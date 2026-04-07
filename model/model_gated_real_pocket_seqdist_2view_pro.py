# -*- coding: utf-8 -*-
"""
更强 pair-scoring 的最小两视角口袋增强版
------------------------------------------------
目标：
1. 保留当前最强 global baseline
   - drug graph -> z_graph
   - pharm      -> z_pharm
   - protein    -> z_esm
   => h_global

2. 把 pocket 从“末端一个 pocket_vec”改成“局部匹配视角”
   - pharm tokens 作为 query source
   - pocket node tokens 作为 local structure source
   - 显式构造 pair 特征并打分
   => h_pair

3. 最终仍做稳妥的残差门控：
   h = h_global + gate(h_global, h_pair) * h_pair

适用：
- 你当前训练脚本
- USE_MULTIVIEW = 0
- USE_PP_ATTN = 0（推荐）
- 先试 top3 + prob0.1 这条最合理的 pocket 线
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# 复用当前 model.py 里的主干模块
from model.model import (
    DrugGraphEncoder,
    PharmTokenEncoder,
    PharmTokenEnhancer,
    ProteinTokenEncoder,
    ScaledDotAttn,
)


# =========================================================
# helpers
# =========================================================

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    """
    x:    [B, L, D]
    mask: [B, L] (bool or 0/1)
    """
    if mask.dtype != torch.bool:
        mask = mask > 0
    m = mask.unsqueeze(-1).float()
    s = (x * m).sum(dim=dim)
    d = m.sum(dim=dim).clamp(min=eps)
    return s / d


def lengths_to_batch(lengths: torch.Tensor) -> torch.Tensor:
    """
    lengths=[3,2] -> [0,0,0,1,1]
    """
    lengths = lengths.long().view(-1)
    if lengths.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=lengths.device)

    out = []
    for i, n in enumerate(lengths.tolist()):
        if n > 0:
            out.append(torch.full((n,), i, dtype=torch.long, device=lengths.device))
    if not out:
        return torch.zeros((0,), dtype=torch.long, device=lengths.device)
    return torch.cat(out, dim=0)


def graph_mean_pool(node_x: torch.Tensor, batch_index: torch.Tensor, num_graphs: Optional[int] = None) -> torch.Tensor:
    """
    node_x:      [N, D]
    batch_index: [N]
    return:      [G, D]
    """
    if node_x.numel() == 0:
        g = int(num_graphs or 0)
        d = int(node_x.shape[-1]) if node_x.dim() == 2 else 0
        return torch.zeros((g, d), dtype=torch.float32, device=node_x.device)

    batch_index = batch_index.long().view(-1)
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0

    out = torch.zeros((num_graphs, node_x.size(-1)), dtype=node_x.dtype, device=node_x.device)
    cnt = torch.zeros((num_graphs, 1), dtype=node_x.dtype, device=node_x.device)

    out.index_add_(0, batch_index, node_x)
    ones = torch.ones((batch_index.numel(), 1), dtype=node_x.dtype, device=node_x.device)
    cnt.index_add_(0, batch_index, ones)

    return out / cnt.clamp(min=1.0)


def _get_esm_tokens(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    if "esm2_tok" in batch:
        return batch["esm2_tok"]
    if "esm2_tokens" in batch:
        return batch["esm2_tokens"]
    raise KeyError("batch missing esm2_tok / esm2_tokens")


# =========================================================
# simple modules
# =========================================================

class AttnPool1D(nn.Module):
    """
    attention pooling for token sequence
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D], mask: [B,L]
        if mask.dtype != torch.bool:
            mask = mask > 0
        score = self.scorer(x).squeeze(-1)                 # [B,L]
        score = score.masked_fill(~mask, float("-inf"))
        alpha = torch.softmax(score, dim=1)
        alpha = torch.where(torch.isfinite(alpha), alpha, torch.zeros_like(alpha))
        alpha = alpha * mask.float()
        alpha = alpha / alpha.sum(dim=1, keepdim=True).clamp(min=1e-8)
        alpha = alpha.unsqueeze(-1)                        # [B,L,1]
        return (x * alpha).sum(dim=1)


class SimpleGatedFusion(nn.Module):
    """
    多向量门控融合，输入若干个 [B,D]
    """
    def __init__(self, dim: int, n_inputs: int, dropout: float = 0.1):
        super().__init__()
        self.n_inputs = int(n_inputs)
        self.dim = int(dim)

        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(dim, dim),
                nn.Sigmoid(),
            )
            for _ in range(self.n_inputs)
        ])

        self.out = nn.Sequential(
            nn.Linear(dim * self.n_inputs, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, xs: List[torch.Tensor]) -> torch.Tensor:
        assert len(xs) == self.n_inputs
        gated = []
        for i, x in enumerate(xs):
            g = self.gates[i](x)
            gated.append(x * g)
        return self.out(torch.cat(gated, dim=-1))


class ResidualPairGate(nn.Module):
    """
    h = h_global + gate * h_pair
    """
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hidden = int(hidden or dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Sigmoid(),
        )

    def forward(self, h_global: torch.Tensor, h_pair: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([h_global, h_pair], dim=-1))
        return h_global + g * h_pair


# =========================================================
# pocket encoder
# =========================================================

class EdgeFeatLayer(nn.Module):
    """
    一个非常轻量的 edge-aware message passing layer
    """
    def __init__(self, hidden: int, edge_dim: int, dropout: float = 0.1):
        super().__init__()
        self.src_proj = nn.Linear(hidden, hidden)
        self.dst_proj = nn.Linear(hidden, hidden)
        self.edge_proj = nn.Linear(edge_dim, hidden)
        self.msg_proj = nn.Linear(hidden * 3, hidden)
        self.upd = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        """
        x: [N,D]
        edge_index: [2,E]
        edge_feat: [E,De]
        """
        if x.numel() == 0:
            return x
        if edge_index.numel() == 0:
            z = self.upd(torch.cat([x, x], dim=-1))
            return self.norm(x + self.dropout(z))

        src = edge_index[0].long()
        dst = edge_index[1].long()

        if edge_feat.dim() == 1:
            edge_feat = edge_feat.view(-1, 1)
        edge_feat = edge_feat.to(dtype=x.dtype, device=x.device)
        expected_edge_dim = int(self.edge_proj.in_features)
        cur_edge_dim = int(edge_feat.shape[1]) if edge_feat.numel() > 0 else 0
        if cur_edge_dim < expected_edge_dim:
            pad = torch.zeros(
                (int(edge_feat.shape[0]), expected_edge_dim - cur_edge_dim),
                dtype=edge_feat.dtype,
                device=edge_feat.device,
            )
            edge_feat = torch.cat([edge_feat, pad], dim=1)
        elif cur_edge_dim > expected_edge_dim:
            edge_feat = edge_feat[:, :expected_edge_dim]

        xs = self.src_proj(x.index_select(0, src))
        xd = self.dst_proj(x.index_select(0, dst))
        ee = self.edge_proj(edge_feat)

        msg = self.msg_proj(torch.cat([xs, xd, ee], dim=-1))

        agg = torch.zeros_like(x)
        msg = msg.to(dtype=agg.dtype)
        agg.index_add_(0, dst, msg)

        deg = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp(min=1.0)

        z = self.upd(torch.cat([x, agg], dim=-1))
        return self.norm(x + self.dropout(z))


class PocketNodeEncoder(nn.Module):
    """
    输入 pocket_node_feat / pocket_edge_index / pocket_edge_feat
    输出：
      pocket_node_repr [N, D]
      pocket_graph_repr [G, D]
    """
    def __init__(
        self,
        node_in: int,
        node_proj: int,
        hidden: int,
        edge_dim: int,
        layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_proj = nn.Sequential(
            nn.Linear(node_in, node_proj),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        if int(node_proj) != int(hidden):
            self.to_hidden = nn.Sequential(
                nn.Linear(node_proj, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.to_hidden = nn.Identity()
        self.layers = nn.ModuleList([
            EdgeFeatLayer(hidden, edge_dim, dropout=dropout)
            for _ in range(int(layers))
        ])

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_batch: torch.Tensor,
        num_graphs: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.in_proj(node_feat)
        x = self.to_hidden(x)
        for layer in self.layers:
            x = layer(x, edge_index, edge_feat)
        g = graph_mean_pool(x, graph_batch, num_graphs=num_graphs)
        return x, g


# =========================================================
# stronger pair-scoring local branch
# =========================================================

class PairScoreLocal(nn.Module):
    """
    更强的局部匹配模块：
    - 不只是一层 cross-attn
    - 显式构造每个 pharm token 与每个 pocket node 的 pair feature
    - 对所有 pair 打分后加权汇聚，得到 pocket-level local evidence
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.r_proj = nn.Linear(dim, dim)

        self.score_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

        self.pair_feat_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )

        self.out = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, q_tokens: torch.Tensor, r_tokens: torch.Tensor) -> torch.Tensor:
        """
        q_tokens: [1, M, D]
        r_tokens: [1, N, D]
        returns : [1, D]
        """
        if q_tokens.size(1) == 0 or r_tokens.size(1) == 0:
            return torch.zeros((1, q_tokens.size(-1)), dtype=q_tokens.dtype, device=q_tokens.device)

        q = q_tokens.squeeze(0)  # [M,D]
        r = r_tokens.squeeze(0)  # [N,D]

        qh = self.q_proj(q)      # [M,D]
        rh = self.r_proj(r)      # [N,D]

        M, D = qh.shape
        N = rh.shape[0]

        qe = qh.unsqueeze(1).expand(M, N, D)
        re = rh.unsqueeze(0).expand(M, N, D)

        pair_in = torch.cat(
            [
                qe,
                re,
                qe * re,
                torch.abs(qe - re),
            ],
            dim=-1,
        )  # [M,N,4D]

        pair_score = self.score_mlp(pair_in).squeeze(-1)   # [M,N]
        pair_feat = self.pair_feat_mlp(pair_in)            # [M,N,D]

        alpha = torch.softmax(pair_score.reshape(-1), dim=0).view(M, N, 1)
        pooled_pair = (pair_feat * alpha).sum(dim=(0, 1), keepdim=False).view(1, D)

        q_ctx = q.mean(dim=0, keepdim=True)   # [1,D]
        r_ctx = r.mean(dim=0, keepdim=True)   # [1,D]

        out = self.out(torch.cat([pooled_pair, q_ctx, r_ctx], dim=-1))
        return self.norm(out)


# =========================================================
# main model
# =========================================================

class BaselineGatedDTIRealPocket(nn.Module):
    """
    更强 pair-scoring 的最小两视角口袋增强版：
    - h_global: strongest baseline 的全局主干
    - h_pair  : pharm <-> pocket nodes 显式局部匹配视角
    - final   : h_global + gated(h_global,h_pair)*h_pair
    """

    def __init__(
        self,
        node_in: int,
        esm_dim: int,
        pharm_num_types: Optional[int] = None,
        *,
        node_hidden: int = 128,
        drug_layers: int = 3,
        drug_dropout: float = 0.1,

        pharm_dim: int = 128,
        pharm_use_graph_attn: bool = True,
        pharm_self_attn: bool = False,
        pharm_pool: str = "attn",

        prot_dim: Optional[int] = None,
        prot_pool: str = "attn",

        use_pp_attn: bool = False,

        use_pocket: bool = False,
        pocket_node_in: Optional[int] = None,
        pocket_edge_feat_dim: int = 2,
        pocket_hidden: int = 128,
        pocket_layers: int = 2,

        fusion_dim: int = 256,
        dropout: float = 0.1,

        **kwargs,
    ):
        super().__init__()

        # compatibility with existing training/config keys
        if pharm_num_types is None:
            pharm_num_types = int(kwargs.get("num_pharm_types", 64))
        node_hidden = int(kwargs.get("gcn_hidden", node_hidden))
        drug_layers = int(kwargs.get("gcn_layers", drug_layers))
        drug_out = int(kwargs.get("gcn_out", node_hidden))
        pharm_dim = int(kwargs.get("pharm_out", pharm_dim))
        pharm_emb_dim = int(kwargs.get("pharm_emb", pharm_dim))
        use_anchor = bool(int(kwargs.get("use_anchor", 1)))
        pocket_dim = int(kwargs.get("pocket_dim", pocket_hidden))
        pocket_hidden = int(kwargs.get("pocket_gnn_hidden", pocket_dim))
        pocket_layers = int(kwargs.get("pocket_gnn_layers", pocket_layers))
        pocket_node_proj = int(kwargs.get("pocket_node_proj", pocket_hidden))
        pocket_edge_hidden = int(kwargs.get("pocket_edge_hidden", pocket_edge_feat_dim))
        pocket_inst_pool = str(kwargs.get("pocket_inst_pool", "softmax")).lower()
        if pocket_inst_pool not in ("softmax", "mean", "avg", "max"):
            pocket_inst_pool = "softmax"
        pocket_struct_dim = int(kwargs.get("pocket_struct_dim", 29))
        pocket_use_struct_feat = bool(int(kwargs.get("pocket_use_struct_feat", 1)))
        pocket_score_prior = bool(int(kwargs.get("pocket_score_prior", 1)))
        pocket_score_scale = float(kwargs.get("pocket_score_scale", 1.0))
        pocket_tau = float(kwargs.get("pocket_tau", 1.0))
        gate_hidden = int(kwargs.get("gate_hidden", fusion_dim))
        mlp_hidden = int(kwargs.get("mlp_hidden", fusion_dim))
        pp_attn_dropout = float(kwargs.get("pp_attn_dropout", dropout))

        self.esm_dim = int(esm_dim)
        esm_proj_dim = kwargs.get("esm_proj_dim", None)
        if esm_proj_dim is not None and int(esm_proj_dim) > 0 and int(esm_proj_dim) != self.esm_dim:
            self.esm_proj = nn.Sequential(
                nn.Linear(self.esm_dim, int(esm_proj_dim)),
                nn.LayerNorm(int(esm_proj_dim)),
            )
            prot_in_dim = int(esm_proj_dim)
        else:
            self.esm_proj = None
            prot_in_dim = self.esm_dim

        prot_dim_cfg = kwargs.get("prot_dim", prot_dim)
        if prot_dim_cfg is None:
            prot_token_dim = int(prot_in_dim)
        else:
            prot_token_dim = int(prot_dim_cfg)
            if prot_token_dim <= 0:
                prot_token_dim = int(prot_in_dim)

        self.use_pp_attn = bool(use_pp_attn)
        self.use_pocket = bool(use_pocket)

        # -------------------------
        # global strongest branches
        # -------------------------
        self.drug_gcn = DrugGraphEncoder(
            node_in=node_in,
            hidden=node_hidden,
            out_dim=drug_out,
            num_layers=drug_layers,
            dropout=drug_dropout,
            return_nodes=True,
        )

        self.pharm_emb = PharmTokenEncoder(
            num_types=pharm_num_types,
            emb_dim=pharm_emb_dim,
            out_dim=pharm_dim,
            dropout=dropout,
        )

        self.pharm_enh = PharmTokenEnhancer(
            pharm_dim=pharm_dim,
            drug_node_dim=drug_out,
            dropout=dropout,
            use_anchor=use_anchor,
            pharm_use_graph_attn=pharm_use_graph_attn,
        )

        self.pharm_self_attn = None
        if pharm_self_attn:
            self.pharm_self_attn = nn.MultiheadAttention(
                embed_dim=pharm_dim,
                num_heads=4,
                dropout=dropout,
                batch_first=True,
            )
            self.pharm_self_ln = nn.LayerNorm(pharm_dim)
        else:
            self.pharm_self_ln = None

        self.pharm_pool_mode = str(pharm_pool).lower()
        self.pharm_pool_attn = AttnPool1D(pharm_dim, dropout=dropout)

        self.prot_enc = ProteinTokenEncoder(
            in_dim=prot_token_dim,
            pool=prot_pool,
        )
        self.prot_out_dim = int(prot_token_dim)
        if int(prot_in_dim) != int(self.prot_out_dim):
            self.prot_token_proj = nn.Sequential(
                nn.Linear(int(prot_in_dim), int(self.prot_out_dim)),
                nn.LayerNorm(int(self.prot_out_dim)),
            )
        else:
            self.prot_token_proj = None

        # keep compatibility, but recommended OFF now
        if self.use_pp_attn:
            self.pp_attn = ScaledDotAttn(
                q_dim=pharm_dim,
                kv_dim=self.prot_out_dim,
                out_dim=pharm_dim,
                dropout=pp_attn_dropout,
            )
            # Make PP-attn output actually affect the protein branch.
            self.pp_to_prot = nn.Linear(pharm_dim, self.prot_out_dim)
            self.pp_ln = nn.LayerNorm(self.prot_out_dim)
            self.pp_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.pp_attn = None
            self.pp_to_prot = None
            self.pp_ln = None
            self.pp_gate = None

        # -------------------------
        # views
        # -------------------------
        self.view_dim = int(fusion_dim)

        self.graph_align = nn.Linear(drug_out, self.view_dim)
        self.pharm_align = nn.Linear(pharm_dim, self.view_dim)
        self.prot_align = nn.Linear(self.prot_out_dim, self.view_dim)
        self.global_fuse = SimpleGatedFusion(self.view_dim, n_inputs=3, dropout=dropout)

        if self.use_pocket:
            if pocket_node_in is None:
                pocket_node_in = int(self.esm_dim + (pocket_struct_dim if pocket_use_struct_feat else 0))
            self.pocket_edge_in = int(pocket_edge_feat_dim)
            self.pocket_edge_dim = int(pocket_edge_hidden) if int(pocket_edge_hidden) > 0 else int(pocket_edge_feat_dim)
            if self.pocket_edge_dim != self.pocket_edge_in:
                self.pocket_edge_proj = nn.Sequential(
                    nn.Linear(self.pocket_edge_in, self.pocket_edge_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                self.pocket_edge_proj = None

            self.pocket_enc = PocketNodeEncoder(
                node_in=int(pocket_node_in),
                node_proj=int(pocket_node_proj),
                hidden=int(pocket_hidden),
                edge_dim=int(self.pocket_edge_dim),
                layers=int(pocket_layers),
                dropout=dropout,
            )
            self.pocket_node_align = nn.Linear(int(pocket_hidden), self.view_dim)
            self.pocket_graph_align = nn.Linear(int(pocket_hidden), self.view_dim)

            self.pair_cross = PairScoreLocal(self.view_dim, dropout=dropout)

            # drug-conditioned pocket relevance scorer across topK pockets
            self.pocket_relevance = nn.Sequential(
                nn.Linear(self.view_dim * 4, self.view_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.view_dim, 1),
            )
        else:
            self.pocket_enc = None
            self.pocket_edge_proj = None
            self.pocket_node_align = None
            self.pocket_graph_align = None
            self.pair_cross = None
            self.pocket_relevance = None

        self.use_pocket_score_prior = bool(pocket_score_prior)
        self.pocket_score_scale = float(pocket_score_scale)
        self.pocket_tau = float(max(1e-6, pocket_tau))
        self.pocket_inst_pool = str(pocket_inst_pool)

        self.pair_residual = ResidualPairGate(self.view_dim, hidden=gate_hidden, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(self.view_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    # -----------------------------------------------------
    # global branch
    # -----------------------------------------------------
    def _run_drug(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        node_repr, z_graph = self.drug_gcn(
            batch["node_feat"],
            batch["edge_index"],
            batch["node_slices"],
            edge_attr=batch.get("edge_attr", None),
        )
        return node_repr, z_graph

    def _run_pharm(self, batch: Dict[str, torch.Tensor], drug_node_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pharm_tokens = self.pharm_emb(batch["pharm_type"], batch["pharm_mask"])  # [B,P,D]
        pharm_anchor = batch.get("pharm_anchor", None)

        pharm_tokens = self.pharm_enh(
            pharm_tokens,
            batch["pharm_mask"],
            drug_node_repr,
            batch["node_slices"],
            pharm_anchor=pharm_anchor,
        )

        if self.pharm_self_attn is not None:
            z, _ = self.pharm_self_attn(pharm_tokens, pharm_tokens, pharm_tokens, need_weights=False)
            pharm_tokens = self.pharm_self_ln(pharm_tokens + z)

        if self.pharm_pool_mode == "attn":
            z_pharm = self.pharm_pool_attn(pharm_tokens, batch["pharm_mask"])
        else:
            z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)

        return pharm_tokens, z_pharm

    def _run_protein(self, batch: Dict[str, torch.Tensor], pharm_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        esm_tok = _get_esm_tokens(batch)
        esm_mask = batch["esm2_mask"]

        if self.esm_proj is not None:
            esm_tok = self.esm_proj(esm_tok)
        if self.prot_token_proj is not None:
            esm_tok = self.prot_token_proj(esm_tok)

        z_esm = self.prot_enc(esm_tok, esm_mask)
        if (
            self.use_pp_attn
            and pharm_tokens is not None
            and self.pp_attn is not None
            and self.pp_to_prot is not None
            and self.pp_ln is not None
            and self.pp_gate is not None
        ):
            pp_tokens = self.pp_attn(pharm_tokens, esm_tok, kv_mask=esm_mask)  # [B,P,Dp]
            pp_mask = batch["pharm_mask"]
            if self.pharm_pool_mode == "attn":
                pp_vec = self.pharm_pool_attn(pp_tokens, pp_mask)
            else:
                pp_vec = masked_mean_pool(pp_tokens, pp_mask, dim=1)
            pp_vec = self.pp_to_prot(pp_vec)
            z_esm = self.pp_ln(z_esm + torch.sigmoid(self.pp_gate) * pp_vec)
        return z_esm

    def _build_h_global(self, z_graph: torch.Tensor, z_pharm: torch.Tensor, z_esm: torch.Tensor) -> torch.Tensor:
        hg = self.graph_align(z_graph)
        hf = self.pharm_align(z_pharm)
        hp = self.prot_align(z_esm)
        return self.global_fuse([hg, hf, hp])

    # -----------------------------------------------------
    # pocket local branch
    # -----------------------------------------------------
    def _get_pocket_batch_index(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
        """
        返回 node-level pocket graph batch index，以及 num_graphs
        """
        if "pocket_graph_slices" in batch:
            slices = batch["pocket_graph_slices"]
            n_nodes = int(batch["pocket_node_feat"].shape[0])
            node_graph = torch.full((n_nodes,), -1, dtype=torch.long, device=batch["pocket_node_feat"].device)
            for gi, (s, e) in enumerate(slices):
                s = int(s)
                e = int(e)
                if e > s:
                    node_graph[s:e] = int(gi)
            num_graphs = int(len(slices))
            if num_graphs == 0 and n_nodes > 0:
                node_graph[:] = 0
                num_graphs = 1
            return node_graph, num_graphs

        if "pocket_node_slices" in batch:
            lengths = batch["pocket_node_slices"].long().view(-1)
            return lengths_to_batch(lengths), int(lengths.numel())

        n = int(batch["pocket_node_feat"].shape[0])
        return torch.zeros((n,), dtype=torch.long, device=batch["pocket_node_feat"].device), (1 if n > 0 else 0)

    def _graph_to_sample_index(self, batch: Dict[str, torch.Tensor], num_graphs: int) -> torch.Tensor:
        """
        graph idx -> sample idx
        """
        B = int(batch["pharm_mask"].shape[0])
        dev = batch["pharm_mask"].device

        if "pocket_graph_batch" in batch:
            gb = batch["pocket_graph_batch"].long().view(-1)
            if int(gb.numel()) == int(num_graphs):
                return gb.to(dev)

        if "pocket_graph_valid" in batch:
            valid = batch["pocket_graph_valid"]
            mapping = []
            if valid.dim() > 1:
                valid2 = valid.view(B, -1).long()
                for i in range(B):
                    for j in range(int(valid2.shape[1])):
                        if int(valid2[i, j].item()) > 0:
                            mapping.append(i)
            else:
                valid1 = valid.view(-1).long()
                if int(valid1.numel()) == int(B):
                    for i in range(B):
                        if int(valid1[i].item()) > 0:
                            mapping.append(i)
                elif B > 0 and int(valid1.numel()) % int(B) == 0:
                    valid2 = valid1.view(B, -1)
                    for i in range(B):
                        for j in range(int(valid2.shape[1])):
                            if int(valid2[i, j].item()) > 0:
                                mapping.append(i)

            if len(mapping) == int(num_graphs):
                return torch.tensor(mapping, dtype=torch.long, device=dev)

        if num_graphs == B:
            return torch.arange(B, dtype=torch.long, device=dev)

        m = min(B, num_graphs)
        out = torch.arange(m, dtype=torch.long, device=dev)
        if m < num_graphs:
            pad = torch.full((num_graphs - m,), m - 1 if m > 0 else 0, dtype=torch.long, device=dev)
            out = torch.cat([out, pad], dim=0)
        return out

    def _build_h_pair(
        self,
        batch: Dict[str, torch.Tensor],
        pharm_tokens: torch.Tensor,
        h_global: torch.Tensor,
    ) -> torch.Tensor:
        """
        局部视角：
        - 每个 pocket graph 先编码成 node_repr / graph_repr
        - 对于属于同一个 sample 的每个 pocket graph:
            pharm_tokens[b] × pocket_nodes[g]
            -> pair_g
        - 如果 sample 有多个 pocket:
            用 drug-conditioned scorer 对多个 pair_g 做 softmax 加权
        - 得到样本级 h_pair
        """
        B = int(batch["pharm_mask"].shape[0])
        dev = h_global.device
        out_dtype = h_global.dtype

        if (not self.use_pocket) or ("pocket_node_feat" not in batch) or (self.pocket_enc is None):
            return torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)

        px = batch["pocket_node_feat"]
        pei = batch["pocket_edge_index"]
        pef = batch["pocket_edge_feat"]

        if px.numel() == 0:
            return torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)
        if self.pocket_edge_proj is not None and pef.numel() > 0:
            pef = self.pocket_edge_proj(pef)

        graph_batch, num_graphs = self._get_pocket_batch_index(batch)
        graph_to_sample = self._graph_to_sample_index(batch, num_graphs)

        pocket_node_repr, pocket_graph_repr = self.pocket_enc(
            px, pei, pef, graph_batch, num_graphs=num_graphs
        )
        pocket_node_repr = self.pocket_node_align(pocket_node_repr)
        pocket_graph_repr = self.pocket_graph_align(pocket_graph_repr)

        sample_pair_reprs: List[List[torch.Tensor]] = [[] for _ in range(B)]
        sample_pair_scores: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for g in range(num_graphs):
            b = int(graph_to_sample[g].item())
            if b < 0 or b >= B:
                continue

            pharm_mask_b = batch["pharm_mask"][b].bool()
            pharm_b = pharm_tokens[b][pharm_mask_b]  # [Mb, Dp]
            if pharm_b.numel() == 0:
                continue

            pharm_b = self.pharm_align(pharm_b).unsqueeze(0)  # [1, Mb, D]

            node_mask = graph_batch == g
            pocket_nodes_g = pocket_node_repr[node_mask]      # [Ng, D]
            if pocket_nodes_g.numel() == 0:
                continue

            pair_g = self.pair_cross(pharm_b, pocket_nodes_g.unsqueeze(0))  # [1,D]
            graph_g = pocket_graph_repr[g:g + 1]                             # [1,D]

            score_in = torch.cat(
                [
                    pair_g,
                    graph_g,
                    h_global[b:b + 1],
                    pair_g * h_global[b:b + 1],
                ],
                dim=-1,
            )
            score_g = self.pocket_relevance(score_in)  # [1,1]

            # 可选：加 pocket 先验 score
            if self.use_pocket_score_prior and "pocket_graph_score" in batch:
                pgs = batch["pocket_graph_score"]
                if pgs.numel() >= g + 1:
                    prior = pgs.view(-1)[g:g + 1].view(1, 1).to(score_g.device, dtype=score_g.dtype)
                    score_g = score_g + self.pocket_score_scale * prior

            sample_pair_reprs[b].append(pair_g)
            sample_pair_scores[b].append(score_g)

        h_pair_list = []
        for b in range(B):
            if len(sample_pair_reprs[b]) == 0:
                h_pair_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                continue

            reps = torch.cat(sample_pair_reprs[b], dim=0)   # [K,D]
            scrs = torch.cat(sample_pair_scores[b], dim=0)  # [K,1]

            if self.pocket_inst_pool in ("mean", "avg"):
                h_pair_b = reps.mean(dim=0, keepdim=True)
            elif self.pocket_inst_pool == "max":
                max_idx = int(torch.argmax(scrs.view(-1)).item())
                h_pair_b = reps[max_idx:max_idx + 1]
            else:
                alpha = torch.softmax((scrs.view(-1).float() / float(self.pocket_tau)), dim=0).unsqueeze(-1).to(dtype=reps.dtype)
                h_pair_b = (reps * alpha).sum(dim=0, keepdim=True)
            h_pair_list.append(h_pair_b)

        return torch.cat(h_pair_list, dim=0).to(dtype=out_dtype)  # [B,D]

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 1) strongest baseline 主干
        drug_node_repr, z_graph = self._run_drug(batch)
        pharm_tokens, z_pharm = self._run_pharm(batch, drug_node_repr)
        z_esm = self._run_protein(batch, pharm_tokens=pharm_tokens)

        h_global = self._build_h_global(z_graph, z_pharm, z_esm)

        # 2) pocket 局部匹配视角
        h_pair = self._build_h_pair(batch, pharm_tokens, h_global)

        # 3) 残差式接入 pocket
        fused = self.pair_residual(h_global, h_pair)

        # 4) 分类
        logit = self.classifier(fused).squeeze(-1)
        return logit


# 兼容旧训练脚本中可能的 import
class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
