# 多源功能单元 + 多视角交互编码（Pair-Transformer / Set-view / Global-view） + 多视角门控融合
# 说明：
# - 在原 BaselineGatedDTIRealPocket 基础上补齐你框架图里的 (C)(D)：
#   1) 局部视角：药效团 token ↔ pocket 残基节点 稀疏配对 → Pair-Transformer → h_pair
#   2) 集合视角：药效团集合 ↔ pocket 子图集合 匹配 → h_set
#   3) 全局视角：z_d/z_p 交互组合 → h_global
#   4) 三视角门控融合：fuse([h_pair,h_set,h_global]) → 预测
#
# - 额外：保留 pocket 节点重要性（node-level）可回溯输出，用于可解释性与 faithfulness 实验。

from __future__ import annotations

from typing import List, Tuple, Optional, Dict, Any

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from baseline_src.model.model import (
    DrugGraphEncoder,
    PharmTokenEncoder,
    PharmTokenEnhancer,
    ProteinTokenEncoder,
    ScaledDotAttn,
    masked_mean_pool,
)
from baseline_src.model.model_gated_multi_pocket import GatedFusion, PharmTokenPool


# -----------------------------
# small helpers
# -----------------------------
def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.numel() == 0:
        if src.dim() == 1:
            return src.new_zeros((dim_size,))
        return src.new_zeros((dim_size, src.size(-1)))
    if src.dim() == 1:
        out = src.new_zeros((dim_size,))
        out.index_add_(0, index, src)
        return out
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    # small dim_size (B or B*G) in this project; keep it simple & stable
    scores = scores.view(-1)
    index = index.view(-1).long()
    out = scores.new_zeros(scores.shape)
    for g in range(int(dim_size)):
        mask_idx = (index == g).nonzero(as_tuple=True)[0]
        if mask_idx.numel() == 0:
            continue
        sg = scores[mask_idx]
        # All -inf / non-finite will make softmax produce NaN.
        if not torch.isfinite(sg).any():
            out[mask_idx] = 0.0
            continue
        sg = torch.nan_to_num(sg, nan=-1e9, posinf=1e9, neginf=-1e9)
        out[mask_idx] = torch.softmax(sg, dim=0).to(dtype=scores.dtype)
    return out


class AttnPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D], mask: [B, T] (1=valid)
        if x.numel() == 0:
            return x.new_zeros((x.size(0), x.size(-1)))
        scores = self.scorer(self.drop(x)).squeeze(-1)  # [B,T]
        neg_inf = torch.finfo(scores.dtype).min
        mask_bool = mask.bool()
        scores = scores.masked_fill(~mask_bool, neg_inf)
        # Guard all-masked rows to avoid NaN from softmax(-inf,...,-inf).
        all_masked = (~mask_bool).all(dim=1, keepdim=True)
        scores = torch.where(all_masked, torch.zeros_like(scores), scores)
        w = torch.softmax(scores, dim=1)  # [B,T]
        w = w * mask_bool.float()
        denom = w.sum(dim=1, keepdim=True).clamp_min(1e-8)
        w = w / denom
        return torch.einsum("bt,btd->bd", w, x)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, num_layers: int, dim_ff: int, dropout: float):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_ff,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.enc = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        # key_padding_mask: [B,T] True for PAD
        return self.enc(x, src_key_padding_mask=key_padding_mask)


# -----------------------------
# pocket GNN (edge feature GCN) with node-attn pooling
# -----------------------------
class EdgeFeatureGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int = 2, edge_hidden: int = 32):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feat_dim, edge_hidden),
            nn.ReLU(),
            nn.Linear(edge_hidden, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_feat: torch.Tensor) -> torch.Tensor:
        h = self.lin(x)
        n = h.size(0)
        if edge_index.numel() == 0 or n == 0:
            return F.relu(h)

        if edge_feat.dim() == 1:
            edge_feat = edge_feat.view(-1, 1)

        src, dst = edge_index[0].long(), edge_index[1].long()
        # [E, out_dim]
        edge_bias = self.edge_mlp(edge_feat)
        msg = h.index_select(0, src) + edge_bias
        out = h.new_zeros((n, h.size(-1)))
        out.index_add_(0, dst, msg)
        deg = h.new_zeros((n,))
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=h.dtype))
        deg = deg.clamp_min(1.0).unsqueeze(-1)
        out = out / deg
        return F.relu(out)


class PocketGraphEncoder(nn.Module):
    """
    Encode pocket residue graph:
      - node_in: ESM2 token dim
      - edge_feat: [dist, seqdist]
    Returns:
      - node_h: [Np, D]
      - graph_h: [Bg, D] (Bg = B*G)
      - node_local_w: [Np] node weights inside its pocket graph (sum to 1 per graph, if graph non-empty)
      - node_graph: [Np] graph id (0..Bg-1)
    """
    def __init__(
        self,
        node_in: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 3,
        edge_feat_dim: int = 2,
        edge_hidden: int = 32,
        dropout: float = 0.1,
        node_proj_dim: Optional[int] = 128,
        node_pool: str = "attn",  # attn | mean
    ):
        super().__init__()
        self.node_pool = str(node_pool).lower()
        if node_proj_dim and int(node_proj_dim) > 0:
            self.node_proj = nn.Sequential(
                nn.Linear(node_in, int(node_proj_dim)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(node_proj_dim), hidden),
            )
            in_dim = hidden
        else:
            self.node_proj = nn.Linear(node_in, hidden)
            in_dim = hidden

        self.layers = nn.ModuleList()
        dims = [in_dim] + [hidden] * max(0, int(num_layers) - 1) + [out_dim]
        for li in range(int(num_layers)):
            self.layers.append(EdgeFeatureGCNLayer(dims[li], dims[li + 1], edge_feat_dim=edge_feat_dim, edge_hidden=edge_hidden))
        self.dropout = nn.Dropout(dropout)

        if self.node_pool == "attn":
            self.attn = nn.Linear(out_dim, 1)
        else:
            self.attn = None

        self.out_dim = int(out_dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_slices: List[Tuple[int, int]],
        return_node: bool = True,
    ):
        if x.numel() == 0:
            node_h = x.new_zeros((0, self.out_dim))
            graph_h = x.new_zeros((len(graph_slices), self.out_dim)) if graph_slices else x.new_zeros((0, self.out_dim))
            node_local_w = x.new_zeros((0,))
            node_graph = x.new_zeros((0,), dtype=torch.long)
            return (node_h, graph_h, node_local_w, node_graph) if return_node else graph_h

        h = self.node_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
            h = self.dropout(h)

        Bg = int(len(graph_slices))
        graph_h_list: List[torch.Tensor] = []
        node_local_w = h.new_zeros((h.size(0),))
        node_graph = torch.full((h.size(0),), -1, dtype=torch.long, device=h.device)

        for gi, (s, e) in enumerate(graph_slices):
            if e <= s:
                graph_h_list.append(h.new_zeros((h.size(-1),)))
                continue
            node_graph[s:e] = gi
            if self.node_pool == "attn" and self.attn is not None:
                scores = self.attn(h[s:e]).squeeze(-1)
                w = torch.softmax(scores, dim=0)
            else:
                w = h.new_full((e - s,), 1.0 / float(e - s))
            node_local_w[s:e] = w
            graph_h_list.append((h[s:e] * w.unsqueeze(-1)).sum(dim=0))

        graph_h = torch.stack(graph_h_list, dim=0) if Bg > 0 else h.new_zeros((0, h.size(-1)))
        if return_node:
            return h, graph_h, node_local_w, node_graph
        return graph_h


# -----------------------------
# multiview fusion
# -----------------------------
class ViewGatedFusion(nn.Module):
    """
    Fuse 3 view vectors (h_pair, h_set, h_global) with context-dependent gates.
    """
    def __init__(self, dim: int, hidden: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        hid = int(hidden or dim)
        self.net = nn.Sequential(
            nn.Linear(dim * 3, hid),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hid, 3),
        )

    def forward(self, h_pair: torch.Tensor, h_set: torch.Tensor, h_global: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([h_pair, h_set, h_global], dim=-1)  # [B, 3D]
        w = torch.softmax(self.net(x), dim=-1)  # [B,3]
        fused = (w[:, 0:1] * h_pair) + (w[:, 1:2] * h_set) + (w[:, 2:3] * h_global)
        return fused, w


# -----------------------------
# main model
# -----------------------------
class BaselineGatedDTIRealPocket(nn.Module):
    """
    Multi-view DTI:
      (A) multi-source encoding:
          drug graph, pharm tokens, protein ESM tokens, pocket residue graphs
      (C) multi-view interaction:
          local(pair) view via Pair-Transformer,
          set view via set self-attn + cross-attn,
          global view via interaction composition
      (D) view-gated fusion + MLP
    """
    def __init__(
        self,
        node_in: int = 74,
        gcn_hidden: int = 128,
        gcn_out: int = 128,
        gcn_layers: int = 2,
        num_pharm_types: int = 32,
        pharm_emb: int = 64,
        pharm_out: int = 64,
        pharm_pool: str = "mean",
        pharm_self_attn: bool = False,
        pharm_self_nhead: int = 4,
        pharm_self_dropout: Optional[float] = None,
        esm_dim: int = 320,
        esm_proj_dim: Optional[int] = None,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        # switches
        use_anchor: bool = True,
        pharm_use_graph_attn: bool = True,
        prot_pool: str = "attn",
        # pharm -> protein cross-attention (optional, mostly for residue-level weights)
        use_pp_attn: bool = False,
        pp_attn_dropout: Optional[float] = None,
        # pocket graph encoder
        use_pocket: bool = True,
        pocket_dim: int = 64,
        pocket_node_proj: Optional[int] = 128,
        pocket_gnn_hidden: int = 128,
        pocket_gnn_layers: int = 3,
        pocket_edge_feat_dim: int = 2,
        pocket_edge_hidden: int = 32,
        pocket_node_pool: str = "attn",
        pocket_inst_pool: str = "attn",  # pocket-graph instance pool: "attn" | "mean"
        pocket_score_prior: bool = True,
        pocket_score_scale: float = 1.0,
        # multiview (Pair-Transformer etc.)
        use_multiview: bool = True,
        view_dim: Optional[int] = None,
        pair_topk: int = 128,
        pair_layers: int = 2,
        pair_nhead: int = 4,
        pair_ff: Optional[int] = None,
        set_layers: int = 1,
        set_nhead: int = 4,
        set_ff: Optional[int] = None,
        # gated fusion of views
        view_gate_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.use_multiview = bool(use_multiview)

        # drug graph
        self.drug_gcn = DrugGraphEncoder(
            node_in=node_in,
            hidden=gcn_hidden,
            out_dim=gcn_out,
            num_layers=gcn_layers,
            dropout=dropout,
            return_nodes=True,
        )

        # pharm tokens
        self.pharm_tok = PharmTokenEncoder(num_types=num_pharm_types, emb_dim=pharm_emb, out_dim=pharm_out, dropout=dropout)
        self.pharm_enh = PharmTokenEnhancer(
            pharm_dim=pharm_out,
            drug_node_dim=gcn_out,
            dropout=dropout,
            use_anchor=use_anchor,
            pharm_use_graph_attn=pharm_use_graph_attn,
        )
        self.pharm_pool = str(pharm_pool).lower()
        self.use_pharm_self_attn = bool(pharm_self_attn)
        if self.use_pharm_self_attn:
            self.pharm_self_attn = nn.MultiheadAttention(
                embed_dim=pharm_out,
                num_heads=int(pharm_self_nhead),
                dropout=float(pharm_self_dropout or dropout),
                batch_first=True,
            )
            self.pharm_self_ln = nn.LayerNorm(pharm_out)
        else:
            self.pharm_self_attn = None
            self.pharm_self_ln = None
        if self.pharm_pool == "attn":
            self.pharm_pooler = PharmTokenPool(pharm_out, dropout=dropout)
        else:
            self.pharm_pooler = None

        # protein tokens
        self.esm_dim = int(esm_dim)
        self.esm_proj_dim = int(esm_proj_dim) if esm_proj_dim else None
        self.use_esm_proj = self.esm_proj_dim is not None and self.esm_proj_dim > 0 and self.esm_proj_dim != self.esm_dim
        prot_dim = self.esm_proj_dim if self.use_esm_proj else self.esm_dim
        self.prot_dim = int(prot_dim)
        if self.use_esm_proj:
            self.esm_proj = nn.Sequential(
                nn.Linear(self.esm_dim, self.esm_proj_dim),
                nn.LayerNorm(self.esm_proj_dim),
            )
        else:
            self.esm_proj = None
        self.prot_enc = ProteinTokenEncoder(in_dim=self.prot_dim, pool=prot_pool)

        # pharm -> protein cross-attention (optional)
        self.use_pp_attn = bool(use_pp_attn)
        if self.use_pp_attn:
            self.pp_attn = ScaledDotAttn(
                q_dim=pharm_out,
                kv_dim=self.prot_dim,
                out_dim=pharm_out,
                dropout=float(pp_attn_dropout or dropout),
            )
        else:
            self.pp_attn = None

        # pocket encoder
        self.use_pocket = bool(use_pocket)
        self.pocket_dim = int(pocket_dim)
        if self.use_pocket:
            self.pocket_gnn = PocketGraphEncoder(
                node_in=self.esm_dim,
                hidden=pocket_gnn_hidden,
                out_dim=self.pocket_dim,
                num_layers=pocket_gnn_layers,
                edge_feat_dim=pocket_edge_feat_dim,
                edge_hidden=pocket_edge_hidden,
                dropout=dropout,
                node_proj_dim=pocket_node_proj,
                node_pool=pocket_node_pool,
            )
            self.pocket_inst_pool = str(pocket_inst_pool).lower()
            self.pocket_score_prior = bool(pocket_score_prior)
            self.pocket_score_scale = float(pocket_score_scale)
            if self.pocket_inst_pool == "attn":
                self.pocket_attn = nn.Linear(self.pocket_dim, 1)
            else:
                self.pocket_attn = None
            # gate for pocket branch
            self.pocket_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.pocket_gnn = None
            self.pocket_attn = None
            self.pocket_gate = None

        # view dims
        self.view_dim = int(view_dim or mlp_hidden)

        # projections for scoring/matching
        self.pharm_proj = nn.Linear(pharm_out, self.view_dim)
        self.pocket_node_proj = nn.Linear(self.pocket_dim, self.view_dim)
        self.pocket_graph_proj = nn.Linear(self.pocket_dim, self.view_dim)
        self.esm_pool_proj = nn.Linear(self.prot_dim, self.view_dim)
        self.drug_pool_proj = nn.Linear(gcn_out, self.view_dim)

        # (C1) Pair-Transformer (local view)
        self.pair_topk = int(pair_topk)
        pair_ff_dim = int(pair_ff or (self.view_dim * 4))
        self.pair_in = nn.Sequential(
            nn.Linear(self.view_dim * 4 + 2, self.view_dim),  # +2: (node_weight, pocket_score)
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.view_dim, self.view_dim),
        )
        self.pair_tf = TransformerBlock(
            d_model=self.view_dim,
            nhead=int(pair_nhead),
            num_layers=int(pair_layers),
            dim_ff=pair_ff_dim,
            dropout=dropout,
        )
        self.pair_pool = AttnPool(self.view_dim, dropout=dropout)

        # (C2) Set view: pocket-set self-attn + pharm->pocket cross-attn
        set_ff_dim = int(set_ff or (self.view_dim * 4))
        self.pocket_set_tf = TransformerBlock(
            d_model=self.view_dim,
            nhead=int(set_nhead),
            num_layers=int(set_layers),
            dim_ff=set_ff_dim,
            dropout=dropout,
        )
        self.set_cross_attn = nn.MultiheadAttention(self.view_dim, int(set_nhead), dropout=dropout, batch_first=True)
        self.set_ln = nn.LayerNorm(self.view_dim)
        self.set_pool = AttnPool(self.view_dim, dropout=dropout)
        self.pocket_score_emb = nn.Sequential(
            nn.Linear(1, self.view_dim),
            nn.Tanh(),
        )

        # (C3) Global view: interaction composition
        self.drug_global = nn.Sequential(
            nn.Linear(self.view_dim * 2, self.view_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.view_dim, self.view_dim),
        )
        self.prot_global = nn.Sequential(
            nn.Linear(self.view_dim * 2, self.view_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.view_dim, self.view_dim),
        )
        self.global_mlp = nn.Sequential(
            nn.Linear(self.view_dim * 4, self.view_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.view_dim, self.view_dim),
        )

        # (D) view fusion + prediction head
        self.view_fuse = ViewGatedFusion(self.view_dim, hidden=view_gate_hidden, dropout=dropout)
        # Initialize view-gate to favor global view at the beginning.
        # softmax([-2,-2,+2]) ~= [0.018, 0.018, 0.964]
        last = self.view_fuse.net[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            with torch.no_grad():
                last.bias.copy_(
                    torch.tensor(
                        [-2.0, -2.0, 2.0],
                        dtype=last.bias.dtype,
                        device=last.bias.device,
                    )
                )
        self.head = nn.Sequential(
            nn.Linear(self.view_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _pool_pockets(
        self,
        pocket_repr: torch.Tensor,        # [Bg, D]
        pocket_batch: torch.Tensor,       # [Bg]
        pocket_valid: torch.Tensor,       # [Bg]
        pocket_score: torch.Tensor,       # [Bg]
        batch_size: int,
        return_weights: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if pocket_repr.numel() == 0 or pocket_batch.numel() == 0:
            z = pocket_repr.new_zeros((batch_size, self.pocket_dim))
            return (z, pocket_repr.new_zeros((0,))) if return_weights else (z, None)

        valid = pocket_valid.bool().view(-1)
        pocket_score = torch.nan_to_num(pocket_score.view(-1), nan=0.0, posinf=0.0, neginf=0.0)
        scores = None
        if self.pocket_inst_pool == "attn" and self.pocket_attn is not None:
            scores = self.pocket_attn(pocket_repr).squeeze(-1)  # [Bg]
            if self.pocket_score_prior:
                scores = scores + self.pocket_score_scale * pocket_score
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~valid, neg_inf)
            w = scatter_softmax(scores, pocket_batch, batch_size)  # [Bg]
            w = w * valid.float()
        else:
            if self.pocket_score_prior:
                w = pocket_score.clamp_min(0.0)
            else:
                w = pocket_repr.new_ones((pocket_repr.shape[0],))
            w = w * valid.float()

        z = scatter_sum(pocket_repr * w.unsqueeze(-1), pocket_batch, batch_size)
        denom = scatter_sum(w, pocket_batch, batch_size).clamp_min(1e-8)
        z = z / denom.unsqueeze(-1)

        if return_weights:
            return z, w
        return z, None

    def _build_pair_tokens(
        self,
        pharm_tok: torch.Tensor,        # [B,P,Dp]
        pharm_mask: torch.Tensor,       # [B,P]
        pocket_node_h: torch.Tensor,    # [Np,Dpocket]
        pocket_node_batch: torch.Tensor,# [Np] sample id
        pocket_node_valid: torch.Tensor,# [Np] bool
        pocket_node_weight: torch.Tensor,# [Np] float (sample-level)
        pocket_node_score: torch.Tensor,# [Np] float (p2rank score copied)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Build sparse pharm-pocket pair tokens:
          - score = dot(pharm_proj, pocket_proj)
          - global topK pairs per sample
        Returns:
          pair_tok: [B,K,D]
          pair_mask: [B,K] bool
          pair_pharm_idx: [B,K] original pharm index (0..P-1, -1 for pad)
          pair_node_idx: [B,K] global pocket node index (0..Np-1, -1 for pad)
        """
        B, P, _ = pharm_tok.shape
        K = self.pair_topk
        device = pharm_tok.device

        pair_tok = pharm_tok.new_zeros((B, K, self.view_dim))
        pair_mask = torch.zeros((B, K), dtype=torch.bool, device=device)
        pair_pharm_idx = torch.full((B, K), -1, dtype=torch.long, device=device)
        pair_node_idx = torch.full((B, K), -1, dtype=torch.long, device=device)

        if pocket_node_h.numel() == 0:
            return pair_tok, pair_mask, pair_pharm_idx, pair_node_idx

        # project all pocket nodes once
        pocket_proj_all = self.pocket_node_proj(pocket_node_h)  # [Np,D]
        pocket_node_weight = torch.nan_to_num(pocket_node_weight.view(-1), nan=0.0, posinf=0.0, neginf=0.0)
        pocket_node_score = torch.nan_to_num(pocket_node_score.view(-1), nan=0.0, posinf=0.0, neginf=0.0)
        for b in range(B):
            pm = pharm_mask[b].bool()
            if not pm.any():
                continue
            node_mask = (pocket_node_batch == b) & pocket_node_valid.bool()
            if not node_mask.any():
                continue

            # valid pharm indices & features
            p_idx = torch.nonzero(pm, as_tuple=False).view(-1)  # [P']
            f = self.pharm_proj(pharm_tok[b].index_select(0, p_idx))  # [P',D]

            # nodes for this sample
            n_idx = torch.nonzero(node_mask, as_tuple=False).view(-1)  # [N']
            n = pocket_proj_all.index_select(0, n_idx)  # [N',D]

            # scores [P',N']
            scores = (f @ n.t()) / math.sqrt(float(self.view_dim))
            scores = torch.nan_to_num(scores, nan=-1e9, posinf=1e9, neginf=-1e9)
            flat = scores.reshape(-1)
            topk = min(K, int(flat.numel()))
            if topk <= 0:
                continue
            v, idx_flat = torch.topk(flat, k=topk, largest=True, sorted=False)
            ii = torch.div(idx_flat, n.size(0), rounding_mode="floor")  # pharm index in [0,P')
            jj = idx_flat % n.size(0)  # node index in [0,N')

            f_sel = f.index_select(0, ii)
            n_sel = n.index_select(0, jj)

            # add 2 scalars: node_weight + node_score
            node_w = pocket_node_weight.index_select(0, n_idx.index_select(0, jj)).clamp_min(0.0).unsqueeze(-1)
            node_s = pocket_node_score.index_select(0, n_idx.index_select(0, jj)).clamp_min(0.0).unsqueeze(-1)
            feat = torch.cat([f_sel, n_sel, f_sel * n_sel, (f_sel - n_sel).abs(), node_w, node_s], dim=-1)
            tok = self.pair_in(feat)  # [topk,D]

            pair_tok[b, :topk] = tok
            pair_mask[b, :topk] = True
            pair_pharm_idx[b, :topk] = p_idx.index_select(0, ii)
            pair_node_idx[b, :topk] = n_idx.index_select(0, jj)

        return pair_tok, pair_mask, pair_pharm_idx, pair_node_idx

    def forward(self, batch: dict, return_attn: bool = False):
        # (A) drug graph
        node_repr, z_graph = self.drug_gcn(
            batch["node_feat"],
            batch["edge_index"],
            batch["node_slices"],
            edge_attr=batch.get("edge_attr", None),
        )
        z_graph_v = self.drug_pool_proj(z_graph)  # [B,D]

        # (A)(B) pharm tokens
        pharm_tokens = self.pharm_tok(batch["pharm_type"], batch["pharm_mask"])
        pharm_anchor = batch.get("pharm_anchor", None)
        pharm_tokens = self.pharm_enh(
            pharm_tokens,
            batch["pharm_mask"],
            node_repr,
            batch["node_slices"],
            pharm_anchor=pharm_anchor,
        )

        if self.pharm_self_attn is not None:
            pharm_mask = batch["pharm_mask"].bool()
            if not pharm_mask.any(dim=1).all():
                pharm_mask_safe = pharm_mask.clone()
                empty = ~pharm_mask.any(dim=1)
                pharm_mask_safe[empty, 0] = True
                pharm_tokens = pharm_tokens.clone()
                pharm_tokens[empty, 0] = 0
                key_padding_mask = ~pharm_mask_safe
            else:
                key_padding_mask = ~pharm_mask
            attn_out, _ = self.pharm_self_attn(
                pharm_tokens, pharm_tokens, pharm_tokens, key_padding_mask=key_padding_mask
            )
            pharm_tokens = self.pharm_self_ln(pharm_tokens + attn_out)
            pharm_tokens = pharm_tokens * pharm_mask.unsqueeze(-1).float()

        # protein ESM tokens
        esm_tok = batch["esm2_tok"]
        if self.esm_proj is not None:
            esm_tok = self.esm_proj(esm_tok)

        # optional pharm->protein cross-attn (useful for residue weights)
        attn_pp = None
        res_scores = None
        if self.pp_attn is not None:
            pharm_tokens, attn_pp = self.pp_attn(
                pharm_tokens, esm_tok, kv_mask=batch["esm2_mask"], return_attn=True
            )  # attn_pp: [B,P,L]
            pharm_tokens = pharm_tokens * batch["pharm_mask"].unsqueeze(-1).float()

            pm = batch["pharm_mask"].float().unsqueeze(-1)  # [B,P,1]
            w = attn_pp * pm
            denom = pm.sum(dim=1).clamp_min(1.0)
            res_scores = w.sum(dim=1) / denom  # [B,L]
            res_scores = res_scores.masked_fill(batch["esm2_mask"] == 0, float("-inf"))

        # IMPORTANT FIX: pool z_pharm AFTER pp_attn so that cross-attn can affect prediction
        z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)
        if self.pharm_pooler is not None:
            z_pharm = self.pharm_pooler(pharm_tokens, batch["pharm_mask"])
        z_pharm_v = self.pharm_proj(z_pharm)  # [B,D]

        # protein global vector
        z_esm = self.prot_enc(esm_tok, batch["esm2_mask"])  # [B,prot_dim]
        z_esm_v = self.esm_pool_proj(z_esm)  # [B,D]

        # pocket branch
        pocket_vec = None
        pocket_graph_w = None
        pocket_node_weight = None
        pocket_node_graph = batch.get("pocket_node_graph", None)
        pocket_node_batch = batch.get("pocket_node_batch", None)
        pocket_node_valid = batch.get("pocket_node_valid", None)
        pocket_node_score = batch.get("pocket_node_score", None)

        pocket_node_h = None
        pocket_graph_h = None
        node_local_w = None
        node_graph = None

        if self.use_pocket and self.pocket_gnn is not None:
            pocket_node_h, pocket_graph_h, node_local_w, node_graph = self.pocket_gnn(
                batch["pocket_node_feat"],
                batch["pocket_edge_index"],
                batch["pocket_edge_feat"],
                batch["pocket_graph_slices"],
                return_node=True,
            )
            pocket_vec, pocket_graph_w = self._pool_pockets(
                pocket_graph_h,
                batch["pocket_graph_batch"],
                batch["pocket_graph_valid"],
                batch["pocket_graph_score"],
                batch_size=int(z_esm.shape[0]),
                return_weights=True,
            )  # pocket_vec: [B,pocket_dim], pocket_graph_w: [Bg]
            if self.pocket_gate is not None:
                pocket_vec = torch.sigmoid(self.pocket_gate) * pocket_vec

            # build node-level weights (sample-level)
            # prefer mapping from batch (precomputed), else compute from node_graph
            if (
                pocket_node_batch is None
                or pocket_node_valid is None
                or pocket_node_score is None
                or pocket_node_graph is None
            ):
                # compute node->graph->sample mapping
                node_graph_safe = node_graph.clone()
                node_graph_safe[node_graph_safe < 0] = 0
                pocket_node_batch = batch["pocket_graph_batch"].index_select(0, node_graph_safe) if batch["pocket_graph_batch"].numel() > 0 else node_graph_safe.new_zeros((node_graph_safe.numel(),))
                pocket_node_valid = batch["pocket_graph_valid"].index_select(0, node_graph_safe) if batch["pocket_graph_valid"].numel() > 0 else node_graph_safe.new_zeros((node_graph_safe.numel(),), dtype=torch.bool)
                pocket_node_score = batch["pocket_graph_score"].index_select(0, node_graph_safe) if batch["pocket_graph_score"].numel() > 0 else node_graph_safe.new_zeros((node_graph_safe.numel(),), dtype=torch.float32)
                pocket_node_graph = node_graph

            # node weight = local attn weight * graph weight
            if pocket_graph_w is None or pocket_graph_w.numel() == 0 or pocket_node_graph is None:
                pocket_node_weight = node_local_w
            else:
                g_safe = pocket_node_graph.clone()
                g_safe[g_safe < 0] = 0
                g_safe = g_safe.clamp_max(int(pocket_graph_w.numel()) - 1)
                pocket_node_weight = node_local_w * pocket_graph_w.index_select(0, g_safe)
            if pocket_node_valid is not None:
                pocket_node_weight = pocket_node_weight * pocket_node_valid.float()
            pocket_node_weight = torch.nan_to_num(pocket_node_weight, nan=0.0, posinf=0.0, neginf=0.0)
            if pocket_node_score is not None:
                pocket_node_score = torch.nan_to_num(pocket_node_score, nan=0.0, posinf=0.0, neginf=0.0)

        # -----------------------
        # (C) multi-view interaction
        # -----------------------
        B = int(z_esm.shape[0])

        # global view
        # drug side global
        z_d = self.drug_global(torch.cat([z_graph_v, z_pharm_v], dim=-1))
        # protein side global
        if self.use_pocket and pocket_vec is not None:
            pocket_vec_v = self.pocket_graph_proj(pocket_vec)  # [B,D]
            z_p = self.prot_global(torch.cat([z_esm_v, pocket_vec_v], dim=-1))
        else:
            # fallback: duplicate z_esm_v
            z_p = self.prot_global(torch.cat([z_esm_v, z_esm_v], dim=-1))

        gfeat = torch.cat([z_d, z_p, z_d * z_p, (z_d - z_p).abs()], dim=-1)
        h_global = self.global_mlp(gfeat)  # [B,D]

        # Optional ablation: disable local/set multiview branches.
        if not self.use_multiview:
            logit = self.head(h_global).squeeze(-1)
            if return_attn:
                return logit, {
                    "view_weights": None,
                    "attn_pp": attn_pp,
                    "res_scores": res_scores,
                    "pocket_node_weight": pocket_node_weight,
                }
            return logit

        # if pocket missing, the other 2 views become zeros
        if not self.use_pocket or pocket_node_h is None or pocket_node_h.numel() == 0:
            logit = self.head(h_global).squeeze(-1)
            if return_attn:
                return logit, {
                    "view_weights": None,
                    "attn_pp": attn_pp,
                    "res_scores": res_scores,
                    "pocket_node_weight": pocket_node_weight,
                }
            return logit

        # local(pair) view via Pair-Transformer
        pair_tok, pair_mask, pair_pharm_idx, pair_node_idx = self._build_pair_tokens(
            pharm_tok=pharm_tokens,
            pharm_mask=batch["pharm_mask"],
            pocket_node_h=pocket_node_h,
            pocket_node_batch=pocket_node_batch,
            pocket_node_valid=pocket_node_valid,
            pocket_node_weight=pocket_node_weight if pocket_node_weight is not None else pocket_node_h.new_ones((pocket_node_h.size(0),)),
            pocket_node_score=pocket_node_score if pocket_node_score is not None else pocket_node_h.new_zeros((pocket_node_h.size(0),)),
        )
        # Guard samples with no valid pair token (all False mask).
        pair_mask_safe = pair_mask.clone()
        if not pair_mask_safe.all(dim=1).all():
            empty_pair = ~pair_mask_safe.any(dim=1)
            if empty_pair.any():
                pair_tok = pair_tok.clone()
                pair_tok[empty_pair, 0, :] = 0.0
                pair_mask_safe[empty_pair, 0] = True
        pair_out = self.pair_tf(pair_tok, key_padding_mask=~pair_mask_safe)  # [B,K,D]
        h_pair = self.pair_pool(pair_out, pair_mask_safe)  # [B,D]

        # set view: pocket-set tokens + pharm->pocket cross-attn
        # build pocket set [B, Gmax, D]
        Bg = int(pocket_graph_h.size(0))
        gmask = batch["pocket_graph_valid"].bool().view(-1)[:Bg]
        # indices per sample
        pocket_set_list = []
        pocket_set_mask_list = []
        score_list = []
        max_g = 0
        for b in range(B):
            gidx = (batch["pocket_graph_batch"].view(-1)[:Bg] == b).nonzero(as_tuple=True)[0]
            max_g = max(max_g, int(gidx.numel()))
            pocket_set_list.append(gidx)
        max_g = max(max_g, 1)

        pocket_set = pocket_graph_h.new_zeros((B, max_g, self.view_dim))
        pocket_set_mask = torch.zeros((B, max_g), dtype=torch.bool, device=pocket_set.device)
        for b in range(B):
            gidx = pocket_set_list[b]
            if gidx.numel() == 0:
                continue
            gg = pocket_graph_h.index_select(0, gidx)  # [G',pocket_dim]
            gg = self.pocket_graph_proj(gg)  # [G',D]
            sc = batch["pocket_graph_score"].view(-1)[:Bg].index_select(0, gidx).unsqueeze(-1)  # [G',1]
            gg = gg + self.pocket_score_emb(sc)
            vv = gmask.index_select(0, gidx)
            m = vv.bool()
            n = int(gg.size(0))
            pocket_set[b, :n] = gg
            pocket_set_mask[b, :n] = m

        # Guard samples with empty pocket set for transformer/cross-attn stability.
        pocket_set_mask_safe = pocket_set_mask.clone()
        if not pocket_set_mask_safe.all(dim=1).all():
            empty_set = ~pocket_set_mask_safe.any(dim=1)
            if empty_set.any():
                pocket_set = pocket_set.clone()
                pocket_set[empty_set, 0, :] = 0.0
                pocket_set_mask_safe[empty_set, 0] = True

        pocket_set_ctx = self.pocket_set_tf(pocket_set, key_padding_mask=~pocket_set_mask_safe)  # [B,G,D]

        # pharm queries
        q = self.pharm_proj(pharm_tokens)  # [B,P,D]
        key_padding_mask = ~pocket_set_mask_safe  # True for PAD
        cross_out, cross_attn = self.set_cross_attn(
            q, pocket_set_ctx, pocket_set_ctx, key_padding_mask=key_padding_mask, need_weights=return_attn
        )
        cross_out = self.set_ln(q + cross_out)
        h_set = self.set_pool(cross_out, batch["pharm_mask"].bool())

        # (D) view gated fusion + prediction
        view_fused, view_w = self.view_fuse(h_pair, h_set, h_global)
        logit = self.head(view_fused).squeeze(-1)

        if return_attn:
            out: Dict[str, Any] = {
                "view_weights": view_w,                 # [B,3]
                "pair_pharm_idx": pair_pharm_idx,       # [B,K]
                "pair_node_idx": pair_node_idx,         # [B,K]
                "pocket_node_weight": pocket_node_weight,  # [Np]
                "attn_pp": attn_pp,
                "res_scores": res_scores,
            }
            # for back-tracing to residues
            if "pocket_node_tok_idx" in batch:
                out["pocket_node_tok_idx"] = batch["pocket_node_tok_idx"]
            if "pocket_node_res_seq" in batch:
                out["pocket_node_res_seq"] = batch["pocket_node_res_seq"]
            if pocket_node_batch is not None:
                out["pocket_node_batch"] = pocket_node_batch
            if pocket_node_graph is not None:
                out["pocket_node_graph"] = pocket_node_graph
            if pocket_graph_w is not None:
                out["pocket_graph_weight"] = pocket_graph_w
            if node_local_w is not None:
                out["pocket_node_weight_local"] = node_local_w
            if return_attn:
                out["set_cross_attn"] = cross_attn  # [B, P, G] (if available)
            return logit, out

        return logit
