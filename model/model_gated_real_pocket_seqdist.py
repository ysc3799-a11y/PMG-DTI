#鍙ｈ鍥?GNN + 娉ㄦ剰鍔?鍧囧€艰仛鍚?+ 铻嶅悎銆?
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import (
    DrugGraphEncoder,
    PharmTokenEncoder,
    PharmTokenEnhancer,
    ProteinTokenEncoder,
    ScaledDotAttn,
    masked_mean_pool,
)
from model.model_gated_multi_pocket import GatedFusion, PharmTokenPool


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

        src, dst = edge_index[0], edge_index[1]
        edge_bias = self.edge_mlp(edge_feat)
        msg = h.index_select(0, src) + edge_bias
        msg = msg.to(h.dtype)

        self_idx = torch.arange(n, device=h.device)
        msg_self = h.index_select(0, self_idx)
        msg_self = msg_self.to(h.dtype)
        dst_all = torch.cat([dst, self_idx], dim=0)
        msg_all = torch.cat([msg, msg_self], dim=0)

        out = torch.zeros_like(h).index_add_(0, dst_all, msg_all)
        deg = torch.zeros((n,), device=h.device).index_add_(0, dst_all, torch.ones_like(dst_all, dtype=torch.float32))
        out = out / deg.clamp(min=1.0).unsqueeze(-1)
        return F.relu(out)


class PocketGraphEncoder(nn.Module):
    def __init__(
        self,
        node_in: int,
        hidden: int,
        out_dim: int,
        num_layers: int = 3,
        edge_feat_dim: int = 2,
        edge_hidden: int = 32,
        dropout: float = 0.1,
        node_proj_dim: Optional[int] = None,
    ):
        super().__init__()
        self.node_proj = None
        if node_proj_dim is not None and int(node_proj_dim) > 0 and int(node_proj_dim) != int(node_in):
            self.node_proj = nn.Sequential(
                nn.Linear(node_in, int(node_proj_dim)),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            node_in = int(node_proj_dim)

        num_layers = int(max(1, num_layers))
        dims = [node_in] + [int(hidden)] * (num_layers - 1) + [int(out_dim)]
        self.layers = nn.ModuleList(
            [
                EdgeFeatureGCNLayer(dims[i], dims[i + 1], edge_feat_dim=edge_feat_dim, edge_hidden=edge_hidden)
                for i in range(len(dims) - 1)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        node_feat: torch.Tensor,
        edge_index: torch.Tensor,
        edge_feat: torch.Tensor,
        graph_slices: List[Tuple[int, int]],
    ) -> torch.Tensor:
        if node_feat.numel() == 0 or not graph_slices:
            return node_feat.new_zeros((len(graph_slices), self.layers[-1].lin.out_features))

        h = node_feat
        if self.node_proj is not None:
            h = self.node_proj(h)
        for layer in self.layers:
            h = layer(h, edge_index, edge_feat)
            h = self.dropout(h)

        z_list = []
        for (s, e) in graph_slices:
            if e <= s:
                z_list.append(h.new_zeros((h.size(-1),)))
            else:
                z_list.append(h[s:e].mean(dim=0))
        return torch.stack(z_list, dim=0)


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    if src.dim() == 1:
        out = src.new_zeros((dim_size,))
        out.index_add_(0, index, src)
        return out
    out = src.new_zeros((dim_size, src.size(-1)))
    out.index_add_(0, index, src)
    return out


def scatter_softmax(scores: torch.Tensor, index: torch.Tensor, dim_size: int) -> torch.Tensor:
    scores = scores.view(-1)
    index = index.view(-1).long()
    out = scores.new_zeros(scores.shape)
    for b in range(int(dim_size)):
        mask_idx = (index == b).nonzero(as_tuple=True)[0]
        if mask_idx.numel() == 0:
            continue
        out[mask_idx] = torch.softmax(scores[mask_idx], dim=0).to(dtype=scores.dtype)
    return out


class BaselineGatedDTIRealPocket(nn.Module):
    """
    Drug graph + pharm tokens + protein tokens + real pocket graph encoder.
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
        # pharm -> protein cross-attention (for residue weights)
        use_pp_attn: bool = False,
        pp_attn_dropout: Optional[float] = None,
        # pocket graph encoder
        use_pocket: bool = True,
        pocket_dim: int = 64,
        pocket_node_in: Optional[int] = None,
        pocket_node_proj: Optional[int] = 128,
        pocket_gnn_hidden: int = 128,
        pocket_gnn_layers: int = 3,
        pocket_struct_dim: int = 29,
        pocket_use_struct_feat: bool = True,
        pocket_edge_feat_dim: int = 21,
        pocket_edge_hidden: int = 32,
        pocket_inst_pool: str = "attn",  # "attn" | "mean"
        pocket_score_prior: bool = True,
        pocket_score_scale: float = 1.0,
        # gated fusion
        fusion_dim: Optional[int] = None,
        gate_hidden: Optional[int] = None,
        fusion_type: str = "gated",
    ):
        super().__init__()
        self.drug_gcn = DrugGraphEncoder(
            node_in=node_in,
            hidden=gcn_hidden,
            out_dim=gcn_out,
            num_layers=gcn_layers,
            dropout=dropout,
            return_nodes=True,
        )
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
            nhead = int(pharm_self_nhead)
            self.pharm_self_attn = nn.MultiheadAttention(
                embed_dim=pharm_out,
                num_heads=nhead,
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
        self.prot_enc = ProteinTokenEncoder(in_dim=prot_dim, pool=prot_pool)

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

        self.use_pocket = bool(use_pocket)
        self.pocket_dim = int(pocket_dim)
        self.pocket_struct_dim = int(max(0, pocket_struct_dim))
        self.pocket_use_struct_feat = bool(pocket_use_struct_feat)
        if pocket_node_in is not None:
            self.pocket_node_dim = int(max(1, pocket_node_in))
        else:
            self.pocket_node_dim = int(self.esm_dim + (self.pocket_struct_dim if self.pocket_use_struct_feat else 0))
        if self.use_pocket:
            self.pocket_gnn = PocketGraphEncoder(
                node_in=self.pocket_node_dim,
                hidden=pocket_gnn_hidden,
                out_dim=self.pocket_dim,
                num_layers=pocket_gnn_layers,
                edge_feat_dim=pocket_edge_feat_dim,
                edge_hidden=pocket_edge_hidden,
                dropout=dropout,
                node_proj_dim=pocket_node_proj,
            )
            self.pocket_inst_pool = str(pocket_inst_pool).lower()
            self.pocket_score_prior = bool(pocket_score_prior)
            self.pocket_score_scale = float(pocket_score_scale)
            if self.pocket_inst_pool == "attn":
                self.pocket_attn = nn.Linear(self.pocket_dim, 1)
            else:
                self.pocket_attn = None
            self.pocket_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.pocket_gnn = None
            self.pocket_attn = None
            self.pocket_gate = None

        self.fusion_type = str(fusion_type or "gated").lower()
        in_dims = [gcn_out, pharm_out, prot_dim]
        if self.use_pocket:
            in_dims.append(self.pocket_dim)
        if self.fusion_type == "concat":
            fusion_dim = int(sum(in_dims))
            self.fusion = None
        else:
            fusion_dim = int(fusion_dim or mlp_hidden)
            self.fusion = GatedFusion(in_dims=in_dims, fusion_dim=fusion_dim, dropout=dropout, gate_hidden=gate_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def _pool_pockets(
        self,
        pocket_repr: torch.Tensor,
        pocket_batch: torch.Tensor,
        pocket_valid: torch.Tensor,
        pocket_score: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        if pocket_repr.numel() == 0 or pocket_batch.numel() == 0:
            return pocket_repr.new_zeros((batch_size, self.pocket_dim))

        valid = pocket_valid.bool()
        if valid.numel() != pocket_repr.shape[0]:
            valid = valid.view(-1)[: pocket_repr.shape[0]]

        if self.pocket_inst_pool == "attn" and self.pocket_attn is not None:
            scores = self.pocket_attn(pocket_repr).squeeze(-1)
            if self.pocket_score_prior:
                scores = scores + self.pocket_score_scale * pocket_score
            neg_inf = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(~valid, neg_inf)
            weights = scatter_softmax(scores, pocket_batch, batch_size)
            weights = weights * valid.float()
        else:
            if self.pocket_score_prior:
                weights = pocket_score.clamp_min(0.0)
            else:
                weights = pocket_repr.new_ones((pocket_repr.shape[0],))
            weights = weights * valid.float()

        z = scatter_sum(pocket_repr * weights.unsqueeze(-1), pocket_batch, batch_size)
        denom = scatter_sum(weights, pocket_batch, batch_size).clamp_min(1e-8)
        z = z / denom.unsqueeze(-1)
        return z

    def forward(self, batch: dict, return_attn: bool = False):
        node_repr, z_graph = self.drug_gcn(
            batch["node_feat"],
            batch["edge_index"],
            batch["node_slices"],
            edge_attr=batch.get("edge_attr", None),
        )

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
                pharm_tokens,
                pharm_tokens,
                pharm_tokens,
                key_padding_mask=key_padding_mask,
            )
            pharm_tokens = self.pharm_self_ln(pharm_tokens + attn_out)
            pharm_tokens = pharm_tokens * pharm_mask.unsqueeze(-1).float()

        z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)
        if self.pharm_pooler is not None:
            z_pharm = self.pharm_pooler(pharm_tokens, batch["pharm_mask"])

        esm_tok = batch["esm2_tok"]
        if self.esm_proj is not None:
            esm_tok = self.esm_proj(esm_tok)

        attn_pp = None
        res_scores = None
        if self.pp_attn is not None:
            # pharm -> protein cross-attention (Query=pharm, KV=protein)
            pharm_tokens, attn_pp = self.pp_attn(
                pharm_tokens, esm_tok, kv_mask=batch["esm2_mask"], return_attn=True
            )  # attn_pp: [B, P, L]
            pharm_tokens = pharm_tokens * batch["pharm_mask"].unsqueeze(-1).float()

            # residue-level scores by averaging over pharm tokens
            pm = batch["pharm_mask"].float().unsqueeze(-1)  # [B, P, 1]
            w = attn_pp * pm
            denom = pm.sum(dim=1).clamp_min(1.0)
            res_scores = w.sum(dim=1) / denom  # [B, L]
            res_scores = res_scores.masked_fill(batch["esm2_mask"] == 0, float("-inf"))

        z_esm = self.prot_enc(esm_tok, batch["esm2_mask"])

        pocket_vec = None
        if self.use_pocket and self.pocket_gnn is not None:
            pocket_repr = self.pocket_gnn(
                batch["pocket_node_feat"],
                batch["pocket_edge_index"],
                batch["pocket_edge_feat"],
                batch["pocket_graph_slices"],
            )
            pocket_vec = self._pool_pockets(
                pocket_repr,
                batch["pocket_graph_batch"],
                batch["pocket_graph_valid"],
                batch["pocket_graph_score"],
                batch_size=int(z_esm.shape[0]),
            )
            if self.pocket_gate is not None:
                pocket_vec = torch.sigmoid(self.pocket_gate) * pocket_vec

        feats = [z_graph, z_pharm, z_esm]
        if self.use_pocket and pocket_vec is not None:
            feats.append(pocket_vec)

        if self.fusion_type == "concat":
            fused = torch.cat(feats, dim=-1)
        else:
            fused, _ = self.fusion(feats)
        logit = self.mlp(fused).squeeze(-1)
        if return_attn:
            return logit, {"attn_pp": attn_pp, "res_scores": res_scores}
        return logit
