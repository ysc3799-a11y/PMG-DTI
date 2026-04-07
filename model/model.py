from typing import List, Tuple, Optional

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """
    x: [..., L, D], mask: [..., L] (0/1)
    """
    mask_f = mask.float()
    while mask_f.dim() < x.dim():
        mask_f = mask_f.unsqueeze(-1)
    x = x * mask_f
    denom = mask_f.sum(dim=dim).clamp_min(1.0)
    return x.sum(dim=dim) / denom


def masked_softmax(scores: torch.Tensor, mask: torch.Tensor = None, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax computed in float32.

    Handles all-masked rows and accidental non-finite scores by falling back to
    zero weights instead of propagating NaN/Inf.
    """
    s = torch.nan_to_num(scores.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    mask_exp = None
    valid = None
    if mask is not None:
        mask_exp = mask.bool() if mask.dtype != torch.bool else mask
        while mask_exp.dim() < s.dim():
            mask_exp = mask_exp.unsqueeze(1)
        s = s.masked_fill(~mask_exp, torch.finfo(s.dtype).min)
        valid = mask_exp.any(dim=dim, keepdim=True)
        s = torch.where(valid, s, torch.zeros_like(s))
    s = s - torch.amax(s, dim=dim, keepdim=True)
    w = torch.softmax(s, dim=dim)
    w = torch.where(torch.isfinite(w), w, torch.zeros_like(w))
    if mask_exp is not None:
        w = w * mask_exp.to(dtype=w.dtype)
        w = w / w.sum(dim=dim, keepdim=True).clamp_min(1e-12)
        w = torch.where(valid, w, torch.zeros_like(w))
    return w.to(dtype=scores.dtype)


def safe_softmax(scores: torch.Tensor, mask: torch.Tensor, tau: float = 1.0) -> torch.Tensor:
    """
    scores: [B, L] (may contain -inf)
    mask: [B, L] bool (True=valid)
    """
    s = scores / max(float(tau), 1e-6)
    return masked_softmax(s, mask=mask, dim=-1)


def build_topk_mask(scores: torch.Tensor, mask: torch.Tensor, k: int = 32, min_keep: int = 8) -> torch.Tensor:
    """
    scores: [B, L]
    mask: [B, L] bool valid
    returns: pocket_mask [B, L] bool
    """
    B, L = scores.shape
    pocket_mask = torch.zeros((B, L), device=scores.device, dtype=torch.bool)

    k_eff = max(int(k), 1)
    min_keep = max(int(min_keep), 1)

    for b in range(B):
        valid = mask[b]
        n_valid = int(valid.sum().item())
        if n_valid == 0:
            continue
        kb = min(k_eff, n_valid)
        kb = max(kb, min_keep)
        kb = min(kb, n_valid)
        sc = scores[b].clone()
        sc[~valid] = torch.finfo(sc.dtype).min
        idx = torch.topk(sc, k=kb, largest=True).indices
        pocket_mask[b, idx] = True
    return pocket_mask


class SimpleGCNLayer(nn.Module):
    """
    Minimal edge-aware GCN layer using edge_index (COO) and degree-normalized
    message passing. Falls back gracefully when edge_attr is missing.
    """
    def __init__(self, in_dim: int, out_dim: int, edge_feat_dim: int = 0, edge_hidden: int = 32):
        super().__init__()
        self.lin = nn.Linear(in_dim, out_dim)
        self.edge_feat_dim = int(max(0, edge_feat_dim))
        if self.edge_feat_dim > 0:
            self.edge_mlp = nn.Sequential(
                nn.Linear(self.edge_feat_dim, int(edge_hidden)),
                nn.ReLU(),
                nn.Linear(int(edge_hidden), out_dim),
            )
        else:
            self.edge_mlp = None

    def _prep_edge_attr(self, edge_attr: Optional[torch.Tensor], num_edges: int, device, dtype) -> Optional[torch.Tensor]:
        if self.edge_mlp is None:
            return None
        if edge_attr is None or edge_attr.numel() == 0:
            return torch.zeros((num_edges, self.edge_feat_dim), device=device, dtype=dtype)
        ea = edge_attr.to(device=device, dtype=dtype)
        if ea.dim() == 1:
            ea = ea.view(-1, 1)
        feat_dim = int(ea.shape[1])
        if feat_dim < self.edge_feat_dim:
            pad = torch.zeros((int(ea.shape[0]), self.edge_feat_dim - feat_dim), device=device, dtype=dtype)
            ea = torch.cat([ea, pad], dim=1)
        elif feat_dim > self.edge_feat_dim:
            ea = ea[:, : self.edge_feat_dim]
        return ea

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: Optional[torch.Tensor] = None) -> torch.Tensor:
        N = x.size(0)
        if N == 0:
            return x
        # Graph message passing is prone to fp16 overflow/NaN under AMP, especially on
        # large or dense molecules. Keep the whole layer in fp32 and return fp32 so the
        # drug branch remains numerically stable through backprop.
        with torch.autocast(device_type=x.device.type, enabled=False):
            x32 = torch.nan_to_num(
                x.to(dtype=torch.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            h = self.lin(x32)
            if edge_index.numel() == 0:
                return torch.nan_to_num(h, nan=0.0, posinf=0.0, neginf=0.0)
            row, col = edge_index[0].long(), edge_index[1].long()  # messages col -> row
            msg = h[col]
            if self.edge_mlp is not None:
                ea = self._prep_edge_attr(edge_attr, int(edge_index.shape[1]), x.device, torch.float32)
                msg = msg + self.edge_mlp(ea)

            self_idx = torch.arange(N, device=x.device, dtype=torch.long)
            dst_all = torch.cat([row, self_idx], dim=0)
            msg_all = torch.cat([msg, h], dim=0)

            out = torch.zeros_like(h)
            out.index_add_(0, dst_all, msg_all)
            deg = torch.zeros((N,), device=x.device, dtype=h.dtype)
            deg.index_add_(0, dst_all, torch.ones_like(dst_all, dtype=h.dtype))
            out = out / deg.clamp_min(1.0).unsqueeze(-1)
            out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
            return out


class DrugGCNEncoder(nn.Module):
    def __init__(self, node_in: int, hidden: int, out: int, layers: int = 2, dropout: float = 0.1, return_nodes: bool = True, edge_feat_dim: int = 12, edge_hidden: int = 32):
        super().__init__()
        self.return_nodes = return_nodes
        self.edge_feat_dim = int(max(0, edge_feat_dim))
        dims = [node_in] + [hidden] * (layers - 1) + [out]
        self.layers = nn.ModuleList([SimpleGCNLayer(dims[i], dims[i + 1], edge_feat_dim=self.edge_feat_dim, edge_hidden=edge_hidden) for i in range(len(dims) - 1)])
        self.drop = nn.Dropout(dropout)

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, node_slices: List[Tuple[int, int]], edge_attr: Optional[torch.Tensor] = None):
        x = node_feat
        for li, layer in enumerate(self.layers):
            x = layer(x, edge_index, edge_attr=edge_attr)
            if li != len(self.layers) - 1:
                x = F.relu(x)
                x = self.drop(x)

        # pool per graph
        outs = []
        for (s, e) in node_slices:
            outs.append(x[s:e].mean(dim=0))
        z_graph = torch.stack(outs, dim=0)  # [B, out]

        if self.return_nodes:
            return z_graph, x
        return z_graph


class DrugGraphEncoder(DrugGCNEncoder):
    """
    Compatibility wrapper for older codepaths that expect:
      node_repr, z_graph = encoder(...)
    while DrugGCNEncoder returns:
      z_graph, node_repr
    """
    def __init__(self, node_in: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.1, return_nodes: bool = True, edge_feat_dim: int = 12, edge_hidden: int = 32):
        super().__init__(
            node_in=node_in,
            hidden=hidden,
            out=out_dim,
            layers=num_layers,
            dropout=dropout,
            return_nodes=return_nodes,
            edge_feat_dim=edge_feat_dim,
            edge_hidden=edge_hidden,
        )

    def forward(self, node_feat: torch.Tensor, edge_index: torch.Tensor, node_slices: List[Tuple[int, int]], edge_attr: Optional[torch.Tensor] = None):
        out = super().forward(node_feat, edge_index, node_slices, edge_attr=edge_attr)
        if self.return_nodes:
            z_graph, node_repr = out
            return node_repr, z_graph
        return out


class ScaledDotAttn(nn.Module):
    """
    Single-head scaled dot-product attention updating Query tokens with K/V tokens.
    Includes a small residual gate for stability.
    """
    def __init__(self, q_dim: int, kv_dim: int, out_dim: Optional[int] = None, dropout: float = 0.1):
        super().__init__()
        out_dim = out_dim or q_dim
        self.q_proj = nn.Linear(q_dim, out_dim)
        self.k_proj = nn.Linear(kv_dim, out_dim)
        self.v_proj = nn.Linear(kv_dim, out_dim)
        self.out_proj = nn.Linear(out_dim, out_dim)
        self.drop = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(out_dim)
        # residual gate (learnable), initialized small
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def forward(self, q: torch.Tensor, kv: torch.Tensor, kv_mask: Optional[torch.Tensor] = None, return_attn: bool = False):
        """
        q: [B, Q, Dq]
        kv: [B, K, Dkv]
        kv_mask: [B, K] (0/1) where 0 = padded / ignore
        returns: updated q: [B, Q, Dout]
        """
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))  # [B, Q, K]
        attn_prob = masked_softmax(scores, mask=kv_mask, dim=-1)
        attn_drop = self.drop(attn_prob)
        ctx = torch.matmul(attn_drop, V)  # [B,Q,D]
        ctx = self.out_proj(ctx)
        out = self.ln(Q + self.gate * ctx)
        if return_attn:
            return out, attn_prob
        return out


class PharmTokenEncoder(nn.Module):
    def __init__(
        self,
        num_types: int,
        emb_dim: int,
        out_dim: int,
        dropout: float = 0.1,
        num_families: int = 0,
        family_emb_dim: int = 0,
        context_dim: int = 0,
        geom_dim: int = 0,
        center_dim: int = 0,
    ):
        super().__init__()
        self.emb = nn.Embedding(num_types, emb_dim)
        self.proj = nn.Linear(emb_dim, out_dim)
        self.drop = nn.Dropout(dropout)

        self.num_families = int(max(0, num_families))
        fam_dim = int(family_emb_dim) if int(family_emb_dim) > 0 else max(8, emb_dim // 4)
        if self.num_families > 0:
            self.family_emb = nn.Embedding(self.num_families, fam_dim)
            self.family_proj = nn.Linear(fam_dim, emb_dim)
        else:
            self.family_emb = None
            self.family_proj = None

        self.context_dim = int(max(0, context_dim))
        if self.context_dim > 0:
            self.context_proj = nn.Sequential(
                nn.LayerNorm(self.context_dim),
                nn.Linear(self.context_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.context_proj = None

        self.geom_dim = int(max(0, geom_dim))
        if self.geom_dim > 0:
            self.geom_proj = nn.Sequential(
                nn.LayerNorm(self.geom_dim),
                nn.Linear(self.geom_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.geom_proj = None

        self.center_dim = int(max(0, center_dim))
        if self.center_dim > 0:
            self.center_proj = nn.Sequential(
                nn.LayerNorm(self.center_dim),
                nn.Linear(self.center_dim, emb_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
        else:
            self.center_proj = None

    def forward(
        self,
        pharm_type: torch.Tensor,
        pharm_mask: torch.Tensor,
        pharm_family: Optional[torch.Tensor] = None,
        pharm_context: Optional[torch.Tensor] = None,
        pharm_geom: Optional[torch.Tensor] = None,
        pharm_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pharm_type: [B, P] int64
        pharm_mask: [B, P] 0/1
        """
        x = self.emb(pharm_type)  # [B,P,E]
        if self.family_emb is not None and self.family_proj is not None and pharm_family is not None:
            x = x + self.family_proj(self.family_emb(pharm_family.long()))
        if self.context_proj is not None and pharm_context is not None:
            x = x + self.context_proj(pharm_context.float())
        if self.geom_proj is not None and pharm_geom is not None:
            x = x + self.geom_proj(pharm_geom.float())
        if self.center_proj is not None and pharm_center is not None:
            x = x + self.center_proj(pharm_center.float())
        x = self.drop(x)
        x = self.proj(x)  # [B,P,D]
        x = x * pharm_mask.unsqueeze(-1).float()
        return x


class PharmTokenEnhancer(nn.Module):
    """
    Optionally:
    - Anchor fusion: concatenate anchor-conditioned node embeddings into each pharm token.
    - Multi-anchor pooling with geometry-aware offsets when 3D coordinates are available.
    - Pharm->drug attention: pharm tokens attend to drug node tokens.
    """
    def __init__(
        self,
        pharm_dim: int,
        drug_node_dim: int,
        dropout: float = 0.1,
        use_anchor: bool = True,
        pharm_use_graph_attn: bool = True,
        use_multi_anchor: bool = True,
        anchor_geom_dim: int = 4,
    ):
        super().__init__()
        self.use_anchor = bool(use_anchor)
        self.pharm_use_graph_attn = bool(pharm_use_graph_attn)
        self.use_multi_anchor = bool(use_multi_anchor)
        self.anchor_geom_dim = int(max(0, anchor_geom_dim))

        if self.use_anchor:
            self.fuse = nn.Linear(pharm_dim + drug_node_dim, pharm_dim)
            if self.anchor_geom_dim > 0:
                self.anchor_geom_proj = nn.Sequential(
                    nn.LayerNorm(self.anchor_geom_dim),
                    nn.Linear(self.anchor_geom_dim, pharm_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
            else:
                self.anchor_geom_proj = None
        else:
            self.fuse = None
            self.anchor_geom_proj = None

        if self.pharm_use_graph_attn:
            self.graph_attn = ScaledDotAttn(q_dim=pharm_dim, kv_dim=drug_node_dim, out_dim=pharm_dim, dropout=dropout)
        else:
            self.graph_attn = None

        self.drop = nn.Dropout(dropout)

    def _align_mask(self, pharm_mask: torch.Tensor, B: int, P: int) -> torch.Tensor:
        pm = pharm_mask
        if pm.dim() == 1:
            pm = pm.view(B, 1).expand(B, P)
        elif pm.size(0) != B or pm.size(1) != P:
            pm2 = pm.new_zeros((B, P))
            m0 = min(B, int(pm.size(0)))
            m1 = min(P, int(pm.size(1)))
            pm2[:m0, :m1] = pm[:m0, :m1]
            pm = pm2
        return pm

    def _pool_anchor_features(
        self,
        drug_node_repr: torch.Tensor,
        node_slices: List[Tuple[int, int]],
        pm: torch.Tensor,
        pharm_anchor: Optional[torch.Tensor] = None,
        pharm_multi_anchor: Optional[torch.Tensor] = None,
        pharm_multi_anchor_mask: Optional[torch.Tensor] = None,
        drug_node_pos: Optional[torch.Tensor] = None,
        pharm_center: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, P = int(pm.size(0)), int(pm.size(1))
        Dg = int(drug_node_repr.size(-1))
        anchor_vec = drug_node_repr.new_zeros((B, P, Dg))

        use_geom = (
            isinstance(drug_node_pos, torch.Tensor)
            and drug_node_pos.dim() == 2
            and int(drug_node_pos.shape[0]) == int(drug_node_repr.shape[0])
            and int(drug_node_pos.shape[1]) == 3
            and isinstance(pharm_center, torch.Tensor)
            and pharm_center.dim() == 3
            and int(pharm_center.size(0)) == B
            and int(pharm_center.size(1)) == P
            and int(pharm_center.size(2)) == 3
        )
        anchor_geom = drug_node_repr.new_zeros((B, P, self.anchor_geom_dim)) if use_geom and self.anchor_geom_dim > 0 else None

        if (
            self.use_multi_anchor
            and pharm_multi_anchor is not None
            and pharm_multi_anchor_mask is not None
            and pharm_multi_anchor.dim() == 3
            and pharm_multi_anchor_mask.dim() == 3
        ):
            K = int(pharm_multi_anchor.size(-1))
            ma = pharm_multi_anchor.long()
            mm = pharm_multi_anchor_mask.long()
            for b, (s, e) in enumerate(node_slices):
                nb = int(e - s)
                if nb <= 0:
                    continue
                valid_tok = torch.nonzero(pm[b] > 0, as_tuple=False).view(-1)
                for pidx in valid_tok.tolist():
                    cur_mask = (mm[b, pidx] > 0) & (ma[b, pidx] >= 0)
                    if not bool(cur_mask.any().item()):
                        continue
                    local_idx = ma[b, pidx][cur_mask].clamp(min=0, max=nb - 1)
                    if local_idx.numel() <= 0:
                        continue
                    global_idx = local_idx + int(s)
                    anchor_nodes = drug_node_repr.index_select(0, global_idx)
                    anchor_vec[b, pidx] = anchor_nodes.mean(dim=0)
                    if anchor_geom is not None:
                        pos_sel = drug_node_pos.index_select(0, global_idx).to(dtype=anchor_geom.dtype, device=anchor_geom.device)
                        pos_mean = pos_sel.mean(dim=0)
                        center = pharm_center[b, pidx].to(dtype=anchor_geom.dtype, device=anchor_geom.device)
                        rel = pos_mean - center
                        spread = torch.norm(pos_sel - pos_mean.view(1, -1), dim=-1).mean()
                        anchor_geom[b, pidx, :3] = rel
                        anchor_geom[b, pidx, 3] = spread
            return anchor_vec, anchor_geom

        if pharm_anchor is None:
            return anchor_vec, anchor_geom

        if pharm_anchor.dim() == 1:
            for b, (s, e) in enumerate(node_slices):
                nb = int(e - s)
                if nb <= 0:
                    continue
                a = int(pharm_anchor[b].item())
                if a < 0:
                    continue
                a = max(0, min(a, nb - 1))
                valid_tok = (pm[b] > 0)
                if valid_tok.any():
                    anchor_vec[b, valid_tok] = drug_node_repr[s + a].view(1, -1).expand(int(valid_tok.sum().item()), -1)
                    if anchor_geom is not None:
                        pos_mean = drug_node_pos[s + a].to(dtype=anchor_geom.dtype, device=anchor_geom.device).view(1, 3)
                        center = pharm_center[b, valid_tok].to(dtype=anchor_geom.dtype, device=anchor_geom.device)
                        anchor_geom[b, valid_tok, :3] = pos_mean.expand_as(center) - center
            return anchor_vec, anchor_geom

        pa = pharm_anchor.long()
        pa2 = pa.new_full((B, P), -1)
        m0 = min(B, int(pa.size(0)))
        m1 = min(P, int(pa.size(1)))
        pa2[:m0, :m1] = pa[:m0, :m1]
        pa = pa2
        for b, (s, e) in enumerate(node_slices):
            nb = int(e - s)
            if nb <= 0:
                continue
            valid_tok = (pm[b] > 0) & (pa[b] >= 0)
            if not valid_tok.any():
                continue
            idx = pa[b][valid_tok].clamp(min=0, max=nb - 1) + int(s)
            anchor_vec[b, valid_tok] = drug_node_repr.index_select(0, idx)
            if anchor_geom is not None:
                pos_sel = drug_node_pos.index_select(0, idx).to(dtype=anchor_geom.dtype, device=anchor_geom.device)
                center = pharm_center[b, valid_tok].to(dtype=anchor_geom.dtype, device=anchor_geom.device)
                anchor_geom[b, valid_tok, :3] = pos_sel - center
        return anchor_vec, anchor_geom

    def forward(
        self,
        pharm_tokens: torch.Tensor,
        pharm_mask: torch.Tensor,
        drug_node_repr: torch.Tensor,
        node_slices: List[Tuple[int, int]],
        pharm_anchor: Optional[torch.Tensor] = None,
        pharm_multi_anchor: Optional[torch.Tensor] = None,
        pharm_multi_anchor_mask: Optional[torch.Tensor] = None,
        drug_node_pos: Optional[torch.Tensor] = None,
        pharm_center: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        pharm_tokens: [B,P,Dp]
        pharm_mask: [B,P]
        drug_node_repr: [N, Dg]
        node_slices: list of (s,e) per graph
        """
        x = pharm_tokens

        if self.use_anchor and self.fuse is not None and (pharm_anchor is not None or pharm_multi_anchor is not None):
            B, P = int(x.size(0)), int(x.size(1))
            pm = self._align_mask(pharm_mask, B, P)
            anchor_vec, anchor_geom = self._pool_anchor_features(
                drug_node_repr=drug_node_repr,
                node_slices=node_slices,
                pm=pm,
                pharm_anchor=pharm_anchor,
                pharm_multi_anchor=pharm_multi_anchor,
                pharm_multi_anchor_mask=pharm_multi_anchor_mask,
                drug_node_pos=drug_node_pos,
                pharm_center=pharm_center,
            )
            x = torch.cat([x, anchor_vec], dim=-1)
            x = F.relu(self.fuse(self.drop(x)))
            if anchor_geom is not None and self.anchor_geom_proj is not None:
                x = x + self.anchor_geom_proj(anchor_geom)
            x = x * pharm_mask.unsqueeze(-1).float()

        if self.pharm_use_graph_attn and self.graph_attn is not None:
            B = len(node_slices)
            nmax = max((e - s) for (s, e) in node_slices)
            node_repr = drug_node_repr
            nodes = torch.zeros((B, nmax, node_repr.size(-1)), device=node_repr.device, dtype=node_repr.dtype)
            nmask = torch.zeros((B, nmax), device=node_repr.device, dtype=torch.long)
            for b, (s, e) in enumerate(node_slices):
                nb = e - s
                nodes[b, :nb] = node_repr[s:e]
                nmask[b, :nb] = 1

            x2 = self.graph_attn(x, nodes, kv_mask=nmask)
            x = x2 * pharm_mask.unsqueeze(-1).float()

        return x


class ProteinTokenEncoder(nn.Module):
    def __init__(self, in_dim: int, pool: str = "attn"):
        super().__init__()
        self.pool = pool
        self.attn = nn.Linear(in_dim, 1)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        tokens: [B, L, d]
        mask: [B, L] (0/1)
        """
        if self.pool == "mean":
            return masked_mean_pool(tokens, mask, dim=1)

        scores = self.attn(tokens).squeeze(-1)  # [B, L]
        weights = masked_softmax(scores, mask=mask, dim=1)
        z = torch.sum(tokens * weights.unsqueeze(-1), dim=1)
        return z


class BaselineConcatDTI(nn.Module):
    """
    Drug: graph GCN pooled (+ nodes)
    Pharm: pharm tokens (type emb) + optional anchor + optional pharm->drug attention
          + optional pharm<->protein cross-attention
    Protein: ESM2 tokens pooled
    Fusion: concat + MLP
    """
    def __init__(
        self,
        node_in: int = 6,
        gcn_hidden: int = 128,
        gcn_out: int = 128,
        gcn_layers: int = 2,
        num_pharm_types: int = 32,
        pharm_emb: int = 64,
        pharm_out: int = 64,
        esm_dim: int = 320,
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        # switches
        use_anchor: bool = True,
        pharm_use_graph_attn: bool = True,
        use_pp_attn: bool = True,
        # protein pooling
        prot_pool: str = "attn",
        # pocket settings
        use_pocket: bool = True,
        pocket_pool: str = "softmax",
        pocket_k: int = 32,
        pocket_min_keep: int = 8,
        pocket_tau: float = 0.3,
        pocket_detach_score: bool = True,
        pocket_dim: int = 64,
        pp_attn_dropout: Optional[float] = None,
    ):
        super().__init__()
        self.drug_gcn = DrugGCNEncoder(node_in=node_in, hidden=gcn_hidden, out=gcn_out, layers=gcn_layers, dropout=dropout, return_nodes=True)

        self.pharm_tok = PharmTokenEncoder(num_types=num_pharm_types, emb_dim=pharm_emb, out_dim=pharm_out, dropout=dropout)
        self.pharm_enh = PharmTokenEnhancer(
            pharm_dim=pharm_out,
            drug_node_dim=gcn_out,
            dropout=dropout,
            use_anchor=use_anchor,
            pharm_use_graph_attn=pharm_use_graph_attn,
        )

        self.use_pp_attn = bool(use_pp_attn)
        if self.use_pp_attn:
            self.pp_attn = ScaledDotAttn(q_dim=pharm_out, kv_dim=esm_dim, out_dim=pharm_out, dropout=float(pp_attn_dropout or dropout))
            # Outer gate to blend pp-attn output with original pharm tokens (stabilizes training)
            self.pp_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  # sigmoid~0.12
        else:
            self.pp_attn = None
            self.pp_gate = None

        self.prot_enc = ProteinTokenEncoder(in_dim=esm_dim, pool=prot_pool)

        # pseudo-pocket settings
        self.use_pocket = bool(use_pocket)
        self.pocket_pool = str(pocket_pool)
        self.pocket_k = int(pocket_k)
        self.pocket_min_keep = int(pocket_min_keep)
        self.pocket_tau = float(pocket_tau)
        self.pocket_detach_score = bool(pocket_detach_score)
        self.pocket_dim = int(pocket_dim)

        if self.use_pocket:
            self.pocket_proj = nn.Linear(pharm_out, self.pocket_dim)
            self.pocket_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  # sigmoid~0.12
        else:
            self.pocket_proj = None
            self.pocket_gate = None

        in_dim = gcn_out + pharm_out + esm_dim + (self.pocket_dim if self.use_pocket else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        # drug graph encoder
        z_graph, node_repr = self.drug_gcn(batch["node_feat"], batch["edge_index"], batch["node_slices"])  # [B, gcn_out], [N, gcn_out]

        # pharm tokens + enhance
        pharm_tokens = self.pharm_tok(batch["pharm_type"], batch["pharm_mask"])  # [B,P,Dp]
        pharm_anchor = batch.get("pharm_anchor", None)
        pharm_tokens = self.pharm_enh(
            pharm_tokens,
            batch["pharm_mask"],
            node_repr,
            batch["node_slices"],
            pharm_anchor=pharm_anchor,
        )  # [B,P,Dp]

        pocket_vec = None

        if self.use_pp_attn and (self.pp_attn is not None):
            # pharm -> protein cross-attention (and get attention weights for pseudo-pocket)
            pp_out, attn_pp = self.pp_attn(
                pharm_tokens, batch["esm2_tok"], kv_mask=batch["esm2_mask"], return_attn=True
            )  # pp_out: [B,P,Dp], attn_pp: [B,P,L]

            # Outer residual-gate: keep original pharm_tokens as the base; let cross-attn contribute gradually.
            alpha = torch.sigmoid(self.pp_gate) if (self.pp_gate is not None) else 1.0
            pharm_tokens = (1.0 - alpha) * pharm_tokens + alpha * pp_out
            pharm_tokens = pharm_tokens * batch["pharm_mask"].unsqueeze(-1).float()

            if self.use_pocket and self.pocket_proj is not None:
                # compute residue scores from attention weights (conditioned on pharm tokens)
                prot_mask = batch["esm2_mask"].bool()
                pm = batch["pharm_mask"].float().unsqueeze(-1)  # [B,P,1]
                w = attn_pp * pm
                denom = pm.sum(dim=1).clamp_min(1.0)
                scores = w.sum(dim=1) / denom  # [B,L]
                scores = scores.masked_fill(~prot_mask, float("-inf"))
                if self.pocket_detach_score:
                    scores = scores.detach()
                neg = torch.finfo(scores.dtype).min
                scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, neg))

                # pool on value-projected protein tokens (aligned with cross-attn space)
                prot_v = self.pp_attn.v_proj(batch["esm2_tok"])
                if self.pocket_pool.lower() == "hard":
                    pocket_mask = build_topk_mask(scores, prot_mask, k=self.pocket_k, min_keep=self.pocket_min_keep)
                    pocket_vec_raw = masked_mean_pool(prot_v, pocket_mask.long(), dim=1)  # [B, Dp]
                else:
                    pocket_w = safe_softmax(scores, prot_mask, tau=self.pocket_tau)  # [B,L]
                    pocket_vec_raw = torch.sum(prot_v * pocket_w.unsqueeze(-1), dim=1)  # [B, Dp]

                pocket_vec = self.pocket_proj(pocket_vec_raw)  # [B, pocket_dim]
                # apply gate (if enabled)
                if self.pocket_gate is not None:
                    pocket_vec = torch.sigmoid(self.pocket_gate) * pocket_vec

        z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)  # [B,Dp]
        z_esm = self.prot_enc(batch["esm2_tok"], batch["esm2_mask"])          # [B,De]

        feats = [z_graph, z_pharm, z_esm]
        if self.use_pocket and pocket_vec is not None:
            feats.append(pocket_vec)

        h = torch.cat(feats, dim=-1)
        logit = self.mlp(h).squeeze(-1)
        return logit
