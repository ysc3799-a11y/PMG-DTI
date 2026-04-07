# -*- coding: utf-8 -*-
"""
3-view pocket-enhanced DTI model
--------------------------------
Views:
1) h_global : drug graph + pharm + protein global fusion
2) h_pair   : local summary = pair evidence + coverage-aware summary
3) h_set    : top high-score pair tokens -> set encoder relation summary

Recommended usage:
- replace the import in your training script with:
    from model.model_gated_real_pocket_seqdist_3view_hset import BaselineGatedDTIRealPocket
- keep your current dataset / collate unchanged
- start with SET_TOP_PAIR_PER_POCKET=4, SET_MAX_TOKENS=12, SET_NUM_LAYERS=1
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import (
    DrugGraphEncoder,
    PharmTokenEncoder,
    PharmTokenEnhancer,
    ProteinTokenEncoder,
    ScaledDotAttn,
    masked_softmax,
)


# =========================================================
# helpers
# =========================================================

def masked_mean_pool(x: torch.Tensor, mask: torch.Tensor, dim: int = 1, eps: float = 1e-8) -> torch.Tensor:
    if mask.dtype != torch.bool:
        mask = mask > 0
    m = mask.unsqueeze(-1).float()
    s = (x * m).sum(dim=dim)
    d = m.sum(dim=dim).clamp(min=eps)
    return s / d


def lengths_to_batch(lengths: torch.Tensor) -> torch.Tensor:
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
    if node_x.numel() == 0:
        g = int(num_graphs or 0)
        d = int(node_x.shape[-1]) if node_x.dim() == 2 else 0
        return torch.zeros((g, d), dtype=torch.float32, device=node_x.device)

    batch_index = batch_index.long().view(-1)
    if num_graphs is None:
        num_graphs = int(batch_index.max().item()) + 1 if batch_index.numel() > 0 else 0

    out = torch.zeros((num_graphs, node_x.size(-1)), dtype=node_x.dtype, device=node_x.device)
    cnt = torch.zeros((num_graphs, 1), dtype=node_x.dtype, device=node_x.device)

    valid = (batch_index >= 0) & (batch_index < int(num_graphs))
    if valid.numel() > 0 and not torch.all(valid):
        batch_index = batch_index[valid]
        node_x = node_x[valid]

    if batch_index.numel() > 0:
        out.index_add_(0, batch_index, node_x)
    cnt.index_add_(
        0,
        batch_index,
        torch.ones((batch_index.numel(), 1), dtype=node_x.dtype, device=node_x.device),
    )
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
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if mask.dtype != torch.bool:
            mask = mask > 0
        score = self.scorer(x).squeeze(-1)
        alpha = masked_softmax(score, mask=mask, dim=1)
        alpha = alpha.unsqueeze(-1)
        return (x * alpha).sum(dim=1)


class SimpleGatedFusion(nn.Module):
    def __init__(self, dim: int, n_inputs: int, dropout: float = 0.1):
        super().__init__()
        self.n_inputs = int(n_inputs)
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
            gated.append(x * self.gates[i](x))
        return self.out(torch.cat(gated, dim=-1))


class DenseFeatureEncoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int, dropout: float = 0.1):
        super().__init__()
        self.in_dim = int(max(0, in_dim))
        self.out_dim = int(out_dim)
        if self.in_dim > 0:
            self.net = nn.Sequential(
                nn.LayerNorm(self.in_dim),
                nn.Linear(self.in_dim, int(hidden)),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(int(hidden), self.out_dim),
            )
        else:
            self.net = None

    def forward(self, x: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
        if self.net is None or x is None:
            return ref.new_zeros((int(ref.shape[0]), self.out_dim))
        if x.dim() != 2 or int(x.shape[1]) != self.in_dim:
            return ref.new_zeros((int(ref.shape[0]), self.out_dim))
        # Dense global descriptors can contain large magnitudes (for example very large
        # molecular weights). Under AMP, casting them to fp16 before LayerNorm can overflow
        # and produce NaNs. Keep this encoder in fp32, then cast the encoded vector back.
        x32 = torch.nan_to_num(
            x.to(device=ref.device, dtype=torch.float32),
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        with torch.autocast(device_type=ref.device.type, enabled=False):
            out = self.net(x32)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out.to(dtype=ref.dtype)


class ResidualPairGate(nn.Module):
    """h_out = h_main + gate(h_main, h_add) * h_add"""
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

    def forward(self, h_main: torch.Tensor, h_add: torch.Tensor) -> torch.Tensor:
        g = self.gate(torch.cat([h_main, h_add], dim=-1))
        return h_main + g * h_add


# =========================================================
# pocket encoder
# =========================================================

class EdgeFeatLayer(nn.Module):
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
            pad = torch.zeros((int(edge_feat.shape[0]), expected_edge_dim - cur_edge_dim), dtype=edge_feat.dtype, device=edge_feat.device)
            edge_feat = torch.cat([edge_feat, pad], dim=1)
        elif cur_edge_dim > expected_edge_dim:
            edge_feat = edge_feat[:, :expected_edge_dim]

        xs = self.src_proj(x.index_select(0, src))
        xd = self.dst_proj(x.index_select(0, dst))
        ee = self.edge_proj(edge_feat)
        msg = self.msg_proj(torch.cat([xs, xd, ee], dim=-1))

        agg = torch.zeros_like(x)
        agg.index_add_(0, dst, msg.to(dtype=agg.dtype))

        deg = torch.zeros((x.size(0), 1), dtype=x.dtype, device=x.device)
        deg.index_add_(0, dst, torch.ones((dst.numel(), 1), dtype=x.dtype, device=x.device))
        agg = agg / deg.clamp(min=1.0)

        z = self.upd(torch.cat([x, agg], dim=-1))
        return self.norm(x + self.dropout(z))


class PocketNodeEncoder(nn.Module):
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
# pair local view
# =========================================================

class PairScoreLocal(nn.Module):
    """
    Pair-level local matcher.
    If return_details=True, it also returns the internal pair map needed by h_set.
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

    def forward(self, q_tokens: torch.Tensor, r_tokens: torch.Tensor, return_details: bool = False):
        if q_tokens.size(1) == 0 or r_tokens.size(1) == 0:
            zero = torch.zeros((1, q_tokens.size(-1)), dtype=q_tokens.dtype, device=q_tokens.device)
            if return_details:
                return {
                    "pair_g": zero,
                    "pair_score": torch.zeros((0, 0), dtype=q_tokens.dtype, device=q_tokens.device),
                    "pair_feat": torch.zeros((0, 0, q_tokens.size(-1)), dtype=q_tokens.dtype, device=q_tokens.device),
                    "q_tokens": torch.zeros((0, q_tokens.size(-1)), dtype=q_tokens.dtype, device=q_tokens.device),
                    "r_tokens": torch.zeros((0, q_tokens.size(-1)), dtype=q_tokens.dtype, device=q_tokens.device),
                }
            return zero

        q = q_tokens.squeeze(0)
        r = r_tokens.squeeze(0)

        qh = self.q_proj(q)
        rh = self.r_proj(r)
        M, D = qh.shape
        N = rh.shape[0]

        qe = qh.unsqueeze(1).expand(M, N, D)
        re = rh.unsqueeze(0).expand(M, N, D)

        pair_in = torch.cat([qe, re, qe * re, torch.abs(qe - re)], dim=-1)
        pair_score = self.score_mlp(pair_in).squeeze(-1)
        pair_feat = self.pair_feat_mlp(pair_in)

        alpha = masked_softmax(pair_score.reshape(-1), dim=0).view(M, N, 1).to(dtype=pair_feat.dtype)
        pooled_pair = (pair_feat * alpha).sum(dim=(0, 1), keepdim=False).view(1, D)

        q_ctx = q.mean(dim=0, keepdim=True)
        r_ctx = r.mean(dim=0, keepdim=True)
        pair_g = self.norm(self.out(torch.cat([pooled_pair, q_ctx, r_ctx], dim=-1)))

        if return_details:
            return {
                "pair_g": pair_g,
                "pair_score": pair_score,
                "pair_feat": pair_feat,
                "q_tokens": qh,
                "r_tokens": rh,
            }
        return pair_g



class CoverageStatsEncoder(nn.Module):
    """
    Build a coverage-aware pocket summary from the pair score map.

    Idea:
    - h_pair should answer: "is local evidence strong enough, and is coverage complete enough?"
    - It therefore uses:
        * pharm coverage   : max_j sigmoid(s_ij)
        * pocket coverage  : max_i sigmoid(s_ij)
        * weighted q/r contexts
        * simple summary stats / coverage ratios
    - h_set, in contrast, will only model top-pair relations.
    """
    def __init__(self, dim: int, dropout: float = 0.1, cover_thr: float = 0.5):
        super().__init__()
        self.cover_thr = float(cover_thr)
        self.stats_proj = nn.Sequential(
            nn.Linear(8, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.out = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def _safe_std(x: torch.Tensor) -> torch.Tensor:
        if x.numel() <= 1:
            return torch.zeros((1,), dtype=x.dtype, device=x.device)
        return x.std(unbiased=False).view(1)

    def forward(
        self,
        pair_score: torch.Tensor,
        q_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        graph_g: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if pair_score.numel() == 0:
            d = int(graph_g.shape[-1])
            zero = torch.zeros((1, d), dtype=graph_g.dtype, device=graph_g.device)
            stats = {
                "pharm_cov": torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
                "pocket_cov": torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
            }
            return zero, stats

        pair_prob = torch.sigmoid(pair_score)               # [M, N]
        pharm_cov = pair_prob.max(dim=1).values             # [M]
        pocket_cov = pair_prob.max(dim=0).values            # [N]

        pharm_w = pharm_cov / pharm_cov.sum().clamp(min=1e-8)
        pocket_w = pocket_cov / pocket_cov.sum().clamp(min=1e-8)

        q_ctx = (q_tokens * pharm_w.unsqueeze(-1)).sum(dim=0, keepdim=True)   # [1, D]
        r_ctx = (r_tokens * pocket_w.unsqueeze(-1)).sum(dim=0, keepdim=True)  # [1, D]
        graph_ctx = graph_g

        stats_raw = torch.cat([
            pharm_cov.mean().view(1),
            pharm_cov.max().view(1),
            self._safe_std(pharm_cov),
            (pharm_cov >= self.cover_thr).float().mean().view(1),
            pocket_cov.mean().view(1),
            pocket_cov.max().view(1),
            self._safe_std(pocket_cov),
            (pocket_cov >= self.cover_thr).float().mean().view(1),
        ], dim=0).view(1, 8)
        stats_emb = self.stats_proj(stats_raw)

        cov_g = self.norm(self.out(torch.cat([q_ctx, r_ctx, graph_ctx, stats_emb], dim=-1)))
        stats = {
            "pharm_cov": pharm_cov,
            "pocket_cov": pocket_cov,
            "stats_raw": stats_raw,
        }
        return cov_g, stats

# =========================================================
# h_set: top-pair tokenizer + set encoder
# =========================================================

class TopPairTokenizer(nn.Module):
    """
    Build pair-set tokens from top-scoring pharm-pocket pairs inside one pocket graph.
    """
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score_proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.Tanh(),
        )
        self.token_mlp = nn.Sequential(
            nn.Linear(dim * 5, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        pair_score: torch.Tensor,
        pair_feat: torch.Tensor,
        q_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        graph_g: torch.Tensor,
        top_t: int = 4,
        max_per_q: int = 2,
        max_per_r: int = 1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            tokens: [T, D]
            scores: [T]
        """
        if pair_score.numel() == 0 or int(top_t) <= 0:
            d = int(graph_g.shape[-1])
            return (
                torch.zeros((0, d), dtype=graph_g.dtype, device=graph_g.device),
                torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
            )

        M, N = pair_score.shape
        flat_score = pair_score.reshape(-1)
        if flat_score.numel() == 0:
            d = int(graph_g.shape[-1])
            return (
                torch.zeros((0, d), dtype=graph_g.dtype, device=graph_g.device),
                torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
            )

        # Diversity-aware top-k:
        # limit how many pairs can come from the same pharm token index
        # and from the same pocket node index.
        order = torch.argsort(flat_score, descending=True)
        selected: List[int] = []
        q_counts: Dict[int, int] = {}
        r_counts: Dict[int, int] = {}
        per_q = int(max_per_q)
        per_r = int(max_per_r)
        target = int(max(top_t, 0))
        for oid in order.tolist():
            if len(selected) >= target:
                break
            i = int(oid // N)
            j = int(oid % N)
            if per_q > 0 and q_counts.get(i, 0) >= per_q:
                continue
            if per_r > 0 and r_counts.get(j, 0) >= per_r:
                continue
            selected.append(int(oid))
            q_counts[i] = q_counts.get(i, 0) + 1
            r_counts[j] = r_counts.get(j, 0) + 1

        if len(selected) == 0:
            d = int(graph_g.shape[-1])
            return (
                torch.zeros((0, d), dtype=graph_g.dtype, device=graph_g.device),
                torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
            )

        topi = torch.tensor(selected, dtype=torch.long, device=pair_score.device)
        topv = flat_score.index_select(0, topi)
        # Use sigmoid(raw logit) as cross-pocket comparable ranking score.
        top_score = torch.sigmoid(topv)
        k = int(topi.numel())

        tokens = []
        graph_ctx = graph_g.squeeze(0)
        for n in range(int(k)):
            idx = int(topi[n].item())
            i = idx // N
            j = idx % N
            score_emb = self.score_proj(top_score[n].view(1, 1)).view(-1)
            tok = self.token_mlp(torch.cat([
                pair_feat[i, j],
                q_tokens[i],
                r_tokens[j],
                graph_ctx,
                score_emb,
            ], dim=-1))
            tokens.append(tok)

        tokens = torch.stack(tokens, dim=0)
        tokens = self.norm(tokens)
        return tokens, top_score


class PairSetEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        dropout: float = 0.1,
        use_cls: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.use_cls = bool(use_cls)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=self.dim,
            nhead=int(num_heads),
            dim_feedforward=self.dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=int(num_layers))

        if self.use_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, self.dim))
            nn.init.normal_(self.cls, mean=0.0, std=0.02)
            self.out_norm = nn.LayerNorm(self.dim)
            self.pool = None
        else:
            self.cls = None
            self.out_norm = nn.LayerNorm(self.dim)
            self.pool = AttnPool1D(self.dim, dropout=dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # tokens: [B, L, D], mask: [B, L] bool
        if tokens.numel() == 0:
            return torch.zeros((tokens.size(0), self.dim), dtype=tokens.dtype, device=tokens.device)
        if mask.dtype != torch.bool:
            mask = mask > 0

        x = tokens
        src_key_padding_mask = ~mask
        if self.use_cls:
            B = x.size(0)
            cls = self.cls.expand(B, 1, -1)
            x = torch.cat([cls, x], dim=1)
            cls_mask = torch.ones((B, 1), dtype=torch.bool, device=x.device)
            src_key_padding_mask = ~torch.cat([cls_mask, mask], dim=1)

        z = self.encoder(x, src_key_padding_mask=src_key_padding_mask)

        if self.use_cls:
            return self.out_norm(z[:, 0, :])
        pooled = self.pool(z, mask)
        return self.out_norm(pooled)


# =========================================================
# main model
# =========================================================

class BaselineGatedDTIRealPocket(nn.Module):
    """
    3-view model:
      h_global : global drug/pharm/protein fusion
      h_pair   : local summary (pair evidence + coverage)
      h_set    : top-pair relation/set-pattern summary

    final:
      h_local = h_pair + gate(h_pair, h_set) * h_set
      h = h_global + gate(h_global, h_local) * h_local
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
        prot_dim: int = 128,
        prot_pool: str = "attn",
        use_pp_attn: bool = False,
        use_pocket: bool = False,
        pocket_node_in: Optional[int] = None,
        pocket_edge_feat_dim: int = 2,
        pocket_hidden: int = 128,
        pocket_layers: int = 2,
        fusion_dim: Optional[int] = 256,
        dropout: float = 0.1,
        **kwargs,
    ):
        super().__init__()

        if pharm_num_types is None:
            pharm_num_types = int(kwargs.get("num_pharm_types", 64))
        node_hidden = int(kwargs.get("gcn_hidden", node_hidden))
        drug_layers = int(kwargs.get("gcn_layers", drug_layers))
        drug_out = int(kwargs.get("gcn_out", node_hidden))
        drug_edge_feat_dim = int(kwargs.get("drug_edge_feat_dim", 12))
        drug_edge_hidden = int(kwargs.get("drug_edge_hidden", 32))

        pharm_dim = int(kwargs.get("pharm_out", pharm_dim))
        pharm_emb_dim = int(kwargs.get("pharm_emb", pharm_dim))
        pharm_num_families = int(kwargs.get("pharm_num_families", kwargs.get("PHARM_NUM_FAMILIES", 0)))
        pharm_context_dim = int(kwargs.get("pharm_context_dim", kwargs.get("PHARM_CONTEXT_DIM", 0)))
        pharm_geom_dim = int(kwargs.get("pharm_geom_dim", kwargs.get("PHARM_GEOM_DIM", 0)))
        pharm_center_dim = int(kwargs.get("pharm_center_dim", kwargs.get("PHARM_CENTER_DIM", 0)))
        pharm_family_emb_dim = int(kwargs.get("pharm_family_emb_dim", kwargs.get("PHARM_FAMILY_EMB_DIM", 24)))
        use_anchor = bool(int(kwargs.get("use_anchor", 1)))
        use_multi_anchor = bool(int(kwargs.get("use_multi_anchor", kwargs.get("USE_MULTI_ANCHOR", 1))))
        use_drug_global_aux = bool(int(kwargs.get("use_drug_global_aux", kwargs.get("USE_DRUG_GLOBAL_AUX", 1))))
        drug_fp_dim = int(kwargs.get("drug_fp_dim", kwargs.get("DRUG_FP_DIM", 0)))
        drug_desc_dim = int(kwargs.get("drug_desc_dim", kwargs.get("DRUG_DESC_DIM", 0)))
        drug_aux_hidden = int(kwargs.get("drug_aux_hidden", kwargs.get("DRUG_AUX_HIDDEN", max(drug_out, 128))))

        pocket_hidden = int(kwargs.get("pocket_gnn_hidden", pocket_hidden))
        pocket_layers = int(kwargs.get("pocket_gnn_layers", pocket_layers))
        pocket_node_proj = int(kwargs.get("pocket_node_proj", pocket_hidden))
        pocket_edge_hidden = int(kwargs.get("pocket_edge_hidden", pocket_edge_feat_dim))
        pocket_struct_dim = int(kwargs.get("pocket_struct_dim", 29))
        pocket_use_struct_feat = bool(int(kwargs.get("pocket_use_struct_feat", 1)))
        pocket_score_prior = bool(int(kwargs.get("pocket_score_prior", 1)))
        pocket_score_scale = float(kwargs.get("pocket_score_scale", 1.0))
        pocket_inst_pool = str(kwargs.get("pocket_inst_pool", kwargs.get("POCKET_INST_POOL", "softmax"))).lower()
        if pocket_inst_pool in ("attn", "softmax"):
            pocket_inst_pool = "softmax"
        elif pocket_inst_pool in ("mean", "avg"):
            pocket_inst_pool = "mean"
        elif pocket_inst_pool == "max":
            pocket_inst_pool = "max"
        else:
            pocket_inst_pool = "softmax"
        pocket_tau = float(kwargs.get("pocket_tau", kwargs.get("POCKET_TAU", 1.0)))
        strict_pocket_graph_mapping = bool(
            int(kwargs.get("strict_pocket_graph_mapping", kwargs.get("STRICT_POCKET_GRAPH_MAPPING", 1)))
        )
        use_graph_branch = kwargs.get("use_graph_branch", kwargs.get("USE_GRAPH_BRANCH", 1))
        use_pharm_branch = kwargs.get("use_pharm_branch", kwargs.get("USE_PHARM_BRANCH", 1))
        use_sequence_branch = kwargs.get("use_sequence_branch", kwargs.get("USE_SEQUENCE_BRANCH", 1))
        fusion_strategy = str(kwargs.get("fusion_strategy", kwargs.get("FUSION_STRATEGY", "hierarchical"))).lower()
        if fusion_strategy in ("hier", "two_stage", "two-stage"):
            fusion_strategy = "hierarchical"
        elif fusion_strategy in ("concat", "concatenate", "direct_concat", "direct-concat"):
            fusion_strategy = "concat"
        elif fusion_strategy in ("single", "single_stage", "single-stage", "single_stage_fusion"):
            fusion_strategy = "single_stage"
        elif fusion_strategy in ("mean", "sum", "mean_sum", "mean-sum", "mean/sum"):
            fusion_strategy = "mean_sum"
        else:
            fusion_strategy = "hierarchical"

        gate_hidden = int(kwargs.get("gate_hidden", fusion_dim or 256))
        mlp_hidden = int(kwargs.get("mlp_hidden", fusion_dim or 256))
        pp_attn_dropout = float(kwargs.get("pp_attn_dropout", dropout))

        # h_set configs
        use_set_view = kwargs.get("use_set_view", kwargs.get("USE_SET_VIEW", 1))
        self.use_set_view = bool(int(use_set_view))
        self.set_top_pair_per_pocket = int(kwargs.get("set_top_pair_per_pocket", kwargs.get("SET_TOP_PAIR_PER_POCKET", 4)))
        self.set_max_tokens = int(kwargs.get("set_max_tokens", kwargs.get("SET_MAX_TOKENS", 12)))
        self.set_max_per_pharm_token = int(
            kwargs.get("set_max_per_pharm_token", kwargs.get("SET_MAX_PER_PHARM_TOKEN", 2))
        )
        self.set_max_per_pocket_node = int(
            kwargs.get("set_max_per_pocket_node", kwargs.get("SET_MAX_PER_POCKET_NODE", 1))
        )
        self.set_num_layers = int(kwargs.get("set_num_layers", kwargs.get("SET_NUM_LAYERS", 1)))
        self.set_num_heads = int(kwargs.get("set_num_heads", kwargs.get("SET_NUM_HEADS", 4)))
        self.set_dropout = float(kwargs.get("set_dropout", kwargs.get("SET_DROPOUT", dropout)))
        self.set_use_cls = bool(int(kwargs.get("set_use_cls", kwargs.get("SET_USE_CLS", 1))))
        self.set_gate_init_bias = float(kwargs.get("set_gate_init_bias", kwargs.get("SET_GATE_INIT_BIAS", -2.0)))
        self.use_pair_coverage = bool(int(kwargs.get("use_pair_coverage", kwargs.get("USE_PAIR_COVERAGE", 1))))
        self.coverage_thr = float(kwargs.get("coverage_thr", kwargs.get("COVERAGE_THR", 0.5)))

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
        # Keep token-level protein dim and optional pooled projection dim explicit.
        self.prot_token_dim = int(prot_in_dim)
        prot_out_dim_cfg = int(kwargs.get("prot_dim", kwargs.get("PROT_DIM", prot_dim)))
        if prot_out_dim_cfg > 0 and prot_out_dim_cfg != self.prot_token_dim:
            self.prot_proj = nn.Sequential(
                nn.Linear(self.prot_token_dim, prot_out_dim_cfg),
                nn.LayerNorm(prot_out_dim_cfg),
            )
            self.prot_out_dim = int(prot_out_dim_cfg)
        else:
            self.prot_proj = None
            self.prot_out_dim = int(self.prot_token_dim)

        self.use_pp_attn = bool(use_pp_attn)
        self.use_pocket = bool(use_pocket)
        self.use_graph_branch = bool(int(use_graph_branch))
        self.use_pharm_branch = bool(int(use_pharm_branch))
        self.use_sequence_branch = bool(int(use_sequence_branch))
        self.fusion_strategy = str(fusion_strategy)
        self.strict_pocket_graph_mapping = strict_pocket_graph_mapping
        self.pharm_dim = int(pharm_dim)
        self.pocket_use_struct_feat = bool(pocket_use_struct_feat)

        inferred_view_dim = max(int(drug_out), int(pharm_dim), int(prot_in_dim), 128)
        self.view_dim = int(fusion_dim) if fusion_dim is not None else inferred_view_dim

        # global branches
        self.drug_gcn = DrugGraphEncoder(
            node_in=node_in,
            hidden=node_hidden,
            out_dim=drug_out,
            num_layers=drug_layers,
            dropout=drug_dropout,
            return_nodes=True,
            edge_feat_dim=drug_edge_feat_dim,
            edge_hidden=drug_edge_hidden,
        )

        self.pharm_emb = PharmTokenEncoder(
            num_types=pharm_num_types,
            emb_dim=pharm_emb_dim,
            out_dim=pharm_dim,
            dropout=dropout,
            num_families=pharm_num_families,
            family_emb_dim=pharm_family_emb_dim,
            context_dim=pharm_context_dim,
            geom_dim=pharm_geom_dim,
            center_dim=pharm_center_dim,
        )
        self.pharm_enh = PharmTokenEnhancer(
            pharm_dim=pharm_dim,
            drug_node_dim=drug_out,
            dropout=dropout,
            use_anchor=use_anchor,
            pharm_use_graph_attn=pharm_use_graph_attn,
            use_multi_anchor=use_multi_anchor,
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

        self.prot_enc = ProteinTokenEncoder(in_dim=prot_in_dim, pool=prot_pool)

        if self.use_pp_attn:
            self.pp_attn = ScaledDotAttn(
                q_dim=pharm_dim,
                kv_dim=self.prot_token_dim,
                out_dim=pharm_dim,
                dropout=pp_attn_dropout,
            )
            self.pp_to_prot = nn.Linear(pharm_dim, self.prot_token_dim)
            self.pp_ln = nn.LayerNorm(self.prot_token_dim)
            self.pp_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        else:
            self.pp_attn = None
            self.pp_to_prot = None
            self.pp_ln = None
            self.pp_gate = None

        # global view
        self.use_drug_global_aux = bool(use_drug_global_aux) and (int(drug_fp_dim) > 0 or int(drug_desc_dim) > 0)
        if self.use_drug_global_aux:
            self.drug_desc_encoder = DenseFeatureEncoder(
                in_dim=drug_desc_dim,
                out_dim=drug_out,
                hidden=drug_aux_hidden,
                dropout=dropout,
            )
            self.drug_fp_encoder = DenseFeatureEncoder(
                in_dim=drug_fp_dim,
                out_dim=drug_out,
                hidden=drug_aux_hidden,
                dropout=dropout,
            )
            self.drug_aux_fuse = SimpleGatedFusion(drug_out, n_inputs=3, dropout=dropout)
        else:
            self.drug_desc_encoder = None
            self.drug_fp_encoder = None
            self.drug_aux_fuse = None
        self.graph_align = nn.Linear(drug_out, self.view_dim)
        self.pharm_align = nn.Linear(pharm_dim, self.view_dim)
        self.prot_align = nn.Linear(self.prot_out_dim, self.view_dim)
        self.global_fuse = SimpleGatedFusion(self.view_dim, n_inputs=3, dropout=dropout)

        # pocket local views
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
            self.coverage_encoder = CoverageStatsEncoder(
                self.view_dim,
                dropout=dropout,
                cover_thr=self.coverage_thr,
            )
            self.hpair_pocket_fuse = nn.Sequential(
                nn.Linear(self.view_dim * 3, self.view_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.view_dim, self.view_dim),
                nn.LayerNorm(self.view_dim),
            )
            self.pocket_relevance = nn.Sequential(
                nn.Linear(self.view_dim * 4, self.view_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.view_dim, 1),
            )

            if self.use_set_view:
                self.top_pair_tokenizer = TopPairTokenizer(self.view_dim, dropout=self.set_dropout)
                self.set_encoder = PairSetEncoder(
                    dim=self.view_dim,
                    num_layers=self.set_num_layers,
                    num_heads=self.set_num_heads,
                    dropout=self.set_dropout,
                    use_cls=self.set_use_cls,
                )
            else:
                self.top_pair_tokenizer = None
                self.set_encoder = None
        else:
            self.pocket_enc = None
            self.pocket_edge_proj = None
            self.pocket_node_align = None
            self.pocket_graph_align = None
            self.pair_cross = None
            self.coverage_encoder = None
            self.hpair_pocket_fuse = None
            self.pocket_relevance = None
            self.top_pair_tokenizer = None
            self.set_encoder = None

        self.use_pocket_score_prior = bool(pocket_score_prior)
        self.pocket_score_scale = float(pocket_score_scale)
        self.pocket_inst_pool = str(pocket_inst_pool)
        self.pocket_tau = float(max(1e-6, pocket_tau))

        self.set_residual = ResidualPairGate(self.view_dim, hidden=gate_hidden, dropout=dropout)
        self.pair_residual = ResidualPairGate(self.view_dim, hidden=gate_hidden, dropout=dropout)
        self.single_stage_fuse = SimpleGatedFusion(self.view_dim, n_inputs=3, dropout=dropout)
        # Cold-start suppress h_set injection at the beginning.
        set_last_linear = None
        for mod in reversed(list(self.set_residual.gate)):
            if isinstance(mod, nn.Linear):
                set_last_linear = mod
                break
        if set_last_linear is not None:
            nn.init.zeros_(set_last_linear.weight)
            nn.init.constant_(set_last_linear.bias, self.set_gate_init_bias)

        self.classifier = nn.Sequential(
            nn.Linear(self.view_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )
        self.concat_classifier = nn.Sequential(
            nn.Linear(self.view_dim * 3, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    # -----------------------------------------------------
    # global branch
    # -----------------------------------------------------
    def _run_drug(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        edge_attr = batch.get("edge_attr", None)
        node_repr, z_graph = self.drug_gcn(
            batch["node_feat"],
            batch["edge_index"],
            batch["node_slices"],
            edge_attr=edge_attr,
        )
        if self.use_drug_global_aux and self.drug_aux_fuse is not None:
            z_desc = self.drug_desc_encoder(batch.get("drug_desc", None), z_graph) if self.drug_desc_encoder is not None else torch.zeros_like(z_graph)
            z_fp = self.drug_fp_encoder(batch.get("drug_fp", None), z_graph) if self.drug_fp_encoder is not None else torch.zeros_like(z_graph)
            z_graph = self.drug_aux_fuse([z_graph, z_desc, z_fp])
        if not self.use_graph_branch:
            node_repr = torch.zeros_like(node_repr)
            z_graph = torch.zeros_like(z_graph)
        return node_repr, z_graph

    def _run_pharm(self, batch: Dict[str, torch.Tensor], drug_node_repr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(batch["pharm_mask"].shape[0])
        P = int(batch["pharm_mask"].shape[1])
        if not self.use_pharm_branch:
            zero_tok = torch.zeros(
                (B, P, self.pharm_dim),
                dtype=self.pharm_emb.proj.weight.dtype,
                device=batch["pharm_mask"].device,
            )
            zero_vec = torch.zeros(
                (B, self.pharm_dim),
                dtype=zero_tok.dtype,
                device=zero_tok.device,
            )
            return zero_tok, zero_vec

        pharm_tokens = self.pharm_emb(
            batch["pharm_type"],
            batch["pharm_mask"],
            pharm_family=batch.get("pharm_family", None),
            pharm_context=batch.get("pharm_context", None),
            pharm_geom=batch.get("pharm_geom", None),
            pharm_center=batch.get("pharm_center", None),
        )
        if self.use_graph_branch:
            pharm_anchor = batch.get("pharm_anchor", None)
            pharm_tokens = self.pharm_enh(
                pharm_tokens,
                batch["pharm_mask"],
                drug_node_repr,
                batch["node_slices"],
                pharm_anchor=pharm_anchor,
                pharm_multi_anchor=batch.get("pharm_multi_anchor", None),
                pharm_multi_anchor_mask=batch.get("pharm_multi_anchor_mask", None),
                drug_node_pos=batch.get("node_pos", None),
                pharm_center=batch.get("pharm_center", None),
            )

        if self.pharm_self_attn is not None:
            pharm_mask = batch["pharm_mask"].bool()
            if not pharm_mask.any(dim=1).all():
                # MultiheadAttention cannot accept rows where every position is padded.
                pharm_mask_safe = pharm_mask.clone()
                empty = ~pharm_mask.any(dim=1)
                pharm_mask_safe[empty, 0] = True
            else:
                pharm_mask_safe = pharm_mask
            z, _ = self.pharm_self_attn(
                pharm_tokens,
                pharm_tokens,
                pharm_tokens,
                key_padding_mask=~pharm_mask_safe,
                need_weights=False,
            )
            pharm_tokens = self.pharm_self_ln(pharm_tokens + z)
            pharm_tokens = pharm_tokens * pharm_mask.unsqueeze(-1).float()

        if self.pharm_pool_mode == "attn":
            z_pharm = self.pharm_pool_attn(pharm_tokens, batch["pharm_mask"])
        else:
            z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)
        return pharm_tokens, z_pharm

    def _run_protein(self, batch: Dict[str, torch.Tensor], pharm_tokens: Optional[torch.Tensor] = None) -> torch.Tensor:
        if not self.use_sequence_branch:
            B = int(batch["esm2_mask"].shape[0])
            return torch.zeros(
                (B, self.prot_out_dim),
                dtype=self.prot_align.weight.dtype,
                device=batch["esm2_mask"].device,
            )

        esm_tok = _get_esm_tokens(batch)
        esm_mask = batch["esm2_mask"]

        if self.esm_proj is not None:
            esm_tok = self.esm_proj(esm_tok)

        z_esm = self.prot_enc(esm_tok, esm_mask)
        if (
            self.use_pp_attn
            and self.use_pharm_branch
            and pharm_tokens is not None
            and self.pp_attn is not None
            and self.pp_to_prot is not None
            and self.pp_ln is not None
            and self.pp_gate is not None
        ):
            pp_tokens = self.pp_attn(pharm_tokens, esm_tok, kv_mask=esm_mask)
            pp_mask = batch["pharm_mask"]
            if self.pharm_pool_mode == "attn":
                pp_vec = self.pharm_pool_attn(pp_tokens, pp_mask)
            else:
                pp_vec = masked_mean_pool(pp_tokens, pp_mask, dim=1)
            pp_vec = self.pp_to_prot(pp_vec)
            z_esm = self.pp_ln(z_esm + torch.sigmoid(self.pp_gate) * pp_vec)
        if self.prot_proj is not None:
            z_esm = self.prot_proj(z_esm)
        return z_esm

    def _build_h_global(self, z_graph: torch.Tensor, z_pharm: torch.Tensor, z_esm: torch.Tensor) -> torch.Tensor:
        hg = self.graph_align(z_graph)
        hf = self.pharm_align(z_pharm)
        hp = self.prot_align(z_esm)
        return self.global_fuse([hg, hf, hp])

    # -----------------------------------------------------
    # pocket local branch helpers
    # -----------------------------------------------------
    def _get_pocket_batch_index(self, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, int]:
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
        B = int(batch["pharm_mask"].shape[0])
        dev = batch["pharm_mask"].device
        if int(num_graphs) <= 0:
            return torch.zeros((0,), dtype=torch.long, device=dev)
        if B <= 0:
            raise RuntimeError(
                f"Invalid batch size B={B} while num_graphs={num_graphs}. "
                "Cannot map pocket graphs to samples."
            )

        if "pocket_graph_batch" in batch:
            gb = batch["pocket_graph_batch"].long().view(-1)
            if int(gb.numel()) == int(num_graphs):
                if self.strict_pocket_graph_mapping:
                    bad = (gb < 0) | (gb >= B)
                    if bool(bad.any().item()):
                        bad_vals = gb[bad][:8].tolist()
                        raise RuntimeError(
                            "Invalid pocket_graph_batch indices. "
                            f"B={B}, num_graphs={num_graphs}, bad_examples={bad_vals}"
                        )
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

        if self.strict_pocket_graph_mapping:
            raise RuntimeError(
                "Cannot infer graph->sample mapping safely. "
                f"B={B}, num_graphs={num_graphs}. "
                "Please provide aligned `pocket_graph_batch` in collate output."
            )

        m = min(B, num_graphs)
        out = torch.arange(m, dtype=torch.long, device=dev)
        if m < num_graphs:
            pad = torch.full((num_graphs - m,), m - 1 if m > 0 else 0, dtype=torch.long, device=dev)
            out = torch.cat([out, pad], dim=0)
        return out

    def _build_local_views(
        self,
        batch: Dict[str, torch.Tensor],
        pharm_tokens: torch.Tensor,
        h_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = int(batch["pharm_mask"].shape[0])
        dev = h_global.device
        out_dtype = h_global.dtype

        zero_pair = torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)
        zero_set = torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)

        if not self.use_pharm_branch:
            return zero_pair, zero_set

        if (not self.use_pocket) or ("pocket_node_feat" not in batch) or (self.pocket_enc is None):
            return zero_pair, zero_set

        px = batch["pocket_node_feat"]
        pei = batch["pocket_edge_index"]
        pef = batch["pocket_edge_feat"]
        if px.numel() == 0:
            return zero_pair, zero_set

        if not self.use_sequence_branch:
            if (not self.pocket_use_struct_feat) or int(px.shape[1]) <= int(self.esm_dim):
                return zero_pair, zero_set
            px = px.clone()
            px[:, : int(self.esm_dim)] = 0.0

        if self.pocket_edge_proj is not None and pef.numel() > 0:
            pef = self.pocket_edge_proj(pef)

        graph_batch, num_graphs = self._get_pocket_batch_index(batch)
        graph_to_sample = self._graph_to_sample_index(batch, num_graphs)

        pocket_node_repr, pocket_graph_repr = self.pocket_enc(px, pei, pef, graph_batch, num_graphs=num_graphs)
        pocket_node_repr = self.pocket_node_align(pocket_node_repr)
        pocket_graph_repr = self.pocket_graph_align(pocket_graph_repr)

        sample_pair_reprs: List[List[torch.Tensor]] = [[] for _ in range(B)]
        sample_pair_scores: List[List[torch.Tensor]] = [[] for _ in range(B)]
        sample_set_tokens: List[List[torch.Tensor]] = [[] for _ in range(B)]
        sample_set_scores: List[List[torch.Tensor]] = [[] for _ in range(B)]

        for g in range(num_graphs):
            b = int(graph_to_sample[g].item())
            if b < 0 or b >= B:
                continue

            pharm_mask_b = batch["pharm_mask"][b].bool()
            pharm_b = pharm_tokens[b][pharm_mask_b]
            if pharm_b.numel() == 0:
                continue
            pharm_b = self.pharm_align(pharm_b).unsqueeze(0)

            node_mask = graph_batch == g
            pocket_nodes_g = pocket_node_repr[node_mask]
            if pocket_nodes_g.numel() == 0:
                continue

            pair_out = self.pair_cross(pharm_b, pocket_nodes_g.unsqueeze(0), return_details=True)
            pair_g = pair_out["pair_g"]
            graph_g = pocket_graph_repr[g:g + 1]

            if self.use_pair_coverage and self.coverage_encoder is not None:
                cov_g, cov_stats = self.coverage_encoder(
                    pair_score=pair_out["pair_score"],
                    q_tokens=pair_out["q_tokens"],
                    r_tokens=pair_out["r_tokens"],
                    graph_g=graph_g,
                )
            else:
                cov_g = torch.zeros_like(graph_g)
                cov_stats = {}

            pair_local_g = self.hpair_pocket_fuse(torch.cat([pair_g, cov_g, graph_g], dim=-1))

            score_in = torch.cat([
                pair_local_g,
                graph_g,
                h_global[b:b + 1],
                pair_local_g * h_global[b:b + 1],
            ], dim=-1)
            score_g = self.pocket_relevance(score_in)

            if self.use_pocket_score_prior and "pocket_graph_score" in batch:
                pgs = batch["pocket_graph_score"]
                if pgs.numel() >= g + 1:
                    prior = pgs.view(-1)[g:g + 1].view(1, 1).to(score_g.device, dtype=score_g.dtype)
                    score_g = score_g + self.pocket_score_scale * prior

            sample_pair_reprs[b].append(pair_local_g)
            sample_pair_scores[b].append(score_g)

            if self.use_set_view and self.top_pair_tokenizer is not None and int(self.set_top_pair_per_pocket) > 0:
                tok_g, tok_score_g = self.top_pair_tokenizer(
                    pair_score=pair_out["pair_score"],
                    pair_feat=pair_out["pair_feat"],
                    q_tokens=pair_out["q_tokens"],
                    r_tokens=pair_out["r_tokens"],
                    graph_g=graph_g,
                    top_t=self.set_top_pair_per_pocket,
                    max_per_q=self.set_max_per_pharm_token,
                    max_per_r=self.set_max_per_pocket_node,
                )
                if tok_g.numel() > 0:
                    pocket_gate = torch.sigmoid(score_g).view(1, 1)
                    tok_g = tok_g * pocket_gate
                    sample_set_tokens[b].append(tok_g)
                    # Use pocket score only once (token scaling above), avoid double counting.
                    sample_set_scores[b].append(tok_score_g)

        h_pair_list = []
        for b in range(B):
            if len(sample_pair_reprs[b]) == 0:
                h_pair_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                continue
            reps = torch.cat(sample_pair_reprs[b], dim=0)
            scrs = torch.cat(sample_pair_scores[b], dim=0).view(-1)
            scrs = torch.nan_to_num(scrs, nan=0.0, posinf=20.0, neginf=-20.0)
            if self.pocket_inst_pool == "mean":
                h_pair_b = reps.mean(dim=0, keepdim=True)
            elif self.pocket_inst_pool == "max":
                max_idx = int(torch.argmax(scrs).item())
                h_pair_b = reps[max_idx:max_idx + 1]
            else:
                alpha = masked_softmax(scrs / float(self.pocket_tau), dim=0).unsqueeze(-1).to(dtype=reps.dtype)
                h_pair_b = (reps * alpha).sum(dim=0, keepdim=True)
            h_pair_list.append(h_pair_b)
        h_pair = torch.cat(h_pair_list, dim=0).to(dtype=out_dtype)

        if (not self.use_set_view) or (self.set_encoder is None):
            return h_pair, zero_set

        h_set_list = []
        for b in range(B):
            if len(sample_set_tokens[b]) == 0:
                h_set_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                continue

            toks = torch.cat(sample_set_tokens[b], dim=0)
            tok_scores = torch.cat(sample_set_scores[b], dim=0)

            if self.set_max_tokens > 0 and toks.size(0) > self.set_max_tokens:
                keep = min(int(self.set_max_tokens), int(toks.size(0)))
                _, idx = torch.topk(tok_scores, k=keep, largest=True)
                toks = toks.index_select(0, idx)

            toks = toks.unsqueeze(0)
            mask = torch.ones((1, toks.size(1)), dtype=torch.bool, device=toks.device)
            h_set_b = self.set_encoder(toks, mask)
            h_set_list.append(h_set_b)

        h_set = torch.cat(h_set_list, dim=0).to(dtype=out_dtype)
        return h_pair, h_set

    # -----------------------------------------------------
    # forward
    # -----------------------------------------------------
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        drug_node_repr, z_graph = self._run_drug(batch)
        pharm_tokens, z_pharm = self._run_pharm(batch, drug_node_repr)
        z_esm = self._run_protein(batch, pharm_tokens=pharm_tokens)

        h_global = self._build_h_global(z_graph, z_pharm, z_esm)
        h_pair, h_set = self._build_local_views(batch, pharm_tokens, h_global)

        h_set_eff = h_set if self.use_set_view else torch.zeros_like(h_pair)
        if self.fusion_strategy == "concat":
            fused = torch.cat([h_global, h_pair, h_set_eff], dim=-1)
            logit = self.concat_classifier(fused).squeeze(-1)
            return logit

        if self.fusion_strategy == "single_stage":
            fused = self.single_stage_fuse([h_global, h_pair, h_set_eff])
        elif self.fusion_strategy == "mean_sum":
            fused = (h_global + h_pair + h_set_eff) / 3.0
        else:
            h_local = self.set_residual(h_pair, h_set_eff) if self.use_set_view else h_pair
            fused = self.pair_residual(h_global, h_local)
        logit = self.classifier(fused).squeeze(-1)
        return logit


class BaselineGatedDTIRealPocket3View(BaselineGatedDTIRealPocket):
    pass


class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
