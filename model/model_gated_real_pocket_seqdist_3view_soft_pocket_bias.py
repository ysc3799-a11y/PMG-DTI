# -*- coding: utf-8 -*-
"""
3-view DTI model with soft pocket bias.

Idea:
- Keep the existing strong 3-view backbone (global + h_pair + h_set).
- Inject pocket information earlier by biasing pharm->protein attention logits.
- Pocket bias is residue-level and soft: each protein token gets a score
  according to its similarity with pocket node embeddings.

This file is intentionally lightweight: it subclasses the current
`model_gated_real_pocket_seqdist_3view_hset.BaselineGatedDTIRealPocket`
so you can drop it into the current codebase with minimal changes.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import masked_softmax
from model.model_gated_real_pocket_seqdist_3view_hset import (
    BaselineGatedDTIRealPocket as _BaseModel,
    _get_esm_tokens,
    masked_mean_pool,
)


class PocketBiasedScaledDotAttn(nn.Module):
    """
    Drop-in replacement for ScaledDotAttn with an additive residue bias.

    scores_{i,j} = q_i^T k_j / sqrt(d) + scale * bias_j
    where bias_j is a pocket-derived scalar for protein residue j.
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
        self.gate = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))
        # Positive scale for additive pocket bias.
        self.bias_log_scale = nn.Parameter(torch.tensor(-0.2, dtype=torch.float32))

    @classmethod
    def from_existing(cls, attn: nn.Module) -> "PocketBiasedScaledDotAttn":
        q_dim = int(attn.q_proj.in_features)
        kv_dim = int(attn.k_proj.in_features)
        out_dim = int(attn.q_proj.out_features)
        try:
            p = next(attn.parameters())
            drop_p = float(getattr(attn.drop, "p", 0.1))
            device = p.device
            dtype = p.dtype
        except StopIteration:
            drop_p = 0.1
            device = None
            dtype = None
        new_attn = cls(q_dim=q_dim, kv_dim=kv_dim, out_dim=out_dim, dropout=drop_p)
        new_attn.load_state_dict(attn.state_dict(), strict=False)
        if device is not None:
            new_attn = new_attn.to(device=device, dtype=dtype)
        return new_attn

    def forward(
        self,
        q: torch.Tensor,
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor] = None,
        additive_bias: Optional[torch.Tensor] = None,
        return_attn: bool = False,
    ):
        """
        q: [B, Q, Dq]
        kv: [B, K, Dkv]
        kv_mask: [B, K] (0/1) where 0 = padded / ignore
        additive_bias: [B, K] or [B, 1, K]
        """
        Q = self.q_proj(q)
        K = self.k_proj(kv)
        V = self.v_proj(kv)

        scores = torch.matmul(Q, K.transpose(1, 2)) / math.sqrt(Q.size(-1))  # [B,Q,K]
        if additive_bias is not None:
            if additive_bias.dim() == 2:
                bias = additive_bias.unsqueeze(1)
            else:
                bias = additive_bias
            scores = scores + F.softplus(self.bias_log_scale) * bias.to(dtype=scores.dtype, device=scores.device)

        attn_prob = masked_softmax(scores, mask=kv_mask, dim=-1)
        attn_prob = self.drop(attn_prob)
        ctx = torch.matmul(attn_prob, V)
        ctx = self.out_proj(ctx)
        out = self.ln(Q + self.gate * ctx)
        if return_attn:
            return out, attn_prob
        return out


class BaselineGatedDTIRealPocket(_BaseModel):
    """
    Existing 3-view model + soft pocket bias injected into pharm->protein attention.
    """

    def __init__(self, *args, **kwargs):
        self.use_soft_pocket_bias = bool(int(kwargs.pop("use_soft_pocket_bias", kwargs.pop("USE_SOFT_POCKET_BIAS", 1))))
        self.pocket_bias_margin = float(kwargs.pop("pocket_bias_margin", kwargs.pop("POCKET_BIAS_MARGIN", 0.45)))
        self.pocket_bias_pool = str(kwargs.pop("pocket_bias_pool", kwargs.pop("POCKET_BIAS_POOL", "max"))).lower()
        self.pocket_bias_topk = int(max(1, kwargs.pop("pocket_bias_topk", kwargs.pop("POCKET_BIAS_TOPK", 4))))
        self.pocket_bias_use_prior = bool(int(kwargs.pop("pocket_bias_use_prior", kwargs.pop("POCKET_BIAS_USE_PRIOR", 1))))
        self.pocket_bias_prior_power = float(kwargs.pop("pocket_bias_prior_power", kwargs.pop("POCKET_BIAS_PRIOR_POWER", 1.0)))
        self.pocket_bias_reduce = str(kwargs.pop("pocket_bias_reduce", kwargs.pop("POCKET_BIAS_REDUCE", "max"))).lower()
        self.pocket_bias_l2norm = bool(int(kwargs.pop("pocket_bias_l2norm", kwargs.pop("POCKET_BIAS_L2NORM", 1))))
        _pp_gate_init = kwargs.pop("pp_gate_init", kwargs.pop("PP_GATE_INIT", None))
        _pp_attn_gate_init = kwargs.pop("pp_attn_gate_init", kwargs.pop("PP_ATTN_GATE_INIT", None))
        self.pp_gate_init = None if _pp_gate_init is None else float(_pp_gate_init)
        self.pp_attn_gate_init = None if _pp_attn_gate_init is None else float(_pp_attn_gate_init)
        super().__init__(*args, **kwargs)

        # Replace existing pp_attn with bias-aware attention while keeping weights.
        if self.use_pp_attn and self.pp_attn is not None and self.use_soft_pocket_bias:
            self.pp_attn = PocketBiasedScaledDotAttn.from_existing(self.pp_attn)
        # Optional experiment controls: force open gates at init.
        with torch.no_grad():
            if self.pp_gate is not None and self.pp_gate_init is not None:
                self.pp_gate.fill_(self.pp_gate_init)
            if self.pp_attn is not None and self.pp_attn_gate_init is not None and hasattr(self.pp_attn, "gate"):
                gate = getattr(self.pp_attn, "gate")
                if isinstance(gate, torch.Tensor):
                    gate.fill_(self.pp_attn_gate_init)
        # Updated every forward; consumed by training script for debug prints.
        self._debug_scalars: Dict[str, float] = {}

    # -----------------------------------------------------
    # soft pocket bias helpers
    # -----------------------------------------------------
    def _extract_pocket_node_esm(self, pocket_node_feat: torch.Tensor) -> torch.Tensor:
        """
        The first self.esm_dim channels always correspond to residue ESM features.
        Works for both nostruct and struct versions.
        """
        if pocket_node_feat.numel() == 0:
            return pocket_node_feat.new_zeros((0, self.esm_dim))
        if int(pocket_node_feat.shape[1]) >= int(self.esm_dim):
            return pocket_node_feat[:, : int(self.esm_dim)]
        # Defensive fallback (should not happen in normal pipeline)
        pad = int(self.esm_dim) - int(pocket_node_feat.shape[1])
        return torch.cat(
            [pocket_node_feat, pocket_node_feat.new_zeros((int(pocket_node_feat.shape[0]), pad))],
            dim=1,
        )

    def _normalize_token_bank(self, x: torch.Tensor) -> torch.Tensor:
        if not self.pocket_bias_l2norm:
            return x
        return F.normalize(x, dim=-1)

    def _score_to_weight(self, score: torch.Tensor) -> torch.Tensor:
        s = score
        if torch.isfinite(s).all() and float(s.min().item()) >= 0.0 and float(s.max().item()) <= 1.0:
            w = s.clamp(0.0, 1.0)
        else:
            w = torch.sigmoid(s)
        if self.pocket_bias_prior_power != 1.0:
            w = torch.pow(w.clamp_min(1e-6), self.pocket_bias_prior_power)
        return w

    def _build_soft_pocket_bias(
        self,
        batch: Dict[str, torch.Tensor],
        prot_tok: torch.Tensor,
        prot_mask: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Build a soft residue bias over protein tokens using pocket node embeddings.

        Returns:
            bias: [B, L] in [0, 1] approximately, or None if unavailable.
        """
        if not self.use_sequence_branch:
            return None
        if (not self.use_soft_pocket_bias) or (not self.use_pocket):
            return None
        if "pocket_node_feat" not in batch or batch["pocket_node_feat"].numel() == 0:
            return None
        if "pocket_graph_slices" not in batch or len(batch["pocket_graph_slices"]) == 0:
            return None

        B, L, D = prot_tok.shape
        dev = prot_tok.device
        dtype = prot_tok.dtype

        prot_mask_bool = prot_mask > 0
        prot_bank = self._normalize_token_bank(prot_tok)

        pocket_node_feat = batch["pocket_node_feat"]
        pocket_node_esm = self._extract_pocket_node_esm(pocket_node_feat).to(device=dev, dtype=dtype)
        if self.esm_proj is not None:
            pocket_node_esm = self.esm_proj(pocket_node_esm)
        pocket_bank = self._normalize_token_bank(pocket_node_esm)

        graph_slices = batch["pocket_graph_slices"]
        graph_to_sample = batch.get("pocket_graph_batch", None)
        if graph_to_sample is None or int(graph_to_sample.numel()) != int(len(graph_slices)):
            graph_to_sample = self._graph_to_sample_index(batch, len(graph_slices))
        graph_to_sample = graph_to_sample.to(device=dev)

        graph_score = batch.get("pocket_graph_score", None)
        bias = torch.zeros((B, L), dtype=dtype, device=dev)
        graph_count = torch.zeros((B,), dtype=dtype, device=dev)

        for g, (s, e) in enumerate(graph_slices):
            s = int(s)
            e = int(e)
            if e <= s:
                continue
            b = int(graph_to_sample[g].item())
            if b < 0 or b >= B:
                continue

            pocket_g = pocket_bank[s:e]  # [Ng, D]
            prot_b = prot_bank[b]        # [L, D]
            sim = torch.matmul(pocket_g, prot_b.transpose(0, 1))  # [Ng, L]
            if self.pocket_bias_pool == "mean":
                residue_score = sim.mean(dim=0)
            elif self.pocket_bias_pool == "topkmean":
                k = min(int(self.pocket_bias_topk), int(sim.shape[0]))
                residue_score = sim.topk(k=k, dim=0).values.mean(dim=0)
            else:
                residue_score = sim.max(dim=0).values

            # sharpen: only keep high-confidence matches to residue positions
            margin = float(self.pocket_bias_margin)
            residue_score = (residue_score - margin).clamp_min(0.0) / max(1e-6, 1.0 - margin)
            residue_score = residue_score * prot_mask_bool[b].float()

            if self.pocket_bias_use_prior and graph_score is not None and graph_score.numel() >= g + 1:
                w = self._score_to_weight(graph_score.view(-1)[g:g + 1].to(device=dev, dtype=dtype)).view(1)
                residue_score = residue_score * w

            if self.pocket_bias_reduce == "sum":
                bias[b] = bias[b] + residue_score
            elif self.pocket_bias_reduce == "mean":
                bias[b] = bias[b] + residue_score
                graph_count[b] = graph_count[b] + 1.0
            else:
                bias[b] = torch.maximum(bias[b], residue_score)

        # per-sample normalize to [0,1] on valid residues only
        for b in range(B):
            valid = prot_mask_bool[b]
            if int(valid.sum().item()) == 0:
                continue
            if self.pocket_bias_reduce == "mean" and float(graph_count[b].item()) > 0.0:
                bias[b, valid] = bias[b, valid] / graph_count[b].clamp_min(1e-6)
            x = bias[b, valid]
            xmax = x.max()
            if torch.isfinite(xmax) and float(xmax.item()) > 0.0:
                bias[b, valid] = x / xmax.clamp_min(1e-6)

        return bias

    # -----------------------------------------------------
    # override protein branch to inject soft pocket bias
    # -----------------------------------------------------
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
            pocket_bias = self._build_soft_pocket_bias(batch, esm_tok, esm_mask)
            if isinstance(self.pp_attn, PocketBiasedScaledDotAttn):
                pp_tokens = self.pp_attn(pharm_tokens, esm_tok, kv_mask=esm_mask, additive_bias=pocket_bias)
            else:
                pp_tokens = self.pp_attn(pharm_tokens, esm_tok, kv_mask=esm_mask)

            pp_mask = batch["pharm_mask"]
            if self.pharm_pool_mode == "attn":
                pp_vec = self.pharm_pool_attn(pp_tokens, pp_mask)
            else:
                pp_vec = masked_mean_pool(pp_tokens, pp_mask, dim=1)
            pp_vec = self.pp_to_prot(pp_vec)
            z_esm = self.pp_ln(z_esm + torch.sigmoid(self.pp_gate) * pp_vec)
        if getattr(self, "prot_proj", None) is not None:
            z_esm = self.prot_proj(z_esm)

        # Debug scalars for monitoring soft pocket bias behavior during training.
        with torch.no_grad():
            dbg: Dict[str, float] = {}
            pocket_bias = None
            if self.use_pp_attn and self.use_pharm_branch and self.use_sequence_branch and pharm_tokens is not None:
                pocket_bias = self._build_soft_pocket_bias(batch, esm_tok, esm_mask)
            if pocket_bias is not None and pocket_bias.numel() > 0:
                valid = (esm_mask > 0)
                x = pocket_bias[valid] if int(valid.sum().item()) > 0 else pocket_bias.view(-1)
                if x.numel() > 0:
                    dbg["pocket_bias.mean"] = float(x.mean().item())
                    dbg["pocket_bias.max"] = float(x.max().item())
                    dbg["pocket_bias.nonzero_ratio"] = float((x > 0).float().mean().item())
                else:
                    dbg["pocket_bias.mean"] = 0.0
                    dbg["pocket_bias.max"] = 0.0
                    dbg["pocket_bias.nonzero_ratio"] = 0.0
            else:
                dbg["pocket_bias.mean"] = 0.0
                dbg["pocket_bias.max"] = 0.0
                dbg["pocket_bias.nonzero_ratio"] = 0.0

            if isinstance(self.pp_attn, PocketBiasedScaledDotAttn):
                dbg["softplus(bias_log_scale)"] = float(F.softplus(self.pp_attn.bias_log_scale.detach()).item())
                dbg["pp_attn.gate"] = float(self.pp_attn.gate.detach().item())
            else:
                dbg["softplus(bias_log_scale)"] = float("nan")
                dbg["pp_attn.gate"] = float("nan")

            if self.pp_gate is not None:
                dbg["sigmoid(pp_gate)"] = float(torch.sigmoid(self.pp_gate.detach()).item())
            else:
                dbg["sigmoid(pp_gate)"] = float("nan")

            self._debug_scalars = dbg
        return z_esm


class BaselineGatedDTIRealPocket3View(BaselineGatedDTIRealPocket):
    pass


class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
