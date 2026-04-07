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
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

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

        if kv_mask is not None:
            neg = torch.finfo(scores.dtype).min
            scores = scores.masked_fill(kv_mask.unsqueeze(1) == 0, neg)

        attn_prob = torch.softmax(scores, dim=-1)
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
        self.aux_mlp_hidden = int(kwargs.pop("aux_mlp_hidden", kwargs.pop("AUX_MLP_HIDDEN", 0)) or 0)
        self.aux_dropout = float(kwargs.pop("aux_dropout", kwargs.pop("AUX_DROPOUT", kwargs.get("dropout", kwargs.get("DROPOUT", 0.1)))))
        self.mil_pool = str(kwargs.pop("mil_pool", kwargs.pop("MIL_POOL", "logsumexp"))).lower()
        self.mil_tau = float(kwargs.pop("mil_tau", kwargs.pop("MIL_TAU", 1.0)))
        self.mil_empty_logit = float(kwargs.pop("mil_empty_logit", kwargs.pop("MIL_EMPTY_LOGIT", -6.0)))
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
        aux_hidden = int(self.aux_mlp_hidden or max(64, self.view_dim))
        self.global_aux_head = self._make_aux_head(aux_hidden, self.aux_dropout)
        self.pair_aux_head = self._make_aux_head(aux_hidden, self.aux_dropout)
        self.set_aux_head = self._make_aux_head(aux_hidden, self.aux_dropout)

    def _make_aux_head(self, hidden: int, dropout: float) -> nn.Module:
        return nn.Sequential(
            nn.Linear(self.view_dim, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden), 1),
        )

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

    def _aggregate_bag_logit(self, instance_logits: torch.Tensor) -> torch.Tensor:
        if instance_logits.numel() == 0:
            return instance_logits.new_full((1,), float(self.mil_empty_logit))

        mode = str(self.mil_pool).lower()
        if mode in {"max", "hardmax"}:
            return instance_logits.max().view(1)
        if mode in {"mean", "avg"}:
            return instance_logits.mean().view(1)
        if mode in {"noisy_or", "noisyor"}:
            p = torch.sigmoid(instance_logits).clamp(1e-6, 1.0 - 1e-6)
            bag_p = 1.0 - torch.prod(1.0 - p)
            bag_p = bag_p.clamp(1e-6, 1.0 - 1e-6)
            return torch.log(bag_p / (1.0 - bag_p)).view(1)

        # Default: normalized logsumexp = smooth max without a raw count bias.
        tau = float(max(1e-6, self.mil_tau))
        norm = float(math.log(max(1, int(instance_logits.numel()))))
        out = tau * (torch.logsumexp(instance_logits / tau, dim=0) - norm)
        return out.view(1)

    def _build_local_views_with_bag(
        self,
        batch: Dict[str, torch.Tensor],
        pharm_tokens: torch.Tensor,
        h_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = int(batch["pharm_mask"].shape[0])
        dev = h_global.device
        out_dtype = h_global.dtype

        zero_pair = torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)
        zero_set = torch.zeros((B, self.view_dim), dtype=out_dtype, device=dev)
        zero_bag = torch.full((B,), float(self.mil_empty_logit), dtype=out_dtype, device=dev)

        if not self.use_pharm_branch:
            return zero_pair, zero_set, zero_bag

        if (not self.use_pocket) or ("pocket_node_feat" not in batch) or (self.pocket_enc is None):
            return zero_pair, zero_set, zero_bag

        px = batch["pocket_node_feat"]
        pei = batch["pocket_edge_index"]
        pef = batch["pocket_edge_feat"]
        if px.numel() == 0:
            return zero_pair, zero_set, zero_bag

        if not self.use_sequence_branch:
            if (not self.pocket_use_struct_feat) or int(px.shape[1]) <= int(self.esm_dim):
                return zero_pair, zero_set, zero_bag
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
                cov_g, _ = self.coverage_encoder(
                    pair_score=pair_out["pair_score"],
                    q_tokens=pair_out["q_tokens"],
                    r_tokens=pair_out["r_tokens"],
                    graph_g=graph_g,
                )
            else:
                cov_g = torch.zeros_like(graph_g)

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
            sample_pair_scores[b].append(score_g.view(1))

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
                    sample_set_scores[b].append(tok_score_g)

        h_pair_list: List[torch.Tensor] = []
        h_set_list: List[torch.Tensor] = []
        bag_logit_list: List[torch.Tensor] = []
        inst_count = []
        inst_top_mean = []
        for b in range(B):
            if len(sample_pair_reprs[b]) == 0:
                h_pair_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                h_set_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                bag_logit_list.append(zero_bag[b:b + 1])
                inst_count.append(0.0)
                inst_top_mean.append(float(self.mil_empty_logit))
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
                alpha = torch.softmax((scrs.float() / float(self.pocket_tau)), dim=0).unsqueeze(-1).to(dtype=reps.dtype)
                h_pair_b = (reps * alpha).sum(dim=0, keepdim=True)
            h_pair_list.append(h_pair_b)
            bag_logit_list.append(self._aggregate_bag_logit(scrs).to(dtype=out_dtype))
            inst_count.append(float(scrs.numel()))
            inst_top_mean.append(float(scrs.max().item()))

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

        h_pair = torch.cat(h_pair_list, dim=0).to(dtype=out_dtype)
        h_set = torch.cat(h_set_list, dim=0).to(dtype=out_dtype)
        bag_logit = torch.cat(bag_logit_list, dim=0).to(dtype=out_dtype)

        with torch.no_grad():
            dbg = dict(self._debug_scalars)
            if inst_count:
                dbg["mil.instance_count_mean"] = float(sum(inst_count) / max(1, len(inst_count)))
                dbg["mil.instance_logit_top_mean"] = float(sum(inst_top_mean) / max(1, len(inst_top_mean)))
                dbg["mil.bag_logit_mean"] = float(bag_logit.mean().item())
            self._debug_scalars = dbg
        return h_pair, h_set, bag_logit

    def forward(self, batch: Dict[str, torch.Tensor]):
        drug_node_repr, z_graph = self._run_drug(batch)
        pharm_tokens, z_pharm = self._run_pharm(batch, drug_node_repr)
        z_esm = self._run_protein(batch, pharm_tokens=pharm_tokens)

        h_global = self._build_h_global(z_graph, z_pharm, z_esm)
        h_pair, h_set, bag_logit = self._build_local_views_with_bag(batch, pharm_tokens, h_global)

        h_set_eff = h_set if self.use_set_view else torch.zeros_like(h_pair)
        if self.fusion_strategy == "concat":
            fused = torch.cat([h_global, h_pair, h_set_eff], dim=-1)
            logit = self.concat_classifier(fused).squeeze(-1)
        else:
            if self.fusion_strategy == "single_stage":
                fused = self.single_stage_fuse([h_global, h_pair, h_set_eff])
            elif self.fusion_strategy == "mean_sum":
                fused = (h_global + h_pair + h_set_eff) / 3.0
            else:
                h_local = self.set_residual(h_pair, h_set_eff) if self.use_set_view else h_pair
                fused = self.pair_residual(h_global, h_local)
            logit = self.classifier(fused).squeeze(-1)

        out = {
            "logit": logit,
            "global_logit": self.global_aux_head(h_global).squeeze(-1),
            "pair_logit": self.pair_aux_head(h_pair).squeeze(-1),
            "set_logit": self.set_aux_head(h_set).squeeze(-1) if self.use_set_view else None,
            "bag_logit": bag_logit,
        }
        return out


class BaselineGatedDTIRealPocket3View(BaselineGatedDTIRealPocket):
    pass


class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
