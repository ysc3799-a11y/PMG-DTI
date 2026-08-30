# -*- coding: utf-8 -*-
"""
3-view DTI model with pharmacophore evidence-guided pocket matching.

This variant keeps the existing soft-pocket-bias backbone and replaces the
local pharm-pocket matcher with an evidence-guided matcher:
  S_ij  : learned pharm-token / pocket-residue compatibility
  M_ij  : bidirectional consistency matching weight
  rho_j : pharmacophore-guided residue relevance

The model also exposes a weakly supervised local evidence logit through
``self._aux_outputs["local_logit"]`` for the training script.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model import masked_softmax
from model.model_gated_real_pocket_seqdist_3view_hset import TopPairTokenizer
from model.model_gated_real_pocket_seqdist_3view_soft_pocket_bias import (
    BaselineGatedDTIRealPocket as _SoftPocketBiasModel,
)


class EvidencePairScoreLocal(nn.Module):
    """
    Pair-level matcher with prediction-contribution calibrated attention.

    The attention score combines content compatibility, pair-level evidence,
    and optional pocket-prior bias. A global-context contribution gate then
    calibrates the local attention before pooling and downstream alignment.
    """

    def __init__(
        self,
        dim: int,
        dropout: float = 0.1,
        tau: float = 0.7,
        evidence_pool: str = "logsumexp",
        evidence_score_scale: float = 0.5,
        pocket_bias_scale: float = 1.0,
        use_contribution_gate: bool = True,
    ):
        super().__init__()
        self.dim = int(dim)
        self.q_proj = nn.Linear(dim, dim)
        self.r_proj = nn.Linear(dim, dim)
        self.score_mlp = nn.Sequential(
            nn.Linear(dim * 4, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.evidence_mlp = nn.Sequential(
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
        self.contrib_gate = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.out = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)
        self.tau = float(max(1e-4, tau))
        self.evidence_pool = str(evidence_pool or "logsumexp").lower()
        self.evidence_score_scale = float(evidence_score_scale)
        self.pocket_bias_scale = float(pocket_bias_scale)
        self.use_contribution_gate = bool(use_contribution_gate)

    def _empty_details(self, ref: torch.Tensor, dim: int) -> Dict[str, torch.Tensor]:
        zero = torch.zeros((1, dim), dtype=ref.dtype, device=ref.device)
        z_score = torch.zeros((0, 0), dtype=ref.dtype, device=ref.device)
        z_pair = torch.zeros((0, 0, dim), dtype=ref.dtype, device=ref.device)
        z_tok = torch.zeros((0, dim), dtype=ref.dtype, device=ref.device)
        return {
            "pair_g": zero,
            "pair_score": z_score,
            "pair_feat": z_pair,
            "pair_match": z_score,
            "pair_conf": z_score,
            "pair_attention": z_score,
            "pair_conf_raw": z_score,
            "content_score": z_score,
            "evidence_score": z_score,
            "contribution_gate": z_score,
            "q_tokens": z_tok,
            "r_tokens": z_tok,
            "pharm_relevance": torch.zeros((0,), dtype=ref.dtype, device=ref.device),
            "residue_relevance": torch.zeros((0,), dtype=ref.dtype, device=ref.device),
            "local_logit": torch.zeros((1,), dtype=ref.dtype, device=ref.device),
        }

    def _normalise_pair_weight(self, pair_weight: torch.Tensor) -> torch.Tensor:
        pair_weight = torch.nan_to_num(pair_weight.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        total = pair_weight.sum()
        if (not torch.isfinite(total)) or float(total.detach().item()) <= 0.0:
            pair_weight = torch.ones_like(pair_weight, dtype=torch.float32)
            total = pair_weight.sum()
        return pair_weight / total.clamp_min(1e-8)

    def pool_from_weight(
        self,
        pair_feat: torch.Tensor,
        q_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        pair_weight: torch.Tensor,
    ) -> torch.Tensor:
        if pair_feat.numel() == 0 or q_tokens.numel() == 0 or r_tokens.numel() == 0:
            return torch.zeros((1, self.dim), dtype=pair_feat.dtype, device=pair_feat.device)

        out_dtype = pair_feat.dtype
        with torch.autocast(device_type=pair_feat.device.type, enabled=False):
            feat = pair_feat.float()
            q = q_tokens.float()
            r = r_tokens.float()
            weight = self._normalise_pair_weight(pair_weight.to(device=feat.device))

            pooled_pair = (feat * weight.unsqueeze(-1)).sum(dim=(0, 1)).view(1, self.dim)
            pharm_rel = weight.sum(dim=1)
            residue_rel = weight.sum(dim=0)
            pharm_w = pharm_rel / pharm_rel.sum().clamp_min(1e-8)
            residue_w = residue_rel / residue_rel.sum().clamp_min(1e-8)
            q_ctx = (q * pharm_w.unsqueeze(-1)).sum(dim=0, keepdim=True)
            r_ctx = (r * residue_w.unsqueeze(-1)).sum(dim=0, keepdim=True)
            pair_g = self.norm(self.out(torch.cat([pooled_pair, q_ctx, r_ctx], dim=-1)))
        return pair_g.to(dtype=out_dtype)

    def forward(
        self,
        q_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        h_global: Optional[torch.Tensor] = None,
        pocket_bias: Optional[torch.Tensor] = None,
        return_details: bool = False,
    ):
        dim = int(q_tokens.size(-1))
        if q_tokens.size(1) == 0 or r_tokens.size(1) == 0:
            details = self._empty_details(q_tokens, dim)
            return details if return_details else details["pair_g"]

        out_dtype = q_tokens.dtype
        with torch.autocast(device_type=q_tokens.device.type, enabled=False):
            q = q_tokens.squeeze(0).float()
            r = r_tokens.squeeze(0).float()
            qh = self.q_proj(q)
            rh = self.r_proj(r)
            m, d = qh.shape
            n = int(rh.shape[0])

            qe = qh.unsqueeze(1).expand(m, n, d)
            re = rh.unsqueeze(0).expand(m, n, d)
            pair_in = torch.cat([qe, re, qe * re, torch.abs(qe - re)], dim=-1)
            content_score = self.score_mlp(pair_in).squeeze(-1)
            evidence_score = self.evidence_mlp(pair_in).squeeze(-1)
            pair_feat = self.pair_feat_mlp(pair_in)

            content_score_safe = torch.nan_to_num(content_score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            evidence_score_safe = torch.nan_to_num(evidence_score, nan=0.0, posinf=20.0, neginf=-20.0).clamp(-20.0, 20.0)
            attn_score = content_score_safe + self.evidence_score_scale * evidence_score_safe
            if pocket_bias is not None and pocket_bias.numel() == n:
                pb = torch.nan_to_num(
                    pocket_bias.to(device=attn_score.device, dtype=torch.float32).view(1, n),
                    nan=0.0,
                    posinf=1.0,
                    neginf=0.0,
                )
                attn_score = attn_score + self.pocket_bias_scale * pb.expand(m, n)

            pair_attention = F.softmax(attn_score / float(self.tau), dim=1)
            pair_attention = torch.nan_to_num(pair_attention, nan=0.0, posinf=0.0, neginf=0.0)
            pair_conf_raw = pair_attention * torch.sigmoid(evidence_score_safe)

            if self.use_contribution_gate and h_global is not None:
                hg = h_global.float().view(1, 1, d).expand(m, n, d)
                contrib_in = torch.cat([pair_feat, hg, pair_feat * hg], dim=-1)
                contribution_gate = torch.sigmoid(self.contrib_gate(contrib_in).squeeze(-1))
            else:
                contribution_gate = torch.ones_like(pair_conf_raw)
            contribution_gate = torch.nan_to_num(contribution_gate, nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)

            pair_conf = pair_conf_raw * contribution_gate
            pair_weight = self._normalise_pair_weight(pair_conf)
            pair_match = pair_weight
            pair_g = self.pool_from_weight(pair_feat, qh, rh, pair_weight)
            pharm_rel = pair_weight.sum(dim=1)
            residue_rel = pair_weight.sum(dim=0)

            if self.evidence_pool == "mean":
                local_logit = (attn_score * pair_weight).sum().view(1)
            elif self.evidence_pool == "max":
                local_logit = attn_score.reshape(-1).max().view(1)
            else:
                local_logit = torch.logsumexp(torch.log(pair_weight.clamp_min(1e-8)) + attn_score, dim=(0, 1)).view(1)

        if return_details:
            return {
                "pair_g": pair_g.to(dtype=out_dtype),
                "pair_score": attn_score.to(dtype=out_dtype),
                "pair_feat": pair_feat.to(dtype=out_dtype),
                "pair_match": pair_match.to(dtype=out_dtype),
                "pair_conf": pair_conf.to(dtype=out_dtype),
                "pair_attention": pair_attention.to(dtype=out_dtype),
                "pair_conf_raw": pair_conf_raw.to(dtype=out_dtype),
                "content_score": content_score_safe.to(dtype=out_dtype),
                "evidence_score": evidence_score_safe.to(dtype=out_dtype),
                "contribution_gate": contribution_gate.to(dtype=out_dtype),
                "q_tokens": qh.to(dtype=out_dtype),
                "r_tokens": rh.to(dtype=out_dtype),
                "pharm_relevance": pharm_rel.to(dtype=out_dtype),
                "residue_relevance": residue_rel.to(dtype=out_dtype),
                "local_logit": local_logit,
            }
        return pair_g.to(dtype=out_dtype)


class EvidenceTopPairTokenizer(TopPairTokenizer):
    """
    Top-pair tokenizer that ranks pairs by bidirectional evidence confidence.
    """

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
        pair_match: torch.Tensor = None,
        pair_conf: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if pair_score.numel() == 0 or int(top_t) <= 0:
            d = int(graph_g.shape[-1])
            return (
                torch.zeros((0, d), dtype=graph_g.dtype, device=graph_g.device),
                torch.zeros((0,), dtype=graph_g.dtype, device=graph_g.device),
            )

        m, n = pair_score.shape
        if pair_conf is None:
            if pair_match is None:
                pair_conf = torch.sigmoid(pair_score)
            else:
                pair_conf = pair_match.to(dtype=pair_score.dtype, device=pair_score.device) * torch.sigmoid(pair_score)

        flat_rank = torch.nan_to_num(pair_conf.reshape(-1), nan=0.0, posinf=1.0, neginf=0.0)
        order = torch.argsort(flat_rank, descending=True)
        selected: List[int] = []
        q_counts: Dict[int, int] = {}
        r_counts: Dict[int, int] = {}
        per_q = int(max_per_q)
        per_r = int(max_per_r)
        target = int(max(top_t, 0))
        for oid in order.tolist():
            if len(selected) >= target:
                break
            i = int(oid // n)
            j = int(oid % n)
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
        top_score = flat_rank.index_select(0, topi).clamp(0.0, 1.0)
        tokens = []
        graph_ctx = graph_g.squeeze(0)
        for k in range(int(topi.numel())):
            idx = int(topi[k].item())
            i = idx // n
            j = idx % n
            score_emb = self.score_proj(top_score[k].view(1, 1)).view(-1)
            tok = self.token_mlp(torch.cat([
                pair_feat[i, j],
                q_tokens[i],
                r_tokens[j],
                graph_ctx,
                score_emb,
            ], dim=-1))
            tokens.append(tok)

        tokens = self.norm(torch.stack(tokens, dim=0))
        return tokens, top_score


class GlobalInteractionAligner(nn.Module):
    """
    Learn a global-context pair interaction map and align it to local evidence.

    S_ij from EvidencePairScoreLocal is a local pharm-pocket compatibility map.
    This module builds a separate G_ij conditioned on h_global, then uses the
    detached local evidence M_ij as a soft target for the global interaction.
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.dim = int(dim)
        self.q_ctx = nn.Linear(self.dim, self.dim)
        self.r_ctx = nn.Linear(self.dim, self.dim)
        self.q_norm = nn.LayerNorm(self.dim)
        self.r_norm = nn.LayerNorm(self.dim)
        self.scorer = nn.Sequential(
            nn.Linear(self.dim * 4, self.dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim, 1),
        )

    def forward(
        self,
        q_tokens: torch.Tensor,
        r_tokens: torch.Tensor,
        h_global: torch.Tensor,
        pair_match: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if q_tokens.numel() == 0 or r_tokens.numel() == 0 or pair_match.numel() == 0:
            zero = h_global.new_zeros(())
            score = h_global.new_zeros((0, 0))
            return score, zero

        with torch.autocast(device_type=h_global.device.type, enabled=False):
            q = q_tokens.float()
            r = r_tokens.float()
            hg = h_global.float().view(1, -1)

            qg = self.q_norm(q + self.q_ctx(hg))
            rg = self.r_norm(r + self.r_ctx(hg))
            m, d = qg.shape
            n = int(rg.shape[0])

            qe = qg.unsqueeze(1).expand(m, n, d)
            re = rg.unsqueeze(0).expand(m, n, d)
            global_in = torch.cat([qe, re, qe * re, torch.abs(qe - re)], dim=-1)
            global_score = self.scorer(global_in).squeeze(-1)

            target = torch.nan_to_num(
                pair_match.detach().float(),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            ).clamp_min(0.0)
            target_sum = target.sum()
            if (not torch.isfinite(target_sum)) or float(target_sum.item()) <= 0.0:
                return global_score.to(dtype=h_global.dtype), h_global.new_zeros(())

            target = target / target_sum.clamp_min(1e-8)
            log_prob = F.log_softmax(global_score.reshape(-1), dim=0)
            align_loss = F.kl_div(log_prob, target.reshape(-1), reduction="sum")
            align_loss = torch.nan_to_num(align_loss, nan=0.0, posinf=0.0, neginf=0.0)

        return global_score.to(dtype=h_global.dtype), align_loss.to(device=h_global.device)


class BaselineGatedDTIRealPocket(_SoftPocketBiasModel):
    """
    Soft-pocket-bias 3-view model with pharmacophore evidence-guided local views.
    """

    def __init__(self, *args, **kwargs):
        self.local_match_tau = float(kwargs.pop("local_match_tau", kwargs.pop("LOCAL_MATCH_TAU", 0.7)))
        self.local_evidence_gate = bool(int(kwargs.pop("local_evidence_gate", kwargs.pop("LOCAL_EVIDENCE_GATE", 1))))
        self.local_evidence_pool = str(kwargs.pop("local_evidence_pool", kwargs.pop("LOCAL_EVIDENCE_POOL", "logsumexp"))).lower()
        self.use_global_interaction_align = bool(
            int(kwargs.pop("use_global_interaction_align", kwargs.pop("USE_GLOBAL_INTERACTION_ALIGN", 1)))
        )
        self.use_contribution_calibration = bool(
            int(kwargs.pop("use_contribution_calibration", kwargs.pop("USE_CONTRIBUTION_CALIBRATION", 1)))
        )
        self.local_evidence_score_scale = float(
            kwargs.pop("local_evidence_score_scale", kwargs.pop("LOCAL_EVIDENCE_SCORE_SCALE", 0.5))
        )
        self.local_pocket_bias_scale = float(
            kwargs.pop("local_pocket_bias_scale", kwargs.pop("LOCAL_POCKET_BIAS_SCALE", 1.0))
        )
        self.contrib_calib_top_frac = float(
            kwargs.pop("contrib_calib_top_frac", kwargs.pop("CONTRIB_CALIB_TOP_FRAC", 0.2))
        )
        self.contrib_calib_min_pairs = int(
            kwargs.pop("contrib_calib_min_pairs", kwargs.pop("CONTRIB_CALIB_MIN_PAIRS", 2))
        )
        super().__init__(*args, **kwargs)

        if self.use_pocket and self.pair_cross is not None:
            old_pair = self.pair_cross
            self.pair_cross = EvidencePairScoreLocal(
                self.view_dim,
                dropout=float(getattr(getattr(old_pair, "score_mlp", [None, None, None])[2], "p", 0.1)),
                tau=self.local_match_tau,
                evidence_pool=self.local_evidence_pool,
                evidence_score_scale=self.local_evidence_score_scale,
                pocket_bias_scale=self.local_pocket_bias_scale,
                use_contribution_gate=self.use_contribution_calibration,
            )
            self.pair_cross.load_state_dict(old_pair.state_dict(), strict=False)
        if self.use_pocket and self.use_set_view and self.top_pair_tokenizer is not None:
            old_tok = self.top_pair_tokenizer
            self.top_pair_tokenizer = EvidenceTopPairTokenizer(self.view_dim, dropout=self.set_dropout)
            self.top_pair_tokenizer.load_state_dict(old_tok.state_dict(), strict=True)
        if self.use_pocket and self.use_global_interaction_align:
            self.global_interaction_aligner = GlobalInteractionAligner(self.view_dim, dropout=self.set_dropout)
        else:
            self.global_interaction_aligner = None

        self._aux_outputs: Dict[str, torch.Tensor] = {}

    def _set_empty_local_aux(self, batch_size: int, ref: torch.Tensor) -> None:
        self._aux_outputs = {
            "local_logit": torch.zeros((batch_size,), dtype=ref.dtype, device=ref.device),
            "local_mask": torch.zeros((batch_size,), dtype=torch.bool, device=ref.device),
            "global_align_loss": ref.new_zeros(()),
            "global_align_valid_ratio": ref.new_zeros(()),
            "contrib_calib_mask": torch.zeros((batch_size,), dtype=torch.bool, device=ref.device),
        }

    def _predict_from_views(self, h_global: torch.Tensor, h_pair: torch.Tensor, h_set: torch.Tensor) -> torch.Tensor:
        h_set_eff = h_set if self.use_set_view else torch.zeros_like(h_pair)
        if self.fusion_strategy == "concat":
            fused = torch.cat([h_global, h_pair, h_set_eff], dim=-1)
            return self.concat_classifier(fused).squeeze(-1)
        if self.fusion_strategy == "single_stage":
            fused = self.single_stage_fuse([h_global, h_pair, h_set_eff])
        elif self.fusion_strategy == "mean_sum":
            fused = (h_global + h_pair + h_set_eff) / 3.0
        else:
            h_local = self.set_residual(h_pair, h_set_eff) if self.use_set_view else h_pair
            fused = self.pair_residual(h_global, h_local)
        return self.classifier(fused).squeeze(-1)

    def _pocket_bias_for_graph(
        self,
        batch: Dict[str, torch.Tensor],
        graph_index: int,
        num_nodes: int,
        ref: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if "pocket_graph_score" not in batch or int(num_nodes) <= 0:
            return None
        pgs = batch["pocket_graph_score"]
        if pgs.numel() < int(graph_index) + 1:
            return None
        prior = torch.nan_to_num(
            pgs.view(-1)[int(graph_index)].to(device=ref.device, dtype=ref.dtype),
            nan=0.0,
            posinf=1.0,
            neginf=0.0,
        ).clamp_min(0.0)
        return torch.full((int(num_nodes),), float(prior.detach().item()), dtype=ref.dtype, device=ref.device)

    def _make_perturb_weights(self, pair_weight: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool]:
        weight = torch.nan_to_num(pair_weight.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
        flat = weight.reshape(-1)
        total_pairs = int(flat.numel())
        if total_pairs < max(2, int(self.contrib_calib_min_pairs)):
            return weight, weight, weight, False

        k = max(1, int(round(float(total_pairs) * float(self.contrib_calib_top_frac))))
        k = min(k, total_pairs - 1)
        order = torch.argsort(flat, descending=True)
        high_idx = order[:k]
        low_idx = order[-k:]
        high_mask = torch.zeros_like(flat, dtype=torch.bool)
        low_mask = torch.zeros_like(flat, dtype=torch.bool)
        high_mask[high_idx] = True
        low_mask[low_idx] = True
        high_mask = high_mask.view_as(weight)
        low_mask = low_mask.view_as(weight)

        drop_high = weight.masked_fill(high_mask, 0.0)
        drop_low = weight.masked_fill(low_mask, 0.0)
        keep_high = weight.masked_fill(~high_mask, 0.0)
        valid = bool(drop_high.sum().detach().item() > 0.0 and keep_high.sum().detach().item() > 0.0)
        return drop_high, drop_low, keep_high, valid

    def _build_pair_local_from_weight(
        self,
        pair_out: Dict[str, torch.Tensor],
        pair_weight: torch.Tensor,
        cov_g: torch.Tensor,
        graph_g: torch.Tensor,
    ) -> torch.Tensor:
        pair_g = self.pair_cross.pool_from_weight(
            pair_out["pair_feat"],
            pair_out["q_tokens"],
            pair_out["r_tokens"],
            pair_weight,
        )
        return self.hpair_pocket_fuse(torch.cat([pair_g, cov_g, graph_g], dim=-1))

    def _score_pair_local(
        self,
        pair_local_g: torch.Tensor,
        graph_g: torch.Tensor,
        h_global_b: torch.Tensor,
        batch: Dict[str, torch.Tensor],
        graph_index: int,
    ) -> torch.Tensor:
        score_in = torch.cat([pair_local_g, graph_g, h_global_b, pair_local_g * h_global_b], dim=-1)
        score_g = self.pocket_relevance(score_in)
        if self.use_pocket_score_prior and "pocket_graph_score" in batch:
            pgs = batch["pocket_graph_score"]
            if pgs.numel() >= int(graph_index) + 1:
                prior = pgs.view(-1)[int(graph_index):int(graph_index) + 1].view(1, 1).to(
                    score_g.device,
                    dtype=score_g.dtype,
                )
                score_g = score_g + self.pocket_score_scale * prior
        return score_g

    def _pool_instance_reprs(self, reps: torch.Tensor, scores: torch.Tensor) -> torch.Tensor:
        scores = torch.nan_to_num(scores.view(-1), nan=0.0, posinf=20.0, neginf=-20.0)
        if self.pocket_inst_pool == "mean":
            return reps.mean(dim=0, keepdim=True)
        if self.pocket_inst_pool == "max":
            max_idx = int(torch.argmax(scores).item())
            return reps[max_idx:max_idx + 1]
        alpha = masked_softmax(scores / float(self.pocket_tau), dim=0).unsqueeze(-1).to(dtype=reps.dtype)
        return (reps * alpha).sum(dim=0, keepdim=True)

    def _encode_set_tokens_or_zero(
        self,
        toks: Optional[torch.Tensor],
        scores: Optional[torch.Tensor],
        ref: torch.Tensor,
    ) -> torch.Tensor:
        if (not self.use_set_view) or (self.set_encoder is None) or toks is None or toks.numel() == 0:
            return torch.zeros((1, self.view_dim), dtype=ref.dtype, device=ref.device)
        tok_scores = scores if scores is not None else torch.ones((toks.size(0),), dtype=toks.dtype, device=toks.device)
        if self.set_max_tokens > 0 and toks.size(0) > self.set_max_tokens:
            keep = min(int(self.set_max_tokens), int(toks.size(0)))
            _, idx = torch.topk(tok_scores, k=keep, largest=True)
            toks = toks.index_select(0, idx)
        toks = toks.unsqueeze(0)
        mask = torch.ones((1, toks.size(1)), dtype=torch.bool, device=toks.device)
        return self.set_encoder(toks, mask)

    def _build_local_views(
        self,
        batch: Dict[str, torch.Tensor],
        pharm_tokens: torch.Tensor,
        h_global: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bsz = int(batch["pharm_mask"].shape[0])
        dev = h_global.device
        out_dtype = h_global.dtype

        zero_pair = torch.zeros((bsz, self.view_dim), dtype=out_dtype, device=dev)
        zero_set = torch.zeros((bsz, self.view_dim), dtype=out_dtype, device=dev)
        self._set_empty_local_aux(bsz, h_global)

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

        sample_pair_reprs: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
        sample_pair_scores: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
        sample_set_tokens: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
        sample_set_scores: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
        sample_local_logits: List[List[torch.Tensor]] = [[] for _ in range(bsz)]
        perturb_names = ("drop_high", "drop_low", "keep_high")
        sample_perturb_reprs: Dict[str, List[List[torch.Tensor]]] = {
            name: [[] for _ in range(bsz)] for name in perturb_names
        }
        sample_perturb_scores: Dict[str, List[List[torch.Tensor]]] = {
            name: [[] for _ in range(bsz)] for name in perturb_names
        }
        sample_perturb_set_tokens: Dict[str, List[List[torch.Tensor]]] = {
            name: [[] for _ in range(bsz)] for name in perturb_names
        }
        sample_perturb_set_scores: Dict[str, List[List[torch.Tensor]]] = {
            name: [[] for _ in range(bsz)] for name in perturb_names
        }
        sample_perturb_valid: List[List[bool]] = [[] for _ in range(bsz)]
        global_align_losses: List[torch.Tensor] = []

        for g in range(int(num_graphs)):
            b = int(graph_to_sample[g].item())
            if b < 0 or b >= bsz:
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

            pocket_bias_g = self._pocket_bias_for_graph(batch, g, int(pocket_nodes_g.size(0)), h_global)
            pair_out = self.pair_cross(
                pharm_b,
                pocket_nodes_g.unsqueeze(0),
                h_global=h_global[b:b + 1],
                pocket_bias=pocket_bias_g,
                return_details=True,
            )
            pair_g = pair_out["pair_g"]
            graph_g = pocket_graph_repr[g:g + 1]

            if self.global_interaction_aligner is not None:
                _, align_loss_g = self.global_interaction_aligner(
                    q_tokens=pair_out["q_tokens"],
                    r_tokens=pair_out["r_tokens"],
                    h_global=h_global[b:b + 1],
                    pair_match=pair_out["pair_match"],
                )
                global_align_losses.append(align_loss_g.to(dtype=torch.float32))

            rho = pair_out.get("residue_relevance", None)
            if rho is not None and rho.numel() == pocket_nodes_g.size(0) and float(rho.detach().sum().item()) > 0.0:
                rho_w = rho / rho.sum().clamp_min(1e-8)
                residue_ctx = (pocket_nodes_g * rho_w.unsqueeze(-1).to(dtype=pocket_nodes_g.dtype)).sum(dim=0, keepdim=True)
                graph_g = 0.5 * graph_g + 0.5 * residue_ctx

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
            score_g = self._score_pair_local(pair_local_g, graph_g, h_global[b:b + 1], batch, g)

            sample_pair_reprs[b].append(pair_local_g)
            sample_pair_scores[b].append(score_g)
            sample_local_logits[b].append(pair_out["local_logit"].view(1).to(dtype=out_dtype))

            if self.use_contribution_calibration:
                drop_high_w, drop_low_w, keep_high_w, perturb_valid = self._make_perturb_weights(pair_out["pair_match"])
                perturb_weights = {
                    "drop_high": drop_high_w,
                    "drop_low": drop_low_w,
                    "keep_high": keep_high_w,
                }
                sample_perturb_valid[b].append(bool(perturb_valid))
                for pname, pweight in perturb_weights.items():
                    p_pair_local_g = self._build_pair_local_from_weight(pair_out, pweight, cov_g, graph_g)
                    p_score_g = self._score_pair_local(p_pair_local_g, graph_g, h_global[b:b + 1], batch, g)
                    sample_perturb_reprs[pname][b].append(p_pair_local_g)
                    sample_perturb_scores[pname][b].append(p_score_g)

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
                    pair_match=pair_out.get("pair_match", None),
                    pair_conf=pair_out.get("pair_conf", None),
                )
                if tok_g.numel() > 0:
                    pocket_gate = torch.sigmoid(score_g).view(1, 1)
                    tok_g = tok_g * pocket_gate
                    sample_set_tokens[b].append(tok_g)
                    sample_set_scores[b].append(tok_score_g)

                if self.use_contribution_calibration:
                    for pname, pweight in perturb_weights.items():
                        tok_p, tok_score_p = self.top_pair_tokenizer(
                            pair_score=pair_out["pair_score"],
                            pair_feat=pair_out["pair_feat"],
                            q_tokens=pair_out["q_tokens"],
                            r_tokens=pair_out["r_tokens"],
                            graph_g=graph_g,
                            top_t=self.set_top_pair_per_pocket,
                            max_per_q=self.set_max_per_pharm_token,
                            max_per_r=self.set_max_per_pocket_node,
                            pair_match=pweight,
                            pair_conf=pweight,
                        )
                        if tok_p.numel() > 0:
                            pocket_gate = torch.sigmoid(sample_perturb_scores[pname][b][-1]).view(1, 1)
                            tok_p = tok_p * pocket_gate
                            sample_perturb_set_tokens[pname][b].append(tok_p)
                            sample_perturb_set_scores[pname][b].append(tok_score_p)

        h_pair_list = []
        perturb_pair_lists: Dict[str, List[torch.Tensor]] = {name: [] for name in perturb_names}
        contrib_calib_mask = torch.zeros((bsz,), dtype=torch.bool, device=dev)
        local_logit = torch.zeros((bsz,), dtype=out_dtype, device=dev)
        local_mask = torch.zeros((bsz,), dtype=torch.bool, device=dev)
        for b in range(bsz):
            if len(sample_pair_reprs[b]) == 0:
                h_pair_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                for pname in perturb_names:
                    perturb_pair_lists[pname].append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
            else:
                reps = torch.cat(sample_pair_reprs[b], dim=0)
                scrs = torch.cat(sample_pair_scores[b], dim=0).view(-1)
                h_pair_list.append(self._pool_instance_reprs(reps, scrs))
                for pname in perturb_names:
                    if self.use_contribution_calibration and len(sample_perturb_reprs[pname][b]) > 0:
                        preps = torch.cat(sample_perturb_reprs[pname][b], dim=0)
                        pscrs = torch.cat(sample_perturb_scores[pname][b], dim=0).view(-1)
                        perturb_pair_lists[pname].append(self._pool_instance_reprs(preps, pscrs))
                    else:
                        perturb_pair_lists[pname].append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
                if self.use_contribution_calibration and any(sample_perturb_valid[b]):
                    contrib_calib_mask[b] = True

            if len(sample_local_logits[b]) > 0:
                vals = torch.cat(sample_local_logits[b], dim=0)
                if self.local_evidence_pool == "mean":
                    local_logit[b] = vals.mean()
                elif self.local_evidence_pool == "max":
                    local_logit[b] = vals.max()
                else:
                    local_logit[b] = torch.logsumexp(vals, dim=0)
                local_mask[b] = True

        h_pair = torch.cat(h_pair_list, dim=0).to(dtype=out_dtype)
        h_pair_perturb = {
            pname: torch.cat(perturb_pair_lists[pname], dim=0).to(dtype=out_dtype) for pname in perturb_names
        }
        if len(global_align_losses) > 0:
            global_align_loss = torch.stack(global_align_losses).mean().to(device=dev)
            global_align_valid_ratio = h_global.new_tensor(float(len(global_align_losses)) / float(max(1, int(num_graphs))))
        else:
            global_align_loss = h_global.new_zeros(())
            global_align_valid_ratio = h_global.new_zeros(())
        aux_outputs = {
            "local_logit": local_logit,
            "local_mask": local_mask,
            "global_align_loss": global_align_loss,
            "global_align_valid_ratio": global_align_valid_ratio,
            "contrib_calib_mask": contrib_calib_mask,
            "h_pair_drop_high": h_pair_perturb["drop_high"],
            "h_pair_drop_low": h_pair_perturb["drop_low"],
            "h_pair_keep_high": h_pair_perturb["keep_high"],
        }

        if (not self.use_set_view) or (self.set_encoder is None):
            aux_outputs.update({
                "h_set_drop_high": zero_set,
                "h_set_drop_low": zero_set,
                "h_set_keep_high": zero_set,
            })
            self._aux_outputs = aux_outputs
            return h_pair, zero_set

        h_set_list = []
        perturb_set_lists: Dict[str, List[torch.Tensor]] = {name: [] for name in perturb_names}
        for b in range(bsz):
            if len(sample_set_tokens[b]) == 0:
                h_set_list.append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))
            else:
                toks = torch.cat(sample_set_tokens[b], dim=0)
                tok_scores = torch.cat(sample_set_scores[b], dim=0)
                h_set_list.append(self._encode_set_tokens_or_zero(toks, tok_scores, h_global))
            for pname in perturb_names:
                if self.use_contribution_calibration and len(sample_perturb_set_tokens[pname][b]) > 0:
                    ptoks = torch.cat(sample_perturb_set_tokens[pname][b], dim=0)
                    pscores = torch.cat(sample_perturb_set_scores[pname][b], dim=0)
                    perturb_set_lists[pname].append(self._encode_set_tokens_or_zero(ptoks, pscores, h_global))
                else:
                    perturb_set_lists[pname].append(torch.zeros((1, self.view_dim), dtype=out_dtype, device=dev))

        h_set = torch.cat(h_set_list, dim=0).to(dtype=out_dtype)
        aux_outputs.update({
            "h_set_drop_high": torch.cat(perturb_set_lists["drop_high"], dim=0).to(dtype=out_dtype),
            "h_set_drop_low": torch.cat(perturb_set_lists["drop_low"], dim=0).to(dtype=out_dtype),
            "h_set_keep_high": torch.cat(perturb_set_lists["keep_high"], dim=0).to(dtype=out_dtype),
        })
        self._aux_outputs = aux_outputs
        return h_pair, h_set

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        self._aux_outputs = {}
        drug_node_repr, z_graph = self._run_drug(batch)
        pharm_tokens, z_pharm = self._run_pharm(batch, drug_node_repr)
        z_esm = self._run_protein(batch, pharm_tokens=pharm_tokens)

        h_global = self._build_h_global(z_graph, z_pharm, z_esm)
        h_pair, h_set = self._build_local_views(batch, pharm_tokens, h_global)

        local_logit = self._aux_outputs.get("local_logit", None)
        local_mask = self._aux_outputs.get("local_mask", None)
        local_gate = None
        if self.local_evidence_gate and local_logit is not None and local_mask is not None:
            local_gate = torch.sigmoid(local_logit.float().clamp(-20.0, 20.0)).unsqueeze(-1).to(dtype=h_pair.dtype)
            local_gate = local_gate * local_mask.float().unsqueeze(-1).to(dtype=h_pair.dtype)
            h_pair = h_pair * local_gate
            h_set = h_set * local_gate

        if self.training and self.use_contribution_calibration:
            for pname in ("drop_high", "drop_low", "keep_high"):
                hp = self._aux_outputs.get(f"h_pair_{pname}", None)
                hs = self._aux_outputs.get(f"h_set_{pname}", None)
                if hp is None or hs is None:
                    continue
                hp = hp.to(device=h_pair.device, dtype=h_pair.dtype)
                hs = hs.to(device=h_set.device, dtype=h_set.dtype)
                if local_gate is not None:
                    hp = hp * local_gate
                    hs = hs * local_gate
                self._aux_outputs[f"contrib_{pname}_logit"] = self._predict_from_views(h_global, hp, hs)

        with torch.no_grad():
            dbg = dict(getattr(self, "_debug_scalars", {}) or {})
            if local_logit is not None and local_mask is not None:
                valid = local_mask.bool()
                x = local_logit[valid] if bool(valid.any().item()) else local_logit.view(-1)
                dbg["local_evidence.mean"] = float(x.mean().item()) if x.numel() > 0 else 0.0
                dbg["local_evidence.valid_ratio"] = float(valid.float().mean().item()) if valid.numel() > 0 else 0.0
                if local_gate is not None:
                    dbg["local_gate.mean"] = float(local_gate.mean().item())
            align_loss = self._aux_outputs.get("global_align_loss", None)
            align_valid = self._aux_outputs.get("global_align_valid_ratio", None)
            if align_loss is not None:
                dbg["global_align.loss"] = float(align_loss.detach().float().item())
            if align_valid is not None:
                dbg["global_align.valid_ratio"] = float(align_valid.detach().float().item())
            contrib_mask = self._aux_outputs.get("contrib_calib_mask", None)
            if contrib_mask is not None:
                dbg["contrib_calib.valid_ratio"] = float(contrib_mask.detach().float().mean().item())
            self._debug_scalars = dbg

        return self._predict_from_views(h_global, h_pair, h_set)


class BaselineGatedDTIRealPocket3View(BaselineGatedDTIRealPocket):
    pass


class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
