from typing import List, Optional, Tuple

import torch
import torch.nn as nn

from model.model import (
    DrugGraphEncoder,
    PharmTokenEncoder,
    PharmTokenEnhancer,
    ProteinTokenEncoder,
    ScaledDotAttn,
    masked_mean_pool,
    safe_softmax,
    build_topk_mask,
)


def build_topk_group_masks(
    scores: torch.Tensor,
    mask: torch.Tensor,
    k: int,
    num_groups: int,
    min_keep: int = 8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build disjoint top-k masks for multiple pocket instances.
    scores: [B, L], mask: [B, L] bool -> group_masks [B, G, L] bool, group_valid [B, G] bool
    """
    if mask.dtype != torch.bool:
        mask = mask.bool()
    B, L = scores.shape
    num_groups = int(max(1, num_groups))
    k = int(k)
    min_keep = int(min_keep)
    group_masks = torch.zeros((B, num_groups, L), device=scores.device, dtype=torch.bool)
    group_valid = torch.zeros((B, num_groups), device=scores.device, dtype=torch.bool)
    if num_groups <= 1:
        group_masks[:, 0] = build_topk_mask(scores, mask, k=k, min_keep=min_keep)
        group_valid[:, 0] = group_masks[:, 0].any(dim=1)
        return group_masks, group_valid

    k = max(k, min_keep)
    for b in range(B):
        remaining = mask[b].clone()
        if not remaining.any():
            continue
        for gi in range(num_groups):
            n_remain = int(remaining.sum().item())
            if n_remain == 0:
                break
            kb = min(k, n_remain)
            sc = scores[b].clone()
            sc[~remaining] = torch.finfo(sc.dtype).min
            idx = torch.topk(sc, k=kb, largest=True).indices
            group_masks[b, gi, idx] = True
            group_valid[b, gi] = True
            remaining[idx] = False
    return group_masks, group_valid


class GatedFusion(nn.Module):
    def __init__(
        self,
        in_dims: List[int],
        fusion_dim: int,
        dropout: float = 0.1,
        gate_hidden: Optional[int] = None,
    ):
        super().__init__()
        if not in_dims:
            raise ValueError("in_dims must be non-empty")
        self.num_modalities = len(in_dims)
        self.projs = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(d, fusion_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                )
                for d in in_dims
            ]
        )
        gate_hidden = int(gate_hidden or fusion_dim)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * self.num_modalities, gate_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gate_hidden, self.num_modalities),
        )

    def forward(self, feats: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(feats) != self.num_modalities:
            raise ValueError(f"Expected {self.num_modalities} modalities, got {len(feats)}")
        proj = [self.projs[i](feats[i]) for i in range(self.num_modalities)]
        gate_in = torch.cat(proj, dim=-1)
        gate_logits = self.gate(gate_in)
        weights = torch.softmax(gate_logits, dim=-1)
        fused = proj[0] * weights[:, 0:1]
        for i in range(1, self.num_modalities):
            fused = fused + proj[i] * weights[:, i:i + 1]
        return fused, weights


class PharmTokenPool(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Linear(dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask = mask.bool()
        scores = self.score(tokens).squeeze(-1)  # [B,P]
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg)
        weights = torch.softmax(scores, dim=1)
        weights = torch.where(torch.isfinite(weights), weights, torch.zeros_like(weights))
        weights = self.drop(weights)
        weights = weights * mask.float()
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return (tokens * weights.unsqueeze(-1)).sum(dim=1)  # [B,D]


class BaselineGatedDTIMultiPocket(nn.Module):
    """
    Same encoders as BaselineConcatDTI, but fuse modalities with gated fusion.
    Multi-instance pocket option pools multiple pseudo-pocket vectors.
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
        use_pp_attn: bool = True,
        prot_pool: str = "attn",
        pp_attn_dropout: Optional[float] = None,
        # pseudo-pocket from pharm->protein attention (conditioned pocket)
        use_pocket: bool = True,
        pocket_pool: str = "softmax",      # "softmax" | "hard"
        pocket_k: int = 32,
        pocket_min_keep: int = 8,
        pocket_tau: float = 0.3,
        pocket_detach_score: bool = True,
        pocket_dim: int = 64,
        pocket_instances: int = 1,
        pocket_inst_pool: str = "mean",  # "mean" | "attn"
        # gated fusion
        fusion_dim: Optional[int] = None,
        gate_hidden: Optional[int] = None,
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
        if self.use_esm_proj:
            self.esm_proj = nn.Sequential(
                nn.Linear(self.esm_dim, self.esm_proj_dim),
                nn.LayerNorm(self.esm_proj_dim),
            )
        else:
            self.esm_proj = None

        self.use_pp_attn = bool(use_pp_attn)
        if self.use_pp_attn:
            self.pp_attn = ScaledDotAttn(q_dim=pharm_out, kv_dim=prot_dim, out_dim=pharm_out, dropout=float(pp_attn_dropout or dropout))

        self.prot_enc = ProteinTokenEncoder(in_dim=prot_dim, pool=prot_pool)
        # pseudo-pocket settings
        self.use_pocket = bool(use_pocket and self.use_pp_attn)
        self.pocket_pool = str(pocket_pool)
        self.pocket_k = int(pocket_k)
        self.pocket_min_keep = int(pocket_min_keep)
        self.pocket_tau = float(pocket_tau)
        self.pocket_detach_score = bool(pocket_detach_score)
        self.pocket_dim = int(pocket_dim)
        self.pocket_instances = int(max(1, pocket_instances))
        self.pocket_inst_pool = str(pocket_inst_pool).lower()
        if self.use_pocket:
            self.pocket_proj = nn.Sequential(
                nn.Linear(pharm_out, self.pocket_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            # learnable gate to let model down-weight noisy pocket early
            self.pocket_gate = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))  # sigmoid~0.12
            if self.pocket_inst_pool == "attn" and self.pocket_instances > 1:
                self.pocket_inst_attn = nn.Linear(self.pocket_dim, 1)
            else:
                self.pocket_inst_attn = None
        else:
            self.pocket_proj = None
            self.pocket_gate = None
            self.pocket_inst_attn = None

        fusion_dim = int(fusion_dim or mlp_hidden)
        in_dims = [gcn_out, pharm_out, prot_dim]
        if self.use_pocket:
            in_dims.append(self.pocket_dim)
        self.fusion = GatedFusion(in_dims=in_dims, fusion_dim=fusion_dim, dropout=dropout, gate_hidden=gate_hidden)

        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, 1),
        )

    def forward(self, batch: dict) -> torch.Tensor:
        node_repr, z_graph = self.drug_gcn(
            batch["node_feat"],
            batch["edge_index"],
            batch["node_slices"],
            edge_attr=batch.get("edge_attr", None),
        )  # [N_total,Dg], [B,Dg]

        pharm_tokens = self.pharm_tok(batch["pharm_type"], batch["pharm_mask"])  # [B,P,Dp]
        pharm_anchor = batch.get("pharm_anchor", None)
        pharm_tokens = self.pharm_enh(
            pharm_tokens,
            batch["pharm_mask"],
            node_repr,
            batch["node_slices"],
            pharm_anchor=pharm_anchor,
        )  # [B,P,Dp]

        esm_tok = batch["esm2_tok"]
        if self.esm_proj is not None:
            esm_tok = self.esm_proj(esm_tok)

        # pharm -> protein cross-attention (Query=pharm tokens, KV=protein tokens)
        pocket_vec = None

        if self.use_pp_attn:
            # pharm -> protein cross-attention (and get attention weights for pseudo-pocket)
            pharm_tokens, attn_pp = self.pp_attn(
                pharm_tokens, esm_tok, kv_mask=batch["esm2_mask"], return_attn=True
            )  # pharm_tokens: [B,P,Dp], attn_pp: [B,P,L]
            pharm_tokens = pharm_tokens * batch["pharm_mask"].unsqueeze(-1).float()

            if self.use_pocket and self.pocket_proj is not None:
                # compute residue scores from attention weights (conditioned on pharm tokens)
                prot_mask = batch["esm2_mask"].bool()
                pm = batch["pharm_mask"].float().unsqueeze(-1)  # [B,P,1]
                w = attn_pp * pm
                denom = pm.sum(dim=1).clamp_min(1.0)

                # Step2-1: attn_pp is already a probability distribution (softmax over residues).
                # Convert to log-prob "logits" so that subsequent safe_softmax(scores, tau)
                # performs meaningful temperature scaling (equivalently p^(1/tau)).
                attn_mean = w.sum(dim=1) / denom  # [B,L] in [0,1]
                attn_mean = attn_mean.masked_fill(~prot_mask, 0.0)  # avoid log(0) on invalid residues
                scores = torch.log(attn_mean.clamp_min(1e-9))       # [B,L] log-prob
                neg = torch.finfo(scores.dtype).min
                scores = scores.masked_fill(~prot_mask, neg)

                if self.pocket_detach_score:
                    scores = scores.detach()
                scores = torch.where(torch.isfinite(scores), scores, torch.full_like(scores, neg))

                # pool on value-projected protein tokens (aligned with cross-attn space)
                prot_v = self.pp_attn.v_proj(esm_tok)
                if self.pocket_instances <= 1:
                    if self.pocket_pool.lower() == "hard":
                        pocket_mask = build_topk_mask(scores, prot_mask, k=self.pocket_k, min_keep=self.pocket_min_keep)
                        pocket_vec_raw = masked_mean_pool(prot_v, pocket_mask.long(), dim=1)  # [B,De]
                    else:
                        pocket_w = safe_softmax(scores, prot_mask, tau=self.pocket_tau)  # [B,L]
                        pocket_vec_raw = (prot_v * pocket_w.unsqueeze(-1)).sum(dim=1)  # [B,De]
                    pocket_vec = self.pocket_proj(pocket_vec_raw)  # [B, pocket_dim]
                else:
                    group_masks, group_valid = build_topk_group_masks(
                        scores,
                        prot_mask,
                        k=self.pocket_k,
                        num_groups=self.pocket_instances,
                        min_keep=self.pocket_min_keep,
                    )
                    pocket_raw = []
                    for gi in range(self.pocket_instances):
                        gmask = group_masks[:, gi, :]
                        if self.pocket_pool.lower() == "hard":
                            vec = masked_mean_pool(prot_v, gmask.long(), dim=1)
                        else:
                            pocket_w = safe_softmax(scores, gmask, tau=self.pocket_tau)
                            vec = (prot_v * pocket_w.unsqueeze(-1)).sum(dim=1)
                        pocket_raw.append(vec)
                    pocket_raw = torch.stack(pocket_raw, dim=1)  # [B,G,De]
                    pocket_vecs = self.pocket_proj(pocket_raw)  # [B,G,pocket_dim]
                    if self.pocket_inst_attn is not None:
                        inst_scores = self.pocket_inst_attn(pocket_vecs).squeeze(-1)  # [B,G]
                        inst_scores = inst_scores.masked_fill(~group_valid, torch.finfo(inst_scores.dtype).min)
                        inst_w = torch.softmax(inst_scores, dim=1)
                        inst_w = inst_w * group_valid.float()
                        inst_w = inst_w / inst_w.sum(dim=1, keepdim=True).clamp_min(1e-8)
                        pocket_vec = (pocket_vecs * inst_w.unsqueeze(-1)).sum(dim=1)
                    else:
                        denom = group_valid.float().sum(dim=1, keepdim=True).clamp_min(1.0)
                        pocket_vec = (pocket_vecs * group_valid.unsqueeze(-1).float()).sum(dim=1) / denom

                # apply gate (if enabled)
                if self.pocket_gate is not None:
                    pocket_vec = torch.sigmoid(self.pocket_gate) * pocket_vec

        if self.pharm_self_attn is not None:
            pharm_mask = batch["pharm_mask"].bool()
            # Avoid NaNs when a sample has no pharm tokens (all mask=0).
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

        z_pharm = masked_mean_pool(pharm_tokens, batch["pharm_mask"], dim=1)  # [B,Dp]
        if self.pharm_pooler is not None:
            z_pharm = self.pharm_pooler(pharm_tokens, batch["pharm_mask"])
        z_esm = self.prot_enc(esm_tok, batch["esm2_mask"])          # [B,De]

        feats = [z_graph, z_pharm, z_esm]
        if self.use_pocket and pocket_vec is not None:
            feats.append(pocket_vec)

        fused, _ = self.fusion(feats)
        logit = self.mlp(fused).squeeze(-1)
        return logit
