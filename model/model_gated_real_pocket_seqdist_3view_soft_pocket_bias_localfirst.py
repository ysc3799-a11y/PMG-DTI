from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn

from model.model_gated_real_pocket_seqdist_3view_soft_pocket_bias import (
    BaselineGatedDTIRealPocket as _BaseModel,
)


class BaselineGatedDTIRealPocket(_BaseModel):
    """
    Soft-pocket-bias 3-view model with local-first fusion.

    The original hierarchical fusion uses:
        fused = pair_residual(h_global, h_local)
    which makes the global branch the backbone and local interaction views
    residual additions.

    This variant flips the direction:
        fused = pair_residual(h_local, alpha * h_global)
    so local interaction views become the backbone and the global branch is
    a gated auxiliary shortcut.
    """

    def __init__(self, *args, **kwargs):
        fusion_strategy = str(kwargs.get("fusion_strategy", kwargs.get("FUSION_STRATEGY", "hierarchical"))).lower()
        self.use_local_first_fusion = fusion_strategy in ("local_first", "local-first", "localfirst")
        self.local_first_global_shortcut_init = float(
            kwargs.pop(
                "local_first_global_shortcut_init",
                kwargs.pop("LOCAL_FIRST_GLOBAL_SHORTCUT_INIT", -1.5),
            )
        )
        super().__init__(*args, **kwargs)

        if self.use_local_first_fusion:
            self.fusion_strategy = "local_first"
            self.global_shortcut_logit = nn.Parameter(
                torch.tensor(self.local_first_global_shortcut_init, dtype=torch.float32)
            )
        else:
            self.global_shortcut_logit = None

    def _fuse_local_first(
        self,
        h_global: torch.Tensor,
        h_pair: torch.Tensor,
        h_set: torch.Tensor,
    ) -> torch.Tensor:
        h_set_eff = h_set if self.use_set_view else torch.zeros_like(h_pair)
        h_local = self.set_residual(h_pair, h_set_eff) if self.use_set_view else h_pair

        if self.global_shortcut_logit is None:
            alpha = h_local.new_tensor(1.0)
        else:
            alpha = torch.sigmoid(self.global_shortcut_logit).to(device=h_local.device, dtype=h_local.dtype)
        fused = self.pair_residual(h_local, alpha * h_global)

        dbg = getattr(self, "_debug_scalars", None)
        if isinstance(dbg, dict):
            dbg = dict(dbg)
            dbg["sigmoid(global_shortcut_logit)"] = float(alpha.detach().float().item())
            self._debug_scalars = dbg
        return fused

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if not self.use_local_first_fusion:
            return super().forward(batch)

        drug_node_repr, z_graph = self._run_drug(batch)
        pharm_tokens, z_pharm = self._run_pharm(batch, drug_node_repr)
        z_esm = self._run_protein(batch, pharm_tokens=pharm_tokens)

        h_global = self._build_h_global(z_graph, z_pharm, z_esm)
        h_pair, h_set = self._build_local_views(batch, pharm_tokens, h_global)
        fused = self._fuse_local_first(h_global, h_pair, h_set)
        logit = self.classifier(fused).squeeze(-1)
        return logit


class BaselineGatedDTIRealPocket3View(BaselineGatedDTIRealPocket):
    pass


class BaselineGatedDTIMultiPocket(BaselineGatedDTIRealPocket):
    pass
