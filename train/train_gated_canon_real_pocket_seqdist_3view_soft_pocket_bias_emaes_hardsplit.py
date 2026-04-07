import argparse
import csv
import json
import os
import random
import sys
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple
from contextlib import nullcontext
from functools import partial

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
for _p in (os.path.dirname(_THIS), _ROOT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.dataset_canon_real_pocket_seqdist import (
    HumanDTIDatasetCanonRealPocket,
    collate_fn_real_pocket,
)
from dataset.metrics import compute_metrics
from model.model_gated_real_pocket_seqdist_3view_soft_pocket_bias_hardsplit import BaselineGatedDTIRealPocket


class EMAHelper:
    def __init__(self, model, decay: float = 0.999, device: str = ""):
        self.decay = float(decay)
        self.device = torch.device(device) if device else None
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                t = param.detach().clone()
                if self.device is not None:
                    t = t.to(self.device)
                self.shadow[name] = t
        for name, buf in model.named_buffers():
            t = buf.detach().clone()
            if self.device is not None:
                t = t.to(self.device)
            self.shadow[f"__buf__::{name}"] = t

    @torch.no_grad()
    def update(self, model):
        one_minus = 1.0 - self.decay
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            data = param.detach()
            if self.device is not None:
                data = data.to(self.device)
            self.shadow[name].mul_(self.decay).add_(data, alpha=one_minus)
        for name, buf in model.named_buffers():
            data = buf.detach()
            if self.device is not None:
                data = data.to(self.device)
            self.shadow[f"__buf__::{name}"].copy_(data)

    @torch.no_grad()
    def store(self, model):
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.detach().clone()
        for name, buf in model.named_buffers():
            self.backup[f"__buf__::{name}"] = buf.detach().clone()

    @torch.no_grad()
    def copy_to(self, model):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            src = self.shadow[name]
            if src.device != param.device:
                src = src.to(param.device)
            param.data.copy_(src)
        for name, buf in model.named_buffers():
            src = self.shadow[f"__buf__::{name}"]
            if src.device != buf.device:
                src = src.to(buf.device)
            buf.data.copy_(src)

    @torch.no_grad()
    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.backup[name])
        for name, buf in model.named_buffers():
            buf.data.copy_(self.backup[f"__buf__::{name}"])
        self.backup = {}


def _get(cfg: Dict[str, Any], key_path: str, default=None):
    cur = cfg
    for k in key_path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def move_batch_to_device(batch: Dict[str, Any], device: str) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def get_model_debug_scalars(model) -> Dict[str, float]:
    m = model.module if hasattr(model, "module") else model
    d = getattr(m, "_debug_scalars", None)
    return d if isinstance(d, dict) else {}


def amp_autocast_ctx(device: str, enabled: bool):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        dev = "cuda" if str(device).startswith("cuda") else "cpu"
        return torch.amp.autocast(dev, enabled=True)
    return torch.cuda.amp.autocast(enabled=enabled)


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    z = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-z))


def select_threshold(y_true: np.ndarray, logits: np.ndarray, metric: str = "f1") -> float:
    _ = sigmoid_np(logits)
    best_thr = 0.5
    best_val = -1.0
    for thr in np.arange(0.05, 0.951, 0.01):
        m = compute_metrics(y_true, logits, threshold=float(thr))
        v = float(m.get(metric, float("nan")))
        if np.isfinite(v) and v > best_val:
            best_val = v
            best_thr = float(thr)
    return best_thr


def summarize_logit_distribution(y_true: np.ndarray, logits: np.ndarray) -> Dict[str, float]:
    pos = logits[y_true > 0.5]
    neg = logits[y_true <= 0.5]
    pos_mean = float(np.mean(pos)) if pos.size else float("nan")
    pos_std = float(np.std(pos)) if pos.size else float("nan")
    neg_mean = float(np.mean(neg)) if neg.size else float("nan")
    neg_std = float(np.std(neg)) if neg.size else float("nan")
    return {
        "pos_logit_mean": pos_mean,
        "pos_logit_std": pos_std,
        "neg_logit_mean": neg_mean,
        "neg_logit_std": neg_std,
        "logit_gap": pos_mean - neg_mean if np.isfinite(pos_mean) and np.isfinite(neg_mean) else float("nan"),
    }


def _dataset_label_array(ds) -> np.ndarray:
    if len(ds) == 0:
        return np.zeros((0,), dtype=np.float32)
    return ds.df.iloc[list(ds.indices)]["Y"].to_numpy(dtype=np.float32).reshape(-1)


def build_train_sampler(labels: np.ndarray, mode: str):
    mode = str(mode or "shuffle").strip().lower()
    if labels.size == 0 or mode in {"", "shuffle", "random", "none", "off", "disable", "disabled"}:
        return None
    pos = int((labels > 0.5).sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return None
    if mode not in {"balanced", "balanced_replacement"}:
        raise ValueError(f"Unsupported TRAIN_SAMPLER={mode}")
    weights = np.where(labels > 0.5, 0.5 / float(pos), 0.5 / float(neg))
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=int(labels.size),
        replacement=True,
    )


def compute_pos_weight(labels: np.ndarray, mode: str, clip_max: float = 4.0, power: float = 1.0, device: str = "cpu"):
    mode = str(mode or "none").strip().lower()
    if labels.size == 0 or mode in {"", "none", "off", "disable", "disabled", "0"}:
        return None
    pos = int((labels > 0.5).sum())
    neg = int(labels.size - pos)
    if pos == 0 or neg == 0:
        return None

    ratio = float(neg) / float(max(pos, 1))
    if mode in {"ratio", "balanced"}:
        value = ratio
    elif mode in {"sqrt", "sqrt_ratio"}:
        value = float(np.sqrt(ratio))
    else:
        raise ValueError(f"Unsupported POS_WEIGHT_MODE={mode}")

    value = max(1.0, float(value))
    if clip_max > 0:
        value = min(value, float(clip_max))
    if power != 1.0:
        value = float(value) ** float(power)
    return torch.tensor([value], dtype=torch.float32, device=device)


def ramp_weight(base: float, epoch: int, warmup_epochs: int) -> float:
    base = float(base)
    warmup_epochs = int(max(0, warmup_epochs))
    if base <= 0.0 or warmup_epochs <= 0:
        return base
    return base * min(1.0, float(epoch) / float(max(1, warmup_epochs)))


def bce_with_optional_smoothing(
    logit: torch.Tensor,
    y: torch.Tensor,
    smoothing: float = 0.0,
    pos_weight: torch.Tensor = None,
) -> torch.Tensor:
    if smoothing > 0.0:
        y = y * (1.0 - smoothing) + 0.5 * smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, y, pos_weight=pos_weight)


def pairwise_auc_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = logit[y > 0.5].view(-1)
    neg = logit[y <= 0.5].view(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return logit.new_zeros(())
    margin = neg.unsqueeze(0) - pos.unsqueeze(1)
    return torch.nn.functional.softplus(margin).mean()


def gather_eval(model, loader, device: str, amp: bool, label_smoothing: float = 0.0) -> Tuple[np.ndarray, np.ndarray, float]:
    model.eval()
    all_logits: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []
    loss_sum = 0.0
    n = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc="eval", leave=False, dynamic_ncols=True)

    with torch.no_grad():
        for batch, y in it:
            batch = move_batch_to_device(batch, device)
            y = y.to(device)
            with amp_autocast_ctx(device, amp):
                logit = model(batch)
                loss = bce_with_optional_smoothing(logit, y, smoothing=label_smoothing)
            loss_sum += float(loss.item()) * int(y.shape[0])
            n += int(y.shape[0])
            all_logits.append(logit.detach().cpu().numpy().reshape(-1))
            all_labels.append(y.detach().cpu().numpy().reshape(-1))

    logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0,), dtype=np.float32)
    labels = np.concatenate(all_labels, axis=0) if all_labels else np.zeros((0,), dtype=np.float32)
    mean_loss = loss_sum / max(n, 1)
    return labels, logits, mean_loss


def run_train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device: str,
    amp: bool,
    grad_accum: int,
    epoch: int,
    ema: EMAHelper = None,
    label_smoothing: float = 0.0,
    pairwise_weight: float = 0.0,
    pos_weight: torch.Tensor = None,
    debug_csv_path: str = "",
    debug_every: int = 1,
) -> Dict[str, float]:
    model.train()
    total_loss_sum = 0.0
    bce_sum = 0.0
    pair_sum = 0.0
    n = 0

    it = loader
    if tqdm is not None:
        it = tqdm(loader, desc=f"train {epoch:03d}", leave=False, dynamic_ncols=True)

    optimizer.zero_grad(set_to_none=True)
    grad_accum = max(1, int(grad_accum))
    step = 0
    has_scaled_grads = False
    debug_every = max(1, int(debug_every))
    dbg_fh = None
    dbg_writer = None
    if debug_csv_path:
        dbg_fh = open(debug_csv_path, "a", newline="", encoding="utf-8")
        dbg_writer = csv.writer(dbg_fh)

    try:
        for step, (batch, y) in enumerate(it, start=1):
            batch = move_batch_to_device(batch, device)
            y = y.to(device)
            with amp_autocast_ctx(device, amp):
                logit = model(batch)
                bce = bce_with_optional_smoothing(logit, y, smoothing=label_smoothing, pos_weight=pos_weight)
                pair = pairwise_auc_loss(logit, y)
                loss_raw = bce + float(pairwise_weight) * pair
                loss = loss_raw / grad_accum

            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            has_scaled_grads = True
            if step % grad_accum == 0:
                if has_scaled_grads:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer)
                    scaler.update()
                    if ema is not None:
                        ema.update(model)
                    has_scaled_grads = False
                optimizer.zero_grad(set_to_none=True)

            bs = int(y.shape[0])
            total_loss_sum += float(loss_raw.item()) * bs
            bce_sum += float(bce.item()) * bs
            pair_sum += float(pair.item()) * bs
            n += bs

            dbg = get_model_debug_scalars(model)
            if dbg_writer is not None and dbg and (step == 1 or step % debug_every == 0):
                dbg_writer.writerow([
                    int(epoch),
                    int(step),
                    float(dbg.get("pocket_bias.mean", float("nan"))),
                    float(dbg.get("pocket_bias.max", float("nan"))),
                    float(dbg.get("pocket_bias.nonzero_ratio", float("nan"))),
                    float(dbg.get("softplus(bias_log_scale)", float("nan"))),
                    float(dbg.get("sigmoid(pp_gate)", float("nan"))),
                    float(dbg.get("pp_attn.gate", float("nan"))),
                ])
                dbg_fh.flush()
            if tqdm is not None:
                it.set_postfix(loss=f"{total_loss_sum / max(n, 1):.4f}")

        if step > 0 and (step % grad_accum) != 0 and has_scaled_grads:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
    finally:
        if dbg_fh is not None:
            dbg_fh.close()

    return {
        "train_loss": total_loss_sum / max(n, 1),
        "train_bce": bce_sum / max(n, 1),
        "train_pairwise": pair_sum / max(n, 1),
    }


def infer_num_pharm_types(pharm_path: str) -> int:
    payload = torch.load(pharm_path, map_location="cpu")
    if isinstance(payload, dict):
        if "num_types" in payload:
            return int(payload["num_types"])
        if "token2id" in payload and isinstance(payload["token2id"], dict) and payload["token2id"]:
            return int(max(payload["token2id"].values()) + 1)
        if "pharm_type" in payload and isinstance(payload["pharm_type"], torch.Tensor):
            return int(payload["pharm_type"].max().item()) + 1
        if "family2id" in payload and isinstance(payload["family2id"], dict):
            return int(len(payload["family2id"]))
    return 64


def build_model(cfg: Dict[str, Any], num_pharm_types: int, esm_dim: int, node_in: int, use_multiview: bool):
    mcfg = cfg.get("MODEL", {})

    common = dict(
        node_in=int(mcfg.get("NODE_IN", node_in)),
        gcn_hidden=int(mcfg.get("GCN_HIDDEN", 128)),
        gcn_out=int(mcfg.get("GCN_OUT", 128)),
        gcn_layers=int(mcfg.get("GCN_LAYERS", 2)),
        num_pharm_types=int(mcfg.get("NUM_PHARM_TYPES", num_pharm_types)),
        pharm_emb=int(mcfg.get("PHARM_EMB", 64)),
        pharm_out=int(mcfg.get("PHARM_OUT", 64)),
        pharm_pool=str(mcfg.get("PHARM_POOL", "mean")),
        pharm_self_attn=bool(int(mcfg.get("PHARM_SELF_ATTN", 0))),
        pharm_self_nhead=int(mcfg.get("PHARM_SELF_NHEAD", 4)),
        pharm_self_dropout=float(mcfg.get("PHARM_SELF_DROPOUT", mcfg.get("DROPOUT", 0.1))),
        esm_dim=int(esm_dim),
        esm_proj_dim=mcfg.get("ESM_PROJ_DIM", None),
        mlp_hidden=int(mcfg.get("MLP_HIDDEN", 256)),
        dropout=float(mcfg.get("DROPOUT", 0.1)),
        use_anchor=bool(int(mcfg.get("USE_ANCHOR", 1))),
        pharm_use_graph_attn=bool(int(mcfg.get("PHARM_USE_GRAPH_ATTN", 1))),
        prot_pool=str(mcfg.get("PROT_POOL", "attn")),
        use_graph_branch=bool(int(mcfg.get("USE_GRAPH_BRANCH", 1))),
        use_pharm_branch=bool(int(mcfg.get("USE_PHARM_BRANCH", 1))),
        use_sequence_branch=bool(int(mcfg.get("USE_SEQUENCE_BRANCH", 1))),
        use_pocket=bool(int(mcfg.get("USE_POCKET", 1))),
        pocket_dim=int(mcfg.get("POCKET_DIM", 64)),
        fusion_dim=mcfg.get("FUSION_DIM", None),
        gate_hidden=mcfg.get("GATE_HIDDEN", None),
        fusion_strategy=str(mcfg.get("FUSION_STRATEGY", "hierarchical")),
    )

    if use_multiview:
        print("[WARN] USE_MULTIVIEW=1 is ignored in 3view training script; forcing 3view model.")

    return BaselineGatedDTIRealPocket(
        **common,
        use_pp_attn=bool(int(mcfg.get("USE_PP_ATTN", 1))),
        pp_attn_dropout=float(mcfg.get("PP_ATTN_DROPOUT", mcfg.get("DROPOUT", 0.1))),
        pocket_node_proj=mcfg.get("POCKET_NODE_PROJ", 128),
        pocket_gnn_hidden=int(mcfg.get("POCKET_GNN_HIDDEN", 128)),
        pocket_gnn_layers=int(mcfg.get("POCKET_GNN_LAYERS", 3)),
        pocket_struct_dim=int(mcfg.get("POCKET_STRUCT_DIM", cfg.get("DATA", {}).get("POCKET_STRUCT_DIM", 29))),
        pocket_use_struct_feat=bool(int(mcfg.get("POCKET_USE_STRUCT_FEAT", cfg.get("DATA", {}).get("POCKET_USE_STRUCT_FEAT", cfg.get("DATA", {}).get("POCKET_STRUCT_FEAT", 1))))),
        pocket_edge_feat_dim=int(mcfg.get("POCKET_EDGE_FEAT_DIM", 1 + int(cfg.get("DATA", {}).get("POCKET_EDGE_RBF_BINS", 16)) + 3 + (1 if bool(int(cfg.get("DATA", {}).get("POCKET_EDGE_USE_SEQDIST", 1))) else 0))),
        pocket_edge_hidden=int(mcfg.get("POCKET_EDGE_HIDDEN", 32)),
        pocket_inst_pool=str(mcfg.get("POCKET_INST_POOL", "attn")),
        pocket_tau=float(mcfg.get("POCKET_TAU", 1.0)),
        pocket_score_prior=bool(int(mcfg.get("POCKET_SCORE_PRIOR", 1))),
        pocket_score_scale=float(mcfg.get("POCKET_SCORE_SCALE", 1.0)),
        use_set_view=bool(int(mcfg.get("USE_SET_VIEW", 1))),
        set_top_pair_per_pocket=int(mcfg.get("SET_TOP_PAIR_PER_POCKET", 4)),
        set_max_tokens=int(mcfg.get("SET_MAX_TOKENS", 12)),
        set_max_per_pharm_token=int(mcfg.get("SET_MAX_PER_PHARM_TOKEN", 2)),
        set_num_layers=int(mcfg.get("SET_NUM_LAYERS", 1)),
        set_num_heads=int(mcfg.get("SET_NUM_HEADS", 4)),
        set_dropout=float(mcfg.get("SET_DROPOUT", mcfg.get("DROPOUT", 0.1))),
        set_use_cls=bool(int(mcfg.get("SET_USE_CLS", mcfg.get("USE_SET_CLS", 1)))),
        set_gate_init_bias=float(mcfg.get("SET_GATE_INIT_BIAS", -2.0)),
        use_soft_pocket_bias=bool(int(mcfg.get("USE_SOFT_POCKET_BIAS", 1))),
        pocket_bias_margin=float(mcfg.get("POCKET_BIAS_MARGIN", 0.45)),
        pocket_bias_pool=str(mcfg.get("POCKET_BIAS_POOL", "max")),
        pocket_bias_use_prior=bool(int(mcfg.get("POCKET_BIAS_USE_PRIOR", 1))),
        pocket_bias_prior_power=float(mcfg.get("POCKET_BIAS_PRIOR_POWER", 1.0)),
        pocket_bias_reduce=str(mcfg.get("POCKET_BIAS_REDUCE", "max")),
        pocket_bias_l2norm=bool(int(mcfg.get("POCKET_BIAS_L2NORM", 1))),
        pp_gate_init=mcfg.get("PP_GATE_INIT", None),
        pp_attn_gate_init=mcfg.get("PP_ATTN_GATE_INIT", None),
    )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cfg", required=True, help="YAML config")
    p.add_argument("--split_path", default=None, help="override DATA.SPLIT")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--save_dir", default=None)
    p.add_argument("--run_name", default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--threshold_metric", default="f1", choices=["f1", "bacc", "mcc", "acc"])
    p.add_argument("--use_multiview", type=int, default=None, help="1: multiview model, 0: baseline gated model")
    p.add_argument("--eval_only_ckpt", default=None, help="evaluate an existing checkpoint and exit")
    return p.parse_args()


def _save_ckpt(path: str, model, epoch: int, metric_name: str, best_val: float, best_thr: float, seed: int, cfg, args, extra: Dict[str, Any] = None):
    payload = {
        "model": deepcopy(model.state_dict()),
        "epoch": int(epoch),
        "best_metric": metric_name,
        "best_val": float(best_val),
        "best_threshold": float(best_thr),
        "seed": int(seed),
        "cfg": cfg,
        "args": vars(args),
    }
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def main():
    args = parse_args()
    cfg = load_yaml(args.cfg)

    proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def _abs(path: str) -> str:
        if not path:
            return path
        return path if os.path.isabs(path) else os.path.normpath(os.path.join(proj_dir, path))

    data = cfg.get("DATA", {})
    solver = cfg.get("SOLVER", {})
    result = cfg.get("RESULT", {})
    model_cfg = cfg.get("MODEL", {})

    csv_path = _abs(data.get("CSV", "data/human/full_with_pid.csv"))
    split_path = _abs(args.split_path or data.get("SPLIT", "data/human/splits/human_random_seed42.json"))
    pharm_path = _abs(data.get("PHARM", "data/human/precomputed/pharm_max64_seed42.pt"))
    esm2_path = _abs(data.get("ESM2", "data/human/precomputed/esm2_tokens_t6_fp16.pt"))

    p2rank_work_dir = _abs(data.get("P2RANK_WORK_DIR", ""))
    p2rank_csv = _abs(data.get("PROTEIN_POCKETS_CSV", "")) if data.get("PROTEIN_POCKETS_CSV") else None
    pdb_index_csv = _abs(data.get("PDB_INDEX_CSV", "")) if data.get("PDB_INDEX_CSV") else None
    p2rank_out_dir = _abs(data.get("P2RANK_OUT_DIR", "")) if data.get("P2RANK_OUT_DIR") else None
    pocket_cache_dir = _abs(data.get("POCKET_CACHE_DIR", "")) if data.get("POCKET_CACHE_DIR") else None
    drug_graph_cache_path = _abs(data.get("DRUG_GRAPH_CACHE", "")) if data.get("DRUG_GRAPH_CACHE") else None

    for need in [csv_path, split_path, pharm_path, esm2_path]:
        if not os.path.isfile(need):
            raise FileNotFoundError(f"Required file not found: {need}")

    seed = int(args.seed if args.seed is not None else solver.get("SEED", 42))
    seed_all(seed)

    device = str(args.device if args.device is not None else solver.get("DEVICE", "cuda:0"))
    if device.startswith("cuda") and not torch.cuda.is_available():
        device = "cpu"

    batch_size = int(args.batch_size if args.batch_size is not None else solver.get("BATCH_SIZE", 16))
    epochs = int(args.epochs if args.epochs is not None else solver.get("EPOCHS", 50))
    lr = float(args.lr if args.lr is not None else solver.get("LR", 3e-4))
    weight_decay = float(solver.get("WEIGHT_DECAY", 3e-5))
    main_metric = str(solver.get("MAIN_METRIC", "auc")).lower()
    amp = bool(int(solver.get("AMP", 1))) and device.startswith("cuda")
    grad_accum = int(max(1, solver.get("GRAD_ACCUM", 1)))
    num_workers = int(max(0, solver.get("NUM_WORKERS", 0)))
    label_smoothing = float(solver.get("LABEL_SMOOTHING", 0.0))
    pairwise_weight = float(solver.get("PAIRWISE_AUC_LOSS_WEIGHT", 0.0))
    pairwise_warmup_epochs = int(max(0, solver.get("PAIRWISE_WARMUP_EPOCHS", 0)))
    use_ema = bool(int(solver.get("EMA", 1)))
    ema_decay = float(solver.get("EMA_DECAY", 0.999))
    ema_start_epoch = int(max(1, solver.get("EMA_START_EPOCH", 1)))
    early_stop_patience = int(max(0, solver.get("EARLY_STOPPING_PATIENCE", 8)))
    early_stop_min_delta = float(solver.get("EARLY_STOPPING_MIN_DELTA", 0.0))
    save_topk = int(max(1, solver.get("SAVE_TOPK_CHECKPOINTS", 5)))
    train_sampler_mode = str(solver.get("TRAIN_SAMPLER", "shuffle"))
    pos_weight_mode = str(solver.get("POS_WEIGHT_MODE", "none"))
    pos_weight_clip_max = float(solver.get("POS_WEIGHT_CLIP_MAX", 4.0))
    pos_weight_power = float(solver.get("POS_WEIGHT_POWER", 1.0))

    use_multiview = bool(int(args.use_multiview if args.use_multiview is not None else model_cfg.get("USE_MULTIVIEW", 0)))
    use_pocket_model = bool(int(model_cfg.get("USE_POCKET", 1)))

    save_dir = _abs(args.save_dir or result.get("SAVE_DIR", "runs/real_pocket_cached_rebuild"))
    run_name = str(args.run_name or result.get("RUN_NAME", "human_real_pocket_cached_seqdist_3view_soft_pocket_bias_emaes"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"{run_name}_{ts}")
    ensure_dir(run_dir)
    soft_bias_log_every = int(max(1, result.get("SOFT_BIAS_LOG_EVERY", model_cfg.get("SOFT_BIAS_LOG_EVERY", 1))))
    soft_bias_debug_csv = os.path.join(run_dir, "soft_pocket_bias_debug.csv")
    with open(soft_bias_debug_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "epoch",
            "step",
            "pocket_bias.mean",
            "pocket_bias.max",
            "pocket_bias.nonzero_ratio",
            "softplus(bias_log_scale)",
            "sigmoid(pp_gate)",
            "pp_attn.gate",
        ])

    if use_pocket_model:
        p2rank_work_dir_ds = p2rank_work_dir if p2rank_work_dir and os.path.isdir(p2rank_work_dir) else None
        p2rank_csv_ds = p2rank_csv
        pdb_index_csv_ds = pdb_index_csv
        p2rank_out_dir_ds = p2rank_out_dir
        pocket_cache_dir_ds = pocket_cache_dir
        drop_no_pocket_ds = bool(int(data.get("DROP_NO_POCKET", 0)))
    else:
        p2rank_work_dir_ds = None
        p2rank_csv_ds = None
        pdb_index_csv_ds = None
        p2rank_out_dir_ds = None
        pocket_cache_dir_ds = None
        drop_no_pocket_ds = False
        print("[INFO] USE_POCKET=0 -> no-pocket data flow enabled (skip pocket cache/p2rank; DROP_NO_POCKET disabled).")

    ds_common = dict(
        csv_path=csv_path,
        split_path=split_path,
        pharm_path=pharm_path,
        esm2_path=esm2_path,
        mask_special_tokens=bool(int(data.get("MASK_SPECIAL_TOKENS", 1))),
        p2rank_work_dir=p2rank_work_dir_ds,
        protein_pockets_csv=p2rank_csv_ds,
        pdb_index_csv=pdb_index_csv_ds,
        p2rank_out_dir=p2rank_out_dir_ds,
        protein_id_col=str(data.get("PROTEIN_ID_COL", "Protein_ID")),
        pocket_cache_dir=pocket_cache_dir_ds,
        pocket_topk=int(data.get("POCKET_TOPK", 3)),
        pocket_max_nodes=int(data.get("POCKET_MAX_NODES", 96)),
        pocket_knn_k=int(data.get("POCKET_KNN_K", 16)),
        pocket_expand_min_nodes=int(data.get("POCKET_EXPAND_MIN_NODES", 0)),
        pocket_seqdist_norm=str(data.get("POCKET_SEQDIST_NORM", "log1p")),
        pocket_seqdist_norm_denom=float(data.get("POCKET_SEQDIST_NORM_DENOM", 128.0)),
        pocket_prob_thres=float(data.get("POCKET_PROB_THRES", 0.0)),
        pocket_min_residues=int(data.get("POCKET_MIN_RESIDUES", 5)),
        drop_no_pocket=drop_no_pocket_ds,
        pocket_use_struct_feat=bool(int(data.get("POCKET_USE_STRUCT_FEAT", data.get("POCKET_STRUCT_FEAT", 1)))),
        pocket_struct_dim=int(data.get("POCKET_STRUCT_DIM", 29)),
        pocket_edge_rbf_bins=int(data.get("POCKET_EDGE_RBF_BINS", 16)),
        pocket_edge_use_seqdist=bool(int(data.get("POCKET_EDGE_USE_SEQDIST", 1))),
        pocket_raw_only=bool(int(data.get("POCKET_RAW_ONLY", 1))),
        pocket_edge_mode=str(data.get("POCKET_EDGE_MODE", "radius")),
        pocket_radius_cutoff=float(data.get("POCKET_RADIUS_CUTOFF", 10.0)),
        pocket_radius_max_neighbors=int(data.get("POCKET_RADIUS_MAX_NEIGHBORS", 32)),
        drug_graph_cache_path=drug_graph_cache_path,
    )

    train_ds = HumanDTIDatasetCanonRealPocket(split="train", **ds_common)
    val_ds = HumanDTIDatasetCanonRealPocket(split="val", **ds_common)
    test_ds = HumanDTIDatasetCanonRealPocket(split="test", **ds_common)
    collate_fn = partial(collate_fn_real_pocket, include_pocket=use_pocket_model)

    train_labels = _dataset_label_array(train_ds)
    train_pos_ratio = float(train_labels.mean()) if train_labels.size else float("nan")
    train_sampler = build_train_sampler(train_labels, train_sampler_mode)
    train_pos_weight = compute_pos_weight(
        train_labels,
        mode=pos_weight_mode,
        clip_max=pos_weight_clip_max,
        power=pos_weight_power,
        device=device,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    num_pharm_types = infer_num_pharm_types(pharm_path)
    node_in = int(train_ds[0][0]["node_feat"].shape[1]) if len(train_ds) > 0 else int(model_cfg.get("NODE_IN", 74))
    esm_dim = int(train_ds.esm_dim)
    model = build_model(cfg, num_pharm_types=num_pharm_types, esm_dim=esm_dim, node_in=node_in, use_multiview=use_multiview).to(device)

    if args.eval_only_ckpt:
        ckpt = torch.load(_abs(args.eval_only_ckpt), map_location=device)
        model.load_state_dict(ckpt["model"], strict=True)
        best_thr = float(ckpt.get("best_threshold", 0.5))
        yt, lt, test_loss = gather_eval(model, test_loader, device, amp=False, label_smoothing=0.0)
        test_m = compute_metrics(yt, lt, threshold=best_thr) if yt.size else {}
        test_m["loss"] = float(test_loss)
        payload = {
            "best_epoch": int(ckpt.get("epoch", -1)),
            "best_metric": ckpt.get("best_metric", main_metric),
            "best_val": float(ckpt.get("best_val", float("nan"))),
            "best_threshold": best_thr,
            "test@thr": test_m,
            "ckpt": _abs(args.eval_only_ckpt),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = None
    if str(solver.get("SCHEDULER", "")).lower() == "cosine":
        min_lr = float(solver.get("MIN_LR", 1e-5))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=min_lr)
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda" if device.startswith("cuda") else "cpu", enabled=amp)
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=amp)
    ema = EMAHelper(model, decay=ema_decay, device="cpu") if use_ema else None

    epoch_csv = os.path.join(run_dir, "epoch_metrics.csv")
    with open(epoch_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "epoch",
            "train_loss",
            "train_bce",
            "train_pairwise",
            "pairwise_weight_now",
            "val_loss",
            "val_auc",
            "val_auprc",
            "val_f1@best",
            "val_bacc@best",
            "val_mcc@best",
            "best_threshold",
            "val_pos_logit_mean",
            "val_pos_logit_std",
            "val_neg_logit_mean",
            "val_neg_logit_std",
            "val_logit_gap",
            "lr",
        ])

    best_epoch = -1
    best_val = -1.0
    best_thr = 0.5
    best_ckpt = os.path.join(run_dir, "best_auc.pt")
    metric_name = main_metric if main_metric in {"auc", "auprc", "f1", "bacc", "mcc", "acc"} else "auc"
    no_improve = 0
    topk_entries: List[Dict[str, Any]] = []
    topk_manifest = os.path.join(run_dir, "topk_checkpoints.json")

    for ep in range(1, epochs + 1):
        pairwise_weight_now = ramp_weight(pairwise_weight, ep, pairwise_warmup_epochs)
        ema_active = ema if (ema is not None and ep >= ema_start_epoch) else None
        tr = run_train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            amp,
            grad_accum,
            ep,
            ema=ema_active,
            label_smoothing=label_smoothing,
            pairwise_weight=pairwise_weight_now,
            pos_weight=train_pos_weight,
            debug_csv_path=soft_bias_debug_csv,
            debug_every=soft_bias_log_every,
        )

        if ema_active is not None:
            ema_active.store(model)
            ema_active.copy_to(model)
        yv, lv, val_loss = gather_eval(model, val_loader, device, amp=False, label_smoothing=0.0)
        val_logit_stats = summarize_logit_distribution(yv, lv) if yv.size else {
            "pos_logit_mean": float("nan"),
            "pos_logit_std": float("nan"),
            "neg_logit_mean": float("nan"),
            "neg_logit_std": float("nan"),
            "logit_gap": float("nan"),
        }

        thr = select_threshold(yv, lv, metric=args.threshold_metric) if yv.size else 0.5
        val_m = compute_metrics(yv, lv, threshold=thr) if yv.size else {}
        val_m["loss"] = float(val_loss)
        score = float(val_m.get(metric_name, float("nan")))
        lr_now = float(optimizer.param_groups[0]["lr"])

        with open(epoch_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                f"{tr['train_loss']:.6f}",
                f"{tr['train_bce']:.6f}",
                f"{tr['train_pairwise']:.6f}",
                f"{pairwise_weight_now:.6f}",
                f"{val_loss:.6f}",
                f"{val_m.get('auc', float('nan')):.6f}",
                f"{val_m.get('auprc', float('nan')):.6f}",
                f"{val_m.get('f1', float('nan')):.6f}",
                f"{val_m.get('bacc', float('nan')):.6f}",
                f"{val_m.get('mcc', float('nan')):.6f}",
                f"{thr:.2f}",
                f"{val_logit_stats.get('pos_logit_mean', float('nan')):.6f}",
                f"{val_logit_stats.get('pos_logit_std', float('nan')):.6f}",
                f"{val_logit_stats.get('neg_logit_mean', float('nan')):.6f}",
                f"{val_logit_stats.get('neg_logit_std', float('nan')):.6f}",
                f"{val_logit_stats.get('logit_gap', float('nan')):.6f}",
                f"{lr_now:.8f}",
            ])

        improved = np.isfinite(score) and (score > (best_val + early_stop_min_delta))
        if np.isfinite(score) and (len(topk_entries) < save_topk or score > min(x['score'] for x in topk_entries)):
            ckpt_path = os.path.join(run_dir, f"topk_epoch{ep:03d}_{metric_name}{score:.6f}.pt")
            _save_ckpt(ckpt_path, model, ep, metric_name, score, thr, seed, cfg, args, extra={"ema_used": bool(use_ema)})
            topk_entries.append({"epoch": ep, "score": score, "path": ckpt_path})
            topk_entries = sorted(topk_entries, key=lambda x: x["score"], reverse=True)[:save_topk]
            keep = {x['path'] for x in topk_entries}
            for p in list(Path for Path in []):
                pass
            for fn in os.listdir(run_dir):
                if fn.startswith("topk_epoch") and fn.endswith(".pt"):
                    fp = os.path.join(run_dir, fn)
                    if fp not in keep:
                        try:
                            os.remove(fp)
                        except OSError:
                            pass
            with open(topk_manifest, "w", encoding="utf-8") as f:
                json.dump(topk_entries, f, ensure_ascii=False, indent=2)

        print(
            f"epoch={ep:03d} train_loss={tr['train_loss']:.4f} train_pair={tr['train_pairwise']:.4f} "
            f"pair_w={pairwise_weight_now:.3f} "
            f"val_auc={val_m.get('auc', float('nan')):.6f} val_auprc={val_m.get('auprc', float('nan')):.6f} "
            f"logit_gap={val_logit_stats.get('logit_gap', float('nan')):.4f} "
            f"best={best_val:.6f}"
        )

        if improved:
            best_val = score
            best_epoch = ep
            best_thr = float(thr)
            _save_ckpt(best_ckpt, model, ep, metric_name, best_val, best_thr, seed, cfg, args, extra={"ema_used": bool(use_ema)})
            no_improve = 0
        else:
            no_improve += 1

        if ema_active is not None:
            ema_active.restore(model)

        if scheduler is not None:
            scheduler.step()

        if early_stop_patience > 0 and no_improve >= early_stop_patience:
            print(f"[EARLY_STOP] no improvement for {no_improve} epochs (patience={early_stop_patience}).")
            break

    if not os.path.isfile(best_ckpt):
        raise RuntimeError("No valid checkpoint was saved. Please check data and metrics.")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    best_thr = float(ckpt.get("best_threshold", 0.5))

    yt, lt, test_loss = gather_eval(model, test_loader, device, amp=False, label_smoothing=0.0)
    test_m = compute_metrics(yt, lt, threshold=best_thr) if yt.size else {}
    test_m["loss"] = float(test_loss)

    payload = {
        "run_dir": run_dir,
        "best_epoch": int(ckpt.get("epoch", best_epoch)),
        "best_metric": metric_name,
        "best_val": float(ckpt.get("best_val", best_val)),
        "best_threshold": best_thr,
        "test@thr": test_m,
        "seed": seed,
        "split_path": split_path,
        "use_multiview": int(use_multiview),
        "ema": int(use_ema),
        "ema_decay": float(ema_decay),
        "ema_start_epoch": int(ema_start_epoch),
        "label_smoothing": float(label_smoothing),
        "pairwise_auc_loss_weight": float(pairwise_weight),
        "pairwise_warmup_epochs": int(pairwise_warmup_epochs),
        "train_sampler": str(train_sampler_mode),
        "train_pos_ratio": float(train_pos_ratio),
        "pos_weight_mode": str(pos_weight_mode),
        "pos_weight_value": float(train_pos_weight.item()) if train_pos_weight is not None else None,
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"[OK] run_dir={run_dir}\n")
        f.write(f"best_epoch={payload['best_epoch']} best_{metric_name}={payload['best_val']:.6f}\n")
        f.write(f"best_threshold(val-{args.threshold_metric})={best_thr:.2f} test@thr={test_m}\n")

    print(f"[OK] run_dir={run_dir}")
    print(f"best_epoch={payload['best_epoch']} best_{metric_name}={payload['best_val']:.6f}")
    print(f"best_threshold(val-{args.threshold_metric})={best_thr:.2f} test@thr={test_m}")


if __name__ == "__main__":
    main()
