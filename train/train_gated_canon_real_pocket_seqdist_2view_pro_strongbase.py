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
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(_THIS))
_THIS_DIR = os.path.dirname(_THIS)
for _p in (_THIS_DIR, _ROOT):
    if _p and _p not in sys.path:
        sys.path.insert(0, _p)

from dataset.dataset_canon_real_pocket_seqdist import (
    HumanDTIDatasetCanonRealPocket,
    collate_fn_real_pocket,
)
from dataset.metrics import compute_metrics
from model.model_gated_real_pocket_seqdist_2view_pro_strongbase import BaselineGatedDTIRealPocket


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
    y_true = np.asarray(y_true).reshape(-1)
    logits = np.asarray(logits).reshape(-1)
    pos = logits[y_true > 0.5]
    neg = logits[y_true <= 0.5]

    def _mean(x: np.ndarray) -> float:
        return float(np.mean(x)) if x.size else float("nan")

    def _std(x: np.ndarray) -> float:
        return float(np.std(x)) if x.size else float("nan")

    out = {
        "pos_logit_mean": _mean(pos),
        "pos_logit_std": _std(pos),
        "neg_logit_mean": _mean(neg),
        "neg_logit_std": _std(neg),
    }
    if np.isfinite(out["pos_logit_mean"]) and np.isfinite(out["neg_logit_mean"]):
        out["logit_gap"] = float(out["pos_logit_mean"] - out["neg_logit_mean"])
    else:
        out["logit_gap"] = float("nan")
    return out


def bce_with_optional_smoothing(logit: torch.Tensor, y: torch.Tensor, smoothing: float = 0.0) -> torch.Tensor:
    if smoothing > 0.0:
        y = y * (1.0 - smoothing) + 0.5 * smoothing
    return torch.nn.functional.binary_cross_entropy_with_logits(logit, y)


def get_model_aux_losses(model) -> Dict[str, torch.Tensor]:
    m = model.module if hasattr(model, "module") else model
    if hasattr(m, "get_aux_losses"):
        d = m.get_aux_losses()
    else:
        d = getattr(m, "_aux_losses", None)
    return d if isinstance(d, dict) else {}


def pairwise_auc_loss(logit: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    pos = logit[y > 0.5].view(-1)
    neg = logit[y <= 0.5].view(-1)
    if pos.numel() == 0 or neg.numel() == 0:
        return logit.new_zeros(())
    margin = neg.unsqueeze(0) - pos.unsqueeze(1)
    return torch.nn.functional.softplus(margin).mean()


def gather_eval(model, loader, device: str, amp: bool) -> Tuple[np.ndarray, np.ndarray, float]:
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
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, y)
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

    for step, (batch, y) in enumerate(it, start=1):
        batch = move_batch_to_device(batch, device)
        y = y.to(device)
        with amp_autocast_ctx(device, amp):
            logit = model(batch)
            bce = bce_with_optional_smoothing(logit, y, smoothing=label_smoothing)
            pair = pairwise_auc_loss(logit, y)
            loss_raw = bce + float(pairwise_weight) * pair
            loss = loss_raw / grad_accum

        if not torch.isfinite(loss):
            continue

        scaler.scale(loss).backward()
        has_scaled_grads = True

        if step % grad_accum == 0 and has_scaled_grads:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(optimizer)
            scaler.update()
            if ema is not None:
                ema.update(model)
            optimizer.zero_grad(set_to_none=True)
            has_scaled_grads = False

        bs = int(y.shape[0])
        total_loss_sum += float(loss_raw.detach().item()) * bs
        bce_sum += float(bce.detach().item()) * bs
        pair_sum += float(pair.detach().item()) * bs
        n += bs
        if tqdm is not None:
            it.set_postfix(loss=f"{total_loss_sum / max(n, 1):.4f}", bce=f"{bce_sum / max(n, 1):.4f}", pair=f"{pair_sum / max(n, 1):.4f}")

    if step > 0 and (step % grad_accum) != 0 and has_scaled_grads:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        scaler.step(optimizer)
        scaler.update()
        if ema is not None:
            ema.update(model)
        optimizer.zero_grad(set_to_none=True)

    denom = max(n, 1)
    return {
        "loss": total_loss_sum / denom,
        "bce": bce_sum / denom,
        "pairwise": pair_sum / denom,
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
        prot_dim=mcfg.get("PROT_DIM", None),
        use_pocket=bool(int(mcfg.get("USE_POCKET", 1))),
        pocket_dim=int(mcfg.get("POCKET_DIM", 64)),
        fusion_dim=mcfg.get("FUSION_DIM", None),
        gate_hidden=mcfg.get("GATE_HIDDEN", None),
    )

    if use_multiview:
        print("[WARN] USE_MULTIVIEW=1 is ignored in 2view_pro training script; forcing 2view_pro model.")

    return BaselineGatedDTIRealPocket(
        **common,
        use_pp_attn=bool(int(mcfg.get("USE_PP_ATTN", 1))),
        pp_attn_dropout=float(mcfg.get("PP_ATTN_DROPOUT", mcfg.get("DROPOUT", 0.1))),
        pocket_node_proj=mcfg.get("POCKET_NODE_PROJ", mcfg.get("POCKET_DIM", 128)),
        pocket_gnn_hidden=int(mcfg.get("POCKET_GNN_HIDDEN", mcfg.get("POCKET_DIM", 128))),
        pocket_gnn_layers=int(mcfg.get("POCKET_GNN_LAYERS", 3)),
        pocket_struct_dim=int(mcfg.get("POCKET_STRUCT_DIM", cfg.get("DATA", {}).get("POCKET_STRUCT_DIM", 29))),
        pocket_use_struct_feat=bool(int(mcfg.get("POCKET_USE_STRUCT_FEAT", cfg.get("DATA", {}).get("POCKET_USE_STRUCT_FEAT", cfg.get("DATA", {}).get("POCKET_STRUCT_FEAT", 1))))),
        pocket_edge_feat_dim=int(mcfg.get("POCKET_EDGE_FEAT_DIM", 1 + int(cfg.get("DATA", {}).get("POCKET_EDGE_RBF_BINS", 16)) + 3 + (1 if bool(int(cfg.get("DATA", {}).get("POCKET_EDGE_USE_SEQDIST", 1))) else 0))),
        pocket_edge_hidden=int(mcfg.get("POCKET_EDGE_HIDDEN", 32)),
        pocket_inst_pool=str(mcfg.get("POCKET_INST_POOL", "softmax")),
        pocket_tau=float(mcfg.get("POCKET_TAU", 1.0)),
        pocket_score_prior=bool(int(mcfg.get("POCKET_SCORE_PRIOR", 1))),
        pocket_score_scale=float(mcfg.get("POCKET_SCORE_SCALE", 1.0)),
        pair_gate_init_bias=float(mcfg.get("PAIR_GATE_INIT_BIAS", -2.0)),
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
    return p.parse_args()


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
    label_smoothing = float(max(0.0, solver.get("LABEL_SMOOTHING", 0.02)))
    pairwise_weight = float(max(0.0, solver.get("PAIRWISE_AUC_LOSS_WEIGHT", 0.20)))
    use_ema = bool(int(solver.get("EMA", 1)))
    ema_decay = float(solver.get("EMA_DECAY", 0.999))
    early_stop_patience = int(max(0, solver.get("EARLY_STOPPING_PATIENCE", 8)))
    early_stop_min_delta = float(solver.get("EARLY_STOPPING_MIN_DELTA", 0.0))

    use_multiview = bool(int(args.use_multiview if args.use_multiview is not None else model_cfg.get("USE_MULTIVIEW", 0)))
    use_pocket_model = bool(int(model_cfg.get("USE_POCKET", 1)))

    save_dir = _abs(args.save_dir or result.get("SAVE_DIR", "runs/real_pocket_cached_rebuild"))
    run_name = str(args.run_name or result.get("RUN_NAME", "human_real_pocket_cached_seqdist_rebuild"))
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_dir, f"{run_name}_{ts}")
    ensure_dir(run_dir)

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
        pocket_edge_rbf_dmin=float(data.get("POCKET_EDGE_RBF_DMIN", 0.0)),
        pocket_edge_rbf_dmax=float(data.get("POCKET_EDGE_RBF_DMAX", 20.0)),
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

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn)

    num_pharm_types = infer_num_pharm_types(pharm_path)
    node_in = int(train_ds[0][0]["node_feat"].shape[1]) if len(train_ds) > 0 else int(model_cfg.get("NODE_IN", 74))
    esm_dim = int(train_ds.esm_dim)

    model = build_model(cfg, num_pharm_types=num_pharm_types, esm_dim=esm_dim, node_in=node_in, use_multiview=use_multiview).to(device)
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
            "epoch", "train_loss", "train_bce", "train_pairwise", "val_loss", "val_auc", "val_auprc",
            "val_f1@best", "val_bacc@best", "val_mcc@best", "best_threshold",
            "val_pos_logit_mean", "val_pos_logit_std", "val_neg_logit_mean", "val_neg_logit_std", "val_logit_gap",
            "lr"
        ])

    best_epoch = -1
    best_val = -1.0
    best_thr = 0.5
    best_ckpt = os.path.join(run_dir, "best_auc.pt")
    metric_name = main_metric if main_metric in {"auc", "auprc", "f1", "bacc", "mcc", "acc"} else "auc"
    bad_epochs = 0

    for ep in range(1, epochs + 1):
        tr = run_train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp=amp,
            grad_accum=grad_accum,
            epoch=ep,
            ema=ema,
            label_smoothing=label_smoothing,
            pairwise_weight=pairwise_weight,
        )

        if ema is not None:
            ema.store(model)
            ema.copy_to(model)
        yv, lv, val_loss = gather_eval(model, val_loader, device, amp=False)
        if ema is not None:
            ema.restore(model)

        thr = select_threshold(yv, lv, metric=args.threshold_metric) if yv.size else 0.5
        val_m = compute_metrics(yv, lv, threshold=thr) if yv.size else {}
        val_logit_stats = summarize_logit_distribution(yv, lv) if yv.size else {
            "pos_logit_mean": float("nan"),
            "pos_logit_std": float("nan"),
            "neg_logit_mean": float("nan"),
            "neg_logit_std": float("nan"),
            "logit_gap": float("nan"),
        }
        val_m["loss"] = float(val_loss)
        score = float(val_m.get(metric_name, float("nan")))

        lr_now = float(optimizer.param_groups[0]["lr"])
        with open(epoch_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                ep,
                f"{tr['loss']:.6f}",
                f"{tr['bce']:.6f}",
                f"{tr['pairwise']:.6f}",
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

        print(
            f"epoch={ep:03d} train_loss={tr['loss']:.4f} (bce={tr['bce']:.4f}, pair={tr['pairwise']:.4f}) "
            f"val_auc={val_m.get('auc', float('nan')):.6f} "
            f"val_auprc={val_m.get('auprc', float('nan')):.6f} "
            f"thr={thr:.2f} "
            f"logit_gap={val_logit_stats.get('logit_gap', float('nan')):.4f} "
            f"pos_mu={val_logit_stats.get('pos_logit_mean', float('nan')):.4f} "
            f"neg_mu={val_logit_stats.get('neg_logit_mean', float('nan')):.4f} "
            f"best={best_val:.6f}"
        )

        improved = np.isfinite(score) and (score > (best_val + early_stop_min_delta))
        if improved:
            best_val = score
            best_epoch = ep
            best_thr = float(thr)
            bad_epochs = 0

            if ema is not None:
                ema.store(model)
                ema.copy_to(model)
            save_state = deepcopy(model.state_dict())
            if ema is not None:
                ema.restore(model)

            torch.save(
                {
                    "model": save_state,
                    "epoch": ep,
                    "best_metric": metric_name,
                    "best_val": best_val,
                    "best_threshold": best_thr,
                    "seed": seed,
                    "cfg": cfg,
                    "args": vars(args),
                    "ema": int(use_ema),
                    "ema_decay": float(ema_decay),
                    "label_smoothing": float(label_smoothing),
                    "pairwise_auc_loss_weight": float(pairwise_weight),
                },
                best_ckpt,
            )
        else:
            bad_epochs += 1

        if scheduler is not None:
            scheduler.step()

        if early_stop_patience > 0 and bad_epochs >= early_stop_patience:
            print(f"[INFO] early stopping triggered at epoch {ep} (best epoch {best_epoch}, patience {early_stop_patience}).")
            break

    if not os.path.isfile(best_ckpt):
        raise RuntimeError("No valid checkpoint was saved. Please check data and metrics.")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    best_thr = float(ckpt.get("best_threshold", 0.5))

    yt, lt, test_loss = gather_eval(model, test_loader, device, amp=False)
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
        "label_smoothing": float(label_smoothing),
        "pairwise_auc_loss_weight": float(pairwise_weight),
    }
    with open(os.path.join(run_dir, "test_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(run_dir, "results.txt"), "w", encoding="utf-8") as f:
        f.write(f"[OK] run_dir={run_dir}\n")
        f.write(f"best_epoch={payload['best_epoch']} best_{metric_name}={payload['best_val']:.6f}\n")
        f.write(f"best_threshold(val-{args.threshold_metric})={best_thr:.2f} test@thr={test_m}\n")
        f.write(
            f"ema={int(use_ema)} ema_decay={ema_decay} label_smoothing={label_smoothing} "
            f"pairwise_auc_loss_weight={pairwise_weight}\n"
        )

    print(f"[OK] run_dir={run_dir}")
    print(f"best_epoch={payload['best_epoch']} best_{metric_name}={payload['best_val']:.6f}")
    print(f"best_threshold(val-{args.threshold_metric})={best_thr:.2f} test@thr={test_m}")


if __name__ == "__main__":
    main()
