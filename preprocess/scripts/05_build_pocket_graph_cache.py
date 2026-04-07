# -*- coding: utf-8 -*-
"""
Build pocket graph cache per protein.

这版和新的 pocket_graph_seqdist.py 对齐，支持：
- topk/top1 pocket
- raw_only
- radius / knn 两种建图
- richer node / edge features
- 将 cache 元信息写完整，方便 dataset 侧校验

用法示例（top1 + 更像 PocketDTA 的轻量版本）：

python 05_build_pocket_graph_cache.py ^
  --in_csv data/interim/human/pairs_with_pid/full_with_pid.csv ^
  --pid_col Protein_ID ^
  --seq_col Protein ^
  --esm2_pt data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt ^
  --protein_pockets_csv data/raw/human/p2rank_raw/protein_pockets.csv ^
  --pdb_index_csv data/raw/human/p2rank_raw/pdb_index.csv ^
  --out_dir data/processed/human/pocket_graph_cache_top1 ^
  --topk 1 ^
  --max_nodes 96 ^
  --knn_k 16 ^
  --expand_min_nodes 0 ^
  --prob_thres 0.0 ^
  --min_residues 8 ^
  --seqdist_norm log1p ^
  --seqdist_norm_denom 128.0 ^
  --raw_only 1 ^
  --edge_mode radius ^
  --radius_cutoff 10.0 ^
  --radius_max_neighbors 32 ^
  --edge_rbf_bins 16 ^
  --edge_use_seqdist 1
"""

from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import torch

# Ensure project root is on sys.path when running as a script.
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ---- import p2rank index helper ----
try:
    from p2rank_pockets_strict import build_index_from_protein_pockets_csv
except (ModuleNotFoundError, ImportError):
    _p2_path = Path(_ROOT) / "p2rank_pockets_strict.py"
    if not _p2_path.is_file():
        raise
    _spec = importlib.util.spec_from_file_location("p2rank_pockets_strict", str(_p2_path))
    if _spec is None or _spec.loader is None:
        raise
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_index_from_protein_pockets_csv = _mod.build_index_from_protein_pockets_csv

# ---- import pocket graph utilities ----
try:
    from pocket_graph_seqdist import (
        load_p2rank_pockets,
        parse_pdb_ca_records,
        parse_pdb_ca_coords,
        build_pocket_templates,
        POCKET_NODE_STRUCT_DIM,
        POCKET_RBF_BINS_DEFAULT,
        POCKET_RADIUS_CUTOFF_DEFAULT,
        POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT,
    )
except (ModuleNotFoundError, ImportError):
    _pg_path = Path(_ROOT) / "pocket_graph_seqdist.py"
    if not _pg_path.is_file():
        raise
    _spec2 = importlib.util.spec_from_file_location("pocket_graph_seqdist", str(_pg_path))
    if _spec2 is None or _spec2.loader is None:
        raise
    _mod2 = importlib.util.module_from_spec(_spec2)
    _spec2.loader.exec_module(_mod2)
    load_p2rank_pockets = _mod2.load_p2rank_pockets
    parse_pdb_ca_records = _mod2.parse_pdb_ca_records
    parse_pdb_ca_coords = _mod2.parse_pdb_ca_coords
    build_pocket_templates = _mod2.build_pocket_templates
    POCKET_NODE_STRUCT_DIM = int(getattr(_mod2, "POCKET_NODE_STRUCT_DIM", 29))
    POCKET_RBF_BINS_DEFAULT = int(getattr(_mod2, "POCKET_RBF_BINS_DEFAULT", 16))
    POCKET_RADIUS_CUTOFF_DEFAULT = float(getattr(_mod2, "POCKET_RADIUS_CUTOFF_DEFAULT", 10.0))
    POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT = int(getattr(_mod2, "POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT", 32))


def _safe_torch_load(path: str) -> Any:
    """
    Safe load for PyTorch >= 2 where weights_only may exist.
    """
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def build_pdb_index_from_csv(pdb_index_csv: str) -> Dict[str, str]:
    """
    Build {Protein_ID -> pdb_path} mapping.
    Supports either absolute or relative pdb_path.
    """
    out: Dict[str, str] = {}
    p = Path(pdb_index_csv)
    if not p.is_file():
        return out

    df = pd.read_csv(p)
    if len(df) == 0:
        return out

    pid_col = None
    for c in ("Protein_ID", "protein_id", "pid"):
        if c in df.columns:
            pid_col = c
            break
    if pid_col is None:
        return out

    path_col = None
    for c in ("pdb_path", "pdb", "structure_path", "path"):
        if c in df.columns:
            path_col = c
            break
    if path_col is None:
        return out

    base = p.resolve().parent
    for _, row in df.iterrows():
        pid = str(row.get(pid_col, "")).strip()
        if not pid:
            continue
        pid = pid.split("_len=")[0]

        raw = str(row.get(path_col, "")).strip()
        if not raw:
            continue

        fp = Path(raw)
        if fp.is_absolute():
            out[pid] = str(fp)
        else:
            out[pid] = str((base / fp).resolve())

    return out


def main():
    ap = argparse.ArgumentParser(description="Build pocket graph cache per protein (Top-K pockets).")

    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--seq_col", default="Protein")

    ap.add_argument("--esm2_pt", default="data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt")
    ap.add_argument("--protein_pockets_csv", default="data/raw/human/p2rank_raw/protein_pockets.csv")
    ap.add_argument("--pdb_index_csv", default="data/raw/human/p2rank_raw/pdb_index.csv")

    ap.add_argument("--out_dir", default="data/processed/human/pocket_graph_cache")

    # pocket selection
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--prob_thres", type=float, default=0.0)
    ap.add_argument("--min_residues", type=int, default=8)

    # graph construction
    ap.add_argument("--max_nodes", type=int, default=96)
    ap.add_argument("--knn_k", type=int, default=16)
    ap.add_argument("--expand_min_nodes", type=int, default=0)
    ap.add_argument("--seqdist_norm", default="log1p")
    ap.add_argument("--seqdist_norm_denom", type=float, default=128.0)

    # richer / pocketdta-like options
    ap.add_argument("--raw_only", type=int, default=1, help="1: only keep raw pocket residues")
    ap.add_argument("--edge_mode", default="radius", choices=["radius", "knn"])
    ap.add_argument("--radius_cutoff", type=float, default=POCKET_RADIUS_CUTOFF_DEFAULT)
    ap.add_argument("--radius_max_neighbors", type=int, default=POCKET_RADIUS_MAX_NEIGHBORS_DEFAULT)
    ap.add_argument("--edge_rbf_bins", type=int, default=POCKET_RBF_BINS_DEFAULT)
    ap.add_argument("--edge_use_seqdist", type=int, default=1)
    ap.add_argument("--use_struct_feat", type=int, default=1)

    ap.add_argument("--limit", type=int, default=0, help="Debug only: limit number of proteins")

    args = ap.parse_args()

    # ---- read source dataframe to get pids / seq len ----
    df = pd.read_csv(args.in_csv)
    if args.pid_col not in df.columns:
        raise ValueError(f"in_csv missing pid_col={args.pid_col}")
    if args.seq_col not in df.columns:
        raise ValueError(f"in_csv missing seq_col={args.seq_col}")

    pid2seqlen: Dict[str, int] = {}
    for _, row in df[[args.pid_col, args.seq_col]].drop_duplicates().iterrows():
        pid = str(row[args.pid_col]).strip().split("_len=")[0]
        seq = str(row[args.seq_col]).strip()
        if pid:
            pid2seqlen[pid] = len(seq)

    # ---- load ESM residue-level embeddings ----
    esm_obj = _safe_torch_load(str(args.esm2_pt))
    if not isinstance(esm_obj, dict):
        raise TypeError(f"ESM2 pt must be a dict, got {type(esm_obj)}")

    tokens = esm_obj.get("tokens", None)
    if tokens is None:
        tokens = esm_obj.get("embeddings", None)
    if tokens is None or not isinstance(tokens, dict):
        raise KeyError("ESM2 pt missing dict 'tokens' (or 'embeddings')")

    # ---- indices for P2Rank and PDB ----
    p2idx = build_index_from_protein_pockets_csv(
        str(args.protein_pockets_csv),
        predictions_col="predictions_csv",
    )
    pdb_idx = build_pdb_index_from_csv(str(args.pdb_index_csv))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pids = sorted(set(p2idx.keys()) & set(pdb_idx.keys()) & set(tokens.keys()))
    if args.limit and int(args.limit) > 0:
        pids = pids[: int(args.limit)]

    n_ok = 0
    n_failed = 0

    for pid in pids:
        res_csv, pred_csv = p2idx.get(pid, ("", ""))
        pdb_path = pdb_idx.get(pid, "")

        if (not res_csv) or (not pred_csv) or (not pdb_path):
            n_failed += 1
            if n_failed <= 5:
                print(f"[WARN] missing inputs pid={pid} res_csv={bool(res_csv)} pred_csv={bool(pred_csv)} pdb={bool(pdb_path)}")
            continue

        try:
            pockets = load_p2rank_pockets(
                pred_csv,
                res_csv,
                top_k=int(args.topk),
                prob_threshold=float(args.prob_thres),
                min_residues=int(args.min_residues),
            )

            pdb_rec = parse_pdb_ca_records(pdb_path)
            if pdb_rec:
                coord_map: Dict[Any, Any] = {}
                resname_map: Dict[Any, str] = {}
                plddt_map: Dict[Any, float] = {}
                for key, rec in pdb_rec.items():
                    xyz = rec.get("coord", None)
                    if xyz is not None:
                        coord_map[key] = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                    resname_map[key] = str(rec.get("resname", "UNK") or "UNK").upper()
                    try:
                        plddt_map[key] = float(rec.get("plddt", 0.0) or 0.0)
                    except Exception:
                        plddt_map[key] = 0.0
            else:
                coord_map = parse_pdb_ca_coords(pdb_path)
                resname_map = {}
                plddt_map = {}

            tok = tokens.get(pid, None)
            if not isinstance(tok, torch.Tensor) or tok.dim() != 2:
                raise TypeError(f"bad token tensor for pid={pid}: {type(tok)}")

            # Use residue embedding length as effective sequence length
            seq_len_eff = int(tok.shape[0])

            templates = build_pocket_templates(
                pockets,
                coord_map,
                seq_len_eff,
                top_k=int(args.topk),
                max_nodes=int(args.max_nodes),
                knn_k=int(args.knn_k),
                expand_min_nodes=int(args.expand_min_nodes),
                seqdist_norm=str(args.seqdist_norm),
                seqdist_norm_denom=float(args.seqdist_norm_denom),
                resname_map=resname_map,
                plddt_map=plddt_map,
                edge_rbf_bins=int(args.edge_rbf_bins),
                raw_only=bool(int(args.raw_only)),
                edge_mode=str(args.edge_mode),
                radius_cutoff=float(args.radius_cutoff),
                radius_max_neighbors=int(args.radius_max_neighbors),
            )

            edge_feat_dim = 1 + int(args.edge_rbf_bins) + 3 + (1 if int(args.edge_use_seqdist) else 0)

            out: Dict[str, Any] = {
                "pid": str(pid),
                "templates": templates,
                "seq_len": int(seq_len_eff),
                "seq_len_full": int(pid2seqlen.get(str(pid), seq_len_eff)),
                "pocket_topk": int(args.topk),
                "pocket_max_nodes": int(args.max_nodes),
                "pocket_knn_k": int(args.knn_k),
                "pocket_prob_thres": float(args.prob_thres),
                "pocket_min_residues": int(args.min_residues),
                "pocket_expand_min_nodes": int(args.expand_min_nodes),
                "pocket_raw_only": int(args.raw_only),
                "pocket_edge_mode": str(args.edge_mode),
                "pocket_radius_cutoff": float(args.radius_cutoff),
                "pocket_radius_max_neighbors": int(args.radius_max_neighbors),
                "pocket_edge_feat_dim": int(edge_feat_dim),
                "pocket_edge_rbf_bins": int(args.edge_rbf_bins),
                "pocket_node_attr_dim": int(POCKET_NODE_STRUCT_DIM),
                "pocket_node_struct_dim": int(POCKET_NODE_STRUCT_DIM),
                "meta": {
                    "pred_csv": str(pred_csv),
                    "res_csv": str(res_csv),
                    "pdb_path": str(pdb_path),
                    "seqdist_norm": str(args.seqdist_norm),
                    "seqdist_norm_denom": float(args.seqdist_norm_denom),
                    "edge_use_seqdist": int(args.edge_use_seqdist),
                    "use_struct_feat": int(args.use_struct_feat),
                },
            }

            out_path = out_dir / f"{pid}.pt"
            torch.save(out, out_path)
            n_ok += 1

        except Exception as e:
            n_failed += 1
            if n_failed <= 5:
                print(f"[WARN] failed pid={pid} err={e}")

    print(f"[OK] wrote pocket cache dir: {out_dir}")
    print(f"[INFO] proteins_in_scope={len(pids)} ok={n_ok} failed_or_missing={n_failed}")


if __name__ == "__main__":
    main()
