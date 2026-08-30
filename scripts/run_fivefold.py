#!/usr/bin/env python3
"""Run the final PMG-DTI entry point over released five-fold splits."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("human", "biosnap"), default="human")
    parser.add_argument("--cfg", default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-fold", type=int, default=0)
    parser.add_argument("--end-fold", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0 <= args.start_fold <= args.end_fold <= 4:
        raise SystemExit("Fold range must satisfy 0 <= start-fold <= end-fold <= 4")

    root = Path(__file__).resolve().parents[1]
    cfg = Path(args.cfg) if args.cfg else root / "configs" / f"{args.dataset}.yaml"
    if not cfg.is_absolute():
        cfg = (root / cfg).resolve()
    train_script = root / "train" / "train_gated_canon_real_pocket_seqdist_3view_pharm_evidence_emaes.py"
    split_dir = root / "splits" / args.dataset / "fivefold_seed42"
    prefix = f"{args.dataset}_fivefold_seed42"

    if not cfg.is_file():
        raise SystemExit(f"Config not found: {cfg}")
    if not train_script.is_file():
        raise SystemExit(f"Training script not found: {train_script}")

    for fold in range(args.start_fold, args.end_fold + 1):
        split = split_dir / f"{prefix}_fold{fold}.json"
        if not split.is_file():
            raise SystemExit(
                f"Split not found: {split}\n"
                f"Generate it first with: {sys.executable} "
                f"scripts/make_fivefold_splits.py --dataset {args.dataset} --seed 42"
            )
        cmd = [
            sys.executable,
            str(train_script),
            "--cfg", str(cfg),
            "--split_path", str(split),
            "--run_name", f"{args.dataset}_pmg_dti_fold{fold}",
            "--seed", str(args.seed),
            "--device", args.device,
        ]
        print("[RUN]", " ".join(cmd), flush=True)
        if not args.dry_run:
            subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()
