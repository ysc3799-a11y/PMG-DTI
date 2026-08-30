#!/usr/bin/env python3
"""Deterministically regenerate the released stratified five-fold splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


DATASETS = {
    "human": {
        "rows": 6728,
        "sha256": "f52517071e32bc28437df974e913521835394d0be75015df477f57f166d3b908",
    },
    "biosnap": {
        "rows": 27464,
        "sha256": "b6056790a65910e22bc2dcddd77fb6d200cff38f5b58474675357005b7eac90b",
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_split(path: Path, train_idx, val_idx, test_idx) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "train": sorted(map(int, train_idx)),
        "val": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def split_stats(df: pd.DataFrame, indices) -> tuple[int, int, int, float]:
    labels = df.loc[list(indices), "Y"].astype(int)
    positives = int(labels.sum())
    total = int(len(labels))
    return total, positives, total - positives, float(positives / max(total, 1))


def resolve_from_root(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def portable_path(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def generate_dataset(
    dataset: str,
    root: Path,
    data_root: str,
    output_root: str,
    seed: int,
    n_splits: int,
    val_fraction: float,
    allow_hash_mismatch: bool,
) -> None:
    expected = DATASETS[dataset]
    csv_path = resolve_from_root(root, data_root) / dataset / "pairs_with_pid" / "full_with_pid.csv"
    if not csv_path.is_file():
        raise SystemExit(f"Input CSV not found: {csv_path}")

    actual_hash = sha256_file(csv_path)
    if actual_hash != expected["sha256"] and not allow_hash_mismatch:
        raise SystemExit(
            f"SHA-256 mismatch for {dataset}: expected {expected['sha256']}, "
            f"got {actual_hash}. Use --allow-hash-mismatch only for a deliberately "
            "different dataset version."
        )

    # The released indices use the stored CSV row order. No sorting or
    # deduplication is performed before reset_index(drop=True).
    df = pd.read_csv(csv_path).reset_index(drop=True)
    required_columns = {"SMILES", "Protein", "Y", "Protein_ID", "Protein_Len"}
    missing = required_columns.difference(df.columns)
    if missing:
        raise SystemExit(f"{dataset} CSV is missing required columns: {sorted(missing)}")
    if len(df) != expected["rows"] and not allow_hash_mismatch:
        raise SystemExit(
            f"Unexpected row count for {dataset}: expected {expected['rows']}, got {len(df)}"
        )

    labels = df["Y"].astype(int).to_numpy()
    if not set(map(int, labels)).issubset({0, 1}):
        raise SystemExit(f"{dataset} Y column must contain only binary labels 0 and 1")
    all_indices = df.index.to_numpy()

    out_dir = resolve_from_root(root, output_root) / dataset / f"fivefold_seed{seed}"
    prefix = f"{dataset}_fivefold_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (dev_pos, test_pos) in enumerate(splitter.split(all_indices, labels)):
        dev_idx = all_indices[dev_pos]
        test_idx = all_indices[test_pos]
        dev_labels = labels[dev_idx]

        validation_splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=val_fraction,
            random_state=seed + fold,
        )
        train_local, val_local = next(validation_splitter.split(dev_idx, dev_labels))
        train_idx = dev_idx[train_local]
        val_idx = dev_idx[val_local]

        split_path = out_dir / f"{prefix}_fold{fold}.json"
        write_split(split_path, train_idx, val_idx, test_idx)

        train = split_stats(df, train_idx)
        val = split_stats(df, val_idx)
        test = split_stats(df, test_idx)
        rows.append(
            {
                "fold": fold,
                "split_path": portable_path(split_path, root),
                "n_train": train[0],
                "pos_train": train[1],
                "neg_train": train[2],
                "pos_rate_train": train[3],
                "n_val": val[0],
                "pos_val": val[1],
                "neg_val": val[2],
                "pos_rate_val": val[3],
                "n_test": test[0],
                "pos_test": test[1],
                "neg_test": test[2],
                "pos_rate_test": test[3],
            }
        )

    stats_path = out_dir / f"{prefix}_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "dataset": dataset,
        "source_csv": portable_path(csv_path, root),
        "source_sha256": actual_hash,
        "row_count": int(len(df)),
        "columns": list(map(str, df.columns)),
        "row_order": "stored CSV order; no sorting or deduplication",
        "split_seed": seed,
        "n_splits": n_splits,
        "validation_fraction_of_development_set": val_fraction,
        "validation_seed_rule": "split_seed + fold_index",
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
    }
    (out_dir / f"{prefix}_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"[OK] {dataset}: wrote {n_splits} folds to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("human", "biosnap", "all"), default="all")
    parser.add_argument("--data-root", default="data/interim")
    parser.add_argument("--output-root", default="splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--val-fraction-of-dev", type=float, default=0.1)
    parser.add_argument("--allow-hash-mismatch", action="store_true")
    args = parser.parse_args()

    if args.n_splits < 2:
        raise SystemExit("--n-splits must be at least 2")
    if not 0.0 < args.val_fraction_of_dev < 1.0:
        raise SystemExit("--val-fraction-of-dev must be between 0 and 1")

    root = Path(__file__).resolve().parents[1]
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    for dataset in datasets:
        generate_dataset(
            dataset=dataset,
            root=root,
            data_root=args.data_root,
            output_root=args.output_root,
            seed=args.seed,
            n_splits=args.n_splits,
            val_fraction=args.val_fraction_of_dev,
            allow_hash_mismatch=args.allow_hash_mismatch,
        )


if __name__ == "__main__":
    main()
