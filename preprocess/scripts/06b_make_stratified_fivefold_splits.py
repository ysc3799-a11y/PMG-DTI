import argparse
import csv
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


def _write_split(path: Path, train_idx, val_idx, test_idx):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {
        "train": sorted(map(int, train_idx)),
        "val": sorted(map(int, val_idx)),
        "test": sorted(map(int, test_idx)),
    }
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _stat(df: pd.DataFrame, idxs):
    y = df.loc[list(idxs), "Y"].astype(int)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    return int(len(y)), pos, neg, float(pos / max(len(y), 1))


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create stratified five-fold splits. For each fold, one fold is used "
            "as test and the remaining four folds are split into train/val at 9:1."
        )
    )
    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--out_dir", default="data/processed/human/splits/fivefold_seed42")
    ap.add_argument("--prefix", default="human_fivefold_seed42")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_splits", type=int, default=5)
    ap.add_argument("--val_fraction_of_dev", type=float, default=0.1)
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).reset_index(drop=True)
    if "Y" not in df.columns:
        raise ValueError("Input CSV must contain a Y column.")
    y = df["Y"].astype(int).to_numpy()
    all_indices = df.index.to_numpy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skf = StratifiedKFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
    for fold, (dev_pos, test_pos) in enumerate(skf.split(all_indices, y)):
        dev_idx = all_indices[dev_pos]
        test_idx = all_indices[test_pos]

        dev_y = y[dev_idx]
        splitter = StratifiedShuffleSplit(
            n_splits=1,
            test_size=args.val_fraction_of_dev,
            random_state=args.seed + fold,
        )
        train_local, val_local = next(splitter.split(dev_idx, dev_y))
        train_idx = dev_idx[train_local]
        val_idx = dev_idx[val_local]

        split_path = out_dir / f"{args.prefix}_fold{fold}.json"
        _write_split(split_path, train_idx, val_idx, test_idx)

        train_n, train_pos, train_neg, train_rate = _stat(df, train_idx)
        val_n, val_pos, val_neg, val_rate = _stat(df, val_idx)
        test_n, test_pos_n, test_neg, test_rate = _stat(df, test_idx)
        rows.append(
            {
                "fold": fold,
                "split_path": str(split_path.as_posix()),
                "n_train": train_n,
                "pos_train": train_pos,
                "neg_train": train_neg,
                "pos_rate_train": train_rate,
                "n_val": val_n,
                "pos_val": val_pos,
                "neg_val": val_neg,
                "pos_rate_val": val_rate,
                "n_test": test_n,
                "pos_test": test_pos_n,
                "neg_test": test_neg,
                "pos_rate_test": test_rate,
            }
        )

    stats_path = out_dir / f"{args.prefix}_stats.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {args.n_splits} split files to {out_dir}")
    print(f"[OK] stats={stats_path}")


if __name__ == "__main__":
    main()
