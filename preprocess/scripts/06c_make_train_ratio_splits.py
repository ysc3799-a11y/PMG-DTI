import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


def _read_split(path: Path):
    obj = json.loads(path.read_text(encoding="utf-8"))
    for key in ("train", "val", "test"):
        if key not in obj:
            raise KeyError(f"split file missing '{key}': {path}")
    return {key: [int(x) for x in obj[key]] for key in ("train", "val", "test")}


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
    rate = float(pos / max(len(y), 1))
    return int(len(y)), pos, neg, rate


def _sample_downsample_pos(df: pd.DataFrame, train_idx, neg_per_pos: int, seed: int):
    rng = random.Random(int(seed))
    pos = [int(i) for i in train_idx if int(df.loc[int(i), "Y"]) == 1]
    neg = [int(i) for i in train_idx if int(df.loc[int(i), "Y"]) == 0]
    rng.shuffle(pos)
    rng.shuffle(neg)

    if neg_per_pos < 1:
        raise ValueError(f"neg_per_pos must be >= 1, got {neg_per_pos}")
    n_pos = min(len(pos), len(neg) // int(neg_per_pos))
    if n_pos <= 0:
        raise ValueError(
            f"Cannot create 1:{neg_per_pos} split from train positives={len(pos)}, negatives={len(neg)}"
        )
    keep_pos = pos[:n_pos]
    keep_neg = neg[: n_pos * int(neg_per_pos)]
    train = keep_pos + keep_neg
    rng.shuffle(train)
    return train


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Create class-ratio robustness splits by changing only the training set. "
            "Validation and test indices are copied from a base split."
        )
    )
    ap.add_argument("--in_csv", required=True)
    ap.add_argument("--base_split", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--ratios", nargs="+", type=int, default=[1, 3, 5, 10])
    ap.add_argument(
        "--mode",
        default="downsample_pos",
        choices=["downsample_pos"],
        help="downsample_pos keeps validation/test fixed and simulates imbalance by subsampling train positives.",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).reset_index(drop=True)
    if "Y" not in df.columns:
        raise ValueError("Input CSV must contain a Y column.")

    base = _read_split(Path(args.base_split))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for ratio in args.ratios:
        train_idx = _sample_downsample_pos(
            df=df,
            train_idx=base["train"],
            neg_per_pos=int(ratio),
            seed=int(args.seed) * 1009 + int(ratio),
        )
        split_path = out_dir / f"{args.prefix}_pos1_neg{int(ratio)}_seed{int(args.seed)}.json"
        _write_split(split_path, train_idx, base["val"], base["test"])

        tr_n, tr_pos, tr_neg, tr_rate = _stat(df, train_idx)
        va_n, va_pos, va_neg, va_rate = _stat(df, base["val"])
        te_n, te_pos, te_neg, te_rate = _stat(df, base["test"])
        rows.append(
            {
                "ratio": f"1:{int(ratio)}",
                "split_path": split_path.as_posix(),
                "n_train": tr_n,
                "pos_train": tr_pos,
                "neg_train": tr_neg,
                "pos_rate_train": tr_rate,
                "n_val": va_n,
                "pos_val": va_pos,
                "neg_val": va_neg,
                "pos_rate_val": va_rate,
                "n_test": te_n,
                "pos_test": te_pos,
                "neg_test": te_neg,
                "pos_rate_test": te_rate,
            }
        )

    stats_path = out_dir / f"{args.prefix}_train_ratio_stats_seed{int(args.seed)}.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] wrote {len(rows)} train-ratio split files to {out_dir}")
    print(f"[OK] stats={stats_path}")


if __name__ == "__main__":
    main()
