import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PAIR_COLUMNS = ["SMILES", "Protein"]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create the manuscript Human cold-pair split. The 5% validation "
            "and 10% test pairs are sampled stratified by label, and training "
            "pairs sharing any held-out drug or protein are removed."
        )
    )
    parser.add_argument(
        "--raw_csv",
        default="data/human/full.csv",
        help="Original Human CSV used to identify the 5997 unique pairs.",
    )
    parser.add_argument(
        "--aligned_csv",
        default="data/interim/human/pairs_with_pid/full_with_pid.csv",
        help="Row-aligned training CSV whose indices are written to the split JSON.",
    )
    parser.add_argument(
        "--out",
        default="data/processed/human/splits/human_cold_pair_seed42.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def subset_stats(df, indices):
    subset = df.loc[indices]
    positives = int(subset["Y"].astype(int).sum())
    return {
        "samples": int(len(subset)),
        "positives": positives,
        "negatives": int(len(subset) - positives),
        "drugs": int(subset["SMILES"].astype(str).nunique()),
        "proteins": int(subset["Protein"].astype(str).nunique()),
    }


def main():
    args = parse_args()
    raw = pd.read_csv(args.raw_csv).reset_index(drop=True)
    aligned = pd.read_csv(args.aligned_csv).reset_index(drop=True)

    required_raw = {"SMILES", "Protein", "Y"}
    required_aligned = {"SMILES", "Protein", "Y"}
    if not required_raw.issubset(raw.columns):
        raise ValueError(f"{args.raw_csv} must contain {sorted(required_raw)}")
    if not required_aligned.issubset(aligned.columns):
        raise ValueError(f"{args.aligned_csv} must contain {sorted(required_aligned)}")
    if len(raw) != len(aligned):
        raise ValueError(
            f"CSV row counts differ: raw={len(raw)}, aligned={len(aligned)}"
        )
    if not raw["Y"].astype(int).equals(aligned["Y"].astype(int)):
        raise ValueError("Y is not row-aligned between the two CSV files")
    if not raw["Protein"].astype(str).equals(aligned["Protein"].astype(str)):
        raise ValueError("Protein is not row-aligned between the two CSV files")

    # Keep the first occurrence of each original drug-target pair. These
    # indices still point to the row-aligned training CSV and feature caches.
    unique_mask = ~raw.duplicated(PAIR_COLUMNS, keep="first")
    unique_indices = raw.index[unique_mask].to_numpy()
    unique_labels = raw.loc[unique_indices, "Y"].astype(int).to_numpy()

    train_candidates, holdout = train_test_split(
        unique_indices,
        test_size=0.15,
        random_state=args.seed,
        stratify=unique_labels,
    )
    holdout_labels = aligned.loc[holdout, "Y"].astype(int).to_numpy()
    val_indices, test_indices = train_test_split(
        holdout,
        test_size=2.0 / 3.0,
        random_state=args.seed,
        stratify=holdout_labels,
    )

    heldout_drugs = set(aligned.loc[val_indices, "SMILES"].astype(str))
    heldout_drugs.update(aligned.loc[test_indices, "SMILES"].astype(str))
    heldout_proteins = set(aligned.loc[val_indices, "Protein"].astype(str))
    heldout_proteins.update(aligned.loc[test_indices, "Protein"].astype(str))

    candidate_rows = aligned.loc[train_candidates]
    keep_train = (
        ~candidate_rows["SMILES"].astype(str).isin(heldout_drugs)
        & ~candidate_rows["Protein"].astype(str).isin(heldout_proteins)
    )
    train_indices = train_candidates[keep_train.to_numpy()]

    split = {
        "train": sorted(int(i) for i in train_indices),
        "val": sorted(int(i) for i in val_indices),
        "test": sorted(int(i) for i in test_indices),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(split, indent=2), encoding="utf-8")

    stats = {
        name: subset_stats(aligned, indices)
        for name, indices in (
            ("Train", train_indices),
            ("Val", val_indices),
            ("Test", test_indices),
        )
    }
    train_drugs = set(aligned.loc[train_indices, "SMILES"].astype(str))
    train_proteins = set(aligned.loc[train_indices, "Protein"].astype(str))

    print(f"[OK] split={out_path}")
    print(
        f"[INFO] unique_pairs={len(unique_indices)} "
        f"train_candidates={len(train_candidates)} "
        f"removed_from_train={len(train_candidates) - len(train_indices)}"
    )
    print("subset,samples,positives,negatives,drugs,proteins")
    for name in ("Train", "Val", "Test"):
        s = stats[name]
        print(
            f"{name},{s['samples']},{s['positives']},{s['negatives']},"
            f"{s['drugs']},{s['proteins']}"
        )
    print(
        "[CHECK] train-heldout overlap: "
        f"drugs={len(train_drugs & heldout_drugs)} "
        f"proteins={len(train_proteins & heldout_proteins)}"
    )


if __name__ == "__main__":
    main()
