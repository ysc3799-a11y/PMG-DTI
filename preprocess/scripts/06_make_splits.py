import argparse
import csv
import json
import random
from pathlib import Path

import pandas as pd


def stratified_indices(y, val_ratio, test_ratio, seed):
    pos = [i for i, v in enumerate(y) if int(v) == 1]
    neg = [i for i, v in enumerate(y) if int(v) == 0]
    rng = random.Random(seed)
    rng.shuffle(pos)
    rng.shuffle(neg)

    def cut(arr):
        n = len(arr)
        n_test = int(round(n * test_ratio))
        n_val = int(round(n * val_ratio))
        test = arr[:n_test]
        val = arr[n_test : n_test + n_val]
        train = arr[n_test + n_val :]
        return train, val, test

    tr_p, va_p, te_p = cut(pos)
    tr_n, va_n, te_n = cut(neg)
    train = tr_p + tr_n
    val = va_p + va_n
    test = te_p + te_n
    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test


def group_split(df, group_col, val_ratio, test_ratio, seed):
    groups = sorted(df[group_col].astype(str).unique().tolist())
    rng = random.Random(seed)
    rng.shuffle(groups)

    n = len(df)
    tgt_test = int(round(n * test_ratio))
    tgt_val = int(round(n * val_ratio))

    g2idx = {}
    for g, sub in df.groupby(group_col):
        g2idx[str(g)] = sub.index.tolist()

    test_g, val_g = set(), set()
    cnt_test = 0
    for g in groups:
        if cnt_test >= tgt_test:
            break
        test_g.add(g)
        cnt_test += len(g2idx[g])

    cnt_val = 0
    for g in groups:
        if g in test_g:
            continue
        if cnt_val >= tgt_val:
            break
        val_g.add(g)
        cnt_val += len(g2idx[g])

    train, val, test = [], [], []
    for g, idxs in g2idx.items():
        if g in test_g:
            test.extend(idxs)
        elif g in val_g:
            val.extend(idxs)
        else:
            train.extend(idxs)
    return train, val, test


def write_split(path: Path, tr, va, te):
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {"train": sorted(map(int, tr)), "val": sorted(map(int, va)), "test": sorted(map(int, te))}
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def summarize(df, tr, va, te):
    def stat(idxs):
        if not idxs:
            return (0, 0.0)
        yy = df.loc[idxs, "Y"].astype(int).tolist()
        return (len(yy), float(sum(yy) / len(yy)))
    ntr, ptr = stat(tr)
    nva, pva = stat(va)
    nte, pte = stat(te)
    return ntr, ptr, nva, pva, nte, pte


def main():
    ap = argparse.ArgumentParser(description="Create random/cold-protein/cold-drug splits.")
    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val_ratio", type=float, default=0.1)
    ap.add_argument("--test_ratio", type=float, default=0.2)
    ap.add_argument("--out_dir", default="data/processed/human/splits")
    ap.add_argument("--prefix", default="human", help="output file prefix, e.g. human or biosnap")
    args = ap.parse_args()

    df = pd.read_csv(args.in_csv).reset_index(drop=True)
    if "Y" not in df.columns:
        raise ValueError("csv must contain Y")
    if "Protein_ID" not in df.columns:
        raise ValueError("csv must contain Protein_ID")
    if "SMILES" not in df.columns:
        raise ValueError("csv must contain SMILES")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tr, va, te = stratified_indices(df["Y"].astype(int).tolist(), args.val_ratio, args.test_ratio, args.seed)
    prefix = str(args.prefix).strip() or "human"
    p_random = out_dir / f"{prefix}_random_seed{args.seed}.json"
    write_split(p_random, tr, va, te)

    trp, vap, tep = group_split(df, "Protein_ID", args.val_ratio, args.test_ratio, args.seed)
    p_prot = out_dir / f"{prefix}_unseen_protein_seed{args.seed}.json"
    write_split(p_prot, trp, vap, tep)

    trd, vad, ted = group_split(df, "SMILES", args.val_ratio, args.test_ratio, args.seed)
    p_drug = out_dir / f"{prefix}_unseen_drug_seed{args.seed}.json"
    write_split(p_drug, trd, vad, ted)

    stats_path = out_dir / f"split_stats_seed{args.seed}.csv"
    with stats_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split_name", "n_train", "pos_train", "n_val", "pos_val", "n_test", "pos_test"])
        w.writerow(["random", *summarize(df, tr, va, te)])
        w.writerow(["unseen_protein", *summarize(df, trp, vap, tep)])
        w.writerow(["unseen_drug", *summarize(df, trd, vad, ted)])

    print(f"[OK] {p_random}")
    print(f"[OK] {p_prot}")
    print(f"[OK] {p_drug}")
    print(f"[OK] {stats_path}")


if __name__ == "__main__":
    main()
