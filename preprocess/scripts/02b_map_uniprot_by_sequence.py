import argparse
from pathlib import Path
from typing import Optional

import pandas as pd


def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def mirror_processed_path(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    if "interim" not in parts:
        return None
    i = parts.index("interim")
    parts[i] = "processed"
    return Path(*parts)


def main():
    ap = argparse.ArgumentParser(description="Map Protein_ID to UniProt accession by exact sequence match.")
    ap.add_argument("--protein_table", default="data/interim/human/pairs_with_pid/protein_table.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--seq_col", default="Protein")
    ap.add_argument("--uniprot_tsv", default="data/raw/human/moltrans/uniprot_human_reviewed.tsv")
    ap.add_argument("--uniprot_acc_col", default="")
    ap.add_argument("--uniprot_seq_col", default="")
    ap.add_argument("--out_map_csv", default="data/interim/human/pairs_with_pid/protein_uniprot_map.csv")
    ap.add_argument("--no_mirror_processed", action="store_true", help="Disable auto mirror to data/processed/*")
    args = ap.parse_args()

    ptab = Path(args.protein_table)
    utab = Path(args.uniprot_tsv)
    if not ptab.is_file():
        raise FileNotFoundError(ptab)
    if not utab.is_file():
        raise FileNotFoundError(utab)

    d1 = pd.read_csv(ptab)
    if args.pid_col not in d1.columns or args.seq_col not in d1.columns:
        raise ValueError(f"protein_table must contain {args.pid_col}/{args.seq_col}")

    d2 = pd.read_csv(utab, sep="\t")
    acc_col = args.uniprot_acc_col or find_col(d2, ["Entry", "accession", "Accession"])
    seq_col = args.uniprot_seq_col or find_col(d2, ["Sequence", "sequence"])
    if acc_col is None or seq_col is None:
        raise ValueError(f"Cannot detect UniProt accession/sequence columns, got={list(d2.columns)}")

    seq_to_acc = {}
    dup_seq = set()
    for _, r in d2[[acc_col, seq_col]].dropna().iterrows():
        acc = str(r[acc_col]).strip()
        seq = str(r[seq_col]).strip()
        if not acc or not seq:
            continue
        if seq in seq_to_acc and seq_to_acc[seq] != acc:
            dup_seq.add(seq)
        else:
            seq_to_acc[seq] = acc

    rows = []
    n_unique = 0
    n_multi = 0
    n_none = 0
    for _, r in d1.iterrows():
        pid = str(r[args.pid_col]).strip()
        seq = str(r[args.seq_col]).strip()
        acc = ""
        status = "no_match"
        if seq in seq_to_acc:
            if seq in dup_seq:
                status = "multi_match"
                n_multi += 1
            else:
                acc = seq_to_acc[seq]
                status = "unique_match"
                n_unique += 1
        else:
            n_none += 1
        rows.append(
            {
                "Protein_ID": pid,
                "Protein_Len": len(seq),
                "UniProt_ID": acc,
                "map_status": status,
            }
        )

    out = pd.DataFrame(rows)
    out_csv = Path(args.out_map_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    if not args.no_mirror_processed:
        m_csv = mirror_processed_path(out_csv)
        if m_csv is not None and m_csv != out_csv:
            m_csv.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(m_csv, index=False)
            print(f"[OK] mirrored: {m_csv}")

    print(f"[OK] {out_csv}")
    print(
        "[STATS]",
        {
            "n_total": int(len(out)),
            "n_unique_match": int(n_unique),
            "n_multi_match": int(n_multi),
            "n_no_match": int(n_none),
            "coverage_unique": float(n_unique / max(len(out), 1)),
        },
    )


if __name__ == "__main__":
    main()
