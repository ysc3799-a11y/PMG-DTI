import argparse
import csv
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Build protein_pockets.csv from P2Rank output dirs.")
    ap.add_argument("--p2rank_out_dir", default="data/raw/human/p2rank_raw/p2rank_out")
    ap.add_argument("--out_csv", default="data/raw/human/p2rank_raw/protein_pockets.csv")
    ap.add_argument("--prefer_pred_pattern", default="*predictions*.csv")
    ap.add_argument("--prefer_res_pattern", default="*residues*.csv")
    args = ap.parse_args()

    out_dir = Path(args.p2rank_out_dir)
    if not out_dir.is_dir():
        raise FileNotFoundError(out_dir)

    rows = []
    for d in sorted([x for x in out_dir.iterdir() if x.is_dir()]):
        pid = d.name.split("__")[0].split("_len=")[0]
        preds = sorted(d.rglob(args.prefer_pred_pattern))
        if not preds:
            continue
        pred = preds[0]
        residues = sorted(d.rglob(args.prefer_res_pattern))
        # IMPORTANT:
        # P2Rank outputs two different csv files:
        #   - *predictions*.csv : pocket list + scores
        #   - *residues*.csv    : residue lists for pockets
        # They are NOT interchangeable.
        # If residues csv is missing, keep residue_csv EMPTY so downstream can
        # correctly treat it as missing instead of silently reading the wrong file.
        res = residues[0] if residues else None
        rows.append(
            {
                "Protein_ID": pid,
                "residue_csv": str(res.resolve()) if res is not None else "",
                "predictions_csv": str(pred.resolve()),
            }
        )

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["Protein_ID", "residue_csv", "predictions_csv"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    print(f"[OK] {out_csv}")
    print(f"[INFO] rows={len(rows)}")


if __name__ == "__main__":
    main()

