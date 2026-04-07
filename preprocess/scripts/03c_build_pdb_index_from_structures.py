import argparse
from pathlib import Path

import pandas as pd


def infer_pid_from_name(stem: str) -> str:
    # Legacy names may be:
    #   <Protein_ID>__<UniProt>__F1
    # New names may be:
    #   <Protein_ID>
    s = str(stem).strip()
    if "__" in s:
        s = s.split("__")[0]
    if "_len=" in s:
        s = s.split("_len=")[0]
    return s


def main():
    ap = argparse.ArgumentParser(description="Build pdb_index.csv from downloaded structure files.")
    ap.add_argument("--struct_dir", default="data/raw/human/alphafold_structures")
    ap.add_argument("--out_csv", default="data/raw/human/p2rank_raw/pdb_index.csv")
    ap.add_argument("--prefer_ext", default=".pdb,.cif,.bcif")
    args = ap.parse_args()

    struct_dir = Path(args.struct_dir)
    if not struct_dir.is_dir():
        raise FileNotFoundError(struct_dir)

    exts = [x.strip().lower() for x in str(args.prefer_ext).split(",") if x.strip()]
    prio = {e: i for i, e in enumerate(exts)}

    files = [p for p in struct_dir.iterdir() if p.is_file() and p.suffix.lower() in prio]
    files = sorted(files, key=lambda p: (prio.get(p.suffix.lower(), 999), p.name))

    pid2path = {}
    for fp in files:
        pid = infer_pid_from_name(fp.stem)
        if not pid:
            continue
        if pid not in pid2path:
            pid2path[pid] = str(fp.resolve())

    out = pd.DataFrame(
        [{"Protein_ID": pid, "pdb_path": path} for pid, path in sorted(pid2path.items())]
    )
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"[OK] {out_csv}")
    print(f"[INFO] n_pid={len(out)}")


if __name__ == "__main__":
    main()

