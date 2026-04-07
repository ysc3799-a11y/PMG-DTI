import argparse
import hashlib
import json
from pathlib import Path
from typing import Optional

import pandas as pd

try:
    from rdkit import Chem
except Exception:
    Chem = None


def sha1_text(x: str) -> str:
    return hashlib.sha1(x.encode("utf-8")).hexdigest()


def pick_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def canon_smiles(s: str) -> str:
    s = str(s).strip()
    if not s or Chem is None:
        return s
    m = Chem.MolFromSmiles(s)
    if m is None:
        return s
    return Chem.MolToSmiles(m, canonical=True)


def mirror_processed_path(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    if "interim" not in parts:
        return None
    i = parts.index("interim")
    parts[i] = "processed"
    return Path(*parts)


def main():
    ap = argparse.ArgumentParser(description="Align Protein_ID and clean csv.")
    ap.add_argument("--in_csv", default="data/raw/human/moltrans/full.csv")
    ap.add_argument("--out_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--out_protein_table", default="data/interim/human/pairs_with_pid/protein_table.csv")
    ap.add_argument("--out_summary_json", default="data/interim/human/pairs_with_pid/summary.json")
    ap.add_argument("--no_mirror_processed", action="store_true", help="Disable auto mirror to data/processed/*")
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    c_smiles = pick_col(df, ["SMILES", "smiles", "Drug", "drug"])
    c_prot = pick_col(df, ["Protein", "protein", "Target", "target", "Sequence", "sequence"])
    c_y = pick_col(df, ["Y", "y", "Label", "label", "Interaction"])

    if c_smiles is None or c_prot is None or c_y is None:
        raise ValueError(f"Required columns not found. got={list(df.columns)}")

    out = pd.DataFrame()
    out["SMILES"] = df[c_smiles].astype(str).map(canon_smiles)
    out["Protein"] = df[c_prot].astype(str).str.strip()
    out["Y"] = pd.to_numeric(df[c_y], errors="coerce").fillna(0).astype(int)
    out = out[(out["SMILES"] != "") & (out["Protein"] != "")]
    out = out.reset_index(drop=True)
    out["Protein_ID"] = out["Protein"].map(sha1_text)
    out["Protein_Len"] = out["Protein"].map(len)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)

    if not args.no_mirror_processed:
        m_csv = mirror_processed_path(out_csv)
        if m_csv is not None and m_csv != out_csv:
            m_csv.parent.mkdir(parents=True, exist_ok=True)
            out.to_csv(m_csv, index=False)
            print(f"[OK] mirrored_csv={m_csv}")

    prot_tbl = out[["Protein_ID", "Protein", "Protein_Len"]].drop_duplicates("Protein_ID")
    out_prot = Path(args.out_protein_table)
    out_prot.parent.mkdir(parents=True, exist_ok=True)
    prot_tbl.to_csv(out_prot, index=False)

    if not args.no_mirror_processed:
        m_prot = mirror_processed_path(out_prot)
        if m_prot is not None and m_prot != out_prot:
            m_prot.parent.mkdir(parents=True, exist_ok=True)
            prot_tbl.to_csv(m_prot, index=False)
            print(f"[OK] mirrored_protein_table={m_prot}")

    summary = {
        "n_rows": int(len(out)),
        "n_pos": int((out["Y"] == 1).sum()),
        "n_neg": int((out["Y"] == 0).sum()),
        "n_unique_smiles": int(out["SMILES"].nunique()),
        "n_unique_protein": int(out["Protein_ID"].nunique()),
        "rdkit_used": bool(Chem is not None),
    }
    out_sum = Path(args.out_summary_json)
    out_sum.parent.mkdir(parents=True, exist_ok=True)
    out_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_mirror_processed:
        m_sum = mirror_processed_path(out_sum)
        if m_sum is not None and m_sum != out_sum:
            m_sum.parent.mkdir(parents=True, exist_ok=True)
            m_sum.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[OK] mirrored_summary={m_sum}")

    print(f"[OK] out_csv={out_csv}")
    print(f"[OK] out_protein_table={out_prot}")
    print(f"[OK] out_summary={out_sum}")
    print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()
