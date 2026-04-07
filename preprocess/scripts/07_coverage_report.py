import argparse
import csv
import json
from pathlib import Path

import pandas as pd
import torch


def load_manifest(path: Path):
    recs = []
    if not path.is_file():
        return recs
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except Exception:
                continue
    return recs


def main():
    ap = argparse.ArgumentParser(description="Compute ESM and pocket cache coverage.")
    ap.add_argument("--full_with_pid_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--esm2_pt", default="data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt")
    ap.add_argument("--manifest_jsonl", default="data/processed/human/pocket_manifest/manifest.jsonl")
    ap.add_argument("--out_summary_csv", default="data/processed/human/coverage/summary.csv")
    ap.add_argument("--out_json", default="data/processed/human/coverage/coverage.json")
    args = ap.parse_args()

    df = pd.read_csv(args.full_with_pid_csv)
    if "Protein_ID" not in df.columns:
        raise ValueError("full_with_pid_csv must contain Protein_ID")
    pids = df["Protein_ID"].astype(str).str.split("_len=").str[0]
    n_rows = len(df)
    unique_pid = sorted(pids.unique().tolist())

    esm = torch.load(args.esm2_pt, map_location="cpu")
    esm_tokens = esm.get("tokens", {}) if isinstance(esm, dict) else {}
    esm_pid = set(str(k).split("_len=")[0] for k in esm_tokens.keys())
    esm_row_cov = float(pids.isin(esm_pid).mean()) if n_rows > 0 else 0.0
    esm_pid_cov = float(sum(1 for x in unique_pid if x in esm_pid) / max(len(unique_pid), 1))

    mf = load_manifest(Path(args.manifest_jsonl))
    status_map = {str(x.get("Protein_ID", "")): str(x.get("status", "")) for x in mf}
    ok_pid = {k for k, v in status_map.items() if v == "ok"}
    valid_pid = {k for k, v in status_map.items() if v in {"ok", "no_valid_pocket"}}
    pocket_ok_row_cov = float(pids.isin(ok_pid).mean()) if n_rows > 0 else 0.0
    pocket_valid_row_cov = float(pids.isin(valid_pid).mean()) if n_rows > 0 else 0.0
    pocket_ok_pid_cov = float(sum(1 for x in unique_pid if x in ok_pid) / max(len(unique_pid), 1))

    out = {
        "n_rows": int(n_rows),
        "n_unique_pid": int(len(unique_pid)),
        "esm_row_coverage": esm_row_cov,
        "esm_pid_coverage": esm_pid_cov,
        "pocket_ok_row_coverage": pocket_ok_row_cov,
        "pocket_valid_row_coverage": pocket_valid_row_cov,
        "pocket_ok_pid_coverage": pocket_ok_pid_cov,
    }

    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

    out_csv = Path(args.out_summary_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for k, v in out.items():
            w.writerow([k, v])

    print(f"[OK] {out_json}")
    print(f"[OK] {out_csv}")
    print("[COVERAGE]", out)


if __name__ == "__main__":
    main()

