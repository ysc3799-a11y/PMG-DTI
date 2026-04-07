import argparse
import json
from pathlib import Path
from typing import Optional
import os
import sys

import pandas as pd

# Ensure project root is on sys.path when running as a script.
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from p2rank_pockets_strict import build_index_from_protein_pockets_csv
from pocket_graph_seqdist import load_p2rank_pockets


def mirror_processed_path(path: Path) -> Optional[Path]:
    parts = list(path.parts)
    if "interim" not in parts:
        return None
    i = parts.index("interim")
    parts[i] = "processed"
    return Path(*parts)


def main():
    ap = argparse.ArgumentParser(description="Parse P2Rank pockets into interim jsonl.")
    ap.add_argument("--full_with_pid_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--protein_pockets_csv", default="data/raw/human/p2rank_raw/protein_pockets.csv")
    ap.add_argument("--p2rank_out_dir", default="data/raw/human/p2rank_raw")
    ap.add_argument("--topk", type=int, default=5)
    ap.add_argument("--prob_thres", type=float, default=0.0)
    ap.add_argument("--min_residues", type=int, default=5)
    ap.add_argument("--out_jsonl", default="data/interim/human/pocket_candidates/pockets.jsonl")
    ap.add_argument("--out_stats_json", default="data/interim/human/pocket_candidates/stats.json")
    ap.add_argument("--no_mirror_processed", action="store_true", help="Disable auto mirror to data/processed/*")
    args = ap.parse_args()

    df = pd.read_csv(args.full_with_pid_csv)
    if "Protein_ID" not in df.columns:
        raise ValueError("full_with_pid_csv must contain Protein_ID")
    pids = sorted(df["Protein_ID"].astype(str).str.split("_len=").str[0].unique().tolist())

    idx = build_index_from_protein_pockets_csv(
        protein_pockets_csv=args.protein_pockets_csv,
        p2rank_out_dir=args.p2rank_out_dir,
        protein_id_col="Protein_ID",
        predictions_col="predictions_csv",
    )

    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n_with_file = 0
    n_with_pocket = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for pid in pids:
            res_csv, pred_csv = idx.get(pid, ("", ""))
            has_files = bool(res_csv)
            if has_files:
                n_with_file += 1
            pockets = load_p2rank_pockets(
                pred_csv,
                res_csv,
                top_k=int(args.topk),
                prob_threshold=float(args.prob_thres),
                min_residues=int(args.min_residues),
            ) if has_files else []
            if pockets:
                n_with_pocket += 1
            rec = {
                "Protein_ID": pid,
                "has_files": has_files,
                "num_pockets": len(pockets),
                "pockets": [
                    {
                        "rank": int(x.get("rank", 0) or 0),
                        "score": float(x.get("score", 0.0) or 0.0),
                        "prob": float(x.get("prob", 0.0) or 0.0),
                        "num_residues": int(len(x.get("residues", []))),
                    }
                    for x in pockets
                ],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "n_pid_total": len(pids),
        "n_pid_with_p2rank_files": n_with_file,
        "n_pid_with_valid_pockets": n_with_pocket,
        "coverage_files": float(n_with_file / max(len(pids), 1)),
        "coverage_pockets": float(n_with_pocket / max(len(pids), 1)),
    }
    out_stats = Path(args.out_stats_json)
    out_stats.parent.mkdir(parents=True, exist_ok=True)
    out_stats.write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")

    if not args.no_mirror_processed:
        m_jsonl = mirror_processed_path(out_jsonl)
        if m_jsonl is not None and m_jsonl != out_jsonl:
            m_jsonl.parent.mkdir(parents=True, exist_ok=True)
            m_jsonl.write_text(out_jsonl.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[OK] mirrored: {m_jsonl}")

        m_stats = mirror_processed_path(out_stats)
        if m_stats is not None and m_stats != out_stats:
            m_stats.parent.mkdir(parents=True, exist_ok=True)
            m_stats.write_text(out_stats.read_text(encoding="utf-8"), encoding="utf-8")
            print(f"[OK] mirrored: {m_stats}")

    print(f"[OK] {out_jsonl}")
    print(f"[OK] {out_stats}")
    print("[STATS]", stats)


if __name__ == "__main__":
    main()
