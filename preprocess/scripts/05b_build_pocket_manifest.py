import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch


def safe_load_pt(path: Path) -> Any:
    try:
        return torch.load(str(path), weights_only=True, map_location="cpu")
    except Exception:
        return torch.load(str(path), map_location="cpu")


def count_valid_templates(templates: List[Any]) -> int:
    n_valid = 0
    for t in templates:
        if not isinstance(t, dict):
            continue
        res_seq = t.get("res_seq", [])
        if isinstance(res_seq, (list, tuple)) and len(res_seq) > 0:
            n_valid += 1
    return int(n_valid)


def normalize_cache_path(path: Path, mode: str) -> str:
    if mode == "absolute":
        return str(path.resolve())
    try:
        return os.path.relpath(str(path.resolve()), start=os.getcwd())
    except Exception:
        return str(path)


def main():
    ap = argparse.ArgumentParser(description="Build pocket manifest jsonl from pocket cache directory.")
    ap.add_argument("--cache_dir", default="data/processed/human/pocket_graph_cache")
    ap.add_argument("--out_jsonl", default="data/processed/human/pocket_manifest/manifest.jsonl")
    ap.add_argument("--path_mode", default="relative", choices=["relative", "absolute"])
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    if not cache_dir.is_dir():
        raise FileNotFoundError(cache_dir)

    files = sorted(cache_dir.glob("*.pt"))
    out_jsonl = Path(args.out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    n_ok = 0
    n_no_valid = 0
    n_bad = 0

    with out_jsonl.open("w", encoding="utf-8") as f:
        for fp in files:
            pid = fp.stem
            rec: Dict[str, Any] = {
                "Protein_ID": pid,
                "status": "load_error",
                "num_pockets_total": 0,
                "num_pockets_valid": 0,
                "cache_path": normalize_cache_path(fp, str(args.path_mode)),
            }
            try:
                obj = safe_load_pt(fp)
                if isinstance(obj, dict):
                    pid = str(obj.get("pid", pid) or pid)
                    templates = obj.get("templates", [])
                    n_total = int(len(templates)) if isinstance(templates, list) else 0
                    n_valid = count_valid_templates(templates if isinstance(templates, list) else [])
                    rec.update(
                        {
                            "Protein_ID": pid,
                            "status": "ok" if n_valid > 0 else "no_valid_pocket",
                            "num_pockets_total": n_total,
                            "num_pockets_valid": n_valid,
                        }
                    )
                    if n_valid > 0:
                        n_ok += 1
                    else:
                        n_no_valid += 1
                else:
                    n_bad += 1
            except Exception:
                n_bad += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[OK] {out_jsonl}")
    print(
        "[STATS]",
        {
            "n_cache_files": int(len(files)),
            "n_ok": int(n_ok),
            "n_no_valid_pocket": int(n_no_valid),
            "n_load_error": int(n_bad),
        },
    )


if __name__ == "__main__":
    main()
