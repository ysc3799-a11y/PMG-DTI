import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Any

import pandas as pd
import torch


def safe_load_pt(path: str) -> Any:
    try:
        return torch.load(path, weights_only=True, map_location="cpu")
    except Exception:
        return torch.load(path, map_location="cpu")


def main():
    ap = argparse.ArgumentParser(description="Sanity checks for PhaDTI preprocessing artifacts.")
    ap.add_argument("--pairs_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--seq_col", default="Protein")
    ap.add_argument("--pharm_pt", default="data/interim/human/pharm_features/pharm_max64_seed42.pt")
    ap.add_argument("--esm2_pt", default="data/interim/human/esm2_tokens/esm2_tokens_t6_fp16.pt")
    ap.add_argument("--pocket_cache_dir", default="data/processed/human/pocket_graph_cache")
    ap.add_argument("--n_samples", type=int, default=5)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    print(f"[pairs] rows={len(df)} unique_pid={df[args.pid_col].nunique()} unique_smiles={df['SMILES'].nunique() if 'SMILES' in df.columns else 'NA'}")

    # --- pharm ---
    pharm = safe_load_pt(args.pharm_pt)
    pt = pharm["pharm_type"].long()
    pm = pharm["pharm_mask"].long()
    pa = pharm.get("pharm_anchor", None)
    valid = (pm > 0)
    n_valid = int(valid.sum().item())
    n_bad0 = int(((pt == 0) & valid).sum().item())
    uniq = sorted(set(pt[valid].view(-1).tolist()))
    print("[pharm] shape=", tuple(pt.shape), "valid_tokens=", n_valid, "valid_with_type0=", n_bad0)
    print("[pharm] uniq_type_ids(valid) sample:", uniq[:20], "... total=", len(uniq))
    if "num_types" in pharm:
        print("[pharm] num_types=", pharm["num_types"])
    if n_valid > 0:
        valid_ids = pt[valid].view(-1).tolist()
        cnt = Counter(int(x) for x in valid_ids if int(x) > 0)
        top5 = sorted(cnt.items(), key=lambda kv: (-int(kv[1]), int(kv[0])))[:5]
        top5_share = (sum(int(v) for _, v in top5) / max(1, int(n_valid))) if top5 else 0.0
        row_uniq = []
        for i in range(int(pt.shape[0])):
            ids_i = pt[i][pm[i] > 0].view(-1).tolist()
            row_uniq.append(len(set(int(x) for x in ids_i if int(x) > 0)))
        med_uniq = float(torch.tensor(row_uniq, dtype=torch.float32).median().item()) if row_uniq else 0.0
        print("[pharm] top5_token_share=", f"{top5_share:.4f}", "median_unique_types_per_row=", f"{med_uniq:.2f}")
    if pa is not None:
        pa = pa.long()
        bad_anchor = int(((pa < 0) & valid).sum().item())
        print("[pharm] anchor_shape=", tuple(pa.shape), "valid_with_anchor<0=", bad_anchor)

    # --- esm ---
    esm = safe_load_pt(args.esm2_pt)
    tokens: Dict[str, torch.Tensor] = esm.get("tokens", {})
    print(f"[esm] num_tokens={len(tokens)}")

    pid2seq = (
        df[[args.pid_col, args.seq_col]]
        .drop_duplicates(args.pid_col)
        .set_index(args.pid_col)[args.seq_col]
        .astype(str)
        .to_dict()
    )

    pids = list(pid2seq.keys())
    random.shuffle(pids)
    trunc = 0
    checked = 0
    for pid in pids:
        if pid not in tokens:
            continue
        seq = pid2seq[pid]
        L_full = len(seq)
        L_tok = int(tokens[pid].shape[0])
        checked += 1
        if L_tok < L_full:
            trunc += 1
        if checked >= max(20, args.n_samples * 5):
            break
    if checked:
        print(f"[esm] checked={checked} truncated={trunc} ({trunc/checked:.1%})")

    # --- pocket cache ---
    cache_dir = Path(args.pocket_cache_dir)
    if cache_dir.is_dir():
        sample_files = list(cache_dir.glob("*.pt"))
        print(f"[pocket_cache] dir={cache_dir} n_files={len(sample_files)}")
        for f in random.sample(sample_files, k=min(args.n_samples, len(sample_files))):
            obj = safe_load_pt(str(f))
            pid = obj.get("pid", f.stem)
            templates = obj.get("templates", [])
            n_valid = sum(1 for t in templates if isinstance(t, dict) and t.get("res_seq"))
            lens = [len(t.get("res_seq", [])) if isinstance(t, dict) else 0 for t in templates]
            print(f"  - pid={pid} seq_len={obj.get('seq_len')} topk={obj.get('pocket_topk')} valid={n_valid}/{len(templates)} lens={lens}")
    else:
        print(f"[pocket_cache] dir not found: {cache_dir}")


if __name__ == "__main__":
    main()
