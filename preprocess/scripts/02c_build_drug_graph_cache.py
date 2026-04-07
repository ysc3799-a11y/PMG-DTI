import argparse
from pathlib import Path
import os
import sys

import pandas as pd
import torch

try:
    from tqdm import tqdm
except Exception:
    tqdm = None

# Ensure project root is on sys.path when running as a script.
_THIS = os.path.abspath(__file__)
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_THIS)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from dataset.featurize_canon import (
    DEFAULT_MORGAN_NBITS,
    DEFAULT_MORGAN_RADIUS,
    DRUG_DESC_DIM,
    smiles_to_drug_data,
)


def main():
    ap = argparse.ArgumentParser(description="Precompute drug graph features cache from SMILES.")
    ap.add_argument("--in_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--out_pt", default="data/processed/human/drug_graph_cache/drug_graph_cache.pt")
    ap.add_argument("--morgan_radius", type=int, default=DEFAULT_MORGAN_RADIUS)
    ap.add_argument("--morgan_nbits", type=int, default=DEFAULT_MORGAN_NBITS)
    args = ap.parse_args()

    in_csv = Path(args.in_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    if args.smiles_col not in df.columns:
        raise ValueError(f"missing smiles col: {args.smiles_col}")

    smiles_list = df[args.smiles_col].astype(str).tolist()
    uniq_smiles = sorted(set(smiles_list))

    graphs = {}
    it = uniq_smiles
    if tqdm is not None:
        it = tqdm(uniq_smiles, desc="build_drug_graph_cache")

    for smi in it:
        data = smiles_to_drug_data(
            smi,
            morgan_radius=int(args.morgan_radius),
            morgan_nbits=int(args.morgan_nbits),
        )
        graphs[smi] = {
            "node_feat": data["node_feat"].cpu(),
            "edge_index": data["edge_index"].cpu(),
            "edge_attr": data["edge_attr"].cpu(),
            "node_pos": data["node_pos"].cpu(),
            "drug_fp": data["drug_fp"].cpu(),
            "drug_desc": data["drug_desc"].cpu(),
        }

    out = {
        "graphs": graphs,
        "n_rows": int(len(df)),
        "n_unique_smiles": int(len(uniq_smiles)),
        "drug_edge_feat_dim": int(next(iter(graphs.values()))["edge_attr"].shape[1]) if graphs else 0,
        "drug_desc_dim": int(DRUG_DESC_DIM),
        "drug_fp_dim": int(args.morgan_nbits),
        "morgan_radius": int(args.morgan_radius),
        "morgan_nbits": int(args.morgan_nbits),
    }
    out_pt = Path(args.out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, out_pt)
    print(f"[OK] saved: {out_pt}")
    print(f"[INFO] n_rows={len(df)} n_unique_smiles={len(uniq_smiles)}")


if __name__ == "__main__":
    main()
