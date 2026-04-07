import os
from typing import Dict, Tuple

import pandas as pd


def build_index_from_protein_pockets_csv(
    protein_pockets_csv: str,
    p2rank_out_dir: str = "",
    protein_id_col: str = "Protein_ID",
    predictions_col: str = "predictions_csv",
) -> Dict[str, Tuple[str, str]]:
    """
    Build pid -> (residue_csv, prediction_csv) index.
    Supports both absolute and relative paths in csv.
    """
    out: Dict[str, Tuple[str, str]] = {}
    if not protein_pockets_csv or not os.path.isfile(protein_pockets_csv):
        return out

    df = pd.read_csv(protein_pockets_csv)
    if protein_id_col not in df.columns:
        return out

    # best effort for residue column naming differences
    residue_col = None
    for c in ["residue_csv", "residues_csv", "res_csv", "residue_list_csv"]:
        if c in df.columns:
            residue_col = c
            break
    if residue_col is None:
        return out

    pred_col = predictions_col if predictions_col in df.columns else None
    if pred_col is None:
        for c in ["prediction_csv", "pred_csv", "predictions"]:
            if c in df.columns:
                pred_col = c
                break

    base = os.path.dirname(os.path.abspath(protein_pockets_csv))
    p2rank_out_dir = str(p2rank_out_dir or "")

    def _resolve(path: str) -> str:
        p = str(path).strip()
        if not p:
            return ""
        if os.path.isabs(p) and os.path.isfile(p):
            return os.path.normpath(p)
        cand = os.path.normpath(os.path.join(base, p))
        if os.path.isfile(cand):
            return cand
        if p2rank_out_dir:
            cand2 = os.path.normpath(os.path.join(p2rank_out_dir, p))
            if os.path.isfile(cand2):
                return cand2
        return os.path.normpath(p)

    for _, row in df.iterrows():
        pid = str(row.get(protein_id_col, "")).strip()
        if not pid:
            continue
        pid = pid.split("_len=")[0]
        res_csv = _resolve(str(row.get(residue_col, "")))
        pred_csv = _resolve(str(row.get(pred_col, ""))) if pred_col is not None else ""
        if res_csv:
            out[pid] = (res_csv, pred_csv)
    return out

