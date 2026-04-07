# scripts/08_collect_pockets.py
import argparse
import re
from pathlib import Path

import pandas as pd


def _safe_strip_cols(df: pd.DataFrame) -> pd.DataFrame:
    # 处理你遇到的列名带大量空格问题： "name     ,  rank,   score, ..."
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _read_predictions(pred_path: Path) -> pd.DataFrame:
    # skipinitialspace=True 会把分隔符后的空格跳过；再对列名 strip
    df = pd.read_csv(pred_path, sep=",", skipinitialspace=True, engine="python")
    df = _safe_strip_cols(df)
    return df


def _protein_id_short(protein_id_long: str) -> str:
    """
    把类似：
      P004E8E304EC9_len=1010_pairs=2
    统一成：
      P004E8E304EC9
    """
    if protein_id_long is None:
        return ""
    s = str(protein_id_long).strip()
    # 最稳：按你流水线里固定出现的 _len= 截断
    return s.split("_len=")[0]


def _parse_outdir_name(outdir_name: str):
    """
    你的 p2rank_out 子目录名通常是：
      <Protein_ID>__<UniProt_AC>__F1
    例如：
      P004E8E304EC9_len=1010_pairs=2__P30153__F1
    """
    parts = outdir_name.split("__")
    protein_id_long = parts[0].strip() if len(parts) >= 1 else outdir_name.strip()
    uniprot = parts[1].strip() if len(parts) >= 2 else ""
    frag = parts[2].strip() if len(parts) >= 3 else ""
    return protein_id_long, uniprot, frag


def main(mapping_csv: Path, p2rank_out_dir: Path, out_pockets_csv: Path, top_k: int):
    p2rank_out_dir = Path(p2rank_out_dir)
    out_pockets_csv = Path(out_pockets_csv)
    out_pockets_csv.parent.mkdir(parents=True, exist_ok=True)

    # mapping_csv 可选用于兜底 UniProt（不强依赖）
    mapping_df = None
    map_dict_long = {}
    map_dict_short = {}
    if mapping_csv and Path(mapping_csv).exists():
        mapping_df = pd.read_csv(mapping_csv, encoding="utf-8-sig")
        if "Protein_ID" in mapping_df.columns and "UniProt_AC" in mapping_df.columns:
            for _, r in mapping_df.iterrows():
                pid_long = str(r["Protein_ID"])
                ac = str(r["UniProt_AC"])
                map_dict_long[pid_long] = ac
                map_dict_short[_protein_id_short(pid_long)] = ac

    rows = []
    n_dirs = 0
    n_with_pred = 0
    n_skipped_no_pred = 0
    n_parse_fail = 0

    for out_sub in sorted([p for p in p2rank_out_dir.iterdir() if p.is_dir()]):
        n_dirs += 1

        # 递归找 predictions（有的版本会放在子层目录）
        pred_files = sorted(out_sub.rglob("*predictions*.csv"))
        if not pred_files:
            n_skipped_no_pred += 1
            continue

        # 选一个最合理的 predictions 文件（通常只有 1 个；多个时取最大的）
        pred_path = max(pred_files, key=lambda x: x.stat().st_size)
        protein_id_long, uniprot_from_name, _frag = _parse_outdir_name(out_sub.name)
        protein_id = _protein_id_short(protein_id_long)

        # UniProt 优先用目录名里的；没有再从 mapping 兜底
        uniprot = uniprot_from_name
        if (not uniprot) and protein_id_long in map_dict_long:
            uniprot = map_dict_long[protein_id_long]
        if (not uniprot) and protein_id in map_dict_short:
            uniprot = map_dict_short[protein_id]

        try:
            df = _read_predictions(pred_path)
        except Exception:
            n_parse_fail += 1
            continue

        # 必要列（列名统一 lower+strip 后应该能匹配）
        required = ["rank", "score", "probability", "center_x", "center_y", "center_z"]
        # 某些版本可能没有 probability；允许缺失
        if "probability" not in df.columns:
            # 有些版本叫 prob
            if "prob" in df.columns:
                df["probability"] = df["prob"]
            else:
                df["probability"] = pd.NA

        missing_core = [c for c in ["rank", "score", "center_x", "center_y", "center_z"] if c not in df.columns]
        if missing_core:
            # 如果核心列缺失，无法使用
            n_parse_fail += 1
            continue

        df = df.sort_values("rank", ascending=True).head(top_k)
        n_with_pred += 1

        rel_pred = str(pred_path.relative_to(p2rank_out_dir)).replace("\\", "/")

        for _, r in df.iterrows():
            rows.append(
                {
                    "Protein_ID": protein_id,
                    "UniProt_AC": uniprot,
                    "pocket_rank": int(r["rank"]) if pd.notna(r["rank"]) else pd.NA,
                    "score": float(r["score"]) if pd.notna(r["score"]) else pd.NA,
                    "probability": float(r["probability"]) if pd.notna(r["probability"]) else pd.NA,
                    "center_x": float(r["center_x"]) if pd.notna(r["center_x"]) else pd.NA,
                    "center_y": float(r["center_y"]) if pd.notna(r["center_y"]) else pd.NA,
                    "center_z": float(r["center_z"]) if pd.notna(r["center_z"]) else pd.NA,
                    "predictions_csv": rel_pred,
                }
            )

    out_df = pd.DataFrame(rows)
    out_df.to_csv(out_pockets_csv, index=False, encoding="utf-8-sig")

    print("[OK] wrote:", out_pockets_csv)
    print("  p2rank_out_dirs:", n_dirs)
    print("  dirs_with_predictions:", n_with_pred)
    print("  skipped_no_predictions:", n_skipped_no_pred)
    print("  parse_failed:", n_parse_fail)
    print("  pockets_rows:", len(out_df))
    if len(out_df) > 0:
        print("  unique proteins:", out_df["Protein_ID"].nunique())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mapping_csv", required=True)
    ap.add_argument("--p2rank_out_dir", required=True)
    ap.add_argument("--out_pockets_csv", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    main(
        mapping_csv=Path(args.mapping_csv),
        p2rank_out_dir=Path(args.p2rank_out_dir),
        out_pockets_csv=Path(args.out_pockets_csv),
        top_k=args.top_k,
    )
