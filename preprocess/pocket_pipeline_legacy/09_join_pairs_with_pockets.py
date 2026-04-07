# scripts/09_join_pairs_with_pockets.py
import argparse
import hashlib
from pathlib import Path

import pandas as pd


def seq_to_pid(seq: str) -> str:
    """
    你的数据里 Protein_ID 的真实规则就是：
      Protein_ID = "P" + MD5(sequence).upper()[:12]
    例：PD400ED5813FF
    """
    h = hashlib.md5(seq.encode("utf-8")).hexdigest().upper()
    return "P" + h[:12]


def main(pairs_csv: Path, protein_map_csv: Path, pockets_csv: Path, out_csv: Path, top_k: int):
    pairs = pd.read_csv(pairs_csv, encoding="utf-8-sig")
    pockets = pd.read_csv(pockets_csv, encoding="utf-8-sig")

    # pairs 必须有 Protein 序列列
    if "Protein" not in pairs.columns:
        raise ValueError("pairs_csv must contain a 'Protein' column (amino acid sequence).")

    # 生成 Protein_ID（与你当前 pairs_with_pockets 里 PD400ED... 一致）
    pairs["Protein_ID"] = pairs["Protein"].astype(str).map(seq_to_pid)

    # pockets 过滤 top_k（保险起见）
    if "pocket_rank" in pockets.columns:
        pockets = pockets[pockets["pocket_rank"].astype("Int64") <= top_k].copy()

    # 关键：按 Protein_ID join，自动展开成 multi-instance（每个 pair 对应 top_k 行）
    out = pairs.merge(pockets, on="Protein_ID", how="left")

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # 打印覆盖率
    n_pairs = len(pairs)
    n_out = len(out)
    n_with_pocket = out["pocket_rank"].notna().sum() if "pocket_rank" in out.columns else 0
    print("[OK] wrote:", out_csv)
    print("  pairs_rows:", n_pairs)
    print("  out_rows:", n_out)
    print("  rows_with_pocket_rank_nonnull:", int(n_with_pocket))
    print("  unique proteins in pairs:", pairs["Protein_ID"].nunique())
    if "Protein_ID" in pockets.columns:
        print("  unique proteins in pockets:", pockets["Protein_ID"].nunique())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", required=True)
    # 为了兼容你之前的命令，保留这个参数但脚本不依赖它（可以随便给一个存在的路径）
    ap.add_argument("--protein_map_csv", required=False, default="")
    ap.add_argument("--pockets_csv", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--top_k", type=int, default=10)
    args = ap.parse_args()

    main(
        pairs_csv=Path(args.pairs_csv),
        protein_map_csv=Path(args.protein_map_csv) if args.protein_map_csv else Path(""),
        pockets_csv=Path(args.pockets_csv),
        out_csv=Path(args.out_csv),
        top_k=args.top_k,
    )
