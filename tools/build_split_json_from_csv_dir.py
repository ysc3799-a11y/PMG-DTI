import argparse
import json
from collections import defaultdict, deque
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import pandas as pd


DEFAULT_KEY_COLS = ("SMILES", "Protein", "Y")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Convert train/val/test CSV split files into the index-based JSON split "
            "format used by the training scripts."
        )
    )
    p.add_argument("--full_csv", required=True, help="Reference full CSV whose row order defines split indices.")
    p.add_argument("--train_csv", required=True, help="Train split CSV.")
    p.add_argument("--val_csv", required=True, help="Validation split CSV.")
    p.add_argument("--test_csv", required=True, help="Test split CSV.")
    p.add_argument("--out", required=True, help="Output split JSON path.")
    p.add_argument(
        "--key_cols",
        nargs="+",
        default=list(DEFAULT_KEY_COLS),
        help="Columns used to match split rows back to the reference full CSV.",
    )
    p.add_argument(
        "--aligned_csv",
        default=None,
        help=(
            "Optional model CSV aligned row-by-row with full_csv. When provided, the script "
            "verifies the two files are row-aligned before writing the split JSON."
        ),
    )
    return p.parse_args()


def _require_cols(df: pd.DataFrame, cols: Sequence[str], path: Path) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {missing}")


def _row_key_iter(df: pd.DataFrame, cols: Sequence[str]):
    for row in df.loc[:, list(cols)].itertuples(index=False, name=None):
        yield tuple("" if pd.isna(v) else str(v) for v in row)


def _build_index(full_df: pd.DataFrame, key_cols: Sequence[str]) -> Dict[Tuple[str, ...], Deque[int]]:
    out: Dict[Tuple[str, ...], Deque[int]] = defaultdict(deque)
    for idx, key in enumerate(_row_key_iter(full_df, key_cols)):
        out[key].append(int(idx))
    return out


def _map_split(
    split_name: str,
    split_df: pd.DataFrame,
    key_cols: Sequence[str],
    locs: Dict[Tuple[str, ...], Deque[int]],
    used: set,
) -> List[int]:
    mapped: List[int] = []
    for row_num, key in enumerate(_row_key_iter(split_df, key_cols), start=1):
        q = locs.get(key)
        if not q:
            raise KeyError(f"{split_name} row {row_num} not found in full_csv for key={key}")
        while q and q[0] in used:
            q.popleft()
        if not q:
            raise RuntimeError(
                f"{split_name} row {row_num} has no unused match left in full_csv for key={key}"
            )
        idx = int(q.popleft())
        used.add(idx)
        mapped.append(idx)
    return mapped


def _verify_alignment(full_df: pd.DataFrame, aligned_df: pd.DataFrame, full_path: Path, aligned_path: Path) -> None:
    if len(full_df) != len(aligned_df):
        raise ValueError(
            f"Aligned CSV length mismatch: {full_path} has {len(full_df)} rows, "
            f"but {aligned_path} has {len(aligned_df)} rows."
        )

    check_cols = [c for c in ("Protein", "Y") if c in full_df.columns and c in aligned_df.columns]
    if not check_cols:
        return

    mismatch_mask = None
    for col in check_cols:
        cur = full_df[col].astype(str) != aligned_df[col].astype(str)
        mismatch_mask = cur if mismatch_mask is None else (mismatch_mask | cur)
    if mismatch_mask is not None and bool(mismatch_mask.any()):
        first_bad = int(mismatch_mask.idxmax())
        raise ValueError(
            f"Aligned CSV check failed at row {first_bad}: {full_path} and {aligned_path} are not row-aligned."
        )


def main() -> None:
    args = parse_args()

    full_path = Path(args.full_csv)
    train_path = Path(args.train_csv)
    val_path = Path(args.val_csv)
    test_path = Path(args.test_csv)
    out_path = Path(args.out)
    key_cols = list(args.key_cols)

    full_df = pd.read_csv(full_path)
    _require_cols(full_df, key_cols, full_path)

    if args.aligned_csv:
        aligned_path = Path(args.aligned_csv)
        aligned_df = pd.read_csv(aligned_path)
        _verify_alignment(full_df, aligned_df, full_path, aligned_path)

    split_paths = {
        "train": train_path,
        "val": val_path,
        "test": test_path,
    }
    split_dfs: Dict[str, pd.DataFrame] = {}
    for split_name, split_path in split_paths.items():
        df = pd.read_csv(split_path)
        _require_cols(df, key_cols, split_path)
        split_dfs[split_name] = df

    locs = _build_index(full_df, key_cols)
    used: set = set()
    split_obj: Dict[str, List[int]] = {}

    for split_name in ("train", "val", "test"):
        split_obj[split_name] = _map_split(split_name, split_dfs[split_name], key_cols, locs, used)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(split_obj, f, ensure_ascii=False, indent=2)

    payload = {
        "full_csv": str(full_path),
        "aligned_csv": str(args.aligned_csv) if args.aligned_csv else None,
        "out": str(out_path),
        "key_cols": key_cols,
        "counts": {k: len(v) for k, v in split_obj.items()},
        "total_used": int(sum(len(v) for v in split_obj.values())),
        "full_rows": int(len(full_df)),
        "unused_rows": int(len(full_df) - len(used)),
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
