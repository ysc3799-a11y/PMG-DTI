import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


POCKET_COLORS = ["#B7F0B1", "#CFC8FF", "#BDE7F2", "#F8D6A5", "#F6B7D2"]
RESIDUE_COLORS = ["#8FBF88", "#8E88CC", "#6FA8B5", "#C9A26B", "#C97BAA"]


def quote_path(path: Path) -> str:
    return '"' + path.resolve().as_posix() + '"'


def compress_ranges(values: List[int]) -> str:
    if not values:
        return ""
    vals = sorted(set(int(v) for v in values))
    ranges: List[str] = []
    start = vals[0]
    prev = vals[0]
    for v in vals[1:]:
        if v == prev + 1:
            prev = v
            continue
        ranges.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = v
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(ranges)


def residue_spec_from_df(df: pd.DataFrame) -> str:
    parts: List[str] = []
    if df.empty:
        return ""
    for chain_id, sub in df.groupby(df["chain_id"].fillna("").astype(str)):
        nums = []
        for x in sub["residue_seq"].tolist():
            try:
                nums.append(int(x))
            except Exception:
                continue
        if not nums:
            continue
        seq = compress_ranges(nums)
        if str(chain_id).strip():
            parts.append(f"#1/{chain_id}:{seq}")
        else:
            parts.append(f"#1:{seq}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " | ".join(parts) + ")"


def context_spec_from_residue_dfs(residue_dfs: List[pd.DataFrame], pad: int = 60) -> str:
    parts: List[str] = []
    if not residue_dfs:
        return ""
    merged = pd.concat(residue_dfs, ignore_index=True)
    for chain_id, sub in merged.groupby(merged["chain_id"].fillna("").astype(str)):
        nums = []
        for x in sub["residue_seq"].tolist():
            try:
                nums.append(int(x))
            except Exception:
                continue
        if not nums:
            continue
        lo = max(1, min(nums) - int(max(0, pad)))
        hi = max(nums) + int(max(0, pad))
        seq = f"{lo}-{hi}" if lo != hi else str(lo)
        if str(chain_id).strip():
            parts.append(f"#1/{chain_id}:{seq}")
        else:
            parts.append(f"#1:{seq}")
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "(" + " | ".join(parts) + ")"


def top_residue_spec(df: pd.DataFrame, n: int = 4) -> str:
    if df.empty:
        return ""
    rank_col = "residue_score_proxy" if "residue_score_proxy" in df.columns else None
    sub = df.copy()
    if rank_col is not None:
        sub = sub.sort_values(rank_col, ascending=False)
    sub = sub.head(int(max(1, n)))
    return residue_spec_from_df(sub)


def select_top_pockets(pockets_df: pd.DataFrame, top_k: int) -> pd.DataFrame:
    if pockets_df.empty:
        return pockets_df
    df = pockets_df.copy()
    for c in ["is_valid", "num_nodes", "learned_pocket_prob", "p2rank_score", "pocket_rank", "pocket_index"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "is_valid" in df.columns:
        df = df[df["is_valid"].fillna(0) > 0].copy()
    if "num_nodes" in df.columns:
        df = df[df["num_nodes"].fillna(0) > 0].copy()
    sort_cols: List[str] = []
    ascend: List[bool] = []
    if "learned_pocket_prob" in df.columns:
        sort_cols.append("learned_pocket_prob")
        ascend.append(False)
    if "p2rank_score" in df.columns:
        sort_cols.append("p2rank_score")
        ascend.append(False)
    if "pocket_rank" in df.columns:
        sort_cols.append("pocket_rank")
        ascend.append(True)
    if sort_cols:
        df = df.sort_values(sort_cols, ascending=ascend, na_position="last")
    return df.head(int(top_k)).reset_index(drop=True)


def panel_a_script(structure_path: Path, pocket_specs: List[Dict], context_spec: str = "") -> str:
    lines = [
        "# Panel A: global protein with top pockets",
        "close all",
        "set bgColor white",
        f"open {quote_path(structure_path)}",
        "hide atoms",
        "hide pbonds",
        "show protein cartoons",
        "hide protein cartoons" if context_spec else "",
        f"show {context_spec} cartoons" if context_spec else "",
        "cartoon style protein width 2.1 thick 0.30",
        "color protein #D9B1C8 cartoons",
    ]
    for i, p in enumerate(pocket_specs, start=1):
        lines.extend(
            [
                f"surface {p['res_spec']} color {p['surface_color']} transparency 45 visiblePatches 1 replace false",
                f"show {p['top_res_spec']} atoms" if p["top_res_spec"] else "",
                f"style {p['top_res_spec']} stick" if p["top_res_spec"] else "",
                f"size {p['top_res_spec']} stickRadius 0.18" if p["top_res_spec"] else "",
                f"color {p['top_res_spec']} {p['residue_color']} atoms" if p["top_res_spec"] else "",
            ]
        )
    lines.append(f"view {context_spec}" if context_spec else "view")
    return "\n".join(x for x in lines if x)


def panel_b_script(structure_path: Path, pocket: Dict, idx: int) -> str:
    lines = [
        f"# Panel B{idx}: local pocket zoom",
        "close all",
        "set bgColor white",
        f"open {quote_path(structure_path)}",
        "hide atoms",
        "hide pbonds",
        "show protein cartoons",
        "cartoon style protein width 1.9 thick 0.26",
        "color protein #E8C9DA cartoons",
        f"surface {pocket['res_spec']} color {pocket['surface_color']} transparency 58 visiblePatches 1",
        f"show {pocket['res_spec']} atoms",
        f"style {pocket['res_spec']} stick",
        f"size {pocket['res_spec']} stickRadius 0.16",
        f"color {pocket['res_spec']} {pocket['residue_color']} atoms",
    ]
    if pocket["top_res_spec"]:
        lines.extend(
            [
                f"show {pocket['top_res_spec']} atoms",
                f"style {pocket['top_res_spec']} stick",
                f"size {pocket['top_res_spec']} stickRadius 0.22",
                "color " + pocket["top_res_spec"] + " #D9D24E atoms",
            ]
        )
    lines.append(f"view {pocket['res_spec']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Generate ChimeraX scripts for A global + Pocket_1/2/3 local panels from one explanation directory.")
    ap.add_argument("--explain_dir", required=True)
    ap.add_argument("--top_k", type=int, default=3)
    ap.add_argument("--out_dir", default="")
    args = ap.parse_args()

    explain_dir = Path(args.explain_dir).resolve()
    summary_path = explain_dir / "summary.json"
    pockets_path = explain_dir / "pockets.csv"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    if not pockets_path.is_file():
        raise FileNotFoundError(pockets_path)

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    structure_path = Path(str(summary.get("pdb_path") or "")).resolve()
    if not structure_path.is_file():
        raise FileNotFoundError(structure_path)

    pockets_df = pd.read_csv(pockets_path)
    top_df = select_top_pockets(pockets_df, top_k=int(args.top_k))
    if top_df.empty:
        raise RuntimeError("No valid pockets found in pockets.csv")

    out_dir = Path(args.out_dir).resolve() if args.out_dir else explain_dir / "multipocket_chimerax"
    out_dir.mkdir(parents=True, exist_ok=True)

    pocket_specs: List[Dict] = []
    selected_residue_dfs: List[pd.DataFrame] = []
    for idx, (_, row) in enumerate(top_df.iterrows(), start=1):
        pocket_index = int(row["pocket_index"])
        residues_path = explain_dir / f"pocket{pocket_index + 1}_residues.csv"
        if not residues_path.is_file():
            continue
        residues_df = pd.read_csv(residues_path)
        res_spec = residue_spec_from_df(residues_df)
        if not res_spec:
            continue
        selected_residue_dfs.append(residues_df)
        pocket_specs.append(
            {
                "panel_index": idx,
                "pocket_index": pocket_index,
                "pocket_rank": int(row["pocket_rank"]) if pd.notna(row.get("pocket_rank")) else None,
                "res_spec": res_spec,
                "top_res_spec": top_residue_spec(residues_df, n=4),
                "surface_color": POCKET_COLORS[(idx - 1) % len(POCKET_COLORS)],
                "residue_color": RESIDUE_COLORS[(idx - 1) % len(RESIDUE_COLORS)],
                "learned_pocket_prob": float(row["learned_pocket_prob"]) if pd.notna(row.get("learned_pocket_prob")) else None,
                "p2rank_score": float(row["p2rank_score"]) if pd.notna(row.get("p2rank_score")) else None,
                "residues_csv": str(residues_path.resolve()),
            }
        )

    if not pocket_specs:
        raise RuntimeError("No pocket residue CSVs available for selected pockets")

    panel_a_path = out_dir / "A_global_top_pockets.cxc"
    context_spec = context_spec_from_residue_dfs(selected_residue_dfs, pad=60)
    panel_a_path.write_text(panel_a_script(structure_path, pocket_specs, context_spec=context_spec), encoding="utf-8")

    manifest_rows = []
    for p in pocket_specs:
        b_path = out_dir / f"B_pocket{p['panel_index']}.cxc"
        b_path.write_text(panel_b_script(structure_path, p, p["panel_index"]), encoding="utf-8")
        manifest_rows.append(
            {
                "panel": f"Pocket_{p['panel_index']}",
                "pocket_index": p["pocket_index"],
                "pocket_rank": p["pocket_rank"],
                "learned_pocket_prob": p["learned_pocket_prob"],
                "p2rank_score": p["p2rank_score"],
                "residues_csv": p["residues_csv"],
                "script_path": str(b_path.resolve()),
            }
        )

    pd.DataFrame(manifest_rows).to_csv(out_dir / "manifest.csv", index=False)

    readme = [
        "# Multi-Pocket ChimeraX",
        "",
        "This folder contains ChimeraX scripts for one explanation case.",
        "",
        "Files:",
        "- `A_global_top_pockets.cxc`: global protein with Top-K predicted pocket surfaces",
    ]
    for p in pocket_specs:
        readme.append(f"- `B_pocket{p['panel_index']}.cxc`: local zoom for Pocket_{p['panel_index']}")
    readme.extend(
        [
            "",
            "Important:",
            "- These panels visualize predicted candidate pockets from your model/P2Rank pipeline on the AlphaFold protein structure.",
            "- They do not contain a real ligand pose. If you want ligand sticks inside each pocket, you still need docking or a matched real complex pose.",
            "",
            "Recommended paper workflow:",
            "1. Render `A_global_top_pockets.cxc` in ChimeraX.",
            "2. Render each `B_pocket*.cxc` separately.",
            "3. Add dashed insets and panel labels in PPT/Illustrator/Inkscape.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    report = {
        "explain_dir": str(explain_dir),
        "structure_path": str(structure_path),
        "n_selected_pockets": len(pocket_specs),
        "panel_a_script": str(panel_a_path.resolve()),
        "manifest_csv": str((out_dir / "manifest.csv").resolve()),
        "out_dir": str(out_dir.resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] {panel_a_path}")
    for p in pocket_specs:
        b_path = out_dir / f"B_pocket{p['panel_index']}.cxc"
        print(f"[OK] {b_path}")
    print("[SUMMARY]", report)


if __name__ == "__main__":
    main()
