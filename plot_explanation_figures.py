import argparse
import io
import json
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from rdkit import Chem
    from rdkit.Chem.Draw import rdMolDraw2D
except Exception:
    Chem = None
    rdMolDraw2D = None

try:
    from PIL import Image
except Exception:
    Image = None


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--explain_dir", required=True, help="path under explanations/<sample_tag>")
    p.add_argument("--pocket_index", type=int, default=None, help="0-based pocket index; default uses best learned pocket")
    p.add_argument("--top_residues", type=int, default=6)
    p.add_argument("--top_evidence", type=int, default=8)
    p.add_argument("--dpi", type=int, default=220)
    return p.parse_args()


def load_csv(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def choose_pocket(pockets: pd.DataFrame, pocket_index: Optional[int]) -> int:
    if pocket_index is not None:
        return int(pocket_index)
    if pockets.empty:
        return 0
    if "learned_pocket_prob" in pockets.columns and pockets["learned_pocket_prob"].notna().any():
        row = pockets.sort_values(by="learned_pocket_prob", ascending=False).iloc[0]
        return int(row["pocket_index"])
    row = pockets.sort_values(by="p2rank_prob", ascending=False).iloc[0]
    return int(row["pocket_index"])


def _norm(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    vmax = float(np.nanmax(values))
    if not np.isfinite(vmax) or vmax <= 1e-12:
        return np.zeros_like(values, dtype=float)
    return np.clip(values / vmax, 0.0, 1.0)


def _truncate(text: str, max_len: int = 18) -> str:
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "…"


def build_mol_image(smiles: str, pharm_df: pd.DataFrame, size: Tuple[int, int] = (900, 600)):
    if Chem is None or rdMolDraw2D is None or Image is None or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    highlight_atoms: List[int] = []
    atom_cols: Dict[int, Tuple[float, float, float]] = {}
    radii: Dict[int, float] = {}
    color_bank = [
        (0.82, 0.24, 0.21),
        (0.17, 0.49, 0.72),
        (0.20, 0.62, 0.39),
        (0.84, 0.68, 0.14),
        (0.53, 0.35, 0.70),
    ]
    if not pharm_df.empty:
        tmp = pharm_df.sort_values(by="pharm_score_proxy", ascending=False).head(8).reset_index(drop=True)
        for i, row in tmp.iterrows():
            atom_idx = int(row.get("anchor_atom_index", -1))
            if atom_idx < 0:
                continue
            if atom_idx not in highlight_atoms:
                highlight_atoms.append(atom_idx)
            atom_cols[atom_idx] = color_bank[i % len(color_bank)]
            radii[atom_idx] = 0.35 + 0.25 * float(row.get("pharm_score_proxy", 0.0))

    drawer = rdMolDraw2D.MolDraw2DCairo(int(size[0]), int(size[1]))
    opts = drawer.drawOptions()
    opts.addAtomIndices = True
    opts.bondLineWidth = 2
    drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, highlightAtomColors=atom_cols, highlightAtomRadii=radii)
    drawer.FinishDrawing()
    png = drawer.GetDrawingText()
    return Image.open(io.BytesIO(png))


def plot_panel_a(ax, summary: Dict, pharm_df: pd.DataFrame):
    img = build_mol_image(summary.get("smiles", ""), pharm_df)
    ax.axis("off")
    if img is not None:
        ax.imshow(np.asarray(img))
    else:
        ax.text(0.5, 0.55, summary.get("smiles", "SMILES unavailable"), ha="center", va="center", fontsize=12)
        ax.text(0.5, 0.42, "RDKit drawing unavailable", ha="center", va="center", fontsize=9, color="#666")
    ax.set_title("(a) Drug 2D + Key Atoms / Pharmacophores", loc="left", fontsize=12, fontweight="bold")

    lines = []
    if not pharm_df.empty:
        tmp = pharm_df.sort_values(by="pharm_score_proxy", ascending=False).head(6)
        for _, row in tmp.iterrows():
            lines.append(
                f"atom {int(row['anchor_atom_index'])}: {_truncate(row['token_label'], 20)} "
                f"({float(row['pharm_score_proxy']):.2f})"
            )
    text = "\n".join(lines) if lines else "No pharmacophore tokens exported."
    ax.text(
        0.02,
        0.02,
        text,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.9, edgecolor="#d0d0d0"),
    )


def plot_panel_b(ax, residues_df: pd.DataFrame, pocket_row: Optional[pd.Series], top_residues: int):
    ax.set_title("(b) Pocket Structure + High-Score Residues", loc="left", fontsize=12, fontweight="bold")
    if residues_df.empty:
        ax.text2D(0.5, 0.5, "No residue coordinates exported.", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return
    score = residues_df["residue_score_proxy"].to_numpy(dtype=float)
    size = 50 + 170 * _norm(score)
    sc = ax.scatter(
        residues_df["x"].to_numpy(dtype=float),
        residues_df["y"].to_numpy(dtype=float),
        residues_df["z"].to_numpy(dtype=float),
        c=score,
        s=size,
        cmap="YlOrRd",
        edgecolors="#333333",
        linewidths=0.4,
        alpha=0.95,
    )
    top_df = residues_df.sort_values(by="residue_score_proxy", ascending=False).head(int(top_residues))
    for _, row in top_df.iterrows():
        label = f"{row['resname']}{int(row['residue_seq'])}"
        ax.text(float(row["x"]), float(row["y"]), float(row["z"]), label, fontsize=8)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    if pocket_row is not None:
        ax.text2D(
            0.02,
            0.98,
            f"Pocket {int(pocket_row['pocket_rank'])}  learned={float(pocket_row.get('learned_pocket_prob', np.nan)):.3f}  p2rank={float(pocket_row.get('p2rank_prob', np.nan)):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white", alpha=0.9, edgecolor="#d0d0d0"),
        )
    return sc


def load_heatmap(path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    df = pd.read_csv(path, index_col=0)
    data = df.to_numpy(dtype=float)
    row_labels = [str(x) for x in df.index.tolist()]
    col_labels = [str(x) for x in df.columns.tolist()]
    return data, row_labels, col_labels


def plot_panel_c(ax, pair_prob_path: str):
    ax.set_title("(c) Pharmacophore-Residue Local Interaction Heatmap", loc="left", fontsize=12, fontweight="bold")
    if not os.path.isfile(pair_prob_path):
        ax.text(0.5, 0.5, "Heatmap csv not found.", ha="center", va="center")
        ax.axis("off")
        return None
    mat, row_labels, col_labels = load_heatmap(pair_prob_path)
    im = ax.imshow(mat, cmap="YlOrRd", aspect="auto", vmin=0.0, vmax=max(0.4, float(np.nanmax(mat))))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_yticklabels([_truncate(x, 24) for x in row_labels], fontsize=8)
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels([_truncate(x, 14) for x in col_labels], rotation=55, ha="right", fontsize=7)
    if mat.size > 0:
        top_flat = np.argsort(mat.reshape(-1))[::-1][: min(3, mat.size)]
        n_cols = mat.shape[1]
        for flat in top_flat:
            i = int(flat // n_cols)
            j = int(flat % n_cols)
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=7, color="#222")
    ax.set_xlabel("Pocket residue / node")
    ax.set_ylabel("Pharmacophore token")
    return im


def plot_panel_d(ax, evid_df: pd.DataFrame, top_evidence: int):
    ax.set_title("(d) High-Score Local Evidence Set", loc="left", fontsize=12, fontweight="bold")
    ax.axis("off")
    if evid_df.empty:
        ax.text(0.5, 0.5, "No set evidence exported.", ha="center", va="center")
        return

    evid_df = evid_df.sort_values(by=["set_token_score", "pair_score_prob"], ascending=False).head(int(top_evidence)).copy()
    pharm_nodes = []
    for _, row in evid_df.iterrows():
        tok = f"p{int(row['pharm_token_index'])}:{_truncate(row['token_label'], 16)}"
        if tok not in pharm_nodes:
            pharm_nodes.append(tok)
    residue_nodes = []
    for _, row in evid_df.iterrows():
        res = f"{row['resname']}{int(row['residue_seq'])}"
        if res not in residue_nodes:
            residue_nodes.append(res)

    py = np.linspace(0.85, 0.15, max(len(pharm_nodes), 1))
    ry = np.linspace(0.85, 0.15, max(len(residue_nodes), 1))
    px = 0.18
    rx = 0.82
    pos = {name: (px, py[i]) for i, name in enumerate(pharm_nodes)}
    pos.update({name: (rx, ry[i]) for i, name in enumerate(residue_nodes)})

    for _, row in evid_df.iterrows():
        left = f"p{int(row['pharm_token_index'])}:{_truncate(row['token_label'], 16)}"
        right = f"{row['resname']}{int(row['residue_seq'])}"
        x0, y0 = pos[left]
        x1, y1 = pos[right]
        w = 1.0 + 5.0 * float(row["set_token_score"])
        ax.plot([x0, x1], [y0, y1], color="#c23b22", alpha=0.35 + 0.5 * float(row["pair_score_prob"]), lw=w)

    for name in pharm_nodes:
        x, y = pos[name]
        ax.scatter([x], [y], s=220, color="#2f6c8f", edgecolors="#16394c", zorder=3)
        ax.text(x - 0.02, y, name, ha="right", va="center", fontsize=8.5)
    for name in residue_nodes:
        x, y = pos[name]
        ax.scatter([x], [y], s=220, color="#c23b22", edgecolors="#6f2012", zorder=3)
        ax.text(x + 0.02, y, name, ha="left", va="center", fontsize=8.5)

    ax.text(0.18, 0.93, "Pharmacophore tokens", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.text(0.82, 0.93, "Pocket residues", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)


def main():
    args = parse_args()
    explain_dir = os.path.abspath(args.explain_dir)
    fig_dir = os.path.join(explain_dir, "figures")
    ensure_dir(fig_dir)

    with open(os.path.join(explain_dir, "summary.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)
    pharm_df = load_csv(os.path.join(explain_dir, "pharm_tokens.csv"))
    pockets_df = load_csv(os.path.join(explain_dir, "pockets.csv"))
    evid_df = load_csv(os.path.join(explain_dir, "set_evidence.csv"))

    pocket_index = choose_pocket(pockets_df, args.pocket_index)
    pocket_row = None
    if not pockets_df.empty and (pockets_df["pocket_index"] == pocket_index).any():
        pocket_row = pockets_df[pockets_df["pocket_index"] == pocket_index].iloc[0]
    residues_path = os.path.join(explain_dir, f"pocket{pocket_index + 1}_residues.csv")
    pair_prob_path = os.path.join(explain_dir, f"pair_score_pocket{pocket_index + 1}_prob.csv")
    residues_df = load_csv(residues_path)
    evid_df = evid_df[evid_df["pocket_index"] == pocket_index].copy() if not evid_df.empty and "pocket_index" in evid_df.columns else evid_df

    fig_a, ax_a = plt.subplots(figsize=(8.5, 5.5))
    plot_panel_a(ax_a, summary, pharm_df)
    fig_a.tight_layout()
    fig_a.savefig(os.path.join(fig_dir, "figure_a_drug.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_a)

    fig_b = plt.figure(figsize=(7.0, 6.2))
    ax_b = fig_b.add_subplot(111, projection="3d")
    sc = plot_panel_b(ax_b, residues_df, pocket_row, args.top_residues)
    if sc is not None:
        fig_b.colorbar(sc, ax=ax_b, shrink=0.7, pad=0.08, label="Residue score proxy")
    fig_b.tight_layout()
    fig_b.savefig(os.path.join(fig_dir, "figure_b_pocket.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_b)

    fig_c, ax_c = plt.subplots(figsize=(8.8, 4.8))
    im = plot_panel_c(ax_c, pair_prob_path)
    if im is not None:
        fig_c.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04, label="Pair score prob")
    fig_c.tight_layout()
    fig_c.savefig(os.path.join(fig_dir, "figure_c_heatmap.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_c)

    fig_d, ax_d = plt.subplots(figsize=(8.8, 4.8))
    plot_panel_d(ax_d, evid_df, args.top_evidence)
    fig_d.tight_layout()
    fig_d.savefig(os.path.join(fig_dir, "figure_d_set_evidence.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_d)

    fig_all = plt.figure(figsize=(16, 11))
    gs = fig_all.add_gridspec(2, 2, hspace=0.26, wspace=0.18)
    ax1 = fig_all.add_subplot(gs[0, 0])
    plot_panel_a(ax1, summary, pharm_df)
    ax2 = fig_all.add_subplot(gs[0, 1], projection="3d")
    sc2 = plot_panel_b(ax2, residues_df, pocket_row, args.top_residues)
    ax3 = fig_all.add_subplot(gs[1, 0])
    im3 = plot_panel_c(ax3, pair_prob_path)
    ax4 = fig_all.add_subplot(gs[1, 1])
    plot_panel_d(ax4, evid_df, args.top_evidence)
    fig_all.suptitle(
        f"Explanation Overview | Protein_ID={summary.get('protein_id', '')} | prob={float(summary.get('prob', np.nan)):.3f}",
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    if sc2 is not None:
        cax1 = fig_all.add_axes([0.92, 0.57, 0.012, 0.22])
        fig_all.colorbar(sc2, cax=cax1, label="Residue score proxy")
    if im3 is not None:
        cax2 = fig_all.add_axes([0.92, 0.15, 0.012, 0.22])
        fig_all.colorbar(im3, cax=cax2, label="Pair score prob")
    fig_all.savefig(os.path.join(fig_dir, "figure_all.png"), dpi=args.dpi, bbox_inches="tight")
    plt.close(fig_all)

    payload = {
        "explain_dir": explain_dir,
        "fig_dir": fig_dir,
        "pocket_index": int(pocket_index),
        "files": [
            "figure_a_drug.png",
            "figure_b_pocket.png",
            "figure_c_heatmap.png",
            "figure_d_set_evidence.png",
            "figure_all.png",
        ],
    }
    with open(os.path.join(fig_dir, "figures_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
