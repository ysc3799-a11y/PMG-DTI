import argparse
import io
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--explain_dir", required=True)
    p.add_argument("--score_col", default="atom_score_proxy_sum", choices=["atom_score_proxy_sum", "atom_score_proxy_max"])
    p.add_argument("--out_name", default="atom_importance_heatmap_blue")
    p.add_argument("--width", type=int, default=980)
    p.add_argument("--height", type=int, default=620)
    return p.parse_args()


def _importance_scale(value: float) -> Tuple[float, float, float]:
    value = max(0.0, min(1.0, float(value)))
    low = (0.98, 0.99, 1.00)
    # Darken the blue endpoint by ~20% so high-score atoms stand out more clearly.
    high = (0.00, 0.28, 0.73)
    return tuple(low[i] + (high[i] - low[i]) * value for i in range(3))


def _rgb_float_to_int(rgb: Tuple[float, float, float]) -> Tuple[int, int, int]:
    return tuple(max(0, min(255, int(round(x * 255)))) for x in rgb)


@lru_cache(maxsize=None)
def _load_font(size: int, bold: bool = False):
    candidates = []
    if bold:
        candidates.extend(
            [
                "C:/Windows/Fonts/arialbd.ttf",
                "C:/Windows/Fonts/segoeuib.ttf",
                "arialbd.ttf",
                "DejaVuSans-Bold.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "C:/Windows/Fonts/arial.ttf",
                "C:/Windows/Fonts/segoeui.ttf",
                "arial.ttf",
                "DejaVuSans.ttf",
            ]
        )
    for font_path in candidates:
        try:
            return ImageFont.truetype(font_path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def load_inputs(explain_dir: Path) -> Tuple[str, pd.DataFrame]:
    summary = json.loads((explain_dir / "summary.json").read_text(encoding="utf-8"))
    smiles = str(summary.get("smiles", "")).strip()
    atom_df = pd.read_csv(explain_dir / "drug_atoms.csv")
    return smiles, atom_df


def prepare_mol(smiles: str, atom_df: pd.DataFrame, score_col: str) -> Tuple[Chem.Mol, List[float]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)

    scores = [0.0] * mol.GetNumAtoms()
    for _, row in atom_df.iterrows():
        idx = int(row["atom_index"])
        if 0 <= idx < len(scores):
            scores[idx] = max(0.0, float(row.get(score_col, 0.0)))

    vmax = max(scores) if scores else 0.0
    norm_scores = [s / vmax if vmax > 1e-12 else 0.0 for s in scores]

    direct_note_cutoff = 0.05
    neighbor_focus_cutoff = 0.65
    for atom in mol.GetAtoms():
        idx = int(atom.GetIdx())
        score = float(norm_scores[idx])
        show_note = score >= direct_note_cutoff
        if not show_note:
            for nbr in atom.GetNeighbors():
                if float(norm_scores[int(nbr.GetIdx())]) >= neighbor_focus_cutoff:
                    show_note = True
                    break
        if show_note:
            atom.SetProp("atomNote", f"{score:.2f}")
    return mol, norm_scores


def build_highlights(
    mol: Chem.Mol, norm_scores: List[float]
) -> Tuple[List[int], Dict[int, Tuple[float, float, float]], Dict[int, float], List[int], Dict[int, Tuple[float, float, float]]]:
    highlight_atoms: List[int] = []
    atom_colors: Dict[int, Tuple[float, float, float]] = {}
    atom_radii: Dict[int, float] = {}
    for idx, score in enumerate(norm_scores):
        if score <= 0.0:
            continue
        highlight_atoms.append(idx)
        atom_colors[idx] = _importance_scale(score)
        atom_radii[idx] = 0.24 + 0.26 * float(score)

    highlight_bonds: List[int] = []
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    for bond in mol.GetBonds():
        bidx = int(bond.GetIdx())
        i = int(bond.GetBeginAtomIdx())
        j = int(bond.GetEndAtomIdx())
        score = max(float(norm_scores[i]), float(norm_scores[j]))
        if score <= 0.0:
            continue
        highlight_bonds.append(bidx)
        bond_colors[bidx] = _importance_scale(score)

    return highlight_atoms, atom_colors, atom_radii, highlight_bonds, bond_colors


def draw_molecule_png(
    mol: Chem.Mol,
    highlight_atoms: List[int],
    atom_colors: Dict[int, Tuple[float, float, float]],
    atom_radii: Dict[int, float],
    highlight_bonds: List[int],
    bond_colors: Dict[int, Tuple[float, float, float]],
    size: Tuple[int, int],
) -> Image.Image:
    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.4
    opts.annotationFontScale = 0.78
    opts.addStereoAnnotation = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def draw_molecule_svg(
    mol: Chem.Mol,
    highlight_atoms: List[int],
    atom_colors: Dict[int, Tuple[float, float, float]],
    atom_radii: Dict[int, float],
    highlight_bonds: List[int],
    bond_colors: Dict[int, Tuple[float, float, float]],
    size: Tuple[int, int],
) -> str:
    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.4
    opts.annotationFontScale = 0.78
    opts.addStereoAnnotation = False
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=atom_colors,
        highlightAtomRadii=atom_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def compose_png(base_img: Image.Image) -> Image.Image:
    left_pad = 14
    right_pad = 120
    canvas = Image.new("RGB", (base_img.width + left_pad + right_pad, base_img.height), "white")
    canvas.paste(base_img, (left_pad, 0))
    draw = ImageDraw.Draw(canvas)
    tick_font = _load_font(18, bold=False)

    bar_w = 22
    bar_h = int(base_img.height * 0.72)
    x0 = left_pad + base_img.width + 18
    y0 = int((base_img.height - bar_h) / 2)
    for i in range(bar_h):
        frac = 1.0 - (i / max(1, bar_h - 1))
        rgb = _rgb_float_to_int(_importance_scale(frac))
        draw.line((x0, y0 + i, x0 + bar_w, y0 + i), fill=rgb)
    draw.rectangle((x0, y0, x0 + bar_w, y0 + bar_h), outline=(110, 120, 140), width=1)
    draw.text((x0 + bar_w + 10, y0 - 4), "1.0", fill="black", font=tick_font)
    draw.text((x0 + bar_w + 10, y0 + bar_h - 12), "0.0", fill="black", font=tick_font)
    return canvas


def compose_svg(base_svg: str, base_size: Tuple[int, int]) -> str:
    base_w, base_h = base_size
    left_pad = 14
    right_pad = 120
    total_w = base_w + left_pad + right_pad
    total_h = base_h

    base_svg = base_svg.replace(f"width='{base_w}px'", f"width='{total_w}px'", 1)
    base_svg = base_svg.replace(f"height='{base_h}px'", f"height='{total_h}px'", 1)
    base_svg = base_svg.replace(f"viewBox='0 0 {base_w} {base_h}'", f"viewBox='0 0 {total_w} {total_h}'", 1)
    base_svg = base_svg.replace("</svg>", "", 1)
    base_svg += f"\n<g transform='translate({left_pad},0)'>\n"
    base_svg += "</g>\n"

    bar_w = 22
    bar_h = int(base_h * 0.72)
    x0 = left_pad + base_w + 18
    y0 = int((base_h - bar_h) / 2)
    steps = max(48, bar_h)
    step_h = bar_h / steps
    extra: List[str] = []
    for i in range(steps):
        frac = 1.0 - (i / max(1, steps - 1))
        rgb = _rgb_float_to_int(_importance_scale(frac))
        extra.append(
            f"<rect x='{x0}' y='{y0 + i * step_h:.2f}' width='{bar_w}' height='{step_h + 0.8:.2f}' "
            f"fill='rgb({rgb[0]},{rgb[1]},{rgb[2]})' stroke='none' />"
        )
    extra.append(f"<rect x='{x0}' y='{y0}' width='{bar_w}' height='{bar_h}' fill='none' stroke='#6E788C' stroke-width='1' />")
    extra.append(
        f"<text x='{x0 + bar_w + 10}' y='{y0 + 14}' font-size='18' font-family='Arial, Helvetica, sans-serif' fill='black'>1.0</text>"
    )
    extra.append(
        f"<text x='{x0 + bar_w + 10}' y='{y0 + bar_h}' font-size='18' font-family='Arial, Helvetica, sans-serif' fill='black'>0.0</text>"
    )
    return base_svg + "\n".join(extra) + "\n</svg>"


def main():
    args = parse_args()
    explain_dir = Path(args.explain_dir).resolve()
    out_dir = explain_dir / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    smiles, atom_df = load_inputs(explain_dir)
    mol, norm_scores = prepare_mol(smiles, atom_df, args.score_col)
    highlight_atoms, atom_colors, atom_radii, highlight_bonds, bond_colors = build_highlights(mol, norm_scores)

    base_size = (args.width, args.height)
    base_png = draw_molecule_png(
        mol,
        highlight_atoms,
        atom_colors,
        atom_radii,
        highlight_bonds,
        bond_colors,
        base_size,
    )
    final_png = compose_png(base_png)
    png_path = out_dir / f"{args.out_name}.png"
    final_png.save(png_path)

    base_svg = draw_molecule_svg(
        mol,
        highlight_atoms,
        atom_colors,
        atom_radii,
        highlight_bonds,
        bond_colors,
        base_size,
    )
    svg_path = out_dir / f"{args.out_name}.svg"
    svg_path.write_text(compose_svg(base_svg, base_size), encoding="utf-8")

    norm_df = atom_df.copy()
    vmax = float(norm_df[args.score_col].max()) if args.score_col in norm_df.columns and len(norm_df) else 0.0
    if vmax > 1e-12:
        norm_df["normalized_score"] = norm_df[args.score_col].astype(float) / vmax
    else:
        norm_df["normalized_score"] = 0.0
    norm_df.to_csv(out_dir / f"{args.out_name}_scores.csv", index=False)

    print(str(png_path))
    print(str(svg_path))


if __name__ == "__main__":
    main()
