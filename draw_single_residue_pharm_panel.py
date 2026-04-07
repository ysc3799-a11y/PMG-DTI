import argparse
import csv
import io
import math
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image, ImageDraw, ImageFont
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D

ROOT = Path(__file__).resolve().parent
for _p in (ROOT, ROOT / "train"):
    _ps = str(_p)
    if _ps not in sys.path:
        sys.path.insert(0, _ps)

from dataset.dataset_canon_real_pocket_seqdist import collate_fn_real_pocket  # noqa: E402
from export_explanation_artifacts import (  # noqa: E402
    attn_pool_weights,
    build_dataset_from_cfg,
    build_model_from_cfg,
    ensure_dir,
    mean_pool_weights,
    move_batch_to_device,
    resolve_sample_position,
    safe_torch_load,
)
from pocket_graph_seqdist import parse_pdb_ca_records  # noqa: E402
from train.train_gated_canon_real_pocket_seqdist_3view_soft_pocket_bias_emaes import (  # noqa: E402
    load_yaml,
    seed_all,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", required=True)
    p.add_argument("--split", default="test", choices=["train", "val", "test"])
    p.add_argument("--row_index", type=int, required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cpu")
    p.add_argument("--title", default="")
    return p.parse_args()


def _display_pharmacophore_label(token_label: str) -> str:
    token = str(token_label)
    type_name, _, anchor = token.partition("|")
    type_map = {
        "PosIonizable": "Pos",
        "NegIonizable": "Neg",
        "Aromatic": "Ar",
        "Donor": "Don",
        "Acceptor": "Acc",
        "Hydrophobe": "Hyd",
        "LumpedHydrophobe": "Hyd",
    }
    shown_type = type_map.get(type_name, type_name)
    return f"{shown_type}-{anchor}" if anchor else shown_type


def _importance_scale(value: float) -> Tuple[float, float, float]:
    value = max(0.0, min(1.0, float(value)))
    low = (0.97, 0.98, 1.00)
    high = (0.00, 0.35, 0.91)
    return tuple(low[i] + (high[i] - low[i]) * value for i in range(3))


def _normalize_importance(linear_norm: float) -> float:
    linear_norm = max(0.0, min(1.0, float(linear_norm)))
    gain = 0.8
    return math.log1p(gain * linear_norm) / math.log1p(gain)


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


def _clear_atom_notes(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.HasProp("atomNote"):
            atom.ClearProp("atomNote")


def prepare_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Failed to parse SMILES: {smiles}")
    AllChem.Compute2DCoords(mol)
    return mol


def build_weighted_mol(smiles: str, link_rows: List[Dict[str, object]]) -> Chem.Mol:
    mol = prepare_mol(smiles)
    _clear_atom_notes(mol)
    for row in link_rows:
        atom_idx = int(row["anchor_atom_index"])
        mol.GetAtomWithIdx(atom_idx).SetProp("atomNote", str(row["label_id"]))
    return mol


def _bond_highlights(
    mol: Chem.Mol, link_rows: List[Dict[str, object]]
) -> Tuple[List[int], Dict[int, Tuple[float, float, float]]]:
    bond_colors: Dict[int, Tuple[float, float, float]] = {}
    bond_scores: Dict[int, float] = {}
    for row in link_rows:
        atom_idx = int(row["anchor_atom_index"])
        color = tuple(row["display_color"])
        weight_norm = float(row["weight_norm"])
        atom = mol.GetAtomWithIdx(atom_idx)
        for bond in atom.GetBonds():
            bond_idx = int(bond.GetIdx())
            if weight_norm >= bond_scores.get(bond_idx, -1.0):
                bond_scores[bond_idx] = weight_norm
                bond_colors[bond_idx] = color
    return sorted(bond_colors.keys()), bond_colors


def _draw_weighted_base_png(
    mol: Chem.Mol, link_rows: List[Dict[str, object]], size: Tuple[int, int]
) -> Image.Image:
    highlight_atoms = [int(row["anchor_atom_index"]) for row in link_rows]
    highlight_colors = {
        int(row["anchor_atom_index"]): tuple(row["display_color"]) for row in link_rows
    }
    highlight_radii = {
        int(row["anchor_atom_index"]): float(row["display_radius"]) for row in link_rows
    }
    highlight_bonds, bond_colors = _bond_highlights(mol, link_rows)

    drawer = rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.6
    opts.addStereoAnnotation = False
    opts.legendFontSize = 24
    opts.annotationFontScale = 1.05
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return Image.open(io.BytesIO(drawer.GetDrawingText())).convert("RGB")


def _draw_weighted_base_svg(
    mol: Chem.Mol, link_rows: List[Dict[str, object]], size: Tuple[int, int]
) -> str:
    highlight_atoms = [int(row["anchor_atom_index"]) for row in link_rows]
    highlight_colors = {
        int(row["anchor_atom_index"]): tuple(row["display_color"]) for row in link_rows
    }
    highlight_radii = {
        int(row["anchor_atom_index"]): float(row["display_radius"]) for row in link_rows
    }
    highlight_bonds, bond_colors = _bond_highlights(mol, link_rows)

    drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    opts = drawer.drawOptions()
    opts.bondLineWidth = 2.6
    opts.addStereoAnnotation = False
    opts.legendFontSize = 24
    opts.annotationFontScale = 1.05
    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightAtomColors=highlight_colors,
        highlightAtomRadii=highlight_radii,
        highlightBonds=highlight_bonds,
        highlightBondColors=bond_colors,
    )
    drawer.FinishDrawing()
    return drawer.GetDrawingText()


def _legend_line(row: Dict[str, object]) -> str:
    return (
        f"{row['residue_label']} ({_display_pharmacophore_label(str(row['token_label']))})"
        f"  w={float(row['display_weight']):.3f}"
    )


def _draw_colorbar_png(
    draw: ImageDraw.ImageDraw,
    x0: int,
    y0: int,
    width: int,
    height: int,
    tick_font: ImageFont.ImageFont,
) -> None:
    for i in range(height):
        frac = 1.0 - (i / max(1, height - 1))
        rgb = _rgb_float_to_int(_importance_scale(frac))
        draw.line((x0, y0 + i, x0 + width, y0 + i), fill=rgb)
    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(110, 120, 140), width=1)
    draw.text((x0 + width + 12, y0 - 4), "1.00", fill="black", font=tick_font)
    draw.text((x0 + width + 12, y0 + height - 12), "0.00", fill="black", font=tick_font)


def _compose_panel_png(base_img: Image.Image, link_rows: List[Dict[str, object]], title: str) -> Image.Image:
    left_w = 108
    right_w = 760
    top_pad = 40 if title else 0
    canvas_h = base_img.height + top_pad
    canvas = Image.new("RGB", (left_w + base_img.width + right_w, canvas_h), "white")
    canvas.paste(base_img, (left_w, top_pad))

    draw = ImageDraw.Draw(canvas)
    title_font = _load_font(26, bold=False)
    list_font = _load_font(28, bold=False)
    tick_font = _load_font(18, bold=False)

    if title:
        title_w = draw.textbbox((0, 0), title, font=title_font)[2]
        draw.text(((canvas.width - title_w) / 2, 8), title, fill="black", font=title_font)

    colorbar_h = min(340, max(220, base_img.height - 180))
    colorbar_w = 22
    colorbar_x = 34
    colorbar_y = top_pad + int((base_img.height - colorbar_h) / 2)
    _draw_colorbar_png(draw, colorbar_x, colorbar_y, colorbar_w, colorbar_h, tick_font)

    right_x = left_w + base_img.width + 36
    line_gap = 52
    list_h = line_gap * len(link_rows)
    y = top_pad + max(72, int((base_img.height - list_h) / 2))
    for row in sorted(link_rows, key=lambda x: int(x["label_id"])):
        rgb = _rgb_float_to_int(tuple(row["display_color"]))
        draw.ellipse((right_x, y + 8, right_x + 20, y + 28), fill=rgb, outline=rgb)
        draw.text((right_x + 34, y), _legend_line(row), fill="black", font=list_font)
        y += line_gap
    return canvas


def _compose_panel_svg(
    base_svg: str, link_rows: List[Dict[str, object]], size: Tuple[int, int], title: str
) -> str:
    base_h = size[1]
    base_w = size[0]
    left_w = 108
    right_w = 760
    top_pad = 40 if title else 0
    total_w = left_w + base_w + right_w
    total_h = base_h + top_pad

    base_svg = base_svg.replace(f"width='{base_w}px'", f"width='{total_w}px'", 1)
    base_svg = base_svg.replace(f"height='{base_h}px'", f"height='{total_h}px'", 1)
    base_svg = base_svg.replace(f"viewBox='0 0 {base_w} {base_h}'", f"viewBox='0 0 {total_w} {total_h}'", 1)
    base_svg = base_svg.replace("<svg", "<svg", 1)
    base_svg = base_svg.replace(">", ">\n", 1)
    base_svg = base_svg.replace("</svg>", f"\n<g transform='translate({left_w},{top_pad})'>", 1)
    base_svg = base_svg.replace("\n<g transform='translate(", "\n<g transform='translate(", 1)
    base_svg = base_svg[:-6] + "\n</g>\n</svg>"

    footer: List[str] = [
        f"<rect x='0' y='0' width='{left_w}' height='{total_h}' fill='white' />",
        f"<rect x='{left_w + base_w}' y='0' width='{right_w}' height='{total_h}' fill='white' />",
    ]

    if title:
        footer.append(
            f"<text x='{total_w / 2:.1f}' y='28' font-size='26' "
            "font-family='Arial, Helvetica, sans-serif' text-anchor='middle' fill='black'>"
            f"{title}</text>"
        )

    colorbar_h = min(340, max(220, base_h - 180))
    colorbar_w = 22
    colorbar_x = 54
    colorbar_y = top_pad + int((base_h - colorbar_h) / 2)
    steps = max(48, colorbar_h)
    step_h = colorbar_h / steps
    for i in range(steps):
        frac = 1.0 - (i / max(1, steps - 1))
        rgb = _rgb_float_to_int(_importance_scale(frac))
        footer.append(
            f"<rect x='{colorbar_x}' y='{colorbar_y + i * step_h:.2f}' width='{colorbar_w}' "
            f"height='{step_h + 0.8:.2f}' fill='rgb({rgb[0]},{rgb[1]},{rgb[2]})' stroke='none' />"
        )
    footer.append(
        f"<rect x='{colorbar_x}' y='{colorbar_y}' width='{colorbar_w}' height='{colorbar_h}' "
        "fill='none' stroke='#6E788C' stroke-width='1' />"
    )
    footer.append(
        f"<text x='{colorbar_x + colorbar_w + 14}' y='{colorbar_y + 14}' font-size='18' "
        "font-family='Arial, Helvetica, sans-serif' fill='black'>1.00</text>"
    )
    footer.append(
        f"<text x='{colorbar_x + colorbar_w + 14}' y='{colorbar_y + colorbar_h}' font-size='18' "
        "font-family='Arial, Helvetica, sans-serif' fill='black'>0.00</text>"
    )

    line_gap = 52
    list_h = line_gap * len(link_rows)
    y = top_pad + max(72, int((base_h - list_h) / 2)) + 2
    right_x = left_w + base_w + 40
    for row in sorted(link_rows, key=lambda x: int(x["label_id"])):
        rgb = _rgb_float_to_int(tuple(row["display_color"]))
        footer.append(
            f"<circle cx='{right_x + 10}' cy='{y - 8}' r='10' fill='rgb({rgb[0]},{rgb[1]},{rgb[2]})' />"
        )
        footer.append(
            f"<text x='{right_x + 38}' y='{y}' font-size='28' "
            "font-family='Arial, Helvetica, sans-serif' fill='black'>"
            f"{_legend_line(row)}</text>"
        )
        y += line_gap
    return base_svg.replace("</svg>", "\n" + "\n".join(footer) + "\n</svg>")


def build_rank_rows(
    model,
    batch: Dict[str, torch.Tensor],
    item_cpu: Dict[str, torch.Tensor],
    id2token: Dict[int, str],
    pdb_path: str,
    top_k: int,
) -> List[Dict[str, object]]:
    with torch.no_grad():
        drug_node_repr, _ = model._run_drug(batch)
        pharm_tokens, _ = model._run_pharm(batch, drug_node_repr)

        esm_tok = batch["esm2_tok"] if "esm2_tok" in batch else batch["esm2_tokens"]
        esm_mask = batch["esm2_mask"]
        esm_tok_proj = model.esm_proj(esm_tok) if getattr(model, "esm_proj", None) is not None else esm_tok

        pocket_bias = None
        if hasattr(model, "_build_soft_pocket_bias"):
            try:
                pocket_bias = model._build_soft_pocket_bias(batch, esm_tok_proj, esm_mask)
            except Exception:
                pocket_bias = None

        if pocket_bias is not None:
            _, attn_prob = model.pp_attn(
                pharm_tokens,
                esm_tok_proj,
                kv_mask=esm_mask,
                additive_bias=pocket_bias,
                return_attn=True,
            )
        else:
            _, attn_prob = model.pp_attn(
                pharm_tokens,
                esm_tok_proj,
                kv_mask=esm_mask,
                return_attn=True,
            )

        if str(getattr(model, "pharm_pool_mode", "attn")).lower() == "attn":
            _, pool_alpha = attn_pool_weights(model.pharm_pool_attn, pharm_tokens, batch["pharm_mask"])
        else:
            pool_alpha = mean_pool_weights(batch["pharm_mask"]).to(dtype=pharm_tokens.dtype)

    valid_token_idx = torch.nonzero(item_cpu["pharm_mask"] > 0, as_tuple=False).view(-1)
    valid_res_idx = torch.nonzero(item_cpu["esm2_mask"] > 0, as_tuple=False).view(-1)
    attn = attn_prob[0, valid_token_idx][:, valid_res_idx].detach().cpu()
    token_weights = pool_alpha[0, valid_token_idx].detach().cpu()
    rec_map = parse_pdb_ca_records(pdb_path)

    candidates: List[Dict[str, object]] = []
    for ti, tok_idx in enumerate(valid_token_idx.tolist()):
        token_id = int(item_cpu["pharm_type"][tok_idx].item())
        token_label = id2token.get(token_id, str(token_id))
        anchor = int(item_cpu["pharm_anchor"][tok_idx].item()) if "pharm_anchor" in item_cpu else -1
        token_weight = float(token_weights[ti].item())
        if anchor < 0:
            continue
        for rj in range(int(attn.shape[1])):
            residue_seq = int(rj + 1)
            rec = rec_map.get(("A", residue_seq), {})
            resname = str(rec.get("resname", "UNK"))
            residue_label = f"{resname}{residue_seq}"
            attention_weight = float(attn[ti, rj].item())
            combined_score = float(attention_weight * token_weight)
            candidates.append(
                {
                    "residue_seq": residue_seq,
                    "residue_label": residue_label,
                    "resname": resname,
                    "token_label": token_label,
                    "anchor_atom_index": int(anchor),
                    "token_weight": token_weight,
                    "attn_prob": attention_weight,
                    "raw_score": combined_score,
                }
            )

    candidates.sort(key=lambda row: float(row["raw_score"]), reverse=True)
    selected: List[Dict[str, object]] = []
    seen_atoms = set()
    seen_residues = set()
    for row in candidates:
        atom_idx = int(row["anchor_atom_index"])
        residue_key = int(row["residue_seq"])
        if atom_idx in seen_atoms or residue_key in seen_residues:
            continue
        selected.append(dict(row))
        seen_atoms.add(atom_idx)
        seen_residues.add(residue_key)
        if len(selected) >= int(top_k):
            break

    if not selected:
        return []

    max_raw = max(float(row["raw_score"]) for row in selected)
    for rank, row in enumerate(selected, start=1):
        linear_norm = float(row["raw_score"]) / max(1e-8, max_raw)
        row["label_id"] = rank
        row["weight_norm"] = _normalize_importance(linear_norm)
        row["display_weight"] = float(row["weight_norm"])
        row["display_color"] = _importance_scale(row["weight_norm"])
        row["display_radius"] = 0.16 + 0.36 * float(row["weight_norm"])
    return selected


def write_link_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fieldnames = [
        "label_id",
        "residue_seq",
        "residue_label",
        "resname",
        "token_label",
        "anchor_atom_index",
        "token_weight",
        "attn_prob",
        "raw_score",
        "display_weight",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main():
    args = parse_args()
    ensure_dir(args.out_dir)
    seed_all(int(args.seed))

    cfg = load_yaml(args.cfg)
    ckpt = safe_torch_load(args.ckpt, map_location="cpu")

    ds, _csv_path, pharm_path, use_pocket_model = build_dataset_from_cfg(cfg, args.split)
    sample_pos = resolve_sample_position(ds, sample_index=0, row_index=args.row_index, protein_id=None)
    ridx = int(ds.indices[sample_pos])
    row = ds.df.iloc[ridx]
    pid = str(row.get(ds.protein_id_col, "")).strip().split("_len=")[0]
    smiles = str(row.get("SMILES", ""))

    model = build_model_from_cfg(cfg, pharm_path, ds)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    item_cpu, y_cpu = ds[sample_pos]
    batch_cpu, y_batch = collate_fn_real_pocket([(item_cpu, y_cpu)], include_pocket=use_pocket_model)
    batch = move_batch_to_device(batch_cpu, args.device)
    model = model.to(args.device)

    pharm_payload = safe_torch_load(pharm_path, map_location="cpu")
    token2id = pharm_payload.get("token2id", {}) if isinstance(pharm_payload, dict) else {}
    id2token = {int(v): str(k) for k, v in token2id.items()} if isinstance(token2id, dict) else {}

    pdb_path = ds.pdb_index.get(pid, "")
    rows = build_rank_rows(model, batch, item_cpu, id2token, pdb_path, top_k=args.top_k)
    if not rows:
        raise RuntimeError("No residue-pharmacophore links could be built for this sample.")

    mol = build_weighted_mol(smiles, rows)
    size = (860, 560)
    base_png = _draw_weighted_base_png(mol, rows, size=size)
    panel_png = _compose_panel_png(base_png, rows, args.title)
    png_path = Path(args.out_dir) / f"row{args.row_index:06d}_residue_pharm_panel.png"
    panel_png.save(png_path)

    base_svg = _draw_weighted_base_svg(mol, rows, size=size)
    svg_path = Path(args.out_dir) / f"row{args.row_index:06d}_residue_pharm_panel.svg"
    svg_path.write_text(_compose_panel_svg(base_svg, rows, size=size, title=args.title), encoding="utf-8")

    csv_path = Path(args.out_dir) / f"row{args.row_index:06d}_residue_pharm_panel_links.csv"
    write_link_csv(csv_path, rows)

    info_path = Path(args.out_dir) / f"row{args.row_index:06d}_residue_pharm_panel_summary.txt"
    info_lines = [
        f"row_index={args.row_index}",
        f"split={args.split}",
        f"sample_pos={sample_pos}",
        f"protein_id={pid}",
        f"smiles={smiles}",
        f"pdb_path={pdb_path}",
        f"label={float(y_batch.view(-1)[0].item())}",
        f"top_k={args.top_k}",
        "",
    ]
    for row in rows:
        info_lines.append(
            f"{row['label_id']}. {row['residue_label']} -> {row['token_label']} "
            f"(atom {row['anchor_atom_index']}, w={float(row['display_weight']):.3f}, "
            f"raw={float(row['raw_score']):.6f})"
        )
    info_path.write_text("\n".join(info_lines), encoding="utf-8")

    print(str(png_path))
    print(str(svg_path))
    print(str(csv_path))


if __name__ == "__main__":
    main()
