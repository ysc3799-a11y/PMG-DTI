import argparse
import json
from pathlib import Path

import pandas as pd


def ligand_spec(comp_id: str) -> str:
    comp_id = str(comp_id or "").strip()
    return f'::name="{comp_id}"'


def quote_path(p: Path) -> str:
    return f'"{p.name}"'


def structure_filename(meta: dict) -> str:
    files = meta.get("local_files", {}) or {}
    cif = str(files.get("cif") or "").strip()
    pdb = str(files.get("pdb") or "").strip()
    return cif or pdb


def overview_script(meta: dict) -> str:
    lig = ligand_spec(meta["comp_id"])
    structure_file = structure_filename(meta)
    return "\n".join(
        [
            "# ChimeraX overview script",
            "close all",
            "set bgColor white",
            f"open {quote_path(Path(structure_file))}",
            "hide atoms",
            "hide pbonds",
            "show protein cartoons",
            "cartoon style protein width 2.2 thick 0.35",
            "color protein light slate gray cartoons",
            f"show {lig} atoms",
            f"style {lig} stick",
            f"size {lig} stickRadius 0.24",
            f"color {lig} orange red atoms",
            "view",
            "",
        ]
    )


def publication_global_script(meta: dict) -> str:
    lig = ligand_spec(meta["comp_id"])
    structure_file = structure_filename(meta)
    near_surface = f'protein & ({lig} :<5)'
    return "\n".join(
        [
            "# ChimeraX publication-style global panel",
            "close all",
            "set bgColor white",
            f"open {quote_path(Path(structure_file))}",
            "hide atoms",
            "hide pbonds",
            "show protein cartoons",
            "cartoon style protein width 2.3 thick 0.34",
            "color protein #E76AB3 cartoons",
            f"surface {near_surface} color #A6F4A1 transparency 58 visiblePatches 1",
            f"show {lig} atoms",
            f"style {lig} stick",
            f"size {lig} stickRadius 0.28",
            f"color {lig} #20D7E8 atoms",
            f"view {lig}",
            "",
        ]
    )


def closeup_script(meta: dict) -> str:
    lig = ligand_spec(meta["comp_id"])
    structure_file = structure_filename(meta)
    near_spec = f'protein & ({lig} :<4.5)'
    return "\n".join(
        [
            "# ChimeraX close-up and H-bond script",
            "close all",
            "set bgColor white",
            f"open {quote_path(Path(structure_file))}",
            "hide atoms",
            "hide pbonds",
            "show protein cartoons",
            "cartoon style protein width 1.8 thick 0.25",
            "color protein gray cartoons",
            f"show {lig} atoms",
            f"style {lig} stick",
            f"size {lig} stickRadius 0.26",
            f"color {lig} orange red atoms",
            f"show {near_spec} atoms",
            f"style {near_spec} stick",
            f"size {near_spec} stickRadius 0.18",
            f"color {near_spec} dodger blue atoms",
            "hbonds delete",
            f"hbonds {lig} restrict protein reveal true showDist true color gold radius 0.08 dashes 6 name ligand_hbonds",
            f"view {lig}",
            "",
        ]
    )


def publication_zoom_script(meta: dict) -> str:
    lig = ligand_spec(meta["comp_id"])
    structure_file = structure_filename(meta)
    near_atoms = f'protein & ({lig} :<4.5)'
    near_surface = f'protein & ({lig} :<5)'
    return "\n".join(
        [
            "# ChimeraX publication-style zoom panel",
            "close all",
            "set bgColor white",
            f"open {quote_path(Path(structure_file))}",
            "hide atoms",
            "hide pbonds",
            "show protein cartoons",
            "cartoon style protein width 2.0 thick 0.28",
            "color protein #E76AB3 cartoons",
            f"surface {near_surface} color #A6F4A1 transparency 63 visiblePatches 1",
            f"show {lig} atoms",
            f"style {lig} stick",
            f"size {lig} stickRadius 0.30",
            f"color {lig} #20D7E8 atoms",
            f"show {near_atoms} atoms",
            f"style {near_atoms} stick",
            f"size {near_atoms} stickRadius 0.18",
            f"color {near_atoms} #D7A7C8 atoms",
            "hbonds delete",
            f"hbonds {lig} restrict protein reveal true showDist true color #FF7A59 radius 0.09 dashes 7 name ligand_hbonds",
            f"view {lig}",
            "",
        ]
    )


def candidate_readme(meta: dict) -> str:
    structure_file = structure_filename(meta)
    lig = ligand_spec(meta["comp_id"])
    return "\n".join(
        [
            f"# {meta['entry_id']} / {meta['comp_id']}",
            "",
            f"- Dataset: `{meta['dataset']}`",
            f"- pair_row: `{meta['pair_row']}`",
            f"- Protein_ID: `{meta['Protein_ID']}`",
            f"- UniProt_ID: `{meta['UniProt_ID']}`",
            f"- Ligand: `{meta['candidate_ligand_name']}`",
            f"- Method: `{meta['method']}`",
            f"- Resolution: `{meta['resolution']}`",
            f"- Match type: `{meta['match_type']}`",
            f"- Structure file: `{structure_file}`",
            f"- Ligand selection in ChimeraX: `{lig}`",
            "",
            "Scripts:",
            "- `01_overview.cxc`: whole-structure cartoon + ligand sticks",
            "- `03_closeup_hbonds.cxc`: close-up, nearby residues, hydrogen bonds",
            "- `04_publication_global.cxc`: publication-style global panel similar to your example panel (a)",
            "- `05_publication_zoom.cxc`: publication-style zoom panel similar to your example panel (b)",
            "",
            "Notes:",
            "- If an entry contains multiple copies of the same ligand residue name, these scripts will select all of them.",
            "- For final publication figures, manually keep the ligand copy you want and hide the others if necessary.",
            "- The dashed zoom box and panel labels (a)/(b) should be added during figure assembly, not inside ChimeraX.",
            "",
        ]
    )


def main():
    ap = argparse.ArgumentParser(description="Generate ChimeraX command scripts for downloaded RCSB visualization assets.")
    ap.add_argument(
        "--assets_dir",
        default="data/interim/shared/pairs_with_pid/rcsb_visualization_assets",
    )
    args = ap.parse_args()

    assets_dir = Path(args.assets_dir)
    manifest_csv = assets_dir / "manifest.csv"
    if not assets_dir.is_dir():
        raise FileNotFoundError(assets_dir)
    if not manifest_csv.is_file():
        raise FileNotFoundError(manifest_csv)

    manifest = pd.read_csv(manifest_csv)
    generated_rows = []
    for _, row in manifest.iterrows():
        subdir = Path(str(row["asset_dir"]))
        meta_path = subdir / "metadata.json"
        if not meta_path.is_file():
            print(f"[WARN] missing metadata: {meta_path}")
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        structure_file = structure_filename(meta)
        if not structure_file:
            print(f"[WARN] no structure file recorded in metadata: {meta_path}")
            continue

        overview_path = subdir / "01_overview.cxc"
        closeup_path = subdir / "03_closeup_hbonds.cxc"
        pub_global_path = subdir / "04_publication_global.cxc"
        pub_zoom_path = subdir / "05_publication_zoom.cxc"
        readme_path = subdir / "CHIMERAX_README.md"
        legacy_pocket_path = subdir / "02_pocket_surface.cxc"

        overview_path.write_text(overview_script(meta), encoding="utf-8")
        closeup_path.write_text(closeup_script(meta), encoding="utf-8")
        pub_global_path.write_text(publication_global_script(meta), encoding="utf-8")
        pub_zoom_path.write_text(publication_zoom_script(meta), encoding="utf-8")
        readme_path.write_text(candidate_readme(meta), encoding="utf-8")
        if legacy_pocket_path.is_file():
            legacy_pocket_path.unlink()

        generated_rows.append(
            {
                "rank": meta["rank"],
                "dataset": meta["dataset"],
                "entry_id": meta["entry_id"],
                "comp_id": meta["comp_id"],
                "asset_dir": str(subdir.resolve()),
                "overview_script": str(overview_path.resolve()),
                "closeup_script": str(closeup_path.resolve()),
                "publication_global_script": str(pub_global_path.resolve()),
                "publication_zoom_script": str(pub_zoom_path.resolve()),
            }
        )

    out_manifest = assets_dir / "chimerax_manifest.csv"
    pd.DataFrame(generated_rows).sort_values(["rank"]).to_csv(out_manifest, index=False)

    root_readme = "\n".join(
        [
            "# ChimeraX Scripts",
            "",
            "This directory contains downloaded RCSB real-complex assets plus generated ChimeraX command files.",
            "",
            "Per candidate directory:",
            "- `01_overview.cxc`",
            "- `03_closeup_hbonds.cxc`",
            "- `04_publication_global.cxc`",
            "- `05_publication_zoom.cxc`",
            "- `CHIMERAX_README.md`",
            "",
            "Usage examples:",
            "- Open a script inside ChimeraX with `File -> Open...`",
            "- Or from terminal: `ChimeraX path\\\\to\\\\01_overview.cxc`",
            "",
            "Main manifest:",
            f"- `{out_manifest.name}`",
            "",
            "For a figure like your example, render `04_publication_global.cxc` and `05_publication_zoom.cxc` separately, then assemble the dashed inset and panel labels outside ChimeraX.",
        ]
    )
    (assets_dir / "CHIMERAX_ROOT_README.md").write_text(root_readme, encoding="utf-8")

    print(f"[OK] generated: {out_manifest}")
    print(f"[OK] root readme: {assets_dir / 'CHIMERAX_ROOT_README.md'}")


if __name__ == "__main__":
    main()
