import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import requests


def slugify(text: str, max_len: int = 48) -> str:
    s = str(text or "").strip()
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("._")
    if not s:
        s = "item"
    return s[:max_len]


def fetch_bytes(url: str, timeout: int, max_retry: int, allow_404: bool = False) -> Optional[bytes]:
    for attempt in range(max(1, int(max_retry))):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "PMG-DTI-RCSB-Asset-Downloader/1.0"})
            if r.status_code == 404 and allow_404:
                return None
            r.raise_for_status()
            return r.content
        except Exception:
            if attempt >= max_retry - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def download_one_file(url: str, out_path: Path, timeout: int, max_retry: int, allow_404: bool = False) -> str:
    data = fetch_bytes(url=url, timeout=timeout, max_retry=max_retry, allow_404=allow_404)
    if data is None:
        return "missing" if allow_404 else "failed"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(data)
    return "ok"


def build_candidate_dir(row: pd.Series) -> str:
    rank = int(row["rank"])
    dataset = slugify(row.get("dataset", "dataset"), max_len=16)
    entry_id = slugify(row.get("entry_id", "entry"), max_len=16).upper()
    comp_id = slugify(row.get("comp_id", "lig"), max_len=16).upper()
    ligand = slugify(row.get("candidate_ligand_name", ""), max_len=36)
    return f"{rank:02d}_{dataset}_{entry_id}_{comp_id}_{ligand}"


def process_one(row: pd.Series, out_dir: Path, timeout: int, max_retry: int, download_pdb: bool) -> Dict:
    rank = int(row["rank"])
    dataset = str(row.get("dataset") or "").strip()
    entry_id = str(row.get("entry_id") or "").strip().upper()
    comp_id = str(row.get("comp_id") or "").strip().upper()
    subdir = out_dir / build_candidate_dir(row)
    subdir.mkdir(parents=True, exist_ok=True)

    cif_url = f"https://files.rcsb.org/download/{entry_id}.cif"
    pdb_url = f"https://files.rcsb.org/download/{entry_id}.pdb"
    ligand_ideal_url = f"https://files.rcsb.org/ligands/view/{comp_id}_ideal.sdf"
    ligand_model_url = f"https://files.rcsb.org/ligands/view/{comp_id}_model.sdf"
    entry_url = f"https://www.rcsb.org/structure/{entry_id}"

    cif_path = subdir / f"{entry_id}.cif"
    pdb_path = subdir / f"{entry_id}.pdb"
    ligand_ideal_path = subdir / f"{comp_id}_ideal.sdf"
    ligand_model_path = subdir / f"{comp_id}_model.sdf"

    cif_status = download_one_file(cif_url, cif_path, timeout=timeout, max_retry=max_retry, allow_404=False)
    pdb_status = "skipped"
    if download_pdb:
        pdb_status = download_one_file(pdb_url, pdb_path, timeout=timeout, max_retry=max_retry, allow_404=True)
    ligand_ideal_status = download_one_file(
        ligand_ideal_url, ligand_ideal_path, timeout=timeout, max_retry=max_retry, allow_404=True
    )
    ligand_model_status = download_one_file(
        ligand_model_url, ligand_model_path, timeout=timeout, max_retry=max_retry, allow_404=True
    )

    meta = {
        "rank": rank,
        "dataset": dataset,
        "pair_row": int(row["pair_row"]),
        "Protein_ID": str(row.get("Protein_ID") or "").strip(),
        "UniProt_ID": str(row.get("UniProt_ID") or "").strip(),
        "entry_id": entry_id,
        "entry_url": entry_url,
        "comp_id": comp_id,
        "candidate_ligand_name": str(row.get("candidate_ligand_name") or "").strip(),
        "SMILES": str(row.get("SMILES") or "").strip(),
        "method": str(row.get("method") or "").strip(),
        "resolution": float(row["resolution"]) if pd.notna(row["resolution"]) else None,
        "match_type": str(row.get("match_type") or "").strip(),
        "title": str(row.get("title") or "").strip(),
        "selection_reason": str(row.get("selection_reason") or "").strip(),
        "recommended_resn_selection": f"resn {comp_id}",
        "recommended_entry_selection": "polymer.protein",
        "download_urls": {
            "cif": cif_url,
            "pdb": pdb_url,
            "ligand_ideal_sdf": ligand_ideal_url,
            "ligand_model_sdf": ligand_model_url,
        },
        "local_files": {
            "cif": cif_path.name if cif_path.is_file() else "",
            "pdb": pdb_path.name if pdb_path.is_file() else "",
            "ligand_ideal_sdf": ligand_ideal_path.name if ligand_ideal_path.is_file() else "",
            "ligand_model_sdf": ligand_model_path.name if ligand_model_path.is_file() else "",
        },
        "download_status": {
            "cif": cif_status,
            "pdb": pdb_status,
            "ligand_ideal_sdf": ligand_ideal_status,
            "ligand_model_sdf": ligand_model_status,
        },
    }
    (subdir / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "rank": rank,
        "dataset": dataset,
        "entry_id": entry_id,
        "comp_id": comp_id,
        "asset_dir": str(subdir.resolve()),
        "cif_status": cif_status,
        "pdb_status": pdb_status,
        "ligand_ideal_sdf_status": ligand_ideal_status,
        "ligand_model_sdf_status": ligand_model_status,
    }


def main():
    ap = argparse.ArgumentParser(description="Download RCSB structures and ligand files for selected visualization candidates.")
    ap.add_argument(
        "--candidates_csv",
        default="data/interim/shared/pairs_with_pid/rcsb_visualization_candidates_top10.csv",
    )
    ap.add_argument(
        "--out_dir",
        default="data/interim/shared/pairs_with_pid/rcsb_visualization_assets",
    )
    ap.add_argument("--max_workers", type=int, default=4)
    ap.add_argument("--timeout", type=int, default=60)
    ap.add_argument("--max_retry", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--no_pdb", action="store_true", help="Skip downloading legacy PDB coordinates.")
    args = ap.parse_args()

    in_csv = Path(args.candidates_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    required = {"rank", "dataset", "pair_row", "Protein_ID", "UniProt_ID", "SMILES", "entry_id", "comp_id"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"candidates_csv missing required columns: {missing}")

    if int(args.limit) > 0:
        df = df.head(int(args.limit)).copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futs = {
            ex.submit(
                process_one,
                row=row,
                out_dir=out_dir,
                timeout=int(args.timeout),
                max_retry=int(args.max_retry),
                download_pdb=(not args.no_pdb),
            ): int(row["rank"])
            for _, row in df.iterrows()
        }
        n_done = 0
        n_total = len(futs)
        for fut in as_completed(futs):
            rows.append(fut.result())
            n_done += 1
            print(f"[PROGRESS] {n_done}/{n_total} candidates downloaded")

    manifest = pd.DataFrame(rows).sort_values(["rank"]).reset_index(drop=True)
    manifest_path = out_dir / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    readme = [
        "# RCSB Visualization Assets",
        "",
        "Each subfolder contains one selected real protein-ligand complex candidate.",
        "",
        "Files per candidate:",
        "- `<entry>.cif`: primary mmCIF structure file from RCSB",
        "- `<entry>.pdb`: legacy PDB coordinate file when available",
        "- `<comp>_ideal.sdf`: idealized ligand coordinates from RCSB chemical component library",
        "- `<comp>_model.sdf`: model ligand coordinates when available",
        "- `metadata.json`: candidate metadata and download status",
        "",
        "Suggested PyMOL ligand selection:",
        "- `resn <comp_id>`",
        "",
        f"Manifest: `{manifest_path.name}`",
    ]
    (out_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")

    report = {
        "n_candidates": int(len(manifest)),
        "n_cif_ok": int((manifest["cif_status"] == "ok").sum()) if not manifest.empty else 0,
        "n_pdb_ok": int((manifest["pdb_status"] == "ok").sum()) if not manifest.empty else 0,
        "n_ligand_ideal_ok": int((manifest["ligand_ideal_sdf_status"] == "ok").sum()) if not manifest.empty else 0,
        "n_ligand_model_ok": int((manifest["ligand_model_sdf_status"] == "ok").sum()) if not manifest.empty else 0,
        "manifest_csv": str(manifest_path.resolve()),
        "assets_dir": str(out_dir.resolve()),
    }
    (out_dir / "report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[OK] manifest:", manifest_path)
    print("[OK] report:", out_dir / "report.json")
    print("[SUMMARY]", report)


if __name__ == "__main__":
    main()
