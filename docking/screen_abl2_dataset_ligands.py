import argparse
import csv
import re
import subprocess
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Lipinski


def canonical_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad smiles: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def compute_box_from_residues(residues_csv: Path, padding: float = 8.0):
    df = pd.read_csv(residues_csv)
    xyz = df[["x", "y", "z"]].to_numpy(float)
    center = xyz.mean(axis=0)
    mins = xyz.min(axis=0)
    maxs = xyz.max(axis=0)
    size = (maxs - mins) + float(padding)
    return center, size


def prepare_ligand_sdf(smiles: str, out_sdf: Path):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"bad smiles: {smiles}")
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 20260322
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError("RDKit embedding failed")
    try:
        AllChem.MMFFOptimizeMolecule(mol)
    except Exception:
        pass
    writer = Chem.SDWriter(str(out_sdf))
    writer.write(mol)
    writer.close()


def prepare_ligand_pdbqt(meeko_python: Path, mk_prepare_ligand: Path, in_sdf: Path, out_pdbqt: Path):
    cmd = [str(meeko_python), str(mk_prepare_ligand), "-i", str(in_sdf), "-o", str(out_pdbqt)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)


def dock_with_vina(vina_exe: Path, receptor_pdbqt: Path, ligand_pdbqt: Path, center, size, exhaustiveness: int, num_modes: int, seed: int, out_pdbqt: Path):
    cmd = [
        str(vina_exe),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", str(round(float(center[0]), 3)),
        "--center_y", str(round(float(center[1]), 3)),
        "--center_z", str(round(float(center[2]), 3)),
        "--size_x", str(round(float(size[0]), 3)),
        "--size_y", str(round(float(size[1]), 3)),
        "--size_z", str(round(float(size[2]), 3)),
        "--exhaustiveness", str(int(exhaustiveness)),
        "--num_modes", str(int(num_modes)),
        "--seed", str(int(seed)),
        "--out", str(out_pdbqt),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    best_aff = None
    for line in proc.stdout.splitlines():
        m = re.match(r"^\s*1\s+(-?\d+\.\d+)\s+", line)
        if m:
            best_aff = float(m.group(1))
            break
    if best_aff is None:
        raise RuntimeError("failed to parse vina score")
    return best_aff, proc.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_csv", required=True)
    ap.add_argument("--protein_id", required=True)
    ap.add_argument("--residues_csv", required=True)
    ap.add_argument("--receptor_pdbqt", required=True)
    ap.add_argument("--vina_exe", required=True)
    ap.add_argument("--meeko_python", required=True)
    ap.add_argument("--mk_prepare_ligand", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--padding", type=float, default=8.0)
    ap.add_argument("--exhaustiveness", type=int, default=12)
    ap.add_argument("--num_modes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=20260322)
    args = ap.parse_args()

    full_csv = Path(args.full_csv)
    residues_csv = Path(args.residues_csv)
    receptor_pdbqt = Path(args.receptor_pdbqt)
    vina_exe = Path(args.vina_exe)
    meeko_python = Path(args.meeko_python)
    mk_prepare_ligand = Path(args.mk_prepare_ligand)
    out_dir = Path(args.out_dir)
    lig_dir = out_dir / "ligands"
    pose_dir = out_dir / "poses"
    lig_dir.mkdir(parents=True, exist_ok=True)
    pose_dir.mkdir(parents=True, exist_ok=True)

    center, size = compute_box_from_residues(residues_csv, padding=args.padding)
    df = pd.read_csv(full_csv)
    df = df[(df["Protein_ID"] == args.protein_id) & (pd.to_numeric(df["Y"], errors="coerce") == 1)].copy()
    if df.empty:
        raise RuntimeError("no positive dataset rows for protein_id")

    rows = []
    seen = set()
    for row_index, row in df.iterrows():
        try:
            canon = canonical_smiles(row["SMILES"])
        except Exception:
            continue
        if canon in seen:
            continue
        seen.add(canon)
        lig_id = f"row{int(row_index):06d}"
        sdf_path = lig_dir / f"{lig_id}.sdf"
        pdbqt_path = lig_dir / f"{lig_id}.pdbqt"
        pose_path = pose_dir / f"{lig_id}_docked.pdbqt"
        log_path = pose_dir / f"{lig_id}.log"
        prepare_ligand_sdf(canon, sdf_path)
        prepare_ligand_pdbqt(meeko_python, mk_prepare_ligand, sdf_path, pdbqt_path)
        best_aff, stdout = dock_with_vina(
            vina_exe=vina_exe,
            receptor_pdbqt=receptor_pdbqt,
            ligand_pdbqt=pdbqt_path,
            center=center,
            size=size,
            exhaustiveness=args.exhaustiveness,
            num_modes=args.num_modes,
            seed=args.seed,
            out_pdbqt=pose_path,
        )
        log_path.write_text(stdout, encoding="utf-8")
        mol = Chem.MolFromSmiles(canon)
        rows.append(
            {
                "row_index": int(row_index),
                "canonical_smiles": canon,
                "mol_wt": round(Descriptors.MolWt(mol), 2),
                "logp": round(Descriptors.MolLogP(mol), 2),
                "rot_bonds": int(Lipinski.NumRotatableBonds(mol)),
                "best_affinity_kcal_mol": best_aff,
                "ligand_sdf": str(sdf_path.resolve()),
                "ligand_pdbqt": str(pdbqt_path.resolve()),
                "docked_pose_pdbqt": str(pose_path.resolve()),
                "vina_log": str(log_path.resolve()),
            }
        )

    out_csv = out_dir / "screen_results.csv"
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda x: x["best_affinity_kcal_mol"]))

    summary = out_dir / "screen_summary.txt"
    summary.write_text(
        "\n".join(
            [
                f"protein_id={args.protein_id}",
                f"residues_csv={residues_csv.resolve()}",
                "center=" + ",".join(f"{x:.3f}" for x in center),
                "size=" + ",".join(f"{x:.3f}" for x in size),
                f"n_unique_positive_ligands={len(rows)}",
                f"results_csv={out_csv.resolve()}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[OK] {out_csv.resolve()}")
    print(f"[OK] {summary.resolve()}")


if __name__ == "__main__":
    main()
