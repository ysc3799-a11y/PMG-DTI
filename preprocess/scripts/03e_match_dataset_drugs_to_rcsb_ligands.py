import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests

try:
    from rdkit import Chem
    from rdkit import RDLogger
    from rdkit.Chem import inchi
except Exception as e:
    raise RuntimeError(
        "RDKit is required for exact ligand matching. Run this script in the conda env that has rdkit installed."
    ) from e

RDLogger.DisableLog("rdApp.*")


CHEMCOMP_URL = "https://data.rcsb.org/rest/v1/core/chemcomp/{comp_id}"


def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def get_json(url: str, timeout: int, max_retry: int) -> Optional[dict]:
    for attempt in range(max(1, int(max_retry))):
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": "Pha-DTI-RCSB-Ligand-Matcher/1.0"})
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt >= max_retry - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def canonicalize_smiles(smiles: str) -> Dict[str, str]:
    s = str(smiles or "").strip()
    if not s:
        return {
            "raw_smiles": "",
            "smiles_iso": "",
            "smiles_flat": "",
            "inchikey": "",
            "status": "empty",
        }
    mol = Chem.MolFromSmiles(s)
    if mol is None:
        return {
            "raw_smiles": s,
            "smiles_iso": "",
            "smiles_flat": "",
            "inchikey": "",
            "status": "parse_failed",
        }
    iso = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    flat = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=False)
    ik = ""
    try:
        ik = inchi.MolToInchiKey(mol)
    except Exception:
        ik = ""
    return {
        "raw_smiles": s,
        "smiles_iso": iso,
        "smiles_flat": flat,
        "inchikey": ik,
        "status": "ok",
    }


def choose_descriptor_rows(rows: List[dict]) -> Tuple[str, str, str]:
    can_smi = ""
    any_smi = ""
    ik = ""
    for r in rows:
        typ = str(r.get("type") or "").strip().upper()
        desc = str(r.get("descriptor") or "").strip()
        if not desc:
            continue
        if typ == "INCHIKEY" and (not ik):
            ik = desc
        elif typ == "SMILES_CANONICAL" and (not can_smi):
            can_smi = desc
        elif typ == "SMILES" and (not any_smi):
            any_smi = desc
    return can_smi, any_smi, ik


def fetch_one_chemcomp(comp_id: str, timeout: int, max_retry: int) -> dict:
    comp_id = str(comp_id or "").strip().upper()
    if not comp_id:
        return {
            "comp_id": "",
            "fetch_status": "empty",
            "pdb_name": "",
            "formula_weight": "",
            "inchi_pdb": "",
            "inchikey_pdb": "",
            "descriptor_smiles_canonical": "",
            "descriptor_smiles_any": "",
            "smiles_iso": "",
            "smiles_flat": "",
            "inchikey_rdkit": "",
        }
    obj = get_json(CHEMCOMP_URL.format(comp_id=comp_id), timeout=timeout, max_retry=max_retry)
    if obj is None:
        return {
            "comp_id": comp_id,
            "fetch_status": "fetch_failed",
            "pdb_name": "",
            "formula_weight": "",
            "inchi_pdb": "",
            "inchikey_pdb": "",
            "descriptor_smiles_canonical": "",
            "descriptor_smiles_any": "",
            "smiles_iso": "",
            "smiles_flat": "",
            "inchikey_rdkit": "",
        }

    chem_comp = obj.get("chem_comp", {}) or {}
    desc_rows = obj.get("pdbx_chem_comp_descriptor", []) or []
    can_smi, any_smi, ik = choose_descriptor_rows(desc_rows)

    inchi_pdb = ""
    for r in desc_rows:
        if str(r.get("type") or "").strip().upper() == "INCHI":
            inchi_pdb = str(r.get("descriptor") or "").strip()
            if inchi_pdb:
                break

    best_smiles = can_smi or any_smi
    canon = canonicalize_smiles(best_smiles) if best_smiles else canonicalize_smiles("")
    return {
        "comp_id": comp_id,
        "fetch_status": "ok",
        "pdb_name": str(chem_comp.get("name") or "").strip(),
        "formula_weight": safe_float(chem_comp.get("formula_weight")),
        "inchi_pdb": inchi_pdb,
        "inchikey_pdb": ik,
        "descriptor_smiles_canonical": can_smi,
        "descriptor_smiles_any": any_smi,
        "smiles_iso": canon["smiles_iso"],
        "smiles_flat": canon["smiles_flat"],
        "inchikey_rdkit": canon["inchikey"],
    }


def ensure_chemcomp_cache(
    comp_ids: List[str],
    cache_csv: Path,
    timeout: int,
    max_retry: int,
    max_workers: int,
) -> pd.DataFrame:
    cache_cols = [
        "comp_id",
        "fetch_status",
        "pdb_name",
        "formula_weight",
        "inchi_pdb",
        "inchikey_pdb",
        "descriptor_smiles_canonical",
        "descriptor_smiles_any",
        "smiles_iso",
        "smiles_flat",
        "inchikey_rdkit",
    ]
    if cache_csv.is_file():
        cache_df = pd.read_csv(cache_csv)
    else:
        cache_df = pd.DataFrame(columns=cache_cols)

    if cache_df.empty:
        known = set()
    else:
        cache_df["comp_id"] = cache_df["comp_id"].astype(str).str.upper().str.strip()
        known = set(cache_df["comp_id"])

    missing = sorted({str(x).upper().strip() for x in comp_ids if str(x).strip()} - known)
    if missing:
        rows = []
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as ex:
            futs = {
                ex.submit(fetch_one_chemcomp, comp_id=cid, timeout=timeout, max_retry=max_retry): cid
                for cid in missing
            }
            n_done = 0
            n_total = len(futs)
            for fut in as_completed(futs):
                rows.append(fut.result())
                n_done += 1
                if (n_done % 100 == 0) or (n_done == n_total):
                    print(f"[CHEMCOMP] {n_done}/{n_total} fetched")
        if rows:
            add_df = pd.DataFrame(rows)
            if cache_df.empty:
                cache_df = add_df.copy()
            else:
                cache_df = pd.concat([cache_df, add_df], ignore_index=True)
            cache_df = cache_df.drop_duplicates(subset=["comp_id"], keep="last").reset_index(drop=True)
            cache_csv.parent.mkdir(parents=True, exist_ok=True)
            cache_df.to_csv(cache_csv, index=False)
    return cache_df


def build_smiles_cache(smiles_list: List[str]) -> pd.DataFrame:
    rows = []
    for s in sorted(set(str(x).strip() for x in smiles_list if str(x).strip())):
        rec = canonicalize_smiles(s)
        rec["SMILES"] = s
        rows.append(rec)
    return pd.DataFrame(rows)


def explode_entry_ligands(entries_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in entries_df.iterrows():
        comp_ids = [x.strip() for x in str(r.get("candidate_ligand_comp_ids") or "").split(";") if x.strip()]
        names = [x.strip() for x in str(r.get("candidate_ligand_names") or "").split(";")]
        fws = [x.strip() for x in str(r.get("candidate_ligand_formula_weight_kda") or "").split(";")]
        n = max(len(comp_ids), len(names), len(fws))
        if n == 0:
            continue
        while len(names) < n:
            names.append("")
        while len(fws) < n:
            fws.append("")
        while len(comp_ids) < n:
            comp_ids.append("")
        for cid, nm, fw in zip(comp_ids, names, fws):
            cid = str(cid).strip().upper()
            if not cid:
                continue
            rows.append(
                {
                    "Protein_ID": str(r.get("Protein_ID") or "").strip(),
                    "UniProt_ID": str(r.get("UniProt_ID") or "").strip(),
                    "entry_id": str(r.get("entry_id") or "").strip(),
                    "entry_url": str(r.get("entry_url") or "").strip(),
                    "title": str(r.get("title") or "").strip(),
                    "method": str(r.get("method") or "").strip(),
                    "resolution": safe_float(r.get("resolution")),
                    "release_date": str(r.get("release_date") or "").strip(),
                    "comp_id": cid,
                    "candidate_ligand_name": str(nm).strip(),
                    "candidate_ligand_formula_weight_kda": safe_float(fw),
                }
            )
    return pd.DataFrame(rows)


def choose_match_type(drug_row: dict, lig_row: dict) -> str:
    d_ik = str(drug_row.get("inchikey") or "").strip()
    l_ik = str(lig_row.get("inchikey_rdkit") or lig_row.get("inchikey_pdb") or "").strip()
    if d_ik and l_ik and d_ik == l_ik:
        return "inchikey"
    d_iso = str(drug_row.get("smiles_iso") or "").strip()
    l_iso = str(lig_row.get("smiles_iso") or "").strip()
    if d_iso and l_iso and d_iso == l_iso:
        return "smiles_iso"
    d_flat = str(drug_row.get("smiles_flat") or "").strip()
    l_flat = str(lig_row.get("smiles_flat") or "").strip()
    if d_flat and l_flat and d_flat == l_flat:
        return "smiles_flat"
    return ""


def match_rank(match_type: str) -> int:
    order = {"inchikey": 0, "smiles_iso": 1, "smiles_flat": 2}
    return order.get(str(match_type), 99)


def main():
    ap = argparse.ArgumentParser(description="Match dataset SMILES to RCSB co-crystal ligands for exact pair-level complex hits.")
    ap.add_argument("--pairs_csv", default="data/interim/human/pairs_with_pid/full_with_pid.csv")
    ap.add_argument("--entries_csv", default="data/interim/human/pairs_with_pid/rcsb_holo_entries.csv")
    ap.add_argument("--map_csv", default="data/interim/human/pairs_with_pid/protein_uniprot_map.csv")
    ap.add_argument("--smiles_col", default="SMILES")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--label_col", default="Y")
    ap.add_argument("--positive_value", default="1")
    ap.add_argument("--all_rows", action="store_true", help="Disable positive-only filtering.")
    ap.add_argument("--limit", type=int, default=0, help="Optional debug limit after filtering.")
    ap.add_argument("--cache_csv", default="data/interim/shared/pairs_with_pid/rcsb_chemcomp_cache.csv")
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max_retry", type=int, default=3)
    ap.add_argument("--max_workers", type=int, default=12)
    ap.add_argument("--out_matches_csv", default="data/interim/human/pairs_with_pid/rcsb_exact_pair_matches.csv")
    ap.add_argument("--out_best_csv", default="data/interim/human/pairs_with_pid/rcsb_exact_pair_best.csv")
    ap.add_argument("--out_report_json", default="data/interim/human/pairs_with_pid/rcsb_exact_pair_report.json")
    args = ap.parse_args()

    pairs_csv = Path(args.pairs_csv)
    entries_csv = Path(args.entries_csv)
    map_csv = Path(args.map_csv)
    cache_csv = Path(args.cache_csv)
    if not pairs_csv.is_file():
        raise FileNotFoundError(pairs_csv)
    if not entries_csv.is_file():
        raise FileNotFoundError(entries_csv)
    if not map_csv.is_file():
        raise FileNotFoundError(map_csv)

    pairs = pd.read_csv(pairs_csv).reset_index().rename(columns={"index": "pair_row"})
    if args.smiles_col not in pairs.columns or args.pid_col not in pairs.columns:
        raise ValueError(f"pairs_csv must contain {args.smiles_col}/{args.pid_col}")
    if (not args.all_rows) and args.label_col in pairs.columns:
        target = str(args.positive_value)
        pairs = pairs[pairs[args.label_col].astype(str) == target].copy()
    if int(args.limit) > 0:
        pairs = pairs.head(int(args.limit)).copy()

    entries = pd.read_csv(entries_csv)
    protein_map = pd.read_csv(map_csv)[[args.pid_col, "UniProt_ID"]].drop_duplicates()

    ligands = explode_entry_ligands(entries)
    if ligands.empty:
        raise RuntimeError("No ligand rows found in entries_csv.")

    comp_ids = sorted(set(ligands["comp_id"].astype(str)))
    chemcomp = ensure_chemcomp_cache(
        comp_ids=comp_ids,
        cache_csv=cache_csv,
        timeout=int(args.timeout),
        max_retry=int(args.max_retry),
        max_workers=int(args.max_workers),
    )
    chemcomp["comp_id"] = chemcomp["comp_id"].astype(str).str.upper().str.strip()

    ligands = ligands.merge(chemcomp, on="comp_id", how="left")
    ligands = ligands[ligands["fetch_status"].astype(str) == "ok"].copy()

    smiles_cache = build_smiles_cache(pairs[args.smiles_col].astype(str).tolist())
    pairs = pairs.merge(smiles_cache[["SMILES", "smiles_iso", "smiles_flat", "inchikey", "status"]], on="SMILES", how="left")
    pairs = pairs[pairs["status"].astype(str) == "ok"].copy()
    pairs = pairs.merge(protein_map, on=args.pid_col, how="left")

    lig_by_pid: Dict[str, List[dict]] = {}
    for _, r in ligands.iterrows():
        pid = str(r.get("Protein_ID") or "").strip()
        if not pid:
            continue
        lig_by_pid.setdefault(pid, []).append(r.to_dict())

    match_rows = []
    for _, pr in pairs.iterrows():
        pid = str(pr.get(args.pid_col) or "").strip()
        if not pid or pid not in lig_by_pid:
            continue
        drug_rec = pr.to_dict()
        for lig in lig_by_pid[pid]:
            mtype = choose_match_type(drug_row=drug_rec, lig_row=lig)
            if not mtype:
                continue
            match_rows.append(
                {
                    "pair_row": int(pr["pair_row"]),
                    args.pid_col: pid,
                    "UniProt_ID": str(pr.get("UniProt_ID") or "").strip(),
                    args.smiles_col: str(pr.get(args.smiles_col) or "").strip(),
                    "drug_smiles_iso": str(pr.get("smiles_iso") or "").strip(),
                    "drug_smiles_flat": str(pr.get("smiles_flat") or "").strip(),
                    "drug_inchikey": str(pr.get("inchikey") or "").strip(),
                    "match_type": mtype,
                    "entry_id": str(lig.get("entry_id") or "").strip(),
                    "entry_url": str(lig.get("entry_url") or "").strip(),
                    "title": str(lig.get("title") or "").strip(),
                    "method": str(lig.get("method") or "").strip(),
                    "resolution": safe_float(lig.get("resolution")),
                    "release_date": str(lig.get("release_date") or "").strip(),
                    "comp_id": str(lig.get("comp_id") or "").strip(),
                    "candidate_ligand_name": str(lig.get("candidate_ligand_name") or lig.get("pdb_name") or "").strip(),
                    "ligand_smiles_iso": str(lig.get("smiles_iso") or "").strip(),
                    "ligand_smiles_flat": str(lig.get("smiles_flat") or "").strip(),
                    "ligand_inchikey": str(lig.get("inchikey_rdkit") or lig.get("inchikey_pdb") or "").strip(),
                    "label": pr.get(args.label_col, ""),
                }
            )

    dmatch = pd.DataFrame(match_rows)
    if dmatch.empty:
        print("[WARN] no exact pair matches found")
        dbest = dmatch.copy()
    else:
        dmatch["match_rank"] = dmatch["match_type"].map(match_rank)
        dmatch = dmatch.sort_values(
            ["pair_row", "match_rank", "resolution", "entry_id", "comp_id"],
            ascending=[True, True, True, True, True],
            na_position="last",
        ).reset_index(drop=True)
        dbest = dmatch.groupby("pair_row", as_index=False).first()
        dmatch = dmatch.drop(columns=["match_rank"])
        dbest = dbest.drop(columns=["match_rank"])

    out_matches = Path(args.out_matches_csv)
    out_best = Path(args.out_best_csv)
    out_report = Path(args.out_report_json)
    out_matches.parent.mkdir(parents=True, exist_ok=True)
    out_best.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    dmatch.to_csv(out_matches, index=False)
    dbest.to_csv(out_best, index=False)

    rep = {
        "n_pairs_considered": int(len(pairs)),
        "n_pairs_matched": int(dbest["pair_row"].nunique()) if not dbest.empty else 0,
        "pair_match_rate": float((dbest["pair_row"].nunique() if not dbest.empty else 0) / max(len(pairs), 1)),
        "n_match_rows": int(len(dmatch)),
        "n_unique_matched_proteins": int(dbest[args.pid_col].nunique()) if not dbest.empty else 0,
        "n_unique_matched_entries": int(dbest["entry_id"].nunique()) if not dbest.empty else 0,
        "cache_csv": str(cache_csv.resolve()),
        "matches_csv": str(out_matches.resolve()),
        "best_csv": str(out_best.resolve()),
    }
    out_report.write_text(json.dumps(rep, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] cache: {cache_csv}")
    print(f"[OK] matches: {out_matches}")
    print(f"[OK] best: {out_best}")
    print(f"[OK] report: {out_report}")
    print("[SUMMARY]", rep)


if __name__ == "__main__":
    main()
