import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests


SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
ENTRY_URL = "https://data.rcsb.org/rest/v1/core/entry/{entry_id}"
NONPOLYMER_URL = "https://data.rcsb.org/rest/v1/core/nonpolymer_entity/{entry_id}/{entity_id}"

# Exclude obvious ions, waters, and common crystallization additives.
SKIP_COMP_IDS = {
    "HOH",
    "WAT",
    "DOD",
    "NA",
    "K",
    "CL",
    "CA",
    "MG",
    "MN",
    "ZN",
    "CU",
    "FE",
    "CO",
    "NI",
    "IOD",
    "BR",
    "SO4",
    "PO4",
    "NO3",
    "NH4",
    "ACE",
    "ACT",
    "FMT",
    "EDO",
    "GOL",
    "PEG",
    "PG4",
    "PGE",
    "MPD",
    "MES",
    "TRS",
    "BME",
    "DTT",
    "EOH",
    "IPA",
    "IMD",
    "CIT",
    "MLI",
}


def safe_float(x) -> Optional[float]:
    try:
        if x is None or x == "":
            return None
        return float(x)
    except Exception:
        return None


def post_json(session: requests.Session, url: str, payload: dict, timeout: int, max_retry: int) -> Optional[dict]:
    for attempt in range(max(1, int(max_retry))):
        try:
            r = session.post(url, json=payload, timeout=timeout)
            if r.status_code == 204:
                return {"total_count": 0, "result_set": []}
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt >= max_retry - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def get_json(session: requests.Session, url: str, timeout: int, max_retry: int) -> Optional[dict]:
    for attempt in range(max(1, int(max_retry))):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r.json()
        except Exception:
            if attempt >= max_retry - 1:
                return None
            time.sleep(0.5 * (attempt + 1))
    return None


def build_search_query(uniprot_id: str, min_ligand_mw_kda: float, rows: int) -> dict:
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_polymer_entity_container_identifiers.reference_sequence_identifiers.database_accession",
                        "operator": "exact_match",
                        "value": str(uniprot_id).strip(),
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_entity_count",
                        "operator": "greater",
                        "value": 0,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_entry_info.nonpolymer_molecular_weight_maximum",
                        "operator": "greater_or_equal",
                        "value": float(min_ligand_mw_kda),
                    },
                },
            ],
        },
        "return_type": "entry",
        "request_options": {
            "paginate": {"start": 0, "rows": int(rows)},
            "results_content_type": ["experimental"],
            "sort": [{"sort_by": "rcsb_entry_info.resolution_combined", "direction": "asc"}],
        },
    }


def extract_resolution(entry_obj: dict) -> Optional[float]:
    info = entry_obj.get("rcsb_entry_info", {}) or {}
    vals = info.get("resolution_combined")
    if isinstance(vals, list) and vals:
        nums = [safe_float(v) for v in vals]
        nums = [v for v in nums if v is not None]
        if nums:
            return min(nums)
    return None


def is_candidate_ligand(comp_id: str, name: str, formula_weight_kda: Optional[float], min_ligand_mw_kda: float) -> bool:
    cid = str(comp_id or "").strip().upper()
    nm = str(name or "").strip().upper()
    if not cid:
        return False
    if cid in SKIP_COMP_IDS:
        return False
    if "WATER" in nm or "ION" in nm:
        return False
    if formula_weight_kda is not None and formula_weight_kda < float(min_ligand_mw_kda):
        return False
    return True


def fetch_entry_detail(
    session: requests.Session,
    entry_id: str,
    timeout: int,
    max_retry: int,
    min_ligand_mw_kda: float,
) -> Optional[dict]:
    obj = get_json(session, ENTRY_URL.format(entry_id=entry_id), timeout=timeout, max_retry=max_retry)
    if obj is None:
        return None

    entry_info = obj.get("rcsb_entry_info", {}) or {}
    ids = obj.get("rcsb_entry_container_identifiers", {}) or {}
    nonpoly_ids = ids.get("non_polymer_entity_ids") or []
    lig_all = []
    lig_keep = []

    for entity_id in nonpoly_ids:
        nobj = get_json(
            session,
            NONPOLYMER_URL.format(entry_id=entry_id, entity_id=entity_id),
            timeout=timeout,
            max_retry=max_retry,
        )
        if nobj is None:
            continue
        ent = nobj.get("pdbx_entity_nonpoly", {}) or {}
        rcsb = nobj.get("rcsb_nonpolymer_entity", {}) or {}
        comp_id = str(ent.get("comp_id") or "").strip()
        name = str(rcsb.get("pdbx_description") or ent.get("name") or "").strip()
        fw = safe_float(rcsb.get("formula_weight"))
        rec = {
            "entity_id": str(entity_id),
            "comp_id": comp_id,
            "name": name,
            "formula_weight_kda": fw,
        }
        lig_all.append(rec)
        if is_candidate_ligand(comp_id=comp_id, name=name, formula_weight_kda=fw, min_ligand_mw_kda=min_ligand_mw_kda):
            lig_keep.append(rec)

    return {
        "entry_id": str(entry_id),
        "entry_url": f"https://www.rcsb.org/structure/{entry_id}",
        "title": str((obj.get("struct", {}) or {}).get("title") or "").strip(),
        "method": str(entry_info.get("experimental_method") or "").strip(),
        "resolution": extract_resolution(obj),
        "release_date": str((obj.get("rcsb_accession_info", {}) or {}).get("initial_release_date") or "").strip(),
        "all_nonpolymer": lig_all,
        "candidate_ligands": lig_keep,
        "has_candidate_ligand": bool(lig_keep),
    }


def summarize_best_entry(entry_details: List[dict]) -> Tuple[str, str, Optional[float], str, str, str]:
    if not entry_details:
        return "", "", None, "", "", ""
    # Search results are already sorted by resolution ascending; keep the first entry that survives ligand filtering.
    e = entry_details[0]
    ligand_ids = ";".join(str(x.get("comp_id") or "") for x in e.get("candidate_ligands", []))
    ligand_names = ";".join(str(x.get("name") or "") for x in e.get("candidate_ligands", []))
    return (
        str(e.get("entry_id") or ""),
        str(e.get("entry_url") or ""),
        safe_float(e.get("resolution")),
        str(e.get("method") or ""),
        ligand_ids,
        ligand_names,
    )


def process_one(
    protein_id: str,
    uniprot_id: str,
    timeout: int,
    max_retry: int,
    min_ligand_mw_kda: float,
    top_entries: int,
) -> Tuple[dict, List[dict]]:
    session = requests.Session()
    session.headers.update({"User-Agent": "Pha-DTI-RCSB-Holo-Mapper/1.0"})

    query = build_search_query(uniprot_id=uniprot_id, min_ligand_mw_kda=min_ligand_mw_kda, rows=top_entries)
    resp = post_json(session, SEARCH_URL, query, timeout=timeout, max_retry=max_retry)
    if resp is None:
        return (
            {
                "Protein_ID": protein_id,
                "UniProt_ID": uniprot_id,
                "search_status": "search_failed",
                "candidate_entry_count": 0,
                "reported_entry_rows": 0,
                "has_real_complex_candidate": 0,
                "best_entry_id": "",
                "best_entry_url": "",
                "best_resolution": "",
                "best_method": "",
                "best_ligand_comp_ids": "",
                "best_ligand_names": "",
            },
            [],
        )

    entry_ids = [str(x.get("identifier") or "").strip() for x in (resp.get("result_set") or [])]
    entry_ids = [x for x in entry_ids if x]
    details = []
    for entry_id in entry_ids:
        d = fetch_entry_detail(
            session=session,
            entry_id=entry_id,
            timeout=timeout,
            max_retry=max_retry,
            min_ligand_mw_kda=min_ligand_mw_kda,
        )
        if d is not None and d.get("has_candidate_ligand"):
            details.append(d)

    best_entry_id, best_entry_url, best_resolution, best_method, best_ligand_ids, best_ligand_names = summarize_best_entry(details)

    summary_row = {
        "Protein_ID": protein_id,
        "UniProt_ID": uniprot_id,
        "search_status": "ok",
        "candidate_entry_count": int(resp.get("total_count") or 0),
        "reported_entry_rows": int(len(details)),
        "has_real_complex_candidate": int(bool(details)),
        "best_entry_id": best_entry_id,
        "best_entry_url": best_entry_url,
        "best_resolution": "" if best_resolution is None else float(best_resolution),
        "best_method": best_method,
        "best_ligand_comp_ids": best_ligand_ids,
        "best_ligand_names": best_ligand_names,
    }

    entry_rows = []
    for d in details:
        lig_ids = ";".join(str(x.get("comp_id") or "") for x in d.get("candidate_ligands", []))
        lig_names = ";".join(str(x.get("name") or "") for x in d.get("candidate_ligands", []))
        lig_fw = ";".join(
            "" if x.get("formula_weight_kda") is None else f"{float(x['formula_weight_kda']):.3f}"
            for x in d.get("candidate_ligands", [])
        )
        all_ids = ";".join(str(x.get("comp_id") or "") for x in d.get("all_nonpolymer", []))
        all_names = ";".join(str(x.get("name") or "") for x in d.get("all_nonpolymer", []))
        entry_rows.append(
            {
                "Protein_ID": protein_id,
                "UniProt_ID": uniprot_id,
                "entry_id": str(d.get("entry_id") or ""),
                "entry_url": str(d.get("entry_url") or ""),
                "title": str(d.get("title") or ""),
                "method": str(d.get("method") or ""),
                "resolution": "" if d.get("resolution") is None else float(d.get("resolution")),
                "release_date": str(d.get("release_date") or ""),
                "candidate_ligand_comp_ids": lig_ids,
                "candidate_ligand_names": lig_names,
                "candidate_ligand_formula_weight_kda": lig_fw,
                "all_nonpolymer_comp_ids": all_ids,
                "all_nonpolymer_names": all_names,
            }
        )
    return summary_row, entry_rows


def main():
    ap = argparse.ArgumentParser(
        description="Query RCSB experimental holo structures for dataset proteins mapped to UniProt."
    )
    ap.add_argument("--map_csv", default="data/interim/human/pairs_with_pid/protein_uniprot_map.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--uniprot_col", default="UniProt_ID")
    ap.add_argument("--status_col", default="map_status")
    ap.add_argument("--accept_status", default="unique_match")
    ap.add_argument("--min_ligand_mw_kda", type=float, default=0.12)
    ap.add_argument("--top_entries", type=int, default=10, help="Max entry rows fetched per UniProt after RCSB sorting.")
    ap.add_argument("--max_workers", type=int, default=8)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max_retry", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="Optional debug limit after filtering.")
    ap.add_argument("--out_summary_csv", default="data/interim/human/pairs_with_pid/rcsb_holo_summary.csv")
    ap.add_argument("--out_entries_csv", default="data/interim/human/pairs_with_pid/rcsb_holo_entries.csv")
    ap.add_argument("--out_report_json", default="data/interim/human/pairs_with_pid/rcsb_holo_report.json")
    args = ap.parse_args()

    in_csv = Path(args.map_csv)
    if not in_csv.is_file():
        raise FileNotFoundError(in_csv)

    df = pd.read_csv(in_csv)
    if args.pid_col not in df.columns or args.uniprot_col not in df.columns:
        raise ValueError(f"map_csv must contain {args.pid_col}/{args.uniprot_col}")

    if args.status_col in df.columns and args.accept_status:
        accepted = {x.strip() for x in str(args.accept_status).split(",") if x.strip()}
        df = df[df[args.status_col].astype(str).isin(accepted)].copy()

    df = df[[args.pid_col, args.uniprot_col]].dropna().copy()
    df[args.pid_col] = df[args.pid_col].astype(str).str.strip()
    df[args.uniprot_col] = df[args.uniprot_col].astype(str).str.strip()
    df = df[(df[args.pid_col] != "") & (df[args.uniprot_col] != "")]
    df = df.drop_duplicates(subset=[args.pid_col])
    if int(args.limit) > 0:
        df = df.head(int(args.limit)).copy()

    tasks = [(str(r[args.pid_col]), str(r[args.uniprot_col])) for _, r in df.iterrows()]

    summary_rows: List[dict] = []
    entry_rows: List[dict] = []

    with ThreadPoolExecutor(max_workers=max(1, int(args.max_workers))) as ex:
        futs = {
            ex.submit(
                process_one,
                protein_id=pid,
                uniprot_id=uid,
                timeout=int(args.timeout),
                max_retry=int(args.max_retry),
                min_ligand_mw_kda=float(args.min_ligand_mw_kda),
                top_entries=int(args.top_entries),
            ): (pid, uid)
            for pid, uid in tasks
        }
        n_done = 0
        n_total = len(futs)
        for fut in as_completed(futs):
            summary_row, rows = fut.result()
            summary_rows.append(summary_row)
            entry_rows.extend(rows)
            n_done += 1
            if (n_done % 50 == 0) or (n_done == n_total):
                n_hit = sum(int(x.get("has_real_complex_candidate", 0)) for x in summary_rows)
                print(f"[PROGRESS] {n_done}/{n_total} proteins processed, hits={n_hit}")

    out_summary = Path(args.out_summary_csv)
    out_entries = Path(args.out_entries_csv)
    out_report = Path(args.out_report_json)
    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_entries.parent.mkdir(parents=True, exist_ok=True)
    out_report.parent.mkdir(parents=True, exist_ok=True)

    dsum = pd.DataFrame(summary_rows).sort_values([args.pid_col]).reset_index(drop=True)
    dents = pd.DataFrame(entry_rows)
    if not dents.empty:
        dents = dents.sort_values([args.pid_col, "resolution", "entry_id"], na_position="last").reset_index(drop=True)

    dsum.to_csv(out_summary, index=False)
    dents.to_csv(out_entries, index=False)

    n_hit = int(dsum["has_real_complex_candidate"].sum()) if (not dsum.empty and "has_real_complex_candidate" in dsum.columns) else 0
    report = {
        "n_query_proteins": int(len(dsum)),
        "n_hit_proteins": int(n_hit),
        "protein_hit_rate": float(n_hit / max(len(dsum), 1)),
        "n_entry_rows": int(len(dents)),
        "min_ligand_mw_kda": float(args.min_ligand_mw_kda),
        "top_entries_per_uniprot": int(args.top_entries),
        "summary_csv": str(out_summary.resolve()),
        "entries_csv": str(out_entries.resolve()),
    }
    out_report.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] summary: {out_summary}")
    print(f"[OK] entries: {out_entries}")
    print(f"[OK] report: {out_report}")
    print("[SUMMARY]", report)


if __name__ == "__main__":
    main()
