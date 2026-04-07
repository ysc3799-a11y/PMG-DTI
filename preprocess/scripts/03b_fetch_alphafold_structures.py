import argparse
import json
import os
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import pandas as pd
try:
    from tqdm import tqdm
except Exception:
    tqdm = None


def build_out_filename(pid: str, uid: str) -> str:
    pid = str(pid).strip()
    uid = str(uid).strip()
    return f"{pid}__{uid}.pdb"


def fetch_json(url: str, timeout: int = 20):
    with urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def fetch_bytes(url: str, timeout: int = 30) -> bytes:
    with urlopen(url, timeout=timeout) as r:
        return r.read()


def detect_pdb_url_from_api(uniprot_id: str):
    api = f"https://alphafold.ebi.ac.uk/api/prediction/{uniprot_id}"
    try:
        obj = fetch_json(api)
        if isinstance(obj, list) and obj:
            rec = obj[0]
            if isinstance(rec, dict):
                pdb_url = rec.get("pdbUrl") or rec.get("pdb_url")
                if pdb_url:
                    return str(pdb_url)
    except Exception:
        return ""
    return ""


def candidate_urls(uniprot_id: str):
    u = str(uniprot_id).strip()
    return [
        f"https://alphafold.ebi.ac.uk/files/AF-{u}-F1-model_v4.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{u}-F1-model_v3.pdb",
        f"https://alphafold.ebi.ac.uk/files/AF-{u}-F1-model_v2.pdb",
    ]


def download_pdb(uniprot_id: str, timeout: int = 30):
    url = detect_pdb_url_from_api(uniprot_id)
    tried = []
    if url:
        tried.append(url)
    tried.extend(candidate_urls(uniprot_id))
    seen = set()
    for u in tried:
        if u in seen:
            continue
        seen.add(u)
        try:
            content = fetch_bytes(u, timeout=timeout)
            head = content[:500].decode("utf-8", errors="ignore")
            if ("ATOM" in head) or ("MODEL" in head) or ("HEADER" in head):
                return content, u
        except (HTTPError, URLError, TimeoutError):
            continue
        except Exception:
            continue
    return b"", ""


def main():
    ap = argparse.ArgumentParser(description="Download AlphaFold PDB and build pdb_index.csv")
    ap.add_argument("--map_csv", default="data/interim/human/pairs_with_pid/protein_uniprot_map.csv")
    ap.add_argument("--pid_col", default="Protein_ID")
    ap.add_argument("--uniprot_col", default="UniProt_ID")
    ap.add_argument("--status_col", default="map_status")
    ap.add_argument("--accept_status", default="unique_match")
    ap.add_argument("--out_dir", default="data/raw/human/alphafold_structures")
    ap.add_argument("--out_pdb_index_csv", default="data/raw/human/p2rank_raw/pdb_index.csv")
    ap.add_argument("--out_report_json", default="data/raw/human/alphafold_structures/download_report.json")
    ap.add_argument("--sleep_sec", type=float, default=0.15)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--max_retry", type=int, default=2)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    m = Path(args.map_csv)
    if not m.is_file():
        raise FileNotFoundError(m)
    df = pd.read_csv(m)
    if args.pid_col not in df.columns or args.uniprot_col not in df.columns:
        raise ValueError(f"map_csv must contain {args.pid_col}/{args.uniprot_col}")

    if args.status_col in df.columns and args.accept_status:
        accepted = set(s.strip() for s in str(args.accept_status).split(",") if s.strip())
        df = df[df[args.status_col].astype(str).isin(accepted)].copy()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    n_ok = 0
    n_skip = 0
    n_fail = 0

    iterator = df.iterrows()
    total = int(len(df))
    if tqdm is not None:
        iterator = tqdm(iterator, total=total, desc="AFDB download", dynamic_ncols=True)

    for i, (_, r) in enumerate(iterator, start=1):
        pid = str(r[args.pid_col]).strip()
        uid = str(r[args.uniprot_col]).strip()
        if not pid or not uid:
            continue
        out_pdb = out_dir / build_out_filename(pid=pid, uid=uid)

        if out_pdb.is_file() and (not args.overwrite):
            n_skip += 1
            records.append(
                {
                    "Protein_ID": pid,
                    "UniProt_ID": uid,
                    "pdb_path": str(out_pdb.resolve()),
                    "source_url": "",
                    "status": "exists",
                }
            )
            if tqdm is not None:
                iterator.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
            continue

        content = b""
        src = ""
        for _ in range(int(max(1, args.max_retry))):
            content, src = download_pdb(uid, timeout=int(args.timeout))
            if content:
                break
            time.sleep(float(args.sleep_sec))

        if content:
            out_pdb.write_bytes(content)
            n_ok += 1
            st = "downloaded"
            ppath = str(out_pdb.resolve())
        else:
            n_fail += 1
            st = "failed"
            ppath = ""

        records.append(
            {
                "Protein_ID": pid,
                "UniProt_ID": uid,
                "pdb_path": ppath,
                "source_url": src,
                "status": st,
            }
        )
        if tqdm is not None:
            iterator.set_postfix(ok=n_ok, skip=n_skip, fail=n_fail)
        elif (i % 20 == 0) or (i == total):
            print(f"[PROGRESS] {i}/{total} ok={n_ok} skip={n_skip} fail={n_fail}")
        time.sleep(float(args.sleep_sec))

    out_idx = Path(args.out_pdb_index_csv)
    out_idx.parent.mkdir(parents=True, exist_ok=True)
    idx = pd.DataFrame(records)
    idx.to_csv(out_idx, index=False)

    summary = {
        "n_candidates": int(len(df)),
        "n_ok_download": int(n_ok),
        "n_skip_exists": int(n_skip),
        "n_failed": int(n_fail),
        "coverage_download_or_exists": float((n_ok + n_skip) / max(len(df), 1)),
        "pdb_index_csv": str(out_idx.resolve()),
        "pdb_dir": str(out_dir.resolve()),
    }
    out_rep = Path(args.out_report_json)
    out_rep.parent.mkdir(parents=True, exist_ok=True)
    out_rep.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"[OK] pdb_index: {out_idx}")
    print(f"[OK] report: {out_rep}")
    print("[SUMMARY]", summary)


if __name__ == "__main__":
    main()
