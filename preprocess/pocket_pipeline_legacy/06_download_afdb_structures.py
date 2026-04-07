#!/usr/bin/env python3
import argparse
import os
import time
import json
import pandas as pd
import requests
from tqdm import tqdm

AFDB_ENDPOINTS = [
    # Prefer EBI endpoint (often more stable in CN).
    "https://www.alphafold.ebi.ac.uk/api/prediction/{ac}",
    # Fallback to alphafold.com
    "https://alphafold.com/api/prediction/{ac}",
]


def safe_get_json(url: str, timeout: int = 30, retries: int = 3, retry_sleep_s: float = 2.0):
    for i in range(int(max(1, retries))):
        try:
            r = requests.get(url, timeout=timeout)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                time.sleep(retry_sleep_s * (i + 1))
                continue
            return None
        except Exception:
            time.sleep(retry_sleep_s * (i + 1))
            continue
    return None


def pick_file_urls(item: dict):
    # Try a few common key variants
    keys = [
        ('pdbUrl', 'pdb'), ('pdb_url', 'pdb'),
        ('cifUrl', 'cif'), ('cif_url', 'cif'),
        ('bcifUrl', 'bcif'), ('bcif_url', 'bcif'),
        ('mmcifUrl', 'cif'), ('mmcif_url', 'cif'),
    ]
    out = []
    for k, ext in keys:
        v = item.get(k)
        if v and isinstance(v, str):
            out.append((ext, v))
    return out


def download(url: str, out_path: str, timeout: int = 60, retries: int = 3, retry_sleep_s: float = 2.0):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    last_err = None
    for i in range(int(max(1, retries))):
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                with open(out_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=1024*1024):
                        if chunk:
                            f.write(chunk)
            return
        except Exception as e:
            last_err = e
            time.sleep(retry_sleep_s * (i + 1))
    if last_err is not None:
        raise last_err


def _load_downloaded_ids(log_path: str):
    if not os.path.exists(log_path):
        return set()
    try:
        df = pd.read_csv(log_path)
        if "Protein_ID" in df.columns and "status" in df.columns:
            return set(df.loc[df["status"] == "downloaded", "Protein_ID"].astype(str))
    except Exception:
        pass
    return set()


def _has_existing_files(pid: str, out_dir: str) -> bool:
    if not os.path.isdir(out_dir):
        return False
    prefix = f"{pid}__"
    for name in os.listdir(out_dir):
        if name.startswith(prefix):
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mapping_csv', required=True)
    ap.add_argument('--out_dir', required=True)
    ap.add_argument('--format', choices=['pdb', 'cif', 'bcif'], default='pdb')
    ap.add_argument('--max_models_per_ac', type=int, default=5)
    ap.add_argument('--sleep_s', type=float, default=0.1)
    ap.add_argument('--timeout_s', type=int, default=30)
    ap.add_argument('--download_timeout_s', type=int, default=120)
    ap.add_argument('--retries', type=int, default=3)
    ap.add_argument('--retry_sleep_s', type=float, default=2.0)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    m = pd.read_csv(args.mapping_csv)
    if 'UniProt_AC' not in m.columns or 'Protein_ID' not in m.columns:
        raise SystemExit('mapping_csv must contain Protein_ID and UniProt_AC columns.')

    # one row per Protein_ID
    m = m.dropna(subset=['UniProt_AC']).copy()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, 'afdb_download_log.csv')
    downloaded_ids = _load_downloaded_ids(log_path) if args.resume else set()

    log_rows = []

    for _, row in tqdm(m.iterrows(), total=len(m), desc='AFDB download'):
        pid = str(row['Protein_ID'])
        ac = str(row['UniProt_AC'])

        if args.resume and (pid in downloaded_ids or _has_existing_files(pid, args.out_dir)):
            log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': 'skipped_existing', 'endpoint': ''})
            continue

        data = None
        used_endpoint = None
        for tpl in AFDB_ENDPOINTS:
            url = tpl.format(ac=ac)
            data = safe_get_json(url, timeout=args.timeout_s, retries=args.retries, retry_sleep_s=args.retry_sleep_s)
            if data is not None:
                used_endpoint = url
                break

        if data is None:
            log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': 'no_api_response', 'endpoint': ''})
            continue

        # API may return a list (common) or a dict
        if isinstance(data, dict):
            items = [data]
        elif isinstance(data, list):
            items = data
        else:
            log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': 'unexpected_json', 'endpoint': used_endpoint})
            continue

        saved = 0
        for idx, item in enumerate(items[: args.max_models_per_ac]):
            urls = pick_file_urls(item)
            # choose requested format if present
            chosen = [u for ext, u in urls if ext == args.format]
            if not chosen:
                # fallback: prefer pdb then cif
                chosen = [u for ext, u in urls if ext in ['pdb', 'cif', 'bcif']]
            if not chosen:
                continue

            model_url = chosen[0]
            # derive a filename
            tag = item.get('entryId') or item.get('entry_id') or item.get('uniprotAccession') or item.get('uniprot_accession')
            if not tag:
                tag = ac
            # Some APIs provide model IDs / fragment IDs
            model_id = item.get('modelId') or item.get('model_id') or item.get('fragment') or item.get('fragmentId')
            if model_id:
                fname = f"{pid}__{tag}__{model_id}.{args.format}"
            else:
                fname = f"{pid}__{tag}__{idx+1}.{args.format}"

            out_path = os.path.join(args.out_dir, fname)

            try:
                download(model_url, out_path, timeout=args.download_timeout_s, retries=args.retries, retry_sleep_s=args.retry_sleep_s)
                saved += 1
                log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': 'downloaded', 'file': fname, 'endpoint': used_endpoint})
            except Exception as e:
                log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': f'download_error:{type(e).__name__}', 'file': fname, 'endpoint': used_endpoint})

            time.sleep(args.sleep_s)

        if saved == 0:
            log_rows.append({'Protein_ID': pid, 'UniProt_AC': ac, 'status': 'no_file_url', 'endpoint': used_endpoint})

    pd.DataFrame(log_rows).to_csv(log_path, index=False)
    print(f"[OK] wrote download log: {log_path}")


if __name__ == '__main__':
    main()
