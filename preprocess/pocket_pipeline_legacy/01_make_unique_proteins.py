#!/usr/bin/env python3
import argparse
import pandas as pd
import hashlib

def stable_id(seq: str) -> str:
    # Stable across runs: md5 of sequence, first 12 hex chars.
    m = hashlib.md5(seq.encode('utf-8')).hexdigest()[:12]
    return f"P{m}".upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs_csv', required=True)
    ap.add_argument('--out_fasta', required=True)
    ap.add_argument('--out_map', required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.pairs_csv)
    if 'Protein' not in df.columns:
        raise SystemExit('CSV must contain a Protein column (amino-acid sequence).')

    # Normalize sequences
    df['Protein'] = df['Protein'].astype(str).str.strip().str.upper()

    # Count occurrences in pairs
    counts = df['Protein'].value_counts().rename_axis('Protein').reset_index(name='count_in_pairs')
    counts['length'] = counts['Protein'].str.len()
    counts['Protein_ID'] = counts['Protein'].apply(stable_id)

    # Write mapping
    counts[['Protein_ID', 'length', 'count_in_pairs', 'Protein']].to_csv(args.out_map, index=False)

    # Write FASTA
    with open(args.out_fasta, 'w') as f:
        for _, row in counts.iterrows():
            pid = row['Protein_ID']
            seq = row['Protein']
            f.write(f">{pid}|len={len(seq)}|pairs={row['count_in_pairs']}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    print(f"[OK] unique proteins: {len(counts)}")
    print(f"[OK] wrote: {args.out_fasta}")
    print(f"[OK] wrote: {args.out_map}")


if __name__ == '__main__':
    main()
