#!/usr/bin/env python3
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--exact_tsv', required=True)
    ap.add_argument('--similarity_tsv', default='')
    ap.add_argument('--out_csv', required=True)
    # thresholds
    ap.add_argument('--high_pident', type=float, default=0.95)
    ap.add_argument('--high_qcov', type=float, default=0.90)
    args = ap.parse_args()

    exact = pd.read_csv(args.exact_tsv, sep='\t')
    # Normalize column names from different exact-mapping scripts.
    if 'Method' in exact.columns and 'method' not in exact.columns:
        exact = exact.rename(columns={'Method': 'method'})
    if 'method' not in exact.columns:
        exact['method'] = 'exact'
    exact['pident'] = 1.0
    exact['qcov'] = 1.0
    exact['tcov'] = 1.0
    exact['evalue'] = 0.0
    exact['bits'] = 0.0

    sim_best = None
    if args.similarity_tsv:
        try:
            sim = pd.read_csv(args.similarity_tsv, sep='\t', header=None,
                              names=['Protein_ID', 'target', 'pident', 'alnlen', 'qcov', 'tcov', 'evalue', 'bits'])
            # target header could be like sp|P12345|... or P12345
            sim['UniProt_AC'] = sim['target'].astype(str).str.split('|').str[1].fillna(sim['target'])
            # pick best hit per Protein_ID: highest bits, then pident, then qcov
            sim = sim.sort_values(['Protein_ID', 'bits', 'pident', 'qcov'], ascending=[True, False, False, False])
            sim_best = sim.drop_duplicates('Protein_ID', keep='first')
            sim_best['method'] = 'mmseqs'
            sim_best = sim_best[['Protein_ID', 'UniProt_AC', 'method', 'pident', 'qcov', 'tcov', 'evalue', 'bits']]
        except FileNotFoundError:
            sim_best = None

    # Combine, exact wins
    merged = exact[['Protein_ID', 'UniProt_AC', 'method', 'pident', 'qcov', 'tcov', 'evalue', 'bits']].copy()
    if sim_best is not None:
        # remove those already exact-mapped
        sim_best = sim_best[~sim_best['Protein_ID'].isin(set(merged['Protein_ID']))]
        merged = pd.concat([merged, sim_best], ignore_index=True)

    # confidence label (for downstream filtering)
    def conf(row):
        if row['method'] == 'exact':
            return 'high'
        return 'high' if (row['pident'] >= args.high_pident and row['qcov'] >= args.high_qcov) else 'low'

    merged['mapping_confidence'] = merged.apply(conf, axis=1)

    merged.to_csv(args.out_csv, index=False)
    print(f"[OK] wrote: {args.out_csv} (n={len(merged)})")


if __name__ == '__main__':
    main()
