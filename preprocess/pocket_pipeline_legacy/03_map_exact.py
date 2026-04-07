#!/usr/bin/env python3
import argparse
import gzip


def read_fasta_any(path: str):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as f:
        header = None
        seq_chunks = []
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if header is not None:
                    yield header, ''.join(seq_chunks)
                header = line[1:].split()[0]
                seq_chunks = []
            else:
                seq_chunks.append(line.strip().upper())
        if header is not None:
            yield header, ''.join(seq_chunks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query_fasta', required=True)
    ap.add_argument('--ref_fasta_gz', required=True)
    ap.add_argument('--out_map_tsv', required=True)
    ap.add_argument('--out_unmapped_fasta', required=True)
    args = ap.parse_args()

    # Build ref dict: seq -> accession
    seq2ac = {}
    for h, s in read_fasta_any(args.ref_fasta_gz):
        # h can look like sp|P12345|NAME... or P12345
        if '|' in h:
            parts = h.split('|')
            ac = parts[1] if len(parts) > 1 else parts[0]
        else:
            ac = h
        # keep first occurrence
        if s not in seq2ac:
            seq2ac[s] = ac

    mapped = 0
    unmapped = []

    out_lines = ["Protein_ID\tUniProt_AC\tmethod"]
    for qh, qs in read_fasta_any(args.query_fasta):
        pid = qh.split('|')[0]
        ac = seq2ac.get(qs)
        if ac:
            mapped += 1
            out_lines.append(f"{pid}\t{ac}\texact")
        else:
            unmapped.append((pid, qs))

    with open(args.out_map_tsv, 'w') as f:
        f.write('\n'.join(out_lines) + '\n')

    with open(args.out_unmapped_fasta, 'w') as f:
        for pid, seq in unmapped:
            f.write(f">{pid}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")

    print(f"[OK] mapped_exact={mapped}")
    print(f"[OK] unmapped={len(unmapped)}")
    print(f"[OK] wrote: {args.out_map_tsv}")
    print(f"[OK] wrote: {args.out_unmapped_fasta}")


if __name__ == '__main__':
    main()
