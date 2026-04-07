#!/usr/bin/env bash
set -euo pipefail

QUERY_FASTA=$1
REF_FASTA_GZ=$2
OUT_TSV=$3

TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

# mmseqs can't read gz directly in all builds; decompress safely.
REF_FASTA="$TMPDIR/ref.fasta"
gzip -dc "$REF_FASTA_GZ" > "$REF_FASTA"

# Format: query,target,pident,alnlen,qcov,tcov,evalue,bits
mmseqs easy-search \
  "$QUERY_FASTA" \
  "$REF_FASTA" \
  "$OUT_TSV" \
  "$TMPDIR/tmp" \
  --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits" \
  --cov-mode 0 -c 0.8 --min-seq-id 0.9 --max-seqs 5

echo "[OK] wrote: $OUT_TSV"
