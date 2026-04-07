#!/usr/bin/env bash
set -euo pipefail

OUT_GZ=${1:-"uniprot_sprot_human.fasta.gz"}

# UniProtKB search query for reviewed human entries (Swiss-Prot)
QUERY='(reviewed:true)%20AND%20(organism_id:9606)'
URL="https://rest.uniprot.org/uniprotkb/search?query=${QUERY}&format=fasta&compressed=true"

echo "[INFO] Downloading: ${URL}"
mkdir -p "$(dirname "$OUT_GZ")"

# Use curl if available, otherwise wget
if command -v curl >/dev/null 2>&1; then
  curl -L --retry 5 --retry-delay 2 -o "$OUT_GZ" "$URL"
else
  wget -O "$OUT_GZ" "$URL"
fi

echo "[OK] wrote: $OUT_GZ"
