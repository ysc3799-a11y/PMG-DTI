#!/usr/bin/env bash
set -euo pipefail

P2RANK_DIR=$1      # directory that contains the prank launcher
STRUCT_DIR=$2      # directory containing downloaded structures
OUT_DIR=$3         # directory to write p2rank outputs
PARALLEL=${4:-4}

PRANK="$P2RANK_DIR/prank"

if [[ ! -x "$PRANK" ]]; then
  echo "ERROR: prank launcher not found or not executable at: $PRANK" >&2
  exit 1
fi

mkdir -p "$OUT_DIR"

# Run one structure per task; P2Rank will create per-input subfolders.
find "$STRUCT_DIR" -type f \( -name "*.pdb" -o -name "*.cif" -o -name "*.bcif" \) -print0 \
  | xargs -0 -n 1 -P "$PARALLEL" -I {} bash -lc '
    in="{}"
    base=$(basename "$in")
    sub="$OUT_DIR/${base%.*}"
    mkdir -p "$sub"
    "$PRANK" predict -f "$in" -o "$sub" >/dev/null 2>&1 || echo "P2Rank failed on $in" >&2
  '

echo "[OK] P2Rank batch done. Outputs in: $OUT_DIR"
