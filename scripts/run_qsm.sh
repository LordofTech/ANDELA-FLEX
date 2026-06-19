#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$ROOT/sample_data/qsm/sub-1/anat}"
OUT="${2:-$ROOT/output/qsm}"

mkdir -p "$OUT"

docker build -t andela-flex-qsm -f "$ROOT/docker/qsm/Dockerfile" "$ROOT"
docker run --rm \
  -v "$DATA:/app/data/sub-1/anat:ro" \
  -v "$OUT:/app/data/derivatives/qsm/sub-1/anat" \
  andela-flex-qsm

echo "chi map -> $OUT/sub-1_MEGRE_Chimap.nii"
