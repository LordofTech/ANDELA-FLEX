#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA="${1:-$ROOT/sample_data/pharm}"
OUT="${2:-$ROOT/output/pharm}"

mkdir -p "$OUT"

docker build -t andela-flex-pharm -f "$ROOT/docker/pharm/Dockerfile" "$ROOT"
docker run --rm \
  -v "$DATA:/root/data:ro" \
  -v "$OUT:/root/results" \
  andela-flex-pharm

echo "poses -> $OUT/docked_poses.sdf"
