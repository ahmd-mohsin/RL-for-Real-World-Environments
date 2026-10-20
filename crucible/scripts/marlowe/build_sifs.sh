#!/bin/bash
# Build the x86_64 SIF images on the Marlowe login node (never ship images built on a Mac: those are arm64).
#   bash scripts/marlowe/build_sifs.sh [name ...]       # default: every defs/*.def; an existing image is kept (-f rebuilds)
# Writes $CRU_SIF/<name>.sif and $CRU_SIF/<name>.sha256.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
FORCE=""
if [ "${1:-}" = "-f" ]; then FORCE=1; shift; fi
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$CRU_ROOT/cache/apptainer-tmp}
mkdir -p "$CRU_SIF" "$APPTAINER_TMPDIR"
names=("$@")
[ ${#names[@]} -eq 0 ] && for d in "$HERE"/defs/*.def; do names+=("$(basename "$d" .def)"); done
for n in "${names[@]}"; do
  out=$CRU_SIF/$n.sif
  if [ -s "$out" ] && [ -z "$FORCE" ]; then echo "[sif] $n: exists"; continue; fi
  echo "[sif] building $n $(date -Is)"
  apptainer build -F "$out.tmp" "$HERE/defs/$n.def"
  mv "$out.tmp" "$out"
  sha256sum "$out" | cut -d' ' -f1 > "$CRU_SIF/$n.sha256"
  echo "[sif] $n: $(du -h "$out" | cut -f1) sha256 $(cat "$CRU_SIF/$n.sha256")"
done
