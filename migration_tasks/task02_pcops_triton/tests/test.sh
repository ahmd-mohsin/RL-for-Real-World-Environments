#!/usr/bin/env bash
# Verifier entry point: apply the agent's pcops patch to a pristine copy, then grade.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ORIG=${PCOPS_ORIGINALS:-/opt/originals}
PATCH=${MODEL_PATCH:-/logs/artifacts/model.patch}
OUT=${VERIFIER_DIR:-/logs/verifier}
mkdir -p "$OUT"
WORK=$(mktemp -d)
cp -a "$ORIG/pcops" "$WORK/pcops"
if [ -s "$PATCH" ] && ! (cd "$WORK/pcops" && git apply --whitespace=nowarn "$PATCH") 2>"$OUT/apply.log"; then
  echo '{"reward": 0.0, "error": "patch does not apply"}' > "$OUT/report.json"; echo 0 > "$OUT/reward.txt"; exit 0
fi
python3 "$HERE/grade.py" --candidate "$WORK/pcops" --originals "$ORIG" --out "$OUT/report.json"
python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['reward'])" "$OUT/report.json" > "$OUT/reward.txt"
cat "$OUT/reward.txt"
