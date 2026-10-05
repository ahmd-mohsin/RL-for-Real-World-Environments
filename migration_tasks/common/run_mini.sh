#!/usr/bin/env bash
# Single-attempt mini-swe-agent run of a migration task inside its GPU sandbox.
#   run_mini.sh <task_dir> <model> "<ssh command to the sandbox>" <out_dir>
# The API key is read from the project .env (OPENROUTER_API_KEY) and stays local.
set -euo pipefail
TASK=$(cd "$1" && pwd); MODEL=$2; SSH=$3; OUT=$4
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
export MSWEA_CONFIGURED=true
export PYTHONPATH="$HERE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
MINI=${MINI:-mini}
"$MINI" -c "$HERE/mswea/migration.yaml" \
  -c "model.model_name=$MODEL" \
  -c "environment.ssh=$SSH" \
  -t "$(cat "$TASK/instruction.md")" \
  --yolo --exit-immediately \
  -o "$OUT/trajectory.json" 2>&1 | tee "$OUT/console.log"
# Collect the agent's work exactly like the verifier's collect step.
$SSH -o BatchMode=yes 'bash -s' > "$OUT/model.patch" <<'COLLECT'
cd /app && git add -A -- . ':(exclude,glob)**/build*/**' ':(exclude)build*' && git diff --binary --cached base
COLLECT
echo "patch: $(wc -l < "$OUT/model.patch") lines -> $OUT/model.patch"
