#!/usr/bin/env bash
# Single-attempt agent run of task04 in a no-network sandbox on the lab GPU host.
#   run_agent.sh <model> <out_dir> [step_limit] [cost_limit]
# Starts a fresh `--network none` container from task04-pcl-gpu-cuda12:v1 on the host, drives
# mini-swe-agent through `docker exec`, then collects the patch exactly like the verifier.
set -euo pipefail
MODEL=$1; OUT=$2; STEPS=${3:-350}; COST=${4:-80}
HERE=$(cd "$(dirname "$0")" && pwd)
COMMON=$(cd "$HERE/../common" && pwd)
ROOT=$(cd "$HERE/../.." && pwd)
HOST=${LAB_HOST:-ahmed@100.100.101.36}
CTR=t4agent_$(date +%s)
SSH="ssh -o BatchMode=yes $HOST"
$SSH "docker run -d --name $CTR --network none --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --shm-size=8g --cpus=${CPUS:-16} --memory=64g task04-pcl-gpu-cuda12:v1 sleep infinity" >/dev/null
echo "sandbox container: $CTR"
export OPENROUTER_API_KEY=$(grep '^OPENROUTER_API_KEY=' "$ROOT/.env" | cut -d= -f2-)
export MSWEA_CONFIGURED=true
export PYTHONPATH="$COMMON${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$OUT"
MINI=${MINI:-mini}
"$MINI" -c "$COMMON/mswea/migration.yaml" -c "$HERE/mswea_sandbox.yaml" \
  -c "model.model_name=$MODEL" -c "environment.ssh=$SSH" -c "environment.docker_container=$CTR" \
  -c "agent.step_limit=$STEPS" -c "agent.cost_limit=$COST" \
  -t "$(cat "$HERE/instruction.md")" --yolo --exit-immediately \
  -o "$OUT/trajectory.json" 2>&1 | tee "$OUT/console.log" || true
$SSH "docker exec -i $CTR bash -lc \"cd /workspace/pcl && git add -A -- . ':(exclude,glob)**/build*/**' ':(exclude)build*' ':(exclude)install*' && git diff --binary --cached base\"" > "$OUT/model.patch"
echo "patch: $(wc -l < "$OUT/model.patch") lines -> $OUT/model.patch (container $CTR left running for inspection)"
