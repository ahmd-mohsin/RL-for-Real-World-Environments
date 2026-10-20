#!/bin/bash
# smoke1 stage rollout: vLLM serves Qwen2.5-Coder-7B-Instruct on this job's GPU (127.0.0.1 only, from the HF cache on
# scratch, offline), then one Crucible rollout through the Apptainer World with it as the policy (rollout_smoke.py).
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
RUN=${1:?run dir}
MODEL=${CRU_ROLLOUT_MODEL:-Qwen/Qwen2.5-Coder-7B-Instruct}
PORT=$(( 18000 + ${SLURM_JOB_ID:-0} % 1000 ))
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1
# PyTorch-native top-p/top-k sampling: flashinfer's sampler is JIT-compiled at start (failed on n02, job 523884: curand.h)
export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
LOG=$RUN/logs/vllm.log
"$CRU_ENV/bin/vllm" serve "$MODEL" --host 127.0.0.1 --port "$PORT" --dtype bfloat16 --max-model-len 16384 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --served-model-name "$MODEL" > "$LOG" 2>&1 &
V=$!
trap 'kill -TERM $V 2>/dev/null; wait $V 2>/dev/null' EXIT
t0=$(date +%s)
until "$PY" -c "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:$PORT/health', timeout=5)" 2>/dev/null; do
  if ! kill -0 "$V" 2>/dev/null; then echo "vLLM exited:"; tail -60 "$LOG"; exit 1; fi
  if [ $(( $(date +%s) - t0 )) -gt 1500 ]; then echo "vLLM not healthy after 1500 s"; tail -60 "$LOG"; exit 1; fi
  sleep 5
done
echo "vLLM healthy after $(( $(date +%s) - t0 )) s ($("$PY" -c 'import vllm; print(vllm.__version__)'))"
"$PY" "$HERE/rollout_smoke.py" --policy vllm --base-url "http://127.0.0.1:$PORT/v1" --model "$MODEL" --out "$RUN"
