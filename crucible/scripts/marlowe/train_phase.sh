#!/bin/bash
# One phase of the 8-GPU training smoke (stage.sh plan smoke8): train_phase.sh a|b <run dir>
#   a: steps 1..4 on Qwen2.5-7B (sharded checkpoint after step 2), then a forced stop: SIGTERM to every rank, as a
#      preemption sends it; the ranks checkpoint and exit. Passes when the reference has every step and the TERM
#      checkpoint exists. A re-run (after a real interruption) starts the phase clean.
#   b: new processes (as after a requeue on any node) restore step 2, redo steps 3..4, compare (train_smoke.py).
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
PH=${1:?phase a|b}; RUN=${2:?run dir}; OUT=$RUN/train
mkdir -p "$OUT"
MODEL=${CRU_TRAIN_MODEL:-Qwen/Qwen2.5-7B-Instruct}
MESH=${CRU_TRAIN_MESH:-2x4}
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0} TORCH_NCCL_ASYNC_ERROR_HANDLING=1 NCCL_DEBUG=${NCCL_DEBUG:-WARN} OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export CRUCIBLE_PG_TIMEOUT_S=${CRUCIBLE_PG_TIMEOUT_S:-1800} HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export MASTER_PORT=$(( 20000 + ${SLURM_JOB_ID:-0} % 20000 + (${SLURM_RESTART_COUNT:-0} * 2) + ([ "$PH" = b ] && echo 1 || echo 0) ))
CMD=(srun --kill-on-bad-exit=1 --ntasks-per-node="${CRU_GPUS:-8}" "$PY" "$HERE/train_smoke.py" --phase "$PH" --model "$MODEL" --mesh "$MESH" --out "$OUT")
echo "[train $PH] $(date) host $(hostname) model $MODEL mesh $MESH port $MASTER_PORT"
nvidia-smi --query-gpu=index,name,memory.used --format=csv,noheader || true
if [ "$PH" = a ]; then
  rm -rf "$OUT/ckpt" "$OUT/ref.json" "$OUT/term.json" "$OUT/verdict.json"
  "${CMD[@]}" &
  S=$!
  while kill -0 "$S" 2>/dev/null; do
    if [ -e "$OUT/ref.json" ]; then
      sleep 10
      echo "[train a] forced stop $(date +%T): SIGTERM to every rank"
      for p in $(pgrep -u "$(id -u)" -f "train_smoke.py --phase a"); do  # the ranks themselves, not srun (Slurm signals them so)
        [ "$(ps -o comm= -p "$p" 2>/dev/null)" = srun ] || kill -TERM "$p" 2>/dev/null
      done
      break
    fi
    sleep 5
  done
  wait "$S"; rc=$?
  echo "[train a] srun exit $rc"
  "$PY" - "$OUT" <<'PY'
import json, pathlib, sys
out = pathlib.Path(sys.argv[1])
ref = json.loads((out / "ref.json").read_text()) if (out / "ref.json").exists() else None
term = json.loads((out / "term.json").read_text()) if (out / "term.json").exists() else None
steps = len(ref["ranks"][0]["hist"]) if ref else 0
ok = bool(ref and term and steps >= 4 and term["signal"] == "SIGTERM" and (out / "ckpt" / "step-2").exists())
print(json.dumps({"steps": steps, "term": term, "pass": ok}))
sys.exit(0 if ok else 1)
PY
  exit $?
fi
"${CMD[@]}"; rc=$?
if [ "$rc" -eq 0 ] && [ -z "${CRU_KEEP_CKPT:-}" ]; then
  du -sh "$OUT/ckpt"/* 2>/dev/null; rm -rf "$OUT/ckpt"   # ~90 GB per checkpoint of a 7B model with AdamW: verdict.json keeps the result
fi
exit $rc
