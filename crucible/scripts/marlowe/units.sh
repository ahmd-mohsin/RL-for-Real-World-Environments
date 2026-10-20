#!/bin/bash
# smoke8 stage units: the tests/unit FSDP2 / HSDP / SPMD scenarios on NCCL with 8 ranks (scripts/slurm_check.py unit),
# then the report; the stage fails when any scenario misses its tolerance.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
OUT=${1:?run dir}/units
export NCCL_CUMEM_ENABLE=${NCCL_CUMEM_ENABLE:-0} TORCH_NCCL_ASYNC_ERROR_HANDLING=1
for sc in fsdp hsdp hf_layer spmd_run; do
  export MASTER_PORT=$(( 30000 + ${SLURM_JOB_ID:-0} % 20000 + RANDOM % 1000 ))
  srun --kill-on-bad-exit=1 --ntasks-per-node="${CRU_GPUS:-8}" "$PY" "$CRU_CODE/scripts/slurm_check.py" unit "$sc" --out "$OUT" || echo "scenario $sc failed"
done
"$PY" "$CRU_CODE/scripts/slurm_check.py" report --out "$OUT"
"$PY" -c "import json,sys; sys.exit(0 if json.load(open('$OUT/stage1.json'))['pass'] else 1)"
