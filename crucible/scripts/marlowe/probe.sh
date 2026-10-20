#!/bin/bash
# Node probe (first stage of every plan): GPUs, driver, topology, the venv's CUDA view, and whether this node grants what
# the Apptainer sandbox asks for (network namespace, cgroup limits, --nv). Writes <run>/probe-<host>.json.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
RUN=${1:?run dir}
H=$(hostname)
echo "== host $H job ${SLURM_JOB_ID:-} restart ${SLURM_RESTART_COUNT:-0} gpus ${CUDA_VISIBLE_DEVICES:-none} $(date)"
nvidia-smi || true
nvidia-smi topo -m 2>/dev/null | head -12 || true
df -h /tmp "$CRU_ROOT" | cat
N=crucible-apt-probe-$$
lim=no; apptainer instance start --containall --cleanenv --no-home --net --network none --writable-tmpfs --pids-limit 512 --memory 4G \
  "$CRU_SIF/world-py312.sif" "$N" >/dev/null 2>&1 && { lim=yes; apptainer instance stop "$N" >/dev/null 2>&1; }
gpu=$(apptainer exec --containall --cleanenv --no-home --nv "$CRU_SIF/cuda128.sif" nvidia-smi -L 2>&1 | grep -c "^GPU" || true)
"$PY" - "$RUN/probe-$H.json" "$lim" "$gpu" <<'PY'
import json, subprocess, sys
import torch
q = subprocess.run(["nvidia-smi", "--query-gpu=index,name,driver_version,memory.total", "--format=csv,noheader"], capture_output=True, text=True).stdout
out = {"host": __import__("socket").gethostname(), "torch": torch.__version__, "torch_cuda": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
       "devices": torch.cuda.device_count(), "nvidia_smi": q.strip().splitlines(), "apptainer_cgroup_limits": sys.argv[2] == "yes",
       "apptainer_nv_gpus_seen": int(sys.argv[3] or 0)}
if torch.cuda.is_available():
    x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    out["matmul_ok"] = bool(torch.isfinite(x @ x).all())
    out["nccl"] = ".".join(map(str, torch.cuda.nccl.version()))
json.dump(out, open(sys.argv[1], "w"), indent=1)
print(json.dumps(out, indent=1))
sys.exit(0 if out["cuda_available"] else 1)
PY
