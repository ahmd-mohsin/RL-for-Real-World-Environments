# Marlowe paths and tools, sourced by every script in scripts/marlowe (docs/research/swarm-pipeline/05-marlowe.md).
# Persistent state lives under CRU_ROOT. /users/muahmed was over its quota on 2026-10-19 (writes fail with EDQUOT), so the
# default root is the m000159 scratch; point CRU_ROOT at /users/muahmed once space there is freed.
export CRU_ROOT=${CRU_ROOT:-/scratch/m000159/muahmed}
export CRU_CODE=${CRU_CODE:-$CRU_ROOT/crucible}
export CRU_ENV=${CRU_ENV:-$CRU_ROOT/crucible-env}
export CRU_RUNS=${CRU_RUNS:-$CRU_ROOT/crucible-runs}
export CRU_SIF=${CRU_SIF:-$CRU_ROOT/sif}
export HF_HOME=${HF_HOME:-$CRU_ROOT/hf}
export UV_CACHE_DIR=${UV_CACHE_DIR:-$CRU_ROOT/cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$CRU_ROOT/uv-python}
export UV_LINK_MODE=copy UV_HTTP_TIMEOUT=${UV_HTTP_TIMEOUT:-300}
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$CRU_ROOT/cache/apptainer}
export APPTAINER_CONFIGDIR=${APPTAINER_CONFIGDIR:-$CRU_ROOT/cache/apptainer-config}  # instance state; the default ~/.apptainer is on the full home
# every cache a library writes under $HOME (triton, inductor, the CUDA JIT, vLLM, flashinfer, torch extensions) goes to
# node-local disk instead: a write under the over-quota home fails with EDQUOT and takes the job down
export CRU_CACHE=${CRU_CACHE:-${TMPDIR:-/tmp}/crucible-cache-$(id -u)}
export XDG_CACHE_HOME=$CRU_CACHE TRITON_CACHE_DIR=$CRU_CACHE/triton TORCHINDUCTOR_CACHE_DIR=$CRU_CACHE/inductor CUDA_CACHE_PATH=$CRU_CACHE/nv \
  TORCH_EXTENSIONS_DIR=$CRU_CACHE/torch_extensions VLLM_CACHE_ROOT=$CRU_CACHE/vllm FLASHINFER_WORKSPACE_BASE=$CRU_CACHE NUMBA_CACHE_DIR=$CRU_CACHE/numba \
  MPLCONFIGDIR=$CRU_CACHE/matplotlib PYTHONPYCACHEPREFIX=$CRU_CACHE/pycache
mkdir -p "$CRU_CACHE" 2>/dev/null || true
# JIT builds (flashinfer, torch extensions) use the venv's own CUDA 13.0 toolkit: the nodes' /usr/local/cuda lacks curand.h,
# and job 523884's vLLM died compiling flashinfer's sampler against it
_cu=$CRU_ENV/lib/python3.12/site-packages/nvidia/cu13
[ -x "$_cu/bin/nvcc" ] && export CUDA_HOME=$_cu
export SLURM_CONF=${SLURM_CONF:-/cm/shared/apps/slurm/var/etc/slurm/slurm.conf}
export PATH=/cm/shared/apps/slurm/current/bin:$HOME/.local/bin:$CRU_ENV/bin:$PATH
export PY=${PY:-$CRU_ENV/bin/python}
# the only usable route (m000215 is blocked and its scratch is 90% full: never use it)
export CRU_ACCOUNT=marlowe-m000159 CRU_PARTITION=preempt
