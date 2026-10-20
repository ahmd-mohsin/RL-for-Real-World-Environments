#!/bin/bash
# Bring Crucible up on the Marlowe login node (no GPU needed). Idempotent: a finished step leaves a marker in
# $CRU_ROOT/.setup and is skipped next time (rm the marker to redo it). Code first: rsync the crucible tree to $CRU_CODE.
#   bash scripts/marlowe/setup_env.sh > $CRU_ROOT/crucible-setup.log 2>&1
# Steps: uv-managed Python 3.12 -> venv -> vLLM (pins torch) -> Crucible [train,llm,answers] + the toy pack, editable ->
# the SIF images (build_sifs.sh) -> the models into $HF_HOME -> an import check -> the uv cache removed (the m000159 group
# has a 512k-file limit on scratch) -> $CRU_ROOT/.setup/ready, which every job checks before it runs a stage.
# Versions: vllm 0.32.0 pins torch 2.13.0 built for CUDA 13.0. The nodes run driver 580.173.02 (CUDA 13.0); a vLLM >= 0.24
# wheel forced onto cu128 torch fails at import on Marlowe (libcudart.so.13 missing), so the CUDA 13 pair is the one that works.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
VLLM=${VLLM:-0.32.0}
MODELS=${MODELS:-"Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-7B-Instruct"}
M=$CRU_ROOT/.setup
mkdir -p "$M" "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$APPTAINER_CACHEDIR" "$HF_HOME" "$CRU_SIF" "$CRU_RUNS/slurm"
done_() { [ -e "$M/$1.done" ]; }
mark() { date +%Y-%m-%dT%H:%M:%S%z > "$M/$1.done"; }
done_ python || { uv python install 3.12; mark python; }
done_ venv || { rm -rf "$CRU_ENV"; uv venv "$CRU_ENV" --python 3.12; mark venv; }
done_ vllm || { uv pip install --python "$PY" "vllm==$VLLM"; mark vllm; }
# --no-sources: the optional framework extras point at sibling checkouts that are not on the cluster
done_ crucible || { uv pip install --python "$PY" --no-sources -e "$CRU_CODE[train,llm,answers]" -e "$CRU_CODE/packs/_toy" pytest pytest-asyncio \
                      hypothesis "huggingface_hub[hf_xet]"; mark crucible; }
done_ sifs || { bash "$HERE/build_sifs.sh"; mark sifs; }
if ! done_ models; then
  ( unset HF_TOKEN HUGGING_FACE_HUB_TOKEN  # public models: no token travels to the cluster
    for m in $MODELS; do "$CRU_ENV/bin/hf" download "$m"; done )
  mark models
fi
uv pip freeze --python "$PY" > "$CRU_ROOT/crucible-env.freeze.txt"
"$PY" - <<'PY'
import importlib, json
out = {}
for m in ("torch", "transformers", "vllm", "safetensors", "crucible", "crucible_pack_toy"):
    mod = importlib.import_module(m)
    out[m] = getattr(mod, "__version__", "ok")
import torch
out["torch.cuda"] = torch.version.cuda
print(json.dumps(out, indent=1))
PY
rm -rf "$UV_CACHE_DIR"
echo "ready $(date +%Y-%m-%dT%H:%M:%S%z) vllm $VLLM sifs $(cd "$CRU_SIF" && cat ./*.sha256 | cut -c1-12 | tr '\n' ' ')" > "$M/ready"
echo "SETUP_EXIT=0"
