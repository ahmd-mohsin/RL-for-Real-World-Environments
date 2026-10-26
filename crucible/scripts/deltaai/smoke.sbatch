#!/bin/bash
# DeltaAI inference smoke (doc 14-deltaai §5): the crucible environment on the node, a World SIF from the Mac's image,
# the student's weights, the vLLM server, the lockdown probe in that World, a throughput measurement, and one short
# student episode on a training task through the debug lane with --backend vllm.
# Submit:  unset SBATCH_ACCOUNT SLURM_ACCOUNT; sbatch --account=bjhg-dtai-gh --partition=ghx4-interactive smoke.sbatch
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=2 --cpus-per-task=32 --mem=200g
#SBATCH --time=02:00:00
#SBATCH --job-name=crucible-infer-smoke
#SBATCH --output=logs/smoke-%j.out
set -u
W=/work/hdd/bjhg/mahmedmohsin/crucible-infer
J=/tmp/crucible-$SLURM_JOB_ID
MODEL=Qwen/Qwen3.6-35B-A3B
TASK=${TASK:-db-go-go-toml-a02114e3}
IMG=crucible-lh-d_go-go-toml-d4fe68839d05
TP=${TP:-2}
mkdir -p "$J/state" "$J/ovl" "$J/cfg" "$J/cache" "$J/atmp" "$J/ws" "$W/sif" "$W/out" "$W/hf" "$W/vcache"
export CRUCIBLE_SIF_DIR=$W/sif CRUCIBLE_SHIM_STATE=$J/state CRUCIBLE_SHIM_SCRATCH=$J/ovl CRUCIBLE_SHIM_CONFIGDIR=$J/cfg CRUCIBLE_SHIM_ALLOW=$J
export APPTAINER_CACHEDIR=$J/cache APPTAINER_TMPDIR=$J/atmp TMPDIR=$J
export PATH=$W/bin:$PATH
O=$W/out/smoke-$SLURM_JOB_ID
mkdir -p "$O"
t() { echo "[$(date +%H:%M:%S)] $*"; }
t "node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ,)"

# 1. the environment (once): uv from PyPI, Python 3.12, crucible editable with the tokenizer and OpenAI client
if [ ! -x "$W/venv/bin/python" ]; then
  t "env: building"
  module load python/3.11.9
  python3 -m venv "$J/uvenv" && "$J/uvenv/bin/pip" -q install uv==0.9.5 > "$O/env.log" 2>&1 || "$J/uvenv/bin/pip" -q install uv >> "$O/env.log" 2>&1
  UV="$J/uvenv/bin/uv"
  $UV venv -q --python 3.12 "$W/venv" >> "$O/env.log" 2>&1
  $UV pip install -q --python "$W/venv/bin/python" -e "$W/code/crucible" -e "$W/code/crucible/packs/coding" "transformers>=4.56" "openai>=1.60" pytest >> "$O/env.log" 2>&1
  t "env: rc=$? $($W/venv/bin/python -c 'import crucible, transformers; print("crucible ok, transformers", transformers.__version__)' 2>&1 | tail -1)"
fi
PY=$W/venv/bin/python
if ! $PY -c "import torch" 2>/dev/null; then  # crucible.train.engine imports torch (the CPU build is enough off the GPU)
  t "env: torch (cpu)"
  module load python/3.11.9; python3 -m venv "$J/uvenv" && "$J/uvenv/bin/pip" -q install uv >> "$O/env.log" 2>&1
  "$J/uvenv/bin/uv" pip install -q --python "$PY" torch --index-url https://download.pytorch.org/whl/cpu >> "$O/env.log" 2>&1
  t "env: torch $($PY -c 'import torch; print(torch.__version__)' 2>&1 | tail -1)"
fi

# 2. the World SIF from the Mac's image (docker save) and its config side file
if [ ! -f "$W/sif/$IMG.sif" ]; then
  t "sif: building $IMG"
  apptainer -q build "$W/sif/$IMG.sif" "docker-archive://$W/images/$IMG.tar" > "$O/sif.log" 2>&1; t "sif: rc=$?"
  cp "$W/images/$IMG.config.json" "$W/sif/$IMG.sif.json"
fi

# 3. the weights (once), into the HF cache the server and the tokenizer read offline
if [ ! -f "$W/hf/.done-$(echo $MODEL | tr / _)" ]; then
  t "weights: downloading $MODEL"
  printf 'HF_TOKEN=%s\nHF_HOME=%s\nHF_HUB_ENABLE_HF_TRANSFER=0\n' "$(cat ~/.hf_token)" "$W/hf" > "$J/hf.env"; chmod 600 "$J/hf.env"
  apptainer exec --cleanenv --env-file "$J/hf.env" -B "$W/hf" /sw/llmhub/vllm.sif python3 -c "
from huggingface_hub import snapshot_download
p = snapshot_download('$MODEL', max_workers=16)
print('weights at', p)" > "$O/weights.log" 2>&1 && touch "$W/hf/.done-$(echo $MODEL | tr / _)"
  rm -f "$J/hf.env"
  t "weights: $(tail -1 $O/weights.log) $(du -sh $W/hf | cut -f1)"
fi
export HF_HOME=$W/hf HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

# 4. the lockdown probe inside the task's World image, through the shim, exactly as an episode starts it
$PY -c "from crucible.world.lockdown import PROBE; open('$J/ws/probe.sh','w').write(PROBE)"
docker run --rm --network none -v "$J/ws:$J/ws" crucible-lh-d:go-go-toml-d4fe68839d05 bash "$J/ws/probe.sh" "$HOME" 0 > "$O/lockdown.txt" 2> "$O/lockdown.err"
$PY -c "
from crucible.world import lockdown
r = lockdown.parse(open('$O/lockdown.txt').read())
p = lockdown.verdict(r, home='$HOME')
print('LOCKDOWN', 'OK' if not p else 'FAILED', p)" | tee "$O/lockdown.verdict"
grep -q "LOCKDOWN OK" "$O/lockdown.verdict" || { t "lockdown failed: no model episodes"; exit 3; }

# 5. the server
cat > "$J/serve.py" <<EOF
import json, sys
from crucible.train.bridges.vllm.serve import ServeSpec
s = ServeSpec(model="$MODEL", name="$MODEL", tp=$TP, seed=0, env={"HF_HOME": "$W/hf"}, cache_dir="$W/vcache")
json.dump(s.record(), open("$O/engine.json", "w"))
print(s.shell(binds=["$W/hf:$W/hf"]))
EOF
SERVE=$($PY "$J/serve.py")
t "serve: $SERVE"
eval "$SERVE" > "$O/vllm.log" 2>&1 &
VPID=$!
$PY -c "
import asyncio
from crucible.train.bridges.vllm.client import VLLMBackend
from crucible.train.bridges.vllm.serve import wait_ready
print('ready after', round(asyncio.run(wait_ready(VLLMBackend('http://127.0.0.1:8000'), timeout_s=2400)), 1), 's')" 2>&1 | tail -1
t "serve: $(grep -E 'KV cache|GPU KV|Maximum concurrency' $O/vllm.log | tail -3 | tr '\n' ' ')"

# 6. throughput: episode-shaped requests
for spec in "8 16000 512" "32 32000 512" "64 64000 256"; do
  set -- $spec
  $PY "$W/code/crucible/scripts/deltaai/throughput.py" --model "$MODEL" --concurrency $1 --prompt $2 --max-tokens $3 --rounds 1 2>&1 | tail -1 | tee -a "$O/throughput.jsonl"
done

# 7. one short student episode on a training task (tool limit 20), the debug lane's own World and grader through the shim
t "episode: $TASK"
cd "$W/code/crucible"
$PY scripts/lh_debug_target.py student --backend vllm --vllm-engine "$O/engine.json" --student "$MODEL" --tasks "$TASK" --k 1 \
  --tool-limit 20 --student-concurrency 1 --slots 2 --grade-parallel 1 --cap 1 --audit-reserve 0 --debug "$W/data/debug" --out "$J/runs/smoke" \
  > "$O/episode.log" 2>&1
t "episode: rc=$? $(tail -3 $O/episode.log | tr '\n' ' ')"
cp -r "$J/runs/smoke" "$O/run" 2>/dev/null
find "$O/run" -name metrics.json -exec cat {} \; | head -c 3000; echo

# 8. one short student episode on a kernel task (gh200-node: the GPU World through the shim, grades in Apptainer mode)
KSIF=/work/hdd/bjhg/mahmedmohsin/crucible/sif/crucible-lh-k_base-12.8.1.sif
[ -e "$W/sif/crucible-lh-k_base-12.8.1.sif" ] || ln -s "$KSIF" "$W/sif/crucible-lh-k_base-12.8.1.sif"
export CRUCIBLE_KERNEL_WORK=$J/kep PATH=$W/bin:$W/venv/bin:$PATH
t "kernel episode: kd-bug-llamacpp-ca21f3a1"
$PY scripts/lh_kernel_target.py student --host gh200-node --kernel "$W/data/kernel_debug" --backend vllm --vllm-engine "$O/engine.json" \
  --student "$MODEL" --tasks kd-bug-llamacpp-ca21f3a1 --k 1 --tool-limit 20 --student-concurrency 1 --slots 1 --cap 1 --audit-reserve 0 \
  --out "$J/runs/ksmoke" > "$O/kepisode.log" 2>&1
t "kernel episode: rc=$? $(tail -3 $O/kepisode.log | tr '\n' ' ')"
cp -r "$J/runs/ksmoke" "$O/krun" 2>/dev/null
find "$O/krun" -name metrics.json -exec cat {} \; | head -c 3000; echo

kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
docker ps
rm -rf "$J"
t SMOKE_JOB_DONE
