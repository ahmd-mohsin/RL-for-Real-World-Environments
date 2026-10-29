#!/bin/bash
# One arm of the sealed benchmark on one GH200 node (doc 14 §4): the student served by vLLM (TP=2, the engine record the
# trained arm must match), k attempts per task at the manifest's seeds through bench.py run (WorkspaceHarness +
# LLMToolPolicy, eval mode, the lane's budget), the Worlds and graders as Apptainer instances through the locked-down shim.
# Prerequisite: bench_prep.sbatch passed (env, SIFs, lockdown, dry run). Results sync to $W/runs/<purpose> every 5 min
# and at the end, so a time limit loses only the tasks in flight (a finished task is never re-run).
# Submit (from $W):  unset SBATCH_ACCOUNT SLURM_ACCOUNT; sbatch --account=bjhg-dtai-gh bench.sbatch
#   env: PURPOSE (baseline), SLOTS (24 Worlds at once), CONC (8 episodes per task), TP (2)
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --gpus-per-node=2 --cpus-per-task=36 --mem=110g
#SBATCH --time=03:50:00
#SBATCH --job-name=crucible-bench
#SBATCH --output=logs/bench-%j.out
source "${CRUCIBLE_BENCH_DIR:-/work/nvme/bjhg/mahmedmohsin/crucible-bench}/code/crucible/scripts/deltaai/bench_env.sh"
PURPOSE=${PURPOSE:-baseline}
TP=${TP:-2}
O=$W/out/bench-$SLURM_JOB_ID
mkdir -p "$O"
t "node=$(hostname) gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | tr '\n' ,) cpus=$(nproc) purpose=$PURPOSE"
[ -x "$PY" ] && $PY -c "import crucible, transformers, torch" || { t "no environment: run bench_prep.sbatch first"; exit 2; }
build_sifs || { t "sif build failed"; exit 2; }
if lockdown_all; then t "lockdown: OK in every World image"; else t "lockdown FAILED: no model episodes"; exit 3; fi
stage_data
cd "$W/code/crucible"
$PY "$B" verify --manifest "$D/debug/manifest.json" --debug "$D/debug" --relocated > "$O/verify.json" 2> "$O/verify.err" || { t "verify failed"; exit 4; }

# the server
cat > "$J/serve.py" <<EOF
import json
from crucible.train.bridges.vllm.serve import ServeSpec
s = ServeSpec(model="$MODEL", name="$MODEL", tp=$TP, seed=0, env={"HF_HOME": "$HFW"}, cache_dir="$W/vcache")
json.dump(s.record(), open("$O/engine.json", "w"))
print(s.shell(binds=["$HFW:$HFW"]))
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
t "serve: $(grep -E 'KV cache|Maximum concurrency' $O/vllm.log | tail -2 | tr '\n' ' ')"

# the arm, synced to $W every 5 minutes
R=$J/runs
mkdir -p "$R" "$W/runs"
( while sleep 300; do rsync -a "$R/" "$W/runs/" 2>/dev/null; done ) &
SYNC=$!
$PY "$B" run --purpose "$PURPOSE" --vllm-engine "$O/engine.json" --expect-harness "$(cat $W/harness.txt)" --measured "$W/measured.json" \
  --manifest "$D/debug/manifest.json" --debug "$D/debug" --out "$R" --concurrency "${CONC:-8}" --slots "${SLOTS:-24}" --tasks "$(cat $W/order.txt)" > "$O/run.log" 2>&1
t "run: rc=$? $(tail -2 $O/run.log | tr '\n' ' ')"
kill $SYNC 2>/dev/null
rsync -a "$R/" "$W/runs/"
$PY "$B" report --manifest "$D/debug/manifest.json" --out "$W/runs" --purpose "$PURPOSE" > "$O/report.json" 2>&1
t "report: $(tr -d '\n ' < $O/report.json | cut -c1-600)"
kill $VPID 2>/dev/null; wait $VPID 2>/dev/null
docker ps
rm -rf "$J"
t BENCH_JOB_DONE
