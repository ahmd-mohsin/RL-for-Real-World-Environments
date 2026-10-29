#!/bin/bash
# The benchmark's CPU-only preparation and dry run (doc 14 §4): no GPU, no weights, no model.
#  1. the environment (once): Python 3.12, crucible editable, transformers, CPU torch, in $W/venv;
#  2. one SIF per World image from the Mac's arm64 archives;
#  3. the lockdown probe in every World image (no network, no host files, no secrets);
#  4. bench.py dryrun: per task its World opened as an episode opens it, the lockdown probe and plain network calls inside
#     it, the null patch (reward 0) and the oracle applied in the World and collected back (reward 1), graded by the gates'
#     own verifier through the shim;
#  5. two scripted one-episode arms (bench.py stub-serve: the tokenizer only) through the real episode path.
# Submit (from $W):  unset SBATCH_ACCOUNT SLURM_ACCOUNT; sbatch --account=bjhg-dtai-gh bench_prep.sbatch
# DeltaAI has no CPU-only partition: on ghx4 with no GPU, 32 CPUs and 48 GB bill 444 per hour (0.44 GPU-hours).
#SBATCH --partition=ghx4
#SBATCH --nodes=1 --ntasks=1 --cpus-per-task=32 --mem=48g
#SBATCH --time=02:00:00
#SBATCH --job-name=crucible-bench-prep
#SBATCH --output=logs/prep-%j.out
# Slurm runs a copy of this file, so the helpers are read from the synced code tree
source "${CRUCIBLE_BENCH_DIR:-/work/nvme/bjhg/mahmedmohsin/crucible-bench}/code/crucible/scripts/deltaai/bench_env.sh"
O=$W/out/prep-$SLURM_JOB_ID
mkdir -p "$O"
t "node=$(hostname) arch=$(uname -m) cpus=$(nproc) gpus=${SLURM_GPUS_ON_NODE:-0}"

# 1. the environment
if [ ! -x "$W/venv/bin/python" ] || ! "$W/venv/bin/python" -c "import crucible, transformers, torch" 2>/dev/null; then
  t "env: building"
  module load python/3.11.9
  python3 -m venv "$J/uvenv" && "$J/uvenv/bin/pip" -q install uv > "$O/env.log" 2>&1
  UV="$J/uvenv/bin/uv"
  [ -x "$W/venv/bin/python" ] || $UV venv -q --python 3.12 "$W/venv" >> "$O/env.log" 2>&1
  $UV pip install -q --python "$W/venv/bin/python" -e "$W/code/crucible" -e "$W/code/crucible/packs/coding" "transformers>=4.56" "openai>=1.60" pytest >> "$O/env.log" 2>&1
  $UV pip install -q --python "$W/venv/bin/python" torch --index-url https://download.pytorch.org/whl/cpu >> "$O/env.log" 2>&1
fi
t "env: $($PY -c 'import crucible, transformers, torch; print("ok transformers", transformers.__version__, "torch", torch.__version__)' 2>&1 | tail -1)"

# 2. SIFs, and the engine's JIT caches seeded from the inference smoke's (read only), so the GPU job does not compile them
build_sifs || { t "sif build failed"; exit 2; }
VC=/work/hdd/bjhg/mahmedmohsin/crucible-infer/vcache
[ -d "$VC" ] && [ -z "$(ls -A $W/vcache 2>/dev/null)" ] && rsync -a --no-acls "$VC/" "$W/vcache/" && t "vcache: seeded $(du -sh $W/vcache | cut -f1)"
t "sifs: $(ls $W/sif/*.sif | wc -l)"

# 3. lockdown in every World image
if lockdown_all; then t "lockdown: OK in every World image"; else t "lockdown FAILED: see $O/lockdown.verdict"; exit 3; fi

# 4. the dry run
stage_data
cd "$W/code/crucible"
$PY "$B" verify --manifest "$D/debug/manifest.json" --debug "$D/debug" --relocated > "$O/verify.json" 2> "$O/verify.err"; t "verify: rc=$? $(tr -d '\n ' < $O/verify.json | cut -c1-300)"
$PY "$B" dryrun --debug "$D/debug" --out "$J/dry" --parallel 4 > "$O/dryrun.agg.json" 2> "$O/dryrun.err"; t "dryrun: rc=$? $(tr -d '\n ' < $O/dryrun.agg.json)"
cp "$J/dry/dryrun.json" "$O/" 2>/dev/null

# 5. the scripted episodes (two tasks named in $W/smoke_tasks.txt, one attempt each)
$PY "$B" stub-serve --port 8123 > "$O/stub.log" 2>&1 &
SPID=$!
sleep 20
$PY "$B" run --purpose smoke --k 1 --seed-offset 1 --concurrency 1 --slots 4 --tasks "$(cat $W/smoke_tasks.txt)" --vllm-url http://127.0.0.1:8123 \
  --expect-harness "$(cat $W/harness.txt)" --manifest "$D/debug/manifest.json" --debug "$D/debug" --out "$J/runs" > "$O/smoke.log" 2>&1
t "smoke: rc=$? $(tail -2 $O/smoke.log | tr '\n' ' ')"
kill $SPID 2>/dev/null
rsync -a "$J/runs/" "$O/runs/" 2>/dev/null
cat "$J"/runs/smoke/*/episodes.jsonl 2>/dev/null | $PY -c "
import json, sys
rows = [json.loads(x) for x in sys.stdin if x.strip()]
print('smoke episodes', len(rows), 'graded', sum(r['reward'] is not None for r in rows), 'masked', sum(bool(r['masked']) for r in rows),
      'network gate fired', sum((r.get('gates') or {}).get('path:network') == 0 for r in rows))" | tee "$O/smoke.summary"
docker ps
rm -rf "$J"
t PREP_JOB_DONE
