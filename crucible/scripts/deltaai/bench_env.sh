# Sourced by bench_prep.sbatch and bench.sbatch (doc 14 §4): the benchmark's own directory, node-local scratch, the
# Apptainer docker shim with its lockdown and CPU pinning, and the job's helpers. Held-out data lives only in $W (mode
# 700) and on the node's /tmp; nothing here names a task.
set -u
W=${CRUCIBLE_BENCH_DIR:-/work/nvme/bjhg/mahmedmohsin/crucible-bench}
HFW=${CRUCIBLE_BENCH_HF:-/work/hdd/bjhg/mahmedmohsin/crucible-infer/hf}   # the student's weights (read, never written)
MODEL=Qwen/Qwen3.6-35B-A3B
J=/tmp/crucible-bench-$SLURM_JOB_ID
umask 077
mkdir -p "$J/state" "$J/ovl" "$J/cfg" "$J/cache" "$J/atmp" "$J/ws" "$W/sif" "$W/out" "$W/runs" "$W/vcache"
export CRUCIBLE_SIF_DIR=$W/sif CRUCIBLE_SHIM_STATE=$J/state CRUCIBLE_SHIM_SCRATCH=$J/ovl CRUCIBLE_SHIM_CONFIGDIR=$J/cfg CRUCIBLE_SHIM_ALLOW=$J
export CRUCIBLE_SHIM_PIN=1   # a World's --cpus 2 becomes 2 pinned cores (no per-container cgroups in fakeroot mode)
export APPTAINER_CACHEDIR=$J/cache APPTAINER_TMPDIR=$J/atmp TMPDIR=$J
export PATH=$W/bin:$W/venv/bin:$PATH
export HF_HOME=$HFW HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PY=$W/venv/bin/python
B=$W/code/crucible/scripts/deltaai/bench.py
t() { echo "[$(date +%H:%M:%S)] $*"; }

# the lane's data on node-local disk without ACLs or xattrs (/work's ACLs ride into the graders' temp trees: smoke 3460204)
stage_data() {
  D=$J/data
  mkdir -p "$D"
  rsync -a --no-acls --no-xattrs "$W/data/debug/" "$D/debug/"
  export CRUCIBLE_DEBUG_ROOT=$D/debug
}

# one SIF per World image (docker save archive + config side file from bench.py bundle --images)
build_sifs() {
  for tar in "$W"/images/*.tar; do
    base=$(basename "$tar" .tar)
    if [ ! -f "$W/sif/$base.sif" ]; then
      t "sif: building $base"
      apptainer -q build "$W/sif/$base.sif" "docker-archive://$tar" >> "$O/sif.log" 2>&1 || { t "sif: $base failed"; return 1; }
    fi
    cp "$W/images/$base.config.json" "$W/sif/$base.sif.json"
  done
}

# the lockdown probe in every World image, through the shim, as an episode starts its World; any problem stops the job
lockdown_all() {
  $PY -c "from crucible.world.lockdown import PROBE; open('$J/ws/probe.sh','w').write(PROBE)"
  ok=1
  for img in $($PY -c "import json; print(' '.join(r['image'] for r in json.load(open('$W/images/images.json'))))"); do
    base=$(echo "$img" | tr '/:' '__')
    docker run --rm --network none -v "$J/ws:$J/ws" "$img" bash "$J/ws/probe.sh" "$HOME" 0 > "$O/lockdown-$base.txt" 2>> "$O/lockdown.err"
    $PY -c "
from crucible.world import lockdown
p = lockdown.verdict(lockdown.parse(open('$O/lockdown-$base.txt').read()), home='$HOME')
print('LOCKDOWN', 'OK' if not p else 'FAILED', '$base', p)" | tee -a "$O/lockdown.verdict"
  done
  ! grep -q "LOCKDOWN FAILED" "$O/lockdown.verdict" && [ "$(grep -c 'LOCKDOWN OK' "$O/lockdown.verdict")" -ge 1 ]
}
