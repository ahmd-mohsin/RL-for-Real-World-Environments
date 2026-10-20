#!/bin/bash
# Stage driver for Marlowe jobs (docs/research/swarm-pipeline/05-marlowe.md).
#
#   stage.sh <plan>          # inside an sbatch job (scripts/marlowe/sbatch/*.sbatch); plans: smoke1 | smoke8 | selftest
#
# - Every stage is guarded by a marker: $RUN/markers/<stage>.done is written only when the stage exits 0, and a finished
#   stage never runs again. A requeued job keeps its job id, so it lands in the same $RUN on any node and resumes at the
#   first unfinished stage (state lives on shared storage, never on node-local disk).
# - Preemption: Slurm sends SIGTERM and kills 900 s later (GraceTime). The driver forwards TERM to the running stage's
#   process group; a stage that holds state (training) checkpoints and exits; the driver records the interruption and
#   exits so that Slurm's requeue starts the job again. The time-limit warning (--signal=B:USR1@600) does the same and then
#   requeues the job itself.
# - A failed stage is retried up to CRU_STAGE_TRIES times per job start (default 2); a stage that keeps failing is marked
#   failed and the plan moves on to stages that do not depend on it (each plan lists its dependencies).
# - Logs: $RUN/logs/<stage>.log per stage, $RUN/driver.log for the driver, $RUN/summary.json at the end.
set -uo pipefail
HERE=$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)
source "$HERE/env.sh"
PLAN=${1:?usage: stage.sh <plan>}
JOBID=${SLURM_JOB_ID:-local$$}
RUN=${CRU_RUN:-$CRU_RUNS/${SLURM_JOB_NAME:-$PLAN}-$JOBID}
M=$RUN/markers
LOG=$RUN/logs
mkdir -p "$M" "$LOG"
export CRU_RUN=$RUN CRU_PLAN=$PLAN
TRIES=${CRU_STAGE_TRIES:-2}
START=$(date +%s)
RESTART=${SLURM_RESTART_COUNT:-0}

now() { date +%Y-%m-%dT%H:%M:%S%z; }
say() { echo "[stage $(now)] $*" | tee -a "$RUN/driver.log"; }

CHILD=""
STOPPING=""
forward() {  # $1 = signal name; stop the running stage's process group, give it the grace period to checkpoint
  local sig=$1
  STOPPING=$sig
  say "got SIG$sig (restart $RESTART): forwarding TERM to the running stage ${CURRENT:-none}"
  if [ -n "$CHILD" ] && kill -0 "$CHILD" 2>/dev/null; then
    kill -TERM -- "-$CHILD" 2>/dev/null || kill -TERM "$CHILD" 2>/dev/null
    local t0; t0=$(date +%s)
    while kill -0 "$CHILD" 2>/dev/null; do
      if [ $(( $(date +%s) - t0 )) -gt "${CRU_GRACE_S:-840}" ]; then say "stage did not stop in ${CRU_GRACE_S:-840}s: KILL"; kill -KILL -- "-$CHILD" 2>/dev/null; break; fi
      sleep 2
    done
    say "stage ${CURRENT:-} stopped after $(( $(date +%s) - t0 ))s"
  fi
  echo "{\"stage\": \"${CURRENT:-}\", \"signal\": \"$sig\", \"restart\": $RESTART, \"at\": \"$(now)\", \"host\": \"$(hostname)\"}" >> "$RUN/interruptions.jsonl"
  if [ "$sig" = USR1 ] && [ -n "${SLURM_JOB_ID:-}" ]; then
    say "time limit near: requeueing job $SLURM_JOB_ID"; scontrol requeue "$SLURM_JOB_ID" || say "scontrol requeue failed"
  fi
  exit 143
}
trap 'forward TERM' TERM
trap 'forward USR1' USR1
trap 'forward INT' INT

# stage <name> [--after a,b] -- <command...> : run once, guarded; dependencies must be done
stage() {
  local name=$1; shift
  local after=""
  if [ "${1:-}" = "--after" ]; then after=$2; shift 2; fi
  [ "${1:-}" = "--" ] && shift
  if [ -e "$M/$name.done" ]; then say "$name: done before, skipped"; return 0; fi
  local d
  for d in ${after//,/ }; do
    if [ ! -e "$M/$d.done" ]; then say "$name: skipped, needs $d"; echo "needs $d" > "$M/$name.skipped"; return 0; fi
  done
  local try rc=1
  for try in $(seq 1 "$TRIES"); do
    CURRENT=$name
    say "$name: start (try $try/$TRIES, restart $RESTART, host $(hostname))"
    local t0; t0=$(date +%s)
    if command -v setsid >/dev/null; then setsid "$@" >> "$LOG/$name.log" 2>&1 < /dev/null & else "$@" >> "$LOG/$name.log" 2>&1 < /dev/null & fi
    CHILD=$!  # setsid: the stage leads its own process group, so TERM reaches everything it started
    wait "$CHILD"; rc=$?
    CHILD=""
    say "$name: exit $rc after $(( $(date +%s) - t0 ))s"
    if [ "$rc" -eq 0 ]; then
      rm -f "$M/$name.failed" "$M/$name.skipped"
      echo "{\"stage\": \"$name\", \"seconds\": $(( $(date +%s) - t0 )), \"try\": $try, \"restart\": $RESTART, \"host\": \"$(hostname)\", \"job\": \"$JOBID\"}" > "$M/$name.done"
      return 0
    fi
  done
  echo "exit $rc" > "$M/$name.failed"
  return 0
}

summary() {
  "$PY" - "$RUN" "$PLAN" <<'PY'
import json, pathlib, sys
run, plan = pathlib.Path(sys.argv[1]), sys.argv[2]
st = {}
for f in sorted((run / "markers").iterdir()):
    name, kind = f.name.rsplit(".", 1)
    st[name] = {"status": kind, **(json.loads(f.read_text()) if kind == "done" else {"why": f.read_text().strip()})}
out = {"plan": plan, "run": str(run), "stages": st, "all_done": all(v["status"] == "done" for v in st.values())}
for r in sorted(run.glob("*.receipt.json")):
    out.setdefault("receipts", {})[r.name] = json.loads(r.read_text())
(run / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps({k: v for k, v in out.items() if k != "receipts"}, indent=1))
PY
}

if [ ! -e "$CRU_ROOT/.setup/ready" ] && [ "$PLAN" != selftest ]; then
  say "the environment is not marked ready ($CRU_ROOT/.setup/ready): requeue instead of burning the slot"
  [ -n "${SLURM_JOB_ID:-}" ] && scontrol requeue "$SLURM_JOB_ID"
  exit 0
fi

say "plan $PLAN job $JOBID restart $RESTART host $(hostname) run $RUN"
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1} TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH=$CRU_CODE/src:$CRU_CODE/scripts:$CRU_CODE/scripts/marlowe${PYTHONPATH:+:$PYTHONPATH}
mkdir -p "$APPTAINER_CONFIGDIR"
P=$HERE

case "$PLAN" in
  selftest)  # login node, no GPU: the driver's guards, signals and the CPU World smoke (CRU_SELFTEST_ONLY=<stage> runs one)
    only() { [ -z "${CRU_SELFTEST_ONLY:-}" ] || [ "$CRU_SELFTEST_ONLY" = "$1" ]; }
    only env && stage env -- "$PY" -c "import crucible, torch, transformers; print(torch.__version__, transformers.__version__)"
    only world && stage world --after env -- "$PY" "$P/rollout_smoke.py" --policy scripted --out "$RUN"
    only sleeper && stage sleeper -- bash -c "trap 'echo got TERM; exit 143' TERM; for i in \$(seq 1 ${CRU_SELFTEST_SLEEP:-0}); do sleep 1; done; true"
    ;;
  smoke1)  # 1 GPU: GPU in a container (--nv), vLLM serving Qwen2.5-Coder-7B, one rollout through the Apptainer World, graded
    stage probe -- bash "$P/probe.sh" "$RUN"
    stage gpu_container -- bash "$P/gpu_container.sh" "$RUN"
    stage rollout -- bash "$P/serve_and_rollout.sh" "$RUN"
    ;;
  smoke8)  # 8 GPUs, one node: NCCL, the FSDP2/HSDP unit scenarios, Qwen2.5-7B training steps with save / forced stop / restore
    stage probe -- bash "$P/probe.sh" "$RUN"
    stage nccl -- srun --kill-on-bad-exit=1 --ntasks-per-node=8 "$PY" "$CRU_CODE/scripts/slurm_check.py" nccl --out "$RUN/nccl"
    stage units -- bash "$P/units.sh" "$RUN"
    stage train_a -- bash "$P/train_phase.sh" a "$RUN"
    stage train_b --after train_a -- bash "$P/train_phase.sh" b "$RUN"
    ;;
  *) say "unknown plan $PLAN"; exit 2 ;;
esac
summary | tee -a "$RUN/driver.log"
say "plan $PLAN finished in $(( $(date +%s) - START ))s"
