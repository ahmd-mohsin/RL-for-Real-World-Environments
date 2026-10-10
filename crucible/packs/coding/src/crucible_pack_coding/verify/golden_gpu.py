"""M7 golden gate, GPU part: the kit-expressed task04 grader re-grades the historical submissions in the real verifier
image (``task04-pcl-gpu-cuda12-verifier:v1``) on the lab RTX 5090.

    python -m crucible_pack_coding.verify.golden_gpu stage      # bundle + runner to the lab, start it (it waits politely)
    python -m crucible_pack_coding.verify.golden_gpu status     # progress
    python -m crucible_pack_coding.verify.golden_gpu collect    # pull results, compare with history, write the report

The lab is shared: the runner grades one submission at a time and, before each, waits until the GPU is nearly idle
and has memory free (another user's job is never pushed out). Expected profiles come from the recorded grades; the
``fresh`` gate draws a new secret seed, so it is reported but only required where the recorded profile passed it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from crucible.verify.stage import bundle
from crucible_pack_coding import PACK_ROOT

CRUCIBLE = PACK_ROOT.parent.parent
REPO = CRUCIBLE.parent
LAB = [str(REPO / "scripts" / "lab_gpu.sh")]
REMOTE = "rlenv/crucible_golden"
HIST = CRUCIBLE / "data" / "m7" / "history" / "task04" / "glogs"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m7_golden_gpu.json"
IMAGE = "task04-pcl-gpu-cuda12-verifier:v1"
# submission -> the gate it must fail (None: must pass every gate)
GRADES = {"oracle3": None, "astra_v3b": "perf", "sol_v3": "ncv", "global": "concurrency", "oob": "ncv", "tex": "ncv", "warp": "sanitizer",
          "empty": "static"}  # empty was graded CPU-only first (it never reaches a GPU gate); its result is reused

RUNNER = r"""#!/bin/bash
# Crucible M7 golden runner (task04). One grade at a time; waits for an idle GPU before each.
cd ~/REMOTE
echo $$ > runner.pid
for N in NAMES; do
  [ -f out/$N/crucible_reward.json ] && continue
  while true; do
    read U M <<< $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
    L=$(cut -d' ' -f1 /proc/loadavg)
    if [ "$U" -le 10 ] && [ "$M" -le 3000 ] && [ "$(echo "$L < 8" | bc)" = 1 ]; then break; fi
    echo "$(date -Is) waiting: gpu ${U}% ${M}MiB load $L" > status.txt; sleep 120
  done
  echo "$(date -Is) grading $N" > status.txt
  mkdir -p out/$N art/$N && cp ~/rlenv/task04/patches/$N.patch art/$N/model.patch
  docker run --rm --name crucible_golden_$N --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network none --init --shm-size 8g \
    -v $HOME/REMOTE/bundle:/crucible_verify:ro -v $HOME/rlenv/task04/taskdir/tests:/tests:ro \
    -v $HOME/REMOTE/art/$N:/logs/artifacts:ro -v $HOME/REMOTE/out/$N:/logs/verifier \
    --entrypoint python3 IMAGE -I /crucible_verify/crucible/verify/runtime/run.py --plan crucible_pack_coding.verify.task04 \
    --out /logs/verifier --require-root --base /opt/originals/pcl --patch /logs/artifacts/model.patch --param tests=/tests > out/$N.console 2>&1
done
echo "$(date -Is) done" > status.txt
"""


# task03: chained after the task04 golden and the forge span runner (one GPU job at a time on the shared lab)
T3 = {"remote": "rlenv/crucible_golden_t3", "image": "task03-mdlite-pallas:v1", "plan": "crucible_pack_coding.verify.task03",
      "plan_file": "task03.py", "tests": "$HOME/rlenv/task03/t3/tests", "patches": "$HOME/rlenv/task03", "base": "/opt/originals/mdlite",
      "python": "/opt/venv/bin/python", "after": ["$HOME/rlenv/crucible_golden/runner.pid", "$HOME/rlenv/crucible_forge/runner.pid"],
      "grades": {"solution": None, "empty": "static", "neg_aserfc": "forces", "neg_floatgrid": "forces", "neg_fp32pos": "forces", "neg_jaxrandom": "dynamics",
                 "neg_truncate": "forces", "gpt1": "api", "cheat_tamper": None},
      "history": {"solution": "gradeh_solution", "empty": "grade_empty", "neg_aserfc": "grade_neg_aserfc", "neg_floatgrid": "grade_neg_floatgrid",
                  "neg_fp32pos": "grade_neg_fp32pos", "neg_jaxrandom": "grade_neg_jaxrandom", "neg_truncate": "grade_neg_truncate", "gpt1": "gradeh_gpt1",
                  "cheat_tamper": "gradeh_cheat_tamper"}}
RUNNER_T3 = r"""#!/bin/bash
cd ~/REMOTE
echo $$ > runner.pid
for P in AFTER; do while kill -0 $(cat $P 2>/dev/null) 2>/dev/null; do echo "$(date -Is) waiting for $P" > status.txt; sleep 300; done; done
for N in NAMES; do
  [ -f out/$N/crucible_reward.json ] && continue
  while true; do
    read U M <<< $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
    L=$(cut -d' ' -f1 /proc/loadavg)
    if [ "$U" -le 10 ] && [ "$M" -le 3000 ] && [ "$(echo "$L < 8" | bc)" = 1 ]; then break; fi
    echo "$(date -Is) waiting: gpu ${U}% ${M}MiB load $L" > status.txt; sleep 120
  done
  echo "$(date -Is) grading $N" > status.txt
  mkdir -p out/$N art/$N && cp PATCHES/$N.patch art/$N/model.patch
  docker run --rm --name crucible_golden_t3_$N --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network none --init --shm-size 8g \
    -v $HOME/REMOTE/bundle:/crucible_verify:ro -v TESTS:/tests:ro -v $HOME/REMOTE/art/$N:/logs/artifacts:ro -v $HOME/REMOTE/out/$N:/logs/verifier \
    --entrypoint PYTHON IMAGE -I /crucible_verify/crucible/verify/runtime/run.py --plan PLAN --out /logs/verifier --require-root \
    --base BASE --patch /logs/artifacts/model.patch --param tests=/tests > out/$N.console 2>&1
done
echo "$(date -Is) done" > status.txt
"""


def lab(cmd: str, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([*LAB, cmd], capture_output=True, text=True, check=False, timeout=kw.pop("timeout", 600), **kw)


def stage() -> None:
    with tempfile.TemporaryDirectory() as d:
        bundle(Path(d) / "bundle", {"crucible_pack_coding.verify.task04": PACK_ROOT / "src" / "crucible_pack_coding" / "verify" / "task04.py"})
        (Path(d) / "run.sh").write_text(RUNNER.replace("REMOTE", REMOTE).replace("NAMES", " ".join(GRADES)).replace("IMAGE", IMAGE))
        tar = subprocess.run(["tar", "-czf", "-", "--no-xattrs", "-C", d, "bundle", "run.sh"], capture_output=True, check=True).stdout
        r = subprocess.run([*LAB, f"mkdir -p ~/{REMOTE} && cd ~/{REMOTE} && rm -rf bundle && tar xzf - && chmod +x run.sh"], input=tar,
                           capture_output=True, check=False, timeout=300)
        if r.returncode:
            raise SystemExit(r.stderr.decode()[-500:])
    # a pidfile, not pgrep -f: a pattern can match this very command line (Appendix B.5)
    try:  # the detached runner can hold the ssh session open: a timeout here means it started
        r = lab(f"cd ~/{REMOTE} && (kill -0 $(cat runner.pid 2>/dev/null) 2>/dev/null || setsid nohup ./run.sh > runner.log 2>&1 < /dev/null &) ; "
                "sleep 3; cat status.txt 2>/dev/null", timeout=60)
        print(r.stdout or r.stderr)
    except subprocess.TimeoutExpired:
        print("started (ssh held open by the detached runner)")


def status() -> None:
    print(lab(f"cd ~/{REMOTE} && cat status.txt; ls out 2>/dev/null").stdout)


def compare(name: str, got: dict, statuses: dict | None = None) -> dict:
    """``statuses``: the kit's per-gate status (report.json). A gate the kit *masked* (``infra``: e.g. node_contention when
    another job took the GPU mid-grade) is not a disagreement with history: it is listed in ``needs_regrade`` and the
    grade is re-run on a quiet GPU (M10 ``Runner.regrade`` semantics), never counted as a mismatch."""
    rec = json.loads((HIST / name / "verifier" / "report.json").read_text())
    want = {g: int(bool(v.get("pass"))) for g, v in rec["gates"].items() if isinstance(v, dict)}
    gates = got.get("gates", {})
    masked = sorted(g for g, st in (statuses or {}).items() if st == "infra")
    expect_fail = GRADES[name]
    diffs = {g: (want.get(g), gates.get(g)) for g in set(want) | set(gates) if want.get(g) != gates.get(g) and g != "fresh" and g not in masked}
    fresh_ok = not (want.get("fresh") == 1 and gates.get("fresh") == 0)
    intended = (got.get("reward") == 1.0 and got.get("infra", {}).get("status") == "ok") if expect_fail is None else gates.get(expect_fail) == 0
    return {"recorded": want, "kit": gates, "kit_status": statuses or {}, "reward": got.get("reward"), "infra": got.get("infra"),
            "intended_gate": expect_fail, "intended": intended, "profile_diffs": diffs, "needs_regrade": masked, "fresh_consistent": fresh_ok,
            "match": intended and not diffs and fresh_ok}


def collect() -> dict:
    tar = subprocess.run([*LAB, f"cd ~/{REMOTE} && tar czf - out status.txt 2>/dev/null"], capture_output=True, check=False, timeout=600).stdout
    dest = CRUCIBLE / "data" / "m7" / "golden_gpu"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest)], input=tar, check=False)
    rows = {}
    for name in GRADES:
        f = dest / "out" / name / "crucible_reward.json"
        rp = dest / "out" / name / "report.json"
        statuses = json.loads(rp.read_text()).get("gates") if rp.exists() else None
        rows[name] = compare(name, json.loads(f.read_text()), statuses) if f.exists() else {"pending": True}
    done = [r for r in rows.values() if "match" in r]
    regrade = sorted(k for k, r in rows.items() if r.get("needs_regrade"))
    rep = {"gate": "M7 golden, GPU part (task04 kit grader in the verifier image)", "graded": f"{len(done)}/{len(GRADES)}",
           "pass": len(done) == len(GRADES) and all(r["match"] for r in done) and not regrade, "needs_regrade": regrade,
           "agree_where_measured": all(r["match"] for r in done), "grades": rows}
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: (v.get("match"), v.get("reward"), v.get("profile_diffs"), v.get("needs_regrade")) if "match" in v else "pending"
                      for k, v in rows.items()}, indent=1))
    return rep


def stage_t3() -> None:
    c = T3
    with tempfile.TemporaryDirectory() as d:
        bundle(Path(d) / "bundle", {c["plan"]: PACK_ROOT / "src" / "crucible_pack_coding" / "verify" / c["plan_file"]})
        script = RUNNER_T3
        for k, v in {"REMOTE": c["remote"], "AFTER": " ".join(c["after"]), "NAMES": " ".join(c["grades"]), "PATCHES": c["patches"], "TESTS": c["tests"],
                     "PYTHON": c["python"], "IMAGE": c["image"], "PLAN": c["plan"], "BASE": c["base"]}.items():
            script = script.replace(k, v)
        (Path(d) / "run.sh").write_text(script)
        tar = subprocess.run(["tar", "-czf", "-", "--no-xattrs", "-C", d, "bundle", "run.sh"], capture_output=True, check=True).stdout
        unpack = f"mkdir -p ~/{c['remote']} && cd ~/{c['remote']} && rm -rf bundle && tar xzf - && chmod +x run.sh && ls"
        r = subprocess.run([*LAB, unpack], input=tar, capture_output=True, check=False, timeout=300)
        if r.returncode:
            raise SystemExit(r.stderr.decode()[-500:])
    start = (f"cd ~/{c['remote']} && (kill -0 $(cat runner.pid 2>/dev/null) 2>/dev/null || setsid nohup ./run.sh > runner.log 2>&1 < /dev/null &); "
             "sleep 3; cat status.txt 2>/dev/null")
    try:
        print(subprocess.run([*LAB, start], capture_output=True, text=True, check=False, timeout=30).stdout)
    except subprocess.TimeoutExpired:
        print("started (ssh held open by the detached runner)")


def collect_t3() -> dict:
    c = T3
    tar = subprocess.run([*LAB, f"cd ~/{c['remote']} && tar czf - out 2>/dev/null"], capture_output=True, check=False, timeout=600).stdout
    dest = CRUCIBLE / "data" / "m7" / "golden_gpu_t3"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest)], input=tar, check=False)
    rows = {}
    for name, gate in c["grades"].items():
        f = dest / "out" / name / "crucible_reward.json"
        if not f.exists():
            rows[name] = {"pending": True}
            continue
        got = json.loads(f.read_text())
        rec = json.loads((CRUCIBLE / "data" / "m7" / "history" / "task03" / c["history"][name] / "verifier" / "report.json").read_text())
        want = {g: int(bool(v.get("pass"))) for g, v in rec["gates"].items() if isinstance(v, dict)}
        gates = got.get("gates", {})
        diffs = {g: (want.get(g), gates.get(g)) for g in set(want) | set(gates) if g in want and want.get(g) != gates.get(g) and g != "fresh"}
        intended = (got.get("reward") == 1.0) if gate is None else gates.get(gate) == 0
        rows[name] = {"recorded": want, "kit": gates, "reward": got.get("reward"), "infra": got.get("infra"), "intended": intended, "profile_diffs": diffs,
                      "match": intended and not diffs}
    done = [r for r in rows.values() if "match" in r]
    rep = {"gate": "M7 golden, GPU part (task03 kit grader in the verifier image)", "graded": f"{len(done)}/{len(rows)}",
           "pass": len(done) == len(rows) and all(r["match"] for r in done), "grades": rows}
    (CRUCIBLE / "docs" / "modules" / "reports" / "m7_golden_gpu_t3.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: (v.get("match"), v.get("reward")) if "match" in v else "pending" for k, v in rows.items()}, indent=1))
    return rep


if __name__ == "__main__":
    {"stage": stage, "status": status, "collect": collect, "stage_t3": stage_t3, "collect_t3": collect_t3}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
