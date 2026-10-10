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
    r = lab(f"cd ~/{REMOTE} && (kill -0 $(cat runner.pid 2>/dev/null) 2>/dev/null || setsid nohup ./run.sh > runner.log 2>&1 < /dev/null &) ; "
            "sleep 3; cat status.txt 2>/dev/null", timeout=60)
    print(r.stdout or r.stderr)


def status() -> None:
    print(lab(f"cd ~/{REMOTE} && cat status.txt; ls out 2>/dev/null").stdout)


def compare(name: str, got: dict) -> dict:
    rec = json.loads((HIST / name / "verifier" / "report.json").read_text())
    want = {g: int(bool(v.get("pass"))) for g, v in rec["gates"].items() if isinstance(v, dict)}
    gates = got.get("gates", {})
    expect_fail = GRADES[name]
    diffs = {g: (want.get(g), gates.get(g)) for g in set(want) | set(gates) if want.get(g) != gates.get(g) and g != "fresh"}
    fresh_ok = not (want.get("fresh") == 1 and gates.get("fresh") == 0)
    intended = (got.get("reward") == 1.0 and got.get("infra", {}).get("status") == "ok") if expect_fail is None else gates.get(expect_fail) == 0
    return {"recorded": want, "kit": gates, "reward": got.get("reward"), "infra": got.get("infra"), "intended_gate": expect_fail,
            "intended": intended, "profile_diffs": diffs, "fresh_consistent": fresh_ok, "match": intended and not diffs and fresh_ok}


def collect() -> dict:
    tar = subprocess.run([*LAB, f"cd ~/{REMOTE} && tar czf - out status.txt 2>/dev/null"], capture_output=True, check=False, timeout=600).stdout
    dest = CRUCIBLE / "data" / "m7" / "golden_gpu"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest)], input=tar, check=False)
    rows = {}
    for name in GRADES:
        f = dest / "out" / name / "crucible_reward.json"
        rows[name] = compare(name, json.loads(f.read_text())) if f.exists() else {"pending": True}
    done = [r for r in rows.values() if "match" in r]
    rep = {"gate": "M7 golden, GPU part (task04 kit grader in the verifier image)", "graded": f"{len(done)}/{len(GRADES)}",
           "pass": len(done) == len(GRADES) and all(r["match"] for r in done), "grades": rows}
    REPORT.write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({k: (v.get("match"), v.get("reward"), v.get("profile_diffs")) if "match" in v else "pending" for k, v in rows.items()}, indent=1))
    return rep


if __name__ == "__main__":
    {"stage": stage, "status": status, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
