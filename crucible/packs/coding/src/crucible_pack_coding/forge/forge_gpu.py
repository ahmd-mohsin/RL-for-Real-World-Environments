"""Behavioural acceptance of the derived task04 span tasks in the verifier image (the GPU half of the M8 coding gate).

    python -m crucible_pack_coding.forge.forge_gpu stage | status | collect

Each span grade runs the kit task04 grader with the span's ``scope``, the span's ``start.patch`` as ``--pre-patch``, and
the solution under test as ``--patch`` (relative to the start): the oracle must score 1 (A1), the empty patch and the
unported mutants 0 at ``static`` (A2/A3), the frontier trap reversion 0 at ``ncv``, Astra's people/NCV port 1 (valid
alternative) and Astra's KinFu ports 0 at ``perf``. The runner starts after the golden runner finishes and waits for
an idle GPU before every grade (the lab is shared).
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
LAB = [str(CRUCIBLE.parent / "scripts" / "lab_gpu.sh")]
REMOTE = "rlenv/crucible_forge"
SPANS = CRUCIBLE / "data" / "forge" / "coding" / "task04_spans"
REPORT = CRUCIBLE / "docs" / "modules" / "reports" / "m8_coding_gpu.json"
IMAGE = "task04-pcl-gpu-cuda12-verifier:v1"

RUNNER = r"""#!/bin/bash
cd ~/REMOTE
echo $$ > runner.pid
while kill -0 $(cat ~/rlenv/crucible_golden/runner.pid 2>/dev/null) 2>/dev/null; do echo "$(date -Is) waiting for the golden runner" > status.txt; sleep 300; done
while kill -0 $(cat ~/rlenv/crucible_labstack/runner.pid 2>/dev/null) 2>/dev/null; do echo "$(date -Is) waiting for the labstack campaign" > status.txt; sleep 300; done
while read SPAN SOL FILE; do
  N=${SPAN}__${SOL}
  [ -f out/$N/crucible_reward.json ] && continue
  while true; do
    read U M <<< $(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits | head -1 | tr -d ',')
    L=$(cut -d' ' -f1 /proc/loadavg)
    if [ "$U" -le 10 ] && [ "$M" -le 3000 ] && [ "$(echo "$L < 8" | bc)" = 1 ]; then break; fi
    echo "$(date -Is) waiting: gpu ${U}% ${M}MiB load $L" > status.txt; sleep 120
  done
  echo "$(date -Is) grading $N" > status.txt
  mkdir -p out/$N
  docker run --rm --name crucible_forge_$N --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --network none --init --shm-size 8g \
    -v $HOME/REMOTE/bundle:/crucible_verify:ro -v $HOME/rlenv/task04/taskdir/tests:/tests:ro -v $HOME/REMOTE/spans/$SPAN:/span:ro \
    -v $HOME/REMOTE/out/$N:/logs/verifier --entrypoint python3 IMAGE -I /crucible_verify/crucible/verify/runtime/run.py \
    --plan crucible_pack_coding.verify.task04 --out /logs/verifier --require-root --base /opt/originals/pcl --pre-patch /span/start.patch \
    --patch /span/$FILE --param tests=/tests --param scope=$SPAN > out/$N.console 2>&1
done < queue.txt
echo "$(date -Is) done" > status.txt
"""


def expected(span: str) -> dict[str, tuple[str | None, str]]:
    """solution name -> (gate it must fail | None, expectation)."""
    negs = json.loads((SPANS / span / "crucible" / "negatives.json").read_text()) if (SPANS / span / "crucible" / "negatives.json").exists() else None
    import yaml

    task = yaml.safe_load((SPANS / span / "crucible" / "task.yaml").read_text())
    out = {"solution": (None, "pass")}
    for n in negs or task.get("negatives", []):
        out[n["id"]] = (n.get("expected_gate"), n.get("expected", "fail"))
    return out


def stage() -> None:
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        bundle(d / "bundle", {"crucible_pack_coding.verify.task04": PACK_ROOT / "src" / "crucible_pack_coding" / "verify" / "task04.py"})
        queue = []
        for span_dir in sorted(p for p in SPANS.iterdir() if p.is_dir()):
            s = span_dir.name
            tgt = d / "spans" / s
            tgt.mkdir(parents=True)
            (tgt / "start.patch").write_text((span_dir / "start.patch").read_text())
            (tgt / "solution.patch").write_text((span_dir / "solution" / "solution.patch").read_text())
            queue.append(f"{s} solution solution.patch")
            (tgt / "neg").mkdir()
            for neg in sorted((span_dir / "crucible" / "negatives").glob("*.patch")):
                # negatives live in neg/: the empty-patch negative is named start.patch and, copied next to the span's own
                # start.patch, overwrote it (every grade of the first run masked: "No valid patches in input")
                (tgt / "neg" / neg.name).write_text(neg.read_text())
                queue.append(f"{s} {neg.stem} neg/{neg.name}")
        empty = sorted(str(q.relative_to(d)) for q in (d / "spans").glob("*/*.patch") if q.stat().st_size == 0)  # start / solution (an empty negative is legitimate)
        if empty:  # the first staging shipped empty start patches (regenerated mid-stage): every grade masked
            raise SystemExit(f"refusing to stage empty patches: {empty}")
        (d / "queue.txt").write_text("\n".join(queue) + "\n")
        (d / "run.sh").write_text(RUNNER.replace("REMOTE", REMOTE).replace("IMAGE", IMAGE))
        import os

        tar = subprocess.run(["tar", "-czf", "-", "--no-xattrs", "-C", str(d), "bundle", "spans", "queue.txt", "run.sh"], capture_output=True, check=True,
                             env={**os.environ, "COPYFILE_DISABLE": "1"}).stdout  # no macOS AppleDouble ._ files
        unpack = (f"mkdir -p ~/{REMOTE} && cd ~/{REMOTE} && rm -rf bundle spans && tar xzf - && chmod +x run.sh && find . -name '._*' -delete && "
                  "test -z \"$(find spans -maxdepth 2 -name '*.patch' -size 0)\" && rm -rf out && echo staged-ok")  # out/: masked results of the bad staging
        r = subprocess.run([*LAB, unpack], input=tar, capture_output=True, check=False, timeout=300)
        if r.returncode or b"staged-ok" not in r.stdout:
            raise SystemExit("staging failed or a patch arrived empty: " + (r.stderr or r.stdout).decode()[-500:])
    start = (f"cd ~/{REMOTE} && (kill -0 $(cat runner.pid 2>/dev/null) 2>/dev/null || setsid nohup ./run.sh > runner.log 2>&1 < /dev/null &); "
             "sleep 3; cat status.txt 2>/dev/null; wc -l queue.txt")
    r = subprocess.run([*LAB, start], capture_output=True, text=True, check=False, timeout=60)
    print(r.stdout or r.stderr)


def status() -> None:
    print(subprocess.run([*LAB, f"cd ~/{REMOTE} && cat status.txt; ls out | grep -v console | wc -l; wc -l < queue.txt"], capture_output=True, text=True,
                         check=False, timeout=60).stdout)


def collect() -> dict:
    tar = subprocess.run([*LAB, f"cd ~/{REMOTE} && tar czf - out 2>/dev/null"], capture_output=True, check=False, timeout=600).stdout
    dest = CRUCIBLE / "data" / "forge" / "coding" / "gpu"
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", "xzf", "-", "-C", str(dest)], input=tar, check=False)
    rows = {}
    for span_dir in sorted(p for p in SPANS.iterdir() if p.is_dir()):
        for sol, (gate, exp) in expected(span_dir.name).items():
            f = dest / "out" / f"{span_dir.name}__{sol}" / "crucible_reward.json"
            if not f.exists():
                rows[f"{span_dir.name}/{sol}"] = {"pending": True}
                continue
            r = json.loads(f.read_text())
            if r.get("infra", {}).get("status") == "masked":  # infrastructure, never a verdict: re-grade
                rows[f"{span_dir.name}/{sol}"] = {"masked": r["infra"].get("failure_kind"), "reason": str(r["infra"].get("reason"))[:200]}
                continue
            ok = (r["reward"] == 1.0) if exp == "pass" else (r["reward"] == 0.0 and r.get("infra", {}).get("status") == "ok" and (gate is None or r["gates"].get(gate) == 0))
            rows[f"{span_dir.name}/{sol}"] = {"reward": r["reward"], "gates": r["gates"], "infra": r.get("infra"), "expected": exp, "expected_gate": gate, "ok": ok}
    done = [v for v in rows.values() if "ok" in v]
    rep = {"graded": f"{len(done)}/{len(rows)}", "pass": len(done) == len(rows) and all(v["ok"] for v in done), "grades": rows}
    REPORT.write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps({k: (v.get("ok"), v.get("reward")) if "ok" in v else ("masked: " + v["masked"] if "masked" in v else "pending") for k, v in rows.items()}, indent=1))
    return rep


if __name__ == "__main__":
    {"stage": stage, "status": status, "collect": collect}[sys.argv[1] if len(sys.argv) > 1 else "status"]()
