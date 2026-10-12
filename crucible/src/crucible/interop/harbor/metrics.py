"""A Harbor job metric that honours Crucible's mask (M7 audit, sweep pick M7-1; DESIGN §4.3).

A masked grade has no ``reward.json``. Stock Harbor's ``Mean`` counts that trial as 0 (``harbor/metrics/base.py``), so a
job's score mixes infrastructure failures with real zeros. Harbor's ``UvScript`` metric hands a script one JSON row per
trial (``null`` for a trial with no reward) on ``-i`` and reads a JSON object from ``-o``
(``harbor/metrics/uv_script.py:17-47``). ``MASKED_MEAN_PY`` is that script: the mean over graded trials plus the mask
rate. ``masked_mean`` is the same computation in process, so tests and reports use one rule.

rLLM's ``ShellScriptEvaluator`` cannot mask at all: no reward file means 0 there
(``rllm/eval/script_evaluator.py:136-137``). The export card says so; ``plan.prewrite_zero`` removes the agent-writable
``/tmp/rllm/reward.json`` that rLLM reads first.
"""

from __future__ import annotations

import math
from pathlib import Path

MASKED_MEAN_PY = '''# /// script
# requires-python = ">=3.10"
# ///
"""Crucible masked mean: mean reward over graded trials (null rows are masked, not 0), plus the mask rate."""
import argparse, json, math

ap = argparse.ArgumentParser()
ap.add_argument("-i", required=True)
ap.add_argument("-o", required=True)
a = ap.parse_args()
vals, masked, n = [], 0, 0
for line in open(a.i):
    line = line.strip()
    if not line:
        continue
    n += 1
    row = json.loads(line)
    v = row.get("reward") if isinstance(row, dict) else None
    if isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v):
        vals.append(float(v))
    else:
        masked += 1
out = {"masked_mean": sum(vals) / len(vals) if vals else None, "graded": len(vals), "masked": masked,
       "mask_rate": masked / n if n else 0.0}
json.dump(out, open(a.o, "w"))
'''


def masked_mean(rows: list[dict | None]) -> dict:
    """One row per trial (``None`` or a row without a finite ``reward`` is masked)."""
    vals = [float(r["reward"]) for r in rows if isinstance(r, dict) and isinstance(r.get("reward"), (int, float))
            and not isinstance(r.get("reward"), bool) and math.isfinite(r["reward"])]
    n = len(rows)
    return {"masked_mean": sum(vals) / len(vals) if vals else None, "graded": len(vals), "masked": n - len(vals),
            "mask_rate": (n - len(vals)) / n if n else 0.0}


def write_masked_mean(dest: Path) -> Path:
    """Write the metric script (reference it from the job config as a ``uv-script`` metric)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(MASKED_MEAN_PY)
    return dest
