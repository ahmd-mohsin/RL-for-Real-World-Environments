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
vals, masked, ambiguous, n = [], 0, 0, 0
for line in open(a.i):
    line = line.strip()
    if not line:
        continue
    n += 1
    row = json.loads(line)
    ok = isinstance(row, dict) and row and all(isinstance(x, (int, float)) and math.isfinite(x) for x in row.values())
    if ok and ("reward" in row or len(row) == 1):  # Harbor's scalar rule: ``reward``, else the single key (bools count)
        vals.append(float(row["reward"] if "reward" in row else next(iter(row.values()))))
    elif ok:
        ambiguous += 1  # several keys and no ``reward``: not a mask, not a score
    else:
        masked += 1
out = {"masked_mean": sum(vals) / len(vals) if vals else None, "graded": len(vals), "masked": masked, "ambiguous": ambiguous,
       "mask_rate": masked / n if n else 0.0}
json.dump(out, open(a.o, "w"))
'''


def masked_mean(rows: list[dict | None]) -> dict:
    """One row per trial. ``None`` or a row with a non-finite value is masked; the scalar is Harbor's (``reward``, else
    the single key; booleans count), and a several-key row without ``reward`` is ambiguous, counted apart (M11 audit
    H1 V12: single-key and boolean rows were masked)."""
    from crucible.interop.harbor.rewards import _finite, row_scalar

    vals, masked, ambiguous = [], 0, 0
    for r in rows:
        v = row_scalar(r)
        if v is not None:
            vals.append(v)
        elif isinstance(r, dict) and r and all(_finite(x) for x in r.values()):
            ambiguous += 1
        else:
            masked += 1
    n = len(rows)
    return {"masked_mean": sum(vals) / len(vals) if vals else None, "graded": len(vals), "masked": masked, "ambiguous": ambiguous,
            "mask_rate": masked / n if n else 0.0}


def write_masked_mean(dest: Path) -> Path:
    """Write the metric script (reference it from the job config as a ``uv-script`` metric)."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(MASKED_MEAN_PY)
    return dest
