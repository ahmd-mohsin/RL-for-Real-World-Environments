"""Difficulty in W&B (swarm-pipeline/12 §5): p̂ over time per task, band membership, curriculum and controller actions,
and the predictor's error. Plain rows in, so this module needs nothing above ``observe``:

- ``rate_rows``: ``forge.difficulty.TaskRate.row`` dicts, one per (task, step) (a trainer's ``difficulty.jsonl``-side
  tracker dumps or the sampler's ``tracker.rows(step)`` per step);
- ``actions``: ``factory.difficulty.KnobAction.to_json`` rows and sampler events (``start_earlier`` / ``start_later``);
- ``predictor``: ``factory.difficulty.predictor_report``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from crucible.observe.run import ObsRun


def p_hat_over_time(run: ObsRun, rate_rows: Sequence[Mapping[str, Any]], *, band: tuple[float | None, float | None] = (0.1, 0.7),
                    key: str = "difficulty") -> list[list[Any]]:
    steps = sorted({int(r["step"]) for r in rate_rows if r.get("step") is not None})
    tasks = sorted({str(r["task"]) for r in rate_rows})
    at = {(str(r["task"]), int(r["step"])): r for r in rate_rows if r.get("step") is not None}
    lo, hi = band
    rows = []
    for r in rate_rows:
        p = r.get("p")
        inside = None if p is None else not ((lo is not None and float(r.get("hi", 1.0)) < lo) or (hi is not None and float(r.get("lo", 0.0)) > hi))
        rows.append([r.get("task"), r.get("step"), p, r.get("lo"), r.get("hi"), r.get("n_eff"), r.get("phi_mean"), r.get("phi_var"), r.get("signal"),
                     inside, r.get("masked"), r.get("budget"), json.dumps(r.get("conditional") or {}, default=str)])
    run.table(f"{key}/p_hat", ["task", "step", "p", "lo", "hi", "n_eff", "phi_mean", "phi_var", "signal", "in_band", "masked", "budget",
                               "p_given_prereqs"], rows)
    if steps and tasks:
        run.lines(f"{key}/p_hat_lines", steps, [[(at.get((t, s)) or {}).get("p") for s in steps] for t in tasks], tasks, title="online p̂ per task",
                  xname="step")
    last = {}
    for r in rate_rows:
        t = str(r["task"])
        if t not in last or int(r.get("step") or 0) >= int(last[t].get("step") or 0):
            last[t] = r
    inband = [t for t, r in last.items() if r.get("p") is not None and not ((lo is not None and float(r.get("hi", 1.0)) < lo)
                                                                          or (hi is not None and float(r.get("lo", 0.0)) > hi))]
    run.summary({f"{key}/tasks": len(last), f"{key}/in_band": len(inband), f"{key}/measured": sum(1 for r in last.values() if r.get("p") is not None)})
    return rows


def actions(run: ObsRun, rows: Sequence[Mapping[str, Any]], *, key: str = "difficulty") -> list[list[Any]]:
    out = [[r.get("task"), r.get("action") or r.get("event"), ",".join(r.get("knobs") or []), r.get("p"), r.get("lo"), r.get("hi"), r.get("n"),
            r.get("generator_reward"), str(r.get("reason") or "")[:500], r.get("step")] for r in rows]
    run.table(f"{key}/actions", ["task", "action", "knobs", "p", "lo", "hi", "n", "generator_reward", "reason", "step"], out)
    kinds: dict[str, int] = {}
    for r in out:
        kinds[str(r[1])] = kinds.get(str(r[1]), 0) + 1
    if kinds:
        run.bar(f"{key}/action_mix", list(kinds), [float(v) for v in kinds.values()], title="difficulty actions", label="action", value="count")
    return out


def predictor(run: ObsRun, report: Mapping[str, Any], *, key: str = "difficulty") -> list[list[Any]]:
    rows = [[p.get("task"), p.get("p_true"), p.get("p_pred"), p.get("baseline"), p.get("n")] for p in report.get("predictions") or []]
    run.table(f"{key}/predictor_loo", ["task", "p_true", "p_pred", "baseline", "n"], rows)
    run.summary({f"{key}/predictor_mae": report.get("mae"), f"{key}/predictor_mae_baseline": report.get("mae_baseline"),
                 f"{key}/predictor_tasks": report.get("tasks")})
    return rows
