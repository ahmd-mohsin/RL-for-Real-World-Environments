"""Run-to-run comparison: a baseline and a candidate on the same benchmark, **clustered by task**.

The NeMo-Gym ``gym eval compare`` shape (per-task flips with a majority rule, regressions first, md + json;
``comparison/diff.py:205-359``), with the fixes the M13 audit found necessary (``docs/audit/2026-10-13``):

- **One declared model per side** (H1 V7, G1 D1). Each side must hold exactly one (policy, harness); a side with more,
  or two episodes with the same (task, sample index), is refused with the names, never silently collapsed (the dict
  used to keep whichever episode came last, so an identical run showed "significant" flips).
- **The task is the unit** (G1 D7, G2-8). Two policies share no seeds (``derive_seed`` hashes the policy), so sample i
  of the base and sample i of the candidate are not a pair, and a task's k samples are not independent. Flips are per
  task (pass rate above ``majority`` on one side and not the other; a rate exactly at ``majority`` is a tie); the exact
  McNemar test runs on those task flips; the delta is the mean of per-task mean deltas with a t interval and an exact
  (or seeded Monte Carlo) sign-flip permutation p-value over tasks.
- **Common support** (G2-1). Only tasks measured on both sides are compared; ``support`` lists the tasks each side
  measured alone. An exposed episode is a scored failure (``Episode.value`` = 0), never dropped.
- McNemar is computed in log space for large counts (G1 D14: big-integer binomials took 2.9 s at b + c = 10 200) and is
  not rounded (H1 V18: 2^-39 printed as 0.0); ``fmt_p`` prints "< 1e-9".
"""

from __future__ import annotations

import itertools
import math
import random
from collections.abc import Iterable, Sequence
from typing import Any

from crucible.bench.metrics import Episode, task_mean_ci


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (pass->fail) and c (fail->pass) discordant units (binomial, p = 1/2)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    if n <= 1000:
        p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    else:  # log space: lgamma terms and a log-sum-exp (D14)
        logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
        top = max(logs)
        p = math.exp(top) * sum(math.exp(x - top) for x in logs)
    return min(1.0, 2 * p)


def fmt_p(p: float | None) -> str:
    if p is None:
        return "—"
    return "< 1e-9" if p < 1e-9 else f"{p:.3g}"


def sign_flip_p(deltas: Sequence[float], *, iters: int = 20000, seed: int = 0, exact_max: int = 16) -> float | None:
    """Two-sided paired permutation test over tasks: under H0 each task's delta is equally likely to have either sign.
    Exact enumeration up to ``exact_max`` nonzero deltas, else seeded Monte Carlo with the +1 correction."""
    d = [float(x) for x in deltas if x != 0.0]
    if not d:
        return 1.0 if deltas else None
    obs = abs(sum(d)) - 1e-12
    if len(d) <= exact_max:
        hits = sum(1 for signs in itertools.product((1, -1), repeat=len(d)) if abs(sum(s * x for s, x in zip(signs, d, strict=True))) >= obs)
        return hits / 2 ** len(d)
    rng = random.Random(seed)
    hits = sum(1 for _ in range(iters) if abs(sum(x if rng.random() < 0.5 else -x for x in d)) >= obs)
    return (hits + 1) / (iters + 1)


def _one_model(side: str, eps: Sequence[Episode]) -> tuple[str, str] | None:
    models = sorted({(e.policy, e.harness) for e in eps})
    if len(models) > 1:
        raise ValueError(f"the {side} side holds {len(models)} (policy, harness) pairs {[f'{p}@{h}' for p, h in models]}: "
                         f"a comparison needs one per side (filter by policy and harness)")
    seen: set[tuple[str, int]] = set()
    for e in eps:
        key = (e.task, e.sample_idx)
        if key in seen:
            raise ValueError(f"the {side} side holds two episodes for task {e.task!r} sample {e.sample_idx}: they would be paired arbitrarily")
        seen.add(key)
    return models[0] if models else None


def compare(base: Sequence[Episode], cand: Sequence[Episode], *, threshold: float = 1.0, majority: float = 0.5, iters: int = 20000,
            seed: int = 0, exclude_tasks: Iterable[str] = ()) -> dict[str, Any]:
    """``exclude_tasks``: tasks a validity manifest marks invalid (broken or flaky oracle, a no-op that passes; campaign
    B4/B5): left out of every number and listed under ``invalid_tasks``."""
    mb, mc = _one_model("base", base), _one_model("candidate", cand)
    invalid = set(exclude_tasks)
    dropped = sorted({e.task for e in [*base, *cand]} & invalid)
    base = [e for e in base if e.task not in invalid]
    cand = [e for e in cand if e.task not in invalid]
    tb: dict[str, list[Episode]] = {}
    tc: dict[str, list[Episode]] = {}
    for e in base:
        if not e.masked:
            tb.setdefault(e.task, []).append(e)
    for e in cand:
        if not e.masked:
            tc.setdefault(e.task, []).append(e)
    common = sorted(set(tb) & set(tc))
    flips: dict[str, Any] = {"pass_to_fail": [], "fail_to_pass": [], "both_pass": 0, "both_fail": 0, "ties": 0}
    deltas, rate_changes = [], []
    for t in common:
        rb = sum(e.passed(threshold) for e in tb[t]) / len(tb[t])
        rc = sum(e.passed(threshold) for e in tc[t]) / len(tc[t])
        deltas.append(sum(float(e.value or 0.0) for e in tc[t]) / len(tc[t]) - sum(float(e.value or 0.0) for e in tb[t]) / len(tb[t]))
        if rb != rc:
            rate_changes.append({"task": t, "base": round(rb, 6), "cand": round(rc, 6), "n_base": len(tb[t]), "n_cand": len(tc[t]), "delta": round(rc - rb, 6)})
        if majority in (rb, rc):
            flips["ties"] += 1
        elif rb > majority > rc:
            flips["pass_to_fail"].append(t)
        elif rc > majority > rb:
            flips["fail_to_pass"].append(t)
        elif rb > majority:
            flips["both_pass"] += 1
        else:
            flips["both_fail"] += 1
    nb, nc = len(flips["pass_to_fail"]), len(flips["fail_to_pass"])
    mean = sum(deltas) / len(deltas) if deltas else None
    rate_changes.sort(key=lambda r: (r["delta"], r["task"]))  # regressions first
    every = {e.task for e in base} | {e.task for e in cand}
    return {"base": f"{mb[0]}@{mb[1]}" if mb else None, "candidate": f"{mc[0]}@{mc[1]}" if mc else None, "unit": "task",
            "support": {"tasks": len(common), "base_only": sorted(set(tb) - set(tc)), "cand_only": sorted(set(tc) - set(tb)),
                        "neither": sorted(every - set(tb) - set(tc))},
            "samples": {"base_measured": sum(len(v) for v in tb.values()), "cand_measured": sum(len(v) for v in tc.values()),
                        "base_masked": sum(e.masked for e in base), "cand_masked": sum(e.masked for e in cand)},
            "mean_delta": round(mean, 6) if mean is not None else None, "delta_ci": task_mean_ci(deltas),
            "permutation_p": sign_flip_p(deltas, iters=iters, seed=seed),
            "flips": {"pass_to_fail": nb, "fail_to_pass": nc, "net": nc - nb, "both_pass": flips["both_pass"], "both_fail": flips["both_fail"],
                      "ties": flips["ties"], "regressions": sorted(flips["pass_to_fail"]), "improvements": sorted(flips["fail_to_pass"])},
            "rate_changes": rate_changes, "mcnemar_p": mcnemar_exact(nb, nc), "threshold": threshold, "majority": majority, "invalid_tasks": dropped}


def to_markdown(name_base: str, name_cand: str, r: dict[str, Any]) -> str:
    f, s, ci = r["flips"], r["support"], r.get("delta_ci")
    ci_txt = (f"95% t interval over {ci['n_tasks']} tasks [{ci['lo']}, {ci['hi']}]" + (" (approximate: fewer than 10 tasks)" if ci["approximate"] else "")
              if ci else "no interval: fewer than 2 common tasks")
    lines = [f"# {name_cand} vs {name_base}", "",
             (f"- common support: {s['tasks']} tasks measured on both sides (base only {len(s['base_only'])}, candidate only {len(s['cand_only'])}, "
              f"neither {len(s['neither'])})"),
             *([f"- {len(r['invalid_tasks'])} invalid tasks left out (validity manifest): {', '.join(r['invalid_tasks'])}"] if r.get("invalid_tasks") else []),
             f"- mean reward delta (mean of per-task deltas): {r['mean_delta']} ({ci_txt}); sign-flip permutation p = {fmt_p(r['permutation_p'])}",
             (f"- task flips (pass rate above {r['majority']}): fail->pass {f['fail_to_pass']}, pass->fail {f['pass_to_fail']}, net {f['net']}, "
              f"ties {f['ties']} (exact McNemar over tasks p = {fmt_p(r['mcnemar_p'])})"),
             (f"- samples measured: base {r['samples']['base_measured']}, candidate {r['samples']['cand_measured']} "
              f"(masked: base {r['samples']['base_masked']}, candidate {r['samples']['cand_masked']})"), ""]
    if f["regressions"]:
        lines += ["## Regressions (task pass -> fail)", *[f"- {x}" for x in f["regressions"]], ""]
    if f["improvements"]:
        lines += ["## Improvements (task fail -> pass)", *[f"- {x}" for x in f["improvements"]], ""]
    if r["rate_changes"]:
        lines += ["## Pass-rate changes (regressions first)",
                  *[f"- {x['task']}: {x['base']} -> {x['cand']} (n {x['n_base']} / {x['n_cand']})" for x in r["rate_changes"]], ""]
    return "\n".join(lines)
