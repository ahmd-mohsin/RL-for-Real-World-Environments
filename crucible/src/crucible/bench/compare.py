"""Run-to-run comparison: a baseline and a candidate on the same benchmark.

The NeMo-Gym ``gym eval compare`` shape (flip analysis, regressions first, md + json; ``comparison/diff.py:205-359``),
with three fixes the study found necessary:
- episodes are **joined on (task, sample index)**, not on row order;
- the delta CI is a **paired bootstrap over tasks** (NeMo-Gym's Welch interval is unpaired and wastes the pairing);
- flips get an **exact McNemar** test (two-sided binomial on the discordant pairs).
Masked episodes are dropped from both sides of a pair; the number of pairs lost to masking is reported.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from typing import Any

from crucible.bench.metrics import Episode


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b (pass->fail) and c (fail->pass) discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return round(min(1.0, 2 * p), 9)


def compare(base: Sequence[Episode], cand: Sequence[Episode], *, threshold: float = 1.0, iters: int = 2000, seed: int = 0) -> dict[str, Any]:
    key = lambda e: (e.task, e.sample_idx)
    b = {key(e): e for e in base}
    c = {key(e): e for e in cand}
    common = sorted(set(b) & set(c))
    pairs = [(b[k], c[k]) for k in common if not b[k].masked and not c[k].masked]
    flips = {"pass_to_fail": [], "fail_to_pass": [], "both_pass": 0, "both_fail": 0}
    deltas: dict[str, list[float]] = {}
    for x, y in pairs:
        px, py = x.passed(threshold), y.passed(threshold)
        if px and not py:
            flips["pass_to_fail"].append(f"{x.task}#{x.sample_idx}")
        elif py and not px:
            flips["fail_to_pass"].append(f"{x.task}#{x.sample_idx}")
        elif px:
            flips["both_pass"] += 1
        else:
            flips["both_fail"] += 1
        deltas.setdefault(x.task, []).append(float(y.reward or 0.0) - float(x.reward or 0.0))
    keys = sorted(deltas)
    mean = sum(v for k in keys for v in deltas[k]) / sum(len(deltas[k]) for k in keys) if keys else None
    ci = None
    if len(keys) >= 2:
        rng = random.Random(seed)
        ms = []
        for _ in range(iters):
            vals = [v for k in (keys[rng.randrange(len(keys))] for _ in keys) for v in deltas[k]]
            ms.append(sum(vals) / len(vals))
        ms.sort()
        ci = (round(ms[int(0.025 * iters)], 6), round(ms[int(0.975 * iters)], 6))
    nb, nc = len(flips["pass_to_fail"]), len(flips["fail_to_pass"])
    return {"pairs": len(pairs), "unmatched": len(set(b) ^ set(c)), "lost_to_masking": len(common) - len(pairs),
            "mean_delta": round(mean, 6) if mean is not None else None, "delta_ci95_paired_bootstrap": ci,
            "flips": {"pass_to_fail": nb, "fail_to_pass": nc, "net": nc - nb, "both_pass": flips["both_pass"], "both_fail": flips["both_fail"],
                      "regressions": sorted(flips["pass_to_fail"]), "improvements": sorted(flips["fail_to_pass"])},
            "mcnemar_p": mcnemar_exact(nb, nc), "threshold": threshold}


def to_markdown(name_base: str, name_cand: str, r: dict[str, Any]) -> str:
    f = r["flips"]
    lines = [f"# {name_cand} vs {name_base}", "",
             f"- pairs: {r['pairs']} (unmatched {r['unmatched']}, lost to masking {r['lost_to_masking']})",
             f"- mean reward delta: {r['mean_delta']} (95% paired bootstrap CI {r['delta_ci95_paired_bootstrap']})",
             f"- flips: fail->pass {f['fail_to_pass']}, pass->fail {f['pass_to_fail']}, net {f['net']} (exact McNemar p = {r['mcnemar_p']})", ""]
    if f["regressions"]:
        lines += ["## Regressions (pass -> fail)", *[f"- {x}" for x in f["regressions"]], ""]
    if f["improvements"]:
        lines += ["## Improvements (fail -> pass)", *[f"- {x}" for x in f["improvements"]], ""]
    return "\n".join(lines)
