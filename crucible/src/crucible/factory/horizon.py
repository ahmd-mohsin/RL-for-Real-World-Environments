"""Long-horizon tests (05 §6.7, R11) for tasks labelled long. The horizon must be intrinsic, not merely long.

- **Milestones from World state.** ``milestone_series`` reads which milestones hold after each oracle step, either from
  the ``verify.progress`` stream (``ProgressTracker`` rows: Φ's own subgoals) or by replaying oracle prefixes through a
  ``read(solution) -> {milestone: bool}`` that queries the World. A milestone counts when it is false at the start and
  true at the end. *v1*: ≥ 5.
- **Order and non-redundancy.** Each milestone is first reached at a distinct step (two reached by the same step are one
  milestone as far as the oracle shows), in the declared order when one is declared, and no two milestones have the same
  truth vector over every observed state (oracle prefixes plus any other end states given).
- **Concatenation test.** Each milestone is also posed as an independent short task (``isolated`` rates). If the
  milestones are solved in context at the rates they are solved alone, the task is N short tasks glued together, not a
  long one. Intrinsic iff, for some milestone, the in-context rate (among full-task episodes that reached its
  predecessors) is significantly below its isolated rate, or the full-task success rate is significantly below the
  product of the isolated rates (one-sided tests, Bonferroni over all of them). An unpowered sample therefore fails.
- **Shortcut probe.** No submission shorter than half the oracle's steps may reach a passing end state. Candidates come
  from the forge's ``MutationHacker`` (state writes only, reads skipped, ...), every short prefix, suffix and write
  subsequence of the oracle, and any extra attempts (an LLM shortcut author); every one is graded, and the count is the
  denominator.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from crucible.factory.floors import call, graded_from


@dataclass
class HorizonConfig:
    min_milestones: int = 5  # v1 N
    shortcut_ratio: float = 0.5  # v1: no passing path under half the oracle's steps
    alpha: float = 0.05
    max_shortcut_attempts: int = 64
    pass_at: float = 1.0


def steps_of(solution: Any) -> int:
    """Steps of a tool-call solution: tool calls and messages (a ``done`` is not a step)."""
    return sum(1 for a in solution if isinstance(a, dict) and (a.get("function_name") or a.get("to")))


# --- milestones -----------------------------------------------------------------------------------------------------


def series_from_progress(rows: Sequence[Mapping[str, Any]]) -> list[frozenset[str]]:
    """Milestones holding after each step, from ``verify.progress`` rows (row 0 = the start state)."""
    cur: set[str] = set()
    out: list[frozenset[str]] = []
    for r in rows:
        if r.get("step") == -1:  # an end-of-episode drift correction
            cur |= set(r.get("gained") or [])
            cur -= set(r.get("lost") or [])
            out[-1] = frozenset(cur)
            continue
        cur |= set(r.get("gained") or [])
        cur -= set(r.get("lost") or [])
        out.append(frozenset(cur))
    return out


async def series_by_replay(oracle: Sequence[Any], read: Callable[[Any], Any]) -> list[frozenset[str]]:
    """Milestones holding after each oracle prefix (``read(prefix) -> {milestone: bool}`` replays it in a fresh World)."""
    out = []
    for k in range(len(oracle) + 1):
        m = await call(read, list(oracle[:k]))
        out.append(frozenset(i for i, v in dict(m).items() if v))
    return out


def milestone_test(series: Sequence[frozenset[str]], *, universe: Sequence[str] | None = None, order: Sequence[str] | None = None,
                   other_states: Sequence[frozenset[str]] = (), config: HorizonConfig | None = None) -> dict:
    cfg = config or HorizonConfig()
    if not series:
        return {"ok": False, "reasons": ["no milestone series"], "milestones": []}
    names = list(universe) if universe is not None else sorted(set().union(*series))
    start, end = series[0], series[-1]
    reached = [m for m in names if m not in start and m in end]
    first = {m: next(k for k, s in enumerate(series) if m in s) for m in reached}
    lost = sorted({m for m in names for a, b in itertools.pairwise(series) if m in a and m not in b})
    reasons = []
    if len(reached) < cfg.min_milestones:
        reasons.append(f"{len(reached)} milestones reached from the start state, need {cfg.min_milestones}")
    same_step: dict[int, list[str]] = {}
    for m, k in first.items():
        same_step.setdefault(k, []).append(m)
    simultaneous = [sorted(v) for v in same_step.values() if len(v) > 1]
    if simultaneous:
        reasons.append(f"milestones first reached by the same step (not separable): {simultaneous}")
    out_of_order: list[str] = []
    if order:
        seq = [m for m in order if m in first]
        out_of_order = [f"{a} after {b}" for a, b in itertools.pairwise(seq) if first[a] > first[b]]
        if out_of_order:
            reasons.append(f"the oracle reaches milestones out of the declared order: {out_of_order}")
        missing = [m for m in order if m not in first]
        if missing:
            reasons.append(f"declared milestones the oracle never reaches from the start: {missing}")
    states = list(series) + list(other_states)
    vec = {m: tuple(m in s for s in states) for m in reached}
    redundant = sorted([a, b] for a, b in itertools.combinations(sorted(reached), 2) if vec[a] == vec[b])
    if redundant:
        reasons.append(f"redundant milestones (identical over every observed state): {redundant}")
    return {"ok": not reasons, "reasons": reasons, "milestones": reached, "first_step": first, "regressions": lost,
            "simultaneous": simultaneous, "redundant": redundant, "out_of_order": out_of_order, "states_observed": len(states)}


# --- concatenation --------------------------------------------------------------------------------------------------


def _p_lower(k1: int, n1: int, k2: int, n2: int) -> float:
    """One-sided p-value that rate 1 (k1/n1) is below rate 2 (k2/n2): pooled two-proportion z test."""
    if not n1 or not n2:
        return 1.0
    p = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return 1.0
    z = (k2 / n2 - k1 / n1) / se
    return 1.0 - NormalDist().cdf(z)


def concatenation_test(full: Sequence[frozenset[str] | set[str]], isolated: Mapping[str, tuple[int, int]], *, order: Sequence[str],
                       full_success: Sequence[bool] | None = None, config: HorizonConfig | None = None) -> dict:
    """``full``: milestones reached in each full-task episode; ``isolated``: milestone -> (solved, attempts) as a short
    task on its own; ``order``: the milestone order; ``full_success`` (default: every milestone reached) per episode."""
    from crucible.verify.calibrate import _binom_cdf

    cfg = config or HorizonConfig()
    n_tests = len(order) + 1
    alpha = cfg.alpha / n_tests  # Bonferroni
    rows = []
    for i, m in enumerate(order):
        pre = [e for e in full if all(p in e for p in order[:i])]
        k_ctx, n_ctx = sum(1 for e in pre if m in e), len(pre)
        k_iso, n_iso = isolated.get(m, (0, 0))
        p = _p_lower(k_ctx, n_ctx, k_iso, n_iso)
        rows.append({"milestone": m, "in_context": [k_ctx, n_ctx], "isolated": [k_iso, n_iso], "p_value": round(p, 6), "dependent": p < alpha})
    succ = list(full_success) if full_success is not None else [all(m in e for m in order) for e in full]
    k_full, n_full = sum(succ), len(succ)
    prod = math.prod((isolated[m][0] / isolated[m][1]) if isolated.get(m, (0, 0))[1] else 0.0 for m in order)
    p_full = _binom_cdf(k_full, n_full, prod) if n_full and 0.0 < prod < 1.0 else 1.0
    harder = p_full < alpha
    intrinsic = harder or any(r["dependent"] for r in rows)
    reasons = [] if intrinsic else [(f"concatenation: in context the milestones are solved at their isolated rates (full {k_full}/{n_full} vs "
                                     f"product of isolated rates {prod:.3f}); the task is independent short tasks, not long-horizon")]
    unmeasured = [m for m in order if not isolated.get(m, (0, 0))[1]]
    if unmeasured:
        reasons.append(f"milestones never posed in isolation: {unmeasured}")
        intrinsic = False
    return {"ok": intrinsic and not unmeasured, "intrinsic": intrinsic, "reasons": reasons, "milestones": rows, "full": [k_full, n_full],
            "isolated_product": round(prod, 6), "p_full_below_product": round(p_full, 6), "alpha_per_test": alpha}


# --- shortcut probe -------------------------------------------------------------------------------------------------


def shortcut_candidates(oracle: Sequence[dict], *, config: HorizonConfig | None = None) -> list[tuple[str, list]]:
    """Submissions shorter than ``shortcut_ratio`` of the oracle's steps (deterministic, deduplicated, capped)."""
    from crucible.forge.hacker import MutationHacker, _is_read

    cfg = config or HorizonConfig()
    limit = cfg.shortcut_ratio * steps_of(oracle)
    acts = [dict(a) for a in oracle]
    out: list[tuple[str, list]] = []
    seen: set[str] = set()

    def add(name: str, sol: list) -> None:
        if steps_of(sol) < limit and repr(sol) not in seen and len(out) < cfg.max_shortcut_attempts:
            seen.add(repr(sol))
            out.append((name, sol))

    for h in MutationHacker(max_attempts=10**6).propose({"solver_trajectories": [acts]}):
        add(f"hacker/{h.strategy}", list(h.actions))
    n = len(acts)
    for k in range(1, n):
        add(f"prefix/{k}", acts[:k])
        add(f"suffix/{k}", acts[n - k:])
    writes = [i for i, a in enumerate(acts) if a.get("function_name") and not _is_read(a)]
    for r in range(1, len(writes)):
        for sub in itertools.combinations(writes, r):
            add(f"writes/{'-'.join(map(str, sub))}", [acts[i] for i in sub])
            if len(out) >= cfg.max_shortcut_attempts:
                break
    return out


async def shortcut_probe(oracle: Sequence[dict], grade: Callable[[Any], Awaitable[Any]], *, extra: Sequence[tuple[str, Any]] = (),
                         config: HorizonConfig | None = None) -> dict:
    """Grade every short candidate; any pass is a shortcut."""
    cfg = config or HorizonConfig()
    limit = cfg.shortcut_ratio * steps_of(oracle)
    cands = shortcut_candidates(oracle, config=cfg) + [(n, s) for n, s in extra if steps_of(s) < limit]
    found, masked = [], 0
    for name, sol in cands:
        g = graded_from(await call(grade, sol))
        if g.masked:
            masked += 1
        elif g.passed(cfg.pass_at):
            found.append({"id": name, "steps": steps_of(sol), "solution": sol})
    reasons = [f"shortcut: {f['id']} passes in {f['steps']} steps (< {limit:g})" for f in found]
    if not cands:
        reasons.append("no shortcut candidate could be built under the step floor")
    return {"ok": not reasons, "reasons": reasons, "oracle_steps": steps_of(oracle), "floor": limit, "attempts": len(cands), "masked": masked, "found": found}


def horizon_card(*, milestones: dict, concatenation: dict, shortcut: dict, expert_minutes: float | None = None) -> dict:
    """``VerifierCard.horizon``: every sub-test with its numbers, and the verdict."""
    reasons = milestones.get("reasons", []) + concatenation.get("reasons", []) + shortcut.get("reasons", [])
    return {"ok": not reasons, "reasons": reasons, "oracle_steps": shortcut.get("oracle_steps"), "expert_minutes": expert_minutes,
            "milestones": len(milestones.get("milestones", [])), "milestone_test": milestones, "concatenation": concatenation,
            "shortcut": {k: v for k, v in shortcut.items() if k != "found"} | {"found": [f["id"] for f in shortcut.get("found", [])]}}
