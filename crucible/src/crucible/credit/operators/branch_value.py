"""C3, branch-rollout values (DESIGN §6.4): V̂(s_t) = the success rate of K continuations started from the World
snapshot at step t. ΔV̂ between consecutive snapshots is an advantage estimate in the style of VinePPO; it costs K
rollouts per branch point, so it is used to label PRM data, to validate C1 (does ΔΦ track ΔV̂?) and on a sampled subset
of steps, never on every step of every rollout.

A branch is ``(WorldRef_t, policy)``: the World is restored from the snapshot (the fingerprint is verified by
``World.restore``) and a *fresh* policy instance continues (``policy_factory(i)``): either with a fresh context plus a
handoff note (measures the value of the *state*, consistent with P12) or seeded with the original conversation prefix
(measures state + context; the gap is the value of the context, research 01 §1.18). Branch only at quiescent steps:
background processes and gitignored build outputs are not part of a snapshot (research 06 §1.5).

Every estimate records the harness (``harness@version``): harness choice moved solve rates by 13 pp in the
multi-harness study, more than a model swap. Masked continuations (infrastructure failures) are excluded from V̂ and
counted; they never count as failures.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.policy import EpisodeLimits, Policy, run_episode


@dataclass
class BranchResult:
    step: int
    k: int
    values: list[float | None] = field(default_factory=list)
    v_hat: float | None = None
    ci: tuple[float, float] | None = None
    masked: int = 0
    harness: str | None = None
    stops: list[str] = field(default_factory=list)


def wilson(successes: float, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


async def branch_value(world: Any, ref: Any, *, step: int, policy_factory: Callable[[int], Policy],
                       verify: Callable[[Any], Awaitable[float | None]], k: int = 8, actor: str = "agent",
                       limits: EpisodeLimits | None = None, harness: str | None = None) -> BranchResult:
    """K continuations from ``ref``; ``verify(world)`` reads the end state through the verifier (1 = success)."""
    out = BranchResult(step=step, k=k, harness=harness)
    try:
        for i in range(k):
            await world.restore(ref)
            res = await run_episode(world, {actor: policy_factory(i)}, reset=False, limits=limits or EpisodeLimits(max_steps=50))
            out.stops.append(res.stop_condition)
            v = None if res.masked else await verify(world)
            if v is None:  # a masked run, or a verifier that could not measure: excluded and counted (M6 audit)
                out.masked += 1
                out.values.append(None)
                continue
            if not (math.isfinite(v) and 0.0 <= v <= 1.0):
                raise ValueError(f"verify() must return a success value in [0, 1] (the outcome, not a composed reward); got {v}")
            out.values.append(v)
    finally:
        await world.restore(ref)  # leave the World at the branch point, not in the last continuation's state
    got = [v for v in out.values if v is not None]
    if got:
        out.v_hat = statistics.fmean(got)
        out.ci = wilson(sum(got), len(got))
    return out


def advantages(results: Sequence[BranchResult]) -> list[float | None]:
    """ΔV̂ between consecutive branch points (None where either side is unmeasured)."""
    out: list[float | None] = []
    prev = None
    for r in results:
        out.append(None if r.v_hat is None or prev is None or prev.v_hat is None else r.v_hat - prev.v_hat)
        prev = r
    return out


def correlation(dv: Sequence[float | None], dphi: Sequence[float], kinds: Sequence[str] | None = None) -> dict[str, Any]:
    """Pearson and Spearman correlation of ΔV̂ and ΔΦ over the measured steps, overall and split into state-changing
    vs investigation steps (ΔΦ is 0 by construction on investigation steps; research 01 §1.8)."""

    def corr(xs: list[float], ys: list[float]) -> dict[str, float | None]:
        if len(xs) < 3 or statistics.pstdev(xs) == 0 or statistics.pstdev(ys) == 0:
            return {"n": len(xs), "pearson": None, "spearman": None}

        def rank(v: list[float]) -> list[float]:
            order = sorted(range(len(v)), key=lambda i: v[i])
            r = [0.0] * len(v)
            i = 0
            while i < len(order):
                j = i
                while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                    j += 1
                for t in range(i, j + 1):
                    r[order[t]] = (i + j) / 2
                i = j + 1
            return r

        return {"n": len(xs), "pearson": round(statistics.correlation(xs, ys), 4), "spearman": round(statistics.correlation(rank(xs), rank(ys)), 4)}

    pts = [(v, p, (kinds[i] if kinds else "?")) for i, (v, p) in enumerate(zip(dv, dphi, strict=True)) if v is not None]
    out: dict[str, Any] = {"all": corr([p for _, p, _ in pts], [v for v, _, _ in pts])}
    if kinds:
        changing = [(v, p) for v, p, k in pts if k in ("edit", "build", "test", "run", "act", "submit", "commit")]
        out["state_changing"] = corr([p for _, p in changing], [v for v, _ in changing])
        out["investigate_dv_mean"] = round(statistics.fmean([v for v, _, k in pts if k == "investigate"]), 4) if any(k == "investigate" for _, _, k in pts) else None
    return out
