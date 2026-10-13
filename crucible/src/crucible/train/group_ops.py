"""Group grading operators for ``Runtime(group_fn=...)`` (M12-infra B; art. §3.6.3.2, MiMo-V2.6 GRS / GAR, StepFun pairwise
GenRM, Kimi K3 tournaments, §3.7 hack zeroing).

A group operator takes a terminal group's scored records and returns them (same order) with rewritten rewards and/or a
sequence ``advantage``. They compose (``compose``): typically hack zeroing → rubric (GRS) or comparison (BT, tournament)
→ GAR. Judges may be sync or async callables; the runtime runs the stage off the critical path under
``group_deadline_s`` and falls back to the raw advantages when it is late.

- ``zero_hacks(is_hack)``: a confirmed hack's reward becomes 0 and it counts as a failure (MiMo: then the group statistics
  are recomputed by whatever follows).
- ``grs(rubric)``: MiMo Groupwise Reward Synthesis, R_i = R_test · S_sol · S_beh. The product anchors rubric scores to the
  test outcome: a failed trajectory stays 0 however well it reads; passing ones become distinguishable.
- ``gar(quality, lambda_max)``: MiMo Groupwise Advantage Redistribution (one implementation: ``credit.gar.gar``). Binary
  R from the raw outcome (``raw/reward``, kept by the first stage that rewrites a reward), A_i = R_i − mean, P = passing
  set, quality f_i ∈ (0, 1] (default: the previous stage's score): λ = Σ_P A_j / Σ_P f_j A_j (capped at ``lambda_max``), A'_i = λ f_i A_i on P, A_i
  elsewhere. Σ_P A' = Σ_P A (positive mass conserved, only redistributed by quality; failing rollouts untouched); when
  the cap binds, the advantages are re-centred to zero mean.
- ``bt_pairwise(judge, reference)``: StepFun's pairwise GenRM against a fixed reference, O(n) comparisons; the judge's
  win probability p_i becomes the reward (Bradley-Terry win rate against the reference).
- ``tournament(compare)``: round-robin, O(n²) comparisons (1 / 0.5 / 0 per pair); the reward is the win share.
"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from crucible.core.schema.rollout import RolloutRecord

GroupOp = Callable[[str, list[RolloutRecord]], Awaitable[list[RolloutRecord]]]


async def _call(fn: Callable, *args: Any) -> Any:
    out = fn(*args)
    return await out if inspect.isawaitable(out) else out


def compose(*ops: GroupOp) -> GroupOp:
    async def run(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        for op in ops:
            members = await op(gid, members)
        return members

    return run


RAW = "raw/reward"  # the scored outcome before any group stage rewrote it: GAR's pass/fail is read from here


def _scored(m: RolloutRecord) -> bool:
    """A member the group stage may rewrite: graded and not masked (M12 audit H1 V16: ops wrote rewards onto masked and
    ungraded members, turning a mask into a scored 0)."""
    return m.reward is not None and not m.mask_sample


def _keep_raw(m: RolloutRecord) -> None:
    m.reward_components.setdefault(RAW, float(m.reward or 0.0))


def zero_hacks(is_hack: Callable[[RolloutRecord], Any]) -> GroupOp:
    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        live = [m for m in members if _scored(m)]
        flags = await asyncio.gather(*(_call(is_hack, m) for m in live))
        for m, hacked in zip(live, flags, strict=True):
            if hacked:
                m.reward_components["hack/zeroed"] = float(m.reward or 0.0)
                m.reward_components[RAW] = 0.0  # a hack is a failure for every later stage, GAR's pass set included
                m.reward = 0.0
                m.advantage = None
        return members

    return op


def grs(rubric: Callable[[RolloutRecord], Any]) -> GroupOp:
    """``rubric(record) -> (solution_score, behaviour_score)`` in [0, 1] each (or None: not scored, reward unchanged)."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        live = [m for m in members if _scored(m) and (m.reward or 0.0) > 0]
        scores = await asyncio.gather(*(_call(rubric, m) for m in live))
        for m, sc in zip(live, scores, strict=True):  # a failed trajectory stays 0: the rubric never pays for a solution that did not pass
            if sc is None:
                continue
            s_sol, s_beh = (float(x) for x in sc)
            if not (0.0 <= s_sol <= 1.0 and 0.0 <= s_beh <= 1.0):
                raise ValueError(f"rubric scores must be in [0, 1], got {sc}")
            _keep_raw(m)
            m.reward_components.update({"grs/solution": s_sol, "grs/behaviour": s_beh, "grs/test": float(m.reward or 0.0)})
            m.reward = float(m.reward or 0.0) * s_sol * s_beh
        return members

    return op


def gar(quality: Callable[[RolloutRecord], Any] | None = None, *, lambda_max: float = 3.0, pass_threshold: float = 1.0,
        floor: float = 1e-3) -> GroupOp:
    """``quality(record) -> f ∈ (0, 1]`` for passing rollouts (None: 1.0, no preference). Pass / fail is the **raw
    outcome** (``reward_components["raw/reward"]``, written by the stage that first rewrote the reward; the reward itself
    when no stage did) at ``>= pass_threshold`` (M6 audit: any reward > 0 passed). M12 audit H1 V1 / G2 H1: GAR read the
    rewritten reward, so after GRS (0.855 for a passing solution), BT or a tournament every non-perfect pass was a
    failure: grs→gar gave the passing "ok" solution the failures' advantage and bt→gar zeroed the group. With
    ``quality=None`` the previous stage's score is the quality: reward / raw (GRS S_sol·S_beh, BT win probability,
    tournament win share), floored at ``floor``. The arithmetic is ``credit.gar.gar(mode="redistribute")``."""
    from crucible.credit.gar import gar as _gar

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        scored = [m for m in members if _scored(m)]
        if not scored:
            return members
        raw = [float(m.reward_components.get(RAW, m.reward)) for m in scored]
        passing = [i for i, x in enumerate(raw) if x >= pass_threshold]
        f: list[float | None] = [None] * len(scored)
        if passing and len(passing) < len(scored):  # a mixed group: redistribute among the passing solutions
            if quality is not None:
                got = await asyncio.gather(*(_call(quality, scored[i]) for i in passing))
            else:
                got = [min(1.0, float(scored[i].reward) / raw[i]) if raw[i] > 0 else 1.0 for i in passing]
                got = [max(floor, x) for x in got]
            for i, x in zip(passing, got, strict=True):
                f[i] = 1.0 if x is None else float(x)
            if any(not 0.0 < f[i] <= 1.0 for i in passing):
                raise ValueError(f"GAR quality factors must be in (0, 1], got {sorted(f[i] for i in passing)}")
        binary = [1.0 if x >= pass_threshold else 0.0 for x in raw]
        adv, _ = _gar(binary, f, mode="redistribute", lambda_max=lambda_max, pass_threshold=1.0)
        if passing and len(passing) < len(scored):
            den = sum(f[i] * (binary[i] - sum(binary) / len(binary)) for i in passing)
            lam = min(lambda_max, sum(binary[i] - sum(binary) / len(binary) for i in passing) / den) if den > 0 else 1.0
            for i in passing:
                scored[i].reward_components["gar/quality"] = float(f[i])
                scored[i].reward_components["gar/lambda"] = lam
        for m, a in zip(scored, adv, strict=True):
            m.advantage = a
        return members

    return op


def bt_pairwise(judge: Callable[[RolloutRecord, Any], Any], reference: Any) -> GroupOp:
    """``judge(candidate, reference) -> P(candidate beats reference)`` in [0, 1] (StepFun: GenRM confidence → BT win
    rate); n comparisons per group instead of n²."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        live = [m for m in members if _scored(m)]
        ps = await asyncio.gather(*(_call(judge, m, reference) for m in live))
        for m, p in zip(live, ps, strict=True):
            p = float(p)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"win probability out of [0, 1]: {p}")
            _keep_raw(m)
            m.reward_components["bt/win_vs_reference"] = p
            m.reward = p
        return members

    return op


def tournament(compare: Callable[[RolloutRecord, RolloutRecord], Any]) -> GroupOp:
    """``compare(a, b) -> 1`` (a wins), ``0`` (b wins) or ``0.5`` (tie). Round robin over the scored members (a masked
    member plays no games); reward = win share."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        live = [m for m in members if _scored(m)]
        n = len(live)
        if n < 2:
            return members
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        res = await asyncio.gather(*(_call(compare, live[i], live[j]) for i, j in pairs))
        wins = [0.0] * n
        for (i, j), r in zip(pairs, res, strict=True):
            r = float(r)
            if r not in (0.0, 0.5, 1.0):
                raise ValueError(f"compare must return 1, 0.5 or 0, got {r}")
            wins[i] += r
            wins[j] += 1.0 - r
        for m, w in zip(live, wins, strict=True):
            _keep_raw(m)
            m.reward_components["tournament/wins"] = w
            m.reward = w / (n - 1)
        return members

    return op


def comparisons(n: int, kind: str) -> int:
    """Judge calls per group of n: the cost axis of the two comparison schemes."""
    return n if kind == "bt_pairwise" else n * (n - 1) // 2


def as_group_fn(op: GroupOp) -> Callable[[str, Sequence[RolloutRecord]], Awaitable[list[RolloutRecord]]]:
    async def run(gid: str, members: Sequence[RolloutRecord]) -> list[RolloutRecord]:
        return await op(gid, list(members))

    return run
