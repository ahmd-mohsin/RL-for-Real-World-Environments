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
- ``gar(quality, lambda_max)``: MiMo Groupwise Advantage Redistribution. With binary R, A_i = R_i − mean, P = passing
  set, quality f_i ∈ (0, 1]: λ = Σ_P A_j / Σ_P f_j A_j (capped at ``lambda_max``), A'_i = λ f_i A_i on P, A_i
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


def zero_hacks(is_hack: Callable[[RolloutRecord], Any]) -> GroupOp:
    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        flags = await asyncio.gather(*(_call(is_hack, m) for m in members))
        for m, hacked in zip(members, flags, strict=True):
            if hacked:
                m.reward_components["hack/zeroed"] = float(m.reward or 0.0)
                m.reward = 0.0
                m.advantage = None
        return members

    return op


def grs(rubric: Callable[[RolloutRecord], Any]) -> GroupOp:
    """``rubric(record) -> (solution_score, behaviour_score)`` in [0, 1] each (or None: not scored, reward unchanged)."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        scores = await asyncio.gather(*(_call(rubric, m) for m in members if (m.reward or 0.0) > 0))
        it = iter(scores)
        for m in members:
            if (m.reward or 0.0) <= 0:
                continue  # a failed trajectory stays 0: the rubric never pays for a solution that did not pass
            sc = next(it)
            if sc is None:
                continue
            s_sol, s_beh = (float(x) for x in sc)
            if not (0.0 <= s_sol <= 1.0 and 0.0 <= s_beh <= 1.0):
                raise ValueError(f"rubric scores must be in [0, 1], got {sc}")
            m.reward_components.update({"grs/solution": s_sol, "grs/behaviour": s_beh, "grs/test": float(m.reward or 0.0)})
            m.reward = float(m.reward or 0.0) * s_sol * s_beh
        return members

    return op


def gar(quality: Callable[[RolloutRecord], Any], *, lambda_max: float = 3.0, pass_threshold: float = 1.0) -> GroupOp:
    """``quality(record) -> f ∈ (0, 1]`` for passing rollouts (None: 1.0, no preference). A rollout passes at
    ``reward >= pass_threshold`` (a composed success is >= 1; M6 audit: any reward > 0 passed, so a failure's partial
    credit κΦ got the success's advantage, 0.333 vs 0.333)."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        scored = [m for m in members if m.reward is not None and not m.mask_sample]
        if not scored:
            return members
        rs = [1.0 if (m.reward or 0.0) >= pass_threshold else 0.0 for m in scored]
        mean = sum(rs) / len(rs)
        adv = [r - mean for r in rs]
        passing = [i for i, r in enumerate(rs) if r == 1.0]
        if passing and len(passing) < len(rs):  # a mixed group: redistribute among the passing solutions
            fs = await asyncio.gather(*(_call(quality, scored[i]) for i in passing))
            f = {i: (1.0 if x is None else float(x)) for i, x in zip(passing, fs, strict=True)}
            if any(not 0.0 < v <= 1.0 for v in f.values()):
                raise ValueError(f"GAR quality factors must be in (0, 1], got {sorted(f.values())}")
            lam = sum(adv[i] for i in passing) / sum(f[i] * adv[i] for i in passing)
            capped = lam > lambda_max
            lam = min(lam, lambda_max)
            for i in passing:
                adv[i] = lam * f[i] * adv[i]
                scored[i].reward_components["gar/quality"] = f[i]
            if capped:
                c = sum(adv) / len(adv)
                adv = [a - c for a in adv]
            for i in passing:
                scored[i].reward_components["gar/lambda"] = lam
        for m, a in zip(scored, adv, strict=True):
            m.advantage = a
        return members

    return op


def bt_pairwise(judge: Callable[[RolloutRecord, Any], Any], reference: Any) -> GroupOp:
    """``judge(candidate, reference) -> P(candidate beats reference)`` in [0, 1] (StepFun: GenRM confidence → BT win
    rate); n comparisons per group instead of n²."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        ps = await asyncio.gather(*(_call(judge, m, reference) for m in members))
        for m, p in zip(members, ps, strict=True):
            p = float(p)
            if not 0.0 <= p <= 1.0:
                raise ValueError(f"win probability out of [0, 1]: {p}")
            m.reward_components["bt/win_vs_reference"] = p
            m.reward = p
        return members

    return op


def tournament(compare: Callable[[RolloutRecord, RolloutRecord], Any]) -> GroupOp:
    """``compare(a, b) -> 1`` (a wins), ``0`` (b wins) or ``0.5`` (tie). Round robin; reward = win share."""

    async def op(gid: str, members: list[RolloutRecord]) -> list[RolloutRecord]:
        n = len(members)
        if n < 2:
            return members
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        res = await asyncio.gather(*(_call(compare, members[i], members[j]) for i, j in pairs))
        wins = [0.0] * n
        for (i, j), r in zip(pairs, res, strict=True):
            r = float(r)
            if r not in (0.0, 0.5, 1.0):
                raise ValueError(f"compare must return 1, 0.5 or 0, got {r}")
            wins[i] += r
            wins[j] += 1.0 - r
        for m, w in zip(members, wins, strict=True):
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
