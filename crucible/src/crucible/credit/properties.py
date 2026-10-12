"""Reward properties R1-R6 (DESIGN §6.4): every task's reward function must pass all six before it trains anything.

A reward function here is ``reward_of(state) -> RewardJson`` over *end states* (P12/P23: reward is a function of the
verified state, not of the trajectory's text), plus the milestone set for the shaping properties.

R1 the oracle end state earns the maximum (outcome 1, reward ≥ 1) and its measured progress is Φ = 1.
R2 the no-op (the base state, nothing done) earns 0.
R3 every negative fails and stays below the success floor. A **crafted** negative (a known-bad end state: a policy
   violation, a reversal, a cheat) must be caught by a gate and score exactly 0; an **honest** failure (a real attempt
   that did not succeed) keeps the bounded partial credit κΦ of §6.4 (M6 audit: DESIGN said 0 for all, the code accepted
   κΦ for all, and the toy's crafted negatives earned 0.13-0.2).
R4 oscillation does not pay: a rollout that breaks and re-fixes things earns no more dense credit than a monotone one
   reaching the same states (ΔΦ and shaping telescope).
R5 stubbing does not pay: satisfying milestones whose prerequisites are unmet adds nothing to Φ.
R6 text does not pay: two rollouts with the same end state but different claims in their messages earn the same reward.
R7 progress cannot be farmed: a mutant that only reaches the shallow milestones (the edit's location, a created file, a
   premature "resolved") earns at most ε (OpenEnv's field evidence: a "+0.2 for submitting anything" term taught agents to
   quick-submit, 0.740 -> 0.178).

A property with nothing to test (no prerequisites for R5, no negatives for R3, a reward that cannot read messages for
R6) is reported as such (``tested: False``), not as passed (M6 audit: Φ ≡ 0 still passed all six).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from crucible.core.schema.milestone import Milestone
from crucible.core.schema.reward import RewardJson
from crucible.workflow.potential import potential, shaping, trajectory

TOL = 1e-9


@dataclass
class PropertyResult:
    prop: str
    ok: bool
    detail: dict[str, Any]
    tested: bool = True  # False: nothing to test here (reported, never counted as a pass)


def r1_oracle(reward_of: Callable[[Any], RewardJson], oracle_state: Any) -> PropertyResult:
    rj = reward_of(oracle_state)
    phi_ok = rj.progress is None or rj.progress >= 1.0 - TOL  # the oracle reaches every milestone
    return PropertyResult("R1", rj.outcome == 1.0 and rj.reward is not None and rj.reward >= 1.0 - TOL and rj.infra.status == "ok" and phi_ok,
                          {"reward": rj.reward, "outcome": rj.outcome, "gates": rj.gates, "progress": rj.progress})


def r2_noop(reward_of: Callable[[Any], RewardJson], base_state: Any) -> PropertyResult:
    rj = reward_of(base_state)
    return PropertyResult("R2", abs(rj.reward) <= TOL and rj.infra.status == "ok", {"reward": rj.reward, "outcome": rj.outcome, "progress": rj.progress})


def r3_negatives(reward_of: Callable[[Any], RewardJson], negatives: Mapping[str, Any], *, crafted: Sequence[str] = ()) -> PropertyResult:
    """``crafted``: the names of known-bad end states, which a gate must catch (reward exactly 0); the rest are honest
    failures, which must fail and stay below the success floor."""
    if not negatives:
        return PropertyResult("R3", False, {"reason": "no negatives to test"}, tested=False)
    out = {name: reward_of(st) for name, st in negatives.items()}

    def succeeded(rj: RewardJson) -> bool:
        return rj.outcome == 1.0 and (not rj.gates or min(rj.gates.values()) == 1)

    bad = {n: rj.reward for n, rj in out.items()
           if rj.reward is None or succeeded(rj) or rj.reward >= 1.0 - TOL or (rj.gates and min(rj.gates.values()) == 0 and rj.reward > TOL)
           or (n in crafted and rj.reward > TOL)}
    return PropertyResult("R3", not bad, {"negatives": {n: {"reward": rj.reward, "outcome": rj.outcome, "crafted": n in crafted} for n, rj in out.items()},
                                          "violations": bad})


def r4_oscillation(milestones: Sequence[Milestone], monotone: Sequence[Mapping[str, float | None]],
                   oscillating: Sequence[Mapping[str, float | None]]) -> PropertyResult:
    """Both series must end in the same state; their summed ΔΦ and shaping must not favour oscillation."""
    a, b = trajectory(milestones, monotone), trajectory(milestones, oscillating)
    da, db = sum(s.delta for s in a), sum(s.delta for s in b)
    sa, sb = sum(shaping([s.phi for s in a])), sum(shaping([s.phi for s in b]))
    same_end = abs(a[-1].phi - b[-1].phi) <= TOL
    return PropertyResult("R4", same_end and db <= da + TOL and sb <= sa + TOL, {"sum_delta": [da, db], "sum_shaping": [sa, sb], "same_end": same_end})


def r5_stubbing(milestones: Sequence[Milestone], with_stubs: Mapping[str, float | None]) -> PropertyResult:
    """``with_stubs`` satisfies some milestones whose prerequisites are unmet; removing those stubs must not lower Φ."""
    if not any(m.requires for m in milestones):
        return PropertyResult("R5", True, {"reason": "no milestone has a prerequisite: nothing can be stubbed", "by_construction": True}, tested=False)
    # a stub: satisfied while a prerequisite is entirely unmet (φ 0 or unmeasured); a prerequisite at 0.5 gives its
    # milestone proportional credit by the formula, which is not stubbing (M6 audit K13)
    pre_unmet = {m.id for m in milestones if (with_stubs.get(m.id) or 0) >= 1 and any(not (with_stubs.get(p) or 0) for p in m.requires)}
    stripped = {k: (0.0 if k in pre_unmet else v) for k, v in with_stubs.items()}
    a, b = potential(milestones, with_stubs)[0], potential(milestones, stripped)[0]
    return PropertyResult("R5", a <= b + TOL, {"phi_with_stubs": a, "phi_without": b, "stubbed": sorted(pre_unmet),
                                               **({} if pre_unmet else {"reason": "the given state stubs nothing"})}, tested=bool(pre_unmet))


def r6_state_only(reward_of_rollout: Callable[[Any, Sequence[str]], RewardJson], end_state: Any, claims: Sequence[Sequence[str]],
                  *, reads_messages: bool = True) -> PropertyResult:
    """``reward_of_rollout(end_state, messages)``: the same end state with different message texts must score the same.
    ``reads_messages=False``: the reward function has no messages argument at all, so R6 holds by construction and is
    reported as untested rather than passed."""
    rewards = [reward_of_rollout(end_state, msgs).reward for msgs in claims]
    same = max(rewards) - min(rewards) <= TOL
    return PropertyResult("R6", same, {"rewards": rewards, **({} if reads_messages else {"by_construction": True})}, tested=reads_messages)


def r7_phi_farm(milestones: Sequence[Milestone], shallow_phis: Mapping[str, float | None], *, kappa: float, eps: float = 0.1) -> PropertyResult:
    """The partial credit κΦ a shallow mutant reaches (only the shallow milestones met) stays ≤ ε."""
    phi = potential(milestones, shallow_phis)[0] if milestones else 0.0
    return PropertyResult("R7", kappa * phi <= eps + TOL, {"shallow_phi": round(phi, 4), "partial_credit": round(kappa * phi, 4), "eps": eps})


def check_all(results: Sequence[PropertyResult]) -> dict[str, Any]:
    """``ok``: every tested property passed, and every untested one holds by construction (R5 without prerequisites,
    R6 on a state-only reward); a property that could not be tested otherwise (R3 without negatives) is not a pass."""
    untested = [r.prop for r in results if not r.tested]
    missing = [r.prop for r in results if not r.tested and not r.detail.get("by_construction")]
    return {"ok": all(r.ok for r in results if r.tested) and not missing, "untested": untested, "not_tested": missing,
            "properties": {r.prop: {"ok": r.ok, "tested": r.tested, **r.detail} for r in results}}
