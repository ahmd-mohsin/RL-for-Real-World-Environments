"""Reward properties R1-R6 (DESIGN §6.4): every task's reward function must pass all six before it trains anything.

A reward function here is ``reward_of(state) -> RewardJson`` over *end states* (P12/P23: reward is a function of the
verified state, not of the trajectory's text), plus the milestone set for the shaping properties.

R1 the oracle end state earns the maximum (outcome 1, reward ≥ 1).
R2 the no-op (the base state, nothing done) earns 0.
R3 every registered negative fails: a gate fails (reward exactly 0) or the outcome is 0 (reward ≤ κΦ < 1, the
   bounded partial credit of §6.4); none reaches the success floor.
R4 oscillation does not pay: a rollout that breaks and re-fixes things earns no more dense credit than a monotone one
   reaching the same states (ΔΦ and shaping telescope).
R5 stubbing does not pay: satisfying milestones whose prerequisites are unmet adds nothing to Φ.
R6 text does not pay: two rollouts with the same end state but different claims in their messages earn the same reward.
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


def r1_oracle(reward_of: Callable[[Any], RewardJson], oracle_state: Any) -> PropertyResult:
    rj = reward_of(oracle_state)
    return PropertyResult("R1", rj.outcome == 1.0 and rj.reward >= 1.0 - TOL and rj.infra.status == "ok",
                          {"reward": rj.reward, "outcome": rj.outcome, "gates": rj.gates})


def r2_noop(reward_of: Callable[[Any], RewardJson], base_state: Any) -> PropertyResult:
    rj = reward_of(base_state)
    return PropertyResult("R2", abs(rj.reward) <= TOL and rj.infra.status == "ok", {"reward": rj.reward, "outcome": rj.outcome, "progress": rj.progress})


def r3_negatives(reward_of: Callable[[Any], RewardJson], negatives: Mapping[str, Any]) -> PropertyResult:
    out = {name: reward_of(st) for name, st in negatives.items()}
    def succeeded(rj: RewardJson) -> bool:
        return rj.outcome == 1.0 and (not rj.gates or min(rj.gates.values()) == 1)

    # a negative must fail (a gate or the outcome) and stay below the success floor; a gate failure must give exactly 0
    bad = {n: rj.reward for n, rj in out.items()
           if succeeded(rj) or rj.reward >= 1.0 - TOL or (rj.gates and min(rj.gates.values()) == 0 and rj.reward > TOL)}
    return PropertyResult("R3", not bad and bool(out), {"negatives": {n: {"reward": rj.reward, "outcome": rj.outcome} for n, rj in out.items()}, "violations": bad})


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
    pre_unmet = {m.id for m in milestones if (with_stubs.get(m.id) or 0) >= 1 and any((with_stubs.get(p) or 0) < 1 for p in m.requires)}
    stripped = {k: (0.0 if k in pre_unmet else v) for k, v in with_stubs.items()}
    a, b = potential(milestones, with_stubs)[0], potential(milestones, stripped)[0]
    return PropertyResult("R5", a <= b + TOL, {"phi_with_stubs": a, "phi_without": b, "stubbed": sorted(pre_unmet)})


def r6_state_only(reward_of_rollout: Callable[[Any, Sequence[str]], RewardJson], end_state: Any, claims: Sequence[Sequence[str]]) -> PropertyResult:
    """``reward_of_rollout(end_state, messages)``: the same end state with different message texts must score the same."""
    rewards = [reward_of_rollout(end_state, msgs).reward for msgs in claims]
    return PropertyResult("R6", max(rewards) - min(rewards) <= TOL, {"rewards": rewards})


def check_all(results: Sequence[PropertyResult]) -> dict[str, Any]:
    return {"ok": all(r.ok for r in results), "properties": {r.prop: {"ok": r.ok, **r.detail} for r in results}}
