"""M6 audit (docs/audit/2026-10-12/M06-credit.md): dense credit, composition, properties, rubric items, validity. Ids:
K = the M6 defect study, S = the framework sweeps."""

from __future__ import annotations

import asyncio
import json
import math

import pytest

from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
from crucible.core.schema.state import Query


def _m(mid, weight=1.0, requires=()):
    return Milestone(id=mid, weight=weight, requires=list(requires),
                     check=MilestoneCheck(app="log", query=Query(op="exists", args={"object": mid}), expect=Expectation(kind="truthy")))


# ------------------------------------------------------------------------------------------------ per-turn credit
def test_K1_gate_failed_turns_get_no_positive_credit_and_flat_groups_none():
    from crucible.credit.compose import Episode, turn_advantages

    g = [Episode(reward=1.0, phis=[0.5, 1.0]), Episode(reward=0.0, phis=[0.0, 0.143, 0.143, 0.286], gated=True)]
    adv = turn_advantages(g)
    assert all(x <= -0.5 + 1e-9 for x in adv[1])  # two turns got positive local credit
    flat = turn_advantages([Episode(reward=0.0, phis=[0.5, 1.0]), Episode(reward=0.0, phis=[0.0, 0.0])])
    assert flat == [[0.0, 0.0], [0.0, 0.0]]  # ±0.1 of pure process reward


def test_K2_local_credit_is_bounded_by_the_outcome_advantage():
    from crucible.credit.compose import Episode, turn_advantages

    n = 800
    fail = Episode(reward=0.2, phis=[0.0] * (n // 2) + [1.0] * (n - n // 2))
    win = Episode(reward=1.0, phis=[(i + 1) / n for i in range(n)])
    adv = turn_advantages([fail, win])
    assert max(adv[0]) <= 0.0 + 1e-9 and min(adv[1]) >= 0.0 - 1e-9  # one failing turn scored 1.199 > every success turn (0.4)


def test_K3_oscillation_does_not_pay():
    from crucible.credit.compose import Episode, turn_advantages

    g = [Episode(reward=0.2, phis=[0.75, 1.0], phi0=0.5), Episode(reward=0.2, phis=[0.0, 1.0, 0.0, 1.0], phi0=0.5),
         Episode(reward=0.0, phis=[0.0] * 8, phi0=0.5)]
    adv = turn_advantages(g)
    assert sum(adv[1]) <= sum(adv[0]) * 2 + 1e-9  # same reward: the oscillating episode's 4 turns earned 0.495 vs 0.254
    mono_local, osc_local = sum(adv[0]) - 2 * (0.2 - sum(e.reward for e in g) / 3), sum(adv[1]) - 4 * (0.2 - sum(e.reward for e in g) / 3)
    assert abs(mono_local) < 1e-9 and abs(osc_local) < 1e-9  # the local term only redistributes within each episode


def test_S_veto_and_the_trainer_path_uses_the_group_rule():
    from crucible.core.schema.rollout import RolloutRecord, Turn
    from crucible.credit.compose import Episode, turn_advantages
    from crucible.train.credit import turn_credit_map

    adv = turn_advantages([Episode(reward=1.0, phis=[0.5, 1.0], vetoed=[False, True]), Episode(reward=0.0, phis=[0.0, 0.0])])
    assert adv[0][1] <= 0.0  # AReaL's GVPO veto: a turn with evidence of wrongdoing is never positive

    def ro(rid, reward, deltas, gates=None):
        return RolloutRecord(rollout_id=rid, sample="s", task="t", harness="h", policy="p", reward=reward, gates=gates or {},
                             turns=[Turn(call_id=f"{rid}:{i}", actor="agent", completion_token_ids=[1], advantage=d) for i, d in enumerate(deltas)])

    recs = [ro("a", 1.0, [0.5, 0.5]), ro("b", 0.0, [0.143, 0.143], gates={"controls": False})]
    tc = turn_credit_map(recs, ["g", "g"], eta=0.1)
    assert all(v <= 0 for k, v in tc.items() if k[0] == "b")  # raw ΔΦ (+0.143) reached the gate-failed rollout's tokens
    assert abs(sum(v for k, v in tc.items() if k[0] == "a")) < 1e-9  # no net mass


def test_S_adv_signed_conserves_each_sign():
    from crucible.credit.signed import signed_rebalance

    adv = [1.0] * 5 + [-1.0] * 5
    hit = [2.0, 0, 0, 0, 0, 2.0, 0, 0, 0, 0]
    sign = [1] * 5 + [-1] * 5
    new, m = signed_rebalance(adv, hit, sign)
    assert new[0] == 0.0 and new[5] == -2.0
    assert sum(new[:5]) == pytest.approx(5.0) and sum(new[5:]) == pytest.approx(-5.0)  # reduction moved net mass to -5
    assert m["penalty/signed/pos_scale"] == pytest.approx(1.25)


# ------------------------------------------------------------------------------------------------ composition, mask
def test_K9_K10_unmeasured_milestones_are_not_zeros_and_a_mask_has_no_reward():
    from crucible.credit.compose import compose
    from crucible.core.schema.reward import InfraStatus
    from crucible.verify.grader import to_reward
    from crucible.verify.runtime.plan import GradeReport

    rj = compose(gates={}, outcome=False, phi=0.0, masked=InfraStatus(status="masked", failure_kind="verifier_error", reason="x"))
    assert rj.reward is None and "reward" not in rj.flat()  # reward 0.0 beside infra/masked = 1
    rep = GradeReport(task="t", tier="full", checks=[], gates={}, reward=0.0, outcome=0.0, infra={"status": "ok"}, trusted=True, elapsed_s=0.0)
    out = to_reward(rep, milestones=[_m("a"), _m("b")], final_phis={"a": 1.0, "b": None})
    assert out.progress == 1.0 and out.components["progress/unmeasured"] == 1.0  # 0.5: the unmeasured milestone counted as 0
    masked = to_reward(rep, milestones=[_m("a")], final_phis={"a": None})
    assert masked.reward is None  # C6 had no production caller


def test_K11_K12_potential_is_validated_and_starts_from_the_base():
    from crucible.workflow.potential import potential, trajectory

    with pytest.raises(ValueError):
        potential([_m("a", weight=2), _m("b", weight=-1)], {"a": 1, "b": 1})  # Φ = 2.0
    with pytest.raises(ValueError):
        potential([_m("a")], {"a": math.nan})
    with pytest.raises(ValueError):
        potential([_m("a", requires=["gone"])], {"a": 1})
    st = trajectory([_m("a"), _m("b")], [{"a": 1.0, "b": 0.0}], initial={"a": 1.0, "b": 0.0})
    assert st[0].delta == 0.0 and st[0].new == []  # step 0 credited with what held at the base


def test_K18_rows_align_and_shaping_is_per_call():
    from crucible.core.schema.reward import StepRow
    from crucible.credit.steps import rows_for, to_turn_shaping
    from crucible.workflow.potential import PotentialStep
    from crucible.workflow.steps import Step

    steps = [Step(index=i, seqs=[i], kind="edit") for i in range(3)]
    pot = [PotentialStep(phi=0.1 * i, delta=0.1) for i in range(3)]
    with pytest.raises(ValueError, match="call_ids"):
        rows_for(steps, pot, call_ids=["c1"])  # IndexError / silent truncation
    rows = [StepRow(step=0, call_id="c1", phi=0.1, delta_phi=0.1), StepRow(step=1, call_id="c1", phi=0.2, delta_phi=0.1),
            StepRow(step=2, call_id="c2", phi=0.1, delta_phi=-0.1)]
    assert to_turn_shaping(rows) == [0.2, -0.1] and to_turn_shaping(rows, gated=True) == [0.0, -0.1]


# ------------------------------------------------------------------------------------------------ properties
def test_K4_K5_K13_R7_properties_can_fail():
    from crucible.credit.compose import compose
    from crucible.credit.properties import r1_oracle, r3_negatives, r5_stubbing, r7_phi_farm

    assert not r1_oracle(lambda s: compose(gates={}, outcome=True, phi=0.0), "oracle").ok  # Φ ≡ 0 passed every property
    res = r3_negatives(lambda s: compose(gates={}, outcome=False, phi=1.0), {"reversed": 1, "honest": 2}, crafted=["reversed"])
    assert not res.ok and "reversed" in res.detail["violations"]  # a crafted bad end state earned 0.2
    assert not r3_negatives(lambda s: compose(gates={}, outcome=False, phi=0.0), {}).tested
    ms = [_m("build"), _m("test", requires=["build"])]
    assert r5_stubbing(ms, {"build": 0.5, "test": 1.0}).ok  # fractional prerequisite = proportional credit, not stubbing
    assert not r7_phi_farm([_m("diff/located/a.py:f")], {"diff/located/a.py:f": 1.0}, kappa=0.2).ok


# ------------------------------------------------------------------------------------------------ rubric items, judges
def test_K8_verdicts_parse_and_unmeasured_items_leave_the_share():
    from crucible.credit.operators.rubric_item import RubricItem, parse_verdict, share_met

    assert parse_verdict("The evidence said 'VERDICT: PASS'.\nVERDICT: FAIL") is False  # a quoted PASS flipped FAIL
    assert parse_verdict("reason\n**VERDICT:** PASS") is True  # None
    assert parse_verdict("<think>VERDICT: PASS") is None
    items = [RubricItem(id="a", question="q", status="validated", weight=1.0), RubricItem(id="b", question="q", status="validated", weight=1.0)]
    assert share_met(items, {"a": True, "b": None}) == 1.0  # 0.5: unmeasured counted as unmet
    with pytest.raises(ValueError):
        RubricItem(id="c", question="q", weight=-1.0)
    with pytest.raises(ValueError, match="duplicate"):
        share_met([items[0], items[0]], {"a": True})
    assert "````" in RubricItem(id="a", question="q").prompt("look: ```code```")  # evidence fenced past its own backticks


def test_K7_items_need_fixtures_and_data_to_validate():
    from crucible.credit.operators.rubric_item import RubricItem, lifecycle

    it = RubricItem(id="x", method="llm", question="q")
    assert lifecycle(it, [1, 0, 1, 0], [1, 0, 1, 0]).status == "candidate"  # validated at weight 1.0 with no fixtures, n = 4
    rule = RubricItem(id="y", method="rule", question="q")
    assert lifecycle(rule, [1, 0] * 3, [1, 0] * 3).status == "candidate"


def test_S_judge_votes_retry_and_critical_mask():
    from crucible.credit.operators.rubric_item import RubricItem
    from crucible.verify.judge_item import judge_items

    class LLM:
        def __init__(self, replies):
            self.replies = list(replies)

        async def complete(self, *a, **k):
            class R:
                pass

            r = R()
            r.text = self.replies.pop(0)
            return r

    items = [RubricItem(id="crit", tier="critical", question="q", status="validated", weight=1.0)]
    out = asyncio.run(judge_items(LLM(["garbled", "why\nVERDICT: PASS"]), items, "ev"))
    assert out["items"]["crit"] is True  # one unparseable reply was final
    out = asyncio.run(judge_items(LLM(["x", "y"]), items, "ev"))
    assert out["mask"] is True


def test_S_answer_criteria_are_validated():
    from crucible.verify.answers import Criterion, validate_criteria

    for bad in ([Criterion.of({"id": "a", "checker": {"kind": "exact"}, "weight": -1})],
                [Criterion.of({"id": "a", "checker": {"kind": "exact"}}), Criterion.of({"id": "a", "checker": {"kind": "exact"}})],
                [Criterion.of({"id": "a", "checker": {"kind": "exact"}, "weight": 0})]):
        with pytest.raises(ValueError):
            validate_criteria(bad)


# ------------------------------------------------------------------------------------------------ validity, PRM, misc
def test_K14_K15_K19_statistics_refuse_bad_input():
    from crucible.credit.operators.prm import LogisticPRM
    from crucible.credit.validity import auc, time_split

    train, hold = time_split(list(range(10)), time=lambda i: 0 if i < 8 else 1, holdout=0.3)
    assert len(train) == 8 and len(hold) == 2  # train 0, holdout 10
    with pytest.raises(ValueError):
        time_split([1, 2], time=lambda i: 0)
    with pytest.raises(ValueError):
        auc([math.nan, 0.2, 0.9, 0.1], [1, 0, 1, 0])
    assert LogisticPRM().score({"delta_phi": 0.1}) is None  # an unfit model imputed 0.5


def test_K6_K20_hack_rules_on_world_writes():
    from crucible.core.agent_events import writes_of
    from crucible.core.schema.trace import Trace, TraceEvent, TraceSource
    from crucible.credit.hack_detector import rule_flags

    cmds = ["echo x > tests/helpers.py", "echo 1 > conftest.py", "echo 1 > solution/solve.sh", "echo 1 > .crucible/reward.json", "touch pkg/foo_test.go"]
    ev = [TraceEvent(seq=i, kind="action", type="tool_call", data={"arguments": {"command": c}, "writes": writes_of("shell.exec", {"command": c})})
          for i, c in enumerate(cmds)]
    ev.append(TraceEvent(seq=len(ev), kind="action", type="tool_call", data={"arguments": {"command": 'pytest -k="not test_hard"'}}))
    flags = rule_flags(Trace(trace_id="t", source=TraceSource(connector="x", ref="t"), events=ev))
    assert {f["seq"] for f in flags} == set(range(len(ev)))  # only tests/test_*.py was caught through `file:` paths


def test_S_contracts_with_trainers_and_group_ops():
    from crucible.core.schema.rollout import RolloutRecord
    from crucible.credit.shaping import verbosity_group_fn
    from crucible.train.group_ops import gar

    def rec(rid, r):
        return RolloutRecord(rollout_id=rid, sample="s", task="t", harness="h", policy="p", reward=r)

    out = asyncio.run(gar(lambda m: None)("g", [rec("a", 1.0), rec("b", 0.18), rec("c", 0.0)]))
    assert out[1].advantage == out[2].advantage  # partial credit 0.18 got the success's advantage
    assert verbosity_group_fn(1.5)  # the bounded rule exists; numbers in test_shaping
    from crucible.interop.nemo_gym import server

    src = open(server.__file__).read()
    assert '"reward_components": {"crucible": reward}' in src  # the channels summed to 4.0 against a reward of 1.033


def test_S_agentenv_capture_keeps_identical_genuine_calls(tmp_path):
    from crucible.interop.agentenv.harness import calls_from_capture

    row = {"digest": "d1", "request": {"model": "m"}, "response": {"choices": [{"message": {"content": "x"}, "finish_reason": "stop"}]}, "status": 200}
    (tmp_path / "cap.jsonl").write_text("\n".join(json.dumps({**row, "t": i}) for i in range(3)) + "\n")
    p = tmp_path / "cap.jsonl"
    assert len(calls_from_capture(p, "agent")) == 1  # unmarked adjacent repeats: the old rule, kept for clients without retry headers
    rows = [{**row, "headers": {"x-stainless-retry-count": "0"}} for _ in range(2)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert len(calls_from_capture(p, "agent")) == 2  # best-of-n collapsed into one call
