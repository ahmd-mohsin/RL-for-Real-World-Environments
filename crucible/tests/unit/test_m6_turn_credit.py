"""M6 R1-R3 (docs/research/improvements/2026-10-14/M06-credit.md §7): per-turn credit follows one rule on every path,
"gated" means a hard rule failed, and live training carries per-turn progress or refuses.

Most checks are properties over random GRPO groups (masked members, hard-rule failures, outcome-only failures, hacks,
RLM sub-calls, child agents), not fixed toy numbers: the rule must hold for any task, not for the helpdesk.
"""

from __future__ import annotations

import asyncio
import json
import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.credit.compose import Episode, hard_gate_failed, turn_advantages
from crucible.train.bridges.common import flatten
from crucible.train.credit import (
    credited_records,
    has_turn_credit,
    is_hack,
    outcome_advantages,
    require_turn_credit,
    turn_credit_map,
)

TOL = 1e-9


# ------------------------------------------------------------------------------------------------------------- builders
def _turn(cid: str, dphi: float | None, *, seg: int = 0, **kw) -> Turn:
    return Turn(call_id=cid, actor="agent", segment=seg, prompt_token_ids=[1, 2, len(cid)], completion_token_ids=[5, 6], logprobs=[-0.1, -0.2],
                policy_version=0, advantage=dphi, sampling={"temperature": 1.0}, **kw)


def _record(rid: str, reward: float | None, dphis: list[float | None], *, gates=None, kinds=None, subs=None, child=0,
            masked=False, comps=None) -> RolloutRecord:
    """Root calls (one segment each, so every call is its own sequence) with their ΔΦ; ``subs``: {root index: n RLM
    sub-calls}; ``child``: a child agent of that many calls spawned by the first root call. Sub-calls and child calls carry
    the ancestor's ΔΦ, as ``runtime.record.build`` writes them."""
    turns = []
    for i, d in enumerate(dphis):
        root = f"{rid}:r{i}"
        turns.append(_turn(root, d, seg=i))
        for j in range((subs or {}).get(i, 0)):
            turns.append(_turn(f"{root}:s{j}", d, seg=i, kind="sub", depends_on=[root]))
    for j in range(child):
        turns.append(_turn(f"{rid}:k{j}", dphis[0] if dphis else None, seg=100 + j, agent="c1", depends_on=[f"{rid}:r0"]))
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, gates=gates or {},
                         gate_kinds=kinds or {}, turns=turns, mask_sample=masked, reward_components=dict(comps or {}),
                         failure_kind="infra" if masked else None)


@st.composite
def groups(draw, max_size: int = 5):
    n = draw(st.integers(1, max_size))
    recs = []
    for k in range(n):
        masked = draw(st.booleans()) and draw(st.booleans())  # ~25 %
        reward = None if masked else draw(st.sampled_from([0.0, 0.0, 0.1, 0.5, 1.0, 1.0, 1.2]))
        dphis = draw(st.lists(st.one_of(st.none(), st.floats(-1, 1, allow_nan=False).map(lambda x: round(x, 3))), min_size=1, max_size=5))
        hard = {f"rule{i}": draw(st.booleans()) for i in range(draw(st.integers(0, 2)))}
        outc = {f"check{i}": draw(st.booleans()) for i in range(draw(st.integers(0, 2)))}
        declared = draw(st.booleans())  # an older record: no kinds at all (every gate a hard rule)
        kinds = ({g: "gate" for g in hard} | {g: draw(st.sampled_from(["outcome", "score"])) for g in outc}) if declared else {}
        comps = draw(st.sampled_from([{}, {}, {}, {"hack/confirmed": 1.0}, {"hack/zeroed": 0.0}, {"gar/hack_zeroed": 1.0}]))
        subs = {0: draw(st.integers(0, 2))}
        recs.append(_record(f"g{k}", reward, dphis, gates=hard | outc, kinds=kinds, subs=subs, child=draw(st.integers(0, 2)), masked=masked,
                            comps=comps))
    return recs


def _trainer_view(recs: list[RolloutRecord], eta: float) -> dict[tuple[str, str], float]:
    """What the torch trainer trains per call: R − group mean + turn_credit_map credit."""
    tc = turn_credit_map(recs, ["g"] * len(recs), eta=eta)
    oa = outcome_advantages(recs, ["g"] * len(recs))
    return {(r.rollout_id, t.call_id): oa[r.rollout_id] + tc.get((r.rollout_id, t.call_id), 0.0) for r in recs for t in r.turns}


def _bridge_view(recs: list[RolloutRecord], eta: float) -> dict[tuple[str, str], list[float]]:
    """What a bridge that flattens credited records (the rLLM tinker path) puts on each call's tokens."""
    oa = outcome_advantages(recs, ["g"] * len(recs))
    out: dict[tuple[str, str], list[float]] = {}
    for r in credited_records(recs, ["g"] * len(recs), turn_weight=eta):
        for s in flatten(r, oa[r.rollout_id], turn_weight=eta):
            for cid, a, b in s.call_spans:
                out[(r.rollout_id, cid)] = s.advantages[a:b]
    return out


# ------------------------------------------------------------------------------------------ R2: the rule's gated branch
@settings(max_examples=300, deadline=None)
@given(st.lists(st.tuples(st.one_of(st.none(), st.sampled_from([0.0, 0.2, 1.0])),
                          st.lists(st.floats(-1, 1, allow_nan=False), min_size=1, max_size=6),
                          st.booleans(), st.booleans()), min_size=1, max_size=6),
       st.sampled_from([0.05, 0.1, 0.5, 1.0, 3.0]))
def test_rule_properties_hold_for_any_group(spec, eta):
    """For any group and η: masked episodes get 0; in a group without outcome spread every turn is the outcome advantage;
    the local term only redistributes within an episode (zero mass) and never flips the sign of the outcome advantage;
    a rule-breaking or hacked episode is not redistributed at all."""
    eps = [Episode(reward=r, phis=phis, gated=g, hack=h) for r, phis, g, h in spec]
    adv = turn_advantages(eps, eta=eta)
    live = [e.reward for e in eps if e.reward is not None]
    mu = sum(live) / len(live) if live else 0.0
    flat = len(set(live)) <= 1
    for e, row in zip(eps, adv, strict=True):
        if e.reward is None:
            assert row == [0.0] * len(e.phis)
            continue
        a_out = e.reward - mu
        assert abs(sum(row) - a_out * len(row)) < 1e-6  # zero mass: the outcome is redistributed, never inflated
        assert all(x >= -TOL for x in row) if a_out > TOL else True
        assert all(x <= TOL for x in row) if a_out < -TOL else True
        if flat or e.gated or e.hack:
            assert row == pytest.approx([a_out] * len(row), abs=TOL)


@pytest.mark.parametrize("eta", [0.05, 0.1, 0.5, 1.0])
def test_an_honest_failure_never_trains_below_a_no_op(eta):
    """The M6 R2 inversion: with the outcome check recorded as a gate, an honest attempt that made progress but missed the
    outcome totalled −1.113 against a no-op's −1.000 (η 0.1; −1.333 at η 0.5). With kinds, the attempt is not rule-
    breaking: its total equals the no-op's (zero mass) and its progress call ranks first in its episode."""
    kinds = {"controls": "gate", "outcome": "outcome"}
    recs = [_record("ok", 1.0, [0.4, 0.3, 0.3], gates={"controls": True, "outcome": True}, kinds=kinds),
            _record("half", 0.0, [0.5, 0.0, 0.0], gates={"controls": True, "outcome": False}, kinds=kinds),
            _record("noop", 0.0, [0.0, 0.0, 0.0], gates={"controls": True, "outcome": False}, kinds=kinds)]
    v = _trainer_view(recs, eta)
    half = [v[("half", f"half:r{i}")] for i in range(3)]
    noop = [v[("noop", f"noop:r{i}")] for i in range(3)]
    assert sum(half) == pytest.approx(sum(noop))
    assert half[0] == max(half) and half[0] > half[1]


def test_a_hard_rule_failure_trains_every_call_at_the_outcome_advantage():
    """The clamp after centring left the call that bought progress by breaking the rule the least punished (−0.5 against
    −0.75); now no call of a rule-breaking episode ranks above another."""
    recs = [_record("ok", 1.0, [0.1, 0.1, 0.8], gates={"no_stub": True}, kinds={"no_stub": "gate"}),
            _record("stub", 0.0, [0.6, 0.0, 0.0], gates={"no_stub": False}, kinds={"no_stub": "gate"})]
    v = _trainer_view(recs, 1.0)
    assert [v[("stub", f"stub:r{i}")] for i in range(3)] == pytest.approx([-0.5, -0.5, -0.5])


def test_hard_gate_failed_reads_kinds_and_defaults_to_hard():
    assert not hard_gate_failed({})
    assert hard_gate_failed({"a": False})  # undeclared: a hard rule (every record before kinds)
    assert hard_gate_failed({"a": 0, "b": 1}, {"b": "outcome"})  # gate bits are 0/1 in harness results
    assert not hard_gate_failed({"tests": False}, {"tests": "outcome"})
    assert not hard_gate_failed({"perf": False}, {"perf": "score"})  # any non-"gate" kind is not a rule (as reward_json)
    assert hard_gate_failed({"tests": False, "path": False}, {"tests": "outcome", "path": "gate"})
    assert not hard_gate_failed({"a": True, "b": True})


@settings(max_examples=200, deadline=None)
@given(groups(), st.sampled_from([0.1, 0.5]))
def test_turn_credit_map_gates_on_hard_rules_only(recs, eta):
    """For any group: a record whose only failed checks are outcome or score checks is credited as honest, and a record
    with a failed hard rule (or a hack) gets no local credit on any call."""
    tc = turn_credit_map(recs, ["g"] * len(recs), eta=eta)
    for r in recs:
        if r.mask_sample or r.reward is None:
            assert all(tc.get((r.rollout_id, t.call_id), 0.0) == 0.0 for t in r.turns)
        elif hard_gate_failed(r.gates, r.gate_kinds) or is_hack(r):
            assert all(abs(tc.get((r.rollout_id, t.call_id), 0.0)) < TOL for t in r.turns)


def test_every_hack_marker_counts():
    """M6 D2: ``zero_hacks`` writes ``hack/zeroed`` (holding the replaced reward, possibly 0.0) and GAR's custom detector
    ``gar/hack_zeroed``; the rule read only ``hack/confirmed``, so a zeroed hack's Φ-raising call kept positive credit."""
    base = {"gates": {}, "kinds": {}}
    for comps in ({"hack/confirmed": 1.0}, {"hack/zeroed": 0.0}, {"hack/zeroed": 1.0}, {"gar/hack_zeroed": 1.0}):
        recs = [_record("ok", 1.0, [0.5, 0.5], **base), _record("h", 0.0, [0.9, 0.0, 0.0], comps=comps, **base),
                _record("f", 0.0, [0.0, 0.0, 0.0], **base)]
        assert is_hack(recs[1])
        tc = turn_credit_map(recs, ["g"] * 3, eta=0.5)
        assert all(abs(tc[("h", f"h:r{i}")]) < TOL for i in range(3)), comps
    assert not is_hack(_record("x", 1.0, [0.1], comps={"hack/confirmed": 0.0}))


# ------------------------------------------------------------------------------- R1: one rule on every trainer path
@settings(max_examples=200, deadline=None)
@given(groups(), st.sampled_from([0.1, 0.5, 1.0]))
def test_a_bridge_trains_exactly_what_the_torch_trainer_trains(recs, eta):
    """For any group (masked members, rule-breaking and hacked members, RLM sub-calls, child agents), the per-token
    advantage a bridge builds from ``credited_records`` + ``flatten`` equals the torch trainer's per-call advantage
    (outcome advantage + ``turn_credit_map``) on every token of every trained call, and a masked record trains nothing.
    M6 R1: the rLLM bridge added raw ΔΦ (+0.1 on a gate-failed call where the rule gives −0.5)."""
    live = [r for r in recs if not r.mask_sample and r.reward is not None]
    if live and not has_turn_credit(recs):  # dense credit asked for, none recorded: refused (R3), never silently absent
        with pytest.raises(ValueError, match="per-turn progress"):
            require_turn_credit(recs, where="test")
        return
    require_turn_credit(recs, where="test")  # nothing live, or progress recorded: passes
    want = _trainer_view(recs, eta)
    got = _bridge_view(recs, eta)
    for r in recs:
        keys = [(r.rollout_id, t.call_id) for t in r.turns]
        if r.mask_sample or r.reward is None:
            assert not any(k in got for k in keys)
            continue
        for k in keys:
            assert got[k] == pytest.approx([want[k]] * len(got[k]), abs=1e-9), k


def test_rlm_sub_calls_and_child_agents_take_their_parents_credit():
    """``record.build`` copies the ancestor's ΔΦ onto every sub-call and child call; flattened raw, a gate-failed root call
    with 4 sub-calls put +0.6 on each of 5 calls. Through the rule they share the parent's credit."""
    recs = [_record("A", 1.0, [0.3, 0.7], gates={"g": True}, kinds={"g": "gate"}),
            _record("B", 0.0, [0.6, 0.0, 0.0], gates={"g": False}, kinds={"g": "gate"}, subs={0: 4}, child=2)]
    got = _bridge_view(recs, 1.0)
    parent = got[("B", "B:r0")][0]
    assert parent == pytest.approx(-0.5)
    assert all(got[("B", f"B:r0:s{j}")][0] == pytest.approx(parent) for j in range(4))
    assert all(got[("B", f"B:k{j}")][0] == pytest.approx(parent) for j in range(2))


def test_outcome_advantages_skip_masked_members_and_match_the_trainer_estimator():
    recs = [_record("a", 1.0, [0.1]), _record("b", 0.0, [0.1]), _record("m", None, [0.1], masked=True), _record("c", 0.5, [0.1])]
    groups_ = ["g", "g", "g", "h"]
    oa = outcome_advantages(recs, groups_)
    assert oa == {"a": pytest.approx(0.5), "b": pytest.approx(-0.5), "m": 0.0, "c": 0.0}
    objectives = pytest.importorskip("crucible.train.objectives")  # torch
    ref = objectives.group_advantages([None if r.mask_sample else r.reward for r in recs], groups_, estimator="mean")
    assert [oa[r.rollout_id] for r in recs] == pytest.approx(ref)


def test_credited_records_needs_a_positive_weight_and_leaves_its_input_untouched():
    recs = [_record("a", 1.0, [0.4, 0.2]), _record("b", 0.0, [0.1, 0.0])]
    before = [r.model_dump() for r in recs]
    with pytest.raises(ValueError, match="turn_weight > 0"):
        credited_records(recs, ["g", "g"], turn_weight=0.0)
    out = credited_records(recs, ["g", "g"], turn_weight=0.5)
    assert [r.model_dump() for r in recs] == before and out[0] is not recs[0]


def test_rllm_bridge_refuses_turn_credit_it_cannot_carry():
    """Checked before rLLM is imported: a misconfiguration fails with its reason wherever rLLM is installed or not."""
    from crucible.train.bridges import rllm_bridge

    recs = [_record("a", 1.0, [0.4]), _record("b", 0.0, [0.1])]
    with pytest.raises(ValueError, match="verl"):
        rllm_bridge.trajectory_groups(recs, backend="verl", turn_weight=0.5)
    with pytest.raises(ValueError, match="trajectory_groups"):
        rllm_bridge.trajectories(recs[0], backend="tinker", turn_weight=0.5)
    with pytest.raises(ValueError, match="per-turn progress"):
        rllm_bridge.trajectory_groups([_record("a", 1.0, [None]), _record("b", 0.0, [None])], backend="tinker", turn_weight=0.5)
    with pytest.raises(ValueError, match="backend"):
        rllm_bridge.trajectory_groups(recs, backend="fireworks")


# --------------------------------------------------------------------------- R3: dense credit is present or refused
def test_require_turn_credit_distinguishes_absent_progress_from_zero_progress():
    none = [_record("a", 1.0, [None, None]), _record("b", 0.0, [None])]
    with pytest.raises(ValueError, match="no live rollout carries per-turn progress"):
        require_turn_credit(none, where="t")
    require_turn_credit([_record("a", 1.0, [0.0, 0.0]), _record("b", 0.0, [None])], where="t")  # 0.0 is progress that did not move
    require_turn_credit([_record("m", None, [None], masked=True)], where="t")  # nothing live: nothing to train either way
    frozen = [_record("a", 1.0, [0.3])]
    assert not has_turn_credit(frozen, frozen_actors=["agent"])  # a frozen actor's calls are context, not credit


def test_torch_trainer_refuses_turn_weight_without_progress():
    pytest.importorskip("torch")
    from crucible.train.engine import TinyPolicy
    from crucible.train.mismatch import Correction
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    def rec(rid, reward, dphi):
        return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=[
            Turn(call_id=f"{rid}:c1", actor="agent", prompt_token_ids=[1, 2, 3], completion_token_ids=[4, 5], logprobs=[-1.0, -1.0], policy_version=0,
                 advantage=dphi, sampling={"temperature": 1.0})])

    tr = TorchTrainer(TinyPolicy(16, 8, seed=0), TrainerConfig(lr=1e-3, turn_weight=0.5, correction=Correction("ppo")))
    with pytest.raises(ValueError, match="TorchTrainer: turn_weight > 0 but no live rollout"):
        tr.step([rec("a", 1.0, None), rec("b", 0.0, None)], ["g", "g"])
    assert "train/loss" in tr.step([rec("a", 1.0, 0.3), rec("b", 0.0, 0.0)], ["g", "g"])
    plain = TorchTrainer(TinyPolicy(16, 8, seed=0), TrainerConfig(lr=1e-3, correction=Correction("ppo")))
    assert "train/loss" in plain.step([rec("a", 1.0, None), rec("b", 0.0, None)], ["g", "g"])  # outcome-only, as asked


def _scripted_backlog_rollout(task, *, skip_ticket=None, short=1, **kw):
    """A live M12 rollout (``world_env.world_rollout``) whose scripted model emits the oracle's tool calls as tokens,
    leaving out one ticket's calls and stopping ``short`` calls early: real, partial progress."""
    from crucible_pack_toy.longhorizon.world import oracle

    from crucible.core.schema.actions import openai_name
    from crucible.train.chat import ByteChatCodec
    from crucible.train.engine import GenRequest, ScriptedEngine
    from crucible.train.world_env import world_rollout

    codec = ByteChatCodec()
    calls = [a for a in oracle(task.b) if a["kind"] == "tool_call" and a["arguments"].get("ticket_id") != skip_ticket]
    calls = calls[: len(calls) - short]

    def script(rid, k, prompt):
        if k < len(calls):
            body = json.dumps({"name": openai_name(calls[k]["function_name"]), "arguments": calls[k]["arguments"]})
            return list(f"<tool_call>{body}</tool_call>".encode()) + [codec.EOS]
        return list(b"Done.") + [codec.EOS]

    eng, k = ScriptedEngine(script, stop_ids=[codec.EOS]), [0]

    async def gen(ids, max_tokens, temperature, *, seed=None, stop_ids=None):
        k[0] += 1
        return await eng.generate(GenRequest(rid="r", call_id=f"r:c{k[0]}", prompt_ids=list(ids), max_tokens=max_tokens, temperature=temperature,
                                             stop_ids=tuple(stop_ids or ())))

    fn = world_rollout(lambda p: task, codec, max_steps=len(calls) + 3, max_tokens=4096, **kw)
    return asyncio.run(fn("r", "p", 0, gen, temperature=1.0))


@pytest.mark.parametrize(("seed", "n"), [(3, 4), (11, 6)])
def test_live_rollouts_record_per_turn_progress_that_adds_up_to_the_grade(seed, n):
    """M6 R3: ``world_rollout`` ran without a progress tracker, so every live record had ``Turn.advantage`` None and
    ``turn_weight > 0`` trained outcome-only. Now a task with subgoals records ΔΦ per call, and it telescopes: the calls'
    ΔΦ sum to Φ_final − Φ_0, which is the task's own share reward."""
    pytest.importorskip("torch")
    from crucible_pack_toy.longhorizon.task import BacklogTask

    task = BacklogTask(seed=seed, n=n, reward="share")
    ro = _scripted_backlog_rollout(task, skip_ticket=task.b.tickets[0].id)
    adv = [t.advantage for t in ro.turns]
    assert all(a is not None for a in adv)
    assert 0.0 < ro.reward < 1.0  # partial progress: the test exercises real credit, not all-or-nothing
    assert sum(adv) == pytest.approx(ro.progress["phi_final"] - ro.progress["phi0"]) == pytest.approx(ro.reward)
    assert ro.progress.get("unattributed_delta", 0.0) == pytest.approx(0.0)
    assert ro.gate_kinds == {}  # share mode grades with no gates


def test_live_rollouts_carry_gate_kinds_and_tasks_without_subgoals_record_no_progress():
    """Across backlogs: the scripted model cannot ask the supervisor, so where a refund needs approval it forges one (a
    broken rule, ``approvals``); elsewhere it only leaves tickets unhandled (an unmet outcome, ``tickets``). The record
    carries both kinds, and "rule broken" is exactly the approvals rule failing, never the unmet outcome."""
    pytest.importorskip("torch")
    from crucible_pack_toy.longhorizon.task import BacklogTask

    seen = set()
    for seed in (3, 5, 7, 9):
        task = BacklogTask(seed=seed, n=4, reward="all")
        ro = _scripted_backlog_rollout(task, skip_ticket=task.b.tickets[0].id)
        assert ro.gate_kinds == {"approvals": "gate", "tickets": "outcome"} and set(ro.gates) == {"approvals", "tickets"}
        assert hard_gate_failed(ro.gates, ro.gate_kinds) == (not ro.gates["approvals"])
        seen.add(ro.gates["approvals"])
    assert seen == {True, False}  # both an honest failure and a rule violation were exercised
    task = BacklogTask(seed=5, n=4, reward="all")

    class NoSubgoals:  # the same task without subgoals: no tracker, no progress file, ΔΦ stays None (unchanged behaviour)
        def __init__(self, inner):
            self._inner = inner

        def __getattr__(self, name):
            if name == "subgoals":
                raise AttributeError(name)
            return getattr(self._inner, name)

    plain = _scripted_backlog_rollout(NoSubgoals(BacklogTask(seed=5, n=4, reward="all")), skip_ticket=task.b.tickets[0].id)
    assert all(t.advantage is None for t in plain.turns) and plain.progress == {}
    off = _scripted_backlog_rollout(BacklogTask(seed=5, n=4, reward="all"), skip_ticket=task.b.tickets[0].id, track_progress=False)
    assert all(t.advantage is None for t in off.turns) and off.progress == {}  # an outcome-only run may skip the subgoal checks


# ------------------------------------------------------------------------------------- R2: kinds travel with the record
def test_grade_report_kinds_and_the_record_round_trip():
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build
    from crucible.verify.runtime.plan import GradeReport

    # M7 R6: every gate declares its kind (an undeclared one raises: test_m7_hardening U5)
    rep = GradeReport(task="t", tier="full", checks=[], gates={"rule": "pass", "t1": "fail", "x": "fail"}, reward=0.0, outcome=0.0,
                      infra={"status": "ok"}, trusted=True, elapsed_s=0.0, meta={"kinds": {"rule": "gate", "t1": "outcome", "x": "gate"}})
    assert rep.gate_kinds() == {"rule": "gate", "t1": "outcome", "x": "gate"}
    rec = RunRecord(run_id="r", task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=1, outcome="completed", reward=0.0,
                    attempt="r-a1", meta={"harness": "h", "gates": rep.gate_bits(), "gate_kinds": rep.gate_kinds()})
    ro = build(rec, calls={})
    assert ro.gate_kinds == {"rule": "gate", "t1": "outcome", "x": "gate"} and ro.gates == {"rule": True, "t1": False, "x": False}
    assert RolloutRecord.model_validate_json(ro.model_dump_json(by_alias=True)) == ro
    old = build(rec.model_copy(update={"meta": {"harness": "h", "gates": rep.gate_bits()}}), calls={})
    assert old.gate_kinds == {} and "gate_kinds" not in json.loads(old.model_dump_json(by_alias=True))  # older records keep their bytes


def test_multi_step_gates_are_outcomes_unless_a_step_broke_a_rule():
    """A multi-step task's ``step/<name>`` gate is the step's ``min_reward`` threshold: an unmet outcome, unless one of the
    step's own hard rules failed (then the episode broke a rule)."""
    import inspect

    from crucible.runtime.harness import native

    src = inspect.getsource(native)
    assert '"gate_kinds": rep.gate_kinds()' in src and '"outcome" if all(v for g, v in r["gates"].items()' in src
    results = [{"name": "a", "gates": {"rule": 1, "t": 0}, "gate_kinds": {"rule": "gate", "t": "outcome"}},
               {"name": "b", "gates": {"rule": 0}, "gate_kinds": {"rule": "gate"}}]
    kinds = {f"step/{r['name']}": "outcome" if all(v for g, v in r["gates"].items() if r["gate_kinds"].get(g, "gate") == "gate") else "gate"
             for r in results}
    assert kinds == {"step/a": "outcome", "step/b": "gate"}


# --------------------------------------------------------------------------------------- slime parity (R2 on slime)
def test_slime_credit_uses_the_same_hard_gate_rule():
    """The slime bridge derives ``gated`` from its sample metadata with the same helper as the trainer, so an outcome-only
    failure is credited as honest there too (its equivalence with the native path is pinned by test_m17_p6_equiv)."""
    from crucible.train.bridges.slime import credit as sc
    from crucible.train.bridges.slime.types import MirrorSample

    def sample(i, reward, dphis, gates, kinds):
        md = {"crucible": {"turn_credit": [{"delta_phi": d, "span": [2 * j, 2 * j + 2]} for j, d in enumerate(dphis)], "gates": gates,
                           "gate_kinds": kinds, "hack_confirmed": 0.0}}
        return MirrorSample(index=i, group_index=0, reward=reward, response_length=2 * len(dphis), metadata=md)

    kinds = {"path": "gate", "tests": "outcome"}
    samples = [sample(0, 1.0, [0.4, 0.3, 0.3], {"path": 1, "tests": 1}, kinds), sample(1, 0.0, [0.5, 0.0, 0.0], {"path": 1, "tests": 0}, kinds),
               sample(2, 0.0, [0.6, 0.0, 0.0], {"path": 0, "tests": 0}, kinds)]
    gc = sc.group_credit(samples, eta=0.5)
    honest, broken = gc[("rollout", 1)]["turns"], gc[("rollout", 2)]["turns"]
    assert honest[0] == max(honest) and honest[0] > 0 and math.isclose(sum(honest), 0.0, abs_tol=1e-9)
    assert broken == pytest.approx([0.0, 0.0, 0.0])
    recs = [_record("0", 1.0, [0.4, 0.3, 0.3], gates={"path": True, "tests": True}, kinds=kinds),
            _record("1", 0.0, [0.5, 0.0, 0.0], gates={"path": True, "tests": False}, kinds=kinds),
            _record("2", 0.0, [0.6, 0.0, 0.0], gates={"path": False, "tests": False}, kinds=kinds)]
    tc = turn_credit_map(recs, ["g"] * 3, eta=0.5)
    for i in range(3):
        assert gc[("rollout", i)]["turns"] == pytest.approx([tc[(str(i), f"{i}:r{j}")] for j in range(3)])


# ---------------------------------------------------------------------------------- D3: the raw reward survives shaping
def test_verbosity_penalty_keeps_a_success_a_success_for_gar_and_the_pass_rate():
    """M6 D3: ``verbosity_group_fn`` rewrote the reward without ``raw/reward``; GAR then read 0.9952 as a failure (−0.5,
    the real failure's advantage) and the pass rate (``r >= 1``) read 0.5 instead of 0.75."""
    from crucible.credit.shaping import RAW, verbosity_group_fn
    from crucible.train import group_ops

    def rec(rid, reward, n_tokens):
        return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward,
                             turns=[Turn(call_id=f"{rid}:c", actor="agent", completion_token_ids=[7] * n_tokens)])

    recs = [rec("a", 1.0, 100), rec("b", 1.0, 100), rec("long", 1.0, 110), rec("f", 0.0, 100)]
    verbosity_group_fn(1.0, min_pass_rate=0.5)("g", recs)
    long = recs[2]
    assert 0.99 < long.reward < 1.0 and long.reward_components[RAW] == 1.0
    assert all(RAW not in r.reward_components for r in (recs[0], recs[1], recs[3]))  # untouched members keep their reward as is
    pass_input = [float((r.reward_components or {}).get(RAW, r.reward)) for r in recs]  # what train.runtime hands the monitor
    assert sum(x >= 1.0 for x in pass_input) / len(recs) == 0.75
    asyncio.run(group_ops.gar()("g", recs))
    assert long.advantage > 0 > recs[3].advantage


def test_the_monitor_reads_the_raw_reward():
    """The runtime builds the pass-rate input from ``raw/reward`` when a group stage rewrote the reward."""
    import inspect

    pytest.importorskip("torch")  # train.runtime imports the engine

    from crucible.train import runtime

    assert '"_rewards": [r.reward if r.reward is None else float((r.reward_components or {}).get(RAW, r.reward)) for r in recs]' in \
        inspect.getsource(runtime)
