"""taskloop-2 (DESIGN §7.7 item 25; runs/taskloop-2.md): the fixes of the findings of taskloop-1, each as a policy rule
the loop obeys.

- R-band v2 ``band_undecided``: a forced final look, a straddling interval or two disagreeing measurements get more fresh
  rollouts on that measurement (to k_cap 32, within the per-lineage cap), then HOLD; a second measurement never places a
  task alone; diagnosis rollouts never enter acceptance. Property tests: the rule never relaxes validity or the band
  edges, at the policy and at the loop.
- R-lemma-lift: accepted repairs are recorded; a recurring counterexample class becomes a candidate; only the gate admits;
  an underpowered gate defers.
- R-p3: an unmeasured P3 on a candidate edit is a named condition with a lawful remedy (HOLD the edit), never a silent
  refusal and never on intent.
- R-too-hard ``family_knobs``: finance_ap has no lawful knob, so the loop reserves at once and never applies one.
A scripted student, classifier, deriver and gate (no model is called)."""

from __future__ import annotations

import asyncio
import json
import random
import tempfile
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from crucible.synth import policy
from crucible.synth.archive import Archive
from crucible.synth.falsify.base import Counterexample, Probe
from crucible.synth.learn import BandResult, BandSpec, settled
from crucible.synth.meta.lemmas import LemmaDecision
from crucible.synth.meta.lift import LemmaLift, RepairPair, pairs_from_run
from crucible.synth.meta.rsi import KnobMove, TaskRules
from crucible.synth.policy import Condition, PolicyLedger, PolicyViolation, resolve
from crucible.synth.policy.amendments import BATCHES, drafts
from crucible.synth.policy.genesis import genesis_entries
from crucible.synth.policy.model import UNSAFE_EFFECTS, relaxes
from tests.unit.test_synth_reskill import ADV, _with
from tests.unit.test_synth_taskloop import Knobs, R, _loop

POL = policy.current()
SPEC = BandSpec(lo=POL.param("R-band", "band_lo"), hi=POL.param("R-band", "band_hi"), alpha=POL.param("R-band", "alpha"))


class VStudent:
    """Pass or fail by the task version and rollout index (no World replay): ``rates[version]`` is a function of i."""

    def __init__(self, rates):
        self.rates, self.asked = rates, []
        self.student, self.checkpoint, self.simulator, self.time_mode = "student", "base", "json-agent/v1", "frozen"

    async def rollouts(self, task, n, *, start=0):
        f = self.rates[task.version]
        self.asked += [(task.id, task.version, i) for i in range(start, start + n)]
        return [R(i, bool(f(i)), None) for i in range(start, start + n)]


THIRD = lambda i: i % 3 == 0
ALL = lambda i: True
NONE = lambda i: False
EDGE = lambda i: i % 5 < 3


async def _clean(t):
    return []


def _reskill_loop(tmp, rates, **rules):
    loop, task, _ = _loop(tmp, tag="band", rules=TaskRules(rollouts=8, kappa=50.0, **rules))
    _with(loop)
    loop.recheck = _clean
    loop.probe = VStudent(rates)
    return loop, task


def _undecided(d):
    return [r for r in d.remedies if r["condition"] == "band_undecided"]


# ---------------------------------------------------------------------------------------------------------- the ledger

def test_the_shipped_ledger_holds_the_taskloop2_batch_after_the_genesis():
    es = PolicyLedger().entries()
    tail = es[len(genesis_entries()):]
    want = drafts("taskloop-2")
    got = [e for e in tail if e.get("batch") == "taskloop-2"]
    assert [(e["kind"], e["draft"]) for e in got] == [(d["kind"], d["draft"]) for d in want]
    for e, d in zip(got, want, strict=True):
        assert {k: e[k] for k in d} == json.loads(json.dumps(d)) and e["source"]["kind"] == "lead" and e["rationale"]
    assert POL.rule("R-band").version == 2 and POL.param("R-band", "k_cap") == 32 and POL.param("R-student-probe", "k_max") == 32
    assert {"R-lemma-lift", "R-p3"} <= set(POL.rules) and POL.param("R-too-hard", "family_knobs")["finance_ap"]["harder"] == []
    assert set(BATCHES) == {"taskloop-2"}


def test_applying_a_batch_twice_changes_nothing(tmp_path):
    led = PolicyLedger(tmp_path / "l.jsonl")
    led.write_genesis(genesis_entries())
    assert len(led.apply_batch("taskloop-2")) == len(BATCHES["taskloop-2"])
    v = led.policy().version
    assert led.apply_batch("taskloop-2") == [] and led.policy().version == v


@pytest.mark.parametrize(("param", "value"), [("band_lo", 0.05), ("band_hi", 0.7), ("alpha", 0.2), ("require_settled", False),
                                              ("place_on_second_alone", True), ("pool_diagnosis_into_acceptance", True)])
def test_no_later_entry_relaxes_the_band_not_even_the_leads(tmp_path, param, value):
    led = PolicyLedger(tmp_path / "l.jsonl")
    led.write_genesis(genesis_entries())
    led.apply_batch("taskloop-2")
    with pytest.raises(PolicyViolation, match="P-valid"):
        led.append({"kind": "amend_rule", "rule_id": "R-band", "set": {param: value}, "source": {"kind": "lead", "who": "lead"}})


# ---------------------------------------------------------------------------------------------------------- R-band v2: the policy property

_FACTS = st.fixed_dictionaries({}, optional={"rollouts": st.integers(min_value=0, max_value=80), "probe_budget_ok": st.booleans(),
                                             "measurements": st.integers(min_value=1, max_value=2), "disagree": st.booleans(),
                                             "straddles": st.booleans(), "lineage_rollouts": st.integers(min_value=0, max_value=200)})
_EXTRA = st.lists(st.fixed_dictionaries({"action": st.sampled_from(["more_rollouts", "hold", "reserve", "descend", "ascend", "reskill",
                                                                    "repair_with_counterexamples", "panel_decide", "executed_check"]),
                                         "rung": st.sampled_from(["evidence", "defer", "variant", "relabel", "repair"]),
                                         "effects": st.lists(st.sampled_from(sorted(UNSAFE_EFFECTS)), max_size=2),
                                         "changes": st.lists(st.sampled_from([
                                             {"rule": "R-band", "param": "band_hi", "old": 0.6, "new": 0.7},
                                             {"rule": "R-band", "param": "band_lo", "old": 0.1, "new": 0.05},
                                             {"rule": "R-band", "param": "place_on_second_alone", "old": False, "new": True},
                                             {"rule": "R-band", "param": "k_cap", "old": 32, "new": 64}]), max_size=1)}), max_size=3)


@settings(max_examples=300, deadline=None)
@given(facts=_FACTS, extra=_EXTRA)
def test_band_undecided_never_relaxes_validity_or_the_band_edges(facts, extra):
    r = resolve(Condition(kind="band_undecided", facts=facts), POL, extra=extra)
    assert not r.changes  # no parameter (a band edge, alpha, k_cap) is ever changed by a remedy, proposed or not
    assert not [x for x in r.changes if relaxes(x.rule, x.param, x.old, x.new)] and not [e for e in r.effects if e in UNSAFE_EFFECTS]
    assert r.rung in ("evidence", "defer") and (r.rung == "evidence" or r.action == "hold")  # never an edit, a relabel or a move
    if r.action == "more_rollouts":  # only within k_cap and the probe cap, even when a script proposes it
        assert facts.get("probe_budget_ok") is True and facts.get("rollouts", 99) < POL.param("R-band", "k_cap")
    elif facts.get("probe_budget_ok") is True and facts.get("rollouts", 99) < POL.param("R-band", "k_cap"):
        assert r.rung == "evidence", "an applicable evidence rung was passed over"
    assert (POL.param("R-band", "band_lo"), POL.param("R-band", "band_hi")) == (0.1, 0.6)


def test_settled_means_confident_and_inside_one_class():
    from crucible.synth.learn import credible_interval, decide

    for n in (4, 8, 12, 16, 32):
        for k in range(n + 1):
            r = decide(k, n, SPEC, final=True)
            lo, hi = credible_interval(k, n, SPEC)
            inside = (r.decision == "too_hard" and hi < SPEC.lo) or (r.decision == "too_easy" and lo > SPEC.hi) or \
                     (r.decision == "in_band" and lo >= SPEC.lo and hi <= SPEC.hi)
            assert settled(r, SPEC) == (r.confident and inside), (k, n)
    assert not settled(decide(0, 16, SPEC, final=True), SPEC) and settled(decide(0, 32, SPEC, final=True), SPEC)  # 0/16 straddles 0.1
    assert not settled(decide(8, 16, SPEC, final=True), SPEC)  # taskloop-1's forced 8/16 "in band"
    assert not settled(None, SPEC)


# ---------------------------------------------------------------------------------------------------------- R-band v2: the loop

async def test_a_reskill_whose_fresh_band_disagrees_is_escalated_then_held_never_placed(tmp_path):
    """taskloop-1 §B.2: only the card changes, so the pass rate cannot; a fresh band that disagrees is the test's error.
    Here the scripted student does change with the version, to force the disagreement."""
    loop, task = _reskill_loop(tmp_path, {1: THIRD, 2: ALL})
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESKILL" and d.accepted and loop.tasks[task.id].intent.skill == ADV.skill  # the evidence-derived label stands
    assert loop.lineages[task.id].status == "held" and d.reskill["placed"] is False
    assert d.band["decision"] == "in_band" and settled(BandResult.model_validate(d.band), SPEC)
    u = _undecided(d)
    assert [x["action"] for x in u] == ["more_rollouts", "hold"] and all(x["facts"]["disagree"] and x["facts"]["measurements"] == 2 for x in u)
    assert u[-1]["facts"]["rollouts"] == 32 and len(d.rollouts["acceptance"]) == 32  # escalated to k_cap on the fresh measurement
    assert set(d.rollouts["acceptance"]).isdisjoint(d.rollouts["diagnosis"])  # never pooled
    assert u[-1]["rule"] == "R-band" and u[-1]["policy_version"] == POL.tag


async def test_a_reskill_whose_fresh_band_agrees_is_placed(tmp_path):
    loop, task = _reskill_loop(tmp_path, {1: THIRD, 2: THIRD})
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESKILL" and d.reskill["placed"] and loop.lineages[task.id].status == "kept"
    assert d.acceptance_band["decision"] == d.band["decision"] == "in_band" and not _undecided(d)


async def test_a_forced_final_look_escalates_to_k_cap_then_places_or_holds(tmp_path):
    loop, task, student = _loop(tmp_path / "a", tag="hard")  # 0/16 is confident, but its interval reaches above 0.1
    (d,) = await loop.run_typed_round(1)
    u = _undecided(d)
    assert u and u[0]["action"] == "more_rollouts" and u[0]["facts"]["straddles"] and u[0]["facts"]["rollouts"] == 16
    assert d.band["decision"] == "too_hard" and d.band["n"] == 32 and settled(BandResult.model_validate(d.band), SPEC) and d.move == "DESCEND"
    assert max(i for t, i in student.asked if t == task.id) >= 31
    loop2, task2, _ = _loop(tmp_path / "b", tag="band")
    loop2.probe = VStudent({1: EDGE})  # p = 0.6: no measurement within the cap can say which side of the edge
    (d2,) = await loop2.run_typed_round(1)
    assert d2.move == "HOLD" and loop2.lineages[task2.id].status == "held" and [x["action"] for x in _undecided(d2)] == ["more_rollouts", "hold"]
    assert "band_undecided" in d2.why[-1] and d2.band["n"] == 32


async def test_the_per_lineage_rollout_cap_stops_the_escalation(tmp_path):
    loop, task, _ = _loop(tmp_path, tag="band", rules=TaskRules(rollouts=8, kappa=50.0, task_rollout_cap=16))
    loop.probe = VStudent({1: EDGE})
    (d,) = await loop.run_typed_round(1)
    (u,) = _undecided(d)
    assert u["action"] == "hold" and u["facts"]["probe_budget_ok"] is False and d.band["n"] == 16 and loop.next_index[task.id] == 16


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(p1=st.sampled_from([0.0, 0.05, 0.2, 0.35, 0.55, 0.65, 0.9, 1.0]), p2=st.sampled_from([0.0, 0.05, 0.2, 0.35, 0.55, 0.65, 0.9, 1.0]),
       seed=st.integers(min_value=0, max_value=10_000))
def test_the_loop_never_places_a_task_on_an_unsettled_or_disagreeing_measurement(p1, p2, seed):
    """Property (any pass rates, any seed): a placement (KEEP, or a RESKILL kept or reserved) rests on settled bands that
    agree; diagnosis and acceptance rollouts are disjoint; every band_undecided remedy is more evidence or HOLD; the
    band edges and validity parameters the loop uses are the ledger's."""

    def bern(p, salt):
        return lambda i: random.Random(f"{seed}|{salt}|{i}").random() < p

    async def go():
        with tempfile.TemporaryDirectory() as td:
            loop, task = _reskill_loop(Path(td), {1: bern(p1, 1), 2: bern(p2, 2)})
            loop.knobs = Knobs(allow=())  # no variant: the placement path alone
            (d,) = await loop.run_typed_round(1)
            lin = loop.lineages[task.id]
            assert set(d.rollouts.get("acceptance", [])).isdisjoint(d.rollouts["diagnosis"])
            for u in _undecided(d):
                assert u["action"] in ("more_rollouts", "hold")
                if u["action"] == "more_rollouts":
                    assert u["facts"]["rollouts"] < 32 and u["facts"]["probe_budget_ok"] is True
            if d.move == "KEEP":
                assert settled(BandResult.model_validate(d.band), SPEC) and d.band["decision"] == "in_band"
            if d.move == "RESKILL" and lin.status in ("kept", "reserved"):
                a, b = BandResult.model_validate(d.band), BandResult.model_validate(d.acceptance_band)
                assert settled(a, SPEC) and settled(b, SPEC) and a.decision == b.decision
                assert (lin.status == "kept") == (b.decision == "in_band")
            if d.move == "RESERVE" and d.band:
                assert settled(BandResult.model_validate(d.band), SPEC)  # out of band, decided
            assert loop.next_index[task.id] <= loop.rules.task_rollout_cap
            assert (loop.rules.band_lo, loop.rules.band_hi, loop.rules.band_alpha) == (0.1, 0.6, 0.1)

    asyncio.run(go())


# ---------------------------------------------------------------------------------------------------------- R-too-hard family_knobs

class CountingKnobs(Knobs):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.applied = []

    async def moves(self, task, direction):
        return [KnobMove(knob="paginate" if direction == "harder" else "fewer_distractors", direction=direction)]

    async def apply(self, task, move):
        self.applied.append(move.knob)
        return await super().apply(task, move)


async def test_finance_ap_has_no_lawful_knob_so_the_loop_reserves_without_trying_one(tmp_path):
    k = CountingKnobs()
    loop, task, _ = _loop(tmp_path / "a", tag="easy", knobs=k)
    fin = task.model_copy(update={"domain": "finance_ap"}, deep=True)
    loop.tasks, loop.lineages, loop.of, loop.cells0 = {}, {}, {}, {}
    loop.add_tasks([fin])
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESERVE" and k.applied == [] and "family_knobs" in " ".join(d.why)
    (r,) = [x for x in d.remedies if x["condition"] == "band_too_easy"]
    assert r["facts"]["harder_knob"] is None and r["action"] == "reserve"
    k2 = CountingKnobs()  # another family: the declared knob is still tried
    loop2, _, _ = _loop(tmp_path / "b", tag="easy", knobs=k2)
    (d2,) = await loop2.run_typed_round(1)
    assert k2.applied == ["paginate"] and d2.move == "ASCEND"


# ---------------------------------------------------------------------------------------------------------- R-p3

async def test_an_unmeasured_p3_is_a_named_condition_with_a_lawful_remedy_never_on_intent(tmp_path):
    loop, _task, _ = _loop(tmp_path, tag="easy")

    async def unmeasured(t):
        return None, ["unmeasured: no valid EVID, SWAP or TWIN trial"]

    loop.p3_check = unmeasured
    (d,) = await loop.run_typed_round(1)
    assert d.move == "ASCEND" and not d.accepted and any(x.startswith("G2: p3_unmeasured") for x in d.refused_by)
    (r,) = [x for x in d.remedies if x["condition"] == "p3_unmeasured"]
    assert (r["rule"], r["action"], r["facts"]["edit"]) == ("R-p3", "hold", "variant")
    assert resolve(Condition(kind="p3_unmeasured", facts={"executed_check_available": True})).action == "executed_check"
    with pytest.raises(PolicyViolation):  # unmeasured never counts as on intent, by construction
        from crucible.synth.policy.model import ParamChange, Remedy

        Remedy(action="hold", rung="defer", disposition="HOLD", changes=[ParamChange(rule="R-p3", param="unmeasured_counts_as_on_intent", old=False, new=True)])


# ---------------------------------------------------------------------------------------------------------- R-lemma-lift

def _pair(task, lineage, defects=("D9",), changed=("contract",)):
    return RepairPair(id=f"pair-{task}", task_id=task, lineage=lineage, domain="finance_ap", from_version=1, to_version=2, defects=list(defects),
                      cx=[f"MUT/SwapEntity/{x}" for x in defects], changed=list(changed), notes="pinned the row the instruction names",
                      evidence={"task_ids": [task], "entity_keys": ["INV-47820"], "literals": ["approved"]})


def _abstract(rule="Pin the entity the instruction names by its key, not by a field of it."):
    calls = []

    async def run(clusters):
        calls.append(clusters)
        assert all("task" not in c and "tasks_ids" not in c for c in clusters)  # clusters only: no task id reaches the abstractor
        return [{"applies_to": "contract_writer", "rule": rule, "evidence": ["r0"]}]

    return run, calls


async def test_a_class_on_one_lineage_is_no_candidate_and_a_recurring_one_is_deferred_when_underpowered(tmp_path):
    ab, calls = _abstract()
    lift = LemmaLift(tmp_path, abstract=ab)
    assert await lift.add(_pair("t1", "t1")) == [] and not calls  # one task's fix is that task's
    assert await lift.add(_pair("t1b", "t1")) == []  # the same lineage again does not recur
    (c,) = await lift.add(_pair("t2", "t2"))
    assert c.key == "D9" and c.status == "deferred" and len(c.lineages) == 2 and calls
    assert c.remedy["rule"] == "R-lemma-lift" and c.remedy["action"] == "no_update" and c.gate["units_available"] == 0
    assert c.gate["units_needed"] >= 40 and lift.bank.lemmas == []  # alpha_t 0.005, gain 0.2, power 0.8: dozens of paired cells
    again = LemmaLift(tmp_path)  # persisted
    assert set(again.pairs) == {"pair-t1", "pair-t1b", "pair-t2"} and again.candidates[c.id].status == "deferred"


async def test_only_the_gate_admits_a_lemma_and_a_leaking_rule_never_reaches_it(tmp_path):
    seen = []

    async def gate(cand, units):
        seen.append(units)
        return LemmaDecision(accepted=True)

    ab, _ = _abstract()
    lift = LemmaLift(tmp_path / "a", abstract=ab, gate=gate, gate_budget_usd=lambda: 100.0)
    await lift.add(_pair("t1", "t1"))
    (c,) = await lift.add(_pair("t2", "t2"))
    assert c.status == "accepted" and seen and len(lift.bank.lemmas) == 1 and lift.bank.lemmas[0].roles == ["writer"]
    assert lift.bank.lemmas[0].sources == c.pairs and lift.bank.lemmas[0].defects == ["D9"]
    (c2,) = await lift.add(_pair("t3", "t3"))
    assert c2.status == "accepted" and len(seen) == 1  # tested once: its alpha is spent once

    async def refuse(cand, units):
        return LemmaDecision(accepted=False, reasons=["paired: the rate did not fall"])

    lift2 = LemmaLift(tmp_path / "b", abstract=ab, gate=refuse, gate_budget_usd=lambda: 100.0)
    await lift2.add(_pair("t1", "t1"))
    (c3,) = await lift2.add(_pair("t2", "t2"))
    assert c3.status == "refused" and lift2.bank.lemmas == []
    leak, _ = _abstract("Pin INV-47820 by its key.")  # names an entity of its evidence
    lift3 = LemmaLift(tmp_path / "c", abstract=leak, gate=gate, gate_budget_usd=lambda: 100.0)
    await lift3.add(_pair("t1", "t1"))
    (c4,) = await lift3.add(_pair("t2", "t2"))
    assert c4.status == "leaking" and c4.leaks and c4.remedy["action"] == "no_update" and lift3.bank.lemmas == []
    lift4 = LemmaLift(tmp_path / "d", abstract=ab, gate=None, gate_budget_usd=lambda: 100.0)  # funded but no gate wired: deferred
    await lift4.add(_pair("t1", "t1"))
    (c5,) = await lift4.add(_pair("t2", "t2"))
    assert c5.status == "deferred" and lift4.bank.lemmas == []


async def test_an_accepted_repair_reaches_the_lift_through_the_loop(tmp_path):
    from tests.unit.test_synth_taskloop import Classifier as C0

    over = {0: ("unintended_pass", "leak", "executed")}

    async def fixed(t, cxs):
        return t.model_copy(update={"version": t.version + 1})

    class Fresh(C0):
        async def classify(self, t, rollouts):
            outs = await super().classify(t, rollouts)
            return [o.model_copy(update={"kind": "intended_pass"}) if o.i >= 8 and o.kind.startswith("unintended") else o for o in outs]

    async def no_replay(t, prog):
        return None

    loop, task, _ = _loop(tmp_path, tag="band", classifier=Fresh(over), repair=fixed)
    loop.replay, loop.recheck = no_replay, _clean
    lift = LemmaLift(tmp_path / "lift", lineage_of=lambda tid: loop.of.get(tid, tid))
    loop.lift_lemmas = lift
    (d,) = await loop.run_typed_round(1)
    assert d.move == "REPAIR" and d.accepted
    (p,) = lift.pairs.values()
    assert p.task_id == task.id and p.from_version == task.version and p.to_version == task.version + 1 and p.defects
    assert d.lessons == {"lemma": []}  # one lineage: recorded, no candidate yet


def test_the_accepted_repairs_of_an_earlier_run_become_pairs(tmp_path):
    run = tmp_path / "run"
    (run / "loop").mkdir(parents=True)
    from crucible.synth.meta.rsi import TaskDecision

    decs = [TaskDecision(task_id="a/t1", version=1, round=1, move="REPAIR", accepted=True, new_version=2),
            TaskDecision(task_id="a/t2", version=1, round=1, move="REPAIR", accepted=False),
            TaskDecision(task_id="a/t3", version=1, round=1, move="KEEP", accepted=True)]
    (run / "loop" / "task_state.json").write_text(json.dumps({"decisions": [x.model_dump(mode="json") for x in decs], "of": {"a/t1": "a/t1"}}))
    (run / "repairs.jsonl").write_text(json.dumps({"task_id": "a/t1", "from_version": 1, "result": "candidate", "changed": ["contract"],
                                                   "notes": "Pinned the status update to invoice key 101 (INV-47820)."}) + "\n")
    Archive(run / "archive").add_all([Counterexample(id="cx-1", domain="finance_ap", task_id="a/t1", task_version=1, cls="MUT", op="SwapEntity",
                                                     site="s", defect="D9", label="confirmed", probe=Probe(cls="MUT", op="SwapEntity", site="s"))])
    (p,) = pairs_from_run(run, LemmaLift(tmp_path / "lift"))
    assert (p.task_id, p.to_version, p.defects, p.changed) == ("a/t1", 2, ["D9"], ["contract"])
    assert "INV-47820" not in p.notes and "101" not in p.notes and p.evidence["entity_keys"]  # scrubbed for the abstractor; kept for the critic
