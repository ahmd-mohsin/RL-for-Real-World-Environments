"""The Part 2 policy layer (DESIGN §7.7 item 24; runs/policy-1.md): principles and rules as ledger data, the lawful-remedy
procedure, and one enforcement test per rule (the code path obeys it, or a conformance check pins the code's constant to
the rule; rules whose call site lives in a file another agent owns carry a wiring-TODO test naming the exact site)."""

from __future__ import annotations

import inspect
import json

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from crucible.synth import policy
from crucible.synth.certificate import AdmissionPolicy, certify, with_ladder
from crucible.synth.falsify.base import ClassResult, Finding, Outcome, Probe
from crucible.synth.policy import Condition, PolicyLedger, PolicyViolation, Remedy, resolve
from crucible.synth.policy.genesis import PRINCIPLES, RULES, genesis_entries
from crucible.synth.policy.ledger import LEDGER_PATH, EscalationQueue, replay
from crucible.synth.policy.model import ACTIONS, ESCALATION_KINDS, RUNG_ORDER, UNSAFE_EFFECTS, VALIDITY, Escalation, ParamChange, relaxes
from crucible.synth.policy.resolve import CONDITIONS, PRINCIPLE_CHECKS, check
from tests.unit import synth_fixtures as fx

POL = policy.current()


def C(kind: str, **facts) -> Condition:
    return Condition(kind=kind, subject="t", facts=facts)


def _ledger(tmp_path) -> PolicyLedger:
    led = PolicyLedger(tmp_path / "ledger.jsonl")
    led.write_genesis(genesis_entries())
    return led


def _result(cls: str, labels: list[str], *, passed: bool | None = None, defect: str = "D2") -> ClassResult:
    fs = [Finding(cls=cls, op="op", site=f"s{i}", label=lab, defect=defect if lab == "confirmed" else None,  # type: ignore[arg-type]
                  probe=Probe(cls=cls, op="op", site=f"s{i}"), outcome=Outcome(passed=(lab != "clean") if passed is None else passed))
          for i, lab in enumerate(labels)]
    return ClassResult(cls=cls, findings=fs)


# ---------------------------------------------------------------------------------------------------------- the ledger

def test_the_shipped_ledger_is_the_genesis_and_its_chain_verifies():
    entries = PolicyLedger(LEDGER_PATH).entries()
    assert len(entries) >= len(genesis_entries())
    for e, g in zip(entries, genesis_entries(), strict=False):
        assert {k: e[k] for k in g} == json.loads(json.dumps(g)), f"ledger entry {e['seq']} differs from genesis"
    pol = replay(entries)
    assert pol.tag == policy.tag() and pol.tag.startswith(f"policy-v{len(entries)}@")
    assert [p.id for p in pol.principles] == ["P-valid", "P-evidence", "P-more-evidence-first", "P-intent", "P-no-delete", "P-student",
                                               "P-equal-budget", "P-family", "P-person", "P-ledger"]
    assert set(PRINCIPLE_CHECKS) == {p.id for p in PRINCIPLES}  # every principle is testable


def test_a_tampered_ledger_does_not_replay(tmp_path):
    led = _ledger(tmp_path)
    rows = led.entries()
    rows[3]["rule"]["params"]["h"] = 0.1  # an edit in place, not an entry
    with pytest.raises(PolicyViolation, match="chain"):
        replay(rows)
    rows = led.entries()
    del rows[2]
    with pytest.raises(PolicyViolation, match="chain"):
        replay(rows)


def test_every_lead_decision_is_a_rule_with_its_source():
    lead = {r.id: r for r in RULES if r.source.kind == "lead"}
    for rid in ("R-ALT-whole", "R-too-hard", "R-holdout", "R-student-probe", "R-budget", "R-scale", "R-admission-fixed", "R-reserve"):
        assert rid in lead and lead[rid].source.date in ("2026-10-29", "2026-10-30") and lead[rid].source.ref
    for r in RULES:
        assert r.principles and all(p in {x.id for x in PRINCIPLES} for p in r.principles)
        for s in r.ladder:
            assert s.action in ACTIONS


def test_a_ladder_out_of_rung_order_is_not_a_rule():
    from crucible.synth.policy.model import Rule, Source, Step

    with pytest.raises(ValueError, match="evidence -> repair"):
        Rule(id="R-x", title="x", statement="x", applies_to=["k"], source=Source(kind="inferred"),
             ladder=[Step(action="hold"), Step(action="check_oracle")])
    with pytest.raises(ValueError, match="closed"):
        Step(action="lower_the_bar")


# ---------------------------------------------------------------------------------------------------------- P-valid by construction

@pytest.mark.parametrize("key", sorted(VALIDITY))
def test_a_remedy_that_relaxes_any_validity_parameter_cannot_be_built(key):
    rid, prm = key
    old = POL.rules[rid].params[prm]
    new = _loosen(VALIDITY[key], old)
    assert relaxes(rid, prm, old, new)
    with pytest.raises(PolicyViolation, match="P-valid"):
        Remedy(action="hold", rung="defer", disposition="HOLD", changes=[ParamChange(rule=rid, param=prm, old=old, new=new)])


def _loosen(direction: str, old):
    return {"lower_strict": lambda v: float(v) * 2 + 0.01, "higher_strict": lambda v: float(v) - 1, "true_strict": lambda v: False,
            "false_strict": lambda v: True, "superset_strict": lambda v: list(v)[:-1], "subset_strict": lambda v: [*v, "derive_to_given"],
            "fixed": lambda v: (v / 2 if isinstance(v, float) else v)}[direction](old)


def test_the_ledger_refuses_a_relaxation_even_from_the_lead_and_takes_it_only_as_a_persons_principle_answer(tmp_path):
    led = _ledger(tmp_path)
    for src in ("inferred", "lead"):
        with pytest.raises(PolicyViolation, match="P-valid"):
            led.append({"kind": "amend_rule", "rule_id": "R-holdout", "set": {"h": 0.1}, "source": {"kind": src}})
    with pytest.raises(PolicyViolation, match="P-valid"):
        led.append({"kind": "amend_rule", "rule_id": "R-admission-fixed", "set": {"require_alt_pass": False}, "source": {"kind": "lead"}})
    v0 = led.policy().version
    esc = resolve(C("principle_change", principle="P-valid", proposal="h = 0.1",
                    amendment={"kind": "amend_rule", "rule_id": "R-holdout", "set": {"h": 0.1}}), led.policy()).escalation
    assert esc is not None and esc.kind == "principle_change"
    led.answer(esc, "accept", by="lead", date="2026-10-31", rationale="test")
    pol = led.policy()
    assert pol.version == v0 + 1 and pol.param("R-holdout", "h") == 0.1 and pol.rule("R-holdout").version == 2


def test_an_inferred_entry_may_tighten_but_not_loosen_or_retire(tmp_path):
    led = _ledger(tmp_path)
    led.append({"kind": "amend_rule", "rule_id": "R-panel", "set": {"min_decided": 8}, "source": {"kind": "inferred", "ref": "a test"}})
    assert led.policy().param("R-panel", "min_decided") == 8
    with pytest.raises(PolicyViolation):
        led.append({"kind": "amend_rule", "rule_id": "R-ALT-whole", "set": {"usd_per_config_task": 5.0}, "source": {"kind": "inferred"}})
    with pytest.raises(PolicyViolation):
        led.append({"kind": "retire_rule", "rule_id": "R-scale", "source": {"kind": "inferred"}})
    with pytest.raises(PolicyViolation, match="P-person"):
        led.append({"kind": "amend_rule", "rule_id": "R-budget", "set": {"part2_total_usd": 500.0}, "source": {"kind": "lead"}})


# ---------------------------------------------------------------------------------------------------------- resolve: properties

_EFFECTS = sorted({*UNSAFE_EFFECTS, "runs_executed_check", "edits_task", "relabels_intent", "spends_student_rollouts", "new_variant"})
_PARAMS = sorted(VALIDITY)


@st.composite
def _remedy_doc(draw):
    action = draw(st.sampled_from(sorted(ACTIONS)))
    rung = draw(st.sampled_from([ACTIONS[action].rung, *RUNG_ORDER]))
    changes = []
    for key in draw(st.lists(st.sampled_from(_PARAMS), max_size=2)):
        old = POL.rules[key[0]].params[key[1]]
        changes.append({"rule": key[0], "param": key[1], "old": old, "new": draw(st.sampled_from([old, _loosen(VALIDITY[key], old)]))})
    return {"action": action, "rung": rung, "effects": draw(st.lists(st.sampled_from(_EFFECTS), max_size=3)), "changes": changes,
            "evidence": draw(st.sampled_from(["executed", "calibrated", "judged", "none"])),
            "params": {"families": draw(st.lists(st.sampled_from(["qwen", "deepseek", "openai", "nemotron"]), max_size=2))}}


_FACT = st.one_of(st.booleans(), st.integers(min_value=0, max_value=80), st.floats(min_value=0, max_value=300, allow_nan=False),
                  st.sampled_from(["fewer_distractors", "derive_to_given", "paginate", "report", "train"]))
_FACT_NAMES = sorted({f for _, fs in CONDITIONS.values() for f in fs} | {"min_useful_usd", "planned_usd", "units_needed", "builders"})


@settings(max_examples=400, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(kind=st.sampled_from([*CONDITIONS, "a_new_condition"]), facts=st.dictionaries(st.sampled_from(_FACT_NAMES), _FACT, max_size=8),
       extra=st.lists(_remedy_doc(), max_size=4))
def test_resolve_never_returns_a_remedy_that_relaxes_validity_or_breaks_a_principle(kind, facts, extra):
    c = Condition(kind=kind, facts=facts)
    r = resolve(c, POL, extra=extra)
    assert not [x for x in r.changes if relaxes(x.rule, x.param, x.old, x.new)]
    assert not [e for e in r.effects if UNSAFE_EFFECTS.get(e) in ("P-valid", "P-evidence", "P-intent", "P-no-delete", "P-student", "P-family")]
    assert check(r, c, POL) == []
    assert all((x.rule, x.param) == ("R-budget", "stage_caps_usd") for x in r.changes)  # P-ledger: only a stage cap, inside the total
    if r.rung == "escalate":
        assert r.escalation is not None and r.escalation.kind in ESCALATION_KINDS
        assert kind in ("over_budget", "sealed_access", "irreversible_action", "principle_change")  # P-person: the four grounds only
    assert r.policy_version == POL.tag


@settings(max_examples=300, deadline=None)
@given(kind=st.sampled_from(sorted(CONDITIONS)), facts=st.dictionaries(st.sampled_from(_FACT_NAMES), _FACT, max_size=8))
def test_more_evidence_comes_first(kind, facts):
    """P-more-evidence-first: a remedy past the evidence rung is returned only when every evidence step of the rules for
    the condition was inapplicable on the facts or refused by a principle."""
    r = resolve(Condition(kind=kind, facts=facts), POL)
    if r.rung == "evidence":
        return
    seen = {x["candidate"] for x in r.considered}
    for rule in POL.rules_for(kind):
        for s in rule.ladder:
            if s.rung == "evidence" and (s.kinds is None or kind in s.kinds):
                assert f"{rule.id}:{s.action}" in seen


def test_an_unknown_blocking_condition_is_resolved_by_the_principles_alone():
    assert resolve(C("a_new_condition", executed_check_available=True)).action == "executed_check"
    r = resolve(C("a_new_condition"))
    assert (r.rule, r.action, r.disposition) == ("R-default", "hold", "HOLD")


def test_an_escalation_on_any_other_ground_is_refused():
    with pytest.raises(ValueError):
        Escalation(id="e", kind="too_hard", condition=C("band_too_hard"), question="?")  # type: ignore[arg-type]
    esc = {"action": "escalate", "rung": "escalate", "escalation": {"id": "e", "kind": "budget_beyond_total", "condition": C("band_too_hard").model_dump(),
                                                                    "question": "ask the lead?"}}
    assert any(w.startswith("P-person") for w in check(Remedy.model_validate(esc), C("band_too_hard")))
    assert resolve(C("band_too_hard"), extra=[esc]).action == "reserve"
    for eff in ("irreversible", "reads_sealed", "exceeds_part2_total", "deletes"):  # without an escalation, refused
        r = resolve(C("a_new_condition"), extra=[{"action": "executed_check", "rung": "evidence", "effects": [eff]}])
        assert r.action == "hold"


# ---------------------------------------------------------------------------------------------------------- R-ALT-whole

def test_r_alt_whole_ladder_then_reject_and_never_segment():
    r = resolve(C("no_solver_pass", truth_available=True, oracle_checked=False, rungs_tried=0, budget_ok=True))
    assert r.action == "check_oracle"
    r = resolve(C("no_solver_pass", truth_available=True, oracle_checked=True, rungs_tried=0, budget_ok=True))
    assert r.action == "escalate_solver_ladder" and r.params["rung"] == 0 and len(r.params["rungs"]) == 3
    r = resolve(C("no_solver_pass", oracle_checked=True, rungs_tried=3, budget_ok=True))
    assert (r.action, r.disposition, r.params["reason"]) == ("reject", "REJECT", "ALT:no_pass after the solver ladder")
    seg = {"action": "escalate_solver_ladder", "rung": "evidence", "effects": ["segments_task"]}
    r = resolve(C("no_solver_pass", oracle_checked=True, rungs_tried=3, budget_ok=True), extra=[seg])
    assert r.action == "reject" and any("segment" in w or "P-valid" in w for x in r.considered for w in x.get("refused", []))
    with pytest.raises(PolicyViolation):  # the rule itself never changes on failure
        Remedy(action="reject", rung="defer", changes=[ParamChange(rule="R-admission-fixed", param="require_alt_pass", old=True, new=False)])


async def test_r_alt_whole_the_solver_ladder_runs_the_whole_task_at_equal_spend_with_independent_seeds(tmp_path):
    from crucible.synth.solvers import LadderRung, SolverConfig, escalate_alt
    from tests.unit.synth_fakes import FakeClient
    from tests.unit.test_synth_compose2 import Book

    task = fx.t1_hold()
    book = Book()
    await book.learn(task)
    quits = FakeClient("openai", lambda _m, _n: {"done": True, "final": None})
    solver = FakeClient("inkling", book.agent())
    builder = FakeClient("deepseek", book.agent())
    cfgs = {"quick": SolverConfig("quick", quits), "strong": SolverConfig("strong", solver), "own": SolverConfig("own", builder)}
    rungs = [LadderRung("L0", ["quick", "own", "absent"], 1.5), LadderRung("L1", ["strong"], 3.0), LadderRung("L2", ["quick"], 3.0)]
    judged: list[str] = []

    async def judge(t, run):
        judged.append(t.id)
        return False  # the failed attempt is invalid (it did nothing)

    out = await escalate_alt(task, 1, cfgs, builders=["deepseek"], judge=judge, budget_usd=0.0, max_attempts=2, rungs=rungs, out=tmp_path)
    assert out.passed and out.rung == 1 and out.rungs_tried == 2  # L2 never ran
    assert any("own" in s and "P-family" in s for s in out.skipped) and any("absent" in s for s in out.skipped)
    assert {r.config for r in out.rows} == {"quick@L0", "strong@L1"} and judged == [task.id]
    assert all("@L" in t for t in quits.tags + solver.tags) and not builder.calls  # tags carry the rung; a builder never solves
    assert out.policy_version == policy.tag() and (tmp_path / f"ladder-{task.id.replace('/', '_')}.json").exists()

    # the certificate: ALT had no pass; the ladder's pass admits it only with every failed attempt adjudicated
    pol = AdmissionPolicy(u_max={}, require_applicable=[])
    cert = certify(task, [_result("MUT", ["clean"] * 3), _result("ALT", ["clean"], passed=False)], pol)
    assert cert.reasons == ["ALT:no_pass"]
    assert with_ladder(cert, out.rows, out.verdicts).status == "certified"
    assert with_ladder(cert, out.rows, {}).reasons == ["ALT:suspects=1"]  # an unjudged failure stays open
    assert with_ladder(cert, out.rows, {0: True}).reasons == ["ALT:confirmed=1"]  # a valid failed attempt is a D6


# ---------------------------------------------------------------------------------------------------------- R-too-hard and R-band

def test_r_too_hard_move_order():
    base = {"truth_available": True, "oracle_checked": False, "unintended_with_evidence": 1, "reskill_attempted": False, "deriver_available": True,
            "easier_knob": "fewer_distractors"}
    order = []
    f = dict(base)
    for flip in ("oracle_checked", "unintended_with_evidence", "reskill_attempted", "easier_knob"):
        order.append(resolve(C("band_too_hard", **f)).action)
        f[flip] = {"oracle_checked": True, "unintended_with_evidence": 0, "reskill_attempted": True, "easier_knob": None}[flip]
    order.append(resolve(C("band_too_hard", **f)).action)
    assert order == ["check_oracle", "repair_with_counterexamples", "reskill", "descend", "reserve"]
    assert resolve(C("band_too_hard", **{**f, "easier_knob": "derive_to_given"})).action == "reserve"  # never a DESCEND
    assert resolve(C("band_too_easy", **{**f, "harder_knob": "paginate"})).action == "ascend"
    r = resolve(C("band_too_hard", **f), extra=[{"action": "descend", "rung": "variant", "params": {"knob": "derive_to_given"}}])
    assert r.action == "reserve" and any("never an easier move" in w for x in r.considered for w in x.get("refused", []))
    r = resolve(C("band_too_hard", **f), extra=[{"action": "repair_with_counterexamples", "rung": "repair", "effects": ["changes_skill"]}])
    assert r.action == "reserve"  # P-intent


def test_r_too_hard_and_r_band_conform_to_taskloop():
    """Conformance: meta/rsi.py (owned by another agent) holds the same constants as the ledger."""
    from crucible.synth.meta.rsi import TaskMove, TaskRules

    tr = TaskRules()
    assert (tr.band_lo, tr.band_hi, tr.band_alpha) == (POL.param("R-band", "band_lo"), POL.param("R-band", "band_hi"), POL.param("R-band", "alpha"))
    assert list(tr.looks) == POL.param("R-band", "looks") and tr.looks[-1] == POL.param("R-band", "k_max")
    assert tr.reskill_cap == POL.param("R-too-hard", "reskill_cap") and tr.kappa == POL.param("R-too-hard", "kappa")
    assert tr.max_rounds == POL.param("R-too-hard", "max_rounds") and tr.uncertain_hold == POL.param("R-uncertain", "uncertain_hold")
    assert {"REPAIR", "RESKILL", "DESCEND", "ASCEND", "KEEP", "RESERVE"} <= set(TaskMove.__args__) and "RETIRE" not in TaskMove.__args__


async def test_taskloop_resolves_through_the_policy(tmp_path):
    """TaskLoop.step raises the named conditions and acts on resolve(c).action, recording the version and the rule
    (DESIGN item 23 wiring of item 24)."""
    from crucible.synth.meta.rsi import KnobMove
    from tests.unit.test_synth_taskloop import Knobs, _loop

    class Illegal(Knobs):  # derive -> given is never an easier move (R-too-hard forbidden_easier)
        async def moves(self, task, direction):
            return [KnobMove(knob="derive_to_given", direction=direction)]

    loop, _task, _ = _loop(tmp_path / "a", tag="hard", knobs=Illegal())
    (d,) = await loop.run_typed_round(1)
    assert d.move == "RESERVE" and d.rule == "R-too-hard" and d.policy_version == policy.tag()
    (r,) = [x for x in d.remedies if x["condition"] == "band_too_hard"]
    assert r["action"] == "reserve" and r["facts"]["easier_knob"] == "derive_to_given" and r["considered"]
    loop2, _, _ = _loop(tmp_path / "b", tag="hard")
    (d2,) = await loop2.run_typed_round(1)
    assert d2.move == "DESCEND" and d2.accepted and d2.remedies[-1]["action"] == "descend"
    loop3, _, _ = _loop(tmp_path / "c", tag="easy")
    (d3,) = await loop3.run_typed_round(1)
    assert d3.move == "ASCEND" and d3.remedies[-1]["condition"] == "band_too_easy" and d3.remedies[-1]["action"] == "ascend"


def test_r_band_more_rollouts_within_the_cap_else_hold():
    """R-band v2 (taskloop-2): an undecided measurement escalates to k_cap (32), then holds."""
    assert POL.param("R-band", "k_cap") == 32 and POL.param("R-band", "k_max") == 16 and POL.rule("R-band").version == 2
    assert resolve(C("band_undecided", rollouts=8, probe_budget_ok=True)).action == "more_rollouts"
    assert resolve(C("band_undecided", rollouts=16, probe_budget_ok=True)).action == "more_rollouts"
    assert resolve(C("band_undecided", rollouts=32, probe_budget_ok=True)).action == "hold"
    assert resolve(C("band_undecided", rollouts=8, probe_budget_ok=False)).action == "hold"
    assert resolve(C("mostly_uncertain_rollouts", task_evidence_run=False)).action == "gather_task_evidence"
    assert resolve(C("mostly_uncertain_rollouts", task_evidence_run=True)).action == "hold"


# ---------------------------------------------------------------------------------------------------------- R-holdout

def test_r_holdout_h_is_the_policys_and_held_out_mutants_never_count():
    from crucible.synth.falsify import HOLDOUT, MUT, population
    from crucible.synth.falsify.mut import MUTHeldOut

    assert HOLDOUT == policy.holdout() == 0.2
    assert inspect.signature(MUT).parameters["holdout"].default == HOLDOUT
    assert inspect.signature(MUTHeldOut).parameters["holdout"].default == HOLDOUT
    assert any(isinstance(f, MUT) and f.split == "shown" for f in population(HOLDOUT))
    pol = AdmissionPolicy(require_applicable=[], require_alt_pass=False)
    cert = certify(fx.t1_hold(), [_result("MUT", ["clean"] * 50), _result("MUT-HO", ["clean"] * 20)], pol)
    assert cert.status == "provisional" and cert.classes["MUT"].n == 50  # 70 mutants in all, 50 shown: the bound uses 50
    assert "MUT-HO" not in pol.u_max
    assert certify(fx.t1_hold(), [_result("MUT", ["clean"] * 60), _result("MUT-HO", ["clean", "confirmed"])], pol).reasons == ["MUT-HO:confirmed=1"]


def test_r_holdout_densify_then_provisional_and_survivors_reject():
    assert resolve(C("underpowered_certificate", shown_n=48, can_densify=True, densify_tries=0)).action == "densify_world"
    r = resolve(C("underpowered_certificate", shown_n=48, can_densify=True, densify_tries=2))
    assert (r.action, r.disposition) == ("provisional", "PROVISIONAL")
    assert resolve(C("underpowered_certificate", shown_n=48, can_densify=False)).action == "provisional"
    assert resolve(C("heldout_survivor", survivors=1)).disposition == "REJECT"
    r = resolve(C("underpowered_certificate", shown_n=48, can_densify=False), extra=[{"action": "provisional", "rung": "defer", "effects": ["counts_heldout"]}])
    assert r.effects == [] and r.action == "provisional"


# ---------------------------------------------------------------------------------------------------------- R-admission-fixed

def test_r_admission_fixed_is_the_certificate_default_and_every_certificate_records_the_version():
    from crucible.synth.pipeline import admission_policy

    p = policy.admission_params()
    pol = admission_policy()
    assert pol == AdmissionPolicy() and set(pol.deterministic) == set(p["zero_confirmed_classes"]) and {"ORACLE", "LEAK", "ALT"} <= set(pol.deterministic)
    assert pol.u_max == {"MUT": p["u_max_mut"]} and pol.require_alt_pass and not pol.allow_open_suspects
    t = fx.t1_hold()
    ok = [_result(k, ["clean"] * (60 if k == "MUT" else 2)) for k in ("MUT", "NULL", "RED", "TAINT", "REP")] + [_result("ALT", ["clean"], passed=True)]
    c = certify(t, ok, pol)
    assert c.status == "certified" and c.policy_version == policy.tag() and c.deviations == []
    for cls in ("ORACLE", "LEAK", "ALT"):
        bad = certify(t, [*ok, _result(cls, ["confirmed"], defect="D6")], pol)
        assert bad.status == "rejected" and f"{cls}:confirmed=1" in bad.reasons
    assert certify(t, [*ok, _result("TWIN", ["suspect"])], pol).reasons == ["TWIN:suspects=1"]
    assert certify(t, ok[:-1], pol).reasons == ["ALT:no_pass"]


def test_a_looser_admission_policy_is_recorded_and_keeps_its_certificate_out_of_the_pool(tmp_path):
    from crucible.synth.falsify.patch import PATCH_POLICY
    from crucible.synth.perfect import load_pool

    loose = certify(fx.t1_hold(), [_result("MUT", ["clean"] * 3)], AdmissionPolicy(u_max={}, require_alt_pass=False, require_applicable=[]))
    assert loose.status == "certified" and "require_alt_pass off" in loose.deviations and any("u_max" in d for d in loose.deviations)
    assert policy.admission_deviations(PATCH_POLICY, PATCH_POLICY.scope) == []  # R-patch-scope: the one recorded exemption
    assert policy.admission_deviations(PATCH_POLICY) != []  # outside its scope it is a deviation
    run = tmp_path / "run1"
    (run / "tasks" / "t01").mkdir(parents=True)
    (run / "report.json").write_text(json.dumps({"tasks": [{"n": 1, "status": "certified"}]}))
    (run / "tasks" / "t01" / "cegis.json").write_text(json.dumps({"task": fx.t1_hold().model_dump(mode="json"), "certificate": loose.model_dump(mode="json")}))
    assert load_pool(tmp_path) == []


# ---------------------------------------------------------------------------------------------------------- R-panel

async def test_r_panel_an_uncalibrated_panel_abstains_and_the_remedy_is_evidence_then_a_person():
    from crucible.synth.falsify.agents import Anchor, Attempt, Panel, PanelMember, calibrate
    from tests.unit.synth_fakes import FakeClient

    t1 = fx.t1_hold()
    sure = lambda msgs, _n: {"valid": True, "confidence": 0.97}
    p = Panel(members=[PanelMember("a", FakeClient("nemotron", sure)), PanelMember("b", FakeClient("openai", sure))])
    anchors = [Anchor(task=t1, attempt=Attempt(), valid=False, kind="noop"),
               Anchor(task=t1, attempt=Attempt(trace=[{"tool": "ap.hold_invoice", "args": {}}]), valid=True, kind="oracle")]
    rep = await calibrate(p, anchors, **policy.panel_calibration())
    assert not rep.calibrated and p.lam > 1  # it decides nothing
    assert resolve(C("uncalibrated_panel", executed_check_available=True)).action == "executed_check"
    assert resolve(C("uncalibrated_panel", executed_check_available=False, panel_calibrated=False, more_anchors_available=True)).action == "add_calibration_anchors"
    r = resolve(C("open_suspects", cls="LEAK", executed_check_available=False, panel_calibrated=False, more_anchors_available=False))
    assert (r.action, r.disposition) == ("human_queue", "HUMAN_QUEUE")
    r = resolve(C("open_suspects", cls="LEAK", executed_check_available=False, panel_calibrated=False, more_anchors_available=False),
                extra=[{"action": "panel_decide", "rung": "evidence"}])
    assert r.action == "human_queue" and any("calibrated" in w for x in r.considered for w in x.get("refused", []))
    assert resolve(C("open_suspects", cls="MUT", executed_check_available=False, panel_calibrated=True)).action == "panel_decide"
    r = resolve(C("open_suspects", cls="MUT", panel_calibrated=True), extra=[{"action": "panel_decide", "rung": "evidence", "effects": ["admits_on_judgement"]}])
    assert "admits_on_judgement" not in r.effects


def test_r_panel_pipeline_calibration_is_strict_by_default():
    from crucible.synth.pipeline import RunConfig

    assert RunConfig().panel_rule == "policy"


# ---------------------------------------------------------------------------------------------------------- R-student-probe

def test_r_student_probe_caps_and_no_reward_in_j():
    from crucible.factory.synth_student_wire import P2_CAP_USD
    from crucible.synth.learn import BandVerdict

    assert P2_CAP_USD == POL.param("R-student-probe", "p2_cap_usd") and POL.param("R-student-probe", "student_reward_in_J") is False
    assert not {"passes", "p_hat", "reward", "pass_rate"} & set(BandVerdict.model_fields)  # only the derived verdict crosses
    for eff in ("student_reward_in_J", "repair_by_pass_rate"):
        r = resolve(C("band_too_hard", oracle_checked=True), extra=[{"action": "repair_with_counterexamples", "rung": "repair", "effects": [eff]}])
        assert r.action == "reserve"
    r = resolve(C("band_undecided", rollouts=4), extra=[{"action": "more_rollouts", "rung": "evidence"}])
    assert r.action == "hold"  # no probe budget measured: no rollouts
    assert resolve(C("student_probe_over_cap")).action == "hold"


# ---------------------------------------------------------------------------------------------------------- R-budget

def test_r_budget_caps_come_from_the_ledger_and_only_the_total_reaches_a_person(tmp_path):
    from crucible.factory import synth_wire as w

    assert (w.WORLDS_CAP_USD, w.RSI_CAP_USD, w.YIELD_CAP_USD) == tuple(policy.stage_cap(s) for s in ("worlds", "rsi", "yield"))
    r = resolve(C("over_budget", stage="compose2", planned_usd=3.0, min_useful_usd=1.0, stage_remaining_usd=2.0, part2_remaining_usd=60.0, arms=1))
    assert (r.action, r.disposition) == ("trim_plan", "TRIM")
    r = resolve(C("over_budget", stage="compose2", planned_usd=3.0, min_useful_usd=1.0, stage_remaining_usd=0.2, part2_remaining_usd=60.0, arms=1))
    assert r.action == "reallocate_within_total" and r.changes[0].new["compose2"] == pytest.approx(8.0 + 2.8)
    r = resolve(C("over_budget", stage="compose2", planned_usd=3.0, min_useful_usd=1.0, stage_remaining_usd=0.2, part2_remaining_usd=60.0, arms=2,
                  arms_equal=False))
    assert r.action != "reallocate_within_total"  # P-equal-budget
    r = resolve(C("over_budget", stage="yield", planned_usd=80.0, min_useful_usd=10.0, stage_remaining_usd=0.0, part2_remaining_usd=12.0, arms=1))
    assert r.rung == "escalate" and r.escalation.kind == "budget_beyond_total"

    led = _ledger(tmp_path)
    q = EscalationQueue(tmp_path / "escalations.jsonl")
    esc = resolve(C("over_budget", stage="yield", planned_usd=80.0, min_useful_usd=10.0, stage_remaining_usd=0.0, part2_remaining_usd=12.0, arms=1),
                  led.policy()).escalation
    q.add(esc)
    q.add(esc)
    assert [e.id for e in q.open()] == [esc.id]  # asked once
    v0 = led.policy().version
    entry = led.answer(esc, "raise_total", by="lead", date="2026-10-31", rationale="RSI-2 needs it", queue=q)
    assert led.policy().version == v0 + 1 and led.policy().param("R-budget", "part2_total_usd") == pytest.approx(150.0 + 68.0)
    assert entry["kind"] == "answer" and entry["source"]["kind"] == "person" and q.open() == []
    with pytest.raises(PolicyViolation):  # a stage cap may rise only inside the recorded remainder
        led.append({"kind": "amend_rule", "rule_id": "R-budget", "set": {"stage_caps_usd": {**POL.param("R-budget", "stage_caps_usd"), "rsi": 400.0}},
                    "source": {"kind": "inferred"}, "data": {"part2_remaining_usd": 10.0}})
    led.append({"kind": "amend_rule", "rule_id": "R-budget", "set": {"stage_caps_usd": {**POL.param("R-budget", "stage_caps_usd"), "rsi": 45.0}},
                "source": {"kind": "inferred"}, "data": {"part2_remaining_usd": 10.0}})


def test_a_declined_answer_is_also_a_version(tmp_path):
    led = _ledger(tmp_path)
    esc = resolve(C("sealed_access", path="data/sealed/synth-locked", purpose="tune"), led.policy()).escalation
    assert esc.kind == "sealed_access" and {o["id"] for o in esc.options} == {"grant_once", "deny"}
    v0, rules0 = led.policy().version, led.policy().rules
    led.answer(esc, "deny", by="lead", date="2026-10-31", rationale="no")
    assert led.policy().version == v0 + 1 and led.policy().rules == rules0  # the genesis rules, unchanged by a "no"
    with pytest.raises(PolicyViolation, match="P-person"):
        led.append({"kind": "answer", "escalation": {"kind": "sealed_access"}, "choice": "deny", "source": {"kind": "inferred"}})


# ---------------------------------------------------------------------------------------------------------- the other rules

def test_r_scale_r_density_r_family_r_seed_conform():
    from crucible.factory.synth_wire import SynthWire, WireClient
    from crucible.synth.families import FamilyConfig
    from crucible.synth.roles import family_problems
    from crucible.synth.worldgates import N_MIN

    assert policy.scale_target() == (10, 10)
    assert N_MIN == POL.param("R-density", "n_min") == POL.param("R-admission-fixed", "n_min")
    assert FamilyConfig.model_fields["k_min"].default == POL.param("R-density", "k_min")
    assert inspect.signature(WireClient).parameters["seed_by_tag"].default is POL.param("R-seed", "seed_by_tag")
    assert inspect.signature(SynthWire).parameters["seed_by_tag"].default is POL.param("R-seed", "seed_by_tag")
    probs = family_problems({"proposer": ["deepseek"], "oracle_programmer": ["deepseek"], "alt_solver": ["deepseek", "qwen"]})
    assert probs  # a builder or the student solving-as-reference is flagged
    assert resolve(C("same_family_check", alternative_family_available=True)).action == "reassign_family"
    r = resolve(C("same_family_check", builders=["deepseek"]), extra=[{"action": "reassign_family", "rung": "repair", "params": {"families": ["deepseek"]}}])
    assert r.action == "hold"


def test_r_scale_r_density_r_rsi_power_r_equal_budget_ladders():
    assert resolve(C("scale_shortfall", families=2, budget_ok=True)).action == "build_more_families"
    assert resolve(C("scale_shortfall", families=5, budget_ok=True)).action == "synthesize_more"
    assert resolve(C("scale_shortfall", families=5, budget_ok=False)).disposition == "SHORTFALL"
    assert resolve(C("density_gate_failing", can_reseed=True, reseed_tries=0)).action == "reseed_denser"
    assert resolve(C("density_gate_failing", can_reseed=False)).disposition == "NOT_USED"
    assert resolve(C("world_gate_failing", restartable=True, rebuild_tries=1)).action == "rebuild_stage"
    assert resolve(C("underpowered_decision", units_available=400, units_needed=240)).action == "larger_fresh_view"
    assert resolve(C("underpowered_decision", units_available=100, units_needed=240)).disposition == "NO_UPDATE"
    assert resolve(C("unequal_budget_comparison", control_rerun_possible=True, budget_ok=True)).action == "rerun_control_equal_spend"
    assert resolve(C("unequal_budget_comparison", control_rerun_possible=False)).disposition == "NO_CLAIM"
    r = resolve(C("unequal_budget_comparison"), extra=[{"action": "no_claim", "rung": "defer", "effects": ["unequal_budget_claim"]}])
    assert r.effects == []


def test_r_oracle_and_r_reserve_and_the_person_grounds():
    assert resolve(C("wrong_oracle", repair_attempts=0)).action == "repair_oracle"
    assert resolve(C("wrong_oracle", repair_attempts=2)).disposition == "REJECT"
    assert resolve(C("disputed_oracle", truth_available=True, oracle_checked=False)).action == "check_oracle"
    assert resolve(C("disputed_oracle", truth_available=False, panel_calibrated=False)).disposition == "HUMAN_QUEUE"
    assert resolve(C("dead_oracle", retries=1)).action == "retry_oracle"
    assert resolve(C("dead_oracle", retries=3)).disposition == "DROPPED"
    assert resolve(C("delete_request", what="t07")).disposition == "RESERVE"
    assert resolve(C("delete_request"), extra=[{"action": "reserve", "rung": "defer", "effects": ["deletes"]}]).effects == []
    assert resolve(C("irreversible_action", action="git push")).escalation.kind == "irreversible_action"
    assert resolve(C("sealed_access", purpose="report")).action == "hold"  # the report channel needs no person


def test_run_reports_record_the_policy_version(tmp_path):
    from crucible.synth.pipeline import RunConfig, Runner
    from tests.unit.test_synth_compose2 import Book, hd_env, part_roles

    runner = Runner(hd_env(), part_roles(Book()), RunConfig(), tmp_path / "r", tmp_path / "a", run_id="r")
    assert runner.report.policy_version == policy.tag()
