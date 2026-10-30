"""crucible.synth.contract: the trusted grader (part2-paper-core §3.3, C1) and crucible.synth.infer (§3.4).

Covers the assignment semantics (one change is one unit of evidence; counts are exact unless widened; duplicates must
be allowed), every predicate, answer and trace check, fail-closed grading, and M1's acceptance: oracle runs pass their
inferred contracts and each hand-planted contract D1..D11 yields the reason codes its defect implies.
"""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from crucible.synth.canon import CanonState
from crucible.synth.contract import AddrRef, AnswerCheck, Contract, EffectPattern, EntityPat, ExecCheck, PAny, PApprox, PEq, PFromAnswer, PIn, PRegex, PRel, TraceConstraint, grade
from crucible.synth.iface import Obs
from crucible.synth.infer import skeleton
from crucible.synth.program import CallRecord, Program, Run, Step, lit, run
from tests.unit import synth_fixtures as fx

S0 = {"ap": {"inv": {"42": {"status": "pending", "amount": 100.0, "reason": None}, "43": {"status": "pending", "amount": 50.0, "reason": None},
                     "44": {"status": "approved", "amount": 10.0, "reason": None}},
             "log": {}}}


def _run(changes: dict, *, answer=None, calls=None, start=None) -> Run:
    import copy

    s = copy.deepcopy(start or S0)
    for (coll, key), row in changes.items():
        if row is None:
            del s["ap"][coll][key]
        else:
            s["ap"][coll][key] = {**s["ap"][coll].get(key, {}), **row}
    return Run(start=CanonState.build(start or S0), end=CanonState.build(s), answer=answer, calls=calls or [])


def hold(key: str = "42", pid: str = "h") -> EffectPattern:
    return EffectPattern(id=pid, app="ap", coll="inv", key=key, field="status", value=PEq(v="on_hold"))


# ---------------------------------------------------------------------------------------------------------- assignment

def test_one_change_satisfies_at_most_one_required_pattern():
    c = Contract(required=[hold("42", "h42"), hold("*", "hany")])
    one = _run({("inv", "42"): {"status": "on_hold"}})
    assert grade(one, c).reasons == ["MISSING_REQUIRED:hany"]
    two = _run({("inv", "42"): {"status": "on_hold"}, ("inv", "43"): {"status": "on_hold"}})
    assert grade(two, c).passed


def test_the_assignment_is_a_matching_not_a_greedy_scan():
    """The wildcard pattern comes first; a greedy scan would give it invoice 42 and starve the 42 pattern."""
    c = Contract(required=[hold("*", "hany"), hold("42", "h42")])
    r = _run({("inv", "42"): {"status": "on_hold"}, ("inv", "43"): {"status": "on_hold"}})
    assert grade(r, c).passed


def test_counts_are_exact_and_a_duplicate_needs_an_allowed_pattern():
    ins = EffectPattern(id="log", app="ap", coll="log", kind="insert", where={"act": PEq(v="hold")})
    r = _run({("log", "1"): {"act": "hold"}, ("log", "2"): {"act": "hold"}})
    assert grade(r, Contract(required=[ins])).reasons == ["FORBIDDEN_EFFECT:ap.log[2]"]
    assert grade(r, Contract(required=[ins.model_copy(update={"count_max": "any"})])).passed
    assert grade(r, Contract(required=[ins.model_copy(update={"count": 2})])).passed
    assert grade(r, Contract(required=[ins.model_copy(update={"count": 3})])).reasons == ["MISSING_REQUIRED:log"]
    allowed = EffectPattern(id="a_log", app="ap", coll="log", kind="insert", why="audit rows are written by every decision")
    assert grade(r, Contract(required=[ins], allowed=[allowed])).passed
    assert grade(_run({}), Contract(required=[ins.model_copy(update={"count": 0, "count_max": 1})])).passed


def test_frame_rejects_unrequested_effects_with_their_address():
    r = _run({("inv", "42"): {"status": "on_hold"}, ("inv", "43"): {"amount": 0.0}, ("inv", "44"): None})
    v = grade(r, Contract(required=[hold()]))
    assert v.reasons == ["FORBIDDEN_EFFECT:ap.inv[43].amount", "FORBIDDEN_EFFECT:ap.inv[44]"] and v.codes == {"FORBIDDEN_EFFECT"}
    assert grade(_run({}), Contract(required=[hold()])).reasons == ["MISSING_REQUIRED:h"]


# ---------------------------------------------------------------------------------------------------------- predicates

@pytest.mark.parametrize("pred,new,ok", [
    (PEq(v=1), 1.0, True), (PEq(v=True), 1, False), (PIn(vs=["a", "b"]), "b", True), (PIn(vs=["a"]), "c", False),
    (PApprox(v=100.0, tol=0.5), 100.4, True), (PApprox(v=100.0, rel=0.01), 102.0, False), (PApprox(v=1.0, tol=1), "1", False),
    (PRegex(pattern=r"INV-\d+"), "INV-12", True), (PRegex(pattern=r"INV"), "INV-12", False), (PAny(), None, True),
    (PRel(rel="gt", ref="old"), 150.0, True), (PRel(rel="eq", ref="old", delta=10.0), 110.0, True), (PRel(rel="lt", ref="old"), 150.0, False),
    (PRel(rel="eq", ref=AddrRef(app="ap", coll="inv", key="43", field="amount")), 50.0, True),
    (PFromAnswer(), 120.0, True), (PFromAnswer(path="total"), 120.0, False),
])
def test_value_predicates(pred, new, ok):
    c = Contract(required=[EffectPattern(id="p", app="ap", coll="inv", key="42", field="amount", value=pred)])
    assert grade(_run({("inv", "42"): {"amount": new}}, answer="120"), c).passed is ok


def test_where_predicates_read_the_inserted_row_and_the_updated_entity():
    ins = EffectPattern(id="log", app="ap", coll="log", kind="insert", where={"inv": PEq(v=42), "act": PIn(vs=["hold"])})
    assert grade(_run({("log", "1"): {"inv": 42, "act": "hold"}}), Contract(required=[ins])).passed
    assert not grade(_run({("log", "1"): {"inv": 43, "act": "hold"}}), Contract(required=[ins])).passed
    upd = hold().model_copy(update={"where": {"reason": PEq(v="pv"), "amount": PRel(rel="eq", ref="old")}})
    ok = _run({("inv", "42"): {"status": "on_hold", "reason": "pv"}})
    c = Contract(required=[upd], allowed=[EffectPattern(id="a", app="ap", coll="inv", key="42", field="reason", why="set by the same call")])
    assert grade(ok, c).passed
    assert not grade(_run({("inv", "42"): {"status": "on_hold", "reason": "pv", "amount": 1.0}}), c).passed


def test_state_patterns_count_entities_of_the_end_state():
    st = EffectPattern(id="st", app="ap", coll="inv", key="*", field="status", kind="state", value=PEq(v="approved"), count=2)
    assert grade(_run({("inv", "42"): {"status": "approved"}}), Contract(required=[st], allowed=[
        EffectPattern(id="a", app="ap", coll="inv", why="approvals")])).passed
    assert grade(_run({}), Contract(required=[st])).reasons == ["MISSING_REQUIRED:st"]


# ---------------------------------------------------------------------------------------------------------- answer, trace, exec

@pytest.mark.parametrize("check,answer,ok", [
    (AnswerCheck(id="q", op="eq", value="Acme Tools"), "  acme   tools ", True),
    (AnswerCheck(id="q", op="eq", value=1200), "1,200.00", True),
    (AnswerCheck(id="q", op="approx", value=1200.0, tol=0.5), 1200.4, True),
    (AnswerCheck(id="q", op="approx", value=1200.0, rel=0.001), 1202.0, False),
    (AnswerCheck(id="q", op="set_eq", value=["a", "B"]), ["b", "a", "a"], True),
    (AnswerCheck(id="q", op="set_eq", value=["a", "b"]), ["a", "b", "c"], False),
    (AnswerCheck(id="q", op="regex", value=r"INV-\d{4}"), "INV-1003", True),
    (AnswerCheck(id="q", op="in", value=["x", "y"]), "Y", True),
    (AnswerCheck(id="q", op="present"), None, False),
    (AnswerCheck(id="q", op="eq", path="total", value=5), {"total": 5}, True),
    (AnswerCheck(id="q", op="eq", ref=AddrRef(app="ap", coll="inv", key="42", field="amount"), ref_state="start"), 100, True),
    (AnswerCheck(id="q", op="eq", value=1), None, False),
])
def test_answer_checks(check, answer, ok):
    assert grade(_run({}, answer=answer), Contract(answer=[check])).passed is ok


def _calls(*spec):
    out = []
    for i, (tool, ro, args, obs, eff) in enumerate(spec):
        out.append(CallRecord(step=f"c{i}", tool=tool, read_only=ro, args=args, obs=Obs(value=obs), effects=eff))
    return out


def test_trace_constraints():
    calls = _calls(("ap.get", True, {"id": "42"}, {"id": "42"}, None), ("ap.hold", False, {"id": "42"}, {}, [["ap", "inv", "42"]]),
                   ("ap.hold", False, {"id": "43"}, {}, [["ap", "inv", "43"]]))
    r = _run({}, calls=calls)

    def ok(m):
        return grade(r, Contract(trace=[m])).passed

    assert ok(TraceConstraint(id="m", op="before", tool="ap.get", then="ap.hold"))
    assert not ok(TraceConstraint(id="m", op="before", tool="ap.hold", then="ap.get"))
    assert ok(TraceConstraint(id="m", op="before", tool="ap.x", then="ap.nothing"))
    assert not ok(TraceConstraint(id="m", op="never", tool="ap.h*")) and ok(TraceConstraint(id="m", op="never", tool="ap.force"))
    assert not ok(TraceConstraint(id="m", op="at_most", tool="ap.hold", n=1)) and ok(TraceConstraint(id="m", op="at_most", tool="ap.hold", n=2))
    assert ok(TraceConstraint(id="m", op="read_before_write", entity=EntityPat(app="ap", coll="inv", key="42")))
    assert not ok(TraceConstraint(id="m", op="read_before_write", entity=EntityPat(app="ap", coll="inv")))  # 43 was never read
    with pytest.raises(ValidationError):
        TraceConstraint(id="m", op="before", tool="x")


def test_exec_checks_are_required_checks():
    reg = {"tests_pass": lambda run, args: run.answer == args["want"]}
    c = Contract(exec=[ExecCheck(id="x", name="tests_pass", args={"want": "ok"})])
    assert grade(_run({}, answer="ok"), c, registry=reg).passed
    assert grade(_run({}, answer="no"), c, registry=reg).reasons == ["MISSING_REQUIRED:x"]


# ---------------------------------------------------------------------------------------------------------- fail-closed

def _boom(run, args):
    raise RuntimeError("secret expected value 1234 leaked in a traceback")


@pytest.mark.parametrize("contract,run_,code", [
    (Contract(exec=[ExecCheck(id="x", name="boom")]), _run({}), "GRADER_ERROR:exec"),
    (Contract(exec=[ExecCheck(id="x", name="missing")]), _run({}), "GRADER_ERROR:exec"),
    (Contract(exec=[ExecCheck(id="x", name="slow")]), _run({}), "GRADER_ERROR:timeout"),
    (Contract(required=[hold().model_copy(update={"value": PRegex(pattern="([")})]), _run({("inv", "42"): {"status": "on_hold"}}), "GRADER_ERROR:required"),
    (Contract(required=[hold().model_copy(update={"value": PRel(rel="eq", ref=AddrRef(app="ap", coll="inv", key="nope", field="x"))})]),
     _run({("inv", "42"): {"status": "on_hold"}}), "GRADER_ERROR:missing_state"),
    (Contract(required=[hold()]), Run(start=None, end=None), "GRADER_ERROR:missing_state"),
    (Contract(required=[hold().model_copy(update={"app": "crm"})]), Run(start=CanonState.build({}, ["crm"]), end=CanonState.build({}, ["crm"])),
     "GRADER_ERROR:missing_state"),
])
def test_grader_fails_closed_with_a_fixed_code(contract, run_, code):
    reg = {"boom": _boom, "slow": lambda run, args: time.sleep(2) or True}
    v = grade(run_, contract, registry=reg, deadline_s=0.3)
    assert not v.passed and v.reasons == [code]
    assert "1234" not in v.model_dump_json()  # no exception text leaves the grader


def test_observation_only_contracts_grade_without_state_and_refuse_effects():
    c = Contract(answer=[AnswerCheck(id="q", value=3)], observation_only=True)
    assert grade(Run(answer=3), c).passed and not grade(Run(answer=4), c).passed
    with pytest.raises(ValidationError):
        Contract(required=[hold()], observation_only=True)


def test_contract_validation():
    with pytest.raises(ValidationError):
        Contract(allowed=[EffectPattern(id="a", app="ap")])  # an allowed effect must say why
    with pytest.raises(ValidationError):
        Contract(required=[hold("42", "x"), hold("43", "x")])
    with pytest.raises(ValidationError):
        EffectPattern(id="i", kind="insert", value=PEq(v=1))
    with pytest.raises(ValidationError):
        EffectPattern(id="i", count=2, count_max=1)
    c = fx.t2_approve_answer().contract
    assert Contract.model_validate_json(c.model_dump_json()) == c and Contract.model_validate_json(c.model_dump_json()).digest() == c.digest()


# ---------------------------------------------------------------------------------------------------------- infer + M1 acceptance

@pytest.mark.parametrize("name", sorted(fx.CORRECT))
async def test_oracles_pass_their_hand_contracts_and_their_inferred_skeletons(name):
    task = fx.CORRECT[name]()
    r = await run(await task.world.build(), task.oracle, task.instruction)
    assert grade(r, task.contract).passed
    sk = skeleton(r, task.contract.normalizers)
    assert grade(r, sk).passed
    assert all(p.value is not None or p.where for p in sk.required)


async def test_skeleton_is_exact_eq_on_every_change():
    task = fx.t1_hold()
    r = await run(fx.ap_world(), task.oracle, task.instruction)
    sk = skeleton(r, fx.AP_N)
    shapes = sorted((p.coll, p.key, p.field, p.kind) for p in sk.required)
    assert shapes == [("audit_log", "*", "*", "insert"), ("invoices", "103", "reason", "update"), ("invoices", "103", "status", "update")]
    assert not sk.allowed and not sk.trace and sk.normalizers == fx.AP_N
    ins = next(p for p in sk.required if p.kind == "insert")
    assert set(ins.where) == {"invoice_id", "action", "reason"}  # id canonicalized away, ``at`` ignored
    # without the normalizer the audit row's clock stamp is pinned: a later run of the same work fails it
    raw = skeleton(r)
    late = await run(fx.ap_world(), Program(steps=[Step(id="s0", tool="ap.list_invoices", args={}), *task.oracle.steps]), task.instruction)
    assert not grade(late, raw).passed and grade(late, sk).passed


async def test_skeleton_answer_check():
    task = fx.t2_approve_answer()
    r = await run(fx.ap_world(), task.oracle, task.instruction)
    sk = skeleton(r, fx.AP_N, answer_tol=0.01)
    assert sk.answer[0].op == "approx" and sk.answer[0].value == 1200.0 and grade(r, sk).passed


async def _r(task, program=None, world=None):
    return await run(world or await task.world.build(), program or task.oracle, task.instruction)


def _drop(task, *ids):
    return task.oracle.model_copy(update={"steps": [s for s in task.oracle.steps if s.id not in ids]})


async def test_planted_contracts_give_the_reason_codes_their_defects_imply():
    """M1 acceptance: for each hand-planted contract, the run that exposes it is graded as the defect predicts (the
    correct contract rejects it with the right code; the planted one accepts it, or rejects a valid run)."""
    t1, t2, t3, t6 = fx.t1_hold(), fx.t2_approve_answer(), fx.t3_resolve_email(), fx.t6_resolve()
    # D1: the email is not required any more: a run that skips it passes the planted contract
    r = await _r(t3, _drop(t3, "s4"))
    assert grade(r, t3.contract).reasons == ["MISSING_REQUIRED:r_mail"] and grade(r, fx.planted("D1").contract).passed
    # D2: an extra hold on another invoice is forbidden, unless the frame is widened
    extra = Step(id="x", tool="ap.hold_invoice", args={"invoice_id": lit(104), "reason": lit("price_variance")})
    r = await _r(t1, t1.oracle.model_copy(update={"steps": [*t1.oracle.steps, extra]}))
    assert grade(r, t1.contract).codes == {"FORBIDDEN_EFFECT"} and grade(r, fx.planted("D2").contract).passed
    # D3: a wrong answer within the relaxed tolerance
    r = await _r(t2, t2.oracle.model_copy(update={"answer": lit(1300.0)}))
    assert grade(r, t2.contract).reasons == ["ANSWER:q_total"] and grade(r, fx.planted("D3").contract).passed
    # D4: the vacuous requirement holds on the do-nothing run
    noop = await _r(t1, Program())
    assert "MISSING_REQUIRED:r_vac" not in grade(noop, fx.planted("D4").contract).reasons
    # D5 / D7 live in the instruction, not the contract: the oracle still passes (TAINT catches them)
    for d in ("D5", "D7"):
        p = fx.planted(d)
        assert grade(await _r(p), p.contract).passed
    # D6: a valid run that skips the purchase-order read is rejected by the unstated order demand
    r = await _r(t1, _drop(t1, "s2"))
    assert grade(r, t1.contract).passed and grade(r, fx.planted("D6").contract).reasons == ["TRACE:m_po_first"]
    # D8: the one-call shortcut passes (a World defect: NULL catches it)
    p8 = fx.planted("D8")
    sc = Program(steps=[Step(id="a", tool="desk.admin_update", args={"ticket_id": lit("T-3"), "status": lit("resolved"), "resolution": lit("KB-2")})])
    assert grade(await _r(p8, sc), p8.contract).passed
    # D9: acting on another invoice passes once the keys are wildcards
    other = t1.oracle.model_copy(update={"steps": [*t1.oracle.steps[:2], Step(id="s3", tool="ap.hold_invoice", args={"invoice_id": lit(104), "reason": lit("price_variance")})]})
    r = await _r(t1, other)
    assert grade(r, t1.contract).codes == {"MISSING_REQUIRED", "FORBIDDEN_EFFECT"} and grade(r, fx.planted("D9").contract).passed
    # D10: no tool can produce the required reason code: the oracle itself is rejected
    p10 = fx.planted("D10")
    assert "MISSING_REQUIRED:r_reason" in grade(await _r(p10), p10.contract).reasons
    # D11: a contract inferred from one run of the nondeterministic World rejects the next run of the same oracle
    w = fx.helpdesk_world(nondet=True)
    first = await _r(t6, world=w)
    pinned = skeleton(first, t6.contract.normalizers)
    assert grade(first, pinned).passed and not grade(await _r(t6, world=w), pinned).passed
