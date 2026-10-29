"""crucible.synth.program: the oracle DSL, the pure library, the interpreter and literal taint (part2-paper-core §3.3, C3)."""

from __future__ import annotations

import pytest

from crucible.synth.iface import Perturbation
from crucible.synth.program import LIBRARY, LibraryError, Program, Step, fn, lit, ref, run, text_spans
from tests.unit import synth_fixtures as fx


def _sources(r, value):
    return {s.kind for e in r.taint if e.value == value for s in e.sources}


def test_program_static_problems_and_json_roundtrip():
    p = Program(steps=[Step(id="s1", tool="a.t", args={"x": ref("s2.y")}), Step(id="s1", tool="a.t", args={"y": fn("nope", lit(1))})])
    probs = p.problems()
    assert any("names no earlier step" in x for x in probs) and any("duplicate step id" in x for x in probs) and any("unknown function" in x for x in probs)
    t = fx.t4_priority().oracle
    assert t.problems() == [] and Program.model_validate_json(t.model_dump_json()) == t
    assert Program.model_validate_json(t.model_dump_json()).digest() == t.digest()


def test_library():
    L = LIBRARY
    assert L["sum"]([1, 2, 3.5]) == 6.5 and L["sum"](1, 2) == 3 and L["min"]([3, 1]) == 1 and L["max"](3, 9) == 9
    assert L["round"](2.345, 2) == 2.35 and L["len"]([1, 2]) == 2 and L["concat"]([1], [2]) == [1, 2] and L["concat"]("a", 1) == "a1"
    rows = [{"id": 1, "v": 5, "c": "Bob"}, {"id": 2, "v": 9, "c": "ann"}]
    assert L["select"](rows, {"c": "bob"}) == [rows[0]]  # filter values compare like literals (case/whitespace-normalised)
    assert L["select"](rows, {"v": {"op": "gt", "v": 6}}) == [rows[1]]
    assert L["get"]({"a": [{"b": 7}]}, "a.0.b") == 7 and L["pluck"](rows, "id") == [1, 2]
    assert L["add"](1, 2) == 3 and L["sub"](5, 2) == 3 and L["mul"](2, 4) == 8 and L["div"](9, 3) == 3
    for bad in (lambda: L["div"](1, 0), lambda: L["sum"](["x"]), lambda: L["select"]("x", {})):
        with pytest.raises(LibraryError):
            bad()


def test_text_spans_normalised_token_match():
    ins = "Approve INV-1001, pay $1,200.00 to Acme  Tools; ticket T-7 (not T-70)."
    assert ins[slice(*text_spans("INV-1001", ins)[0])] == "INV-1001"
    assert len(text_spans(1200, ins)) == 1 and text_spans("acme tools", ins) and text_spans("T-7", ins) == [text_spans("T-7", ins)[0]]
    assert len(text_spans("T-7", ins)) == 1  # word boundaries: T-70 is not T-7
    assert text_spans("price_variance", "a price variance") == []


async def test_run_records_calls_states_and_taint():
    task = fx.t1_hold()
    w = fx.ap_world()
    r = await run(w, task.oracle, task.instruction, seed=0, given=task.intent.given)
    assert r.error is None and [c.step for c in r.calls] == ["s1", "s2", "s3"]
    assert r.calls[2].args == {"invoice_id": 103, "reason": "price_variance"} and r.calls[2].obs.value["status"] == "on_hold"
    assert r.start.entity("ap", "invoices", "103")["status"] == "pending" and r.end.entity("ap", "invoices", "103")["status"] == "on_hold"
    lit_inv = next(e for e in r.taint if e.value == "INV-1003")
    span = next(s.span for s in lit_inv.sources if s.kind == "instruction")
    assert task.instruction[span[0]:span[1]] == "INV-1003"
    assert _sources(r, "price_variance") == {"tool_spec"}  # stated only by the tool catalogue, never by the instruction
    assert r.calls[0].effects is None and sorted(map(tuple, r.calls[2].effects)) == [("ap", "audit_log", "1"), ("ap", "invoices", "103")]
    assert r.read_only["ap.find_invoice"] and not r.read_only["ap.hold_invoice"]
    assert set(r.anchors) == {"s2.po_id", "s3.invoice_id"}


async def test_taint_observation_given_unsourced_and_structural():
    w = fx.ap_world()
    p = Program(steps=[Step(id="s1", tool="ap.list_invoices", args={"status": lit("pending")}),
                       Step(id="s2", tool="ap.hold_invoice", args={"invoice_id": lit(104), "reason": lit("missing_receipt")}),
                       Step(id="s3", tool="ap.get_invoice", args={"invoice_id": lit(999)}),
                       Step(id="s4", tool="ap.get_receipts", args={"po_id": lit("PO-9")})],
                answer=fn("get", ref("s1.invoices"), lit("0.id")))
    r = await run(w, p, "Hold one pending invoice.", given=["PO-9"])
    assert "observation" in _sources(r, 104)  # read from list_invoices before it was used
    assert next(e for e in r.taint if e.value == 104).sources[0].kind == "observation"
    assert _sources(r, 999) == set()  # nobody ever showed 999: UNSOURCED
    assert _sources(r, "PO-9") == {"given"}
    assert _sources(r, "0.id") == {"structural"}  # get's path is schema, not task information
    assert r.answer == 101
    assert {"instruction", "tool_spec"} <= _sources(r, "pending")


async def test_execution_semantics_tool_errors_continue_and_ref_errors_stop():
    w = fx.ap_world()
    p = Program(steps=[Step(id="s1", tool="ap.get_invoice", args={"invoice_id": lit(1)}),  # tool error: an observation
                       Step(id="s2", tool="ap.approve_invoice", args={"invoice_id": lit(101)}),
                       Step(id="s3", tool="ap.approve_invoice", args={"invoice_id": ref("s1.id")}),  # ref into an errored step
                       Step(id="s4", tool="ap.approve_invoice", args={"invoice_id": lit(102)})])
    r = await run(w, p)
    assert r.calls[0].obs.error is not None and r.calls[1].obs.ok and len(r.calls) == 2
    assert r.error.startswith("step s3: RefError")
    assert r.end.entity("ap", "invoices", "101")["status"] == "approved" and r.end.entity("ap", "invoices", "102")["status"] == "pending"
    bad = Program(steps=[Step(id="s1", tool="ap.get_receipts", args={"po_id": lit("PO-1")})], answer=fn("div", ref("s1.total"), lit(0)))
    r2 = await run(w, bad)
    assert r2.answer is None and r2.error.startswith("answer: LibraryError")


async def test_anchors_reresolve_list_refs_after_a_shuffle():
    """A ref into a list follows the same element (by content) when the list order changes: PERT's re-resolution."""
    w = fx.ap_world()
    p = Program(steps=[Step(id="s1", tool="ap.list_invoices", args={"status": lit("pending")}),
                       Step(id="s2", tool="ap.approve_invoice", args={"invoice_id": ref("s1.invoices.2.id")})])
    r = await run(w, p)
    target = r.calls[1].args["invoice_id"]

    async def shuffle(world):
        await world.perturb(Perturbation(op="shuffle", app="ap", coll="invoices", seed=3))

    plain = await run(w, p, prepare=shuffle)
    again = await run(w, p, prepare=shuffle, anchors=r.anchors)
    assert plain.calls[1].args["invoice_id"] != target  # positionally, another invoice
    assert again.calls[1].args["invoice_id"] == target and again.error is None


async def test_run_is_deterministic_from_the_seed():
    task = fx.t3_resolve_email()
    w = fx.helpdesk_world()
    a = await run(w, task.oracle, task.instruction)
    b = await run(w, task.oracle, task.instruction)
    assert a.model_dump(exclude={"start", "end"}) == b.model_dump(exclude={"start", "end"})
    assert a.end.fingerprint() == b.end.fingerprint() and a.start.fingerprint() == b.start.fingerprint()
