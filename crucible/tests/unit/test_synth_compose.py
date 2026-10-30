"""crucible.synth.compose: seq/par composition, dependency depth, knobs, ladders and chains (part2-paper-core §10).

Fixture tasks on the AP and helpdesk test Worlds (``synth_fixtures``); every model role is a scripted fake.
"""

from __future__ import annotations

from typing import Any

import pytest

from crucible.synth.archive import Archive
from crucible.synth.canon import Normalizers
from crucible.synth.cegis import refine
from crucible.synth.compose import (
    ACTOR_TOOL,
    Composition,
    CompositionError,
    KnobNotApplicable,
    KnobRecord,
    KnobSpec,
    PageSpec,
    Target,
    VariantSpec,
    alpha_digest,
    apply_knob,
    build_chain,
    compose,
    concat,
    direction_holds,
    ladder,
    page_of,
    part_programs,
    rename,
    select,
    single,
    split_world,
    student_check,
)
from crucible.synth.contract import AddrRef, Contract, PRel, TraceConstraint, grade
from crucible.synth.falsify import FalsifyContext, cheap
from crucible.synth.infer import skeleton
from crucible.synth.program import Program, Step, fn, lit, ref
from crucible.synth.task import IntentCard, Task, WorldRef
from tests.unit import synth_fixtures as fx
from tests.unit.synth_fakes import FakeClient, const, section

AP = WorldRef(factory=f"{fx.MOD}:ap_world")
HD = WorldRef(factory=f"{fx.MOD}:helpdesk_world")


def S(sid: str, tool: str, **args: Any) -> Step:
    return Step(id=sid, tool=tool, args={k: (v if hasattr(v, "op") else lit(v)) for k, v in args.items()})


async def mk(tid: str, ins: str, steps: list[Step], *, answer: Any = None, world: WorldRef = AP, norm: Normalizers = fx.AP_N,
             derive: tuple[Any, ...] = (), trace: list[TraceConstraint] | None = None, contract: Contract | None = None) -> Task:
    """A fixture task whose contract is the exact skeleton of its oracle run (plus ``trace``)."""
    prog = Program(steps=steps, answer=answer)
    t = Task(id=tid, domain="ap-compose", world=world, instruction=ins, contract=Contract(normalizers=norm), oracle=prog,
             intent=IntentCard(skill=tid, derive=list(derive)))
    ctx = FalsifyContext()
    try:
        r = await ctx.run(t, prog, taint=True)
        assert r.error is None and not any(c.obs.error for c in r.calls), (r.error, [c.obs.error for c in r.calls])
        c = contract or skeleton(r, norm)
        if trace:
            c = c.model_copy(update={"trace": trace})
        return t.model_copy(update={"contract": c})
    finally:
        await ctx.close()


def after(*tasks: Task, base: WorldRef = AP) -> WorldRef:
    """The World a follow-up link is built on: the base World after the earlier links' programs."""
    return VariantSpec(base=base, prefix=concat([rename(t.oracle, f"t{k + 1}_") for k, t in enumerate(tasks)])).ref()


HOLD = "Invoice INV-1006 bills more than its goods receipts add up to. Put it on hold with the right reason code."
APPROVE = "Approve the other pending invoices from the vendor of the invoice you just put on hold."
REPORT = "Reply with the total amount of all approved invoices, as a number."


async def chain3() -> list[Task]:
    t1 = await mk("hold", HOLD, [S("s1", "ap.find_invoice", number="INV-1006"), S("s2", "ap.get_receipts", po_id=ref("s1.invoices.0.po")),
                                 S("s3", "ap.hold_invoice", invoice_id=ref("s1.invoices.0.id"), reason="missing_receipt")])
    sel = fn("select", ref("s2.invoices"), {"vendor": "Borealis Freight"})
    t2 = await mk("approve", APPROVE, [S("s1", "ap.list_invoices", status="on_hold"), S("s2", "ap.list_invoices", status="pending"),
                                       S("s3", "ap.approve_invoice", invoice_id=fn("get", sel, "0.id")),
                                       S("s4", "ap.approve_invoice", invoice_id=fn("get", sel, "1.id"))], world=after(t1))
    t3 = await mk("report", REPORT, [S("s1", "ap.list_invoices", status="approved")], answer=fn("sum", fn("pluck", ref("s1.invoices"), "amount")),
                  world=after(t1, t2))
    return [t1, t2, t3]


# ---------------------------------------------------------------------------------------------------------- programs

def test_rename_repoints_refs_and_alpha_digest_ignores_names():
    p = Program(steps=[S("s1", "ap.find_invoice", number="INV-1"), S("s2", "ap.get_invoice", invoice_id=fn("get", ref("s1.invoices.0"), "id"))],
                answer=ref("s2.amount"))
    q = rename(p, "t2_")
    assert [s.id for s in q.steps] == ["t2_s1", "t2_s2"]
    assert q.steps[1].args["invoice_id"].args[0].path == "t2_s1.invoices.0" and q.answer.path == "t2_s2.amount"
    assert alpha_digest(q) == alpha_digest(p) and alpha_digest(rename(q, "t9_")) == alpha_digest(p)
    assert alpha_digest(Program(steps=p.steps[:1])) != alpha_digest(p)


async def test_variant_world_replays_the_prefix_and_pages_and_answers():
    t1 = (await chain3())[0]
    spec = split_world(after(t1))
    assert spec.prefix is not None and spec.base == AP
    ctx = FalsifyContext()
    stub = t1.model_copy(update={"world": VariantSpec(base=AP, prefix=spec.prefix, paginate={"ap.list_invoices": PageSpec(field="invoices", size=2)},
                                                      actors=[{"name": "requester", "facts": {"number": "INV-1006"}}]).ref()})
    w = await ctx.world(stub)
    await w.reset(0)
    c = await w.canon()
    assert c.entity("ap", "invoices", "106")["status"] == "on_hold"  # the prefix (link 1) ran at reset
    o = await w.call("ap.list_invoices", {"status": "pending"})
    assert len(o.value["invoices"]) == 2 and o.value["next_page"] == 1 and o.value["pages"] == 3
    last = await w.call("ap.list_invoices", {"status": "pending", "page": 2})
    assert last.value["next_page"] is None and (await w.call("ap.list_invoices", {"status": "pending", "page": 3})).error
    a = await w.call(ACTOR_TOOL, {"actor": "requester", "topic": "number"})
    assert a.value["answer"] == "INV-1006" and (await w.call(ACTOR_TOOL, {"actor": "boss", "topic": "number"})).error
    names = {t.name: t for t in await w.tools()}
    assert "page" in names["ap.list_invoices"].input_schema["properties"] and names[ACTOR_TOOL].read_only
    assert page_of([1, 2, 3], PageSpec(size=2), 0).value == {"items": [1, 2], "page": 0, "pages": 2, "next_page": 1}
    await ctx.close()


# ---------------------------------------------------------------------------------------------------------- seq / par

async def test_seq_of_three_dependent_links_has_depth_two():
    links = await chain3()
    c = await compose(links)
    t, comp = c.task, c.comp
    assert [s.id for s in t.oracle.steps][:3] == ["t1_s1", "t1_s2", "t1_s3"] and comp.horizon == 8 and comp.writes == 3
    assert t.world == AP  # the composite starts from the base World's s0
    assert grade(c.run, t.contract).passed and c.run.answer == 1710.0
    assert comp.dep == 2 and comp.dep_unconfirmed >= 2
    conf = {(e.src, e.dst, e.kind) for e in comp.edges if e.confirmed}
    assert (0, 1, "literal") in conf and (1, 2, "answer") in conf
    lit_edge = next(e for e in comp.edges if e.confirmed and e.kind == "literal")
    assert lit_edge.value == "Borealis Freight" and lit_edge.entity.endswith("[106]") and lit_edge.obs_step == "t2_s1"
    assert "Borealis Freight" in t.intent.derive and comp.derive_added == ["Borealis Freight"]  # TAINT D7 guards it from now on
    assert comp.instruction_source == "joined" and all(x.instruction in t.instruction for x in links)
    assert comp.signature["dep"] == 2 and comp.signature["parts"] == 3
    # the parts again, from the composite
    parts = part_programs(t, comp)
    assert [len(p.steps) for p in parts] == [3, 4, 1] and parts[2].answer is not None


async def test_a_literal_the_instruction_states_is_no_dependency():
    """The vendor named in the first part's instruction: the agent needs no earlier result to know it (dep falls to 1)."""
    t1, t2, t3 = await chain3()
    t1 = t1.model_copy(update={"instruction": HOLD.replace("Invoice INV-1006", "Invoice INV-1006 from Borealis Freight")})
    c = await compose([t1, t2, t3])
    assert c.comp.dep == 1 and not any(e.kind == "literal" for e in c.comp.edges)


async def test_information_kept_from_an_earlier_observation_is_carry_not_state():
    """Part 2 needs the purchase order part 1 read (before its write): a context dependency (carry 1), not a state one."""
    t1 = (await chain3())[0]
    t_po = await mk("po", "Reply with the approved amount of the purchase order of the invoice you just put on hold, as a number.",
                    [S("s1", "ap.get_purchase_order", po_id="PO-6")], answer=ref("s1.amount"), world=after(t1))
    c = await compose([t1, t_po])
    assert c.comp.dep == 0 and c.comp.carry == 1
    e = next(x for x in c.comp.edges if x.kind == "carry")
    assert e.value == "PO-6" and e.obs_step in ("t1_s1", "t1_s2") and e.src == 0 and not e.confirmed  # the last earlier sighting


async def test_tasks_in_a_row_have_depth_zero_and_par_needs_disjoint_writes():
    t1 = (await chain3())[0]
    t4 = await mk("approve4", "Approve invoice INV-1004.", [S("s1", "ap.find_invoice", number="INV-1004"),
                                                             S("s2", "ap.approve_invoice", invoice_id=ref("s1.invoices.0.id"))])
    c = await compose([t1, t4])
    assert c.comp.dep == 0 and c.comp.horizon == 5
    p = await compose([t1, t4], op="par")
    assert p.comp.op == "par" and p.comp.dep == 0 and grade(p.run, p.task.contract).passed
    t5 = await mk("hold4", "Put invoice INV-1004 on hold for a price variance.",
                  [S("s1", "ap.find_invoice", number="INV-1004"), S("s2", "ap.hold_invoice", invoice_id=ref("s1.invoices.0.id"), reason="price_variance")])
    with pytest.raises(CompositionError, match="does not run after the earlier parts"):  # t4 approved it: t5's hold now errors
        await compose([t4, t5])


async def test_par_refuses_parts_that_write_the_same_addresses():
    a = await mk("pa", "Set ticket T-4's priority to high.", [S("s1", "desk.set_priority", ticket_id="T-4", priority="high")], world=HD, norm=fx.HD_N)
    b = await mk("pb", "Set ticket T-4's priority to low.", [S("s1", "desk.set_priority", ticket_id="T-4", priority="low")], world=HD, norm=fx.HD_N)
    with pytest.raises(CompositionError, match="write the same addresses"):
        await compose([a, b], op="par")


async def test_a_later_write_supersedes_and_a_start_relative_pattern_is_rebased():
    p1 = await mk("p1", "Set ticket T-5's priority to high.", [S("s1", "desk.set_priority", ticket_id="T-5", priority="high")], world=HD, norm=fx.HD_N)
    p2_steps = [S("s1", "desk.get_ticket", ticket_id="T-5"), S("s2", "desk.set_priority", ticket_id="T-4", priority=ref("s1.priority"))]
    p2 = await mk("p2", "Give ticket T-4 the same priority ticket T-5 has now.", p2_steps, world=after(p1, base=HD), norm=fx.HD_N)
    # the writer's generalisation: T-4's new priority equals T-5's priority at the part's start
    r0 = p2.contract.required[0]
    rel = r0.model_copy(update={"value": PRel(rel="eq", ref=AddrRef(app="desk", coll="tickets", key="T-5", field="priority"))})
    p2 = p2.model_copy(update={"contract": p2.contract.model_copy(update={"required": [rel]})})
    p3 = await mk("p3", "Set ticket T-5's priority back to normal.", [S("s1", "desk.set_priority", ticket_id="T-5", priority="normal")],
                  world=after(p1, p2, base=HD), norm=fx.HD_N)
    c = await compose([p1, p2, p3])
    assert "t1." + p1.contract.required[0].id in c.comp.superseded  # T-5 = high is overwritten by part 3
    assert "t2." + rel.id in c.comp.rebased and any(x.startswith("x.") for x in c.comp.rebased)  # relative to s0 it no longer holds
    assert grade(c.run, c.task.contract).passed
    # part 2 copies the priority part 1 wrote, but part 1's instruction states it ("high"): no dependency on the result
    assert c.comp.dep == 0 and c.comp.dep_unconfirmed == 0


async def test_a_pattern_a_merged_normalizer_re_keys_is_rebased_not_dropped():
    """Part A's insert pattern is keyed by the raw id; part B's id_canon re-keys inserts by content in the merge. The
    requirement must survive as the exact skeleton pattern (compose-1: it was dropped as "superseded" because a later
    glob matched it, and MUT found the lost requirement as DropStep D1). Equal-content inserts are two rows."""
    send = [S("s1", "mail.send", to="ann@example.com", subject="Invoice copy", body="Attached.")]
    a = await mk("ma", "Email ann@example.com the subject Invoice copy with the body Attached.", send, world=HD, norm=Normalizers())
    b = await mk("mb", "Email ann@example.com the subject Invoice copy with the body Attached, once more.", send, world=after(a, base=HD), norm=fx.HD_N)
    c = await compose([a, b])
    assert "t1." + a.contract.required[0].id in c.comp.rebased and not c.comp.superseded
    assert grade(c.run, c.task.contract).passed
    ctx = FalsifyContext()
    dropped = Program(steps=c.task.oracle.steps[1:])  # without part A's mail: one row instead of two
    r = await ctx.run(c.task, dropped)
    assert not grade(r, c.task.contract).passed
    await ctx.close()


async def test_answers_merge_into_one_list_with_per_part_paths():
    t1, t2, t3 = await chain3()
    t0 = await mk("onhold", "Reply with the total amount of all invoices on hold, as a number.", [S("s1", "ap.list_invoices", status="on_hold")],
                  answer=fn("sum", fn("pluck", ref("s1.invoices"), "amount")))
    t1b = t1.model_copy(update={"world": AP})
    c = await compose([t0, t1b, t2.model_copy(update={"world": after(t0, t1b)}), t3.model_copy(update={"world": after(t0, t1b, t2)})])
    assert c.comp.answer_parts == [0, 3] and c.run.answer == [500.0, 1710.0]
    assert sorted(q.path for q in c.task.contract.answer) == ["0", "1"]
    swapped = c.run.model_copy(update={"answer": [1710.0, 500.0]})
    assert not grade(swapped, c.task.contract).passed and "list of the values" in c.task.instruction


async def test_trace_constraints_merge_and_conflicts_refuse():
    t1 = (await chain3())[0]
    never = TraceConstraint(id="m0", op="never", tool="ap.approve_invoice", stated="Do not approve anything.")
    t_never = await mk("nv", "Do not approve anything. Put invoice INV-1004 on hold for a price variance.",
                       [S("s1", "ap.find_invoice", number="INV-1004"), S("s2", "ap.hold_invoice", invoice_id=ref("s1.invoices.0.id"), reason="price_variance")],
                       trace=[never])
    t4 = await mk("approve4", "Approve invoice INV-1004.", [S("s1", "ap.find_invoice", number="INV-1004"), S("s2", "ap.approve_invoice", invoice_id=ref("s1.invoices.0.id"))])
    t_app1 = await mk("approve1", "Approve invoice INV-1001.", [S("s1", "ap.find_invoice", number="INV-1001"),
                                                                 S("s2", "ap.approve_invoice", invoice_id=ref("s1.invoices.0.id"))])
    with pytest.raises(CompositionError, match="forbids"):
        await compose([t_never, t_app1])
    most = TraceConstraint(id="m0", op="at_most", tool="ap.find_invoice", n=1)
    t_most = t1.model_copy(update={"contract": t1.contract.model_copy(update={"trace": [most]})})
    c = await compose([t_most, t4])
    m = next(x for x in c.task.contract.trace if x.op == "at_most")
    assert m.id == "t1.m0" and m.n == 2  # the other part's own find_invoice call is added


# ---------------------------------------------------------------------------------------------------------- the writer

def writer_script(replies: list[str]):
    def f(msgs, n):
        return {"instruction": replies[min(n, len(replies) - 1)], "notes": ""}

    return f


async def test_the_writer_is_checked_by_execution_and_gets_feedback():
    links = await chain3()
    good = ("Invoice INV-1006 bills more than its goods receipts add up to: put it on hold with the right reason code. Then approve the other "
            "pending invoices from that invoice's vendor. Finally reply with the total amount of all approved invoices, as a number.")
    leaks = good.replace("from that invoice's vendor", "from Borealis Freight")
    hides = good.replace("Invoice INV-1006", "The invoice")
    w = FakeClient("deepseek", writer_script([leaks, hides, good]))
    c = await compose(links, writer=w)
    assert c.comp.instruction_source == "writer" and c.task.instruction == good and c.comp.writer["attempts"] == 3
    assert "Borealis Freight" in w.calls[1][-1]["content"]  # the feedback names the value it must not state
    assert "loses information" in w.calls[2][-1]["content"]  # TAINT: INV-1006 is unsourced without it
    first = w.calls[0][1]["content"]
    assert "## parts" in first and "## must_not_state" in first and "## contract" not in first and "s1" not in section(w.calls[0], "parts")[0]["instruction"]
    assert c.comp.dep == 2 and grade(c.run, c.task.contract).passed


async def test_a_failing_writer_falls_back_to_the_joined_instruction():
    links = await chain3()
    w = FakeClient("deepseek", writer_script(["Do everything for Borealis Freight now please, thanks a lot."]))
    c = await compose(links, writer=w)
    assert c.comp.instruction_source == "joined" and c.comp.writer["error"] and c.task.instruction.startswith("Do the following")


# ---------------------------------------------------------------------------------------------------------- knobs

async def test_distractors_move_the_signature_only_when_the_oracle_sees_them():
    links = await chain3()
    ctx = FalsifyContext()
    one = await single(links[0], ctx)
    v1 = await apply_knob(KnobSpec(kind="distractors", params={"k": 2}), one, ctx=ctx)
    assert not v1.record.kept and v1.record.reason  # find_invoice is exact: the oracle never sees a clone, so nothing moved
    c = await compose(links, ctx=ctx)
    v = await apply_knob(KnobSpec(kind="distractors", params={"k": 2}), c, ctx=ctx)
    assert v.record.kept and v.record.after["distractors"] > v.record.before["distractors"]
    assert v.record.after["observed_entities"] > v.record.before["observed_entities"]
    assert v.composed.task.id.endswith("~d1") or v.composed.task.id.endswith("~d2")
    assert grade(v.composed.run, v.composed.task.contract).passed and split_world(v.composed.task.world).perturb
    await ctx.close()


async def test_paginate_rewrites_the_oracle_to_read_every_page():
    links = await chain3()
    ctx = FalsifyContext()
    c = await compose(links, ctx=ctx)
    v = await apply_knob(KnobSpec(kind="paginate", params={"size": 2}), c, ctx=ctx)
    t = v.composed.task
    assert v.record.kept and v.composed.comp.signature["pages"] >= 2 and v.composed.comp.horizon > c.comp.horizon
    assert any(s.id.endswith("__p1") and s.args["page"].path.endswith(".next_page") for s in t.oracle.steps)
    assert grade(v.composed.run, t.contract).passed and v.composed.comp.dep == 2
    assert sum(len(s.steps) for s in v.composed.comp.segments) == len(t.oracle.steps)
    with pytest.raises(KnobNotApplicable):
        await apply_knob(KnobSpec(kind="paginate", params={"size": 50}), c, ctx=ctx)
    await ctx.close()


async def test_actor_knob_moves_an_instruction_literal_to_a_person():
    t1 = (await chain3())[0]
    ctx = FalsifyContext()
    one = await single(t1, ctx)
    with pytest.raises(KnobNotApplicable, match="writer"):
        await apply_knob(KnobSpec(kind="actor"), one, ctx=ctx)
    good = "An invoice bills more than its goods receipts add up to; ask the requester for its invoice number, then put it on hold with the right reason code."
    w = FakeClient("deepseek", writer_script(["An invoice is short on receipts. Put INV-1006 on hold with the right reason code.", good]))
    v = await apply_knob(KnobSpec(kind="actor"), one, ctx=ctx, writer=w)
    t = v.composed.task
    assert t.oracle.steps[0].tool == ACTOR_TOOL and t.instruction == good and v.record.kept and v.composed.comp.signature["asks"] == 1
    assert split_world(t.world).actors[0].facts == {"number": "INV-1006"}
    assert not any(e.unsourced for e in v.composed.run.taint)  # the literal now comes from the person's answer
    await ctx.close()


async def test_derive_knob_drops_a_stated_literal_the_records_show():
    ins = "Invoice INV-1006 from Borealis Freight is short on receipts: hold it. Then approve the other pending Borealis Freight invoices."
    sel = fn("select", ref("s3.invoices"), {"vendor": "Borealis Freight"})
    t = await mk("hv", ins, [S("s1", "ap.find_invoice", number="INV-1006"), S("s2", "ap.hold_invoice", invoice_id=ref("s1.invoices.0.id"), reason="missing_receipt"),
                             S("s3", "ap.list_invoices", status="pending"), S("s4", "ap.approve_invoice", invoice_id=fn("get", sel, "0.id")),
                             S("s5", "ap.approve_invoice", invoice_id=fn("get", sel, "1.id"))])
    ctx = FalsifyContext()
    one = await single(t, ctx)
    good = "Invoice INV-1006 is short on receipts: hold it. Then approve the other pending invoices from the same vendor."
    v = await apply_knob(KnobSpec(kind="derive"), one, ctx=ctx, writer=FakeClient("deepseek", writer_script([good])))
    assert v.record.kept and "Borealis Freight" in v.composed.task.intent.derive and v.composed.task.instruction == good
    assert v.record.after["derived_literals"] > v.record.before["derived_literals"]
    await ctx.close()


def test_student_hook_is_a_protocol_nothing_here_calls():
    assert direction_holds(0.6, 0.3, "harder") and not direction_holds(0.3, 0.6, "harder") and direction_holds(None, 0.3, "harder") is None


async def test_student_check_records_the_measured_direction():
    class Probe:
        async def p_hat(self, task: Task) -> float | None:
            return 0.2 if "~" in task.id else 0.5

    t1 = (await chain3())[0]
    rec = await student_check(KnobRecord(kind="paginate"), t1, t1.model_copy(update={"id": t1.id + "~p5"}), Probe())
    assert rec.student == "moved" and rec.p_before == 0.5 and rec.p_after == 0.2


# ---------------------------------------------------------------------------------------------------------- ladders, selection

async def test_ladder_emits_new_candidates_of_increasing_difficulty():
    links = await chain3()
    ctx = FalsifyContext()
    lad = await ladder(links, ctx=ctx, knobs=(KnobSpec(kind="distractors", params={"k": 2}), KnobSpec(kind="paginate", params={"size": 2}),
                                              KnobSpec(kind="actor")))
    labels = [(r.label, r.kept) for r in lad.rungs]
    assert labels[:3] == [("base", True), ("depth", True), ("depth", True)]
    assert ("distractors", True) in labels and ("paginate", True) in labels and ("actor", False) in labels  # no writer: dropped, recorded
    ids = [c.task.id for c in lad.candidates]
    assert len(ids) == len(set(ids)) == sum(1 for r in lad.rungs if r.kept)
    hz = [c.comp.horizon for c in lad.candidates]
    assert hz == sorted(hz) and lad.candidates[-1].comp.dep == 2
    await ctx.close()


def test_selection_follows_the_target_and_prefers_depth():
    pool = [Composition(id=f"c{i}", horizon=h, dep=d) for i, (h, d) in enumerate([(5, 0), (20, 0), (20, 1), (25, 2), (30, 2), (50, 2), (8, 1)])]
    got = select(pool, 3)
    assert got[0].dep == 2 and 15 <= got[0].horizon <= 40 and all(c.dep >= 1 for c in got)
    t = Target(dep_share={0: 1.0})
    assert select(pool, 1, t)[0].dep == 0


# ---------------------------------------------------------------------------------------------------------- certification

async def test_a_composite_runs_through_the_full_deterministic_population(tmp_path):
    links = await chain3()
    ctx = FalsifyContext(max_mutants=240)
    c = await compose(links, ctx=ctx, task_id="ap-compose/c3")

    async def no_repair(task, cxs):
        return None

    rep = await refine(c.task, repair=no_repair, ctx=ctx, archive=Archive(tmp_path), falsifiers=cheap(), max_rounds=1)
    cert = rep.certificate
    assert cert is not None
    classes = {k: (s.n, s.c) for k, s in cert.classes.items()}
    assert classes["REP"][1] == 0 and classes["NULL"][1] == 0 and classes["RED"][1] == 0
    assert classes["TAINT"][1] == 0  # the cross-part literal is sourced from an earlier observation, the earlier result is no leak
    assert classes["MUT"][0] >= 20  # dedup by distinct effect still leaves a large neighbourhood on 8 steps
    # TWIN across boundaries: no sibling a later part acts on (σ collision), no status ι names only for a read moved through
    # C ("the approved invoices"): both were false D9s before compose-1
    assert classes["TWIN"][1] == 0 and classes["TWIN"][0] >= 2
    await ctx.close()


# ---------------------------------------------------------------------------------------------------------- chains (swarm)

async def test_build_chain_grows_dependent_links_with_the_swarm_roles(tmp_path):
    from crucible.synth.pipeline import DomainEnv, RunRoles

    links = await chain3()
    by_ins = {t.instruction: t for t in links}

    def proposer(msgs, n):
        dom = section(msgs, "domain")
        ins = REPORT if "Approve the other" in dom else (APPROVE if "Earlier work" in dom else HOLD)
        return {"tasks": [{"skill": "x", "instruction": ins, "derive": []}]}

    def programmer(msgs, n):
        t = by_ins[section(msgs, "task")]
        from crucible.synth.roles import program_doc

        return program_doc(t.oracle)

    roles = RunRoles(proposer=FakeClient("deepseek", proposer), oracle_programmer=FakeClient("deepseek", programmer),
                     contract_writer=FakeClient("glm", const({"edits": []})), repairer=FakeClient("deepseek", const({"give_up": True})))
    env = DomainEnv(name="ap-compose", description="Accounts payable.", seeds=(0,), normalizers=fx.AP_N, world=lambda s: AP)
    got, log = await build_chain(env, roles, length=3, run_dir=tmp_path / "chain", archive_root=tmp_path / "arch", seed=0, candidates=1)
    assert [t.instruction for t in got] == [HOLD, APPROVE, REPORT] and [x.dep for x in log] == [0, 1, 2]
    assert split_world(got[1].world).prefix is not None  # link 2 was built after link 1
    assert "Earlier work" in roles.proposer.calls[1][1]["content"] and "follow-up" in section(roles.proposer.calls[1], "cell")
    c = await compose(got)
    assert c.comp.dep == 2
