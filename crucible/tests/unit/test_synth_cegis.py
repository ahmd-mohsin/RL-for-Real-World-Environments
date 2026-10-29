"""crucible.synth.archive + cegis: the monotone counterexample archive and the per-task loop (part2-paper-core §7)."""

from __future__ import annotations

import json

import pytest

from crucible.synth.archive import Archive, CorruptArchive, regress
from crucible.synth.cegis import refine, synthesize
from crucible.synth.certificate import AdmissionPolicy
from crucible.synth.contract import AnswerCheck, Contract, EffectPattern
from crucible.synth.falsify import ALT, MUT, AgentSpec, FalsifyContext, cheap, falsify
from crucible.synth.falsify.base import Counterexample
from crucible.synth.program import Program, Step, lit
from crucible.synth.task import Task
from tests.unit import synth_fixtures as fx

LENIENT = AdmissionPolicy(u_max={}, require_alt_pass=False)


async def _cxs(task: Task, ctx: FalsifyContext | None = None) -> list[Counterexample]:
    r = await falsify(task, MUT(), ctx or FalsifyContext())
    return [Counterexample.of(task, f) for f in r.confirmed]


async def test_archive_is_append_only_hash_chained_and_deduplicated(tmp_path):
    a = Archive(tmp_path)
    cxs = await _cxs(fx.planted("D2"))
    assert len(cxs) == 4 and len({c.id for c in cxs}) == 4
    assert a.add_all(cxs) == [c.id for c in cxs] and a.add_all(cxs) == []  # the same counterexample is archived once
    assert a.verify("synth-test") == 4 and [c.id for c in a.records("synth-test", "planted/D2")] == [c.id for c in cxs]
    assert a.records("synth-test", "other") == [] and a.records("empty-domain") == []
    p = a.path("synth-test")
    lines = p.read_text().splitlines()
    assert [json.loads(x)["seq"] for x in lines] == [1, 2, 3, 4]
    # a torn final line (a crash mid-append) is ignored by reads and dropped by the next append
    p.write_text("\n".join(lines) + "\n" + lines[0][:40])
    assert a.verify("synth-test") == 4
    extra = (await _cxs(fx.always_pass()))[0]
    assert a.add(extra) and a.verify("synth-test") == 5 and p.read_text().endswith("\n")
    # an edited line breaks the chain
    lines = p.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["rec"]["defect"] = "D9"
    lines[1] = json.dumps(rec)
    p.write_text("\n".join(lines) + "\n")
    with pytest.raises(CorruptArchive):
        a.records("synth-test")
    with pytest.raises(ValueError):
        a.path("../escape")


async def test_regressions_replay_archived_counterexamples_against_a_version(tmp_path):
    a = Archive(tmp_path)
    bad = fx.planted("D2")
    a.add_all(await _cxs(bad))
    ctx = FalsifyContext()
    fixed = bad.next_version(contract=fx.t1_hold().contract)
    rep = await regress(a, fixed, ctx, cheap())
    assert rep.checked == 4 and rep.ok
    rep = await regress(a, bad.next_version(), ctx, cheap())  # a version that still widens the frame re-opens them all
    assert len(rep.reopened) == 4
    assert (await regress(a, fixed, ctx, [])).skipped  # no falsifier of the class: skipped, never silently passed


def _combined() -> Task:
    """T2 with two defects: a widened frame (D2) and a relaxed answer tolerance (D3)."""
    t = fx.t2_approve_answer()
    wide = [EffectPattern(id="a_inv", app="ap", coll="invoices", kind="any", why="invoice edits are routine"),
            EffectPattern(id="a_log", app="ap", coll="audit_log", kind="any", why="audit rows are routine")]
    c = t.contract.model_copy(update={"allowed": wide, "answer": [AnswerCheck(id="q_total", op="approx", value=1200.0, tol=500.0)]})
    return t.model_copy(update={"id": "combined/D2+D3", "contract": c})


async def test_cegis_refuses_a_repair_that_reopens_an_archived_counterexample(tmp_path):
    good = fx.t2_approve_answer().contract
    calls: list[list[str]] = []

    async def repair(task: Task, cxs: list[Counterexample]) -> Task | None:
        calls.append(sorted({c.defect for c in cxs}))
        n = len(calls)
        if n == 1:  # fix the frame only
            return task.next_version(contract=task.contract.model_copy(update={"allowed": []}))
        if n == 2:  # fix the tolerance but widen the frame again: re-opens the archived D2 counterexamples
            return task.next_version(contract=_combined().contract.model_copy(update={"answer": good.answer}))
        return task.next_version(contract=task.contract.model_copy(update={"answer": good.answer}))

    a = Archive(tmp_path)
    rep = await refine(_combined(), repair=repair, ctx=FalsifyContext(), archive=a, policy=LENIENT)
    assert [r.repair for r in rep.rounds] == ["accepted", "refused", "accepted"]
    assert calls[0] == ["D2", "D3"] and calls[1] == ["D3"] and calls[2] == ["D3"]
    refused = rep.rounds[1]
    d2 = {c.id for c in a.records("synth-test") if c.defect == "D2"}
    assert refused.refused_for and set(refused.refused_for) <= d2
    assert rep.status == "admitted" and rep.task.version == 3 and rep.certificate.clean and not rep.certificate.regressions_reopened
    assert rep.task.contract.answer == good.answer and rep.task.contract.allowed == []
    # every archived counterexample is rejected by the admitted version
    assert (await regress(a, rep.task, FalsifyContext(), cheap())).ok


async def test_cegis_round_cap_and_declined_repairs(tmp_path):
    async def stubborn(task, cxs):
        return task.next_version()  # changes nothing

    rep = await refine(fx.planted("D2"), repair=stubborn, ctx=FalsifyContext(), archive=Archive(tmp_path / "a"), policy=LENIENT)
    assert len(rep.rounds) == 3 and rep.status == "rejected" and any(r.startswith("MUT:confirmed") for r in rep.reasons)

    async def give_up(task, cxs):
        return None

    rep = await refine(fx.planted("D5"), repair=give_up, ctx=FalsifyContext(), archive=Archive(tmp_path / "b"), policy=LENIENT)
    assert [r.repair for r in rep.rounds] == ["declined"] and rep.status == "rejected" and "TAINT:confirmed=1" in rep.reasons
    clean = await refine(fx.t1_hold(), repair=give_up, ctx=FalsifyContext(), archive=Archive(tmp_path / "c"), policy=LENIENT)
    assert clean.status == "admitted" and len(clean.rounds) == 1 and clean.rounds[0].counterexamples == []


async def _lazy(view, world):
    await world.call("ap.approve_invoice", {"invoice_id": 102})
    return "done"


async def test_agent_counterexamples_replay_their_recorded_trace():
    t = fx.t5_batch()
    alt = ALT([AgentSpec("lazy", "fam-y", _lazy)])
    ctx = FalsifyContext()
    r = await falsify(t, alt, ctx)
    (sus,) = r.suspects  # adjudicated (M4) as a valid solution the contract rejects: an archived D6
    cx = Counterexample.of(t, sus)
    assert cx.probe.program is not None and cx.probe.expect == "pass" and cx.probe.program.steps[0].tool == "ap.approve_invoice"
    assert await alt.replay(cx, t, ctx)  # the same version still rejects it: re-opened


async def test_synthesize_with_scripted_roles(tmp_path):
    proto = fx.t1_hold()

    async def propose(cell, domain):
        return proto.model_copy(update={"oracle": Program(), "contract": Contract(normalizers=fx.AP_N), "domain": domain})

    async def build_oracle(task):
        return proto.oracle

    async def generalise(task, sk, run):
        assert run.error is None and sk.required
        return sk

    async def repair(task, cxs):
        return None

    rep = await synthesize("hold-on-variance", "synth-test", propose=propose, build_oracle=build_oracle, generalise=generalise, repair=repair,
                           ctx=FalsifyContext(), archive=Archive(tmp_path), policy=LENIENT)
    assert rep.status == "admitted" and rep.certificate.clean and len(rep.task.contract.required) == 3

    async def dead_oracle(task):
        return Program(steps=[Step(id="s1", tool="ap.hold_invoice", args={"invoice_id": lit(107), "reason": lit("price_variance")})])

    rep = await synthesize("x", "synth-test", propose=propose, build_oracle=dead_oracle, generalise=generalise, repair=repair, ctx=FalsifyContext(),
                           archive=Archive(tmp_path))
    assert rep.status == "dropped" and rep.reasons == ["oracle:dead"]

    async def nothing(cell, domain):
        return None

    rep = await synthesize("x", "synth-test", propose=nothing, build_oracle=build_oracle, generalise=generalise, repair=repair, ctx=FalsifyContext(),
                           archive=Archive(tmp_path))
    assert rep.status == "dropped"
