"""crucible.synth.iface: PyWorld and the adapter over a Crucible World (part2-paper-core §3.1).

M1 acceptance for the adapter: finance_ap's oracle runs through ``CrucibleWorldAdapter`` (calls on the public policy
path, ``World.step``), passes the contract inferred from its run, agrees with the pack's own grader, and replays exactly.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.synth.contract import grade
from crucible.synth.falsify import REP, TAINT, FalsifyContext, falsify
from crucible.synth.iface import App, CrucibleWorldAdapter, Perturbable, Perturbation, PyWorld, SynthWorld, ToolError, UndeclaredWrite
from crucible.synth.infer import skeleton
from crucible.synth.program import Program, Step, lit, run
from crucible.synth.task import IntentCard, Task, WorldRef
from tests.unit import synth_fixtures as fx


async def test_pyworld_is_deterministic_and_describes_its_tools():
    w = fx.ap_world()
    assert isinstance(w, SynthWorld) and isinstance(w, Perturbable)
    a = await w.reset(3)
    s1 = await w.canon()
    await w.call("ap.approve_invoice", {"invoice_id": 101})
    assert (await w.reset(3)) == a and (await w.canon()).fingerprint() == s1.fingerprint()
    tools = {t.name: t for t in await w.tools()}
    hold = tools["ap.hold_invoice"]
    assert not hold.read_only and hold.writes == ["ap.invoices", "ap.audit_log"] and hold.input_schema["required"] == ["invoice_id", "reason"]
    assert hold.input_schema["properties"]["invoice_id"]["type"] == "integer" and "price_variance" in hold.description
    assert tools["ap.list_invoices"].read_only and tools["ap.list_invoices"].input_schema["properties"]["status"]["default"] == "pending"


async def test_pyworld_calls_are_atomic_and_errors_are_observations():
    w = fx.ap_world()
    await w.reset(0)
    before = (await w.canon()).fingerprint()
    assert (await w.call("ap.hold_invoice", {"invoice_id": 101, "reason": "bogus"})).error.startswith("ToolError")
    assert (await w.call("ap.approve_invoice", {"invoice_id": 107})).error  # not pending
    assert (await w.call("ap.nope", {})).error and (await w.call("ap.get_invoice", {"wrong": 1})).error.startswith("bad arguments")
    assert (await w.canon()).fingerprint() == before


async def test_pyworld_enforces_declared_writes():
    app = App("x", {"a": {1: {"v": 0}}, "b": {}})

    @app.tool(read_only=True)
    def sneaky(ctx) -> dict:
        ctx.tables["a"][1]["v"] = 9
        return {}

    @app.tool(writes=["b"])
    def wrong(ctx) -> dict:
        ctx.tables["a"][1]["v"] = 5
        return {}

    @app.tool(writes=["a"])
    def fails(ctx) -> dict:
        ctx.tables["a"][1]["v"] = 7
        raise ToolError("no")

    w = PyWorld([app])
    await w.reset(0)
    for t in ("x.sneaky", "x.wrong"):
        with pytest.raises(UndeclaredWrite):
            await w.call(t, {})
    assert (await w.call("x.fails", {})).error
    assert w.tables["x"]["a"][1]["v"] == 0  # every refused or failed call left no write
    with pytest.raises(ValueError):
        app.tool(lambda ctx: None)  # a write tool must declare what it writes


async def test_pyworld_snapshot_restore_and_perturb():
    w = fx.helpdesk_world(queue=True)
    await w.reset(0)
    ref = await w.snapshot()
    s0 = await w.canon()
    await w.call("desk.resolve_ticket", {"ticket_id": "T-1", "resolution": "KB-5"})
    await w.restore(ref)
    assert (await w.canon()).fingerprint() == s0.fingerprint()
    await w.perturb(Perturbation(op="clone", app="desk", coll="tickets", key="T-1", new_key="T-1-D", fields={"subject": "x"}, position=0))
    assert next(iter(w.tables["desk"]["tickets"])) == "T-1-D" and w.tables["desk"]["tickets"]["T-1-D"]["id"] == "T-1-D"
    await w.perturb(Perturbation(op="set", app="desk", coll="queue", key="0", fields={"state": "done"}))
    assert w.tables["desk"]["queue"]["q1"]["state"] == "done"  # a positional collection's key is the row position
    await w.perturb(Perturbation(op="shuffle", app="desk", coll="queue", seed=1))
    assert [r["ticket"] for r in w.tables["desk"]["queue"].values()] != ["T-1", "T-2", "T-3", "T-4", "T-5"]
    c = await w.canon()
    assert sorted(c.collection("desk", "queue")) == ["0", "1", "2", "3", "4"]


async def test_world_ref_builds_the_fixture_world():
    w = await WorldRef(factory=f"{fx.MOD}:helpdesk_world", params={"admin": True}).build()
    assert "desk.admin_update" in {t.name for t in await w.tools()}
    with pytest.raises(ValueError):
        await WorldRef(factory="no_colon").build()


# ---------------------------------------------------------------------------------------------------------- finance_ap

@pytest.fixture
async def ap_task():
    pytest.importorskip("crucible_pack_finance_ap")
    from crucible_pack_finance_ap.studio import APTask

    t = APTask(0)
    d = tempfile.TemporaryDirectory()
    w = await t.open(Path(d.name) / "work")
    try:
        yield t, w, CrucibleWorldAdapter(w, seed=0), Path(d.name)
    finally:
        await w.close()
        t.cleanup()
        d.cleanup()


def _oracle_program(t, *, list_first: bool = False) -> Program:
    steps = [Step(id=f"o{i}", tool=a["function_name"], args={k: lit(v) for k, v in a["arguments"].items()}) for i, a in enumerate(t.oracle)]
    if list_first:
        steps.insert(0, Step(id="lst", tool="erp.list_invoices", args={"status": lit("pending")}))
    return Program(steps=steps)


async def test_finance_ap_oracle_runs_through_the_adapter_and_passes_its_inferred_contract(ap_task):
    t, w, a, root = ap_task
    tools = {x.name: x for x in await a.tools()}
    assert tools["erp.get_invoice"].read_only and tools["kb.read_doc"].read_only and not tools["erp.hold_invoice"].read_only
    s0 = await a.canon()
    assert s0.unobserved == [] and set(s0.apps) == {"erp", "kb"} and "ap_policy.md" in s0.collection("kb", "files")
    pending = [k for k, r in s0.collection("erp", "invoices").items() if r["status"] == "pending"]
    assert sorted(int(k) for k in pending) == sorted(c.invoice_id for c in t.w.cases)  # rows keyed by the primary key
    r = await run(a, _oracle_program(t), t.instruction, seed=0)
    assert r.error is None and not any(c.obs.error for c in r.calls)
    sk = skeleton(r)
    assert grade(r, sk).passed
    assert (await t.grade(w, root)).reward == 1.0  # the pack's own grader agrees on the same end state
    # a wrong decision (approve an invoice a rule stops) fails the inferred contract with the right codes
    stopped = next(c for c in t.w.cases if c.expected[0] != "approved")
    bad = _oracle_program(t)
    for s in bad.steps:
        if s.tool in ("erp.hold_invoice", "erp.reject_invoice") and s.args["invoice_id"].v == stopped.invoice_id:
            s.tool, s.args = "erp.approve_invoice", {"invoice_id": lit(stopped.invoice_id)}
    v = grade(await run(a, bad, t.instruction, seed=0), sk)
    assert not v.passed and {"MISSING_REQUIRED", "FORBIDDEN_EFFECT"} <= v.codes
    assert f"FORBIDDEN_EFFECT:erp.invoices[{stopped.invoice_id}].status" in v.reasons


async def test_finance_ap_adapter_restore_reset_and_rep(ap_task):
    t, _w, a, _ = ap_task
    await a.reset(0)
    s0 = await a.canon()
    ref = await a.snapshot()
    await run(a, _oracle_program(t), t.instruction, seed=0, reset=False)
    assert (await a.canon()).fingerprint() != s0.fingerprint()
    await a.restore(ref)
    assert (await a.canon()).fingerprint() == s0.fingerprint()
    with pytest.raises(ValueError):
        await a.reset(1)  # another seed needs a factory
    r = await run(a, _oracle_program(t, list_first=True), t.instruction, seed=0)
    task = Task(id="finance_ap/s0", domain="finance-ap", world=WorldRef(factory="adapter"), instruction=t.instruction, contract=skeleton(r),
                oracle=_oracle_program(t, list_first=True), intent=IntentCard(skill="AP-3"))
    ctx = FalsifyContext(make_world=lambda _task: a)
    rep = await falsify(task, REP(), ctx)
    assert rep.n == 3 and rep.c == 0
    # TAINT: with the pending list read first, every literal of the pack's oracle has a source; without it, the
    # invoice ids are privileged knowledge (the pack oracle never lists the invoices it decides)
    assert (await falsify(task, TAINT(), ctx)).c == 0
    raw = task.model_copy(update={"oracle": _oracle_program(t)})
    hidden = (await falsify(raw, TAINT(), FalsifyContext(make_world=lambda _task: a))).confirmed
    assert hidden and {f.defect for f in hidden} == {"D5"} and {f.probe.data["value"] for f in hidden} <= {c.invoice_id for c in t.w.cases}
