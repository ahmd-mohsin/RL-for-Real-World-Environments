"""crucible.synth.worldgen: tool-first World synthesis with blind, frozen tests and sandboxed code (M4, F1).

Every model reply is scripted (fixtures for the framework test, not synthesized content).
"""

from __future__ import annotations

import json

import pytest

from crucible.synth.worldgen import open_world, worldgen
from tests.unit.synth_fakes import FakeClient, const, seq

P = {"name": "ticket_id", "type": "string", "description": "the ticket id"}
DESIGN = {"app": "desk", "description": "An IT helpdesk", "tools": [
    {"name": "list_tickets", "description": "Tickets with a status: open or closed.", "read_only": True,
     "params": [{"name": "status", "type": "string", "required": False, "enum": ["open", "closed"], "default": "open"}],
     "returns": {"tickets": [{"id": "T-1", "status": "open"}]}},
    {"name": "get_ticket", "description": "One ticket.", "read_only": True, "params": [P], "errors": ["unknown ticket"]},
    {"name": "list_notes", "description": "The notes of a ticket.", "read_only": True, "params": [P], "returns": {"notes": [{"id": 1, "text": "x"}]}},
    {"name": "close_ticket", "description": "Close an open ticket with a code: fixed or wontfix.", "writes": ["tickets"],
     "params": [P, {"name": "code", "type": "string", "enum": ["fixed", "wontfix"]}], "post": ["status = closed", "code set"]},
    {"name": "reopen_ticket", "description": "Reopen a closed ticket (status open, code cleared).", "writes": ["tickets"], "params": [P]},
    {"name": "add_note", "description": "Add a note to a ticket.", "writes": ["notes"], "params": [P, {"name": "text", "type": "string"}],
     "returns": {"ticket_id": "T-1", "id": 1}}]}
SCHEMA = {"tables": [{"name": "tickets", "key": "id", "fields": {"id": "string", "status": "string", "code": "string?"}},
                     {"name": "notes", "key": "id", "fields": {"id": "integer", "ticket_id": "string", "text": "string"}}]}

CODE = '''
def build(app):
    @app.tool(read_only=True)
    def list_tickets(ctx, status: str = "open") -> dict:
        """Tickets with a status."""
        return {"tickets": [dict(r) for r in ctx.tables["tickets"].values() if r["status"] == status]}

    @app.tool(read_only=True)
    def get_ticket(ctx, ticket_id: str) -> dict:
        """One ticket."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None:
            raise ToolError("unknown ticket")
        return dict(t)

    @app.tool(read_only=True)
    def list_notes(ctx, ticket_id: str) -> dict:
        """Notes."""
        return {"notes": [dict(n) for n in ctx.tables["notes"].values() if n["ticket_id"] == ticket_id]}

    @app.tool(writes=["tickets"])
    def close_ticket(ctx, ticket_id: str, code: str) -> dict:
        """Close."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None or t["status"] != "open" or code not in ("fixed", "wontfix"):
            raise ToolError("cannot close")
        t["status"], t["code"] = "closed", code
        return {"id": ticket_id, "status": "closed"}

    @app.tool(writes=["tickets"])
    def reopen_ticket(ctx, ticket_id: str) -> dict:
        """Reopen."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None or t["status"] != "closed":
            raise ToolError("cannot reopen")
        t["status"], t["code"] = "REOPEN_STATUS", None
        return {"id": ticket_id, "status": t["status"]}

    @app.tool(writes=["notes"])
    def add_note(ctx, ticket_id: str, text: str) -> dict:
        """Add a note."""
        if ticket_id not in ctx.tables["tickets"]:
            raise ToolError("unknown ticket")
        nid = max(ctx.tables["notes"], default=0) + 1
        ctx.tables["notes"][nid] = {"id": nid, "ticket_id": ticket_id, "text": text}
        return {"ticket_id": ticket_id, "id": nid}
'''
BUGGY, FIXED = CODE.replace("REOPEN_STATUS", "opened"), CODE.replace("REOPEN_STATUS", "open")
FIX = {"tickets": [{"id": "A", "status": "open", "code": None}, {"id": "B", "status": "closed", "code": "fixed"}]}
TESTS = {"tests": [
    {"id": "list", "fixture": FIX, "calls": [{"tool": "list_tickets", "args": {"status": "open"}, "result": {"tickets": [{"id": "A"}]}}]},
    {"id": "get", "fixture": FIX, "calls": [{"tool": "get_ticket", "args": {"ticket_id": "A"}, "result": {"status": "open"}}]},
    {"id": "get_missing", "fixture": FIX, "calls": [{"tool": "get_ticket", "args": {"ticket_id": "Z"}, "expect": "error"}]},
    {"id": "close", "fixture": FIX, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "fixed"}}],
     "expect_state": {"tickets": {"A": {"status": "closed", "code": "fixed"}}}},
    {"id": "close_closed", "fixture": FIX, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "B", "code": "fixed"}, "expect": "error"}]},
    {"id": "close_badcode", "fixture": FIX, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "bogus"}, "expect": "error"}]},
    {"id": "reopen", "fixture": FIX, "calls": [{"tool": "reopen_ticket", "args": {"ticket_id": "B"}}], "expect_state": {"tickets": {"B": {"status": "open"}}}},
    {"id": "reopen_open", "fixture": FIX, "calls": [{"tool": "reopen_ticket", "args": {"ticket_id": "A"}, "expect": "error"}]},
    {"id": "note", "fixture": FIX, "calls": [{"tool": "add_note", "args": {"ticket_id": "A", "text": "hi"}, "result": {"ticket_id": "A"}},
                                             {"tool": "list_notes", "args": {"ticket_id": "A"}, "result": {"notes": [{"text": "hi"}]}}]},
    {"id": "note_missing", "fixture": FIX, "calls": [{"tool": "add_note", "args": {"ticket_id": "Z", "text": "x"}, "expect": "error"}]},
    {"id": "wrong", "fixture": FIX, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "duplicate"}}]},  # contradicts the enum
]}
SEED = {"tables": {"tickets": [{"id": f"T-{i}", "status": "open" if i % 3 else "closed", "code": None if i % 3 else "fixed"} for i in range(1, 9)],
                   "notes": [{"id": 1, "ticket_id": "T-1", "text": "printer jam"}]}}


def clients(*, debugger=None, approver=None, coder=None):
    return {"tool_designer": FakeClient("kimi", const(DESIGN)), "schema_deriver": FakeClient("kimi", const(SCHEMA)),
            "app_coder": coder or FakeClient("kimi", const({"code": BUGGY})), "blind_tester": FakeClient("deepseek", const(TESTS)),
            "debugger": debugger or FakeClient("kimi", const({"code": FIXED, "disputes": [{"test": "wrong", "why": "the code enum is fixed or wontfix"}]})),
            "test_approver": approver or FakeClient("nemotron", const({"test_is_wrong": True, "why": "the spec lists fixed and wontfix only"})),
            "seeder": FakeClient("kimi", const(SEED))}


async def test_worldgen_accepts_after_code_only_repair_and_an_approved_dispute(tmp_path):
    cl = clients()
    rep = await worldgen("IT helpdesk", tmp_path / "w", cl)
    assert rep.status == "accepted", rep.reasons
    assert rep.debug_rounds == 1 and rep.tests_dropped == ["wrong"] and rep.pass_history[-1] == 1.0 and rep.pass_history[0] < 1.0
    assert "code" not in cl["blind_tester"].calls[0][1]["content"].split("## design")[0]  # the tester never sees the code
    assert "## code" not in cl["blind_tester"].calls[0][1]["content"] and "## code" not in cl["test_approver"].calls[0][1]["content"]
    man = json.loads((tmp_path / "w" / "world.json").read_text())
    assert man["tests_digest"] == rep.tests_digest and man["dropped"] == ["wrong"]
    tests_now = json.loads((tmp_path / "w" / "tests.json").read_text())
    assert [t["id"] for t in tests_now["tests"]] == [t["id"] for t in TESTS["tests"]]  # dropped, never edited
    w = open_world(str(tmp_path / "w"), str(tmp_path / "work"))
    try:
        await w.reset(0)
        assert len((await w.canon()).collection("desk", "tickets")) == 8
        assert (await w.call("desk.reopen_ticket", {"ticket_id": "T-3"})).value["status"] == "open"
    finally:
        await w.close()


async def test_a_refused_dispute_keeps_the_test_and_the_world_is_rejected(tmp_path):
    no = FakeClient("nemotron", const({"test_is_wrong": False, "why": "keep it"}))
    rep = await worldgen("IT helpdesk", tmp_path / "w", clients(approver=no), max_debug_rounds=2)
    assert rep.status == "rejected:tests" and rep.disputes_refused and "wrong" in rep.reasons[0]
    assert not (tmp_path / "w" / "world.json").exists()


async def test_the_frozen_tests_cannot_change_under_the_debugger(tmp_path):
    out = tmp_path / "w"

    def tamper(_m, _n):
        p = out / "tests.json"
        doc = json.loads(p.read_text())
        doc["tests"] = [t for t in doc["tests"] if t["id"] != "reopen"]
        p.write_text(json.dumps(doc))
        return {"code": FIXED}

    with pytest.raises(RuntimeError, match="frozen"):
        await worldgen("IT helpdesk", out, clients(debugger=FakeClient("kimi", tamper)))


async def test_code_failing_the_static_check_goes_back_to_the_coder(tmp_path):
    coder = FakeClient("kimi", seq({"code": "import os\n" + FIXED}, {"code": FIXED}))
    rep = await worldgen("IT helpdesk", tmp_path / "w", clients(coder=coder))
    assert rep.status == "accepted" and len(coder.calls) == 2 and "static check" in coder.calls[1][-1]["content"]


async def test_the_seed_density_knob_and_reseed(tmp_path):
    """§6 / §14.1: denser seeded Worlds. ``rows`` reaches the seeder; ``reseed`` adds swarm-written sibling rows to an
    accepted World, table by table, never changing an existing row, and re-runs the smoke test."""
    from crucible.synth.worldgen import reseed
    from tests.unit.synth_fakes import section

    cl = clients()
    rep = await worldgen("IT helpdesk", tmp_path / "w", cl, rows=(30, 40))
    assert rep.status == "accepted" and section(cl["seeder"].calls[0], "rows") == "30 to 40 rows per table"

    def add(msgs, n):
        have = section(msgs, "existing")
        if n == 0:  # first a reply that reuses an existing key: refused with feedback
            return {"tables": {"tickets": [have["tickets"][0]]}}
        return {"tables": {"tickets": [{"id": f"T-{i}", "status": "open", "code": None} for i in range(100, 130)]}}

    seeder = FakeClient("kimi", add)
    rr = await reseed(tmp_path / "w", tmp_path / "dense", seeder, target={"tickets": 38, "nope": 5})
    assert rr.status == "accepted" and rr.before["tickets"] == 8 and rr.after["tickets"] == 38 and rr.added == {"tickets": 30}
    assert "nope" in rr.refused and "already used" in seeder.calls[1][-1]["content"]
    assert "add exactly 30 new rows to table tickets" in section(seeder.calls[0], "rows")
    old = json.loads((tmp_path / "w" / "seed.json").read_text())["tables"]["tickets"]
    new = json.loads((tmp_path / "dense" / "seed.json").read_text())["tables"]["tickets"]
    assert new[: len(old)] == old  # existing rows unchanged
    assert (tmp_path / "dense" / "app.py").read_bytes() == (tmp_path / "w" / "app.py").read_bytes()
    w = open_world(str(tmp_path / "dense"), str(tmp_path / "work2"))
    try:
        await w.reset(0)
        assert len((await w.canon()).collection("desk", "tickets")) == 38
    finally:
        await w.close()


def test_conformance_refuses_parameters_the_design_lacks_and_tests_may_not_invent_arguments():
    # World-families pilot 1: a hidden acting_user_id parameter on every write tool (impersonation), passed by the tests
    from crucible.synth.roles import SchemaOut, ToolDesignOut, ToolTestsOut
    from crucible.synth.worldgen import check_tests, conformance

    d = ToolDesignOut.model_validate(DESIGN)
    info = {"tools": [{"name": t["name"], "read_only": t.get("read_only", False), "writes": t.get("writes", []),
                       "params": [p["name"] for p in t["params"]] + (["acting_user_id"] if t["name"] == "close_ticket" else [])} for t in DESIGN["tools"]]}
    assert any("parameters not in the design ['acting_user_id']" in p for p in conformance(info, d))
    bad = json.loads(json.dumps(TESTS))
    bad["tests"][3]["calls"][0]["args"]["acting_user_id"] = "U-1"
    with pytest.raises(ValueError, match="has no parameters"):
        check_tests(ToolTestsOut.model_validate(bad), d, SchemaOut.model_validate(SCHEMA))
