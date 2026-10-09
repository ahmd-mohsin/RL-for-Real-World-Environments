"""M1 World runtime on the local backend: tools, ACL, clock, events, responders, snapshot/restore, reset."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from crucible.core.schema.actions import Done, Message, ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import ActorTurn, World, WorldError

TOOLS = textwrap.dedent(
    '''
    """Ticketing system tools (MiMo style: plain functions, db relative to this file)."""
    import sqlite3
    from pathlib import Path

    _DB = Path(__file__).resolve().parent.parent / "db" / "tickets.db"

    def _con():
        c = sqlite3.connect(_DB); c.row_factory = sqlite3.Row; return c

    def list_tickets(status: str = "open") -> list:
        """List tickets by status."""
        with _con() as c:
            return [dict(r) for r in c.execute("SELECT id, subject, status FROM tickets WHERE status = ? ORDER BY id", (status,))]

    def set_status(ticket_id: int, status: str) -> dict:
        """Set a ticket's status."""
        if status not in ("open", "pending", "resolved"):
            raise ValueError("bad status")
        with _con() as c:
            n = c.execute("UPDATE tickets SET status = ? WHERE id = ?", (status, ticket_id)).rowcount
        if not n:
            raise ValueError(f"no ticket {ticket_id}")
        return {"id": ticket_id, "status": status}

    def refund(ticket_id: int, amount: float) -> dict:
        """Issue a refund."""
        with _con() as c:
            c.execute("INSERT INTO refunds(ticket_id, amount) VALUES (?, ?)", (ticket_id, amount))
        return {"ok": True}
    '''
)

SCHEMA = """
CREATE TABLE tickets(id INTEGER PRIMARY KEY, subject TEXT, status TEXT);
CREATE TABLE refunds(id INTEGER PRIMARY KEY, ticket_id INTEGER, amount REAL);
INSERT INTO tickets(subject, status) VALUES ('printer broken', 'open'), ('refund please', 'open');
"""


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    root = tmp_path / "pack"
    (root / "kb").mkdir(parents=True)
    (root / "kb" / "refund_policy.md").write_text("Refunds above 50 USD need supervisor approval.\n")
    (root / "schema.sql").write_text(SCHEMA)
    (root / "tools.py").write_text(TOOLS)
    return root


def spec(transport: str = "inprocess") -> WorldSpec:
    return WorldSpec.model_validate(
        {
            "world": "test/helpdesk",
            "apps": [
                {"id": "tickets", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "helpdesk/db/tickets.db"}},
                 "params": {"schema_sql": "schema.sql", "tools_module": "helpdesk/tools/tickets.py", "transport": transport}},
                {"id": "kb", "template": "common/doc_library", "state": {"adapter": "fs", "params": {"root": "kb"}}, "params": {"seed_dir": "kb"}},
                {"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}},
            ],
            "actors": [
                {"id": "agent", "kind": "policy", "tools": ["tickets.*", "kb.*", "mail.*"]},
                {"id": "supervisor", "kind": "simulated", "tools": ["tickets.list_tickets"]},
                {"id": "ops", "kind": "system"},
            ],
            "clock": {"mode": "step", "t0": "2026-03-02T09:00:00Z", "step_dt_s": 600},
            "events": [
                {"id": "sla", "when": {"after": "30m"}, "actor": "ops", "action": {"type": "nl", "text": "SLA warning: ticket 1 is 30 minutes old", "to": "agent"}},
                {"id": "freeze", "when": {"after": "1h"}, "actor": "ops", "action": {"type": "permission", "to": "agent", "revoke": ["tickets.refund"]}},
                {"id": "audit", "when": {"on_tool": "tickets.refund"}, "actor": "ops", "action": {"type": "tool", "tool": "mail.send_email",
                 "args": {"to": ["agent@world.local"], "subject": "refund audit", "body": "a refund was issued"}}},
            ],
        }
    )


class Supervisor:
    actor_id = "supervisor"

    def __init__(self):
        self.seen = 0

    async def respond(self, turn: ActorTurn):
        self.seen += len(turn.inbox)
        if not turn.tool_results:
            return [ToolCall(tool_call_id="s1", function_name="tickets.list_tickets", arguments={"status": "open"})]
        n = turn.tool_results[0].content.count('"id"')
        return [Message(to="agent", text=f"approved; {n} open tickets remain")]

    def state_dict(self):
        return {"seen": self.seen}

    def load_state_dict(self, s):
        self.seen = s["seen"]


def _materialize_tools(world: World, pack: Path) -> None:
    # tools file must sit in <app dir>/tools/ next to <app dir>/db/ (MiMo layout)
    d = world.workdir / "apps" / "helpdesk" / "tools"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tickets.py").write_text((pack / "tools.py").read_text())


async def _world(pack: Path, tmp_path: Path, transport: str = "inprocess") -> World:
    w = World(spec(transport), instructions={"agent": "Resolve open tickets."}, store=Store(tmp_path / "store"), pack_root=pack, workdir=tmp_path / f"w-{transport}")
    _materialize_tools(w, pack)
    w.attach(Supervisor())
    await w.start()
    return w


@pytest.mark.parametrize("transport", ["inprocess", "mcp"])
async def test_world_episode_acl_events_snapshot(pack, tmp_path, transport):
    w = await _world(pack, tmp_path, transport)
    try:
        obs = await w.reset()
        o = obs["agent"]
        assert o.instruction == "Resolve open tickets." and o.time == "2026-03-02T09:00:00Z"
        names = {t.qualified for t in o.tools}
        assert {"tickets.list_tickets", "tickets.refund", "kb.read_doc", "mail.send_email"} <= names
        r = await w.step("agent", ToolCall(tool_call_id="1", function_name="tickets__list_tickets", arguments={}))
        assert r.observation.tool_results[0].ok and "printer broken" in r.observation.tool_results[0].content
        r = await w.step("agent", ToolCall(tool_call_id="2", function_name="kb.read_doc", arguments={"path": "refund_policy.md"}))
        assert "supervisor approval" in r.observation.tool_results[0].content
        # 3rd step crosses 30 min -> SLA nl event lands in the inbox
        r = await w.step("agent", Message(to="supervisor", text="May I refund ticket 2 (80 USD)?"))
        texts = [m.text for m in r.observation.inbox]
        assert any("approved; 2 open tickets" in t for t in texts) and any("SLA warning" in t for t in texts)
        mid = await w.snapshot()
        r = await w.step("agent", ToolCall(tool_call_id="3", function_name="tickets.refund", arguments={"ticket_id": 2, "amount": 80}))
        assert r.observation.tool_results[0].ok
        r = await w.step("agent", ToolCall(tool_call_id="4", function_name="mail.list_inbox", arguments={}))
        assert "refund audit" in r.observation.tool_results[0].content  # on_tool event fired a system tool call
        # past 1h: permission revoked; tools list re-sent
        r = await w.step("agent", ToolCall(tool_call_id="5", function_name="tickets.set_status", arguments={"ticket_id": 2, "status": "resolved"}))
        assert r.observation.tools and "tickets.refund" not in {t.qualified for t in r.observation.tools}
        r = await w.step("agent", ToolCall(tool_call_id="6", function_name="tickets.refund", arguments={"ticket_id": 1, "amount": 5}))
        assert r.observation.tool_results[0].error == "permission_denied"
        r = await w.step("agent", ToolCall(tool_call_id="7", function_name="tickets.set_status", arguments={"ticket_id": 99, "status": "resolved"}))
        assert not r.observation.tool_results[0].ok and "no ticket 99" in r.observation.tool_results[0].content
        end = await w.snapshot()
        assert end.fingerprint != mid.fingerprint
        view = w.verifier_view()
        assert (await view.query("tickets", {"op": "scalar", "args": {"sql": "SELECT COUNT(*) FROM refunds"}})).value == 1
        # restore the mid-episode state: db, mailbox, clock, ACL, events, responder memory all rewind
        await w.restore(mid)
        again = await w.snapshot()
        assert again.fingerprint == mid.fingerprint
        assert (await view.query("tickets", {"op": "scalar", "args": {"sql": "SELECT COUNT(*) FROM refunds"}})).value == 0
        assert w.acl.allows("agent", "tickets.refund") and w.clock.now_iso() == mid.clock.now
        assert w.responders["supervisor"].state_dict() == mid.actors["supervisor"]
        # replaying the same actions from `mid` reproduces `end` exactly (determinism)
        for call in [ToolCall(tool_call_id="3", function_name="tickets.refund", arguments={"ticket_id": 2, "amount": 80}),
                     ToolCall(tool_call_id="4", function_name="mail.list_inbox", arguments={}),
                     ToolCall(tool_call_id="5", function_name="tickets.set_status", arguments={"ticket_id": 2, "status": "resolved"}),
                     ToolCall(tool_call_id="6", function_name="tickets.refund", arguments={"ticket_id": 1, "amount": 5}),
                     ToolCall(tool_call_id="7", function_name="tickets.set_status", arguments={"ticket_id": 99, "status": "resolved"})]:
            await w.step("agent", call)
        assert (await w.snapshot()).fingerprint == end.fingerprint
        res = await w.step("agent", Done(reason="all resolved"))
        assert res.done
        trace = w.to_trace()
        assert any(e.type == "event:permission" for e in trace.events) and any(e.actor == "supervisor" and e.type == "message" for e in trace.events)
        # reset returns to the initial state
        await w.reset()
        assert (await w.snapshot()).fingerprint == w.initial.fingerprint
    finally:
        await w.close()


async def test_restore_into_fresh_world(pack, tmp_path):
    """Snapshot in one world, restore into a freshly started one: identical fingerprint (M1 gate shape)."""
    w1 = await _world(pack, tmp_path)
    try:
        await w1.reset()
        await w1.step("agent", ToolCall(tool_call_id="1", function_name="tickets.set_status", arguments={"ticket_id": 1, "status": "pending"}))
        await w1.step("agent", ToolCall(tool_call_id="2", function_name="mail.send_email", arguments={"to": ["c@x"], "subject": "hi", "body": "b"}))
        ref = await w1.snapshot()
    finally:
        await w1.close()
    w2 = World(spec(), store=Store(tmp_path / "store"), pack_root=pack, workdir=tmp_path / "w2")
    _materialize_tools(w2, pack)
    w2.attach(Supervisor())
    await w2.start()
    try:
        assert (await w2.snapshot()).fingerprint != ref.fingerprint
        await w2.restore(ref)
        assert (await w2.snapshot()).fingerprint == ref.fingerprint
    finally:
        await w2.close()


async def test_guards(pack, tmp_path):
    w = await _world(pack, tmp_path)
    try:
        await w.reset()
        with pytest.raises(WorldError, match="not a policy"):
            await w.step("supervisor", Done())
        r = await w.step("agent", ToolCall(tool_call_id="x", function_name="nope.tool", arguments={}))
        assert r.observation.tool_results[0].error == "unknown_app"
        r = await w.step("agent", ToolCall(tool_call_id="y", function_name="kb.read_doc", arguments={"path": "../../etc/passwd"}))
        assert "outside the workspace" in r.observation.tool_results[0].content
        r = await w.step("agent", Message(to="ghost", text="hello"))
        assert "No such recipient" in r.observation.inbox[0].text
        other = WorldSpec.model_validate({**spec().model_dump(), "world": "test/other"})
        w_other = World(other, store=w.store, workdir=tmp_path / "other")
        with pytest.raises(WorldError, match="belongs to world"):
            await w_other.restore(w.initial)
    finally:
        await w.close()


async def test_failed_start_tears_down(tmp_path):
    """A World whose App fails during start is closed: nothing keeps running."""
    from crucible.core.schema.world import WorldSpec as WS

    spec_bad = WS.model_validate({"world": "test/bad", "apps": [
        {"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "x.db"}}, "params": {"tools_module": "missing.py"}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}]})
    w = World(spec_bad, store=Store(tmp_path / "s"))
    with pytest.raises(FileNotFoundError, match="missing.py"):
        await w.start()
    assert not w.workdir.exists()  # close() ran and removed the world's scratch dir


async def test_subprocesses_never_see_host_credentials(monkeypatch):
    from crucible.world.exec import LocalExec

    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-secret")
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    monkeypatch.setenv("HARMLESS_SETTING", "visible")
    out = (await LocalExec().run("env")).stdout
    assert "sk-or-secret" not in out and "hf_secret" not in out and "HARMLESS_SETTING=visible" in out
    assert "DATABASE_URL=postgres://x" in (await LocalExec().run("env", env={"DATABASE_URL": "postgres://x"})).stdout
