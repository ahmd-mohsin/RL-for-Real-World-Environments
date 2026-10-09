"""World audit: a sound World passes every check that can run (and names the ones it cannot); each planted
defect is caught by the check meant for it."""

from __future__ import annotations

import textwrap
from pathlib import Path

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.audit import audit_world
from crucible.world.world import World

TOOLS = textwrap.dedent('''
    """Ticket tools."""
    import sqlite3, time
    from pathlib import Path

    def _con():
        c = sqlite3.connect(Path(__file__).resolve().parent / "tickets.db")  # MiMo convention: the DB sits next to the tools
        c.execute("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, body TEXT, at REAL)")
        return c

    def add_note(body: str) -> dict:
        """Add a note."""
        with _con() as c:
            c.execute("INSERT INTO notes (body, at) VALUES (?, ?)", (body, time.time()))
        return {"ok": True}

    def list_notes() -> list:
        """List notes."""
        with _con() as c:
            return [r[0] for r in c.execute("SELECT body FROM notes ORDER BY id")]
''')


def spec(tmp: Path, *, virtual_time: bool = True, instruction_leak: bool = False) -> WorldSpec:
    (tmp / "tools.py").write_text(TOOLS)
    return WorldSpec.model_validate({
        "world": "test/audit",
        "apps": [
            {"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}},
            {"id": "tix", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": str(tmp / "tickets.db")}},
             "params": {"tools_module": str(tmp / "tools.py"), "virtual_time": virtual_time, "env": {}}},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*", "tix.*"]}],
        "clock": {"mode": "step", "t0": "2026-03-02T09:00:00Z"},
    })


GOLD = [ToolCall(tool_call_id="g1", function_name="ws.write_file", arguments={"path": "answer.md", "content": "resolved"}),
        ToolCall(tool_call_id="g2", function_name="tix.add_note", arguments={"body": "customer refunded"})]


async def verifier(view) -> bool:
    r = await view.query("ws", Query(op="exists", args={"path": "answer.md"}))
    return bool(r.value and r.value.get("exists"))


async def test_sound_world_passes(tmp_path: Path):
    s = spec(tmp_path)
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.instructions = {"agent": "Resolve the ticket and write answer.md."}
    await w.start()
    try:
        rep = await audit_world(w, gold=GOLD, verifier=verifier, forbidden=["SECRET-ORACLE"])
    finally:
        await w.close()
    st = {c.id: c.status for c in rep.checks}
    assert rep.ok, rep.summary()
    assert st["episode_determinism"] == "pass" and st["oracle_passes"] == "pass" and st["floor_gap.nop_fails"] == "pass"
    assert st["runtime.host_containment"] == "skip"  # local world: named skip, never a silent pass


async def test_planted_defects_are_caught(tmp_path: Path):
    s = spec(tmp_path, virtual_time=False)  # the tool module reads the wall clock again
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    w.instructions = {"agent": "Fix the bug; see https://github.com/acme/app/pull/7 for the expected change."}
    await w.start()

    async def boom(**kw):
        raise RuntimeError("bug in the environment")

    w.apps["ws"].tool_list_dir = boom
    leak_gold = [*GOLD, ToolCall(tool_call_id="g3", function_name="ws.write_file", arguments={"path": "x.md", "content": "SECRET-ORACLE"}),
                 ToolCall(tool_call_id="g4", function_name="ws.read_file", arguments={"path": "x.md"})]
    try:
        rep = await audit_world(w, gold=leak_gold, forbidden=["SECRET-ORACLE"])
    finally:
        await w.close()
    st = {c.id: c.status for c in rep.checks}
    assert not rep.ok
    assert st["tool_smoke"] == "fail"  # list_dir crashes
    assert st["no_solution_leakage.static"] == "fail"  # PR link in the instruction
    assert st["no_solution_leakage.dynamic"] == "fail"  # the oracle string reached an observation
    assert st["episode_determinism"] == "fail"  # wall-clock timestamps differ between replays
    assert st["oracle_passes"] == "skip"
