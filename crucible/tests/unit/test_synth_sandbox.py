"""crucible.synth.sandbox: model-written App code runs only in a restricted child process (M4, F1).

The App modules here are test fixtures (framework tests), not synthesized content.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from crucible.synth.falsify import NULL, REP, FalsifyContext, falsify
from crucible.synth.iface import Perturbation, UndeclaredWrite
from crucible.synth.sandbox import CodeRejected, SandboxError, SandboxWorld, _env, check_code, run_tool_tests, static_info
from crucible.synth.task import Task, WorldRef

DESIGN = {"app": "desk", "description": "tickets", "tools": [
    {"name": "list_tickets", "description": "Tickets with a status (open or closed).", "read_only": True,
     "params": [{"name": "status", "type": "string", "required": False, "enum": ["open", "closed"]}]},
    {"name": "close_ticket", "description": "Close an open ticket with a resolution code: fixed or wontfix.", "read_only": False, "writes": ["tickets"],
     "params": [{"name": "ticket_id", "type": "string"}, {"name": "code", "type": "string", "enum": ["fixed", "wontfix"]}]},
    {"name": "spin", "description": "A tool that never returns.", "read_only": True, "params": []},
    {"name": "sneak", "description": "Writes what it does not declare.", "read_only": False, "writes": ["notes"], "params": []},
    {"name": "boom", "description": "Crashes.", "read_only": True, "params": []}]}
SCHEMA = {"tables": [{"name": "tickets", "key": "id", "fields": {"id": "string", "status": "string", "code": "string?", "at": "string?"}},
                     {"name": "notes", "key": "id", "fields": {"id": "integer", "text": "string"}}]}
SEED = {"tables": {"tickets": [{"id": "T-1", "status": "open", "code": None, "at": None}, {"id": "T-2", "status": "open", "code": None, "at": None},
                               {"id": "T-3", "status": "closed", "code": "fixed", "at": None}], "notes": []}}
CODE = '''
import math

def build(app):
    @app.tool(read_only=True)
    def list_tickets(ctx, status: str = "open") -> dict:
        """Tickets with a status."""
        return {"tickets": [dict(r) for r in ctx.tables["tickets"].values() if r["status"] == status]}

    @app.tool(writes=["tickets"])
    def close_ticket(ctx, ticket_id: str, code: str) -> dict:
        """Close an open ticket."""
        t = ctx.tables["tickets"].get(ticket_id)
        if t is None or t["status"] != "open":
            raise ToolError(f"ticket {ticket_id} is not open")
        if code not in ("fixed", "wontfix"):
            raise ToolError("unknown code")
        t["status"], t["code"], t["at"] = "closed", code, ctx.now()
        return {"id": ticket_id, "status": "closed"}

    @app.tool(read_only=True)
    def spin(ctx) -> dict:
        """Never returns."""
        while True:
            pass

    @app.tool(writes=["notes"])
    def sneak(ctx) -> dict:
        """Undeclared write."""
        ctx.tables["tickets"]["T-1"]["status"] = "gone"
        return {}

    @app.tool(read_only=True)
    def boom(ctx) -> dict:
        """Crashes."""
        return {"x": math.floor(1 / 0)}
'''


def make_app(root: Path, code: str = CODE) -> Path:
    d = root / "app"
    d.mkdir(parents=True, exist_ok=True)
    (d / "app.py").write_text(code)
    for name, obj in (("design.json", DESIGN), ("schema.json", SCHEMA), ("seed.json", SEED)):
        (d / name).write_text(json.dumps(obj))
    return d


@pytest.mark.parametrize("bad, why", [
    ("import os\ndef build(app):\n    pass\n", "import os"),
    ("def build(app):\n    open('x')\n", "open"),
    ("def build(app):\n    return app.__class__\n", "dunder"),
    ("def build(app):\n    eval('1')\n", "eval"),
    ("import socket\ndef build(app):\n    pass\n", "import socket"),
    ("from .x import y\ndef build(app):\n    pass\n", "from"),
    ("def nobuild(app):\n    pass\n", "build"),
])
def test_static_check_refuses_dangerous_code(bad, why):
    probs = check_code(bad)
    assert probs and any(why.split()[0] in p for p in probs)
    assert check_code(CODE) == []


def test_child_environment_is_scrubbed():
    env = _env()
    assert set(env) == {"PATH", "LANG", "PYTHONHASHSEED"} and not any("KEY" in k or "TOKEN" in k for k in env)


async def test_sandbox_world_serves_the_app(tmp_path):
    d = make_app(tmp_path)
    w = SandboxWorld(d, tmp_path / "work", timeout_s=5.0)
    try:
        fp = await w.reset(0)
        tools = {t.name: t for t in await w.tools()}
        assert tools["desk.list_tickets"].read_only and tools["desk.close_ticket"].writes == ["desk.tickets"]
        assert tools["desk.close_ticket"].input_schema["properties"]["code"]["enum"] == ["fixed", "wontfix"]  # the design's catalogue
        s0 = await w.canon()
        assert set(s0.collection("desk", "tickets")) == {"T-1", "T-2", "T-3"}
        ref = await w.snapshot()
        o = await w.call("desk.close_ticket", {"ticket_id": "T-1", "code": "fixed"})
        assert o.ok and (await w.canon()).collection("desk", "tickets")["T-1"]["status"] == "closed"
        assert (await w.call("desk.close_ticket", {"ticket_id": "T-3", "code": "fixed"})).error.startswith("ToolError")
        assert "ZeroDivisionError" in (await w.call("desk.boom", {})).error  # a crash is an observation
        with pytest.raises(UndeclaredWrite):
            await w.call("desk.sneak", {})
        await w.restore(ref)
        assert (await w.canon()).fingerprint() == s0.fingerprint()
        await w.perturb(Perturbation(op="clone", app="desk", coll="tickets", key="T-2", new_key="T-9", fields={"status": "open"}))
        assert "T-9" in (await w.canon()).collection("desk", "tickets")
        assert await w.reset(0) == fp
        with pytest.raises(SandboxError):  # a tool that never returns: the child is killed
            await w.call("desk.spin", {})
        assert await w.reset(0) == fp  # the next reset starts a fresh child
    finally:
        await w.close()


async def test_sandbox_cannot_write_outside_its_work_dir_and_has_no_key(tmp_path, monkeypatch):
    monkeypatch.setenv("TINKER_API_KEY", "sk-not-a-real-key")
    code = CODE + '''
    @app.tool(read_only=True)
    def env(ctx) -> dict:
        """What the child can see."""
        import json
        return {"ok": True}
'''
    d = make_app(tmp_path, code)
    w = SandboxWorld(d, tmp_path / "work", timeout_s=5.0)
    try:
        await w.reset(0)
        assert (await w.call("desk.env", {})).ok
    finally:
        await w.close()
    assert os.environ["TINKER_API_KEY"] == "sk-not-a-real-key"  # the parent's environment is untouched; the child got _env() only


async def test_refuses_code_that_fails_the_static_check(tmp_path):
    d = make_app(tmp_path, "import os\n" + CODE)
    with pytest.raises(CodeRejected):
        SandboxWorld(d, tmp_path / "work")
    res = await run_tool_tests(d, tmp_path / "w2", [{"id": "t1", "calls": []}])
    assert res[0]["passed"] is False and "code rejected" in res[0]["detail"]


async def test_tool_tests_run_in_the_sandbox(tmp_path):
    d = make_app(tmp_path)
    tests = [
        {"id": "ok", "fixture": {"tickets": [{"id": "A", "status": "open"}]}, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "fixed"},
                                                                                        "result": {"status": "closed"}}],
         "expect_state": {"tickets": {"A": {"status": "closed", "code": "fixed"}}}},
        {"id": "refuse", "fixture": {"tickets": [{"id": "A", "status": "closed"}]}, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "fixed"},
                                                                                             "expect": "error"}]},
        {"id": "wrong", "fixture": {"tickets": [{"id": "A", "status": "open"}]}, "calls": [{"tool": "close_ticket", "args": {"ticket_id": "A", "code": "fixed"}}],
         "expect_state": {"tickets": {"A": {"code": "wontfix"}}}},
        {"id": "crash", "calls": [{"tool": "boom", "args": {}, "expect": "error"}]},
    ]
    res = {r["id"]: r for r in await run_tool_tests(d, tmp_path / "work", tests)}
    assert res["ok"]["passed"] and res["refuse"]["passed"]
    assert not res["wrong"]["passed"] and "code" in res["wrong"]["detail"]
    assert not res["crash"]["passed"] and "crashed instead of refusing" in res["crash"]["detail"] and "ZeroDivisionError" in res["crash"]["detail"]
    info = await static_info(d, tmp_path / "work2")
    assert {t["name"] for t in info["tools"]} == {"list_tickets", "close_ticket", "spin", "sneak", "boom"}


def sandbox_world(dir: str, work: str) -> SandboxWorld:
    return SandboxWorld(dir, work, timeout_s=5.0)


async def test_falsifiers_run_on_a_sandboxed_world(tmp_path):
    from crucible.synth.contract import Contract, EffectPattern, PEq
    from crucible.synth.program import Program, Step, lit, ref

    spin = CODE[CODE.index("    @app.tool(read_only=True)\n    def spin"):CODE.index("    @app.tool(writes=[\"notes\"])")]
    d = make_app(tmp_path, CODE.replace(spin, ""))  # no never-returning tool: NULL's read-all would wait out every timeout
    task = Task(id="sbx/close-T1", domain="synth-test", world=WorldRef(factory="tests.unit.test_synth_sandbox:sandbox_world",
                                                                    params={"dir": str(d), "work": str(tmp_path / "work")}),
                instruction="Close ticket T-1 as fixed.",
                oracle=Program(steps=[Step(id="s1", tool="desk.list_tickets", args={"status": lit("open")}),
                                      Step(id="s2", tool="desk.close_ticket", args={"ticket_id": ref("s1.tickets.0.id"), "code": lit("fixed")})]),
                contract=Contract(required=[EffectPattern(id="r_s", app="desk", coll="tickets", key="T-1", field="status", value=PEq(v="closed")),
                                            EffectPattern(id="r_c", app="desk", coll="tickets", key="T-1", field="code", value=PEq(v="fixed")),
                                            EffectPattern(id="r_at", app="desk", coll="tickets", key="T-1", field="at", value=PEq(v="2026-01-01T00:00:02Z"))]))
    ctx = FalsifyContext()
    try:
        rep = await falsify(task, REP(), ctx)
        assert rep.n == 3 and rep.c == 0
        nul = await falsify(task, NULL(), ctx)
        assert nul.n > 0 and nul.c == 0
    finally:
        await ctx.close()
