"""Agent-caused tool errors carry the policy fault and say how to recover (mvdan-sh dc951aeb-e1, 2026-10-26: 13-14 failed
calls per budget-stopped episode, ``old_string not found`` and a bad regex labelled ``fault: tool``, unknown tool names
answered with no list of the valid ones)."""

from __future__ import annotations

from pathlib import Path

from crucible.core.schema.actions import ToolCall
from crucible.core.store import Store
from crucible.factory.lanes.windows_world import world_spec
from crucible.world.world import World


async def _call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=tool, function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_agent_caused_errors_are_the_policys_and_point_at_the_fix(tmp_path: Path) -> None:
    base = tmp_path / "base"
    base.mkdir()
    (base / "printer.go").write_text("package syntax\n\nfunc (p *Printer) multiLine() bool {\n\treturn p.line > 0\n}\n")
    ws = tmp_path / "ws"
    ws.mkdir()
    w = World(world_spec(ws, base), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        r = await _call(w, "repo.edit_file", path="printer.go", old_string="func (p *Printer) multiline() bool {\n", new_string="x")
        assert not r.ok and r.error == "tool_error" and r.fault == "policy"
        assert "old_string not found" in r.content and "closest lines" in r.content and "line 3:" in r.content  # numbered, near-identical line
        assert "return p.line" not in (ws / "printer.go").read_text() or "multiLine" in (ws / "printer.go").read_text()  # nothing applied
        r = await _call(w, "repo.search", pattern="semiRsrv\"[\\(\\]")
        assert not r.ok and r.fault == "policy" and "search failed" in r.content
        r = await _call(w, "shell", command="ls")
        assert not r.ok and r.error == "bad_tool_name" and r.fault == "policy"
        assert "Did you mean repo.shell?" in r.content and "Valid tools:" in r.content and "repo.edit_file" in r.content
        r = await _call(w, "execcmd", command="ls")
        assert "Valid tools:" in r.content and not r.ok  # a hint only: nothing ran
    finally:
        await w.close()
