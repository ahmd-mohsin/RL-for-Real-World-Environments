"""The M12 learning-evidence task (``runtime.WordFileTask``): seeded, gold solves it, the untouched task and a wrong
word do not, and held-out seeds (offset 10^6) differ from training seeds."""

from __future__ import annotations

import pytest
from crucible_pack_toy.runtime import WordFileTask

from crucible.actors.policy import ScriptedPolicy, run_episode
from crucible.core.schema.actions import ToolCall


@pytest.mark.parametrize("content,expected", [("gold", 1.0), (None, 0.0), ("wrong", 0.0)])
async def test_wordfile(tmp_path, content, expected):
    t = WordFileTask(7)
    word = {"gold": t.word + "\n", "wrong": "nope\n", None: None}[content]
    script = [ToolCall(tool_call_id="1", function_name="ws.write_file", arguments={"path": f"{t.name}.txt", "content": word})] if word else []
    w = await t.open(tmp_path / "w")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, reset=False)
        assert (await t.grade(w, tmp_path / "g")).reward == expected
    finally:
        await w.close()
    assert WordFileTask(7).instruction == t.instruction and WordFileTask(1_000_007).instruction != t.instruction
