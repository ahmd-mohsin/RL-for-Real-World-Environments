"""Part 1 A1 (2026-10-29, docs/research/env-synthesis/part1-framework-improvements.md §1.1): ``WorkspaceHarness`` (the
student path's harness, ``factory.envs.make_env``) overrides ``NativeHarness.run`` and never ran the level-1 hack rules
or counted malformed tool calls, so a workspace rollout carried only the grade's flags and every ``Turn.format_errors``
stayed 0. Both are recorded now, as the native harness records them; the reward is unchanged (record only)."""

from __future__ import annotations

import json

from crucible_pack_toy.runtime import FibTask

from crucible.actors.policy import LLMToolPolicy
from crucible.core.schema.rollout import RolloutRecord
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.inject import FakeLLM, tool_reply
from crucible.runtime.run import Runner, RunSpec

FIB = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"


async def test_a_workspace_rollout_records_level1_flags_and_format_errors(tmp_path):
    acts = [tool_reply("ws__write_file", {"path": "fib.py", "content": FIB}, 1),
            tool_reply("ws__write_file", {"path": "tests/test_fib.py", "content": "def test_ok():\n    pass\n"}, 2),
            tool_reply("ws__no_such_tool", {}, 3), "done"]
    pol = PolicyRef(id="fake:fib", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(list(acts)))}, config={"fib": "a1"})
    await Runner(tmp_path).run(RunSpec(run_id="a1", tasks=[FibTask()], policies=[pol], harnesses=[WorkspaceHarness()], k=1))
    adir = min((tmp_path / "a1" / "samples").glob("*/a*"))
    rec = json.loads((adir / "record.json").read_text())
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    assert [(f["rule"], f["evidence"]) for f in ro.flags] == [("writes_test_file", "tests/test_fib.py")]  # before: []
    assert sorted(t.format_errors for t in ro.turns if t.format_errors) == [1]  # the unknown tool's call; before: none
    assert rec["reward"] == 1.0 and not rec["mask_sample"]  # recorded only: the correct fix still scores 1
