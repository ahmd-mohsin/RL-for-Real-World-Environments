"""CI App with the local runner: hidden checks never enter the policy's world, the policy sees pass/fail only
(plus scrubbed output for ``show_output`` checks), runs cost virtual time and a budget that is itself world state."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

HIDDEN_UNIT = "import sys\nsys.path.insert(0, '.')\nimport calc\nsys.exit(0 if calc.add(2, 3) == 5 else 1)\n"
HIDDEN_LINT = "import sys\nprint('see https://github.com/acme/calc/pull/7 for the fix')\nprint('lint: calc.py:2 suspicious operator')\nsys.exit(1)\n"


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack"
    (p / "repo").mkdir(parents=True)
    (p / "repo" / "calc.py").write_text("def add(a, b):\n    return a - b\n")
    (p / "hidden").mkdir()
    (p / "hidden" / "check_add.py").write_text(HIDDEN_UNIT)
    (p / "hidden" / "lint.py").write_text(HIDDEN_LINT)
    return p


def spec() -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/ci",
        "apps": [
            {"id": "repo", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "repo"}}, "params": {"seed_dir": "repo"}},
            {"id": "ci", "template": "common/ci", "state": {"adapter": "json_doc", "params": {"path": "ci.json"}},
             "params": {"source": "repo", "target_root": "/src", "hidden": {"src": "hidden", "dest": "/opt/hidden"}, "max_runs": 2, "latency": "20m",
                        "checks": [{"name": "unit", "cmd": "python3 {hidden}/check_add.py"},
                                   {"name": "lint", "cmd": "python3 {hidden}/lint.py", "show_output": True}]}},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*", "ci.*"]}],
        "clock": {"mode": "step", "t0": "2026-03-02T09:00:00Z"},
    })


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=tool, function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_ci_hidden_checks(pack: Path, tmp_path: Path):
    w = World(spec(), store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=pack)
    await w.start()
    try:
        await w.reset()
        s0 = await w.snapshot()
        # the hidden checks are nowhere in the policy's world
        assert "check_add" not in (await call(w, "repo.list_dir", path=".", depth=5)).content
        t0 = w.clock.now()
        r = await call(w, "ci.run_ci")
        assert r.ok and "unit: FAIL" in r.content and "lint: FAIL" in r.content
        assert "check_add" not in r.content and "sys.exit" not in r.content  # results only, never the check code
        assert "suspicious operator" in r.content and "github.com" not in r.content and "pull/7" not in r.content  # shown output is scrubbed
        assert (w.clock.now() - t0).total_seconds() >= 20 * 60  # CI costs virtual time
        await call(w, "repo.edit_file", path="calc.py", old_string="a - b", new_string="a + b")
        r = await call(w, "ci.run_ci", checks=["unit"])
        assert "unit: PASS" in r.content and "lint" not in r.content and "#2 of 2" in r.content
        r = await call(w, "ci.run_ci")
        assert not r.ok and "budget exhausted" in r.content
        r = await call(w, "ci.run_ci", checks=["secret_check"])
        assert not r.ok
        # the budget is world state: restoring the reset snapshot gives the runs back
        await w.restore(s0)
        hist = await call(w, "ci.ci_history")
        assert '"runs": 0' in hist.content
        r = await call(w, "ci.run_ci", checks=["unit"])
        assert "unit: FAIL" in r.content  # the restored candidate is the buggy base again
    finally:
        await w.close()
