"""OpenEnv import (``interop.openenv.import_env``): any OpenEnv Environment as a Crucible World. The PortSimEnv /
generic-env cases run inside ``crucible/.envs/openenv`` (``_openenv_import_cases.py``); the World's
environment-terminated episodes (``ToolResult.ends_episode``) are tested here directly."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

CASES = Path(__file__).with_name("_openenv_import_cases.py")
PY = Path(__file__).resolve().parents[2] / ".envs" / "openenv" / "bin" / "python"


def _case(name: str, tmp_path: Path) -> None:
    if not PY.exists():
        pytest.skip("OpenEnv env not built (crucible/.envs/openenv)")
    probe = subprocess.run([str(PY), "-c", "import berth_openenv, torch"], capture_output=True, check=False)
    if probe.returncode:
        pytest.skip("PortSimEnv / torch not installed in crucible/.envs/openenv")
    r = subprocess.run([str(PY), str(CASES), name, str(tmp_path)], capture_output=True, text=True, check=False, timeout=900)
    assert r.returncode == 0 and r.stdout.strip().endswith("OK"), (r.stdout[-1500:], r.stderr[-4000:])


def test_portsim_rewards_match_the_native_environment(tmp_path):
    """oracle 1.0 = native rubric; serial baseline = native (≤ 0.2); no submission 0 (scored, not masked); a call
    after submit_plan never runs (the environment ended the episode)."""
    _case("portsim_rewards", tmp_path)


def test_restore_replays_the_journal_and_verifies_digests(tmp_path):
    _case("restore", tmp_path)


def test_generic_action_environment(tmp_path):
    """A non-MCP OpenEnv env: one ``step`` tool from its Action schema; last / sum reward modes; schema-invalid args."""
    _case("generic_env", tmp_path)


def test_nondeterministic_environment_refuses_restore(tmp_path):
    _case("nondeterministic", tmp_path)


def test_cross_runtime_identity_import_replay_reexport(tmp_path):
    """native PortSim = Crucible import = canonical replay = OpenEnv re-export (step rewards telescope to the final)."""
    _case("cross_runtime", tmp_path)


def test_portsim_through_the_m12_runtime(tmp_path):
    _case("m12_runtime", tmp_path)


async def test_world_ends_the_episode_when_a_tool_result_says_so(tmp_path):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
    from crucible.core.registry import get_registry
    from crucible.core.schema.actions import ToolCall, ToolResult, ToolSpec
    from crucible.core.schema.world import WorldSpec
    from crucible.core.store import Store
    from crucible.world.apps.base import App
    from crucible.world.world import World

    class Ender(App):
        template = "test/ender"
        PATH_PARAMS = ("path",)

        async def list_tools(self):
            return [ToolSpec(app=self.id, name="answer", description="final answer", input_schema={"type": "object", "properties": {}})]

        async def call(self, name, args, *, actor, call_id):
            return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.answer", ok=True, content="graded", ends_episode=True)

    get_registry().register("app_template", "test/ender", Ender, replace=True)
    spec = WorldSpec.model_validate({"world": "test/ender", "apps": [{"id": "e", "template": "test/ender", "state": {"adapter": "json_doc", "params": {"path": str(tmp_path / "j.json")}}}],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["e.*"]}]})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    script = [ToolCall(tool_call_id=f"a{i}", function_name="e.answer", arguments={}) for i in range(3)]
    ep = await run_episode(w, {"agent": ScriptedPolicy("agent", script)}, limits=EpisodeLimits(max_steps=10))
    assert ep.steps == 1 and "agent" in w.done
    assert [e.text for e in w.log if e.type == "done"] == ["environment_terminal"]
    await w.close()
