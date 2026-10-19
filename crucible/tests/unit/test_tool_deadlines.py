"""A tool's own deadline fires before the World's (M1 audit 2026-10-14 R-4): a policy's hung command is its own scored
``[TIMEOUT]``, and only a tool hung past its own deadline is a masked ``transport_timeout``. Equal caps (3600 s each)
used to turn every policy hang into a mask, which a policy could use to escape a 0."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world import world as world_mod
from crucible.world.world import World


@pytest.fixture
async def world(tmp_path: Path):
    shell = {"shell": True, "allow_unsafe_local_shell": True}  # a scripted test policy on the local backend
    w = World(WorldSpec.model_validate({
        "world": "test/deadlines", "limits": {"tool_timeout_s": 2},
        "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}, "params": shell},
                 {"id": "box", "template": "common/toolchain", "state": {"adapter": "process", "params": {"cwd": "."}}, "params": shell}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*", "box.*"]}],
    }), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    yield w
    await w.close()


async def call(w: World, name: str, **args):
    return (await w.step("agent", ToolCall(tool_call_id="c", function_name=name, arguments=args))).observation.tool_results[0]


@pytest.mark.parametrize("tool", ["ws.shell", "box.shell"])
async def test_a_policys_hung_command_is_its_own_scored_timeout(world: World, tool: str):
    t0 = time.monotonic()
    r = await call(world, tool, command="echo started; sleep 30", timeout_s=99)  # asks past the World's 2 s
    assert r.ok and r.content.startswith("[TIMEOUT]") and "started" in r.content and world.failures == []
    assert time.monotonic() - t0 < 10  # its own deadline: the World's 2 s, not the 99 s asked or the World's grace


async def test_a_tool_hung_past_its_own_deadline_is_infrastructure(world: World, monkeypatch):
    monkeypatch.setattr(world_mod, "TOOL_GRACE_S", 0.5)

    async def hang(**kw):
        await asyncio.sleep(30)

    world.apps["ws"].tool_list_dir = hang
    r = await call(world, "ws.list_dir")
    assert not r.ok and r.error == "timeout" and r.fault == "infra"
    assert [f.kind for f in world.failures] == ["transport_timeout"] and "own deadline (2.0s + 0.5s)" in world.failures[0].reason


async def test_the_deadline_a_tool_runs_with(world: World):
    ws = world.apps["ws"]
    assert ws.deadline(99999) == 2.0 and ws.deadline(1.5) == 1.5 and ws.deadline(10, cap=1.2) == 1.2
    assert ws.deadline(0) == 1.0 and ws.deadline(-5) == 1.0  # 0 meant "no deadline" to the exec: a hang, so a mask
