"""A restore of a WorldRef already in place is skipped (M1 audit 2026-10-14 R-5): start, reset and a harness's reset used
to restore every App up to three times before step 1. A skip needs both: nothing ran since the state was set (a
fingerprint cannot see file times, owners or an App's own memory) and the App's live fingerprint still matches."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World


@pytest.fixture
async def world(tmp_path: Path):
    schema = tmp_path / "s.sql"
    schema.write_text("CREATE TABLE t (id INTEGER PRIMARY KEY AUTOINCREMENT, x TEXT);")
    w = World(WorldSpec.model_validate({
        "world": "test/r5",
        "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}},
                 {"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "db.sqlite"}},
                  "params": {"schema_sql": str(schema), "sql_tools": True, "sql_write": True}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*", "db.*"]}],
    }), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    yield w
    await w.close()


async def call(w: World, name: str, **args):
    return (await w.step("agent", ToolCall(tool_call_id="c", function_name=name, arguments=args))).observation.tool_results[0]


async def test_no_app_is_restored_before_step_1(world: World):
    await world.reset()  # right after start
    assert world.restored_apps == []
    await world.reset()  # a harness's own reset
    assert world.restored_apps == []


async def test_after_a_tool_call_every_app_is_restored(world: World):
    await world.reset()
    await call(world, "ws.write_file", path="a.txt", content="x")
    await world.reset()
    assert world.restored_apps == ["ws", "db"] and not (Path(world.apps["ws"].root) / "a.txt").exists()


async def test_a_change_made_underneath_is_restored(world: World):
    await world.reset()
    (Path(world.apps["ws"].root) / "planted.txt").write_text("a setup hook wrote this")  # no tool call: only the fingerprint sees it
    await world.reset()
    assert world.restored_apps == ["ws"] and not (Path(world.apps["ws"].root) / "planted.txt").exists()


async def test_a_verifier_query_counts_as_activity(world: World):
    await world.reset()
    await world.verifier_view().query("db", {"op": "count", "args": {"table": "t"}})
    await world.reset()
    assert world.restored_apps == ["ws", "db"]


async def test_any_tool_call_means_a_full_restore_even_with_equal_fingerprints(world: World):
    await world.reset()
    before = await world.verifier_view().fingerprints()
    assert (await call(world, "ws.list_dir")).ok  # read-only: every fingerprint is unchanged
    assert await world.verifier_view().fingerprints() == before
    await world.reset()
    assert world.restored_apps == ["ws", "db"]  # what a call may have changed, a fingerprint cannot always see
