"""A policy's SQL statement stops at its own deadline (M1 audit 2026-10-14 R-12): a runaway query is a scored
``query_timeout`` within the deadline, a runaway write is rolled back, and the thread running it ends (on the local
backend it used to run until the World masked the call, and kept burning CPU after)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

FOREVER = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) "


@pytest.fixture
async def world(tmp_path: Path):
    schema = tmp_path / "s.sql"
    schema.write_text("CREATE TABLE t (x); INSERT INTO t VALUES (1), (2);")
    w = World(WorldSpec.model_validate({
        "world": "t/sql-deadline", "limits": {"tool_timeout_s": 5},
        "apps": [{"id": "db", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "db.sqlite"}},
                  "params": {"schema_sql": str(schema), "sql_tools": True, "sql_write": True, "sql_deadline_s": 1}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["db.*"]}],
    }), store=Store(tmp_path / "st"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    yield w
    await w.close()


async def call(w: World, tool: str, sql: str):
    return (await w.step("agent", ToolCall(tool_call_id="c", function_name=f"db.{tool}", arguments={"sql": sql}))).observation.tool_results[0]


async def test_a_runaway_query_stops_at_its_deadline(world: World):
    t0 = time.monotonic()
    r = await call(world, "sql_query", FOREVER + "SELECT max(x) FROM c")
    assert not r.ok and "query_timeout: the statement ran past 1s" in r.content and r.fault != "infra" and world.failures == []
    assert time.monotonic() - t0 < 4  # its own 1 s, not the World's 5 s + 60 s grace (then masked)
    assert "2" in (await call(world, "sql_query", "SELECT count(*) FROM t")).content  # the next query runs normally


async def test_a_runaway_write_is_rolled_back(world: World):
    r = await call(world, "sql_execute", FOREVER + "INSERT INTO t SELECT x FROM c")
    assert not r.ok and "query_timeout" in r.content and "nothing it did was kept" in r.content
    assert (await world.verifier_view().query("db", {"op": "scalar", "args": {"sql": "SELECT count(*) FROM t"}})).value == 2
