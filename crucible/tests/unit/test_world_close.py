"""World.close finishes its teardown, bounded (M1 audit 2026-10-14 R-9): a cancel during close no longer skips the
backend teardown for good, a hung App stop is cut off, Apps stop in reverse start order, and a teardown runs once. Since
the 2026-10-20 merge into dev the shielded, bounded close itself is M10 lab audit R-6's (test_m10_lab_audit_r6_exit_paths)."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world import world as world_mod
from crucible.world.world import World


async def started(tmp_path: Path) -> tuple[World, list[str]]:
    """A World with Apps ``a`` then ``b``; the returned list records ``stop:<app>`` and ``teardown`` as they happen."""
    w = World(WorldSpec.model_validate({
        "world": "test/close", "actors": [{"id": "agent", "kind": "policy", "tools": []}],
        "apps": [{"id": x, "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": x}}} for x in ("a", "b")],
    }), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    seen: list[str] = []
    for app in w.apps.values():
        async def stop(app_id=app.id):
            seen.append(f"stop:{app_id}")
        app.stop = stop
    teardown = w.backend.teardown

    async def counted():
        seen.append("teardown")
        await teardown()

    w.backend.teardown = counted
    return w, seen


async def test_a_cancelled_close_still_tears_down(tmp_path: Path):
    w, seen = await started(tmp_path)

    async def slow():
        await asyncio.sleep(0.5)
        seen.append("stop:b")

    w.apps["b"].stop = slow
    t = asyncio.create_task(w.close())
    await asyncio.sleep(0.1)
    t.cancel()  # a second Ctrl-C, a task group shutting down: lands in the first App's stop
    with pytest.raises(asyncio.CancelledError):
        await t  # the caller still learns it was cancelled
    assert seen == ["stop:b", "stop:a", "teardown"] and w.closed  # before: no teardown, and closed already set
    await w.close()
    assert seen.count("teardown") == 1  # a later close waits for the same teardown, never a second one


async def test_a_hung_stop_is_cut_off_and_the_teardown_still_runs(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(world_mod, "APP_STOP_S", 0.2)
    w, seen = await started(tmp_path)

    async def hang():
        await asyncio.sleep(3600)

    async def broken():
        seen.append("teardown")
        raise RuntimeError("compose down failed")

    w.apps["b"].stop, w.backend.teardown = hang, broken
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="compose down failed"):  # the teardown's own error, to the first caller (M10 lab audit R-6)
        await w.close()
    await w.close()  # not to the second
    assert time.monotonic() - t0 < 5 and seen == ["stop:a", "teardown"] and w.closed  # b was cut off at 0.2 s


async def test_concurrent_closes_tear_down_once(tmp_path: Path):
    w, seen = await started(tmp_path)
    await asyncio.gather(w.close(), w.close(), w.close())
    assert seen == ["stop:b", "stop:a", "teardown"]  # reverse start order, one teardown
