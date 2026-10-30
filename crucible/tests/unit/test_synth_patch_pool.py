"""M5's migration requirement on the real training pool (part2-paper-core §14 M5; the evidence doc §12 "migration without
breaking what exists"): every stored gate verdict of the 12 built training tasks recomputes identically through the
``PatchTask`` adapter. Needs Docker and the pool (``CRUCIBLE_DEBUG_POOL`` or ``crucible/data/debug-pool``); about 10
minutes for all 12 (``CRUCIBLE_SYNTH_POOL_TASKS=a,b`` runs a subset)."""

from __future__ import annotations

import asyncio
import json
import os
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
POOL = Path(os.environ.get("CRUCIBLE_DEBUG_POOL") or ROOT / "data" / "debug-pool")
MANIFEST = ROOT / "docs" / "research" / "swarm-pipeline" / "07-debug-lane" / "training-pool" / "pool.json"

pytestmark = [pytest.mark.docker, pytest.mark.slow,
              pytest.mark.skipif(not (POOL / "gen" / "store" / "train").is_dir() or shutil.which("docker") is None, reason="no debug pool or no Docker")]


def ready() -> list[str]:
    ids = sorted(t for t, r in json.loads(MANIFEST.read_text())["tasks"].items() if r.get("status") == "ready")
    want = [t for t in (os.environ.get("CRUCIBLE_SYNTH_POOL_TASKS") or "").split(",") if t]
    return [t for t in ids if not want or t in want]


@pytest.mark.parametrize("wid", ready())
def test_stored_gate_verdicts_recompute_identically(wid):
    from crucible.factory.synth_patch_wire import lane_runner, open_pool, patch_env, remove_containers
    from crucible.synth.patch import migrate

    pool = open_pool(POOL, parallel=4)
    task = pool.task(wid)
    env = patch_env(task)
    runner = lane_runner(task, parallel=4)
    try:
        rep = asyncio.run(migrate(env, runner, pool.gates(wid)))
    finally:
        runner.close()
        remove_containers()
    assert not rep.errors, rep.errors
    assert rep.equal, [d.model_dump() for d in rep.differences()]
    fields = {(r.case, r.field) for r in rep.rows}
    assert {("oracle_patch", "verdict"), ("oracle_patch", "milestones"), ("no_op", "verdict"), ("no_op", "start_any_milestone_test")} <= fields
