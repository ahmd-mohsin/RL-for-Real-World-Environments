"""Tool modules that read the wall clock or os.urandom become deterministic under the World's virtual clock and
seed (virtual_env.py): the same call from the same WorldRef yields the same database, and timestamps agree with
the World's time. Reproduces the survey's finding on a real MiMo environment (1,174 modules call datetime.now)."""

from __future__ import annotations

import json
import textwrap
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

MODULE = textwrap.dedent('''
    """Tools that use every common source of nondeterminism."""
    import datetime as dt
    import random
    import time
    import uuid
    from datetime import date, datetime, timezone
    from time import time as now_ts
    from uuid import uuid4


    def stamp() -> dict:
        """Return times and ids."""
        return {"aware": datetime.now(timezone.utc).isoformat(), "naive": datetime.now().isoformat(), "utc": datetime.utcnow().isoformat(),
                "mod": dt.datetime.now(dt.timezone.utc).isoformat(), "today": date.today().isoformat(), "ts": now_ts(), "t": time.time(),
                "id": str(uuid4()), "id2": str(uuid.uuid4()), "r": random.random(), "ri": random.randint(1, 10**9),
                "seq": [datetime.now(timezone.utc).strftime("%H%M%S%f") for _ in range(3)]}
''')


@pytest.fixture
def pack(tmp_path: Path) -> Path:
    p = tmp_path / "pack"
    p.mkdir()
    (p / "tools.py").write_text(MODULE)
    return p


def spec(pack: Path, seed: int = 7) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/virtual-env", "seed": seed,
        "apps": [{"id": "erp", "template": "common/sqlite_app", "state": {"adapter": "sql", "params": {"path": "erp.db"}}, "params": {"tools_module": str(pack / "tools.py")}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["erp.*"]}],
        "clock": {"mode": "step", "t0": "2026-03-02T09:00:00Z", "step_dt_s": 60},
    })


async def stamp(w: World) -> dict:
    r = await w.step("agent", ToolCall(tool_call_id="s", function_name="erp.stamp", arguments={}))
    res = r.observation.tool_results[0]
    assert res.ok, res.content
    return json.loads(res.content)


async def test_tool_time_and_randomness_follow_the_world(pack: Path, tmp_path: Path):
    w = World(spec(pack), store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=pack)
    await w.start()
    try:
        await w.reset()
        a = await stamp(w)
        assert a["aware"].startswith("2026-03-02T09:00:00") and a["mod"].startswith("2026-03-02T09:00") and a["today"] == "2026-03-02"
        assert a["naive"].startswith("2026-03-02T09:00") and a["utc"].startswith("2026-03-02T09:00") and int(a["ts"]) // 100 == 17724420
        assert len(set(a["seq"])) == 3 and a["seq"] == sorted(a["seq"])  # unique, monotonic ids within one instant
        b = await stamp(w)
        assert b["aware"] > a["aware"] and b["id"] != a["id"]  # the clock moved one step
        await w.reset()
        assert await stamp(w) == a  # same WorldRef, same call -> same values
    finally:
        await w.close()
    w2 = World(spec(pack, seed=8), store=Store(tmp_path / "s2"), workdir=tmp_path / "w2", pack_root=pack)
    await w2.start()
    try:
        await w2.reset()
        c = await stamp(w2)
        assert c["aware"] == a["aware"] and c["id"] != a["id"] and c["r"] != a["r"]  # the seed changes randomness, not time
    finally:
        await w2.close()


MIMO = Path(__file__).resolve().parents[3] / "RL-Environments/01-lab-releases/xiaomi-mimo-v2.6-rl-oss/general/envs/s3k_0027_accounting_audit_tax_en_t2_rl_004"


@pytest.mark.skipif(not MIMO.is_dir(), reason="MiMo release not present")
async def test_mimo_replay_is_deterministic(tmp_path: Path):
    from crucible.interop.mimo.importer import import_mimo_env

    env = import_mimo_env(MIMO)
    from crucible.core.schema.world import ClockSpec

    spec = env.world.model_copy(update={"clock": ClockSpec(mode="step", t0="2026-03-02T09:00:00Z")})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w", pack_root=MIMO)
    await w.start()
    try:
        await w.reset()
        app = next(a for a in w.apps if "ecosys" in a)
        row = (await w.verifier_view().query(app, Query(op="sql", args={"sql": "SELECT schedule_version_id, work_package_code FROM annual_schedule_lines LIMIT 1"}))).value
        sv, wp = row["rows"][0] if isinstance(row, dict) else row[0]
        args = {"schedule_version_id": sv, "requested_by_entity_id": "ENT-1", "change_reference": "REF-1", "change_reason_code": "SCOPE",
                "requested_amount_integer": 10, "affected_work_package_code": wp, "requested_effective_year": 2027, "justification": "audit"}
        fps = []
        for _ in range(2):
            await w.reset()
            r = await w.step(spec.policy_actors[0].id, ToolCall(tool_call_id="c", function_name=f"{app}.submit_schedule_change", arguments=args))
            assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
            assert "2026-03-02T09:00" in r.observation.tool_results[0].content  # the tool's created_at is the World's time
            fps.append((await w.snapshot()).apps[app].fingerprint)
        assert fps[0] == fps[1]
    finally:
        await w.close()
