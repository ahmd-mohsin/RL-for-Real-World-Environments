"""P5b, ``scripts/lh_run.py``'s ``harness()``: the sidecar is ``envs.InfraWatch`` (V8: a World container lost mid-episode is
recorded as ``crucible:env_container_lost`` and the attempt is masked, never graded as an ordinary 0; P6 found the native
path graded it 0), around the checkpoint sidecar when checkpoints are on and around nothing when they are off."""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from test_m17_p6_equiv import HostWindow
from test_m17_w_task import make_task

from crucible.factory.envs import InfraWatch

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("lh_run_p5b", ROOT / "scripts" / "lh_run.py")
lh = importlib.util.module_from_spec(_spec)
sys.modules["lh_run_p5b"] = lh
_spec.loader.exec_module(lh)  # type: ignore[union-attr]


def test_harness_wraps_the_checkpoint_sidecar_in_infra_watch(tmp_path: Path) -> None:
    task, _rec, _repo = make_task(tmp_path)
    task.__class__ = HostWindow
    h = lh.harness(lh.EpisodeCfg(tool_limit=150))
    assert h.limits.max_steps == 152
    side = h.sidecar(SimpleNamespace(task=task, dir=tmp_path / "a1"))
    assert isinstance(side, InfraWatch) and type(side.inner).__name__ == "CheckpointSidecar" and side.inner.cfg.every == 10 and side.inner.cfg.cap == 20
    off = lh.harness(lh.EpisodeCfg(), checkpoints=False).sidecar(SimpleNamespace(task=task, dir=tmp_path / "a2"))
    assert isinstance(off, InfraWatch) and off.inner is None
    other = lh.harness(lh.EpisodeCfg()).sidecar(SimpleNamespace(task=object(), dir=tmp_path / "a3"))  # not a milestone task: no checkpoints
    assert isinstance(other, InfraWatch) and other.inner is None


def test_a_dead_world_is_recorded_as_infra_by_the_harness_sidecar(tmp_path: Path) -> None:
    faults: list[tuple[str, str]] = []
    world = SimpleNamespace(_crucible_alive=lambda: False, _infra=lambda kind, why: faults.append((kind, why)))
    side = lh.harness(lh.EpisodeCfg(), checkpoints=False).sidecar(SimpleNamespace(task=object(), dir=tmp_path))

    async def go() -> dict:
        await side.on_start(world)
        await side.on_step(world, 3, "agent", None, None)
        return await side.on_end(world)

    out = asyncio.run(go())
    assert faults == [("crucible:env_container_lost", "the World's sandbox died (after step 3)")] and out == {"infra_dead": "after step 3"}
