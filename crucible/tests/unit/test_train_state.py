"""The state plane (M12 checklist #1): run checkpoints at the sample barrier; resume continues the run. Synchronous
mode resumes bit for bit; partial rollout restarts in-flight rollouts, keeps finished-untrained ones, never reuses a
prompt and continues the version sequence; corrupt checkpoints are refused."""

from __future__ import annotations

import pytest
import torch

from crucible.train import state
from crucible.train.critic import CriticConfig, TinyValue
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _rt(topology: str, steps: int, ckdir=None, every: int = 0, critic: bool = False, delay: float = 0.0):
    model = TinyPolicy(32, 24, seed=3)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=delay)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2), critic=TinyValue(32, 24, seed=4) if critic else None, critic_cfg=CriticConfig())
    rt = Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=3, warmup_batches=0, checkpoint_every=every,
                               checkpoint_dir=str(ckdir) if ckdir else None),
                 SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng),
                 lambda: CopyTask(turns=2, symbols=8, latency_s=delay, long_tail=0.3 if delay else 0.0), {"copy": [f"p{i}" for i in range(400)]})
    return rt, model, tr


@pytest.mark.parametrize("critic", [False, True])
async def test_synchronous_resume_is_bit_identical(tmp_path, critic):
    ref, ref_model, ref_tr = _rt("colocated_sync", 4, critic=critic)
    await ref.run()
    a, _, _ = _rt("colocated_sync", 2, tmp_path, every=2, critic=critic)
    await a.run()
    assert state.latest(tmp_path) is not None
    b, b_model, b_tr = _rt("colocated_sync", 4, tmp_path, critic=critic)
    assert await b.resume() == 2 and b.version == 2
    rep = await b.run()
    assert [s.step for s in rep.steps] == [3, 4] and [s.version for s in rep.steps] == [3, 4]
    for (k, x), y in zip(ref_model.state_dict().items(), b_model.state_dict().values(), strict=True):
        assert torch.equal(x, y), k
    if critic:
        assert all(torch.equal(x, y) for x, y in zip(ref_tr.critic.state_dict().values(), b_tr.critic.state_dict().values(), strict=True))


async def test_partial_rollout_resume_restarts_in_flight_and_keeps_the_rest(tmp_path):
    a, _, _ = _rt("colocated_partial", 3, tmp_path, every=3, delay=0.003)
    rep_a = await a.run()
    ck = state.load(state.latest(tmp_path))
    live = [r for r, t in ck["pool"]["tags"].items() if t["status"] in ("pending", "running", "grading")]
    kept = [r for r, t in ck["pool"]["tags"].items() if t["status"] == "finished"]
    assert live and set(kept) <= set(ck["pool"]["values"])  # finished-untrained values travel with the checkpoint
    b, _, _ = _rt("colocated_partial", 6, tmp_path, delay=0.003)
    await b.resume()
    assert b.report.restarted_rollouts == sum(1 for r in live if ck["pool"]["tags"][r]["status"] != "pending")
    rep = await b.run()
    assert [s.step for s in rep.steps] == [4, 5, 6] and rep.steps[0].version == ck["version"] + 1
    prompts_a = {g["meta"]["prompt"] for g in ck["pool"]["groups"].values()}
    new = {g.meta["prompt"] for gid, g in b.pool.groups.items() if gid not in ck["pool"]["groups"]}
    assert new and not (new & prompts_a)  # the source queue continued: no prompt drawn twice
    assert rep_a.publishes == 3 and b.report.publishes == 6


def test_corrupt_checkpoint_is_refused(tmp_path):
    p = state.save({"schema": "crucible.run_checkpoint/v1", "step": 1, "version": 1, "x": 1}, tmp_path)
    assert state.load(p)["x"] == 1
    (p / "state.pt").write_bytes((p / "state.pt").read_bytes()[:-3] + b"xyz")
    with pytest.raises(ValueError, match="corrupt"):
        state.load(p)
    state.save({"schema": "crucible.run_checkpoint/v1", "step": 7, "version": 7}, tmp_path)
    assert state.latest(tmp_path).name == "step-000007" and not list(tmp_path.glob(".ckpt-*"))  # no temp dirs left behind
