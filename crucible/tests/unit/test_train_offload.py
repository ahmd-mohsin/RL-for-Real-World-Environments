"""Offload (M12 checklist #7): the trainer's state leaves the accelerator during rollouts (``cpu`` / ``nvme``) and
returns bit for bit; measured on a real accelerator (MPS / CUDA) when present: ≤ 5 % of the training peak stays
resident; the runtime offloads after each publish and onloads before each step (colocated only); KV modes are checked
against what each engine supports."""

from __future__ import annotations

import copy

import pytest
import torch

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _dev() -> str | None:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return None


def _alloc(dev: str) -> int:
    return torch.cuda.memory_allocated() if dev == "cuda" else torch.mps.current_allocated_memory()


def _recs(n: int = 8) -> tuple[list[RolloutRecord], list[str]]:
    g = torch.Generator().manual_seed(0)
    recs = [RolloutRecord(rollout_id=f"r{i}", sample="s", task="t", harness="h", policy="p", reward=float(i % 2),
                          turns=[Turn(call_id=f"r{i}:c", actor="agent", prompt_token_ids=torch.randint(0, 32, (10,), generator=g).tolist(),
                                      completion_token_ids=torch.randint(0, 32, (5,), generator=g).tolist(), logprobs=[-3.0] * 5, policy_version=0)])
            for i in range(n)]
    return recs, [f"g{i // 4}" for i in range(n)]


@pytest.mark.parametrize("mode", ["cpu", "nvme"])
def test_offload_round_trip_is_exact(mode, tmp_path):
    dev = _dev() or "cpu"
    recs, groups = _recs()
    a = TorchTrainer(TinyPolicy(32, 256, seed=0).to(dev), TrainerConfig(lr=1e-3))
    b = TorchTrainer(copy.deepcopy(a.model), TrainerConfig(lr=1e-3))
    for t in (a, b):
        t.step(recs, groups)
    a.offload(mode, path=str(tmp_path / "state.pt") if mode == "nvme" else None)
    p0 = next(a.model.parameters())
    assert p0.device.type == "cpu" and (p0.numel() == 0 if mode == "nvme" else p0.numel() > 0)  # nvme: host memory released too
    a.onload()
    assert next(a.model.parameters()).device.type == torch.device(dev).type
    for t in (a, b):
        t.step(recs, groups)  # moments survived: the second Adam step is identical
    tol = 0.0 if dev == "cpu" else 1e-6  # MPS kernels are not bitwise deterministic (two never-offloaded copies differ by ~6e-8)
    assert max(float((x - y).abs().max()) for x, y in zip(a.model.state_dict().values(), b.model.state_dict().values(), strict=True)) <= tol


@pytest.mark.skipif(_dev() is None, reason="needs an accelerator (MPS or CUDA)")
@pytest.mark.parametrize("mode", ["cpu", "nvme"])
def test_offload_frees_the_accelerator(mode, tmp_path):
    dev = _dev()
    recs, groups = _recs()
    tr = TorchTrainer(TinyPolicy(32, 768, seed=0).to(dev), TrainerConfig(lr=1e-3))
    tr.step(recs, groups)
    if dev == "mps":
        torch.mps.synchronize()
    peak = _alloc(dev)  # parameters + gradients + Adam moments resident
    tr.offload(mode, path=str(tmp_path / "s.pt") if mode == "nvme" else None)
    held = _alloc(dev)
    assert held <= 0.05 * peak, (held, peak)
    tr.onload()
    # parameters + Adam moments back on the device (``peak`` also held gradients and, on CUDA, the cuBLAS workspace that
    # offload frees and the next matmul re-creates, so it is not the onload target)
    state = sum(p.numel() * p.element_size() for p in tr.model.parameters())
    state += sum(t.numel() * t.element_size() for st in tr.opt.state.values() for t in st.values() if torch.is_tensor(t) and t.device.type == dev)
    assert _alloc(dev) >= 0.95 * state and all(p.device.type == dev for p in tr.model.parameters())
    assert tr.step(recs, groups)["train/items"] == len(recs)


async def test_runtime_offloads_after_publish_and_onloads_before_training():
    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    log: list[str] = []
    def spy(name: str, f):
        def wrapped(*a, **k):
            log.append(name)
            return f(*a, **k)

        return wrapped

    for name in ("offload", "onload", "step", "state_dict"):
        setattr(tr, name, spy(name, getattr(tr, name)))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=3, max_turn_tokens=2, warmup_batches=0, train_state_on_rollout="cpu"),
                 SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2), {"copy": [f"p{i}" for i in range(200)]})
    rep = await rt.run()
    assert len(rep.steps) == 3
    seq = [x for x in log if x in ("offload", "onload", "step", "state_dict")]
    assert seq[0] == "offload" and seq[1:] == ["onload", "step", "state_dict", "offload"] * 3  # trained resident, published, then freed


def test_offload_and_kv_mode_validation():
    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=8)])
    tr = TorchTrainer(model, TrainerConfig())
    mk = lambda **kw: Runtime(RuntimeConfig(**kw), SchedulerConfig(n=2, B=2), eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(), {"c": ["p"]})
    with pytest.raises(ValueError, match="keeps the trainer resident"):
        mk(topology="disaggregated_stream", train_state_on_rollout="cpu")
    with pytest.raises(ValueError, match="supports kv_on_train"):
        mk(kv_on_train="cpu_pool")
    with pytest.raises(ValueError, match="none \\| cpu \\| nvme"):
        mk(train_state_on_rollout="gpu")
