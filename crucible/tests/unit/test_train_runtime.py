"""M12 runtime end to end on CPU: a real tiny torch policy learns a multi-turn token task through the asyncio runtime,
the shared scheduler, a bf16 engine (a real train–inference gap), the reference trainer and in-process weight publishing,
under every topology and both version-switch modes. Invariants checked on every run: staleness ≤ τ, version purity
(call-level) or valid version spans (token-level), aborts at the colocated barrier, no pause in disaggregated mode,
masked infra failures never trained."""

from __future__ import annotations

import pytest

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


async def run(topology: str, *, steps: int = 30, vs: str = "call", tau: int = 2, infra_fail: float = 0.0, correction: str = "dis"):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.002, version_switch=vs)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction(correction, old_logprob_source="rollout" if correction != "dis" else None)))
    rt = Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=3, version_switch=vs), SchedulerConfig(n=8, B=4, max_staleness=tau),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, long_tail=0.1, infra_fail=infra_fail),
                 {"copy": [f"p{i}" for i in range(4000)]})
    rep = await rt.run()
    return rt, rep


def learned(rep) -> tuple[float, float]:
    r = rep.series("reward/mean")
    return sum(r[:5]) / 5, sum(r[-10:]) / 10


@pytest.mark.parametrize("topology,vs", [("colocated_sync", "call"), ("colocated_partial", "call"), ("colocated_partial", "token"),
                                         ("disaggregated_stream", "call")])
async def test_runtime_learns_and_keeps_invariants(topology, vs):
    n = 40 if topology == "disaggregated_stream" else 30
    _, rep = await run(topology, vs=vs, steps=n)
    first, last = learned(rep)
    assert len(rep.steps) == n and rep.publishes == n and last > first + 0.08, (first, last)
    assert max(rep.series("staleness/max")) <= (0 if topology == "colocated_sync" else 2)
    assert rep.max_untrained <= (4 * 8 if topology == "colocated_sync" else 2 * 4 * 8)  # S·B·n untrained cap
    for r in rep.trained_records:
        for t in r.turns:
            assert t.spans_valid()
            if vs == "call":
                assert t.version_spans is None  # a call is served by exactly one version
    if topology == "colocated_partial":
        assert rep.aborted_calls > 0 and rep.reprefill_tokens > 0
        if vs == "token":
            assert rep.resumed_calls > 0 and any(t.version_spans for r in rep.trained_records for t in r.turns)
    if topology == "disaggregated_stream":
        assert rep.aborted_calls == 0  # generation never stops for training
    assert all(s.version >= 1 for s in rep.steps)


async def test_infra_failures_are_masked_not_trained():
    rt, rep = await run("colocated_partial", steps=10, infra_fail=0.05)
    assert rep.masked_rollouts > 0 and rt.sched.counters["evicted_masked"] + rep.masked_rollouts > 0  # failures happened
    assert all(r.reward is not None for r in rep.trained_records)  # none of them reached the trainer as a 0


async def test_corrections_run_in_the_loop():
    for c in ("tis", "icepop", "mimo4", "cispo"):
        _, rep = await run("colocated_partial", steps=6, correction=c)
        assert len(rep.steps) == 6 and "mismatch/masked_token_share" in rep.steps[-1].metrics
