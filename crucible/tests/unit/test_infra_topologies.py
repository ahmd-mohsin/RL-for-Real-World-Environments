"""M12-infra batch A: the elastic topology (LongCat DORA), partial-rollout round boundaries (Kimi K3 completion fraction λ,
Ring-1T C3PO++ token budget Φ), retention σ (Ring), and KV-preserving resume (DeepSeek-V4.1). Each is checked in the
discrete-event simulator (the same scheduler, deterministic) and in the asyncio runtime with a real tiny policy."""

from __future__ import annotations

import pytest

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.sim import SimConfig, default_shapes, simulate
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher

SRC = {"a": [f"a{i}" for i in range(600)], "b": [f"b{i}" for i in range(600)]}
SHAPES = default_shapes(3, tail_frac=0.2, tail_mult=12.0)


def sim(topology="colocated_partial", *, steps=15, **kw):
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in SimConfig.__dataclass_fields__}
    return simulate(SchedulerConfig(n=4, B=4, **kw), SimConfig(topology=topology, steps=steps, **sim_kw), SRC, SHAPES)


# ----------------------------------------------------------------------------------------------- simulator
def test_elastic_sits_between_colocated_and_disaggregated():
    kw = {"slots": 32, "train_s_base": 60.0, "max_staleness": 3}
    col, ela, dis = sim("colocated_partial", **kw), sim("elastic", elastic_frac=0.5, **kw), sim("disaggregated_stream", **kw)
    assert ela.steps == col.steps == dis.steps == 15
    assert ela.engine_down_s == 0.0 and col.engine_down_s > 0  # the resident group never stops
    assert 0 < ela.busy_while_training_s and col.busy_while_training_s == 0  # generation overlaps training
    assert dis.time <= ela.time < col.time  # throughput ordering on a long-training workload
    assert ela.max_staleness() <= 3


def test_completion_fraction_barrier():
    lo, hi = sim(partial_barrier="completion_frac", barrier_frac=0.5, max_staleness=4), sim(partial_barrier="completion_frac", barrier_frac=0.95, max_staleness=4)
    assert lo.steps == hi.steps == 15
    assert any(b < 4 for b in lo.batch_sizes) and all(b >= 1 for b in lo.batch_sizes)  # trains what finished (B, or more at the bound)
    assert lo.max_staleness() <= 4 and hi.max_staleness() <= 4
    assert lo.time < hi.time  # an earlier boundary trains sooner


def test_token_budget_barrier():
    r = sim(partial_barrier="token_budget", barrier_tokens=2_000, max_staleness=4)
    assert r.steps == 15 and all(b >= 1 for b in r.batch_sizes) and any(b < 4 for b in r.batch_sizes) and r.max_staleness() <= 4
    big = sim(partial_barrier="token_budget", barrier_tokens=50_000, max_staleness=4)
    assert big.time > r.time  # a bigger budget means longer rounds


def test_retention_caps_cuts():
    r = sim(max_cuts=1, max_staleness=8, train_s_base=30.0)
    assert r.evicted.get("retention", 0) > 0
    assert max(s for b in r.spans for s in b) <= 2  # a trained rollout was cut at most once: at most two versions
    free = sim(max_staleness=8, train_s_base=30.0)
    assert max(s for b in free.spans for s in b) > 2  # without σ the long tail spans more versions


def test_kv_preserving_resume_saves_prefill_and_decode():
    base, kept = sim(max_staleness=4), sim(max_staleness=4, resume_keeps_kv=True)
    assert base.reprefill_tokens > 0 and base.wasted_decode_tokens > 0
    assert kept.reprefill_tokens == 0 and kept.wasted_decode_tokens == 0 and kept.time <= base.time


def test_staleness_bound_holds_across_topologies_and_overflow_modes():
    """``wait`` + ``bound_overflow=batch`` keeps every trained rollout within τ (a group's finished members age while it
    waits for its last runner; groups at the bound beyond B train in the same step); ``allow`` shows what that prevents."""
    for topo in ("colocated_partial", "disaggregated_stream", "elastic"):
        for seed in (1, 3, 5):
            kw = {"slots": 32, "train_s_base": 60.0, "elastic_frac": 0.5}
            ok = simulate(SchedulerConfig(n=4, B=4, max_staleness=3), SimConfig(topology=topo, steps=20, **kw), SRC, default_shapes(seed, tail_frac=0.2, tail_mult=12.0))
            assert ok.steps == 20 and ok.max_staleness() <= 3, (topo, seed)
    loose = [simulate(SchedulerConfig(n=4, B=4, max_staleness=3, bound_overflow="allow"), SimConfig(topology="elastic", steps=20, slots=32, train_s_base=60.0),
                      SRC, default_shapes(seed, tail_frac=0.2, tail_mult=12.0)).max_staleness() for seed in (1, 3, 5)]
    assert max(loose) == 4
    drops = [simulate(SchedulerConfig(n=4, B=4, max_staleness=3, bound_overflow="drop"), SimConfig(topology="elastic", steps=20, slots=32, train_s_base=60.0),
                      SRC, default_shapes(seed, tail_frac=0.2, tail_mult=12.0)) for seed in (1, 3, 5)]
    assert all(d.max_staleness() <= 3 and all(b <= 4 for b in d.batch_sizes) for d in drops)
    assert sum(d.evicted.get("stale", 0) for d in drops) > 0


def test_validation():
    with pytest.raises(ValueError, match="elastic"):
        sim("elastic", elastic_frac=1.0)


# ----------------------------------------------------------------------------------------------- runtime
async def _run(topology="colocated_partial", *, engines=1, steps=12, vs="call", keep_kv=False, **cfg):
    model = TinyPolicy(32, 48, seed=1)
    engs = [TorchEngine(model, capacity=32, numerics="bf16", decode_delay_s=0.002, version_switch=vs) for _ in range(engines)]
    for e in engs:
        e.resume_keeps_kv = keep_kv
    pool = EnginePool(engs)
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    rt = Runtime(RuntimeConfig(topology=topology, steps=steps, max_turn_tokens=3, version_switch=vs, **cfg), SchedulerConfig(n=4, B=4, max_staleness=3),
                 pool, tr, InprocPublisher(tr, pool), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001, long_tail=0.2),
                 {"copy": [f"p{i}" for i in range(4000)]})
    return rt, await rt.run()


async def test_runtime_elastic_topology():
    rt, rep = await _run("elastic", engines=2, elastic_engines=1, steps=20)
    first, last = (sum(rep.series("reward/mean")[:5]) / 5, sum(rep.series("reward/mean")[-8:]) / 8)
    assert len(rep.steps) == 20 and rep.publishes == 20 and last > first
    assert max(rep.series("staleness/max")) <= 3 and rt.engines.held == set()  # the elastic group is always returned
    for r in rep.trained_records:
        assert all(t.spans_valid() and t.version_spans is None for t in r.turns)  # call-level purity on both groups
    with pytest.raises(ValueError, match="resident generation group"):
        await _run("elastic", engines=1, elastic_engines=1)


@pytest.mark.parametrize("barrier,extra", [("completion_frac", {"barrier_frac": 0.6}), ("token_budget", {"barrier_tokens": 60})])
async def test_runtime_round_boundaries(barrier, extra):
    _, rep = await _run(partial_barrier=barrier, steps=12, **extra)
    sizes = rep.series("batch/groups")
    assert len(rep.steps) == 12 and all(1 <= b <= 4 for b in sizes)
    assert max(rep.series("staleness/max")) <= 3


async def test_runtime_retention_and_kv_resume():
    _, rep = await _run(max_cuts=0, steps=10)
    assert rep.retention_evictions > 0 and len(rep.steps) == 10  # the long tail cut once is evicted, never trained over σ
    _, rep = await _run(vs="token", keep_kv=True, steps=10)
    assert rep.aborted_calls > 0 and rep.kv_resumes == rep.resumed_calls > 0 and rep.reprefill_tokens == 0
