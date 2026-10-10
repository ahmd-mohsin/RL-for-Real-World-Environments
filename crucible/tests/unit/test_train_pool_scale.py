"""Pool hot path and memory (m13 audit 05): admission counts untrained rollouts from an index (cost flat in the run's
history), trained / evicted work is released, finished tasks are pruned, and the report keeps a bounded record window."""

from __future__ import annotations

import time

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.pool import DataPool
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import Scheduler, SchedulerConfig
from crucible.train.weights import InprocPublisher


def _admit_cost(history: int) -> float:
    pool = DataPool()
    ids = iter(range(10**7))
    s = Scheduler(SchedulerConfig(n=4, B=4, max_staleness=2, staleness_strategy="none"), pool, {"x": lambda: f"p{next(ids)}"})
    for _ in range(history // 16):  # run history: admit a cap's worth, finish and train it, release it
        gs = s.admit(0)
        for g in gs:
            for r in g.members:
                pool.finish(r, reward=1.0)
            pool.mark_trained(g.gid)
            pool.release(g.gid)
    t0 = time.perf_counter()
    for _ in range(200):
        assert s.untrained_count() == 0 and s.free_slots("x") > 0
    return time.perf_counter() - t0


def test_admission_cost_is_flat_in_history():
    small, big = _admit_cost(160), _admit_cost(50_000)
    assert big < 5 * small + 1e-3, (small, big)


async def test_long_run_memory_is_bounded():
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    sc = SchedulerConfig(n=2, B=2, max_staleness=2)
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=60, max_turn_tokens=2, warmup_batches=0, keep_records=16), sc,
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=1), {"c": [f"p{i}" for i in range(5000)]})
    rep = await rt.run()
    assert len(rep.steps) == 60
    cap = sc.max_inflight()
    assert len(rt.pool.tags) <= 2 * cap and len(rt.pool.groups) <= 2 * cap // sc.n  # open work only, not 60 steps of history
    assert rt.pool.released_groups >= 60 * sc.B and len(rt.tasks) <= cap
    assert len(rep.trained_records) == 16
