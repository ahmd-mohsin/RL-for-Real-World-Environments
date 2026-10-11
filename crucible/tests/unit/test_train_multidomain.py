"""m13 C-6 (A2-53, A2-55): one runtime, several sources, each with its own rollout path, reward and harness; every
record tagged with its source; per-source reward / length / active-time metrics."""

from __future__ import annotations

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


async def fn_rollout(rid, prompt, seed, gen, *, temperature=1.0, harness=""):
    res = await gen([1, 2, 3], 3, temperature)
    return RolloutRecord(rollout_id=rid, sample=rid, task=prompt, harness=harness, policy="native",
                         reward=float(len(res.completion_ids) == 3),
                         turns=[Turn(call_id=f"{rid}:c1", actor="agent", prompt_token_ids=[1, 2, 3], completion_token_ids=list(res.completion_ids),
                                     logprobs=list(res.logprobs), policy_version=res.version)])


async def test_three_sources_one_runtime():
    judged = []

    async def judge(rec):
        judged.append(rec.source)
        return 0.25  # the judge's score replaces the env's

    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    env = lambda: CopyTask(turns=2, symbols=8, latency_s=0.001)
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=6, max_turn_tokens=3),
                 SchedulerConfig(n=2, B=3, max_staleness=2, mix={"copy": 1, "fn": 1, "judged": 1}), eng, tr, InprocPublisher(tr, eng),
                 {"copy": env, "judged": env}, {s: [f"{s}{i}" for i in range(2000)] for s in ("copy", "fn", "judged")},
                 rollout_fn={"fn": fn_rollout}, reward_fn={"judged": judge},
                 harness={"copy": "native:copy@1", "fn": "native:fn@1", "judged": "native:judged@1"})
    rep = await rt.run()
    assert len(rep.steps) == 6
    by = {}
    for r in rep.trained_records:
        by.setdefault(r.source, []).append(r)
    assert set(by) == {"copy", "fn", "judged"}
    assert all(r.harness == f"native:{s}@1" for s, rs in by.items() for r in rs)
    assert all(r.reward == 0.25 for r in by["judged"]) and set(judged) == {"judged"}  # the judge ran for its source only
    assert all("active_s" in r.timings for r in rep.trained_records)
    keys = set().union(*(s.metrics for s in rep.steps))
    for src in ("copy", "fn", "judged"):
        assert {f"reward/mean/{src}", f"len/p99/{src}", f"active_s/p99/{src}", f"records/{src}"} <= keys
