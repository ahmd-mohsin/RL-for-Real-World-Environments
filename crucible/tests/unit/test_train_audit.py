"""Full-tier audit of training rollouts (DESIGN §10.2 / A12; m13 A-6): a deterministic sample is re-graded; a fast-tier
pass the full tier rejects is a confirmed hack — counted in the report, the per-step metrics and the monitor's
confirmed_hack_rate (which nothing fed before), and written into the record. A failing audit is never a verdict."""

from __future__ import annotations

import zlib

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _rt(audit_fn, rate=1.0):
    model = TinyPolicy(32, 24, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2))
    return Runtime(RuntimeConfig(topology="colocated_partial", steps=8, max_turn_tokens=3, warmup_batches=0, audit_rate=rate),
                   SchedulerConfig(n=4, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=4),
                   {"c": [f"p{i}" for i in range(5000)]}, audit_fn=audit_fn)


async def test_audit_confirms_hacks_and_feeds_the_monitor():
    def full_tier(rec):  # the full tier rejects some of what the fast tier paid
        return 0.0 if zlib.crc32(rec.rollout_id.encode()) % 2 else rec.reward

    rt = _rt(full_tier)
    rep = await rt.run()
    assert rep.audits == len(rep.trained_records)  # every trained rollout sampled at rate 1 (audits finish before run() returns)
    assert rep.confirmed_hacks > 0
    assert 0 < sum(rep.series("audit/fast_pass_full_fail")) <= rep.confirmed_hacks  # (audits finishing after the last step are in the total only)
    assert any(v > 0 for v in rep.series("confirmed_hack_rate"))
    assert any(r.reward_components.get("hack/confirmed") == 1.0 for r in rep.trained_records)


async def test_audit_failures_are_infra_not_verdicts():
    def broken(rec):
        raise OSError("full-tier grader container lost")

    rep = await _rt(broken).run()
    assert rep.confirmed_hacks == 0 and rep.failures.get("infra:OSError", 0) > 0 and rep.stop_reason == "steps"


async def test_audit_sampling_is_deterministic_and_rate_bound():
    from crucible.verify.tiers import audit_sample

    ids = [f"r{i}" for i in range(4000)]
    share = sum(audit_sample(i, rate=0.05) for i in ids) / len(ids)
    assert 0.03 < share < 0.07 and [audit_sample(i) for i in ids[:50]] == [audit_sample(i) for i in ids[:50]]
