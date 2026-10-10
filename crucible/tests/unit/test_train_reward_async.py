"""Asynchronous reward with deadlines (M12 checklist #40): a late outcome stays pending and is never trained as 0; a
hard timeout or a crashing grader masks the rollout; a late auxiliary credit channel falls back to the outcome-only
advantage and records ``credit_fallback``; a grading rollout counts as untrained and its group is not selectable."""

from __future__ import annotations

import asyncio
import zlib

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _rt(*, reward_fn=None, aux_fn=None, steps: int = 3, **cfg):
    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    seen = []
    step = tr.step

    def spy(recs, groups):
        seen.extend(recs)
        return step(recs, groups)

    tr.step = spy
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=steps, max_turn_tokens=2, warmup_batches=0, **cfg), SchedulerConfig(n=2, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8), {"copy": [f"p{i}" for i in range(400)]},
                 reward_fn=reward_fn, aux_fn=aux_fn)
    return rt, seen


def _slow(rid: str) -> bool:
    return zlib.crc32(rid.encode()) % 3 == 0


async def test_late_rewards_stay_pending_and_are_never_zero():
    async def grade(rec):
        await asyncio.sleep(0.15 if _slow(rec.rollout_id) else 0.0)
        return 0.75

    rt, seen = _rt(reward_fn=grade, reward_deadline_s=0.05, reward_timeout_s=10)
    rep = await rt.run()
    assert rep.late_rewards > 0 and rep.reward_timeouts == 0 and rep.masked_rollouts == 0
    assert seen and all(r.reward == 0.75 for r in seen)  # every trained reward came from the grader, late or not


async def test_hard_timeout_and_grader_crash_are_masked():
    async def grade(rec):
        if _slow(rec.rollout_id):
            await asyncio.sleep(30)
        if zlib.crc32(rec.rollout_id.encode()) % 3 == 1:
            raise RuntimeError("verifier container died")
        return 1.0

    rt, seen = _rt(reward_fn=grade, reward_timeout_s=0.1, steps=8)
    rep = await rt.run()
    assert rep.reward_timeouts > 0 and rep.masked_rollouts >= rep.reward_timeouts
    assert seen and all(r.reward == 1.0 for r in seen)  # nothing masked reached the trainer (as 0 or otherwise)


async def test_late_aux_channel_falls_back_to_outcome_only():
    async def aux(rec):
        await asyncio.sleep(1.0 if _slow(rec.rollout_id) else 0.0)
        return {t.call_id: 0.25 for t in rec.turns}

    rt, seen = _rt(aux_fn=aux, aux_deadline_s=0.05)
    rep = await rt.run()
    late = [r for r in seen if r.credit_fallback]
    on_time = [r for r in seen if not r.credit_fallback]
    assert rep.credit_fallbacks > 0 and late and on_time
    assert all(t.advantage is None for r in late for t in r.turns)
    assert all(t.advantage == 0.25 for r in on_time for t in r.turns)


async def test_grading_rollouts_are_untrained_and_not_selectable():
    gate = asyncio.Event()

    async def grade(rec):
        await gate.wait()
        return 1.0

    rt, seen = _rt(reward_fn=grade, steps=1)
    run = asyncio.create_task(rt.run())
    for _ in range(200):
        await asyncio.sleep(0.01)
        if sum(1 for t in rt.pool.tags.values() if t.status == "grading") >= 4:
            break
    grading = [t for t in rt.pool.tags.values() if t.status == "grading"]
    assert len(grading) >= 4 and not rt.pool.terminal_groups() and not seen  # nothing selectable, nothing trained
    assert rt.sched.untrained_count() >= len(grading)  # still counted against the admission cap
    gate.set()
    rep = await asyncio.wait_for(run, 30)
    assert len(rep.steps) == 1 and all(r.reward == 1.0 for r in seen)
