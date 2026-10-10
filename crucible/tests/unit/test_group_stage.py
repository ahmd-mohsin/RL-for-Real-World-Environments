"""m13 C-2 / C-3: the group grading stage (``Runtime(group_fn=...)``) and its first operators, MiMo GAR and GRS."""

from __future__ import annotations

import asyncio
import random

import pytest

from crucible.credit.gar import check_rubric_grounding, gar, gar_group_fn, grs
from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.mismatch import Correction
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def test_gar_golden_case():
    adv, r = gar([1, 1, 0, 0], [1.0, 0.5, None, None], lam=1.0)
    assert r == [1.0, 1.0, 0.0, 0.0]
    assert adv == pytest.approx([2 / 3, 1 / 3, -0.5, -0.5], abs=1e-12)  # passing mass 1.0 split 1 : 0.5
    adv, _ = gar([1, 1, 0, 0], [1.0, 0.5, None, None], lam=0.5)
    assert adv == pytest.approx([0.5 + 0.5 * (2 / 3 - 0.5), 0.5 + 0.5 * (1 / 3 - 0.5), -0.5, -0.5], abs=1e-12)  # λ caps the move


def test_gar_conserves_mass_and_leaves_failures_and_uniform_groups():
    rng = random.Random(0)
    for _ in range(500):
        n = rng.randint(2, 16)
        rew = [rng.choice([0.0, 1.0, None, rng.random()]) for _ in range(n)]
        q = [rng.uniform(0.05, 1.0) for _ in range(n)]
        base, _ = gar(rew, q, lam=0.0)
        adv, _ = gar(rew, q, lam=rng.random())
        P = [i for i, x in enumerate(rew) if x is not None and x >= 1.0]
        assert abs(sum(adv[i] for i in P) - sum(base[i] for i in P)) < 1e-9  # passing mass conserved
        assert all(adv[i] == base[i] for i, x in enumerate(rew) if x is not None and x < 1.0)  # failures unchanged
        assert abs(sum(a for a in adv if a is not None)) < 1e-9  # still centred
        assert all((a is None) == (x is None) for a, x in zip(adv, rew, strict=True))  # masked stays masked
    assert gar([1, 1, 1], [0.1, 0.5, 1.0], lam=1.0)[0] == [0.0, 0.0, 0.0]  # all-pass: untouched
    assert gar([0, 0], [None, None], lam=1.0)[0] == [0.0, 0.0]  # all-fail: untouched
    with pytest.raises(ValueError):
        gar([1, 0], [0.0, None])  # quality must be in (0, 1]


def test_gar_zeroes_hacks_then_recomputes_the_group():
    adv, r = gar([1, 1, 0, 0], [1.0, 1.0, None, None], [False, True, False, False], lam=1.0)
    assert r == [1.0, 0.0, 0.0, 0.0] and adv == pytest.approx([0.75, -0.25, -0.25, -0.25])


def test_grs_and_rubric_grounding():
    assert grs(0.0, 0.9, 0.9) == 0.0 and grs(None, 1, 1) is None and grs(1.0, 0.5, 0.8) == pytest.approx(0.4)
    with pytest.raises(ValueError):
        grs(1.0, 1.2, 1.0)
    items = ["the fix touches only the parser module", "like rollout 3, it adds a cache", "mirrors r-abc123's approach"]
    assert check_rubric_grounding(items, ["r-abc123"]) == items[1:]


def _rt(group_fn, *, deadline=30.0, steps=6):
    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    return Runtime(RuntimeConfig(topology="colocated_partial", steps=steps, max_turn_tokens=3, group_deadline_s=deadline),
                   SchedulerConfig(n=4, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng),
                   lambda: CopyTask(turns=2, symbols=8, latency_s=0.001), {"copy": [f"p{i}" for i in range(4000)]}, group_fn=group_fn)


async def test_runtime_applies_gar_through_the_group_stage():
    seen = []

    async def quality(passing):
        seen.append(len(passing))
        return [1.0 / (k + 1) for k in range(len(passing))]

    rep = await _rt(gar_group_fn(quality, lam=1.0, pass_threshold=0.5)).run()
    assert len(rep.steps) == 6 and sum(s.metrics["group/graded"] for s in rep.steps) == 12 and rep.series("group/fallback") == [0.0] * 6
    recs = rep.trained_records
    assert all(r.advantage is not None and "raw" in r.reward_components for r in recs)
    by: dict[str, list] = {}
    for r in recs:
        by.setdefault(r.group, []).append(r)
    for g in by.values():  # every trained group is centred after the stage
        assert abs(sum(r.advantage for r in g)) < 1e-9


def test_trainer_uses_the_stage_advantage():
    from crucible.train.rollout import RolloutRecord, Turn

    tr = TorchTrainer(TinyPolicy(32, 16, seed=0), TrainerConfig(item_unit="call"))
    mk = lambda rid, rw, adv: RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=rw, advantage=adv,
                                            turns=[Turn(call_id=f"{rid}:c1", actor="a", prompt_token_ids=[1, 2], completion_token_ids=[3], logprobs=[-0.1], policy_version=0)])
    items = tr._items([mk("a", 1.0, 0.9), mk("b", 0.0, -0.9)], ["g", "g"])
    assert [it.adv for it in items] == [0.9, -0.9]
    items = tr._items([mk("a", 1.0, 0.9), mk("b", 0.0, None)], ["g", "g"])  # partial override: the group statistic
    assert [it.adv for it in items] == [0.5, -0.5]


async def test_late_or_broken_stage_falls_back_to_raw_advantages():
    async def slow(gid, members):
        await asyncio.sleep(5)
        return members

    rep = await _rt(slow, deadline=0.05, steps=4).run()
    assert len(rep.steps) == 4 and sum(rep.series("group/fallback")) == 8 and sum(rep.series("group/graded")) == 0
    assert all(r.group_credit_fallback and r.advantage is None for r in rep.trained_records)

    def broken(gid, members):
        raise RuntimeError("grader crashed")

    rep = await _rt(broken, steps=3).run()
    assert len(rep.steps) == 3 and rep.failures.get("harness:RuntimeError") == 6 and rep.stop_reason == "steps"
