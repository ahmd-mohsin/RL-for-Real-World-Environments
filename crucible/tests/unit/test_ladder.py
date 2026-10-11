"""m13 G-4: the solver ladder: sequential rung sampling, unbiased pass@k, monotonicity (an inversion blocks acceptance),
acceptance, steering and retirement; graded on a real forged toy task."""

from __future__ import annotations

import random

import pytest

from crucible.forge.ladder import Rung, accept, pass_at_k, retire, run_ladder, steer


def test_pass_at_k_is_the_unbiased_estimator():
    assert pass_at_k(10, 0, 8) == 0.0 and pass_at_k(10, 10, 1) == 1.0 and pass_at_k(10, 3, 8) == 1.0  # n - c < k
    assert pass_at_k(10, 1, 1) == pytest.approx(0.1) and pass_at_k(20, 5, 4) == pytest.approx(1 - (15 * 14 * 13 * 12) / (20 * 19 * 18 * 17))


@pytest.fixture(scope="module")
def task():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return next(t for t, _ in col if t.task_id.endswith("t1_duplicate"))


def _rung(task, name, p, *, seed, k=1):
    """A policy that solves the task with probability p (the oracle) and otherwise does nothing: graded by the real grader."""
    rng = random.Random(seed)

    async def run(n):
        out = []
        for _ in range(n):
            rep, _ = await task.grade(task.oracle if rng.random() < p else [])
            out.append(rep.reward == 1.0)
        return out

    return Rung(name, run, k=k)


async def test_ladder_on_a_real_task_accepts_and_steers(task):
    rep = await run_ladder([_rung(task, "L1", 0.1, seed=1), _rung(task, "L2", 0.4, seed=2), _rung(task, "L3", 0.8, seed=3), _rung(task, "L4", 0.8, seed=4, k=8)])
    assert rep.ok() and set(rep.rates) == {"L1", "L2", "L3", "L4"} and rep.rates["L1"].p <= rep.rates["L3"].p
    assert 0.0 <= rep.pass_at_k["L4"] <= 1.0 and rep.pass_at_k["L4"] >= rep.rates["L4"].p
    ok, why = accept(rep, split="train")
    assert ok == (0.1 <= rep.rates["L2"].p <= 0.7), why
    easy = await run_ladder([_rung(task, "L2", 1.0, seed=5)])
    assert steer(easy) == "compose_one_more" and retire(0.9) and not retire(0.5)
    hard = await run_ladder([_rung(task, "L2", 0.0, seed=6)])
    assert steer(hard) == "drop_outermost"


async def test_an_inversion_blocks_acceptance(task):
    async def flaky_cheap(n):  # a style-grading bug: the cheap rung "passes" without solving
        return [True] * n

    rep = await run_ladder([Rung("L1", flaky_cheap), _rung(task, "L3", 0.0, seed=7)])
    assert rep.inversions and "flaky grader" in rep.inversions[0]
    ok, why = accept(rep, split="bench")
    assert not ok and any("flaky grader" in w for w in why) and steer(rep) == "fix_grader"


async def test_a10_blocks_on_a_ladder_inversion(task):
    import tempfile
    from pathlib import Path

    from crucible.accept.policy import Policy
    from crucible.accept.run import accept_task

    async def flaky_cheap(n):
        return [True] * n

    inverted = await run_ladder([Rung("L1", flaky_cheap), _rung(task, "L3", 0.0, seed=8)])
    task.calibration_runs = lambda: [{"policy": "llm:m", "reward": 1.0 if i < 4 else 0.0, "failing": [] if i < 4 else ["outcome"]} for i in range(10)]
    task.ladder = lambda: inverted
    try:
        r = await accept_task(task, policy=Policy(required=["A10"]), cache_root=Path(tempfile.mkdtemp()), only=["A10"], reuse=False)
        res = r.results["A10"]
        assert str(res.status) == "fail" and any("solver ladder inversion" in x for x in res.reasons) and "ladder" in res.measured
    finally:
        del task.ladder
