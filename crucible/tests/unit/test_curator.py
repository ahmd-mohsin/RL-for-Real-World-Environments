"""m13 G-12: the Curator. A task whose L2 rises above 0.7 in training is retired and re-seeded; the re-composition loop
yields a composite back in the train band (graded by the real grader: the simulated policy solves each component with
its own probability). Placement, dedup and lineage sealing (no train task derives from a bench lineage)."""

from __future__ import annotations

import random

import pytest

from crucible.forge.compose import Parts, chain
from crucible.forge.curator import Curator
from crucible.forge.dedup import Candidate
from crucible.forge.ladder import Rung, run_ladder

# the training policy's per-component skill (after training, t1 is nearly solved)
SKILL = {"t1_duplicate": 0.92, "t3_in_transit": 0.97, "t5_compose": 0.5}


@pytest.fixture(scope="module")
def forged():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return {t.task_id.split("/")[-1]: t for t, _ in col}


def _l2(task, parts, seed):
    """The L2 rung on a (composite) task: each component solved with its skill, failed components' actions dropped."""
    rng = random.Random(seed)

    async def run(n):
        out = []
        for _ in range(n):
            acts = []
            if parts is None:
                acts = list(task.oracle) if rng.random() < SKILL[task.task_id.split("/")[-1]] else []
            else:
                for c in task.composite.components:
                    if rng.random() < SKILL[c.source.split("/")[-1]]:
                        acts += task.composite.component_oracle(c.id)
            rep, _ = await task.grade(acts)
            out.append(rep.reward == 1.0)
        return out

    return Rung("L2", run)


async def test_retired_task_is_recomposed_back_into_the_band(forged):
    from crucible_pack_toy.forge.composed import ToyComposedTask

    cur = Curator(max_rounds=3)
    t1 = forged["t1_duplicate"]
    first = cur.place(t1.task_id, await run_ladder([_l2(t1, None, 1)]), lineage=["seed:t1"])
    assert first.action == "hold" and first.steer == "compose_one_more"  # 0.92 is above the band already
    partners = ["t3_in_transit", "t5_compose"]  # round 0 pairs with an easy task (still too easy), round 1 with a harder one

    async def compose(task_id, rnd):
        b = forged[partners[rnd]]
        comp = chain(Parts.of(t1), Parts.of(b), task_id=f"toy/helpdesk_v2/chain_t1_{partners[rnd]}")
        t = ToyComposedTask(comp, [t1, b])
        rep = await run_ladder([_l2(t, comp, 10 + rnd)])
        return comp.spec.task_id, rep, [f"seed:{partners[rnd]}"], Candidate(comp.spec.task_id, instruction=comp.spec.instruction)

    d = await cur.recompose(t1.task_id, 0.92, compose)
    assert d.action == "train" and d.parent == t1.task_id and d.round == 1, cur.report()
    assert 0.1 <= d.rates["L2"] <= 0.7 and d.lineage == ["seed:t1", "seed:t5_compose"]
    rep = cur.report()
    assert rep["retired"] == [t1.task_id] and [r["round"] for r in rep["recomposed"]] == [0, 1] and rep["recomposed"][0]["action"] == "hold"
    assert await cur.recompose("x", 0.5, compose) is not None and "x" not in cur.decisions  # not retired: nothing happens


def _rep(**rates):
    from crucible.forge.ladder import LadderReport, rate_of

    r = LadderReport()
    for k, (succ, n) in rates.items():
        r.rates[k] = rate_of([True] * succ + [False] * (n - succ), k)
    return r


def test_placement_dedup_and_lineage_sealing():
    cur = Curator()
    assert cur.place("b1", _rep(L3=(0, 16)), lineage=["seed:A"], prefer="bench").action == "bench"
    d = cur.place("t_from_A", _rep(L2=(6, 16)), lineage=["seed:A", "seed:B"])
    assert d.action == "hold" and d.steer == "lineage_sealed" and "sealed to bench" in d.reasons[0]
    assert cur.place("t2", _rep(L2=(6, 16)), lineage=["seed:B"], candidate=Candidate("t2", instruction="Refund ticket 7")).action == "train"
    dup = cur.place("t3", _rep(L2=(6, 16)), lineage=["seed:C"], candidate=Candidate("t3", instruction="refund   ticket 7"))
    assert dup.action == "reject" and "duplicate of t2" in dup.reasons[0]
    hard = cur.place("t4", _rep(L2=(0, 16)), lineage=["seed:D"])
    assert hard.action == "hold" and hard.steer == "drop_outermost"
    inv = _rep(L1=(16, 16), L3=(0, 16))
    from crucible.forge.ladder import monotonicity

    inv.inversions = monotonicity(inv.rates)
    assert cur.place("t5", inv).action == "reject" and cur.decisions["t5"].steer == "fix_grader"
    assert cur.report()["train"] == ["t2"] and cur.report()["sealed_lineages"] == {"seed:A": "bench", "seed:B": "train"}
