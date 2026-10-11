"""m13 G-13 / G-14. The Judge ranks accepted variants by quality and diversity, respects the diversity quota, and never
changes an M9 verdict (random pools). A family specialist re-finds a known hack on a second task of the family with
zero expensive-hacker episodes (replayed from memory), after one episode on the first."""

from __future__ import annotations

import random
import tempfile
from pathlib import Path

import pytest

from crucible.forge.economy import StopRules, YieldBook
from crucible.forge.judge import Variant, quality, select


def test_judge_never_changes_verdicts_and_prefers_diverse_in_band():
    rng = random.Random(0)
    for _ in range(200):
        vs = [Variant(f"t{i}", f"s{rng.randrange(3)}", rng.choice(["chain", "fuse", "nest"]), rng.random() < 0.6, rng.random(), rng.randrange(8), rng.randrange(3))
              for i in range(rng.randrange(1, 12))]
        before = {v.task_id: v.accepted for v in vs}
        picked = select(vs, rng.randrange(1, 6))
        assert {v.task_id: v.accepted for v in vs} == before  # verdicts untouched
        assert all(before[t] for t in picked) and len(set(picked)) == len(picked)
    a, b, c = Variant("a", "s1", "chain", True, 0.4), Variant("b", "s1", "chain", True, 0.38), Variant("c", "s2", "fuse", True, 0.3)
    assert quality(a) > quality(c) and select([a, b, c], 2) == ["a", "c"]  # diversity beats a near-duplicate
    assert select([Variant("x", "s", "chain", False, 0.4)], 3) == []


def test_judge_respects_the_quota():
    book = YieldBook(StopRules(quota_floor=4, max_share=0.3))
    for i, op in enumerate(["chain", "fuse", "nest", "stitch"]):
        book.record_construction(f"k{i}", "g", operator=op, seed=f"s{i}", family="toy")
        book.record_acceptance(f"k{i}", accepted=True)
        book.admit(f"k{i}")
    book.record_construction("new_chain", "g", operator="chain", seed="s9", family="toy")
    book.record_acceptance("new_chain", accepted=True)
    vs = [Variant("new_chain", "s9", "chain", True, 0.4), Variant("other", "s8", "perturb", True, 0.2)]
    assert select(vs, 2, book=book) == ["other"]  # a 2nd chain of 5 would hold 40% > 30%


@pytest.fixture(scope="module")
def weak_tasks():
    import asyncio

    from crucible_pack_toy.forge.task import ToyForgedTask
    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    out = {}
    for t, _ in col:
        name = t.task_id.split("/")[-1]
        if name in ("t2_approval", "t4_decoy"):
            cs = t.cs.model_copy(update={"checks": [c for c in t.cs.checks if c.id != "policy/approval_given_in_episode"]})
            out[name] = ToyForgedTask(t.spec, cs, t.oracle, t.negatives, pack_root=t.pack_root, world_file=t.world_file, scenario=t.sc, canary=t.canary)
    return out


async def test_specialist_replays_known_hacks_at_zero_episodes(weak_tasks, tmp_path):
    from crucible.accept.policy import Policy
    from crucible.accept.run import accept_task
    from crucible.forge.hacker import MutationHacker
    from crucible.forge.specialists import FamilyMemory, SpecialistHacker

    mem = FamilyMemory(tmp_path / "memory.jsonl")

    class CostlyHacker(MutationHacker):  # stands in for an LLM hacker: every propose is a paid episode
        pass

    async def run(task):
        sp = SpecialistHacker("toy/helpdesk_v2", mem, fallback=CostlyHacker())
        task.hacker = lambda: sp
        r = await accept_task(task, policy=Policy(required=["A16"]), cache_root=Path(tempfile.mkdtemp()), only=["A16"], reuse=False)
        return sp, r.results["A16"]

    sp1, a1 = await run(weak_tasks["t2_approval"])
    assert sp1.episodes == 1 and sp1.source == "fallback" and a1.measured["confirmed_hacks"] >= 1
    mem.remember("toy/helpdesk_v2", "t2_approval", sorted({h["strategy"] for h in a1.measured["hacks"]}))
    sp2, a2 = await run(weak_tasks["t4_decoy"])
    assert sp2.episodes == 0 and sp2.source == "memory" and a2.measured["confirmed_hacks"] >= 1  # found again, no hacker episode
    assert FamilyMemory(tmp_path / "memory.jsonl").strategies("toy/helpdesk_v2") == ["skip_interactions"]
