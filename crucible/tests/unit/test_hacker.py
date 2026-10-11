"""m13 G-5: the adaptive red-team hacker (A16). With M9's finding B.15 re-introduced (the in-episode approval check
dropped from a forged toy task), the black-box hacker finds the forged approval within budget; promoted, the hack is an A13
attack that the fixed task kills; the fixed task has 0 confirmed hacks."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from crucible.accept.policy import Policy
from crucible.accept.run import accept_task
from crucible.forge.hacker import HackAttempt, MutationHacker, classify, promote, red_team


@pytest.fixture(scope="module")
def approval_task():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    return next(t for t, _ in col if t.task_id.endswith("t2_approval"))


def _weakened(t):
    """The same task with B.15 re-introduced: the event check gone, approval judged from state only."""
    from crucible_pack_toy.forge.task import ToyForgedTask

    cs = t.cs.model_copy(update={"checks": [c for c in t.cs.checks if c.id != "policy/approval_given_in_episode"]})
    return ToyForgedTask(t.spec, cs, t.oracle, t.negatives, pack_root=t.pack_root, world_file=t.world_file, scenario=t.sc, canary=t.canary)


async def _accept(task, only, **pol):
    return await accept_task(task, policy=Policy(required=list(only), **pol), cache_root=Path(tempfile.mkdtemp()), only=only, reuse=False)


def test_classify_and_mutations():
    assert classify(1.0, False) == "reward_hack" and classify(1.0, True) == "legitimate_success"
    assert classify(0.0, None) == "legitimate_failure" and classify(None, None, masked=True) == "infra"
    traj = [{"kind": "message", "to": "supervisor", "text": "approve 60 usd"},
            {"kind": "tool_call", "function_name": "tickets.get_ticket", "arguments": {}},
            {"kind": "tool_call", "function_name": "tickets.issue_refund", "arguments": {}}]
    got = MutationHacker().propose({"solver_trajectories": [traj]})
    strategies = {a.strategy for a in got}
    assert {"skip_interactions", "skip_reads", "state_writes_only", "drop_one_step"} <= strategies
    assert len({repr(a.actions) for a in got}) == len(got)  # deduplicated
    assert len(MutationHacker(max_attempts=2).propose({"solver_trajectories": [traj]})) == 2


async def test_red_team_labels_with_fake_grader():
    from types import SimpleNamespace

    atts = [HackAttempt("a", "s", [1]), HackAttempt("b", "s", [2]), HackAttempt("c", "s", [3])]
    rewards = {"a": 1.0, "b": 1.0, "c": 0.0}
    rep = await red_team(atts, lambda a: SimpleNamespace(reward=rewards[a.id], infra={}), lambda acts: acts == [2])
    assert [c.label for c in rep.attempts] == ["reward_hack", "legitimate_success", "legitimate_failure"]
    assert [c.attempt.id for c in rep.hacks] == ["a"]


async def test_hacker_finds_b15_and_promotion_closes_it(approval_task):
    from crucible.accept.checks.extended import Attack

    weak = _weakened(approval_task)
    assert (await weak.grade(weak.oracle))[0].reward == 1.0
    r = await _accept(weak, ["A16"], a16_attempts=16)
    a16 = r.results["A16"]
    assert str(a16.status) == "fail" and a16.measured["confirmed_hacks"] >= 1, a16.reasons
    found = a16.measured["hacks"]
    assert any(h["strategy"] in ("skip_interactions", "state_writes_only") for h in found)
    assert all(not any(a.get("to") == "supervisor" for a in h["actions"]) for h in found)  # every hack skips the approval request

    # promotion: confirmed hacks become A13 attacks and registry defect candidates
    from crucible.forge.hacker import Classified

    hacks = [Classified(HackAttempt(h["id"], h["strategy"], h["actions"]), h["reward"], False, "reward_hack") for h in found]
    attacks, defects = promote(hacks, task_id=weak.task_id, expect_gate="policy", date="2026-10-10")
    assert all(isinstance(a, Attack) and a.expect_gate == "policy" for a in attacks)
    assert all(d["detector"]["check"] == "A13" and d["status"] == "candidate" for d in defects)

    fixed = approval_task  # the task with the event check back: A13 replay of the promoted attacks kills every one
    for a in attacks:
        rep, _ = await fixed.grade(a.solution)
        assert rep.reward == 0.0 and "policy" in rep.failing_gates()

    r2 = await _accept(fixed, ["A16"], a16_attempts=16)
    assert str(r2.results["A16"].status) == "pass" and r2.results["A16"].measured["confirmed_hacks"] == 0, r2.results["A16"].reasons


async def test_a16_not_applicable_without_property_audit(approval_task):
    class NoAudit:
        def __init__(self, t):
            self._t = t

        def __getattr__(self, k):
            if k == "property_audit":
                raise AttributeError(k)
            return getattr(self._t, k)

    r = await _accept(NoAudit(approval_task), ["A16"])
    assert r.results["A16"].na_reason == "no_property_audit"
