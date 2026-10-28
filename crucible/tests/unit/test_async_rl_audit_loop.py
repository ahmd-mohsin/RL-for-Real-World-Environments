"""The asynchronous-RL audit's fixes on the framework-neutral loop (docs/research/async-rl-audit.md): the scheduler never
puts one group in a batch twice, the staleness heap follows a call served by an older version, implicit π_θold is
refused for stale rollouts, and mimo4's moving bands are trainer state."""

from __future__ import annotations

import copy

import pytest

torch = pytest.importorskip("torch")

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.runtime.lock import jsonable
from crucible.train.engine import TinyPolicy
from crucible.train.mismatch import Correction
from crucible.train.pool import DataPool
from crucible.train.scheduler import Scheduler, SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig


def _finish_all(p: DataPool, version_of=None) -> None:
    for i, rid in enumerate(p.pending()):
        p.start(rid)
        if version_of is not None:
            p.call(rid, version_of(rid))
        p.finish(rid, reward=float(i % 2), now=1.0)


def test_the_relaxed_fill_never_adds_a_group_already_in_the_batch() -> None:
    # the tail of a finite prompt set: 3 groups for B = 4. The relaxed fill re-added a picked group (a:p0 twice)
    cfg = SchedulerConfig(n=2, B=4, max_staleness=4, staleness_strategy="wait")
    p = DataPool()
    q = ["p0", "p1", "p2"]
    s = Scheduler(cfg, p, {"a": lambda: q.pop(0) if q else None})
    s.admit(0)
    _finish_all(p)
    ids = [g.gid for g in s.select(0).groups]
    assert len(ids) == len(set(ids)) == 3 and s.counters["short_tail_batches"] == 1  # every group once; the tail trains short


def test_the_held_back_off_policy_group_fills_the_batch_when_nothing_else_can_arrive() -> None:
    cfg = SchedulerConfig(n=2, B=2, max_staleness=4, staleness_strategy="wait", max_offpolicy_frac=0.0)
    p = DataPool()
    q = ["on", "off"]
    s = Scheduler(cfg, p, {"a": lambda: q.pop(0) if q else None})
    s.admit(1)
    off = next(g for g in p.groups.values() if "off" in g.gid)
    _finish_all(p, version_of=lambda rid: 0 if p.tags[rid].group == off.gid else 1)
    ids = [g.gid for g in s.select(1).groups]
    assert sorted(ids) == sorted([off.gid, next(g for g in p.groups if "on" in g)])  # was ['a:on@0', 'a:on@0']


def test_the_staleness_heap_follows_a_call_served_by_an_older_version() -> None:
    # admitted at version 5 while a remote engine still serves 4 (the publish returned before the load): wait must see 4
    p = DataPool()
    s = Scheduler(SchedulerConfig(n=2, B=1, max_staleness=1, staleness_strategy="wait"), p, {"a": iter(["x"]).__next__})
    g = p.submit_group("a:x@0", "a", 2, 5)
    for r in g.members:
        p.start(r)
        p.call(r, 4)
    assert p.min_live_oldest("a") == p.group_oldest(g.gid) == 4 and s.select(5).blocked == "wait_stale"


def _rec(i: int, versions: list[int], lp: float) -> RolloutRecord:
    turns = [Turn(sampling={"temperature": 1.0}, call_id=f"r{i}:c{k}", actor="agent", prompt_token_ids=[1, 2, 3] if k == 0 else [4],
                  prompt_is_delta=k > 0, completion_token_ids=[5, 6], logprobs=[lp, lp], policy_version=v) for k, v in enumerate(versions)]
    return RolloutRecord(rollout_id=f"r{i}", sample=f"r{i}", task="t", harness="h", policy="p", reward=float(i % 2), turns=turns)


def test_implicit_old_logprobs_are_refused_for_a_stale_single_version_rollout() -> None:
    """π_θold recomputed with the current weights equals π_θ: PPO then ignores the behaviour logprob. The guard refused
    only multi-version rollouts, so a rollout 10 versions stale trained with no off-policy correction."""
    tr = TorchTrainer(TinyPolicy(48, 8, seed=0), TrainerConfig(correction=Correction("ppo")))
    tr.policy_version = 10
    with pytest.raises(ValueError, match="not the served version 10"):
        tr.step([_rec(0, [0, 0], -1.0), _rec(1, [0, 0], -1.0)], ["g", "g"])
    tr.step([_rec(0, [10, 10], -1.0), _rec(1, [10, 10], -1.0)], ["g", "g"])  # on-policy: allowed
    ok = TorchTrainer(TinyPolicy(48, 8, seed=0), TrainerConfig(correction=Correction("ppo", old_logprob_source="recompute_current")))
    ok.policy_version = 10
    ok.step([_rec(0, [0, 0], -1.0), _rec(1, [0, 0], -1.0)], ["g", "g"])  # the explicit choice stays available


def test_mimo4_bands_are_trainer_state_and_survive_a_resume() -> None:
    from crucible.train.state import apply_trainer_state

    corr = Correction("mimo4")
    cfg = TrainerConfig(correction=corr, entropy_adaptive={"low": 5.0, "high": 50.0, "rate": 0.1})
    before = copy.deepcopy(jsonable(cfg))
    tr = TorchTrainer(TinyPolicy(48, 8, seed=0), cfg)
    recs = [_rec(i, [0], -1.0) for i in range(2)]
    tr.step(recs, ["g", "g"])
    assert jsonable(cfg) == before and corr.params == {}  # the run identity is unchanged: a resume is not refused
    assert tr.bands and tr.bands["pos_high"] > 5.0
    fresh = TorchTrainer(TinyPolicy(48, 8, seed=0), TrainerConfig(correction=Correction("mimo4"), entropy_adaptive=cfg.entropy_adaptive))
    apply_trainer_state(fresh, {"model": tr.model.state_dict(), "opt": tr.opt.state_dict(), "updates": tr.updates, "bands": dict(tr.bands)})
    assert fresh.bands == tr.bands
