"""M12 audit (2026-10-12) regression tests for the `train` control and data planes: runtime, scheduler, pool, simulator,
monitors, admission and the environment worker pool. Each test is named after its finding (H1/H2/H3 study defects, G1-G4
framework sweeps, ``docs/audit/2026-10-12/``) and fails on the M11 commit (3b522cb).

The runtime tests drive the real ``Runtime`` / ``Scheduler`` / ``DataPool`` with a small token engine (version pinned per
call, cooperative abort, pause), a fake trainer that checks the training invariants on every step (no double train, no
masked record, staleness within τ), and adversarial environments."""

from __future__ import annotations

import asyncio
import itertools
import os
import random
import subprocess
import sys
import zlib
from collections import defaultdict

import pytest

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import EnginePool, GenRequest, GenResult
from crucible.train.envs import EnvInfraError, Step
from crucible.train.monitors import StepMonitor
from crucible.train.pool import DataPool
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import Scheduler, SchedulerConfig
from crucible.train.sim import SimConfig, simulate


# ----------------------------------------------------------------------------------------------- harness
class Eng:
    """Token-by-token engine: version pinned at call start, cooperative abort, pause; records pause / abort order."""

    def __init__(self, seed: int = 1, *, tok_s: float = 0.0005, fail=None, always_abort: bool = False, exc=ConnectionError, caps=None):
        self.exc, self.caps, self.reqs = exc, caps or {"token_ids": True}, []
        self.version, self.seed, self.tok_s, self.fail, self.always_abort, self.capacity = 0, seed, tok_s, fail, always_abort, 64
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._epoch = 0
        self.inflight = 0
        self.log: list[str] = []

    async def generate(self, req: GenRequest) -> GenResult:
        await self._unpaused.wait()
        self.reqs.append(req)
        if self.fail is not None and self.fail(req.call_id):
            raise self.exc("engine connection reset")
        if self.always_abort:  # an engine error the gateway turns into "abort" (H3 V9)
            await asyncio.sleep(0)
            return GenResult([], [], self.version, "abort", None, len(req.prompt_ids))
        self.inflight += 1
        epoch, version = self._epoch, self.version
        out, lps = list(req.resume_completion), list(req.resume_logprobs)
        r = random.Random(zlib.crc32(f"{self.seed}|{req.call_id}|{len(out)}".encode()))
        try:
            while len(out) < req.max_tokens:
                if self._epoch != epoch:
                    return GenResult(out, lps, version, "abort", None, len(req.prompt_ids))
                out.append(r.randint(4, 31))
                lps.append(-1.0)
                await asyncio.sleep(self.tok_s * (20 if r.random() < 0.05 else 1))
            return GenResult(out, lps, version, "length", None, len(req.prompt_ids))
        finally:
            self.inflight -= 1

    async def abort_all(self) -> int:
        self.log.append("abort")
        self._epoch += 1
        return self.inflight

    async def pause(self) -> None:
        self.log.append("pause")
        self._unpaused.clear()

    async def resume(self) -> None:
        self._unpaused.set()

    def capabilities(self):
        return dict(self.caps)

    async def load_weights(self, state, version: int) -> None:
        self.version = version


class Env:
    """Random turn count, long tails, and optional infra masks, crashes, NaN rewards and constant rewards."""

    def __init__(self, seed: int, *, p_infra=0.0, p_crash=0.0, p_nan=0.0, lat=0.001, tail=0.1, max_turns=2, const=False):
        self.seed, self.p_infra, self.p_crash, self.p_nan, self.lat, self.tail, self.max_turns, self.const = seed, p_infra, p_crash, p_nan, lat, tail, max_turns, const

    async def reset(self, prompt, seed):
        self.r = random.Random(f"{self.seed}|{prompt}|{seed}")
        self.turns, self.k = self.r.randint(1, self.max_turns), 0
        await asyncio.sleep(self.lat)
        return [1, 5, 2]

    async def step(self, completion):
        await asyncio.sleep(self.lat * (30 if self.r.random() < self.tail else 1))
        x = self.r.random()
        if x < self.p_infra:
            raise EnvInfraError("sandbox lost")
        if x < self.p_infra + self.p_crash:
            raise ValueError("env bug")
        self.k += 1
        if self.k >= self.turns:
            if self.r.random() < self.p_nan:
                return Step([], True, float("nan"))
            return Step([], True, 1.0 if self.const else float(self.r.random() < 0.5))
        return Step([2, 6, 2], False)


class Trainer:
    """Checks the invariants on every step."""

    def __init__(self):
        self.rt, self.seen, self.violations, self.updates = None, set(), [], 0

    def step(self, recs, groups):
        cfg = self.rt.sched_cfg
        for r in recs:
            if r.rollout_id in self.seen:
                self.violations.append(f"double-train {r.rollout_id}")
            self.seen.add(r.rollout_id)
            if r.mask_sample or r.reward is None:
                self.violations.append(f"masked trained {r.rollout_id}")
            vs = [v for t in r.turns for v in ([v for _, v in t.version_spans] if t.version_spans else [t.policy_version])]
            if vs and cfg.staleness_strategy != "none" and self.rt.version - min(vs) > cfg.tau(r.source or ""):
                self.violations.append(f"stale {r.rollout_id}")
        lost = [t.rid for t in self.rt.pool.tags.values() if t.status in ("running", "grading") and t.rid not in self.rt.tasks]
        if lost:
            self.violations.append(f"lost {lost[:3]}")
        self.updates += 1
        return {"train/loss": 0.0}


class Pub:
    def __init__(self, engines):
        self.engines, self.version = engines, 0

    async def publish(self):
        self.version += 1
        await self.engines.load_weights(None, self.version)
        return self.version

    async def restore(self, version):
        self.version = version


def build(seed=1, *, rcfg=None, scfg=None, env=None, eng=None, sources=None, **kw):
    eng = EnginePool([eng or Eng(seed)])
    tr = Trainer()
    rc = {"topology": "colocated_partial", "max_turn_tokens": 3, "poll_s": 0.001, "warmup_batches": 0, "max_wall_s": 8.0, **(rcfg or {})}
    rt = Runtime(RuntimeConfig(**rc), SchedulerConfig(**{"n": 4, "B": 2, "max_staleness": 2, **(scfg or {})}), eng, tr, Pub(eng),
                 (lambda: Env(seed, **(env or {}))) if "rollout_fn" not in kw else None, sources or {"a": [f"a{i}" for i in range(100000)]}, **kw)
    tr.rt = rt
    return rt, tr


def record(rid, res, *, reward=1.0, rollout_id=None, actor="agent", **kw):
    return RolloutRecord(rollout_id=rollout_id or rid, sample=rid, task="t", harness="h", policy="p", reward=reward, actor=actor,
                         turns=[Turn(call_id=f"{rid}:c1", actor=actor, prompt_token_ids=[1, 2, 3], completion_token_ids=res.completion_ids,
                                     logprobs=res.logprobs, policy_version=res.version)], **kw)


def no_lost(rt):
    """No rollout left live without its task (checked by the trainer at every step) and every masked rollout named."""
    return not [v for v in rt.trainer.violations if v.startswith("lost")] and sum(rt.report.masked_kinds.values()) == rt.report.masked_rollouts


# ----------------------------------------------------------------------------------------------- H2 V1: nothing is lost
async def test_H2_V1_env_crash_is_masked_counted_and_the_run_progresses():
    rt, tr = build(1, rcfg={"steps": 8, "max_harness_share": 1.0}, env={"p_crash": 0.05})
    rep = await rt.run()
    assert len(rep.steps) == 8 and rep.stop_reason == "steps" and not tr.violations
    assert rep.failures.get("harness:ValueError", 0) > 0 and rep.masked_kinds.get("harness:ValueError") == rep.failures["harness:ValueError"]
    assert no_lost(rt)


async def test_H2_V1_sync_reward_fn_and_nan_rewards_mask_instead_of_losing():
    rt, tr = build(2, rcfg={"steps": 5}, scfg={"n": 2}, env={"max_turns": 1}, reward_fn=lambda rec: 1.0)  # synchronous grader
    rep = await rt.run()
    assert len(rep.steps) == 5 and rep.stop_reason == "steps" and all(r.reward == 1.0 for r in rep.trained_records)

    async def nan_half(rec):
        return float("nan") if zlib.crc32(rec.rollout_id.encode()) % 2 else 1.0

    rt, tr = build(2, rcfg={"steps": 4, "max_masked_share": 1.0}, scfg={"n": 2}, env={"max_turns": 1}, reward_fn=nan_half)
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.masked_kinds.get("grader_nonfinite", 0) > 0 and not tr.violations and no_lost(rt)
    rt, tr = build(3, rcfg={"steps": 4, "max_masked_share": 1.0}, scfg={"n": 2}, env={"max_turns": 1, "p_nan": 0.3})
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.masked_kinds.get("crucible:env_nonfinite_reward", 0) > 0 and no_lost(rt)


async def test_H2_V1_engine_transport_error_is_masked_as_infra():
    rt, _tr = build(2, rcfg={"steps": 5, "max_masked_share": 1.0}, scfg={"n": 2}, eng=Eng(2, fail=lambda cid: zlib.crc32(cid.encode()) % 23 == 0))
    rep = await rt.run()
    assert len(rep.steps) == 5 and rep.failures.get("infra:ConnectionError", 0) > 0
    assert rep.masked_kinds.get("infra:ConnectionError") == rep.failures["infra:ConnectionError"] and no_lost(rt)


# ----------------------------------------------------------------------------------------------- H2 V2, V7, V16: group stage
async def test_H2_V2_refilled_members_go_through_the_group_stage():
    def stage(gid, members):
        for m in members:
            m.reward_components["stage/ran"] = 1.0
            m.reward = 0.123
        return members

    rt, _tr = build(5, rcfg={"steps": 10, "breaker_min": 10**9}, scfg={"group_backfill": "refill_member"}, env={"p_infra": 0.2}, group_fn=stage)
    rep = await rt.run()
    by = defaultdict(set)
    for r in rep.trained_records:
        by[r.group].add(r.reward_components.get("stage/ran"))
    assert len(rep.steps) == 10 and rt.sched.counters["refilled_members"] > 0
    assert all(v == {1.0} for v in by.values()), {g: v for g, v in by.items() if v != {1.0}}
    assert rep.restaged_groups > 0


async def test_H2_V7_group_stage_writes_back_by_position_not_rollout_id():
    async def fn(rid, prompt, seed, gen, temperature=1.0, harness=""):
        res = await gen([1, 2, 3], 3, temperature)
        return record(rid, res, rollout_id=f"{rid}-a1", reward=float(zlib.crc32(rid.encode()) % 2))  # a World record's id

    def stage(gid, members):
        for m in members:
            m.reward_components["stage/ran"] = 1.0
        return members

    rt, _tr = build(1, rcfg={"steps": 2}, scfg={"n": 2}, rollout_fn=fn, group_fn=stage)
    rep = await rt.run()
    assert len(rep.steps) == 2 and rep.trained_records and all(r.reward_components.get("stage/ran") == 1.0 for r in rep.trained_records)


async def test_H2_V16_stage_entries_of_evicted_groups_are_released():
    rt, _tr = build(4, rcfg={"steps": 30, "max_wall_s": 20.0}, scfg={"n": 2, "filter_zero_variance": True}, env={"max_turns": 1}, group_fn=lambda g, m: m)
    rep = await rt.run()
    assert len(rep.steps) == 30 and rt.sched.counters["evicted_zero_var"] > 0
    assert not [g for g in rt._group_stage if g not in rt.pool.groups or rt.pool.groups[g].status == "evicted"]


# ----------------------------------------------------------------------------------------------- H2 V8 / G2 D-T2-5, H3 V21
async def test_H2_V8_masked_record_from_a_rollout_fn_is_masked_and_trips_the_breaker():
    async def fn(rid, prompt, seed, gen, temperature=1.0, harness=""):
        res = await gen([1, 2, 3], 3, temperature)
        masked = random.Random(rid).random() < 0.5
        return record(rid, res, reward=1.0 if masked else float(random.Random(rid + "r").random() < 0.5), mask_sample=masked,
                      failure_kind="importer:masked" if masked else None)

    rt, tr = build(1, rcfg={"steps": 4, "breaker_min": 4, "max_masked_share": 0.2}, scfg={"n": 2}, rollout_fn=fn)
    rep = await rt.run()
    assert rep.stop_reason == "masked_share" and rep.masked_rollouts > 0 and rep.masked_kinds.get("importer:masked") == rep.masked_rollouts
    assert not tr.violations


async def test_H3_V21_token_incomplete_records_are_masked_before_training():
    async def fn(rid, prompt, seed, gen, temperature=1.0, harness=""):
        res = await gen([1, 2, 3], 3, temperature)
        rec = record(rid, res)
        if zlib.crc32(rid.encode()) % 3 == 0:
            rec.turns[0].logprobs = None  # the capture lost the behaviour log-probs
        return rec

    rt, _tr = build(1, rcfg={"steps": 4, "max_masked_share": 1.0}, scfg={"n": 2}, rollout_fn=fn)
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.masked_kinds.get("crucible:capture_incomplete", 0) > 0
    assert all(t.logprobs is not None for r in rep.trained_records for t in r.turns)
    assert list(rt._recent).count("masked") >= rep.masked_kinds["crucible:capture_incomplete"]  # the breaker sees them


# ----------------------------------------------------------------------------------------------- H3 V9: bounded re-issue
async def test_H3_V9_an_engine_that_always_aborts_masks_the_rollout_by_name():
    rt, _tr = build(1, rcfg={"steps": 2, "max_wall_s": 3.0, "max_call_reissues": 3, "reissue_backoff_s": 0.0}, scfg={"n": 2}, eng=Eng(always_abort=True))
    rep = await rt.run()
    assert rep.stop_reason == "masked_share" and rep.masked_kinds.get("crucible:reissue_limit", 0) >= 20
    assert rep.aborted_calls <= 4 * rt._seeds  # at most max_call_reissues + 1 aborts per started rollout


# ----------------------------------------------------------------------------------------------- H2 V3-V6, V11: scheduler
def _sched(**kw):
    pool, ids = DataPool(), itertools.count()
    return pool, Scheduler(SchedulerConfig(**kw), pool, {"x": lambda: f"t{next(ids)}"})


def _finish_all(pool, version):
    for t in list(pool.live()):
        if t.status == "pending":
            pool.start(t.rid)
        pool.call(t.rid, version, 1)
        pool.finish(t.rid, reward=float(hash(t.rid) % 2))


def test_H2_V3_allow_is_bounded_at_tau_plus_one():
    pool, s = _sched(n=1, B=1, max_staleness=1, admission_staleness=4, bound_overflow="allow")
    s.admit(0)
    _finish_all(pool, 0)
    seen = []
    for v in range(1, 5):
        sel = s.select(v)
        seen += sel.staleness
        s.trained(sel)
    assert seen and max(seen) <= 2


def test_H2_V4_offpolicy_cap_keeps_tau_and_never_deadlocks():
    src = {"a": [f"a{i}" for i in range(400)], "b": [f"b{i}" for i in range(400)]}
    r = simulate(SchedulerConfig(n=4, B=4, max_staleness=3, max_offpolicy_frac=0.25), SimConfig(topology="colocated_partial", steps=15), src)
    assert r.steps == 15 and r.max_staleness() <= 3
    pool, s = _sched(n=1, B=2, max_staleness=2, admission_staleness=2, max_offpolicy_frac=0.0)
    s.admit(0)
    _finish_all(pool, 0)
    s.admit(1)
    assert s.select(1).groups  # every buffered group off-policy, the cap full, nothing live: the held-back groups train


def test_G3_D2_offpolicy_cap_does_not_deadlock_disaggregated():
    src = {"a": [f"a{i}" for i in range(400)], "b": [f"b{i}" for i in range(400)]}
    for frac in (0.25, 0.5):
        r = simulate(SchedulerConfig(n=4, B=4, max_staleness=2, max_offpolicy_frac=frac), SimConfig(topology="disaggregated_stream", steps=30), src)
        assert r.steps == 30 and r.max_staleness() <= 2


def test_H2_V5_windowed_fifo_with_window_below_batch_trains():
    pool, s = _sched(n=1, B=4, max_staleness=2, buffer_policy="windowed_fifo")
    assert s.cfg.window() < s.cfg.B
    s.admit(0)
    _finish_all(pool, 0)
    assert len(s.select(0).groups) == 4


def test_H2_V6_windowed_fifo_respects_tau():
    src = {"a": [f"a{i}" for i in range(600)]}
    for seed in range(6):
        for tau in (1, 2):
            from crucible.train.sim import default_shapes

            r = simulate(SchedulerConfig(n=2, B=2, max_staleness=tau, buffer_policy="windowed_fifo", fifo_window_frac=0.5),
                         SimConfig(topology="colocated_partial", steps=20), src, default_shapes(seed, tail_frac=0.3, tail_mult=15.0))
            assert r.steps == 20 and r.max_staleness() <= tau, (seed, tau)


def test_H2_V11_windowed_lag_bound():
    # the article's case, batch 8 and W 4: it ran 0 steps, so the lag ≤ 10 assertion was vacuous
    r = simulate(SchedulerConfig(n=1, B=8, buffer_policy="windowed_fifo", fifo_window_frac=4 / 8, admission_staleness=1, staleness_strategy="none"),
                 SimConfig(topology="disaggregated_stream", steps=12, publish_every=1), {"x": [f"p{i}" for i in range(500)]})
    assert r.steps == 12 and r.max_staleness() <= 10


# ----------------------------------------------------------------------------------------------- H2 V9, V10: bounded loops
async def test_H2_V9_dynamic_sampling_is_bounded():
    rt, _tr = build(3, rcfg={"steps": 5, "max_wall_s": 6.0}, env={"const": True, "max_turns": 1}, scfg={"filter_zero_variance": True})
    rep = await rt.run()
    assert len(rep.steps) == 5 and rep.stop_reason == "steps" and rt.sched.counters.get("zero_variance_exhausted", 0) > 0


async def test_H2_V10_refill_member_is_capped():
    class Bad(Env):
        first = False

        async def reset(self, prompt, seed):
            out = await super().reset(prompt, seed)
            self._fail = prompt == "a1" and Bad.first
            Bad.first = Bad.first or prompt == "a1"
            return out

        async def step(self, completion):
            if self._fail:
                await asyncio.sleep(0.002)
                raise EnvInfraError("this task's sandbox never starts")
            return await super().step(completion)

    eng = EnginePool([Eng(11)])
    tr = Trainer()
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=20, max_turn_tokens=3, poll_s=0.001, warmup_batches=0, max_wall_s=8.0, breaker_min=10**9),
                 SchedulerConfig(n=4, B=2, max_staleness=2, group_backfill="refill_member"), eng, tr, Pub(eng), lambda: Bad(11),
                 {"a": [f"a{i}" for i in range(100000)]})
    tr.rt = rt
    rep = await rt.run()
    assert len(rep.steps) == 20 and rt.sched.counters.get("refill_exhausted") == 1 and not tr.violations


# ----------------------------------------------------------------------------------------------- H2 V12-V20: smaller
def test_H2_V12_checkpoints_land_on_publish_steps():
    with pytest.raises(ValueError, match="multiple of publish_every"):
        build(rcfg={"checkpoint_every": 3, "publish_every": 2, "checkpoint_dir": "/nonexistent"})


def test_H2_V13_hack_rate_uses_the_audit_denominator():
    m = StepMonitor(min_audits=10)
    out = {}
    for step in range(10):  # 30 % of audits confirm a hack; the audit samples 2 rollouts of a 64-record batch per step
        out = m.record(step, rewards=[1.0] * 64, hacks=0, audit_runs=2, audit_confirmed=1 if step % 3 == 0 else 0)
    assert out["audit/confirmed_rate"] == pytest.approx(0.2) and out["confirmed_hack_rate"] == pytest.approx(0.2)
    assert any(a["kind"] == "confirmed_hack_rate" for a in m.alarms)


def test_H2_V14_simulator_launches_refilled_members():
    def shape(prompt, i):
        from crucible.train.sim import RolloutShape

        return RolloutShape([(100, 50, 1.0)], 1.0 if i % 2 else 0.0, masked=(i == 1 and int(prompt[1:]) % 2 == 0))

    r = simulate(SchedulerConfig(n=4, B=4, max_staleness=2, group_backfill="refill_member"), SimConfig(topology="colocated_partial", steps=15),
                 {"a": [f"a{i}" for i in range(400)]}, shape)
    assert r.steps == 15 and r.evicted.get("refilled_members", 0) > 0


def test_H2_V15_simulator_is_deterministic_across_processes():
    code = "from crucible.train.sim import default_shapes as d; s = d(0); print([s(f'p{i}', 0).reward for i in range(40)])"
    outs = {subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": str(h)}, check=True).stdout
            for h in (1, 2, 3)}
    assert len(outs) == 1


def test_H2_V17_masked_member_is_never_marked_trained():
    pool = DataPool()
    g = pool.submit_group("g", "a", 2, 0)
    for i, rid in enumerate(g.members):
        pool.start(rid)
        pool.finish(rid, reward=1.0, status="masked" if i else "finished")
    pool.mark_trained("g")
    assert [pool.tags[r].status for r in g.members] == ["trained", "evicted"]


async def test_H2_V18_episodes_on_env_workers_clean_up_at_shutdown():
    started, cleaned = [], []

    async def episode(rid, prompt, seed, gen, temperature=1.0, harness=""):
        started.append(rid)
        try:
            res = await gen([1, 2, 3], 2, temperature)
            await asyncio.sleep(0.05 if int(prompt[1:]) % 3 else 5.0)
            return record(rid, res)
        finally:
            await asyncio.sleep(0.01)  # async cleanup, as World.close is
            cleaned.append(rid)

    rt, _tr = build(1, rcfg={"steps": 2, "env_workers": 2}, scfg={"n": 2}, rollout_fn=episode)
    rep = await rt.run()
    assert len(rep.steps) == 2 and set(started) == set(cleaned)


def test_H2_V19_dict_verdicts_screen_on_their_reward():
    from crucible.train.admission import _passed

    assert not _passed({"reward": 0.0, "detail": "x"}) and _passed({"reward": 1.0}) and not _passed({"detail": "no reward"})


async def test_H2_V20_zero_variance_reads_every_seat():
    async def game(rid, prompt, seed, gen, temperature=1.0, harness=""):
        res = await gen([1, 2, 3], 3, temperature)
        b = float(zlib.crc32(rid.encode()) % 2)
        return [record(rid, res, reward=1.0, actor="alice"), record(rid, res, reward=b, actor="bob", rollout_id=f"{rid}#bob")]

    rt, _tr = build(1, rcfg={"steps": 3, "max_wall_s": 5.0}, scfg={"n": 4, "filter_zero_variance": True}, rollout_fn=game)
    rep = await rt.run()
    assert len(rep.steps) == 3 and {r.actor for r in rep.trained_records} == {"alice", "bob"}

    async def judge(rec):  # a source's reward_fn grades every seat (it was skipped for games)
        return 0.25 if rec.actor == "bob" else 0.75

    rt, _tr = build(1, rcfg={"steps": 2}, scfg={"n": 2}, rollout_fn=game, reward_fn=judge)
    rep = await rt.run()
    assert len(rep.steps) == 2 and {(r.actor, r.reward) for r in rep.trained_records} == {("alice", 0.75), ("bob", 0.25)}


# ----------------------------------------------------------------------------------------------- H1 V8, G1, G2, G3, G4
def test_H1_V8_teacher_pool_needs_the_student_vocabulary():
    from crucible.train.teachers import TeacherPool

    class T:
        id, vocab_id = "t", "other-vocab"

        async def score(self, pairs):
            return None

    with pytest.raises(ValueError, match="student_vocab_id"):
        build(teacher_pool=TeacherPool([T()], default="t"))


def test_G1_H5_trainer_temperature_must_match_sampling():
    class Hot(Trainer):
        class cfg:
            temperature = 0.7

    eng = EnginePool([Eng()])
    with pytest.raises(ValueError, match="temperature"):
        Runtime(RuntimeConfig(temperature=1.0), SchedulerConfig(n=2, B=1), eng, Hot(), Pub(eng), lambda: Env(1), {"a": ["a0"]})


def test_G1_H12a_a_graded_rollout_is_never_cut():
    rt, _ = build(1, rcfg={"max_cuts": 0})
    rt.sched.admit(0)
    a, b = rt.pool.pending()[:2]
    rt.pool.start(a)
    rt.pool.start(b)
    rt.pool.grading(b)
    rt._cut_live()
    assert rt.pool.tags[b].cuts == 0 and rt.pool.tags[b].status == "evicted"  # its group was evicted for `a`, but b was not cut
    rt, _ = build(1, rcfg={"max_cuts": 0})
    rt.sched.admit(0)
    b = rt.pool.pending()[0]
    rt.pool.start(b)
    rt.pool.grading(b)
    rt._cut_live()
    assert rt.pool.tags[b].cuts == 0 and rt.pool.tags[b].status == "grading"


async def test_G3_D10_engines_pause_before_the_barrier_abort():
    e = Eng(1)
    rt, _tr = build(1, rcfg={"steps": 2}, scfg={"n": 2}, eng=e)
    await rt.run()
    first = [x for x in e.log[:2]]
    assert first == ["pause", "abort"]


async def test_G4_pick9_drop_evicts_and_cancels_doomed_groups():
    pool, s = _sched(n=2, B=1, max_staleness=1, staleness_strategy="drop", admission_staleness=2)
    gs = s.admit(0)
    for g in gs:
        pool.start(g.members[0])
        pool.call(g.members[0], 0)
    doomed = s.evict_doomed(2)  # oldest version 0, two versions behind with τ = 1: it can never train
    assert doomed and all(pool.tags[r].status == "evicted" for r in doomed) and s.counters["evicted_doomed"] == len(gs)
    rt, tr = build(1, rcfg={"steps": 10}, scfg={"n": 4, "B": 2, "max_staleness": 0, "admission_staleness": 3, "staleness_strategy": "drop"}, env={"tail": 0.3})
    rep = await rt.run()
    assert len(rep.steps) == 10 and rep.doomed_evictions > 0 and not tr.violations and no_lost(rt)


async def test_G3_pick12_group_redundancy_trains_the_first_n():
    src = {"a": [f"a{i}" for i in range(600)]}
    r = simulate(SchedulerConfig(n=4, B=2, max_staleness=2, group_redundancy=2), SimConfig(topology="colocated_partial", steps=10), src)
    assert r.steps == 10 and all(len(b) == 8 for b in r.staleness) and r.evicted.get("evicted_redundant", 0) > 0
    rt, tr = build(1, rcfg={"steps": 6}, scfg={"n": 4, "B": 2, "group_redundancy": 2})
    rep = await rt.run()
    by = defaultdict(int)
    for x in rep.trained_records:
        by[x.group] += 1
    assert len(rep.steps) == 6 and set(by.values()) == {4} and rt.sched.counters["evicted_redundant"] > 0 and not tr.violations
    with pytest.raises(ValueError, match="no group could ever be admitted"):
        SchedulerConfig(n=4, B=1, max_staleness=1, group_redundancy=8)


async def test_G2_DT2_4_zero_variance_filter_reads_the_group_stage():
    def separate(gid, members):  # a GRS / tournament stage: all-pass members get distinct advantages
        for i, m in enumerate(members):
            m.advantage = i - (len(members) - 1) / 2
        return members

    rt, _tr = build(3, rcfg={"steps": 4, "max_wall_s": 6.0}, env={"const": True, "max_turns": 1},
                   scfg={"filter_zero_variance": True}, group_fn=separate)
    rep = await rt.run()
    assert len(rep.steps) == 4 and rt.sched.counters["evicted_zero_var"] == 0


def test_impossible_scheduler_configurations_are_refused():
    with pytest.raises(ValueError, match="inflight_cap"):
        SchedulerConfig(n=4, inflight_cap={"a": 2})
    with pytest.raises(ValueError, match="group_backfill"):
        SchedulerConfig(group_backfill="refill")


# ----------------------------------------------------------------------------------------------- engine / weight-plane hand-off
async def test_gen_forwards_seed_and_stop_ids_and_defaults_stops_from_the_codec():
    from crucible.train.envpool import EnvWorkerPool
    from crucible.train.world_env import _accepts

    async def fn(rid, prompt, seed, gen, temperature=1.0, harness=""):
        assert _accepts(gen, "seed") and _accepts(gen, "stop_ids")
        res = await gen([1, 2, 3], 3, temperature, seed=5, stop_ids=[7])
        await gen([1, 2, 3], 3, temperature)
        return record(rid, res)

    class Codec:
        eos_ids = (9,)

    fn.codec = Codec()
    e = Eng(1)
    rt, _ = build(1, rcfg={"steps": 1}, scfg={"n": 2, "B": 1}, eng=e, rollout_fn=fn)
    await rt.run()
    assert (5, (7,)) in {(r.seed, tuple(r.stop_ids)) for r in e.reqs} and (None, (9,)) in {(r.seed, tuple(r.stop_ids)) for r in e.reqs}
    pool = EnvWorkerPool(1)
    try:
        async def g(ids, n, temp, *, seed=None, stop_ids=()):
            return None

        assert _accepts(pool.bridge(g), "seed")  # the worker bridge keeps gen's signature
    finally:
        pool.close()


def test_token_version_switch_needs_token_resume_engines():
    eng = EnginePool([Eng()])
    with pytest.raises(ValueError, match="token_resume"):
        Runtime(RuntimeConfig(version_switch="token"), SchedulerConfig(n=2, B=1), eng, Trainer(), Pub(eng), lambda: Env(1), {"a": ["a0"]})
    eng = EnginePool([Eng(caps={"token_ids": True, "token_resume": True})])
    Runtime(RuntimeConfig(version_switch="token"), SchedulerConfig(n=2, B=1), eng, Trainer(), Pub(eng), lambda: Env(1), {"a": ["a0"]})


def test_H3_V9_reissue_limit_is_a_registered_infra_kind():
    from crucible.core.schema.run import default_blame, is_failure_kind
    from crucible.train.runtime import CallReissueLimit

    assert is_failure_kind(CallReissueLimit.failure_kind) and default_blame(CallReissueLimit.failure_kind) == "infra"


async def test_gateway_engine_call_failed_masks_as_infra():
    from crucible.train.gateway import EngineCallFailed

    e = Eng(2, fail=lambda cid: zlib.crc32(cid.encode()) % 7 == 0, exc=EngineCallFailed)
    rt, tr = build(2, rcfg={"steps": 4, "max_masked_share": 1.0, "max_harness_share": 0.0}, scfg={"n": 2}, eng=e)
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.stop_reason == "steps" and rep.failures.get("infra:EngineCallFailed", 0) > 0
    assert rep.masked_kinds.get("provider_unavailable") == rep.failures["infra:EngineCallFailed"] and not tr.violations
