"""Per-step monitoring and the remaining scheduler properties (M12 checklist #5, #13, #20, #24, #27, #51, #52)."""

from __future__ import annotations

from typing import ClassVar

from crucible.train.engine import EnginePool, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask
from crucible.train.pool import DataPool
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import Scheduler, SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _rt(steps=6, *, sched=None, publisher_kw=None, reward_fn=None, on_alarm=None, env=None, decode_delay_s=0.002, warmup_batches=0, **cfg):
    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=decode_delay_s)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=steps, max_turn_tokens=3, warmup_batches=warmup_batches, **cfg),
                 sched or SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr, InprocPublisher(tr, eng, **(publisher_kw or {})),
                 env or (lambda: CopyTask(turns=2, symbols=8, latency_s=0.002, long_tail=0.2)), {"copy": [f"p{i}" for i in range(2000)]},
                 reward_fn=reward_fn, on_alarm=on_alarm)
    return rt, tr


async def test_every_step_logs_the_monitor_set():
    rt, _ = _rt(steps=5)
    rep = await rt.run()
    keys = ("weights/publish_s", "rollout/reprefill_tokens", "admission_blocked_s", "staleness/max", "staleness_worst/max", "trajectory_spans/max",
            "truncation_share", "zero_variance_groups", "entropy", "confirmed_hack_rate", "length/p99", "pass_rate")
    for s in rep.steps:
        assert all(k in s.metrics for k in keys), [k for k in keys if k not in s.metrics]
        assert s.metrics["staleness_worst/max"] >= s.metrics["staleness/mean"] and s.metrics["entropy"] > 0
    assert sum(rep.series("rollout/reprefill_tokens")) <= rep.reprefill_tokens  # per-step deltas of the run total
    assert any(v > 0 for v in rep.series("admission_blocked_s"))  # the untrained cap binds in partial rollout


async def test_drop_strategy_logs_evicted_staleness_over_tau():
    rt, _ = _rt(steps=12, sched=SchedulerConfig(n=2, B=2, max_staleness=1, staleness_strategy="drop", admission_staleness=4),
                env=lambda: CopyTask(turns=2, symbols=8, latency_s=0.005, long_tail=0.3, tail_mult=30))  # long tails age past τ
    rep = await rt.run()
    ev = rep.series("evicted_staleness/max")
    assert ev and min(ev) > 1  # only groups older than τ = 1 were evicted
    assert max(rep.series("staleness/max")) <= 1


async def test_publish_every_k_and_optimizer_reset():
    rt, tr = _rt(steps=6, publish_every=2, publisher_kw={"reset_optimizer_on_publish": True})
    rep = await rt.run()
    assert rep.publishes == 3 and [s.version for s in rep.steps] == [0, 1, 1, 2, 2, 3]
    assert len(tr.opt.state) == 0  # GLM: moments zeroed at every publish (the last step published)


async def test_pass_rate_jump_triggers_a_full_tier_reaudit():
    audited = []

    async def grade(rec):
        return 1.0 if rec.turns[0].policy_version >= 7 else 0.0  # a sudden "solve": the classic hack signature

    async def reaudit(step, records):
        audited.append((step, len(records)))

    rt, _ = _rt(steps=11, reward_fn=grade, on_alarm=reaudit)
    rep = await rt.run()
    assert rt.monitor.alarms and audited and audited[0][1] > 0
    assert any(s.metrics.get("alarm/reaudit_records") for s in rep.steps)


def test_drop_eviction_refills_exactly_one_group_each():
    pool = DataPool()
    ids = iter(range(1000))
    s = Scheduler(SchedulerConfig(n=1, B=4, max_staleness=1, staleness_strategy="drop", admission_staleness=2), pool, {"x": lambda: f"t{next(ids)}"})
    first = s.admit(0)
    assert len(first) == 8 and s.admit(0) == []  # the cap binds
    for g in first[:3]:
        pool.finish(g.members[0], reward=1.0)
    sel = s.select(5)  # v_train = 5: the three finished groups are stale (> τ)
    assert len(sel.evicted) == 3 and sel.blocked == "need_more"
    assert len(s.admit(5)) == 3  # one new group per evicted group, no more


def test_windowed_fifo_limits_equal_fifo_and_greedy():
    def run(frac: float) -> list[int]:
        pool = DataPool()
        ids = iter(range(100))
        s = Scheduler(SchedulerConfig(n=1, B=1, buffer_policy="windowed_fifo", fifo_window_frac=frac, admission_staleness=4, staleness_strategy="none"),
                      pool, {"x": lambda: f"t{next(ids)}"})
        gs = s.admit(0)
        for i, t in ((2, 1.0), (1, 2.0), (3, 3.0)):  # finish order: T2, T1, T3; T0 still running
            pool.finish(gs[i].members[0], reward=1.0, now=t)
        out = []
        for _ in range(3):
            sel = s.select(0)
            if sel.blocked:
                break
            out.append(int(sel.groups[0].gid.split(":t")[1].split("@")[0]))
            s.trained(sel)
        return out

    assert run(1 / 4) == []  # W = 1: strict FIFO, nothing until T0 is done
    assert run(1.0) == [2, 1, 3]  # W = N: greedy, earliest finished first


async def test_same_records_give_the_same_loss_under_every_topology():
    """#6: loss / credit / record code does not depend on the topology (golden: one batch through each runtime's _train)."""
    import copy

    from crucible.core.schema.rollout import RolloutRecord, Turn
    from crucible.train.scheduler import Selection

    base = TinyPolicy(32, 16, seed=5)
    losses = {}
    for topo in ("colocated_sync", "colocated_partial", "disaggregated_stream"):
        model = copy.deepcopy(base)
        eng = EnginePool([TorchEngine(model, capacity=8)])
        tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
        rt = Runtime(RuntimeConfig(topology=topo), SchedulerConfig(n=2, B=1), eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(), {"c": ["p"]})
        g = rt.pool.submit_group("g", "c", 2, 0)
        for i, rid in enumerate(g.members):
            rt.pool.start(rid)
            rec = RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=float(i),
                                turns=[Turn(sampling={"temperature": 1.0}, call_id=f"{rid}:c", actor="agent", prompt_token_ids=[1, 2, 3], completion_token_ids=[4 + i, 5], logprobs=[-1.0, -2.0],
                                            policy_version=0)])
            rt.pool.finish(rid, reward=rec.reward, value=rec)
        m = await rt._train(Selection([rt.pool.groups["g"]], {}))
        losses[topo] = m["train/loss"]
    assert len(set(losses.values())) == 1, losses


class CountingCopy(CopyTask):
    steps: ClassVar[dict] = {}

    async def reset(self, prompt, seed):
        self.key = (prompt, seed)
        return await super().reset(prompt, seed)

    async def step(self, completion):
        CountingCopy.steps[self.key] = CountingCopy.steps.get(self.key, 0) + 1
        return await super().step(completion)


async def test_colocated_barrier_never_replays_environment_actions():
    """#11: aborting in-flight calls at the sample barrier leaves environments untouched: every finished rollout took
    exactly ``turns`` environment steps, although calls were aborted and re-issued."""
    CountingCopy.steps = {}
    rt, _ = _rt(steps=8, env=lambda: CountingCopy(turns=3, symbols=8, latency_s=0.002, long_tail=0.3, tail_mult=15), decode_delay_s=0.005)
    rep = await rt.run()
    assert rep.aborted_calls > 0
    finished = [r for r in rep.trained_records]
    assert finished and all(len(r.turns) == 3 for r in finished)
    assert all(v <= 3 for v in CountingCopy.steps.values())  # never more steps than turns, for any rollout


async def test_warmup_batches_gate_the_first_step():
    """#14: with warmup_batches = 1 the first optimizer step waits for B × (1 + 1) terminal groups."""
    seen = []
    rt, tr = _rt(steps=2, warmup_batches=1)
    step = tr.step

    def spy(recs, groups):
        seen.append(len(rt.pool.terminal_groups()) + len(set(groups)))  # still-terminal + the batch being trained
        return step(recs, groups)

    tr.step = spy
    await rt.run()
    assert seen[0] >= 2 * 2


def test_pad_and_drop_group_backfill():
    """#17: a masked member is padded (the group trains without it) or the whole group is dropped."""
    for mode, expect in (("pad", "trained"), ("drop_group", "evicted")):
        pool = DataPool()
        q = [f"p{i}" for i in range(10)]
        s = Scheduler(SchedulerConfig(n=2, B=1, group_backfill=mode, staleness_strategy="none"), pool, {"x": lambda q=q: q.pop(0) if q else None})
        g = s.admit(0)[0]
        pool.finish(g.members[0], reward=1.0)
        pool.finish(g.members[1], reward=None, status="masked")
        sel = s.select(0)
        if expect == "trained":
            assert [x.gid for x in sel.groups] == [g.gid]
        else:
            assert sel.evicted.get(g.gid) == "masked_member" and not sel.groups
