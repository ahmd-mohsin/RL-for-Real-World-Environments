"""M12 control plane: the scheduler driven by the discrete-event simulator (research m11-m12/06 checklist items
10, 12, 15, 20–23, 25, 27–30). Deterministic; no GPU."""

from __future__ import annotations

from crucible.train.pool import DataPool
from crucible.train.scheduler import Scheduler, SchedulerConfig
from crucible.train.sim import SimConfig, ks_distance, simulate

SRC = {"a": [f"a{i}" for i in range(400)], "b": [f"b{i}" for i in range(400)]}


def run(topology="colocated_partial", **kw):
    sim_kw = {k: kw.pop(k) for k in list(kw) if k in SimConfig.__dataclass_fields__}
    return simulate(SchedulerConfig(n=4, B=4, **kw), SimConfig(topology=topology, steps=15, **sim_kw), SRC)


def test_colocated_sync_is_on_policy():
    r = run("colocated_sync")
    assert r.steps == 15 and r.max_staleness() == 0 and all(s == 1 for b in r.spans for s in b)  # item 10


def test_partial_rollout_wait_never_drops_and_bounds_staleness():
    r = run("colocated_partial", max_staleness=2, staleness_strategy="wait")
    assert r.steps == 15 and "stale" not in r.evicted and r.max_staleness() <= 2  # item 21
    assert r.inflight_violations == 0 and r.max_inflight_seen <= 2 * 4 * 4  # item 22: ≤ S·B·n untrained rollouts
    assert r.reprefill_tokens > 0  # aborted calls were re-issued and paid re-prefill (item 5)


def test_drop_strategy_never_trains_over_tau():
    r = run("colocated_partial", max_staleness=1, staleness_strategy="drop", admission_staleness=4)
    assert r.max_staleness() <= 1 and r.evicted.get("stale", 0) > 0  # item 20


def test_disaggregated_keeps_engine_busy_while_training():
    r = run("disaggregated_stream", slots=32, train_s_base=40.0)
    assert r.engine_down_s == 0.0  # generation never pauses (item 12)
    assert r.busy_while_training_s / max(r.time - r.trainer_idle_s, 1e-9) > 0.9


def test_staleness_unit_glm_example():
    pool = DataPool()
    g = pool.submit_group("g", "s", 2, 6)
    a, b = g.members
    for v in (6, 7, 8):
        pool.call(a, v)
    for v in (7, 8, 9):
        pool.call(b, v)
    assert pool.tags[a].staleness(10) == 4 and pool.tags[b].staleness(10) == 3  # τ = 3: (6,7,8) dropped, (7,8,9) kept (item 23)
    assert pool.tags[a].spans() == 3


def test_per_source_caps_and_group_admission():
    pool = DataPool()
    q = {"a": [f"a{i}" for i in range(50)], "b": [f"b{i}" for i in range(50)]}
    s = Scheduler(SchedulerConfig(n=4, B=2, max_staleness=4, inflight_cap={"a": 8, "b": 16}), pool,
                  {k: (lambda k=k: q[k].pop(0) if q[k] else None) for k in q})
    s.admit(0)
    assert len(pool.live("a")) == 8 and len(pool.live("b")) == 16  # whole groups, within caps (item 15)
    assert len(pool.live()) <= s.cfg.max_inflight()


def test_windowed_fifo_blocks_out_of_window_and_bounds_lag():
    # Fig. 9: with W groups, finished groups past head + W are not taken while the head is untrained
    pool = DataPool()
    ids = iter(range(100))
    s = Scheduler(SchedulerConfig(n=1, B=1, buffer_policy="windowed_fifo", fifo_window_frac=4 / 8, admission_staleness=8, staleness_strategy="none"),
                  pool, {"x": lambda: f"t{next(ids)}"})
    gs = s.admit(0)
    assert len(gs) == 8 and s.cfg.window() == 4
    for i in (1, 3, 4, 5, 7):  # T0 still running
        pool.finish(gs[i].members[0], reward=1.0, now=float(i))
    sel = s.select(0)
    assert [g.gid for g in sel.groups] == [gs[1].gid]  # earliest finished inside [T0, T3]
    s.trained(sel)
    sel = s.select(0)
    assert [g.gid for g in sel.groups] == [gs[3].gid]
    s.trained(sel)
    assert s.select(0).blocked == "need_more"  # T4, T5, T7 finished but outside the window while T0 is untrained (item 27)
    r = simulate(SchedulerConfig(n=1, B=8, buffer_policy="windowed_fifo", fifo_window_frac=4 / 8, admission_staleness=1, staleness_strategy="none"),
                 SimConfig(topology="disaggregated_stream", steps=12, publish_every=1), {"x": [f"p{i}" for i in range(500)]})
    assert r.max_staleness() <= 10  # the article's bound: batch 8, W 4 -> 8 + 3 − 1 (item 28)


def test_dynamic_sampling_and_backfill():
    pool = DataPool()
    q = [f"p{i}" for i in range(10)]
    s = Scheduler(SchedulerConfig(n=2, B=1, filter_zero_variance=True, group_backfill="refill_member", staleness_strategy="none"), pool,
                  {"x": lambda: q.pop(0) if q else None})
    g1, g2, *_ = s.admit(0)
    pool.finish(g1.members[0], reward=1.0)
    pool.finish(g1.members[1], reward=1.0)  # zero variance -> evicted
    pool.finish(g2.members[0], reward=1.0)
    pool.finish(g2.members[1], reward=None, status="masked")  # infra -> refill a member
    sel = s.select(0)
    assert sel.evicted.get(g1.gid) == "zero_variance" and sel.blocked == "need_more" and s.counters["refilled_members"] == 1
    new = pool.groups[g2.gid].members[-1]
    pool.finish(new, reward=0.0)
    sel = s.select(0)
    assert [g.gid for g in sel.groups] == [g2.gid]


def test_max_offpolicy_frac_and_ks():
    assert ks_distance([1, 2, 3], [1, 2, 3]) == 0.0 and ks_distance([1, 1, 1], [9, 9, 9]) == 1.0
    r = run("colocated_partial", max_staleness=3, max_offpolicy_frac=0.25)
    assert all(sum(1 for s in b if s > 0) <= max(1, len(b) // 4) * 4 for b in r.staleness)  # ≤ 25% of groups off-policy (item 25)


def test_monitors_alarm_and_length_bias():
    from crucible.train.monitors import StepMonitor

    mon = StepMonitor(steady_window=5)
    for s in range(8):
        mon.record(s, rewards=[0.0, 0.0, 1.0, 0.0], lengths=[1000, 1200, 900, 1100])
    m = mon.record(8, rewards=[1.0, 1.0, 1.0, 1.0], lengths=[100, 120, 90, 110], staleness=[0, 1, 2, 0], spans=[1, 2, 3, 1])
    assert m["alarm/pass_rate_jump"] == 1.0 and mon.alarms[-1]["action"].startswith("full-tier")  # item 52
    assert m["length/ks_vs_steady"] == 1.0 and m["on_policy_share"] == 0.5 and m["staleness/max"] == 2  # items 24, 29
