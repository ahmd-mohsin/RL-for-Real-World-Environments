"""Discrete-event simulator of the RL loop (research m11-m12/06 checklist: "Sim" items). Deterministic, no GPU.

It drives the **real** ``Scheduler`` / ``DataPool`` with a timing model, so every scheduling and synchronization rule
is tested as it will run, and topology / staleness / window settings can be swept before spending GPU time.

Model:
- a rollout is a sequence of turns; each turn is a **model call** (prefill + decode on the engine) followed by an
  **environment step** (on a separate CPU worker pool: the World never waits for weights, and an env step holds no
  engine slot);
- the **engine** has ``slots`` concurrent sequences, decodes ``tok_per_s`` per sequence and prefills ``prefill_tok_per_s``;
  a call is served by exactly one policy version (call-level version switching, checklist 2–3);
- the **trainer** takes ``train_s(tokens)`` per optimizer step; a publish takes ``publish_s``.

Topologies and their synchronization points (S1 sample barrier, S2 publish, S3 in-flight calls at publish):
- ``colocated_sync``: admit one batch, wait until it is terminal, train, publish (staleness 0);
- ``colocated_partial``: generate until the scheduler can select a batch (S1); then **abort every in-flight call**
  (its partial decode is wasted, it is re-issued whole after the publish and pays re-prefill), the engine is down
  for train + publish while environments keep running, then generation resumes on the new version;
- ``disaggregated_stream``: the engine never stops; the trainer trains whenever a batch is selectable; every K updates
  a publish switches the engine at call boundaries (in-flight calls finish on their version).
"""

from __future__ import annotations

import heapq
import random
from collections.abc import Callable
from dataclasses import dataclass, field

from crucible.train.pool import DataPool
from crucible.train.scheduler import Scheduler, SchedulerConfig


@dataclass
class RolloutShape:
    turns: list[tuple[int, int, float]]  # (prompt tokens added, completion tokens, env seconds) per turn
    reward: float
    masked: bool = False


def default_shapes(seed: int = 0, *, tail_frac: float = 0.2, tail_mult: float = 10.0, turns: tuple[int, int] = (2, 6)) -> Callable[[str, int], RolloutShape]:
    """Per (prompt, member) shape: most rollouts short, ``tail_frac`` of them ``tail_mult``× longer (the long tail)."""
    def shape(prompt: str, member: int) -> RolloutShape:
        r = random.Random(f"{seed}|{prompt}|{member}")
        long = r.random() < tail_frac
        k = r.randint(*turns) * (2 if long else 1)
        ts = [(r.randint(100, 400), int(r.randint(80, 300) * (tail_mult if long else 1)), r.uniform(0.5, 3.0)) for _ in range(k)]
        p = (hash(prompt) % 7) / 8
        return RolloutShape(ts, 1.0 if r.random() < p else 0.0)
    return shape


@dataclass
class SimConfig:
    topology: str = "colocated_partial"  # colocated_sync | colocated_partial | disaggregated_stream
    slots: int = 64
    tok_per_s: float = 50.0
    prefill_tok_per_s: float = 5_000.0
    env_workers: int = 256
    train_s_base: float = 20.0
    train_s_per_ktok: float = 0.05
    publish_s: float = 2.0
    publish_every: int = 1
    warmup_batches: int = 1
    steps: int = 20
    max_time: float = 1e7


@dataclass
class _R:
    rid: str
    shape: RolloutShape
    turn: int = 0
    context: int = 0
    call_started: float | None = None
    call_version: int | None = None
    gen_event: int | None = None
    aborted: bool = False  # the current call was cut at a barrier: its re-issue re-prefills the whole context


@dataclass
class SimResult:
    steps: int
    time: float
    staleness: list[list[int]] = field(default_factory=list)
    spans: list[list[int]] = field(default_factory=list)
    batch_lengths: list[list[int]] = field(default_factory=list)
    batch_sources: list[list[str]] = field(default_factory=list)
    trainer_idle_s: float = 0.0
    engine_busy_s: float = 0.0
    engine_down_s: float = 0.0
    slot_seconds: float = 0.0
    reprefill_tokens: int = 0
    wasted_decode_tokens: int = 0
    max_inflight_seen: int = 0
    inflight_violations: int = 0
    evicted: dict[str, int] = field(default_factory=dict)
    blocked: dict[str, int] = field(default_factory=dict)
    publishes: int = 0
    trained_while_generating_s: float = 0.0
    busy_while_training_s: float = 0.0

    def max_staleness(self) -> int:
        return max((s for b in self.staleness for s in b), default=0)

    def engine_utilization(self, slots: int) -> float:
        return self.slot_seconds / max(slots * self.time, 1e-9)


def simulate(sched_cfg: SchedulerConfig, sim: SimConfig, sources: dict[str, list[str]], shape: Callable[[str, int], RolloutShape] | None = None) -> SimResult:
    shape = shape or default_shapes()
    queues = {s: list(ps) for s, ps in sources.items()}
    pool = DataPool()
    if sim.topology == "colocated_sync":
        sched_cfg = SchedulerConfig(**{**sched_cfg.__dict__, "admission_staleness": 1, "staleness_strategy": "none"})
    sched = Scheduler(sched_cfg, pool, {s: (lambda s=s: queues[s].pop(0) if queues[s] else None) for s in sources})
    res = SimResult(steps=0, time=0.0)
    ev: list[tuple[float, int, str, str]] = []
    seq = 0
    now = 0.0
    version = 0
    rs: dict[str, _R] = {}
    engine_q: list[str] = []
    active: set[str] = set()
    env_q: list[str] = []
    env_busy = 0
    engine_up = True
    training = False
    updates_since_publish = 0
    publish_pending: int | None = None  # disaggregated: version becoming servable at its event
    last_t = 0.0

    def push(t: float, kind: str, rid: str = "") -> int:
        nonlocal seq
        seq += 1
        heapq.heappush(ev, (t, seq, kind, rid))
        return seq

    cancelled: set[int] = set()

    def account(t: float) -> None:
        nonlocal last_t
        dt = t - last_t
        if dt > 0:
            res.slot_seconds += len(active) * dt
            if active:
                res.engine_busy_s += dt
                if training:
                    res.busy_while_training_s += dt
            if not engine_up:
                res.engine_down_s += dt
            if not training:
                res.trainer_idle_s += dt
        last_t = t

    def admit() -> None:
        for g in sched.admit(version):
            for i, rid in enumerate(g.members):
                rs[rid] = _R(rid, shape(g.meta["prompt"], i))
                pool.start(rid)
                engine_q.append(rid)
        live = sched.untrained_count()
        res.max_inflight_seen = max(res.max_inflight_seen, live)
        if live > sched_cfg.max_inflight():
            res.inflight_violations += 1

    def start_calls() -> None:
        while engine_up and engine_q and len(active) < sim.slots:
            rid = engine_q.pop(0)
            r = rs[rid]
            add, comp, _ = r.shape.turns[r.turn]
            prefill = r.context + add if r.aborted else add  # prefix cache holds the context, except after an abort (KV discarded)
            r.aborted = False
            dur = prefill / sim.prefill_tok_per_s + comp / sim.tok_per_s
            r.call_started, r.call_version = now, version
            active.add(rid)
            r.gen_event = push(now + dur, "gen_done", rid)

    def abort_all() -> None:
        for rid in list(active):
            r = rs[rid]
            if r.gen_event is not None:
                cancelled.add(r.gen_event)
            _, comp, _ = r.shape.turns[r.turn]
            res.wasted_decode_tokens += int(min(comp, (now - (r.call_started or now)) * sim.tok_per_s))
            res.reprefill_tokens += r.context + r.shape.turns[r.turn][0]
            pool.tags[rid].cuts += 1
            pool.tags[rid].reprefill_tokens += r.context
            r.call_started, r.call_version, r.gen_event, r.aborted = None, None, None, True
            active.discard(rid)
            engine_q.insert(0, rid)  # aborted calls resume first

    def try_train() -> None:
        nonlocal training, engine_up
        if training or res.steps >= sim.steps:
            return
        if res.steps == 0 and sim.topology != "colocated_sync" and sim.warmup_batches:
            ready = len(pool.terminal_groups())
            if ready < sched_cfg.B * (1 + sim.warmup_batches) and sched.live_count() > 0:
                return
        sel = sched.select(version)
        for reason in sel.evicted.values():
            res.evicted[reason] = res.evicted.get(reason, 0) + 1
        if sel.blocked:
            res.blocked[sel.blocked] = res.blocked.get(sel.blocked, 0) + 1
            admit()
            return
        res.staleness.append(sel.staleness)
        res.spans.append(sel.spans)
        res.batch_lengths.append([t.length for g in sel.groups for t in pool.members(g.gid, {"finished", "failure"})])
        res.batch_sources.append([g.source for g in sel.groups])
        tokens = sum(res.batch_lengths[-1])
        training = True
        if sim.topology in ("colocated_partial", "colocated_sync"):
            abort_all()
            engine_up = False
        sched.trained(sel)
        push(now + sim.train_s_base + sim.train_s_per_ktok * tokens / 1000, "train_done")

    push(0.0, "tick")
    while ev and now < sim.max_time and res.steps < sim.steps:
        t, sid, kind, rid = heapq.heappop(ev)
        if sid in cancelled:
            continue
        account(t)
        now = t
        if kind == "tick":
            admit()
        elif kind == "gen_done":
            r = rs[rid]
            active.discard(rid)
            add, comp, _env = r.shape.turns[r.turn]
            pool.call(rid, r.call_version if r.call_version is not None else version, comp)
            r.context += add + comp
            r.call_started, r.gen_event = None, None
            env_q.append(rid)
        elif kind == "env_done":
            env_busy -= 1
            r = rs[rid]
            r.turn += 1
            if r.turn >= len(r.shape.turns):
                pool.finish(rid, reward=r.shape.reward, status="masked" if r.shape.masked else "finished", now=now)
                admit()
            else:
                engine_q.append(rid)
        elif kind == "train_done":
            training = False
            res.steps += 1
            updates_since_publish += 1
            if updates_since_publish >= sim.publish_every:
                updates_since_publish = 0
                if sim.topology == "disaggregated_stream":
                    publish_pending = version + 1
                    push(now + sim.publish_s, "publish_done")
                else:
                    push(now + sim.publish_s, "publish_done")
            elif sim.topology != "disaggregated_stream":
                engine_up = True
        elif kind == "publish_done":
            version = publish_pending if publish_pending is not None else version + 1
            publish_pending = None
            res.publishes += 1
            engine_up = True
            admit()
        while env_q and env_busy < sim.env_workers:
            rid2 = env_q.pop(0)
            env_busy += 1
            push(now + rs[rid2].shape.turns[rs[rid2].turn][2], "env_done", rid2)
        start_calls()
        try_train()
        start_calls()
        if not ev and (engine_q or env_q):
            push(now + 1e-6, "tick")
    res.time = now
    res.evicted.update({k: v for k, v in sched.counters.items() if v and k not in res.evicted})
    return res


def ks_distance(a: list[int], b: list[int]) -> float:
    """Two-sample Kolmogorov–Smirnov statistic (length-bias monitor, checklist 29)."""
    if not a or not b:
        return 0.0
    xs = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    import bisect

    return max(abs(bisect.bisect_right(sa, x) / len(sa) - bisect.bisect_right(sb, x) / len(sb)) for x in xs)
