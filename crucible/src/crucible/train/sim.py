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
    topology: str = "colocated_partial"  # colocated_sync | colocated_partial | disaggregated_stream | elastic
    elastic_frac: float = 0.5  # elastic: share of the engine slots handed to the trainer at each barrier (DORA)
    partial_barrier: str = "groups"  # groups | completion_frac | token_budget (see runtime.RuntimeConfig)
    barrier_frac: float = 0.8
    barrier_tokens: int = 0
    max_cuts: int | None = None  # retention σ
    resume_keeps_kv: bool = False  # an aborted call resumes its decode where it stopped, without re-prefill (DeepSeek-V4.1)
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
    done_decode: int = 0  # resume_keeps_kv: decode tokens of the cut call already produced (kept, not redone)


@dataclass
class SimResult:
    steps: int
    time: float
    staleness: list[list[int]] = field(default_factory=list)
    staleness_by_source: dict[str, list[int]] = field(default_factory=dict)
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
    batch_sizes: list[int] = field(default_factory=list)  # groups per trained batch (budget barriers train fewer than B)

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
    publishing = False
    last_t = 0.0
    round_tokens = 0
    elastic_slots = round(sim.slots * sim.elastic_frac) if sim.topology == "elastic" else 0
    if sim.topology == "elastic" and not 0 < elastic_slots < sim.slots:
        raise ValueError("elastic needs 0 < elastic_frac·slots < slots")

    def capacity() -> int:
        return sim.slots - elastic_slots if training and sim.topology == "elastic" else sim.slots

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
        while engine_up and engine_q and len(active) < capacity():
            rid = engine_q.pop(0)
            r = rs[rid]
            add, comp, _ = r.shape.turns[r.turn]
            prefill = r.context + add if r.aborted else add  # prefix cache holds the context, except after an abort (KV discarded)
            if r.done_decode:  # resumed with its KV: no prefill, only the rest of the decode
                prefill, comp = 0, comp - r.done_decode
            r.aborted = False
            dur = prefill / sim.prefill_tok_per_s + comp / sim.tok_per_s
            r.call_started, r.call_version = now, version
            active.add(rid)
            r.gen_event = push(now + dur, "gen_done", rid)

    def abort_all(keep: int = 0) -> None:
        """Cut in-flight calls (all, or all but the ``keep`` earliest-started: the elastic resident group's share)."""
        victims = sorted(active, key=lambda x: rs[x].call_started or 0.0)[keep:]
        for rid in victims:
            r = rs[rid]
            if r.gen_event is not None:
                cancelled.add(r.gen_event)
            _, comp, _ = r.shape.turns[r.turn]
            produced = int(min(comp - r.done_decode, (now - (r.call_started or now)) * sim.tok_per_s))
            if sim.resume_keeps_kv:
                r.done_decode += produced
            else:
                res.wasted_decode_tokens += produced
                res.reprefill_tokens += r.context + r.shape.turns[r.turn][0]
                pool.tags[rid].reprefill_tokens += r.context
                r.done_decode = 0
            r.call_started, r.call_version, r.gen_event, r.aborted = None, None, None, not sim.resume_keeps_kv
            active.discard(rid)
            engine_q.insert(0, rid)  # aborted calls resume first

    def cut_live() -> None:
        over = set()
        for rid, r in rs.items():
            t = pool.tags.get(rid)
            if t is not None and t.status == "running":
                t.cuts += 1
                if sim.max_cuts is not None and t.cuts > sim.max_cuts:
                    over.add(t.group)
        for gid in over:
            for rid in pool.evict_group(gid):
                if rid in active:
                    r = rs[rid]
                    if r.gen_event is not None:
                        cancelled.add(r.gen_event)
                    active.discard(rid)
                for q in (engine_q, env_q):
                    while rid in q:
                        q.remove(rid)
                rs.pop(rid, None)
            res.evicted["retention"] = res.evicted.get("retention", 0) + 1

    def boundary() -> bool:
        if sim.partial_barrier == "token_budget":
            return round_tokens >= sim.barrier_tokens or not (active or engine_q or env_q)
        if sim.partial_barrier == "completion_frac":
            untrained = [t for t in pool.untrained() if t.status != "evicted"]
            done = sum(1 for t in untrained if t.status in ("finished", "failure", "masked"))
            return not untrained or done / len(untrained) >= sim.barrier_frac
        return True

    def try_train() -> None:
        nonlocal training, engine_up
        if training or res.steps >= sim.steps:
            return
        if publishing and sim.topology in ("colocated_partial", "colocated_sync"):
            return  # the GPU is still busy with the publish: the next step cannot start (it would overlap the weight load)
        if res.steps == 0 and sim.topology != "colocated_sync" and sim.warmup_batches:
            ready = len(pool.terminal_groups())
            if ready < sched_cfg.B * (1 + sim.warmup_batches) and sched.live_count() > 0:
                return
        if not boundary():
            return
        # staleness is measured against the trainer's committed version: a publish still in flight (disaggregated /
        # elastic publish asynchronously) already holds the weights the next batch trains from
        v_commit = publish_pending if publish_pending is not None else version
        sel = sched.select(v_commit, min_groups=None if sim.partial_barrier == "groups" else 1)
        for reason in sel.evicted.values():
            res.evicted[reason] = res.evicted.get(reason, 0) + 1
        if sel.blocked:
            res.blocked[sel.blocked] = res.blocked.get(sel.blocked, 0) + 1
            admit()
            return
        res.staleness.append(sel.staleness)
        for src, xs in sel.staleness_by_source.items():
            res.staleness_by_source.setdefault(src, []).extend(xs)
        res.spans.append(sel.spans)
        res.batch_lengths.append([t.length for g in sel.groups for t in pool.members(g.gid, {"finished", "failure"})])
        res.batch_sources.append([g.source for g in sel.groups])
        tokens = sum(res.batch_lengths[-1])
        res.batch_sizes.append(len(sel.groups))
        nonlocal round_tokens
        round_tokens = 0
        if sim.topology in ("colocated_partial", "colocated_sync"):
            abort_all()
            engine_up = False
        elif sim.topology == "elastic":  # the elastic slots go to the trainer; the resident ones keep their calls
            abort_all(keep=sim.slots - elastic_slots)
        training = True
        if sim.topology in ("colocated_partial", "elastic"):
            cut_live()
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
            if rid not in rs:  # evicted (retention) while its call ran
                continue
            r = rs[rid]
            active.discard(rid)
            add, comp, _env = r.shape.turns[r.turn]
            pool.call(rid, r.call_version if r.call_version is not None else version, comp)
            round_tokens += comp - r.done_decode
            r.context += add + comp
            r.call_started, r.gen_event, r.done_decode = None, None, 0
            env_q.append(rid)
        elif kind == "env_done":
            env_busy -= 1
            if rid not in rs:
                continue
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
                publishing = True
                if sim.topology in ("disaggregated_stream", "elastic"):
                    publish_pending = version + 1
                push(now + sim.publish_s, "publish_done")
            elif sim.topology not in ("disaggregated_stream", "elastic"):
                engine_up = True
        elif kind == "publish_done":
            version = publish_pending if publish_pending is not None else version + 1
            publish_pending = None
            publishing = False
            res.publishes += 1
            engine_up = True
            admit()
        while env_q and env_busy < sim.env_workers:
            rid2 = env_q.pop(0)
            if rid2 not in rs:
                continue
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
