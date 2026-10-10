"""The scheduler: admission, staleness, buffer policy and batch selection (research m11-m12/06 §2.5–2.8, §3.1–3.5,
checklist items 15–30). Pure and deterministic: given the pool's tags and the training version it decides, it never
waits or sleeps itself. The discrete-event simulator (``sim.py``) and the asyncio runtime (``runtime.py``) drive the
same object, so every scheduling rule is tested without GPUs.

**Dispatch unit = one rollout.** Groups are admitted whole and their advantage is computed only when terminal, but
members run independently (no group barrier: DeepSeek-V4.1 found batch-level dispatch oscillates and group-level
dispatch is held by the in-group long tail).

**Admission** (``admit``), two caps layered:
- per source ``inflight_cap[source]`` (DeepSeek: a group of n is admitted when ``free_slots ≥ n``; it also fixes the
  source mix against length bias);
- global ``max_inflight = S × B × n`` **untrained** rollouts — generating *or* finished and waiting in the buffer (Ling's
  staleness manager counts in-flight rollouts; counting the buffer too is what makes the bound hold, because buffered
  rollouts age as well). The trainer takes the B oldest groups per step, so a newly admitted rollout is trained within
  about S steps (S = ``admission_staleness``, default ``max_staleness``). With S = 1 this is exactly synchronous.
- sources are chosen by deficit against ``mix`` weights (largest gap between target share and admitted share).

**Selection** (``select``) at the sample barrier S1, in order:
1. group backfill for infra-masked members: ``pad`` (train the rest; masked members get no advantage), ``refill_member``
   (submit a replacement; the group waits for it), ``drop_group``; a group with every member masked is evicted;
2. staleness: ``drop`` evicts terminal groups with staleness > τ (freed slots are refilled by ``admit``);
   ``wait`` blocks selection while any live rollout has staleness ≥ τ, so the stale work finishes and is trained
   (no drop; MiMo "dropless");
3. dynamic sampling: groups whose rewards are all equal (n > 1) are evicted when ``filter_zero_variance`` (DAPO);
4. the buffer policy orders the candidates: ``oldest_first`` (MiMo), ``greedy`` (earliest finished), or
   ``windowed_fifo`` (MiniMax: only groups whose submission index < head + W, W = ⌈frac × N⌉, earliest finished first);
5. ``max_offpolicy_frac`` caps the share of the batch with staleness ≥ 1 (DeepSeek);
6. ``drop_early_short``: during the first ``early_steps`` steps, groups shorter than the running length quantile are
   evicted (DeepSeek's length-bias control).
A batch is returned only when B candidates remain; otherwise the reason is reported (``need_more`` / ``wait_stale``).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from crucible.train.pool import LIVE, DataPool, Group


@dataclass
class SchedulerConfig:
    n: int = 8  # group size (1 = critic mode, no groups)
    B: int = 4  # groups per optimizer step
    max_staleness: int = 4  # τ (MiMo's published value)
    staleness_strategy: str = "wait"  # wait | drop | none
    admission_staleness: int | None = None  # S for max_inflight = S·B·n (default τ)
    inflight_cap: dict[str, int] = field(default_factory=dict)  # per source, in rollouts
    mix: dict[str, float] = field(default_factory=dict)  # target source shares (default: uniform over sources)
    buffer_policy: str = "oldest_first"  # oldest_first | greedy | windowed_fifo
    fifo_window_frac: float = 0.3
    filter_zero_variance: bool = False
    group_backfill: str = "pad"  # pad | refill_member | drop_group
    max_offpolicy_frac: float | None = None
    drop_early_short: float | None = None  # length quantile in (0, 1)
    early_steps: int = 0

    def max_inflight(self) -> int:
        s = self.admission_staleness if self.admission_staleness is not None else self.max_staleness
        return max(1, s) * self.B * self.n

    def window(self) -> int:
        return max(1, math.ceil(self.fifo_window_frac * self.max_inflight() / self.n))


@dataclass
class Selection:
    groups: list[Group]
    evicted: dict[str, str]  # gid -> reason
    blocked: str | None = None  # need_more | wait_stale
    staleness: list[int] = field(default_factory=list)
    spans: list[int] = field(default_factory=list)
    worst: list[int] = field(default_factory=list)  # per selected group: its stalest member (checklist #24 staleness_worst)
    evicted_staleness: list[int] = field(default_factory=list)  # per group evicted for staleness at this barrier


class Scheduler:
    def __init__(self, cfg: SchedulerConfig, pool: DataPool, sources: dict[str, Callable[[], str | None]]):
        """``sources``: name -> callable returning the next prompt id (None when the source is exhausted)."""
        self.cfg, self.pool, self.sources = cfg, pool, sources
        self.admitted: dict[str, int] = {s: 0 for s in sources}
        self.exhausted: set[str] = set()
        self.steps = 0
        self.lengths_seen: list[int] = []
        self.counters = {"evicted_stale": 0, "evicted_zero_var": 0, "evicted_short": 0, "evicted_masked": 0, "refilled_members": 0}
        self._ev_pending: list[int] = []

    # ---- admission ---------------------------------------------------------------------------------------------
    def _cap(self, source: str) -> int:
        return self.cfg.inflight_cap.get(source, self.cfg.max_inflight())

    def free_slots(self, source: str) -> int:
        return self._cap(source) - self.pool.untrained_count(source)

    def admit(self, version: int) -> list[Group]:
        out = []
        self.cap_blocked = False  # the untrained cap (not the sources) stopped admission (checklist #24 admission_blocked_s)
        while True:
            if self.pool.untrained_count() + self.cfg.n > self.cfg.max_inflight():
                self.cap_blocked = any(s not in self.exhausted for s in self.sources)
                break
            open_src = [s for s in self.sources if s not in self.exhausted and self.free_slots(s) >= self.cfg.n]
            if not open_src:
                break
            total = sum(self.admitted.values()) or 1
            w = self.cfg.mix or {s: 1.0 for s in self.sources}
            wsum = sum(w.get(s, 0.0) for s in self.sources) or 1.0
            src = max(open_src, key=lambda s: (w.get(s, 0.0) / wsum - self.admitted[s] / total, -self.admitted[s], s))
            pid = self.sources[src]()
            if pid is None:
                self.exhausted.add(src)
                continue
            gid = f"{src}:{pid}@{self.pool._next_index}"  # monotonic: released groups never let an id repeat
            out.append(self.pool.submit_group(gid, src, self.cfg.n, version, backfill=self.cfg.group_backfill, meta={"prompt": pid}))
            self.admitted[src] += 1
        return out

    # ---- selection at the sample barrier ------------------------------------------------------------------------
    def _backfill(self, version: int, evicted: dict[str, str]) -> None:
        for g in self.pool.terminal_groups():
            ms = self.pool.members(g.gid)
            masked = [t for t in ms if t.status == "masked"]
            if not masked:
                continue
            if len(masked) == len([t for t in ms if t.status != "evicted"]):
                self.pool.evict_group(g.gid)
                evicted[g.gid] = "all_masked"
                self.counters["evicted_masked"] += 1
            elif g.backfill == "drop_group":
                self.pool.evict_group(g.gid)
                evicted[g.gid] = "masked_member"
                self.counters["evicted_masked"] += 1
            elif g.backfill == "refill_member":
                for t in masked:
                    self.pool._set_status(t.rid, "evicted")
                    self.pool.refill_member(g.gid, version)
                    self.counters["refilled_members"] += 1

    def select(self, v_train: int) -> Selection:
        c = self.cfg
        evicted: dict[str, str] = {}
        self._backfill(v_train, evicted)
        ev_st = self._ev_pending  # evictions at blocked barriers count too: reported with the next delivered batch
        if c.staleness_strategy == "drop":
            for g in self.pool.terminal_groups():
                gs = self.pool.group_staleness(g.gid, v_train)
                if gs > c.max_staleness:
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "stale"
                    ev_st.append(gs)
                    self.counters["evicted_stale"] += 1
        elif c.staleness_strategy == "wait" and any(t.staleness(v_train) >= c.max_staleness for t in self.pool.live()):
            return Selection([], evicted, blocked="wait_stale")
        if c.filter_zero_variance and c.n > 1:
            for g in self.pool.terminal_groups():
                rs = [t.reward for t in self.pool.members(g.gid, {"finished", "failure"})]
                if len(rs) > 1 and len(set(rs)) == 1:
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "zero_variance"
                    self.counters["evicted_zero_var"] += 1
        if c.drop_early_short is not None and self.steps < c.early_steps and self.lengths_seen:
            qs = sorted(self.lengths_seen)
            cut = qs[min(len(qs) - 1, int(c.drop_early_short * len(qs)))]
            for g in self.pool.terminal_groups():
                ls = [t.length for t in self.pool.members(g.gid, {"finished", "failure"})]
                if ls and sum(ls) / len(ls) < cut:
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "early_short"
                    self.counters["evicted_short"] += 1
        cands = self.pool.terminal_groups()
        done_at = {g.gid: max((t.done_at or 0.0) for t in self.pool.members(g.gid)) for g in cands}
        if c.buffer_policy == "oldest_first":
            cands.sort(key=lambda g: g.submit_index)
        elif c.buffer_policy == "greedy":
            cands.sort(key=lambda g: (done_at[g.gid], g.submit_index))
        elif c.buffer_policy == "windowed_fifo":
            pending_idx = [g.submit_index for g in self.pool.groups.values() if g.status in ("open", "terminal")]
            head = min(pending_idx) if pending_idx else 0
            cands = [g for g in cands if g.submit_index < head + c.window()]
            cands.sort(key=lambda g: (done_at[g.gid], g.submit_index))
        else:
            raise ValueError(c.buffer_policy)
        if c.max_offpolicy_frac is not None:
            limit = math.floor(c.max_offpolicy_frac * c.B)
            on = [g for g in cands if self.pool.group_staleness(g.gid, v_train) == 0]
            off = [g for g in cands if self.pool.group_staleness(g.gid, v_train) > 0][:limit]
            keep = {g.gid for g in on + off}
            cands = [g for g in cands if g.gid in keep]
        if len(cands) < c.B:
            return Selection([], evicted, blocked="need_more")
        batch = cands[: c.B]
        st, sp, worst = [], [], []
        for g in batch:
            gst = []
            for t in self.pool.members(g.gid, {"finished", "failure"}):
                gst.append(t.staleness(v_train))
                sp.append(t.spans())
                self.lengths_seen.append(t.length)
            st += gst
            if gst:
                worst.append(max(gst))
        self._ev_pending = []
        return Selection(batch, evicted, staleness=st, spans=sp, worst=worst, evicted_staleness=ev_st)

    def trained(self, sel: Selection) -> None:
        for g in sel.groups:
            self.pool.mark_trained(g.gid)
        self.steps += 1

    def live_count(self) -> int:
        return sum(1 for t in self.pool.tags.values() if t.status in LIVE)

    def untrained_count(self) -> int:
        return self.pool.untrained_count()
