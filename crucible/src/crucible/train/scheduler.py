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
  rollouts age as well). The cap bounds *how much* work is ahead of the trainer, not how stale it is when trained: a
  long-tail group can sit in the buffer while newer groups train past it, so with ``staleness_strategy="none"`` the
  trained staleness grows well past S (M12 audit G1 H12b: S = 2 → 5, S = 4 → 10). Only ``wait`` / ``drop`` bound it
  (by τ). With S = 1 this is exactly synchronous. AReaL's ``max_head_offpolicyness`` η keeps η + 1 batches in flight:
  η ≡ S − 1, and ``staleness_strategy="none"`` with that S is AReaL's capacity-only mode (M12 audit G3 §1.1, pick 17);
- sources are chosen by deficit against ``mix`` weights (largest gap between target share and admitted share);
- ``group_redundancy`` r admits n + r members per group and keeps the first n scored (``trim_redundant``; ROLL's
  ``group_size_redundancy``, M12 audit G3 pick 12). Impossible configurations (a cap below one group, an unknown
  policy, τ < 0) are refused when the config is built, never found as a deadlock.

**Staleness unit.** τ counts *versions behind* the trained weights (v_train − the group's oldest call version), so
τ = 0 is on-policy. MiMo's replay-buffer threshold T counts versions *spanned* (lag + 1): MiMo's T maps to τ = T − 1,
so MiMo's published T = 4 is τ = 3 here (M12 audit G2 H10). The default τ = 4 is one version looser than MiMo's T = 4.

**Selection** (``select``) at the sample barrier S1, in order:
1. group backfill for infra-masked members: ``pad`` (train the rest; masked members get no advantage), ``refill_member``
   (submit a replacement; the group waits for it; at most ``max_refills`` per group, then the group is evicted as
   ``refill_exhausted``: one broken prompt never stalls the run, M12 audit H2 V10), ``drop_group``; a group with every
   member masked is evicted;
2. staleness: ``drop`` evicts terminal groups with staleness > τ (freed slots are refilled by ``admit``; after each
   publish ``evict_doomed`` also evicts *live* groups that can no longer be trained within τ, so their compute stops:
   prime-rl, M12 audit G4 pick 9); ``wait`` blocks selection while any live rollout has staleness ≥ τ, so the stale
   work finishes and is trained (no drop; MiMo "dropless");
3. dynamic sampling: groups whose rewards are all equal (n > 1) are evicted when ``filter_zero_variance`` (DAPO). With
   a group stage (``stage_view``) the decision reads the stage's output, not the raw rewards (a GRS / tournament stage
   separates all-pass groups: M12 audit G2 D-T2-4). At most ``zero_variance_max_rounds × B`` groups are evicted between
   two trained steps (DAPO's ``max_num_gen_batches``); past that the filter is suspended until the next step and
   ``zero_variance_exhausted`` is counted (a solved source no longer spins silently: M12 audit H2 V9);
4. **groups at their staleness bound** (staleness ≥ τ) are taken first and are exempt from the buffer policy's window
   and from ``max_offpolicy_frac``: waiting one more publish would exceed τ (M12 audit H2 V4, V6);
5. the buffer policy orders the rest: ``oldest_first`` (MiMo), ``greedy`` (earliest finished), or ``windowed_fifo``
   (MiniMax: the trainer takes only from the first W *untrained* groups in submission order, earliest finished first;
   each taken group leaves the queue, so the window slides as the batch fills; W = ⌈frac × max_inflight / n⌉ may be
   smaller than B, as in the article's batch 8 / W 4). Trained or evicted groups never hold window slots (the window
   counted raw submission indexes and deadlocked once holes left fewer than B finished groups inside: M12 audit H2 V5);
6. ``max_offpolicy_frac`` caps the share of the batch with staleness ≥ 1 among groups below their bound (DeepSeek);
7. ``drop_early_short``: during the first ``early_steps`` steps, groups shorter than the running length quantile are
   evicted (DeepSeek's length-bias control).
A batch is returned only when B candidates remain; otherwise the reason is reported (``need_more`` / ``wait_stale``).
The window and the off-policy cap are preferences, never a deadlock: when the batch is short, admission cannot add
work (the untrained cap is full or every source is closed) and nothing is live, the groups they held back fill the
batch (counted as ``preference_relaxed``; M12 audit H2 V4/V5, G3 D2).
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
    max_staleness: int = 4  # τ, versions behind (MiMo's T counts versions spanned: τ = T − 1; MiMo's T = 4 is τ = 3)
    max_staleness_by_source: dict[str, int] = field(default_factory=dict)  # per-source τ (m13 C-6, A2-54); default max_staleness
    staleness_strategy: str = "wait"  # wait | drop | none
    admission_staleness: int | None = None  # S for max_inflight = S·B·n (default τ)
    inflight_cap: dict[str, int] = field(default_factory=dict)  # per source, in rollouts
    mix: dict[str, float] = field(default_factory=dict)  # target source shares (default: uniform over sources)
    buffer_policy: str = "oldest_first"  # oldest_first | greedy | windowed_fifo
    fifo_window_frac: float = 0.3
    filter_zero_variance: bool = False
    group_backfill: str = "pad"  # pad | refill_member | drop_group
    max_refills: int | None = None  # refill_member: replacements per group before it is evicted (None: 2·n; M12 audit H2 V10)
    zero_variance_max_rounds: int | None = 10  # zero-variance evictions between steps ≤ this × B (None: unbounded; H2 V9)
    max_offpolicy_frac: float | None = None
    drop_early_short: float | None = None  # length quantile in (0, 1)
    early_steps: int = 0
    # groups at their staleness bound when more than B of them are selectable: they would exceed τ by waiting one more
    # publish. ``batch`` (default) trains them all this step (the batch grows past B, recorded), ``drop`` evicts the
    # overflow as stale, ``allow`` leaves the ones at exactly τ for the next step and trains every group already at τ + 1
    # (the batch grows past B): trained staleness ≤ τ + 1 exactly (it had no bound: S = 4 trained at τ + 3, M12 audit H2 V3)
    bound_overflow: str = "batch"
    # ROLL ``group_size_redundancy`` (M12 audit G3 pick 12): each group is admitted with n + r members and trains the
    # first n to be scored; the r slowest are evicted (cancelled) the moment the n-th member finishes, so one long tail
    # never holds its group. It selects for speed, so it biases toward short responses: ``length_bias_ks`` (the length-
    # bias monitor) measures that shift, and the evictions are counted as ``evicted_redundant``. 0 = off.
    group_redundancy: int = 0

    def group_size(self) -> int:
        """Members admitted per group (n + r)."""
        return self.n + self.group_redundancy

    def __post_init__(self) -> None:
        """Impossible configurations are refused here, never discovered as a deadlock at run time."""
        if self.n < 1 or self.B < 1:
            raise ValueError(f"n and B must be ≥ 1 (got n={self.n}, B={self.B})")
        if self.max_staleness < 0 or any(v < 0 for v in self.max_staleness_by_source.values()):
            raise ValueError("τ (max_staleness, max_staleness_by_source) must be ≥ 0")
        if self.group_backfill not in ("pad", "refill_member", "drop_group"):
            raise ValueError(f"group_backfill: pad | refill_member | drop_group (got {self.group_backfill!r})")
        if self.group_redundancy < 0:
            raise ValueError("group_redundancy must be ≥ 0")
        if self.max_inflight() < self.group_size():
            raise ValueError(f"max_inflight S·B·n = {self.max_inflight()} < n + group_redundancy = {self.group_size()}: no group could ever be admitted")
        small = {s: c for s, c in self.inflight_cap.items() if c < self.group_size()}
        if small:
            raise ValueError(f"inflight_cap below one group ({self.group_size()} rollouts) for {small}: those sources could never admit")
        if self.max_refills is not None and self.max_refills < 0:
            raise ValueError("max_refills must be ≥ 0")
        if self.max_offpolicy_frac is not None and not 0 <= self.max_offpolicy_frac <= 1:
            raise ValueError("max_offpolicy_frac must be in [0, 1]")
        if self.drop_early_short is not None and not 0 < self.drop_early_short < 1:
            raise ValueError("drop_early_short is a length quantile in (0, 1)")
        if self.bound_overflow not in ("batch", "drop", "allow"):
            raise ValueError(f"bound_overflow: batch | drop | allow (got {self.bound_overflow!r})")
        if self.buffer_policy not in ("oldest_first", "greedy", "windowed_fifo"):
            raise ValueError(f"buffer_policy: oldest_first | greedy | windowed_fifo (got {self.buffer_policy!r})")
        if self.staleness_strategy not in ("wait", "drop", "none"):
            raise ValueError(f"staleness_strategy: wait | drop | none (got {self.staleness_strategy!r})")

    def tau(self, source: str) -> int:
        return int(self.max_staleness_by_source.get(source, self.max_staleness))

    def max_inflight(self) -> int:
        s = self.admission_staleness if self.admission_staleness is not None else self.max_staleness
        return max(1, s) * self.B * self.n

    def window(self) -> int:
        """W, in untrained groups (``windowed_fifo``)."""
        return max(1, math.ceil(self.fifo_window_frac * self.max_inflight() / self.n))

    def refill_cap(self) -> int:
        return self.max_refills if self.max_refills is not None else 2 * self.n


@dataclass
class Selection:
    groups: list[Group]
    evicted: dict[str, str]  # gid -> reason
    blocked: str | None = None  # need_more | wait_stale
    staleness: list[int] = field(default_factory=list)
    spans: list[int] = field(default_factory=list)
    worst: list[int] = field(default_factory=list)  # per selected group: its stalest member (checklist #24 staleness_worst)
    evicted_staleness: list[int] = field(default_factory=list)  # per group evicted for staleness at this barrier
    staleness_by_source: dict[str, list[int]] = field(default_factory=dict)


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
        # the group stage's view of a terminal group for the zero-variance filter (set by the runtime when a ``group_fn``
        # runs): the values to compare (post-stage rewards or advantages), or None while the stage is still running
        self.stage_view: Callable[[str], list[float | None] | None] | None = None
        self._zv_since_step = 0
        self._zv_exhausted = False

    # ---- admission ---------------------------------------------------------------------------------------------
    def _cap(self, source: str) -> int:
        return self.cfg.inflight_cap.get(source, self.cfg.max_inflight())

    def free_slots(self, source: str) -> int:
        return self._cap(source) - self.pool.untrained_count(source)

    def admit(self, version: int) -> list[Group]:
        out = []
        self.cap_blocked = False  # the untrained cap (not the sources) stopped admission (checklist #24 admission_blocked_s)
        k = self.cfg.group_size()
        while True:
            if self.pool.untrained_count() + k > self.cfg.max_inflight():
                self.cap_blocked = any(s not in self.exhausted for s in self.sources)
                break
            open_src = [s for s in self.sources if s not in self.exhausted and self.free_slots(s) >= k]
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
            out.append(self.pool.submit_group(gid, src, k, version, backfill=self.cfg.group_backfill, meta={"prompt": pid}))
            self.admitted[src] += 1
        return out

    # ---- selection at the sample barrier ------------------------------------------------------------------------
    def _backfill(self, version: int, evicted: dict[str, str]) -> None:
        if not self.pool.has_masked():
            return
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
            elif g.backfill == "refill_member" and g.refills + len(masked) > self.cfg.refill_cap():
                # M12 audit H2 V10 / G3 D5: a prompt that masks on every attempt was refilled at every barrier forever
                # (1,779 refills, the run stalled on wait_stale or tripped the breaker); past the cap it is dropped
                self.pool.evict_group(g.gid)
                evicted[g.gid] = "refill_exhausted"
                self.counters["evicted_masked"] += 1
                self.counters["refill_exhausted"] = self.counters.get("refill_exhausted", 0) + 1
            elif g.backfill == "refill_member":
                for t in masked:
                    self.pool._set_status(t.rid, "evicted")
                    self.pool.refill_member(g.gid, version)
                    self.counters["refilled_members"] += 1

    def select(self, v_train: int, *, min_groups: int | None = None) -> Selection:
        """``min_groups``: the fewest groups a batch may have (default B). A budget-driven partial-rollout barrier (Kimi λ,
        C3PO++ Φ) trains whatever has finished, up to B groups, once its round boundary is reached."""
        c = self.cfg
        evicted: dict[str, str] = {}
        self._backfill(v_train, evicted)
        ev_st = self._ev_pending  # evictions at blocked barriers count too: reported with the next delivered batch
        if c.staleness_strategy == "drop":
            for g in self.pool.terminal_groups():
                gs = self.pool.group_staleness(g.gid, v_train)
                if gs > c.tau(g.source):
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "stale"
                    ev_st.append(gs)
                    self.counters["evicted_stale"] += 1
        elif c.staleness_strategy == "wait" and any((o := self.pool.min_live_oldest(s)) is not None and v_train - o >= c.tau(s)
                                                    for s in self.pool.live_sources()):
            # a group's staleness counts its finished members too: they age while the group waits for its last runner
            # (per source: the oldest version over groups with a live member, from the pool's lazy heap, not a scan)
            return Selection([], evicted, blocked="wait_stale")
        if c.filter_zero_variance and c.n > 1:
            self._filter_zero_variance(evicted)
        if c.drop_early_short is not None and self.steps < c.early_steps and self.lengths_seen:
            qs = sorted(self.lengths_seen)
            cut = qs[min(len(qs) - 1, int(c.drop_early_short * len(qs)))]
            for g in self.pool.terminal_groups():
                ls = [t.length for t in self.pool.members(g.gid, {"finished", "failure"})]
                if ls and sum(ls) / len(ls) < cut:
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "early_short"
                    self.counters["evicted_short"] += 1
        if c.max_staleness_by_source and c.staleness_strategy != "none":
            # per-source τ: finished groups keep aging in the buffer, which "wait" alone does not see; a group that could
            # not be trained within its bound is dropped as stale, never trained over it
            for g in self.pool.terminal_groups():
                gs = self.pool.group_staleness(g.gid, v_train)
                if gs > c.tau(g.source):
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "stale"
                    ev_st.append(gs)
                    self.counters["evicted_stale"] += 1
        cands = self.pool.terminal_groups()
        st = {g.gid: self.pool.group_staleness(g.gid, v_train) for g in cands}
        done_at = ({g.gid: max((t.done_at or 0.0) for t in self.pool.members(g.gid)) for g in cands}
                   if c.buffer_policy in ("greedy", "windowed_fifo") else {})
        if c.buffer_policy == "oldest_first":
            cands.sort(key=lambda g: g.submit_index)
        else:
            cands.sort(key=lambda g: (done_at[g.gid], g.submit_index))
        need = c.B if min_groups is None else max(1, min(min_groups, c.B))
        bounded = c.staleness_strategy != "none"
        # Groups at their staleness bound train first, whatever the window or the off-policy cap say (M12 audit H2 V4, V6:
        # both filters ran before this rule, so a held-back group aged past τ, to 8 at τ = 3); furthest past the bound
        # first (``allow`` lets a group reach τ + 1, and that one must train now), stable within the buffer policy
        forced = sorted((g for g in cands if bounded and st[g.gid] >= c.tau(g.source)), key=lambda g: c.tau(g.source) - st[g.gid])
        forced_ids = {g.gid for g in forced}
        batch = forced[: c.B]
        taken = {g.gid for g in batch}
        rest = [g for g in cands if g.gid not in forced_ids]
        off_left = None
        if c.max_offpolicy_frac is not None:
            off_left = max(0, math.floor(c.max_offpolicy_frac * c.B) - sum(1 for g in batch if st[g.gid] > 0))
        batch += self._pick(rest, c.B - len(batch), taken, st, done_at, off_left)
        if len(batch) < need and self._admission_closed() and self.pool.live_count() == 0:
            # nothing can arrive: the groups the window / off-policy cap held back fill the batch (they occupy the slots
            # that on-policy work would need: G3 D2, every buffered group off-policy deadlocked disaggregated_stream)
            extra = [g for g in rest if g.gid not in taken][: c.B - len(batch)]
            batch += extra
            taken |= {g.gid for g in extra}
            self.counters["preference_relaxed"] = self.counters.get("preference_relaxed", 0) + len(extra)
        if len(batch) < need:
            return Selection([], evicted, blocked="need_more")
        over = forced[c.B:]
        if over and bounded:
            if c.bound_overflow == "batch":
                batch = batch + over
                self.counters["bound_overflow_trained"] = self.counters.get("bound_overflow_trained", 0) + len(over)
            elif c.bound_overflow == "drop":
                for g in over:
                    self.pool.evict_group(g.gid)
                    evicted[g.gid] = "stale"
                    ev_st.append(st[g.gid])
                    self.counters["evicted_stale"] += 1
            else:  # allow: a group at exactly τ may wait one more publish (τ + 1); one already at τ + 1 trains now
                late = [g for g in over if st[g.gid] > c.tau(g.source)]
                batch = batch + late
                self.counters["bound_overflow_trained"] = self.counters.get("bound_overflow_trained", 0) + len(late)
        stale, sp, worst, by_src = [], [], [], {}
        for g in batch:
            gst = []
            for t in self.pool.members(g.gid, {"finished", "failure"}):
                gst.append(t.staleness(v_train))
                by_src.setdefault(g.source, []).append(gst[-1])
                sp.append(t.spans())
                self.lengths_seen.append(t.length)
            stale += gst
            if gst:
                worst.append(max(gst))
        self._ev_pending = []
        return Selection(batch, evicted, staleness=stale, spans=sp, worst=worst, evicted_staleness=ev_st, staleness_by_source=by_src)

    def _filter_zero_variance(self, evicted: dict[str, str]) -> None:
        c = self.cfg
        limit = None if c.zero_variance_max_rounds is None else c.zero_variance_max_rounds * c.B
        for g in self.pool.terminal_groups():
            if limit is not None and self._zv_since_step >= limit:
                if not self._zv_exhausted:  # once per suspension: the metric counts episodes, not barriers
                    self._zv_exhausted = True
                    self.counters["zero_variance_exhausted"] = self.counters.get("zero_variance_exhausted", 0) + 1
                return
            if self.stage_view is not None:
                rs = self.stage_view(g.gid)
                if rs is None:
                    continue  # the group stage is still running: decided on its output, not the raw rewards
            else:
                rs = [t.reward for t in self.pool.members(g.gid, {"finished", "failure"})]
            if len(rs) > 1 and len(set(rs)) == 1:
                self.pool.evict_group(g.gid)
                evicted[g.gid] = "zero_variance"
                self.counters["evicted_zero_var"] += 1
                self._zv_since_step += 1

    def _admission_closed(self) -> bool:
        """``admit`` cannot add a group now: the untrained cap is full, or every source is exhausted or at its cap."""
        c = self.cfg
        if self.pool.untrained_count() + c.group_size() > c.max_inflight():
            return True
        return not any(s not in self.exhausted and self.free_slots(s) >= c.group_size() for s in self.sources)

    def _pick(self, rest: list[Group], room: int, taken: set[str], st: dict[str, int], done_at: dict[str, float], off_left: int | None) -> list[Group]:
        """Up to ``room`` groups from ``rest`` (already in buffer-policy order), honouring the off-policy budget and, for
        ``windowed_fifo``, the window over untrained groups not yet taken (MiniMax: the head advances as items leave)."""
        out: list[Group] = []
        if room <= 0:
            return out

        def ok(g: Group) -> bool:
            return off_left is None or st[g.gid] == 0 or off_left > 0

        if self.cfg.buffer_policy != "windowed_fifo":
            for g in rest:
                if len(out) >= room:
                    break
                if ok(g):
                    out.append(g)
                    if st[g.gid] > 0 and off_left is not None:
                        off_left -= 1
            return out
        W = self.cfg.window()
        eligible = {g.gid: g for g in rest}
        queue = sorted((g for g in self.pool.groups.values() if g.status in ("open", "terminal")), key=lambda g: g.submit_index)
        while len(out) < room:
            window = [g for g in queue if g.gid not in taken][:W]
            options = [eligible[g.gid] for g in window if g.gid in eligible and ok(eligible[g.gid])]
            if not options:
                break
            g = min(options, key=lambda x: (done_at[x.gid], x.submit_index))
            out.append(g)
            taken.add(g.gid)
            if st[g.gid] > 0 and off_left is not None:
                off_left -= 1
        return out

    def evict_doomed(self, v_next: int) -> list[str]:
        """``drop`` only, after a publish: evict every live group whose oldest version is already more than τ behind
        ``v_next``. It would be evicted the moment it turned terminal; stopping it now frees its compute and its slots
        (prime-rl cancels doomed in-flight groups before the weight swap; M12 audit G4 pick 9). Returns the evicted rids
        still to cancel; counted as ``evicted_doomed`` and in the evicted-staleness metric."""
        c = self.cfg
        if c.staleness_strategy != "drop":
            return []
        out: list[str] = []
        for gid in self.pool.live_groups():
            g = self.pool.groups.get(gid)
            if g is None or g.status in ("trained", "evicted"):
                continue
            # under ``refill_member`` a masked member leaves the group before the staleness check (``_backfill`` runs
            # first), so it cannot doom its group; under ``pad`` / ``drop_group`` it stays and counts
            skip = ("evicted", "masked") if g.backfill == "refill_member" else ("evicted",)
            vs = [t.oldest() for t in self.pool.members(gid) if t.status not in skip]
            if not vs:
                continue
            gs = v_next - min(vs)
            if gs > c.tau(g.source):
                out += self.pool.evict_group(gid)
                self._ev_pending.append(gs)
                self.counters["evicted_doomed"] = self.counters.get("evicted_doomed", 0) + 1
        return out

    def trim_redundant(self, gid: str) -> list[str]:
        """``group_redundancy``: once n members of ``gid`` are scored (finished or failure), every other member still live
        is evicted (the caller cancels it) and masked spares leave too, so the group is terminal with exactly n scored
        members. Returns the evicted rids that were live."""
        c = self.cfg
        if c.group_redundancy <= 0:
            return []
        g = self.pool.groups.get(gid)
        if g is None or g.status in ("trained", "evicted"):
            return []
        ms = self.pool.members(gid)
        if sum(1 for t in ms if t.status in ("finished", "failure")) < c.n:
            return []
        live = [t.rid for t in ms if t.status in LIVE]
        spare = live + [t.rid for t in ms if t.status == "masked"]
        if not spare:
            return []
        self.pool.evict_members(gid, spare)
        self.counters["evicted_redundant"] = self.counters.get("evicted_redundant", 0) + len(spare)
        return live

    def trained(self, sel: Selection) -> None:
        for g in sel.groups:
            self.pool.mark_trained(g.gid)
        self.steps += 1
        self._zv_since_step, self._zv_exhausted = 0, False

    def live_count(self) -> int:
        return self.pool.live_count()

    def untrained_count(self) -> int:
        return self.pool.untrained_count()
