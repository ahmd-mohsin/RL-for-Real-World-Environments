"""The data pool (research m11-m12/06 §6.1, checklist 39): tags separate from values.

A **tag** is the small, indexed record the scheduler decides on: status, group, source, harness, submission index and
version, every policy version the rollout's model calls touched, length, reward, mask, finish time. The **value** is the
``crucible.rollout/v1`` record (or a reference to it), fetched only when a batch is built. Scheduling reads tags only
(MiMo TransferQueue tag/value split; MiniMax Forge Data Pool).

Status machine of one rollout::

    pending --admit--> running --finish--> finished
                          |-----fail----> failure     (agent-caused or policy failure: trainable, reward 0)
                          |-----mask----> masked      (infrastructure: never trained, reward None)
                          `-----evict---> evicted     (staleness drop / dynamic-sampling filter / group dropped)
    finished|failure --train--> trained

A **group** (GRPO prompt group of n samples, or n = 1 in critic mode) is terminal when every member is finished,
failure or masked. Groups are an admission and advantage unit, never a dispatch barrier: members run independently.
"""

from __future__ import annotations

import heapq
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

TERMINAL = {"finished", "failure", "masked"}
LIVE = {"pending", "running", "grading"}  # grading: generation done, reward not yet in (checklist #40): untrained, not selectable


@dataclass
class Tag:
    rid: str
    group: str
    source: str
    submit_index: int
    submit_version: int
    harness: str = ""
    status: str = "pending"
    versions: list[int] = field(default_factory=list)  # policy versions of the model calls made so far, in order
    length: int = 0  # completion tokens so far
    reward: float | None = None
    done_at: float | None = None
    cuts: int = 0  # times interrupted at a sample barrier (colocated partial rollout)
    reprefill_tokens: int = 0

    def oldest(self) -> int:
        return min(self.versions) if self.versions else self.submit_version

    def spans(self) -> int:
        return (max(self.versions) - min(self.versions) + 1) if self.versions else 1

    def staleness(self, v_train: int) -> int:
        """v_train − oldest version (GLM: only the oldest version matters; the earliest tokens deviate most)."""
        return v_train - self.oldest()


@dataclass
class Group:
    gid: str
    source: str
    n: int
    submit_index: int
    submit_version: int
    members: list[str] = field(default_factory=list)
    status: str = "open"  # open | terminal | trained | evicted
    backfill: str = "pad"
    refills: int = 0
    meta: dict[str, Any] = field(default_factory=dict)


class DataPool:
    def __init__(self) -> None:
        self.tags: dict[str, Tag] = {}
        self.groups: dict[str, Group] = {}
        self.values: dict[str, Any] = {}
        self._next_index = 0
        self._rid = 0
        # indexes over tag status (m13 audit 05: admission scanned every tag ever created, O(N²) over a run)
        self._untrained: dict[str, set[str]] = {}  # source -> rids in flight or finished-untrained
        self._live: dict[str, set[str]] = {}  # source -> rids pending / running / grading
        # M12-infra C (control plane at 10⁴–10⁵ live rollouts): groups with a live member (count), masked rids, and each
        # group's oldest non-evicted version, cached and invalidated by the writes that can change it
        self._live_groups: dict[str, int] = {}
        self._masked: set[str] = set()
        self._g_oldest: dict[str, int] = {}
        self._live_heap: dict[str, list[tuple[int, int, str]]] = {}  # source -> (lower bound of oldest, submit index, gid), lazy
        self._terminal: set[str] = set()
        self._pending: dict[str, None] = {}  # insertion-ordered: rollouts waiting to launch, in creation order
        self.released_groups = 0

    def _set_status(self, rid: str, status: str) -> None:
        """The one place a tag's status changes: keeps the untrained / live indexes exact."""
        t = self.tags[rid]
        was_live = rid in self._live.get(t.source, ())
        if (t.status == "evicted") != (status == "evicted"):
            self._g_oldest.pop(t.group, None)
        t.status = status
        if was_live != (status in LIVE):
            n = self._live_groups.get(t.group, 0) + (1 if status in LIVE else -1)
            if n > 0:
                if n == 1:  # the group became live: its submission version bounds every member's oldest from below
                    g = self.groups[t.group]
                    heapq.heappush(self._live_heap.setdefault(g.source, []), (g.submit_version, g.submit_index, g.gid))
                self._live_groups[t.group] = n
            else:
                self._live_groups.pop(t.group, None)
        (self._masked.add if status == "masked" else self._masked.discard)(rid)
        if status == "pending":
            self._pending[rid] = None
        else:
            self._pending.pop(rid, None)
        un, lv = self._untrained.setdefault(t.source, set()), self._live.setdefault(t.source, set())
        (un.add if (status in LIVE or status in TERMINAL) else un.discard)(rid)
        (lv.add if status in LIVE else lv.discard)(rid)

    def reindex(self) -> None:
        """Rebuild the indexes from the tags (after a state-plane restore rewrote them)."""
        self._untrained, self._live, self._live_groups, self._masked, self._g_oldest, self._live_heap = {}, {}, {}, set(), {}, {}
        self._pending = {}
        for rid, t in self.tags.items():
            st, t.status = t.status, ""  # not live, not evicted: _set_status counts it afresh
            self._set_status(rid, st)
        self._terminal = {gid for gid, g in self.groups.items() if g.status == "terminal"}

    def release(self, gid: str) -> None:
        """Forget a trained or evicted group: its tags and values (the trainer already consumed the values), so a long run's
        pool holds only open work."""
        g = self.groups.get(gid)
        if g is None or g.status not in ("trained", "evicted"):
            return
        if any(self.tags[r].status in LIVE for r in g.members if r in self.tags):
            return  # a member still runs (an evicted group's straggler): released later, when it finishes
        for r in g.members:
            t = self.tags.pop(r, None)
            if t is not None:
                self._untrained.get(t.source, set()).discard(r)
                self._live.get(t.source, set()).discard(r)
                self._masked.discard(r)
                self._pending.pop(r, None)
            self.values.pop(r, None)
        del self.groups[gid]
        self._terminal.discard(gid)
        self._g_oldest.pop(gid, None)
        self._live_groups.pop(gid, None)
        self.released_groups += 1

    # ---- writes -------------------------------------------------------------------------------------------------
    def submit_group(self, gid: str, source: str, n: int, version: int, *, harness: str = "", backfill: str = "pad", meta: dict | None = None) -> Group:
        g = Group(gid=gid, source=source, n=n, submit_index=self._next_index, submit_version=version, backfill=backfill, meta=dict(meta or {}))
        self._next_index += 1
        self.groups[gid] = g
        for _ in range(n):
            self._add_member(g, version, harness)
        return g

    def _add_member(self, g: Group, version: int, harness: str) -> Tag:
        self._rid += 1
        t = Tag(rid=f"{g.gid}/{self._rid}", group=g.gid, source=g.source, submit_index=g.submit_index, submit_version=version, harness=harness)
        self.tags[t.rid] = t
        t.status = ""
        self._set_status(t.rid, "pending")
        g.members.append(t.rid)
        self._g_oldest.pop(g.gid, None)
        return t

    def refill_member(self, gid: str, version: int) -> Tag:
        g = self.groups[gid]
        g.refills += 1
        g.status = "open"
        self._terminal.discard(gid)
        return self._add_member(g, version, self.tags[g.members[0]].harness if g.members else "")

    def start(self, rid: str) -> None:
        self._set_status(rid, "running")

    def grading(self, rid: str) -> None:
        """Generation finished; the reward is being computed asynchronously. The rollout still counts as untrained
        (admission, staleness wait) and its group is not terminal until the reward arrives."""
        self._set_status(rid, "grading")

    def call(self, rid: str, version: int, tokens: int = 0) -> None:
        t = self.tags[rid]
        if not t.versions or t.versions[-1] != version:
            t.versions.append(version)
            self._g_oldest.pop(t.group, None)
        t.length += tokens

    def finish(self, rid: str, *, reward: float | None, status: str = "finished", value: Any = None, now: float = 0.0) -> None:
        if status not in TERMINAL:
            raise ValueError(status)
        t = self.tags[rid]
        self._set_status(rid, status)
        t.reward, t.done_at = (None if status == "masked" else reward), now
        if value is not None:
            self.values[rid] = value
        self._update_group(t.group)

    def _update_group(self, gid: str) -> None:
        g = self.groups[gid]
        if g.status in ("trained", "evicted"):
            return
        sts = [self.tags[r].status for r in g.members]
        active = [s for s in sts if s != "evicted"]
        if active and all(s in TERMINAL for s in active):
            g.status = "terminal"
            self._terminal.add(gid)

    def evict_group(self, gid: str) -> list[str]:
        g = self.groups[gid]
        g.status = "evicted"
        self._terminal.discard(gid)
        out = []
        for r in g.members:
            t = self.tags[r]
            if t.status != "trained":
                self._set_status(r, "evicted")
                out.append(r)
        return out

    def mark_trained(self, gid: str) -> None:
        g = self.groups[gid]
        g.status = "trained"
        self._terminal.discard(gid)
        for r in g.members:
            if self.tags[r].status in TERMINAL:
                self._set_status(r, "trained")

    # ---- reads (tags only) --------------------------------------------------------------------------------------
    def _rids(self, index: dict[str, set[str]], source: str | None) -> list[str]:
        if source is not None:
            return sorted(index.get(source, ()))
        return sorted(r for rs in index.values() for r in rs)

    def live(self, source: str | None = None) -> list[Tag]:
        return [self.tags[r] for r in self._rids(self._live, source)]

    def untrained(self, source: str | None = None) -> list[Tag]:
        """In flight or finished-but-not-yet-trained: what the admission cap must count for the staleness bound to hold
        (the buffer ages too)."""
        return [self.tags[r] for r in self._rids(self._untrained, source)]

    def live_count(self) -> int:
        return sum(len(rs) for rs in self._live.values())

    def live_groups(self) -> list[str]:
        """Groups with at least one pending / running / grading member (incrementally indexed)."""
        return list(self._live_groups)

    def min_live_oldest(self, source: str) -> int | None:
        """The oldest version over the source's groups with a live member, without scanning them: a lazy min-heap of
        lower bounds (a group's oldest only rises: first calls, evictions, refills at the current version), fixed up
        at the top until the top entry is exact."""
        h = self._live_heap.get(source)
        while h:
            bound, idx, gid = h[0]
            if gid not in self._live_groups:
                heapq.heappop(h)
                continue
            true = self.group_oldest(gid)  # a live member is never evicted: not None
            if true > bound:
                heapq.heapreplace(h, (true, idx, gid))
                continue
            return bound
        return None

    def pending(self) -> list[str]:
        """Rollouts waiting to launch, in creation order (indexed: no scan over every tag)."""
        return list(self._pending)

    def live_sources(self) -> list[str]:
        return [s for s, rs in self._live.items() if rs]

    def has_masked(self) -> bool:
        return bool(self._masked)

    def untrained_count(self, source: str | None = None) -> int:
        if source is not None:
            return len(self._untrained.get(source, ()))
        return sum(len(rs) for rs in self._untrained.values())

    def terminal_groups(self) -> list[Group]:
        return sorted((self.groups[gid] for gid in self._terminal), key=lambda g: g.submit_index)  # submission order, as before the index

    def members(self, gid: str, statuses: Iterable[str] | None = None) -> list[Tag]:
        st = set(statuses) if statuses else None
        return [self.tags[r] for r in self.groups[gid].members if st is None or self.tags[r].status in st]

    def group_oldest(self, gid: str) -> int | None:
        """The oldest policy version among the group's non-evicted members (cached; None: every member evicted)."""
        if gid not in self._g_oldest:
            vs = [self.tags[r].oldest() for r in self.groups[gid].members if self.tags[r].status != "evicted"]
            self._g_oldest[gid] = min(vs) if vs else None  # type: ignore[assignment]
        return self._g_oldest[gid]

    def group_staleness(self, gid: str, v_train: int) -> int:
        o = self.group_oldest(gid)
        return 0 if o is None else v_train - o
