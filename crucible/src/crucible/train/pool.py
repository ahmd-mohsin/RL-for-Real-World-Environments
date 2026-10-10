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
        self.released_groups = 0

    def _set_status(self, rid: str, status: str) -> None:
        """The one place a tag's status changes: keeps the untrained / live indexes exact."""
        t = self.tags[rid]
        t.status = status
        un, lv = self._untrained.setdefault(t.source, set()), self._live.setdefault(t.source, set())
        (un.add if (status in LIVE or status in TERMINAL) else un.discard)(rid)
        (lv.add if status in LIVE else lv.discard)(rid)

    def reindex(self) -> None:
        """Rebuild the indexes from the tags (after a state-plane restore rewrote them)."""
        self._untrained, self._live = {}, {}
        for rid, t in self.tags.items():
            self._set_status(rid, t.status)

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
            self.values.pop(r, None)
        del self.groups[gid]
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
        self._set_status(t.rid, t.status)
        g.members.append(t.rid)
        return t

    def refill_member(self, gid: str, version: int) -> Tag:
        g = self.groups[gid]
        g.refills += 1
        g.status = "open"
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

    def evict_group(self, gid: str) -> list[str]:
        g = self.groups[gid]
        g.status = "evicted"
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

    def untrained_count(self, source: str | None = None) -> int:
        if source is not None:
            return len(self._untrained.get(source, ()))
        return sum(len(rs) for rs in self._untrained.values())

    def terminal_groups(self) -> list[Group]:
        return [g for g in self.groups.values() if g.status == "terminal"]

    def members(self, gid: str, statuses: Iterable[str] | None = None) -> list[Tag]:
        st = set(statuses) if statuses else None
        return [self.tags[r] for r in self.groups[gid].members if st is None or self.tags[r].status in st]

    def group_staleness(self, gid: str, v_train: int) -> int:
        ts = [t for t in self.members(gid) if t.status != "evicted"]
        return max((t.staleness(v_train) for t in ts), default=0)
