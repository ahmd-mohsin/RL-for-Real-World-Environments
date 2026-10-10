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
LIVE = {"pending", "running"}


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
        g.members.append(t.rid)
        return t

    def refill_member(self, gid: str, version: int) -> Tag:
        g = self.groups[gid]
        g.refills += 1
        g.status = "open"
        return self._add_member(g, version, self.tags[g.members[0]].harness if g.members else "")

    def start(self, rid: str) -> None:
        self.tags[rid].status = "running"

    def call(self, rid: str, version: int, tokens: int = 0) -> None:
        t = self.tags[rid]
        if not t.versions or t.versions[-1] != version:
            t.versions.append(version)
        t.length += tokens

    def finish(self, rid: str, *, reward: float | None, status: str = "finished", value: Any = None, now: float = 0.0) -> None:
        if status not in TERMINAL:
            raise ValueError(status)
        t = self.tags[rid]
        t.status, t.reward, t.done_at = status, (None if status == "masked" else reward), now
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
                t.status = "evicted"
                out.append(r)
        return out

    def mark_trained(self, gid: str) -> None:
        g = self.groups[gid]
        g.status = "trained"
        for r in g.members:
            if self.tags[r].status in TERMINAL:
                self.tags[r].status = "trained"

    # ---- reads (tags only) --------------------------------------------------------------------------------------
    def live(self, source: str | None = None) -> list[Tag]:
        return [t for t in self.tags.values() if t.status in LIVE and (source is None or t.source == source)]

    def untrained(self, source: str | None = None) -> list[Tag]:
        """In flight or finished-but-not-yet-trained: what the admission cap must count for the staleness bound to hold
        (the buffer ages too)."""
        return [t for t in self.tags.values() if (t.status in LIVE or t.status in TERMINAL) and (source is None or t.source == source)]

    def terminal_groups(self) -> list[Group]:
        return [g for g in self.groups.values() if g.status == "terminal"]

    def members(self, gid: str, statuses: Iterable[str] | None = None) -> list[Tag]:
        st = set(statuses) if statuses else None
        return [self.tags[r] for r in self.groups[gid].members if st is None or self.tags[r].status in st]

    def group_staleness(self, gid: str, v_train: int) -> int:
        ts = [t for t in self.members(gid) if t.status != "evicted"]
        return max((t.staleness(v_train) for t in ts), default=0)
