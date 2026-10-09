"""Virtual clock and event scheduling.

Semantics follow the agentenv gateway (``agent_env/env/gateway/clock.py``, ``triggers.py``):
wall mode reads ``t0 + real_elapsed * rate`` (rate <= 86400); time triggers support ``at`` (absolute),
relative-to-t0 offsets, ``every`` with ``count``; actions are ``nl`` / ``tool`` / ``permission``.
Crucible adds **step mode** (time advances only per world step and by explicit advances), which
makes episodes reproducible for RL, and **on_tool** triggers (fire after a matching tool call).
Durations accept "2d", "1h30m", "45s" and ISO-8601 "P1DT2H", "PT30M".
"""

from __future__ import annotations

import fnmatch
import re
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from crucible.core.schema.state import ClockState, PendingEvent
from crucible.core.schema.world import ClockSpec, EventSpec

_SIMPLE = re.compile(r"(\d+(?:\.\d+)?)\s*(w|d|h|m|s|ms)")
_ISO = re.compile(r"^P(?:(\d+(?:\.\d+)?)W)?(?:(\d+(?:\.\d+)?)D)?(?:T(?:(\d+(?:\.\d+)?)H)?(?:(\d+(?:\.\d+)?)M)?(?:(\d+(?:\.\d+)?)S)?)?$")
_UNIT = {"w": 604800, "d": 86400, "h": 3600, "m": 60, "s": 1, "ms": 0.001}


def parse_duration(text: str) -> timedelta:
    t = text.strip()
    m = _ISO.match(t)
    if m and t != "P" and t != "PT":
        w, d, h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return timedelta(weeks=w, days=d, hours=h, minutes=mi, seconds=s)
    parts = _SIMPLE.findall(t.replace(" ", ""))
    if not parts or "".join(f"{n}{u}" for n, u in parts) != t.replace(" ", ""):
        raise ValueError(f"bad duration {text!r} (use e.g. '2d', '1h30m', 'PT45M')")
    return timedelta(seconds=sum(float(n) * _UNIT[u] for n, u in parts))


def parse_time(text: str) -> datetime:
    dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError(f"time {text!r} must carry a timezone (RFC 3339)")
    return dt.astimezone(UTC)


def iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S") + (f".{dt.microsecond:06d}" if dt.microsecond else "") + "Z"


class VirtualClock:
    def __init__(self, spec: ClockSpec | None):
        self.spec = spec
        self.mode = spec.mode if spec else "off"
        self._t0 = parse_time(spec.t0) if spec else datetime(1970, 1, 1, tzinfo=UTC)
        self._base = self._t0
        self._offset = timedelta()
        self._real0 = time.monotonic()

    def now(self) -> datetime:
        if self.mode == "wall":
            assert self.spec is not None
            return self._base + timedelta(seconds=(time.monotonic() - self._real0) * self.spec.rate) + self._offset
        return self._base + self._offset

    def now_iso(self) -> str:
        return iso(self.now())

    @property
    def t0(self) -> datetime:
        return self._t0

    def on_step(self) -> None:
        if self.mode == "step" and self.spec is not None:
            self._offset += timedelta(seconds=self.spec.step_dt_s)

    def advance(self, delta: timedelta) -> None:
        if delta < timedelta():
            raise ValueError("time cannot go backwards")
        self._offset += delta

    def reset(self) -> None:
        self._base = self._t0
        self._offset = timedelta()
        self._real0 = time.monotonic()

    def state(self) -> ClockState:
        if self.spec is None:
            return ClockState(mode="off")
        return ClockState(mode=self.mode, t0=iso(self._t0), now=self.now_iso(), rate=self.spec.rate)

    def restore(self, st: ClockState) -> None:
        if st.mode == "off" or st.now is None:
            return
        self._base = parse_time(st.now)
        self._offset = timedelta()
        self._real0 = time.monotonic()


@dataclass
class Fired:
    event: EventSpec
    index: int
    at: datetime


class EventScheduler:
    def __init__(self, events: list[EventSpec], clock: VirtualClock):
        self.events = events
        self.clock = clock
        self.pending: dict[str, PendingEvent] = {}
        self.fired_counts: dict[str, int] = {}
        self.reset()

    def reset(self) -> None:
        self.pending = {}
        self.fired_counts = {e.id: 0 for e in self.events}
        for i, ev in enumerate(self.events):
            due = self._first_due(ev)
            if due is not None:
                self.pending[ev.id] = PendingEvent(id=ev.id, due=iso(due), spec_index=i)

    def _first_due(self, ev: EventSpec) -> datetime | None:
        w = ev.when
        if w.at:
            if w.at.startswith("P"):
                return self.clock.t0 + parse_duration(w.at)
            return parse_time(w.at)
        if w.after:
            return self.clock.t0 + parse_duration(w.after)
        return None

    def pop_due(self) -> list[Fired]:
        now = self.clock.now()
        fired: list[Fired] = []
        # Fire in due-time order (then spec order) so the log is deterministic.
        for pe in sorted(self.pending.values(), key=lambda p: (p.due, p.spec_index)):
            due = parse_time(pe.due)
            if due > now:
                continue
            ev = self.events[pe.spec_index]
            fired.append(Fired(ev, pe.spec_index, due))
            self.fired_counts[ev.id] = self.fired_counts.get(ev.id, 0) + 1
            if ev.when.every and (ev.when.count is None or self.fired_counts[ev.id] < ev.when.count):
                pe.due = iso(due + parse_duration(ev.when.every))
                pe.fired_count = self.fired_counts[ev.id]
            else:
                del self.pending[ev.id]
        return fired

    def on_tool(self, qualified_tool: str, actor: str) -> list[Fired]:
        out: list[Fired] = []
        for i, ev in enumerate(self.events):
            w = ev.when
            if not w.on_tool or not fnmatch.fnmatchcase(qualified_tool, w.on_tool):
                continue
            if w.on_actor and w.on_actor != actor:
                continue
            limit = w.count if w.count is not None else (None if w.every else 1)
            if limit is not None and self.fired_counts.get(ev.id, 0) >= limit:
                continue
            self.fired_counts[ev.id] = self.fired_counts.get(ev.id, 0) + 1
            out.append(Fired(ev, i, self.clock.now()))
        return out

    def state(self) -> list[PendingEvent]:
        return [p.model_copy() for p in sorted(self.pending.values(), key=lambda p: p.spec_index)]

    def restore(self, pending: list[PendingEvent], fired_counts: dict[str, int]) -> None:
        self.pending = {p.id: p.model_copy() for p in pending}
        self.fired_counts = dict(fired_counts)


class AccessControl:
    """Per-actor tool permissions: base patterns from the spec, then ordered grants/revokes (last match wins)."""

    def __init__(self, base: dict[str, list[str]]):
        self.base = {a: list(p) for a, p in base.items()}
        self.rules: dict[str, list[tuple[str, str]]] = {}

    def allows(self, actor: str, qualified: str) -> bool:
        ok = any(fnmatch.fnmatchcase(qualified, p) for p in self.base.get(actor, []))
        for op, pat in self.rules.get(actor, []):
            if fnmatch.fnmatchcase(qualified, pat):
                ok = op == "grant"
        return ok

    def grant(self, actor: str, patterns: list[str]) -> None:
        self.rules.setdefault(actor, []).extend(("grant", p) for p in patterns)

    def revoke(self, actor: str, patterns: list[str]) -> None:
        self.rules.setdefault(actor, []).extend(("revoke", p) for p in patterns)

    def state(self) -> dict[str, list[list[str]]]:
        return {a: [[op, p] for op, p in rules] for a, rules in self.rules.items()}

    def restore(self, st: dict[str, list[list[str]]]) -> None:
        self.rules = {a: [(op, p) for op, p in rules] for a, rules in st.items()}

    def reset(self) -> None:
        self.rules = {}
