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
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

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
    dt = datetime.fromisoformat(text)  # 3.11+ accepts a trailing "Z"
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
        self._paused_at: datetime | None = None

    def now(self) -> datetime:
        if self._paused_at is not None:
            return self._paused_at
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

    def advance_to(self, t: datetime) -> None:
        """Move to absolute time ``t`` (sim mode: the simulation reports where it is); never backwards."""
        self.advance(t - self.now())

    def advance_sim_seconds(self, seconds_since_t0: float) -> None:
        self.advance_to(self._t0 + timedelta(seconds=float(seconds_since_t0)))

    def reset(self) -> None:
        self._base = self._t0
        self._offset = timedelta()
        self._real0 = time.monotonic()
        self._paused_at = None

    def pause(self) -> None:
        """Freeze a wall clock (m13 S6: the World is paused while the trainer holds the GPU); step clocks never move by themselves."""
        if self.mode == "wall" and self._paused_at is None:
            self._paused_at = self.now()

    def resume(self) -> None:
        """Continue from the frozen time: no jump for the paused interval."""
        at = self._paused_at
        if at is not None:
            self._base, self._offset, self._real0, self._paused_at = at, timedelta(), time.monotonic(), None

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
    ctx: dict | None = None  # on_tool: the provoking call {"args", "result"} (templating, where)
    bindings: dict = field(default_factory=dict)  # state: the check pipeline's bindings


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
            due, draws = self._first_due(ev)
            if due is not None:
                self.pending[ev.id] = PendingEvent(id=ev.id, due=iso(due), spec_index=i, draws=draws or None)

    def _interval(self, ev: EventSpec, k: int) -> timedelta:
        """The k-th interval of a recurring event: fixed, or the k-th draw of its seeded exponential stream."""
        from crucible.core.schema.world import EveryDist
        from crucible.core.triggers import exp_interval

        every = ev.when.every
        if isinstance(every, EveryDist):
            return timedelta(seconds=exp_interval(int(ev.when.seed or 0), parse_duration(every.mean).total_seconds(), k))
        return parse_duration(every)

    def _mark(self, text: str) -> datetime:
        return self.clock.t0 + parse_duration(text) if text.startswith("P") else parse_time(text)

    def _first_due(self, ev: EventSpec) -> tuple[datetime | None, int]:
        w = ev.when
        if w.at:
            return self._mark(w.at), 0
        if w.after:
            return self.clock.t0 + parse_duration(w.after), 0
        if w.every and not (w.on_tool or w.state or w.after_event):  # recurring with no start: one interval after t0
            return self.clock.t0 + self._interval(ev, 0), 1
        return None, 0

    def _until(self, ev: EventSpec) -> datetime | None:
        return self._mark(ev.when.until) if ev.when.until else None

    def pop_due(self) -> list[Fired]:
        from crucible.core.schema.world import EveryDist

        now = self.clock.now()
        fired: list[Fired] = []
        # Fire in due-time order (then spec order) so the log is deterministic.
        for pe in sorted(self.pending.values(), key=lambda p: (p.due, p.spec_index)):
            due = parse_time(pe.due)
            if due > now:
                continue
            ev = self.events[pe.spec_index]
            until = self._until(ev)
            if until is not None and due > until:  # retired without firing
                del self.pending[ev.id]
                continue
            fired.append(Fired(ev, pe.spec_index, due))
            self.fired_counts[ev.id] = self.fired_counts.get(ev.id, 0) + 1
            if ev.when.every and not ev.when.on_tool and (ev.when.count is None or self.fired_counts[ev.id] < ev.when.count):
                nxt = due + self._interval(ev, pe.draws or 0)
                pe.draws = (pe.draws or 0) + 1 if isinstance(ev.when.every, EveryDist) else None
                if until is not None and nxt > until:
                    del self.pending[ev.id]
                else:
                    pe.due = iso(nxt)
                    pe.fired_count = self.fired_counts[ev.id]
            else:
                del self.pending[ev.id]
        return fired

    def anchor(self, event_id: str) -> list[str]:
        """``event_id`` fired: events anchored on it (``after_event`` + ``offset``) get their mark, now + offset, once."""
        now = self.clock.now()
        out = []
        for i, ev in enumerate(self.events):
            if ev.when.after_event == event_id and ev.id not in self.pending and not self.fired_counts.get(ev.id):
                self.pending[ev.id] = PendingEvent(id=ev.id, due=iso(now + parse_duration(ev.when.offset or "0s")), spec_index=i)
                out.append(ev.id)
        return out

    def on_tool(self, qualified_tool: str, actor: str, args: dict | None = None, result: Any = None) -> list[Fired]:
        from crucible.core.triggers import where_matches

        ctx = {"args": args or {}, "result": result}
        out: list[Fired] = []
        for i, ev in enumerate(self.events):
            w = ev.when
            if not w.on_tool or not fnmatch.fnmatchcase(qualified_tool, w.on_tool):
                continue
            if w.on_actor and w.on_actor != actor:
                continue
            limit = w.count if w.count is not None else (None if (w.every or w.repeat) else 1)
            if limit is not None and self.fired_counts.get(ev.id, 0) >= limit:
                continue
            if w.where and not where_matches(w.where, ctx):
                continue
            self.fired_counts[ev.id] = self.fired_counts.get(ev.id, 0) + 1
            out.append(Fired(ev, i, self.clock.now(), ctx=ctx))
        return out

    def state_events(self) -> list[tuple[int, EventSpec]]:
        """Armed state triggers (they fire once, when their check first passes)."""
        return [(i, ev) for i, ev in enumerate(self.events) if ev.when.state and not self.fired_counts.get(ev.id)]

    def mark_fired(self, event_id: str) -> None:
        self.fired_counts[event_id] = self.fired_counts.get(event_id, 0) + 1

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
