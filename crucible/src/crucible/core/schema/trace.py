"""Workflow traces ``crucible.trace/v1`` (DESIGN §4.2) and grounding records.

A trace is the envelope around one recorded execution of a workflow: non-agent events (commits,
reviews, CI runs, emails, DB updates), embedded agent sessions (ATIF), the outcome label and the
derived expert signals. ``Record`` is the unit simulated actors may cite (ARCHITECTURE M2): a
text span someone actually wrote, with its author and time.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.signal import Signal, SignalTarget

EventKind = Literal["state", "action", "observation", "message", "agent_session", "check"]
Privacy = Literal["public", "internal", "restricted"]


class TraceSource(CrucibleModel):
    connector: str
    ref: str
    url: str | None = None
    privacy: Privacy = "internal"
    fetched_at: str | None = None


class TraceActor(CrucibleModel):
    id: str
    role: str  # author, reviewer, approver, customer, ci, ...
    kind: Literal["human", "agent", "system"]
    handle: str | None = None  # pseudonymized identifier


class TraceEvent(CrucibleModel):
    seq: int
    time: str | None = None  # RFC 3339
    kind: EventKind
    actor: str | None = None
    type: str  # connector-defined subtype: commit, review_comment, ci_run, email, sql_update ...
    text: str | None = None
    target: SignalTarget | None = None
    state_ref: str | None = None  # e.g. "git:<sha>", "store:<key>"
    diff_ref: str | None = None
    atif_ref: str | None = None
    reply_to: int | None = None  # seq of the event this one answers
    data: dict[str, Any] = Field(default_factory=dict)


def event_key(e: TraceEvent) -> str:
    """A stable name for an event across re-ingests: the source's own row id when the connector kept one (a GitHub
    comment id, a ticket-log row), else its time, type, actor and text. Seqs are positions: one earlier row shifts
    them all, so signal ids, hand labels and carried signals name events by this key (M4 audit)."""
    import hashlib
    import json

    rid = e.data.get("row_id") if isinstance(e.data, dict) else None
    if rid is not None:
        return f"row:{rid}"
    return "ev:" + hashlib.sha256(json.dumps([e.time, e.type, e.actor, e.text], ensure_ascii=False).encode()).hexdigest()[:24]


class TraceOutcome(CrucibleModel):
    label: Literal["good", "bad", "unknown"] = "unknown"
    label_source: str | None = None  # e.g. "merged + 30d no revert"
    details: dict[str, Any] = Field(default_factory=dict)


class Trace(VersionedDocument):
    SCHEMA = "crucible.trace/v1"
    trace_id: str
    family: str | None = None
    source: TraceSource
    actors: list[TraceActor] = Field(default_factory=list)
    events: list[TraceEvent] = Field(default_factory=list)
    outcome: TraceOutcome = Field(default_factory=TraceOutcome)
    signals: list[Signal] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check(self) -> Trace:
        seqs = [e.seq for e in self.events]
        if seqs != sorted(seqs) or len(set(seqs)) != len(seqs):
            raise ValueError("event seqs must be strictly increasing")
        known = set(seqs)
        actor_ids = {a.id for a in self.actors}
        for e in self.events:
            if e.actor is not None and actor_ids and e.actor not in actor_ids:
                raise ValueError(f"event {e.seq} names unknown actor {e.actor!r}")
            if e.reply_to is not None and e.reply_to not in known:
                raise ValueError(f"event {e.seq} replies to unknown seq {e.reply_to}")
        for s in self.signals:
            bad = [g for g in s.grades if g not in known]
            if bad:
                raise ValueError(f"signal {s.id} grades unknown events {bad}")
        return self


class Record(CrucibleModel):
    """A citable span of recorded text. Simulated actors may only say what records support."""

    id: str  # stable: "<trace_id>#<seq>" or "<source>:<key>"
    source: str  # trace id or document id
    author: str | None = None
    role: str | None = None
    time: str | None = None  # RFC 3339; records later than the world clock are invisible
    text: str
    target: SignalTarget | None = None
    reply_to: str | None = None
    tags: list[str] = Field(default_factory=list)
    meta: dict[str, Any] = Field(default_factory=dict)


def records_from_trace(trace: Trace, types: set[str] | None = None) -> list[Record]:
    """Every event with text (optionally only some subtypes) becomes a citable Record."""
    roles = {a.id: a.role for a in trace.actors}
    out: list[Record] = []
    for e in trace.events:
        if not e.text or (types is not None and e.type not in types):
            continue
        out.append(
            Record(
                id=f"{trace.trace_id}#{e.seq}",
                source=trace.trace_id,
                author=e.actor,
                role=roles.get(e.actor or ""),
                time=e.time,
                text=e.text,
                target=e.target,
                reply_to=f"{trace.trace_id}#{e.reply_to}" if e.reply_to is not None else None,
                tags=[e.type],
                meta={k: v for k, v in e.data.items() if isinstance(v, (str, int, float, bool))},
            )
        )
    return out
