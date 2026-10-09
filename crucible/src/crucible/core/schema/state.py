"""State references, diffs and verifier queries (domain-agnostic).

A ``StateRef`` names one App's state at one moment: its adapter kind, its canonical fingerprint
and (when the state was captured) the content-addressed blob in the store that can restore it.
Two refs with equal fingerprints are the same state, whatever backend produced them.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

# "browser" and "http_mock" are reserved for packs that need them (no built-in adapter yet).
AdapterKind = Literal["fs", "fs_git", "sql", "postgres", "json_doc", "process", "http_mock", "browser"]


class StateRef(CrucibleModel):
    app: str
    adapter: AdapterKind
    fingerprint: str
    blob: str | None = None  # store key of the captured payload (tarball, db copy, json); None = fingerprint only
    size_bytes: int | None = None
    meta: dict[str, Any] = Field(default_factory=dict)  # e.g. {"files": 3412, "head": "abc123", "tables": 7}


class FileChange(CrucibleModel):
    path: str
    change: Literal["added", "removed", "modified", "mode"]
    before: str | None = None  # content hash
    after: str | None = None


class RowChange(CrucibleModel):
    table: str
    change: Literal["inserted", "deleted"]
    row: dict[str, Any]


class StateDiff(CrucibleModel):
    app: str
    adapter: AdapterKind
    before: str
    after: str
    files: list[FileChange] = Field(default_factory=list)
    rows: list[RowChange] = Field(default_factory=list)
    tables_changed: list[str] = Field(default_factory=list)
    detail: dict[str, Any] = Field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.before == self.after


class Query(CrucibleModel):
    """A verifier-side read of state. ``op`` is adapter-specific; adapters document their ops."""

    op: str
    args: dict[str, Any] = Field(default_factory=dict)
    timeout_s: float | None = None


class QueryResult(CrucibleModel):
    ok: bool
    value: Any = None
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    duration_s: float | None = None
    error: str | None = None


class ClockState(CrucibleModel):
    mode: Literal["off", "step", "wall"] = "off"
    t0: str | None = None  # RFC 3339
    now: str | None = None  # virtual time at capture
    rate: float = 1.0
    frozen: bool = False


class PendingEvent(CrucibleModel):
    id: str
    due: str  # RFC 3339 virtual time
    spec_index: int
    fired_count: int = 0


class WorldRef(CrucibleModel):
    """A whole-world snapshot: every App's state, the clock, pending events, actor memory, step index."""

    world: str  # world spec content hash
    step: int
    apps: dict[str, StateRef]
    clock: ClockState = Field(default_factory=ClockState)
    pending_events: list[PendingEvent] = Field(default_factory=list)
    actors: dict[str, dict[str, Any]] = Field(default_factory=dict)  # actor id -> serialized actor state
    fired_counts: dict[str, int] = Field(default_factory=dict)
    acl_rules: dict[str, list[list[str]]] = Field(default_factory=dict)  # actor -> [[grant|revoke, pattern]]
    inboxes: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)  # undelivered messages per actor
    done: list[str] = Field(default_factory=list)  # actors that declared done
    backend: dict[str, Any] = Field(default_factory=dict)  # backend-held runtime state (e.g. agentenv gateway clock/triggers)
    fingerprint: str = ""  # hash over everything above except `step` (same state reached in different #steps is the same state)
