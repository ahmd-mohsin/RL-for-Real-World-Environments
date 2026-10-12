"""Traces as content-addressed documents in the Store (tenant-local): put is idempotent, the latest version of a
trace id wins, and signals/outcomes added later are new versions of the same trace id (provenance kept).

M3 audit (2026-10-12):
- The store directory is owner-only (0700; blobs were 0444 and the index 0644, readable by every local user).
- Every version records the redaction rules it passed (``redaction`` provenance); ``stale()`` lists traces whose head
  was redacted under older rules, and ``purge_superseded()`` removes old versions (a version stored before a
  redaction fix stayed addressable).
- The head of a trace is an explicit tag, so re-ingesting content identical to an earlier version makes it the head
  again (the first insert time decided "latest").
- Re-ingesting a grown source no longer drops the signals M4 wrote on the previous head: they are carried to the new
  version through stable event identities (the source row id, else time, type, actor and text); a signal whose events
  are gone is counted in ``outcome.details.signals_dropped``.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

from crucible.core.prov import Provenance
from crucible.core.schema.trace import Trace
from crucible.core.store import Store

KIND = "trace"


def _head(trace_id: str) -> str:
    return f"trace-head:{trace_id}"


class TraceStore:
    def __init__(self, store: Store):
        self.store = store
        os.chmod(store.root, 0o700)
        for f in store.root.glob("index.sqlite*"):
            os.chmod(f, 0o600)

    @staticmethod
    def _identity(e) -> tuple:
        rid = e.data.get("row_id") if isinstance(e.data, dict) else None
        return ("row", rid) if rid is not None else ("event", e.time, e.type, e.actor, e.text)

    def _carry_signals(self, trace: Trace) -> Trace:
        head = self._head_key(trace.trace_id)
        if head is None or trace.signals:
            return trace
        old = self.store.get_doc(head, Trace)
        if not old.signals:
            return trace
        new_seq = {self._identity(e): e.seq for e in trace.events}
        old_key = {e.seq: self._identity(e) for e in old.events}

        def remap(seq):
            return None if seq is None else new_seq.get(old_key.get(seq, ("missing",)))

        kept, dropped = [], 0
        for sig in old.signals:
            grades = [remap(g) for g in sig.grades]
            src = remap(sig.source_event) if sig.source_event is not None else None
            if any(g is None for g in grades) or (sig.source_event is not None and src is None):
                dropped += 1
                continue
            kept.append(sig.model_copy(update={"grades": grades, "source_event": src}))
        details = {**trace.outcome.details, **({"signals_dropped": dropped} if dropped else {})}
        return trace.model_copy(update={"signals": kept, "outcome": trace.outcome.model_copy(update={"details": details})})

    def put(self, trace: Trace, *, redaction: str | None = None) -> str:
        from crucible.connectors.redact import REDACTION_VERSION

        trace = self._carry_signals(trace)
        prov = Provenance(created_by="crucible.connectors.trace_store", params={"redaction": redaction or REDACTION_VERSION})
        key = self.store.put_doc(KIND, trace, name=trace.trace_id, prov=prov)
        self.store.tag(_head(trace.trace_id), key)
        return key

    def _head_key(self, trace_id: str) -> str | None:
        row = self.store._db.execute("SELECT key FROM tags WHERE name=?", (_head(trace_id),)).fetchone()
        if row:
            return row[0]
        rows = self.store.list_docs(KIND, name=trace_id)
        return rows[-1].key if rows else None

    def get(self, trace_id: str) -> Trace:
        key = self._head_key(trace_id)
        if key is None:
            raise KeyError(f"no trace {trace_id!r}")
        return self.store.get_doc(key, Trace)

    def redaction_of(self, key: str) -> str | None:
        row = self.store._db.execute("SELECT provenance FROM doc_names WHERE kind=? AND key=? ORDER BY created_at DESC LIMIT 1", (KIND, key)).fetchone()
        if not row or not row[0]:
            return None
        return Provenance.model_validate_json(row[0]).params.get("redaction")  # type: ignore[return-value]

    def stale(self) -> list[str]:
        """Trace ids whose head was redacted under other rules than today's (re-ingest them, then ``purge_superseded``)."""
        from crucible.connectors.redact import REDACTION_VERSION

        return [t for t in self.ids() if self.redaction_of(self._head_key(t) or "") != REDACTION_VERSION]

    def purge_superseded(self) -> list[str]:
        """Forget every version that is not a trace's head; returns the keys removed."""
        gone = []
        for t in self.ids():
            head = self._head_key(t)
            for row in self.store.list_docs(KIND, name=t):
                if row.key != head:
                    self.store.forget_doc(KIND, row.key)
                    gone.append(row.key)
        return gone

    def ids(self, family: str | None = None) -> list[str]:
        seen: dict[str, None] = {}
        for row in self.store.list_docs(KIND):
            if row.name and row.name not in seen:
                seen[row.name] = None
        out = list(seen)
        if family is not None:
            out = [t for t in out if self.get(t).family == family]
        return out

    def __iter__(self) -> Iterator[Trace]:
        for t in self.ids():
            yield self.get(t)
