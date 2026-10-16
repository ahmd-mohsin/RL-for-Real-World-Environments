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

M0 audit 2026-10-14 R-16: the head moves by compare-and-set on its revision. A put that lost a race re-reads the head
and carries its signals again (they were dropped when another put moved the head in between), and a version whose
source was fetched before the current head's (``fetched_at``, stamped by ``ingest`` before the fetch) is stored but
never becomes the head: a slow ingest that tags last can no longer move the head back, and ``purge_superseded`` then
delete the newer version.
"""

from __future__ import annotations

from collections.abc import Iterator

from crucible.core.prov import Provenance
from crucible.core.schema.trace import Trace, event_key
from crucible.core.store import Store, StoreError, TagConflict, TagInfo

KIND = "trace"
PUT_ATTEMPTS = 8  # a put that keeps losing the head to other writers gives up (TagConflict) after this many tries


def _head(trace_id: str) -> str:
    return f"trace-head:{trace_id}"


class TraceStore:
    def __init__(self, store: Store):
        self.store = store  # owner-only: Store tightens its root and index on open (M0 audit 2026-10-16 R-4)

    @staticmethod
    def _identity(e) -> str:
        return event_key(e)

    def _carry_signals(self, trace: Trace, head: str | None) -> Trace:
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

    def put(self, trace: Trace, *, redaction: str | None = None, fetched_at: str | None = None) -> str:
        """Store ``trace`` as a version of its trace id and make it the head, unless its source was read before the
        head's (``fetched_at``, ``core.prov.utc_now`` format; None: no order known, it becomes the head). Returns the
        version's key, the head or not."""
        from crucible.connectors.redact import REDACTION_VERSION

        prov = Provenance(created_by="crucible.connectors.trace_store", params={"redaction": redaction or REDACTION_VERSION})
        for _ in range(PUT_ATTEMPTS):
            info = self.head(trace.trace_id)
            version = self._carry_signals(trace, info.key if info else self._head_key(trace.trace_id))
            key = self.store.put_doc(KIND, version, name=trace.trace_id, prov=prov)
            head_fetched = (info.note or {}).get("fetched_at") if info else None
            if fetched_at and head_fetched and fetched_at < head_fetched:
                return key  # an older read of the source than the head's: kept as a superseded version
            try:
                self.store.tag(_head(trace.trace_id), key, expect=info.rev if info else 0,
                               note={"fetched_at": fetched_at} if fetched_at else None)
                return key
            except TagConflict:
                continue  # another put moved the head: read it again (its signals, its fetch time)
        raise StoreError(f"trace {trace.trace_id!r}: the head kept moving under {PUT_ATTEMPTS} attempts to store a version")

    def head(self, trace_id: str) -> TagInfo | None:
        """The head tag of a trace id (key, revision, the fetch time it was recorded with); None if never tagged."""
        return self.store.tag_info(_head(trace_id))

    def _head_key(self, trace_id: str) -> str | None:
        info = self.head(trace_id)
        if info:
            return info.key
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
