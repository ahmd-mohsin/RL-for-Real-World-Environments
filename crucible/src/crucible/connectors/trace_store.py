"""Traces as content-addressed documents in the Store (tenant-local): put is idempotent, the latest version of a
trace id wins, and signals/outcomes added later are new versions of the same trace id (provenance kept)."""

from __future__ import annotations

from collections.abc import Iterator

from crucible.core.schema.trace import Trace
from crucible.core.store import Store

KIND = "trace"


class TraceStore:
    def __init__(self, store: Store):
        self.store = store

    def put(self, trace: Trace) -> str:
        return self.store.put_doc(KIND, trace, name=trace.trace_id)

    def get(self, trace_id: str) -> Trace:
        rows = self.store.list_docs(KIND, name=trace_id)
        if not rows:
            raise KeyError(f"no trace {trace_id!r}")
        return self.store.get_doc(rows[-1].key, Trace)

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
