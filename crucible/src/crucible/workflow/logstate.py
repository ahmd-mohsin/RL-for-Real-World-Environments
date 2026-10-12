"""State of a business-object workflow, reconstructed from an event log (the horizontal-workflow path: tickets, ERP
audit rows, CRM history exported through ``file_drop`` / ``sql_audit_log``).

Each event that *writes* objects (``data.writes``, e.g. ``ticket:T1001.status``) sets those objects' values: the first
declared value field present on the event (``status``, ``amount``, ``team``, ...) or, failing that, the event type.
``LogStateView`` answers milestone queries over that object store, with the same interface as a World's
``VerifierView``:

  ``get``    {object}            -> the value (``None`` if never written)
  ``exists`` {object}            -> {"exists": bool}
  ``match``  {object, pattern}   -> {"exists": bool, "matches": bool} (regex over the value)
  ``count``  {prefix}            -> number of written objects under a prefix

Objects in milestone specs may be templated with ``{case}`` (the trace's case id), so one family milestone set covers
every ticket. ``states(trace)`` gives the state after every event (only writing events change it).
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.state import Query, QueryResult
from crucible.core.schema.trace import Trace

VALUE_FIELDS = ("status", "amount", "team", "score", "plan", "value", "state", "decision")


@dataclass
class LogStateView:
    objects: dict[str, Any] = field(default_factory=dict)
    case: str = ""

    def _o(self, obj: str) -> str:
        return obj.replace("{case}", self.case)

    async def query(self, app: str, q: Query | dict[str, Any]) -> QueryResult:
        q = q if isinstance(q, Query) else Query.model_validate(q)
        a = q.args
        try:
            if q.op == "get":
                return QueryResult(ok=True, value=self.objects.get(self._o(a["object"])))
            if q.op == "exists":
                return QueryResult(ok=True, value={"exists": self._o(a["object"]) in self.objects})
            if q.op == "match":
                o = self._o(a["object"])
                v = self.objects.get(o)
                return QueryResult(ok=True, value={"exists": o in self.objects, "matches": v is not None and re.search(a["pattern"], str(v)) is not None})
            if q.op == "count":
                pre = self._o(a["prefix"])
                return QueryResult(ok=True, value={"count": sum(1 for k in self.objects if k.startswith(pre))})
        except (KeyError, re.error) as e:
            return QueryResult(ok=False, error=f"{q.op}: {e}")
        return QueryResult(ok=False, error=f"op {q.op!r} not supported on an event-log state")


def value_of(data: dict[str, Any], typ: str, fields: Sequence[str] = VALUE_FIELDS) -> Any:
    return next((data[f] for f in fields if data.get(f) not in (None, "")), typ)


def states(trace: Trace, *, case: str | None = None, fields: Sequence[str] = VALUE_FIELDS,
           effects: Callable[[Any], dict[str, Any]] | None = None) -> list[tuple[int, LogStateView]]:
    """(seq, state after the event) for every event of the trace. ``effects(event)`` (pack-supplied) may return extra
    object writes an event implies (e.g. an approval recorded as ``<object>.approved``) or override values
    (a reversal setting the refund to ``"reversed"``)."""
    case = case or trace.source.ref
    cur: dict[str, Any] = {}
    out = []
    for e in trace.events:
        args = e.data.get("arguments") if isinstance(e.data.get("arguments"), dict) else None
        for w in e.data.get("writes") or []:
            obj_field = w.split(":", 1)[-1]
            if args is not None and "." in obj_field and obj_field.rsplit(".", 1)[-1] in args:
                cur[w] = args[obj_field.rsplit(".", 1)[-1]]  # a World App call: the value it wrote (M5 audit: the event type was stored)
            else:
                cur[w] = value_of(e.data, e.type, fields)
        if effects is not None:
            cur.update(effects(e))
        out.append((e.seq, LogStateView(objects=dict(cur), case=case)))
    return out
