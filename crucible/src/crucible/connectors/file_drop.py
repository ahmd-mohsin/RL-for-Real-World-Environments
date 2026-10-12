"""``file_drop``: event logs exported from any system (JSONL or CSV) -> one Trace per case.

An enterprise system that can export "who did what to which object when" is enough: a ticketing tool, an ERP
audit log, a CRM history. Rows are grouped by a case key (ticket id, invoice id), sorted by time, and mapped to
TraceEvents through ``FieldMap``. ``writes`` (the objects an event changed, e.g. ``ticket:42.status``) is what
generic step attribution in M4 uses: a signal about an object grades the event that last wrote it.

M3 audit (2026-10-12): exports are read as real exports are written (a byte-order mark, Latin-1, ``.CSV``, U+2028 inside
JSON strings); rows without a case key are counted, not fatal; events are ordered by instant (mixed offsets sorted as
strings) with ties broken by the row's content (export order changed the hash); a row repeated by overlapping exports is
one event; times are normalized to RFC 3339 (``03/02/2026 09:00`` was stored as is and failed later); the source row id
is kept (``data.row_id``) so a grown log can be matched to the old one; ``Bot`` / ``AI`` actor kinds are agents.
"""

from __future__ import annotations

import csv
import glob
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource

KINDS = {"state", "action", "observation", "message", "agent_session", "check"}


class FieldMap(CrucibleModel):
    case: str = "case"  # rows sharing this value form one trace
    time: str = "time"
    actor: str = "actor"
    role: str = "role"
    actor_kind: str = "actor_kind"  # human | agent | system (default human)
    kind: str = "kind"  # state | action | observation | message | check (default action)
    type: str = "type"
    text: str = "text"
    object: str = "object"  # what the event is about / changed
    writes: str = "writes"  # list (or ';'-separated) of objects the event changed
    reply_to: str = "reply_to"  # id of an earlier row of the same case
    id: str = "id"
    family: str | None = None
    privacy: str = "internal"
    extra: list[str] = Field(default_factory=list)  # further columns copied into event.data


def _decode(data: bytes) -> str:
    """UTF-8 (with or without a byte-order mark), UTF-16 with a BOM, else Windows-1252 (old ERP / CRM exports)."""
    if data.startswith((b"\xff\xfe", b"\xfe\xff")):
        return data.decode("utf-16")
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


_ROWS: dict[tuple[str, int, int], list[dict[str, Any]]] = {}  # (path, size, mtime) -> rows: each file is parsed once, not once per case


def _rows(path: Path) -> list[dict[str, Any]]:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _ROWS:
        if len(_ROWS) > 64:
            _ROWS.clear()
        _ROWS[key] = _parse_rows(path)
    return _ROWS[key]


def _parse_rows(path: Path) -> list[dict[str, Any]]:
    text = _decode(path.read_bytes())
    # spreadsheet exports start with a byte-order mark, which became part of the first column's name
    if path.suffix.lower() == ".csv":
        import io

        return list(csv.DictReader(io.StringIO(text, newline="")))
    out = []
    for n, line in enumerate(text.split("\n"), 1):  # not splitlines(): U+2028 inside a JSON string is not a line break
        if line.strip():
            try:
                out.append(json.loads(line))
            except ValueError as e:
                raise ValueError(f"{path}:{n}: not a JSON object ({e})") from None
    return out


def _instant(value: Any) -> tuple:
    """Sort key: the instant when the time parses (offsets honoured), else the raw string after all parsed times.
    String order put 10:00+02:00 (08:00Z) after 09:00Z."""
    from crucible.connectors.base import parse_time

    t = parse_time(value)
    return (0, t.timestamp(), "") if t is not None else (1, 0.0, str(value or ""))


class FileDropConnector:
    name = "file_drop"

    def __init__(self, fields: FieldMap | None = None):
        self.fields = fields or FieldMap()
        self.skipped = 0  # rows without a case key in the last discover (they belong to no trace)

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        """scope: {"glob": "logs/*.jsonl"} -> one ref per case found in the matching files. Rows without a case key are
        counted in ``skipped`` (one such row raised KeyError and lost the whole export)."""
        f = self.fields
        cases: dict[str, set[str]] = {}
        self.skipped = 0
        for p in sorted(glob.glob(scope["glob"])):
            for row in _rows(Path(p)):
                if row.get(f.case) in (None, ""):
                    self.skipped += 1
                    continue
                cases.setdefault(case_key(row[f.case]), set()).add(p)
        for case, files in sorted(cases.items()):
            yield SourceRef(connector=self.name, key=case, meta={"files": sorted(files)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        f = self.fields
        rows = [r for p in ref.meta["files"] for r in _rows(Path(p)) if r.get(f.case) not in (None, "") and case_key(r[f.case]) == ref.key]
        return RawRecord(ref=ref, payload={"rows": rows}, privacy=f.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        return rows_to_trace(raw.payload["rows"], self.fields, case=raw.ref.key, connector=self.name, privacy=raw.privacy)


def case_key(v: Any) -> str:
    """A case id as text: 1, "1" and 1.0 are one case."""
    if isinstance(v, float) and v.is_integer():
        v = int(v)
    return str(v)


def _row_hash(r: dict[str, Any]) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(r, sort_keys=True, default=str, ensure_ascii=False).encode()).hexdigest()


AGENT_KINDS = {"agent", "bot", "ai", "llm", "assistant", "copilot", "automation"}
SYSTEM_KINDS = {"system", "service", "integration", "job", "cron"}


def _actor_kind(v: Any) -> str:
    k = str(v or "human").strip().lower()
    return "agent" if k in AGENT_KINDS else "system" if k in SYSTEM_KINDS else "human"


def rows_to_trace(rows: list[dict[str, Any]], f: FieldMap, *, case: str, connector: str, privacy: str = "internal") -> Trace:
    """Rows of one case (any tabular export: a CSV, an audit table, a log) -> a Trace, via the field map."""
    from crucible.connectors.base import rfc3339

    seen: set[str] = set()
    unique = []
    for r in rows:  # overlapping exports repeat rows: one event per row id, else per identical row
        k = f"id:{r[f.id]}" if r.get(f.id) not in (None, "") else _row_hash(r)
        if k not in seen:
            seen.add(k)
            unique.append(r)
    rows = sorted(unique, key=lambda r: (_instant(r.get(f.time)), str(r.get(f.id) or ""), _row_hash(r)))
    actors: dict[str, TraceActor] = {}
    events: list[TraceEvent] = []
    by_id: dict[str, int] = {}
    for seq, r in enumerate(rows):
        who = r.get(f.actor)
        if who not in (None, "") and str(who) not in actors:
            actors[str(who)] = TraceActor(id=str(who), role=str(r.get(f.role) or "participant"), kind=_actor_kind(r.get(f.actor_kind)), handle=str(who))  # type: ignore[arg-type]
        writes = r.get(f.writes) or []
        if isinstance(writes, str):
            writes = [w for w in writes.split(";") if w]
        kind = r.get(f.kind) or "action"
        obj = r.get(f.object)
        data = {k: r[k] for k in f.extra if k in r and r[k] is not None}
        if writes:
            data["writes"] = list(writes)
        if r.get(f.id) not in (None, ""):
            data["row_id"] = str(r[f.id])
        when = rfc3339(r.get(f.time))
        if when is None and r.get(f.time) not in (None, ""):
            data["time_raw"] = str(r[f.time])
        reply = by_id.get(str(r.get(f.reply_to))) if r.get(f.reply_to) not in (None, "") else None
        events.append(TraceEvent(seq=seq, time=when, kind=kind if kind in KINDS else "action",
                                 actor=str(who) if who not in (None, "") else None, type=str(r.get(f.type) or "event"), text=None if r.get(f.text) is None else str(r.get(f.text)),
                                 target=SignalTarget(object=str(obj)) if obj not in (None, "") else None, reply_to=reply, data=data))
        if r.get(f.id) not in (None, ""):
            by_id[str(r[f.id])] = seq
    # with a family, file_drop keeps "<family>:<case>"; other tabular connectors add their name (an audit table and
    # an export of the same ticket overwrote each other)
    tid = f"{f.family}:{case}" if f.family and connector == "file_drop" else f"{f.family}:{connector}:{case}" if f.family else f"{connector}:{case}"
    return Trace(trace_id=tid, family=f.family,
                 source=TraceSource(connector=connector, ref=case, privacy=privacy), actors=list(actors.values()), events=events)  # type: ignore[arg-type]
