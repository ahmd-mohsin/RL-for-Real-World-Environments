"""``file_drop``: event logs exported from any system (JSONL or CSV) -> one Trace per case.

An enterprise system that can export "who did what to which object when" is enough: a ticketing tool, an ERP
audit log, a CRM history. Rows are grouped by a case key (ticket id, invoice id), sorted by time, and mapped to
TraceEvents through ``FieldMap``. ``writes`` (the objects an event changed, e.g. ``ticket:42.status``) is what
generic step attribution in M4 uses: a signal about an object grades the event that last wrote it.
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


def _rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".csv":
        with path.open(newline="") as f:
            return list(csv.DictReader(f))
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class FileDropConnector:
    name = "file_drop"

    def __init__(self, fields: FieldMap | None = None):
        self.fields = fields or FieldMap()

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        """scope: {"glob": "logs/*.jsonl"} -> one ref per case found in the matching files."""
        f = self.fields
        cases: dict[str, set[str]] = {}
        for p in sorted(glob.glob(scope["glob"])):
            for row in _rows(Path(p)):
                cases.setdefault(str(row[f.case]), set()).add(p)
        for case, files in sorted(cases.items()):
            yield SourceRef(connector=self.name, key=case, meta={"files": sorted(files)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        f = self.fields
        rows = [r for p in ref.meta["files"] for r in _rows(Path(p)) if str(r[f.case]) == ref.key]
        return RawRecord(ref=ref, payload={"rows": rows}, privacy=f.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        f = self.fields
        rows = sorted(raw.payload["rows"], key=lambda r: (str(r.get(f.time) or ""), str(r.get(f.id) or "")))
        actors: dict[str, TraceActor] = {}
        events: list[TraceEvent] = []
        by_id: dict[str, int] = {}
        for seq, r in enumerate(rows):
            who = r.get(f.actor)
            if who and who not in actors:
                actors[who] = TraceActor(id=str(who), role=str(r.get(f.role) or "participant"), kind=r.get(f.actor_kind) or "human", handle=str(who))
            writes = r.get(f.writes) or []
            if isinstance(writes, str):
                writes = [w for w in writes.split(";") if w]
            kind = r.get(f.kind) or "action"
            obj = r.get(f.object)
            data = {k: r[k] for k in f.extra if k in r}
            if writes:
                data["writes"] = list(writes)
            reply = by_id.get(str(r.get(f.reply_to))) if r.get(f.reply_to) not in (None, "") else None
            events.append(TraceEvent(seq=seq, time=r.get(f.time), kind=kind if kind in KINDS else "action", actor=str(who) if who else None,
                                     type=str(r.get(f.type) or "event"), text=r.get(f.text), target=SignalTarget(object=str(obj)) if obj else None,
                                     reply_to=reply, data=data))
            if r.get(f.id) not in (None, ""):
                by_id[str(r[f.id])] = seq
        return Trace(trace_id=f"{f.family or 'file_drop'}:{raw.ref.key}", family=f.family,
                     source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy), actors=list(actors.values()), events=events)
