"""The connector contract (DESIGN §5.1) and idempotent ingest.

A connector turns one source object (a pull request, a ticket's event log, an agent session) into one Trace:
``discover`` lists source refs, ``fetch`` returns the raw payload (cached by the connector), ``normalize``
builds the Trace, and ingest always runs ``redact`` before anything is stored. Ingesting the same source twice
yields the same content hash and stores nothing new.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.trace import Privacy, Trace

_TIME_FORMATS = ("%m/%d/%Y %H:%M:%S", "%m/%d/%Y %H:%M", "%m/%d/%Y", "%d.%m.%Y %H:%M:%S", "%d.%m.%Y %H:%M", "%d.%m.%Y",
                 "%Y/%m/%d %H:%M:%S", "%Y/%m/%d %H:%M", "%Y/%m/%d", "%d-%b-%Y %H:%M:%S", "%d %b %Y %H:%M")


def parse_time(value: Any):
    """An instant from what exports contain: ISO 8601 (any offset), a ``datetime``, epoch seconds or milliseconds, or a
    common spreadsheet format (naive times are UTC); ``None`` when it is none of these."""
    from datetime import UTC, datetime

    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)) or (isinstance(value, str) and value.strip().replace(".", "", 1).isdigit()):
        x = float(value)
        if x > 1e11:  # milliseconds
            x /= 1000
        return datetime.fromtimestamp(x, tz=UTC)
    s = str(value).strip()
    try:
        t = datetime.fromisoformat(s)  # 3.11+: "Z" and any offset
        return t if t.tzinfo else t.replace(tzinfo=UTC)
    except ValueError:
        pass
    for fmt in _TIME_FORMATS:
        try:
            return datetime.strptime(s, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def rfc3339(value: Any) -> str | None:
    """An event time as stored in traces: RFC 3339, the source's own text when it already is ISO 8601 (offsets kept)."""
    t = parse_time(value)
    if t is None:
        return None
    s = str(value).strip() if isinstance(value, str) else ""
    import re

    if s and "T" in s and s[:4].isdigit() and re.search(r"(?:Z|[+-]\d{2}:?\d{2})$", s):  # already RFC 3339 with a zone
        return s
    return t.isoformat().replace("+00:00", "Z")


class SourceRef(CrucibleModel):
    connector: str
    key: str  # stable id of the source object, e.g. "PointCloudLibrary/pcl#6430"
    url: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class RawRecord(CrucibleModel):
    ref: SourceRef
    payload: dict[str, Any]
    privacy: Privacy = "internal"
    fetched_at: str | None = None


@runtime_checkable
class Connector(Protocol):
    name: str

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]: ...

    def fetch(self, ref: SourceRef) -> RawRecord: ...

    def normalize(self, raw: RawRecord) -> Trace: ...


class IngestError(RuntimeError):
    """Some sources failed; the others were stored. ``keys`` are the stored traces, ``failed`` the (ref, error) pairs."""

    def __init__(self, keys: list[str], failed: list[tuple[SourceRef, BaseException]]):
        self.keys, self.failed = keys, failed
        names = ", ".join(f"{r.key}: {type(e).__name__}: {str(e)[:120]}" for r, e in failed[:5])
        super().__init__(f"{len(failed)} of {len(keys) + len(failed)} sources failed ({names}{', ...' if len(failed) > 5 else ''})")


def ingest(connector: Connector, refs: list[SourceRef], store, *, policy=None, errors: list | None = None) -> list[str]:
    """Fetch, normalize, redact and store; returns the trace document keys (idempotent: same source, same key).

    One failing source no longer aborts the rest (M3 audit: a vanished file lost a whole export). Every source is
    tried; with ``errors`` the failures are appended there as (ref, error), otherwise ``IngestError`` is raised after
    the others are stored."""
    from crucible.connectors.redact import RedactionPolicy, Redactor, redact

    policy = policy or RedactionPolicy()
    if policy.salt is None:  # resolve the tenant key once, not once per trace
        policy = policy.model_copy(update={"salt": Redactor(policy).salt})
    keys: list[str] = []
    failed: list[tuple[SourceRef, BaseException]] = []
    from crucible.core.prov import utc_now

    for ref in refs:
        fetched_at = utc_now()  # taken before the read: a slower read of the same source never becomes the head (R-16)
        try:
            trace = redact(connector.normalize(connector.fetch(ref)), policy)
        except Exception as e:  # noqa: BLE001 - reported per source
            failed.append((ref, e))
            continue
        keys.append(store.put(trace, fetched_at=fetched_at))
    if failed:
        if errors is None:
            raise IngestError(keys, failed)
        errors.extend(failed)
    return keys
