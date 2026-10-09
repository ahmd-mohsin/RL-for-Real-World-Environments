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


def ingest(connector: Connector, refs: list[SourceRef], store, *, policy=None) -> list[str]:
    """Fetch, normalize, redact and store; returns the trace document keys (idempotent: same source, same key)."""
    from crucible.connectors.redact import RedactionPolicy, redact

    policy = policy or RedactionPolicy()
    keys = []
    for ref in refs:
        trace = redact(connector.normalize(connector.fetch(ref)), policy)
        keys.append(store.put(trace))
    return keys
