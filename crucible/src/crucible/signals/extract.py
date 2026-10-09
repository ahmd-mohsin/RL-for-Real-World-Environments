"""Signal extraction: run the pack's mapper and outcome labeler on a trace, attribute what the mapper left
ungraded, mark decision points, and return the trace's next version.

Mappers and labelers are pack plugins, looked up by the trace's ``family`` in the registry (kinds
``signal_mapper`` / ``outcome_labeler``; the longest registered family prefix wins), so the core never imports a
pack. Signal ids are deterministic (trace, type, source event), so re-extracting the same trace is idempotent.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from crucible.core.ids import sha256_text
from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceOutcome


@runtime_checkable
class SignalMapper(Protocol):
    """Pack-supplied: raw events of one family -> generic signals (types from ``signals.taxonomy``)."""

    def map(self, trace: Trace) -> list[Signal]: ...


@runtime_checkable
class OutcomeLabeler(Protocol):
    """Pack-supplied: the terminal outcome of a trace (merged and kept 30 days, ticket solved and rated, ...)."""

    def label(self, trace: Trace) -> TraceOutcome: ...


def signal_id(trace_id: str, type_: str, source_event: int | None, extra: str = "") -> str:
    return "sig_" + sha256_text(f"{trace_id}|{type_}|{source_event}|{extra}")[:16]


def _for_family(kind: str, family: str | None):
    from crucible.core.registry import get_registry

    reg = get_registry()
    names = sorted((n for n in reg.names(kind) if family and (family == n or family.startswith(n + "/"))), key=len, reverse=True)
    return reg.get(kind, names[0])() if names else None


def mapper_for(family: str | None) -> SignalMapper | None:
    return _for_family("signal_mapper", family)


def labeler_for(family: str | None) -> OutcomeLabeler | None:
    return _for_family("outcome_labeler", family)


def extract(trace: Trace, *, mapper: SignalMapper | None = None, labeler: OutcomeLabeler | None = None) -> Trace:
    from crucible.signals.attribute import attribute
    from crucible.signals.pivots import mark_pivots

    mapper = mapper or mapper_for(trace.family)
    labeler = labeler or labeler_for(trace.family)
    if mapper is None:
        raise LookupError(f"no signal_mapper registered for family {trace.family!r}")
    signals = attribute(trace, mapper.map(trace))
    seen: set[str] = set()
    for s in signals:
        if s.id in seen:
            raise ValueError(f"duplicate signal id {s.id} from the {type(mapper).__name__} mapper")
        seen.add(s.id)
    out = trace.model_copy(update={"signals": signals})
    if labeler is not None:
        out = out.model_copy(update={"outcome": labeler.label(out)})
    return mark_pivots(out)
