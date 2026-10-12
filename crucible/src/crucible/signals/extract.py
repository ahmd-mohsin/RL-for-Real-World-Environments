"""Signal extraction: run the pack's mapper and outcome labeler on a trace, attribute what the mapper left
ungraded, mark decision points, and return the trace's next version.

Mappers and labelers are pack plugins, looked up by the trace's ``family`` in the registry (kinds
``signal_mapper`` / ``outcome_labeler``; the longest registered family prefix wins), so the core never imports a
pack. The generic session rules (``signals.session``: interrupts, rejected tool use) run on every trace beside the
pack's mapper. Signal ids are deterministic (trace, type, the source event's stable key), so re-extracting the same
trace is idempotent.

M4 audit (2026-10-12):
- ids were keyed on the source event's seq, a position: one earlier row changed every later id, and the M3 store
  carried signals under ids a fresh extract no longer produced (forge provenance and trap checks cite ids). The key
  is now ``core.schema.trace.event_key`` (the connector's row id, else time, type, actor and text);
- extraction replaced every signal on the trace: hand-added signals and computed reliabilities were lost. Mapper
  output now replaces mapper output by id; signals another producer added (``meta.by`` not ``mapper:...``) stay;
- output is validated: duplicate ids, unknown grades and polarities against the taxonomy raise here, naming the mapper,
  instead of failing later when another stage loads the stored trace; grades of steps after the signal are dropped and
  flagged (``meta.dropped_grades``);
- a family with a labeler (or only session events) but no mapper is labelled instead of raising.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from crucible.core.ids import sha256_text
from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceEvent, TraceOutcome, event_key


@runtime_checkable
class SignalMapper(Protocol):
    """Pack-supplied: raw events of one family -> generic signals (types from ``signals.taxonomy``)."""

    def map(self, trace: Trace) -> list[Signal]: ...


@runtime_checkable
class OutcomeLabeler(Protocol):
    """Pack-supplied: the terminal outcome of a trace (merged and kept 30 days, ticket solved and rated, ...)."""

    def label(self, trace: Trace) -> TraceOutcome: ...


class ExtractError(ValueError):
    """A mapper produced signals that cannot be stored (duplicate ids, unknown or future grades, wrong polarity)."""


def signal_id(trace_id: str, type_: str, source: TraceEvent | int | None, extra: str = "") -> str:
    """Pass the source event itself: its stable key survives re-ingests. A bare seq (or None) is kept for signals with
    no source event and for callers that predate the key."""
    key = event_key(source) if isinstance(source, TraceEvent) else source
    return "sig_" + sha256_text(f"{trace_id}|{type_}|{key}|{extra}")[:16]


def _for_family(kind: str, family: str | None):
    from crucible.core.registry import get_registry

    reg = get_registry()
    names = sorted((n for n in reg.names(kind) if family and (family == n or family.startswith(n + "/"))), key=len, reverse=True)
    return reg.get(kind, names[0])() if names else None


def mapper_for(family: str | None) -> SignalMapper | None:
    return _for_family("signal_mapper", family)


def labeler_for(family: str | None) -> OutcomeLabeler | None:
    return _for_family("outcome_labeler", family)


def _by(s: Signal) -> str:
    return str(s.meta.get("by") or "")


def validate(trace: Trace, signals: list[Signal], producer: str) -> None:
    """Fail at extraction, naming the producer, rather than in whichever later stage loads the stored trace."""
    from crucible.signals.taxonomy import TAXONOMY

    known = {e.seq for e in trace.events}
    seen: set[str] = set()
    for s in signals:
        where = f"{producer} on {trace.trace_id}: {s.type} signal {s.id}"
        if s.id in seen:
            raise ExtractError(f"{where}: duplicate signal id (two signals of one type from one event need distinct `extra`)")
        seen.add(s.id)
        if bad := [g for g in s.grades if g not in known]:
            raise ExtractError(f"{where}: grades unknown events {bad}")
        if s.source_event is not None and s.source_event not in known:
            raise ExtractError(f"{where}: source event {s.source_event} is not in the trace")
        want = TAXONOMY[s.type].polarity
        if want != 0 and s.polarity != want:
            raise ExtractError(f"{where}: polarity {s.polarity}, but {s.type} is always {want:+d} (taxonomy)")


def _not_after(s: Signal) -> Signal:
    """A signal grades steps before the event that carries it (a check may sit on its own step's event, e.g. a frontier
    run's final patch). Grades after it come from clock skew in the source (a CI run "completed" before the commit it
    ran on): they are dropped and the signal says so, rather than crediting a later step or failing the whole trace."""
    if s.source_event is None:
        return s
    late = [g for g in s.grades if g > s.source_event or (g == s.source_event and s.type != "check_result")]
    if not late:
        return s
    keep = [g for g in s.grades if g not in late]
    meta = {**s.meta, "dropped_grades": late, **({} if keep else {"attribution": "unattributed", "reason": "graded steps are after the signal"})}
    return s.model_copy(update={"grades": keep, "meta": meta})


def extract(trace: Trace, *, mapper: SignalMapper | None = None, labeler: OutcomeLabeler | None = None) -> Trace:
    from crucible.signals.attribute import attribute
    from crucible.signals.pivots import mark_pivots
    from crucible.signals.session import SessionSignalMapper

    mapper = mapper or mapper_for(trace.family)
    labeler = labeler or labeler_for(trace.family)
    produced: list[Signal] = []
    for m in ([mapper] if mapper is not None else []) + [SessionSignalMapper()]:
        name = f"mapper:{type(m).__name__}"
        got = [s if _by(s).startswith("mapper:") else s.model_copy(update={"meta": {**s.meta, "by": name}}) for s in m.map(trace)]
        validate(trace, produced + got, name)
        produced += [_not_after(s) for s in got]
    produced = attribute(trace, produced)
    old = {s.id: s for s in trace.signals}
    # signals another producer added (a hand label, a review tool) declare ``meta.by``; unmarked ones are older mapper output
    kept = [s for s in trace.signals if _by(s) and not _by(s).startswith("mapper:") and s.id not in {p.id for p in produced}]
    produced = [p.model_copy(update={"reliability": old[p.id].reliability}) if p.reliability is None and p.id in old and old[p.id].reliability is not None
                else p for p in produced]
    signals = produced + kept
    validate(trace, signals, "extract")
    out = trace.model_copy(update={"signals": signals})
    if labeler is not None:
        out = out.model_copy(update={"outcome": labeler.label(out)})
    return mark_pivots(out)
