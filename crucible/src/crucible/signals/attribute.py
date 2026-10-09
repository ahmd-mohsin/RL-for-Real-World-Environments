"""Step attribution: which event each signal grades (ARCHITECTURE M4).

The generic rule: a signal about an object grades **the event that last wrote that object** before the signal
(``event.data["writes"]`` lists the objects an event changed; ``ticket:42`` covers ``ticket:42.status``). Packs
with a better rule set ``grades`` themselves -- the coding pack uses blame within the PR, computed by its
connector -- and this rule only fills signals they left ungraded.
"""

from __future__ import annotations

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace


def _covers(written: str, obj: str) -> bool:
    return written == obj or obj.startswith((written + ".", written + ":")) or written.startswith(obj + ".")


def last_writer(trace: Trace, obj: str, before: int, *, exclude_actors: set[str] | None = None) -> int | None:
    """Seq of the latest event before ``before`` that wrote ``obj`` (optionally ignoring some actors' writes)."""
    for e in reversed(trace.events):
        if e.seq >= before:
            continue
        if exclude_actors and e.actor in exclude_actors:
            continue
        if any(_covers(w, obj) for w in e.data.get("writes", [])):
            return e.seq
    return None


def attribute(trace: Trace, signals: list[Signal]) -> list[Signal]:
    """Fill ``grades`` of step signals that have a target object but no grades yet (generic last-writer rule)."""
    from crucible.signals.taxonomy import TAXONOMY

    out = []
    for s in signals:
        if not s.grades and TAXONOMY[s.type].grades == "step" and s.target and s.target.object and s.source_event is not None:
            src = next((e for e in trace.events if e.seq == s.source_event), None)
            w = last_writer(trace, s.target.object, s.source_event, exclude_actors={src.actor} if src and src.actor and s.meta.get("exclude_self") else None)
            if w is not None:
                s = s.model_copy(update={"grades": [w], "meta": {**s.meta, "attributed_by": "last_writer"}})
        out.append(s)
    return out
