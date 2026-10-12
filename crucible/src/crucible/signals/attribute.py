"""Step attribution: which event each signal grades (ARCHITECTURE M4).

The generic rule: a signal about an object grades **the event that last wrote that object** before the signal
(``event.data["writes"]`` lists the objects an event changed; ``ticket:42`` covers ``ticket:42.status``). Packs
with a better rule set ``grades`` themselves -- the coding pack uses blame within the PR, computed by its
connector -- and this rule only fills signals they left ungraded.

M4 audit (2026-10-12):
- ``file:`` objects are paths: ``/`` is their only hierarchy separator (``.`` made ``a.py.bak`` cover ``a.py`` and a
  write to ``setup`` cover ``setup.py``), they are normalised (``./a.py`` is ``a.py``), a directory covers what is in
  it, and an absolute path matches a relative one it ends with (an agent writes ``/home/u/repo/src/a.py``, a review
  names ``src/a.py``; NeMo-Gym ``swe_pivot`` compares path suffixes the same way). Record objects keep ``.``/``:``.
- the App counts: a signal about ``ws_a``'s ``file:a.py`` is not graded by a peer's write in ``ws_b``;
- a rejection, violation or approval never grades its own author's write (a reviewer's fixup commit), and excluding an anonymous
  author excludes anonymous writes rather than nothing;
- a step signal left ungraded says why (``meta.attribution = "unattributed"``, ``reason``): it was indistinguishable
  from a trace-level signal.
"""

from __future__ import annotations

import posixpath

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace

NOBODY = "\0anonymous"  # stands for actor None in exclusion sets
# verdicts on someone else's work: their author's own writes are never the graded step (a revert, a rework or an
# escalation is often the actor undoing or handing off its own step, so those keep it)
JUDGEMENTS = frozenset({"rejection", "violation", "approval"})


def _path(p: str) -> str:
    p = posixpath.normpath(p)
    return p.removeprefix("./")


def _covers(written: str, obj: str) -> bool:
    if written.startswith("file:") or obj.startswith("file:"):
        if not (written.startswith("file:") and obj.startswith("file:")):
            return False
        w, o = _path(written[5:]), _path(obj[5:])
        if w == o or o.startswith(w.rstrip("/") + "/") or w.startswith(o.rstrip("/") + "/"):
            return True
        if w.startswith("/") != o.startswith("/"):  # one absolute, one relative to some root: suffix on a component boundary
            a, r = (w, o) if w.startswith("/") else (o, w)
            return not r.startswith("../") and a.endswith("/" + r)
        return False
    return written == obj or obj.startswith((written + ".", written + ":")) or written.startswith(obj + ".")


def last_writer(trace: Trace, obj: str, before: int, *, exclude_actors: set[str] | None = None, app: str | None = None) -> int | None:
    """Seq of the latest event before ``before`` that wrote ``obj`` (in ``app`` when given; optionally ignoring some
    actors' writes, ``NOBODY`` standing for events without an actor)."""
    for e in reversed(trace.events):
        if e.seq >= before:
            continue
        if exclude_actors and (e.actor if e.actor is not None else NOBODY) in exclude_actors:
            continue
        if app and e.target is not None and e.target.app and e.target.app != app:
            continue
        if any(_covers(w, obj) for w in e.data.get("writes", [])):
            return e.seq
    return None


def attribute(trace: Trace, signals: list[Signal]) -> list[Signal]:
    """Fill ``grades`` of step signals that have a target object but no grades yet (generic last-writer rule)."""
    from crucible.signals.taxonomy import TAXONOMY

    out = []
    for s in signals:
        if s.grades or TAXONOMY[s.type].grades != "step" or s.meta.get("attribution"):
            out.append(s)
            continue
        reason = None
        if s.source_event is None:
            reason = "no_source_event"
        elif not (s.target and s.target.object):
            reason = "no_target"
        else:
            src = next((e for e in trace.events if e.seq == s.source_event), None)
            author = (src.actor if src and src.actor is not None else NOBODY) if src is not None else None
            mine = s.meta.get("exclude_self") or (s.type in JUDGEMENTS and not s.meta.get("include_self"))
            w = last_writer(trace, s.target.object, s.source_event, exclude_actors={author} if mine and author else None, app=s.target.app)
            if w is not None:
                s = s.model_copy(update={"grades": [w], "meta": {**s.meta, "attributed_by": "last_writer"}})
            else:
                reason = "no_writer"
        if reason:
            s = s.model_copy(update={"meta": {**s.meta, "attribution": "unattributed", "reason": reason}})
        out.append(s)
    return out
