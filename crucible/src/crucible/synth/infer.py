"""The skeleton contract of an oracle run (part2-paper-core §3.4): R = Δ* with every value ``eq``, A = ∅, Q from the
answer, M = ∅.

The oracle passes its skeleton by construction (``tests`` assert it), so the contract writer only ever *relaxes* an
exact verifier, and every relaxation is a falsifiable edit. Per change:
- an update becomes ``(app, coll, key, field, update, value eq new)``;
- an insert becomes ``(app, coll, key, "", insert, where {every field eq})``, with key ``*`` in an ``id_canon``
  collection (its key is a content hash, and the ``where`` already pins the content);
- a delete becomes ``(app, coll, key, "", delete)``.
An answer becomes one ``eq`` check (``approx`` with ``answer_tol`` for a number when given).
"""

from __future__ import annotations

from typing import Any

from crucible.synth.canon import Normalizers, diff, is_number
from crucible.synth.contract import AnswerCheck, Contract, EffectPattern, PEq
from crucible.synth.program import Run


def skeleton(run: Run, normalizers: Normalizers | None = None, *, answer_tol: float | None = None) -> Contract:
    if run.start is None or run.end is None:
        if run.answer is None:
            raise ValueError("a black-box run without an answer has nothing to infer")
        return Contract(answer=[_answer_check(run.answer, answer_tol)], observation_only=True)
    n = normalizers or Normalizers()
    req: list[EffectPattern] = []
    for i, ch in enumerate(diff(run.start, run.end, n)):
        a = ch.addr
        rid = f"r{i}"
        if ch.kind == "update":
            req.append(EffectPattern(id=rid, app=a.app, coll=a.coll, key=_lit_glob(a.key), field=_lit_glob(a.field), kind="update", value=PEq(v=ch.new)))
        elif ch.kind == "insert":
            key = "*" if n.id_canon_for(a.app, a.coll) is not None else _lit_glob(a.key)
            req.append(EffectPattern(id=rid, app=a.app, coll=a.coll, key=key, field="*", kind="insert", where={f: PEq(v=v) for f, v in sorted(ch.new.items())}))
        else:
            req.append(EffectPattern(id=rid, app=a.app, coll=a.coll, key=_lit_glob(a.key), field="*", kind="delete"))
    answer = [_answer_check(run.answer, answer_tol)] if run.answer is not None else []
    return Contract(required=req, answer=answer, normalizers=n)


def _answer_check(v: Any, tol: float | None) -> AnswerCheck:
    if tol is not None and is_number(v):
        return AnswerCheck(id="q0", op="approx", value=v, tol=tol)
    return AnswerCheck(id="q0", op="eq", value=v)


def _lit_glob(s: str) -> str:
    """A literal key or field as a glob that matches only itself (``[`` ``*`` ``?`` bracketed)."""
    return "".join(f"[{c}]" if c in "*?[" else c for c in s)

