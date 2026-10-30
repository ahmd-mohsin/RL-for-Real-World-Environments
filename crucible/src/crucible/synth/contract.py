"""Contracts as data and the one trusted grader (part2-paper-core §3.3, claim C1).

A contract ``C = (R, A, Q, M, N)`` plus ``exec`` checks:
- **R, required** ``EffectPattern``s: an address pattern (globs on app, collection, key and field; ``*`` matches
  anything), a ``kind`` (``insert``, ``update``, ``delete``, ``any`` or ``state``), a ``value`` predicate (an update's
  new value; a state pattern's field value), ``where`` predicates on the entity's fields (an insert's new row, a delete's
  old row, an update's or state pattern's row in the end state), and a ``count`` (exactly ``count`` changes unless
  ``count_max`` widens it; ``"any"`` = at least ``count``).
- **A, allowed** patterns in the same language, each with a ``why`` (an allowed effect is a claim that must be audited).
- **Q, answer** checks on the final answer: ``eq``, ``approx``, ``set_eq``, ``regex``, ``in``, ``present``, against a
  constant or a start/end-state value.
- **M, trace** constraints: ``before``, ``never``, ``at_most``, ``read_before_write``; each may quote the instruction
  sentence that ``stated`` it (MUT's equivalence test ignores unstated ones, so an unstated order demand is a D6).
- **N, normalizers** (``canon.Normalizers``).
- **exec** checks (code domains): a named black-box predicate from a registry (``register_exec``), part of R.

Predicates: ``eq``, ``in``, ``approx(tol, rel)``, ``regex`` (full match), ``any`` (present), ``rel(op, ref, delta,
tol)`` (a relation to the start state: ``ref="old"`` is the same cell at s₀, or an explicit start address),
``from_answer(path, tol)``.

``grade(run, C)`` (the pseudo-code of §3.3, fail-closed):

    Δ ← diff(N(s₀), N(s_T))
    for r in R:  no assignment of Δ satisfies r with its count    → MISSING_REQUIRED:<r.id>
    for δ in Δ:  δ is covered by neither R (assigned) nor A        → FORBIDDEN_EFFECT:<addr>
    for q in Q:  not q(answer, s₀, s_T)                            → ANSWER:<q.id>
    for m in M:  not m(trace)                                      → TRACE:<m.id>
    for x in exec: the predicate is false                          → MISSING_REQUIRED:<x.id>
    any exception, timeout, missing or unobserved state           → GRADER_ERROR:<stage>  (and nothing else)

**The assignment.** One change is one unit of evidence: it can satisfy at most **one** required pattern (decision; two
patterns that both match a change, e.g. "invoice 42 is held" and "some invoice is held", need two distinct changes, so
overlapping patterns add up instead of collapsing; patterns on two fields of one row are two changes anyway, since an
update is per field). A change matching a required pattern beyond its count is not covered by it (a duplicate write) and
must be covered by A. Formally, with each pattern ``r`` expanded into ``count`` required and ``count_max - count``
optional slots, the run passes the state checks iff some matching covers every required slot and every change outside
A. By the Mendelsohn–Dulmage theorem that holds iff a matching covers every required slot (else MISSING_REQUIRED for
the under-filled patterns) **and** a matching covers every non-A change (else FORBIDDEN_EFFECT for the uncovered ones),
so the two loops of the pseudo-code are exactly two maximum bipartite matchings (Kuhn's algorithm, deterministic order).
State patterns consume no change: they count entities of the end state.

Verdicts carry reason codes only (no values, no exception text): a grader message must not leak the contract.
"""

from __future__ import annotations

import concurrent.futures
import fnmatch
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import MISSING, CanonState, Change, Normalizers, cjson, content_hash, diff, is_number, norm_text, same_value
from crucible.synth.program import Run, leaves

GLOB = re.compile(r"[*?\[]")
ESCAPED = re.compile(r"\[[*?\[]\]")  # ``[*]``, ``[?]``, ``[[]``: a literal character (``infer._lit_glob``)


def is_glob(s: str) -> bool:
    """Does the glob match more than one string? (Escaped single characters are literals.)"""
    return bool(GLOB.search(ESCAPED.sub("", s)))


def unglob(s: str) -> str:
    """The one string a non-glob pattern matches."""
    return ESCAPED.sub(lambda m: m.group(0)[1], s)


class AddrRef(CrucibleModel):
    app: str
    coll: str
    key: str
    field: str = ""


# ---------------------------------------------------------------------------------------------------------- predicates

class PEq(CrucibleModel):
    op: Literal["eq"] = "eq"
    v: Any = None


class PIn(CrucibleModel):
    op: Literal["in"] = "in"
    vs: list[Any] = Field(default_factory=list)


class PApprox(CrucibleModel):
    op: Literal["approx"] = "approx"
    v: float
    tol: float = 0.0
    rel: float = 0.0


class PRegex(CrucibleModel):
    op: Literal["regex"] = "regex"
    pattern: str


class PAny(CrucibleModel):
    op: Literal["any"] = "any"


class PRel(CrucibleModel):
    op: Literal["rel"] = "rel"
    rel: Literal["eq", "ne", "lt", "le", "gt", "ge"] = "eq"
    ref: Literal["old"] | AddrRef = "old"
    delta: float = 0.0
    tol: float = 0.0


class PFromAnswer(CrucibleModel):
    op: Literal["from_answer"] = "from_answer"
    path: str | None = None
    tol: float = 0.0


Pred = Annotated[PEq | PIn | PApprox | PRegex | PAny | PRel | PFromAnswer, Field(discriminator="op")]


# ---------------------------------------------------------------------------------------------------------- patterns

class EffectPattern(CrucibleModel):
    id: str
    app: str = "*"
    coll: str = "*"
    key: str = "*"
    field: str = "*"
    kind: Literal["insert", "update", "delete", "any", "state"] = "update"
    value: Pred | None = None
    where: dict[str, Pred] = Field(default_factory=dict)
    count: int = 1
    count_max: int | Literal["any"] | None = None  # None: exactly ``count``
    why: str | None = None

    @model_validator(mode="after")
    def _check(self) -> EffectPattern:
        if self.count < 0:
            raise ValueError(f"{self.id}: count must be >= 0")
        if isinstance(self.count_max, int) and self.count_max < self.count:
            raise ValueError(f"{self.id}: count_max < count")
        if self.kind in ("insert", "delete") and (self.value is not None or self.field not in ("*", "")):
            raise ValueError(f"{self.id}: an {self.kind} pattern names a whole entity (no field, no value predicate; use where)")
        return self

    def optional_slots(self, n_changes: int) -> int:
        if self.count_max is None:
            return 0
        if self.count_max == "any":
            return n_changes
        return self.count_max - self.count

    def concrete(self) -> bool:
        return not any(is_glob(x) for x in (self.app, self.coll, self.key, self.field))


class AnswerCheck(CrucibleModel):
    id: str
    op: Literal["eq", "approx", "set_eq", "regex", "in", "present"] = "eq"
    path: str | None = None
    value: Any = None
    tol: float = 0.0
    rel: float = 0.0
    ref: AddrRef | None = None  # compare with a state value instead of ``value``
    ref_state: Literal["start", "end"] = "end"


class EntityPat(CrucibleModel):
    app: str = "*"
    coll: str = "*"
    key: str = "*"

    def covers(self, ent: Sequence[str]) -> bool:
        return fnmatch.fnmatchcase(ent[0], self.app) and fnmatch.fnmatchcase(ent[1], self.coll) and fnmatch.fnmatchcase(ent[2], self.key)


class TraceConstraint(CrucibleModel):
    """``before``: a call matching ``tool`` precedes the first call matching ``then``; ``never``: no call matches
    ``tool``; ``at_most``: at most ``n`` calls match ``tool``; ``read_before_write``: before the first call that changes
    an entity matching ``entity``, a read-only call's arguments or observation named its key. Errored calls count: an
    attempt is an act. ``stated`` quotes the instruction text that states the constraint."""

    id: str
    op: Literal["before", "never", "at_most", "read_before_write"]
    tool: str | None = None
    then: str | None = None
    n: int | None = None
    entity: EntityPat | None = None
    stated: str | None = None

    @model_validator(mode="after")
    def _check(self) -> TraceConstraint:
        need = {"before": ("tool", "then"), "never": ("tool",), "at_most": ("tool", "n"), "read_before_write": ("entity",)}[self.op]
        missing = [f for f in need if getattr(self, f) is None]
        if missing:
            raise ValueError(f"{self.id}: {self.op} needs {missing}")
        return self


class ExecCheck(CrucibleModel):
    """A black-box predicate from the exec registry. ``kind`` says what it is for: a ``require`` check is evidence the
    task was done (a hidden fail-to-pass test); a ``guard`` must hold but holds by design at the start state too (a
    pass-to-pass regression suite), so RED does not call it vacuous. Both gate the verdict the same way."""

    id: str
    name: str
    args: dict[str, Any] = Field(default_factory=dict)
    expect: Any = True
    kind: Literal["require", "guard"] = "require"


class Contract(CrucibleModel):
    required: list[EffectPattern] = Field(default_factory=list)
    allowed: list[EffectPattern] = Field(default_factory=list)
    answer: list[AnswerCheck] = Field(default_factory=list)
    trace: list[TraceConstraint] = Field(default_factory=list)
    normalizers: Normalizers = Field(default_factory=Normalizers)
    exec: list[ExecCheck] = Field(default_factory=list)
    observation_only: bool = False  # a black-box World: no canon, so no R/A; the certificate records no frame guarantee

    @model_validator(mode="after")
    def _check(self) -> Contract:
        ids = [x.id for x in [*self.required, *self.allowed, *self.answer, *self.trace, *self.exec]]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            raise ValueError(f"duplicate contract ids {dup}")
        for a in self.allowed:
            if not (a.why or "").strip():
                raise ValueError(f"allowed pattern {a.id} has no why")
            if a.kind == "state":
                raise ValueError(f"allowed pattern {a.id}: a state pattern allows nothing")
        if self.observation_only and (self.required or self.allowed):
            raise ValueError("an observation-only contract has no effect patterns")
        return self

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 32)


class Verdict(CrucibleModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)

    @property
    def codes(self) -> set[str]:
        return {r.split(":", 1)[0] for r in self.reasons}


# ---------------------------------------------------------------------------------------------------------- exec registry

ExecFn = Callable[[Run, dict[str, Any]], Any]
EXEC: dict[str, ExecFn] = {}


def register_exec(name: str, f: ExecFn) -> None:
    """Register a black-box predicate (the debug lane's test runner comes in M5)."""
    EXEC[name] = f


# ---------------------------------------------------------------------------------------------------------- evaluation

class _MissingState(LookupError):
    pass


class _Timeout(Exception):
    pass


def answer_at(answer: Any, path: str | None) -> Any:
    if path is None or path == "":
        return MISSING if answer is None else answer
    cur = answer
    for seg in path.split("."):
        if isinstance(cur, dict) and seg in cur:
            cur = cur[seg]
        elif isinstance(cur, list) and re.fullmatch(r"-?\d+", seg) and -len(cur) <= int(seg) < len(cur):
            cur = cur[int(seg)]
        else:
            return MISSING
    return cur


def _num(v: Any) -> float | None:
    from crucible.verify.answers import to_number

    if is_number(v):
        return float(v)
    if isinstance(v, str):
        return to_number(v)
    return None


def answer_equal(a: Any, b: Any, tol: float = 0.0) -> bool:
    """Answer equality: numbers (also numeric strings, ``"1,250.00"``) within ``tol``; strings case- and
    whitespace-normalised; lists element-wise; else ``same_value``."""
    if a is MISSING or b is MISSING:
        return False
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(answer_equal(x, y, tol) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(answer_equal(a[k], b[k], tol) for k in a)
    x, y = _num(a), _num(b)
    if x is not None and y is not None and not isinstance(a, bool) and not isinstance(b, bool):
        return abs(x - y) <= tol or x == y
    if isinstance(a, str) and isinstance(b, str):
        return norm_text(a) == norm_text(b)
    return same_value(a, b)


def _set_key(v: Any) -> str:
    x = _num(v)
    if x is not None and not isinstance(v, bool):
        return f"n:{x!r}"
    if isinstance(v, str):
        return "s:" + norm_text(v)
    return "j:" + cjson(v)


class _Ctx:
    def __init__(self, s0: CanonState, sT: CanonState, answer: Any, deadline: float):
        self.s0, self.sT, self.answer, self.deadline = s0, sT, answer, deadline

    def tick(self) -> None:
        if time.monotonic() > self.deadline:
            raise _Timeout


def _relate(op: str, v: Any, r: Any, delta: float, tol: float) -> bool:
    if is_number(v) and is_number(r):
        t = r + delta
        return {"eq": abs(v - t) <= tol, "ne": abs(v - t) > tol, "lt": v < t, "le": v <= t + tol, "gt": v > t, "ge": v >= t - tol}[op]
    if op in ("eq", "ne") and delta == 0:
        return same_value(v, r) == (op == "eq")
    return False


def eval_pred(p: Any, v: Any, old: Any, ctx: _Ctx) -> bool:
    """One predicate on value ``v``; ``old`` is the same cell at s₀ (``MISSING`` for an insert)."""
    if isinstance(p, PAny):
        return v is not MISSING
    if v is MISSING:
        return False
    if isinstance(p, PEq):
        return same_value(v, p.v)
    if isinstance(p, PIn):
        return any(same_value(v, x) for x in p.vs)
    if isinstance(p, PApprox):
        return is_number(v) and math.isfinite(v) and abs(v - p.v) <= max(p.tol, p.rel * abs(p.v))
    if isinstance(p, PRegex):
        return re.fullmatch(p.pattern, v if isinstance(v, str) else cjson(v)) is not None
    if isinstance(p, PRel):
        if p.ref == "old":
            r = old
        else:
            r = ctx.s0.get(p.ref)
            if r is MISSING:
                raise _MissingState(str(p.ref))
        return r is not MISSING and _relate(p.rel, v, r, p.delta, p.tol)
    if isinstance(p, PFromAnswer):
        a = answer_at(ctx.answer, p.path)
        return answer_equal(v, a, p.tol)
    raise TypeError(f"unknown predicate {type(p).__name__}")


def _globs(p: EffectPattern, app: str, coll: str, key: str) -> bool:
    return fnmatch.fnmatchcase(app, p.app) and fnmatch.fnmatchcase(coll, p.coll) and fnmatch.fnmatchcase(key, p.key)


def _where(p: EffectPattern, row: Mapping[str, Any] | None, start_row: Mapping[str, Any] | None, ctx: _Ctx) -> bool:
    if p.where and row is None:
        return False
    for f, pred in p.where.items():
        v = row.get(f, MISSING) if row is not None else MISSING
        old = start_row.get(f, MISSING) if start_row is not None else MISSING
        if not eval_pred(pred, v, old, ctx):
            return False
    return True


def matches(p: EffectPattern, ch: Change, ctx: _Ctx) -> bool:
    """Does change ``ch`` satisfy pattern ``p`` (address globs, kind, value and where predicates)?"""
    if p.kind == "state" or (p.kind != "any" and p.kind != ch.kind):
        return False
    a = ch.addr
    if not _globs(p, a.app, a.coll, a.key):
        return False
    start_row = ctx.s0.entity(a.app, a.coll, a.key)
    if ch.kind == "update":
        if not fnmatch.fnmatchcase(a.field, p.field):
            return False
        if p.value is not None and not eval_pred(p.value, ch.new, ch.old, ctx):
            return False
        row = ctx.sT.entity(a.app, a.coll, a.key)
    elif ch.kind == "insert":
        if p.value is not None:
            return False
        row, start_row = ch.new, None
    else:
        if p.value is not None:
            return False
        row = ch.old
    return _where(p, row, start_row, ctx)


def state_count(p: EffectPattern, ctx: _Ctx, state: CanonState | None = None) -> int:
    """Entities of ``state`` (default: the end state) that satisfy a state pattern."""
    s = ctx.sT if state is None else state
    n = 0
    for app, coll, key, row in s.entities():
        if not _globs(p, app, coll, key):
            continue
        start_row = ctx.s0.entity(app, coll, key)
        if p.value is not None:
            fields = [f for f in row if fnmatch.fnmatchcase(f, p.field)]
            if not any(eval_pred(p.value, row[f], start_row.get(f, MISSING) if start_row else MISSING, ctx) for f in fields):
                continue
        if _where(p, row, start_row, ctx):
            n += 1
    return n


def state_ok(p: EffectPattern, n: int) -> bool:
    hi = p.count if p.count_max is None else (math.inf if p.count_max == "any" else p.count_max)
    return p.count <= n <= hi


def _kuhn(adj: list[list[int]], n_right: int, ctx: _Ctx) -> list[int]:
    """Maximum bipartite matching; ``adj[l]`` lists the right nodes of left node ``l``. Returns left -> right (or -1)."""
    match_r = [-1] * n_right
    match_l = [-1] * len(adj)

    def augment(u: int, seen: list[bool]) -> bool:
        for w in adj[u]:
            if seen[w]:
                continue
            seen[w] = True
            if match_r[w] == -1 or augment(match_r[w], seen):
                match_r[w], match_l[u] = u, w
                return True
        return False

    for u in range(len(adj)):
        ctx.tick()
        augment(u, [False] * n_right)
    return match_l


def assign(required: Sequence[EffectPattern], allowed: Sequence[EffectPattern], delta: Sequence[Change], ctx: _Ctx) -> tuple[list[str], list[Change]]:
    """The under-filled required patterns and the uncovered changes (see the module docstring)."""
    change_pats = [p for p in required if p.kind != "state"]
    edges = [[i for i, p in enumerate(change_pats) if matches(p, ch, ctx)] for ch in delta]
    in_a = [any(matches(a, ch, ctx) for a in allowed) for ch in delta]
    req_slots: list[int] = []  # slot -> pattern index
    opt_slots: list[int] = []
    for i, p in enumerate(change_pats):
        req_slots += [i] * p.count
        opt_slots += [i] * p.optional_slots(len(delta))
    # (a) every required slot gets a distinct matching change
    by_pat: dict[int, list[int]] = {}
    for c, pats in enumerate(edges):
        for i in pats:
            by_pat.setdefault(i, []).append(c)
    m = _kuhn([by_pat.get(i, []) for i in req_slots], len(delta), ctx)
    short = sorted({change_pats[req_slots[s]].id for s, c in enumerate(m) if c == -1}, key=[p.id for p in change_pats].index)
    # (b) every change outside A gets a distinct slot of a pattern it matches
    slots = req_slots + opt_slots
    slots_of: dict[int, list[int]] = {}
    for s, i in enumerate(slots):
        slots_of.setdefault(i, []).append(s)
    must = [c for c in range(len(delta)) if not in_a[c]]
    m2 = _kuhn([[s for i in edges[c] for s in slots_of.get(i, [])] for c in must], len(slots), ctx)
    uncovered = [delta[must[k]] for k, s in enumerate(m2) if s == -1]
    return short, uncovered


def _trace_ok(m: TraceConstraint, run: Run) -> bool:
    calls = run.calls
    hit = [i for i, c in enumerate(calls) if m.tool is not None and fnmatch.fnmatchcase(c.tool, m.tool)]
    if m.op == "never":
        return not hit
    if m.op == "at_most":
        return len(hit) <= (m.n or 0)
    if m.op == "before":
        later = [i for i, c in enumerate(calls) if fnmatch.fnmatchcase(c.tool, m.then or "")]
        return not later or (bool(hit) and hit[0] < later[0])
    assert m.entity is not None
    done: set[tuple[str, ...]] = set()
    for i, c in enumerate(calls):
        if c.read_only:
            continue
        if c.effects is None:
            raise _MissingState("per-call effects were not tracked")
        for ent in c.effects:
            t = tuple(ent)
            if t in done or not m.entity.covers(ent):
                continue
            done.add(t)
            key = ent[2]
            if not any(r.read_only and (any(str(v) == key for _, v in leaves(r.args)) or any(str(v) == key for _, v in leaves(r.obs.value)))
                       for r in calls[:i]):
                return False
    return True


def _answer_ok(q: AnswerCheck, run: Run, ctx: _Ctx) -> bool:
    a = answer_at(run.answer, q.path)
    if q.op == "present":
        return a is not MISSING and a is not None
    if q.ref is not None:
        st = ctx.s0 if q.ref_state == "start" else ctx.sT
        e = st.get(q.ref)
        if e is MISSING:
            raise _MissingState(str(q.ref))
    else:
        e = q.value
    if a is MISSING:
        return False
    if q.op == "eq":
        return answer_equal(a, e)
    if q.op == "approx":
        x, y = _num(a), _num(e)
        return x is not None and y is not None and abs(x - y) <= max(q.tol, q.rel * abs(y))
    if q.op == "set_eq":
        return isinstance(a, list) and isinstance(e, list) and {_set_key(x) for x in a} == {_set_key(x) for x in e}
    if q.op == "regex":
        return re.fullmatch(str(e), a if isinstance(a, str) else cjson(a)) is not None
    if q.op == "in":
        return isinstance(e, list) and any(answer_equal(a, x) for x in e)
    raise ValueError(q.op)


def _exec_ok(x: ExecCheck, run: Run, registry: Mapping[str, ExecFn], remaining: float) -> bool:
    f = registry[x.name]
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    try:
        res = pool.submit(f, run, dict(x.args)).result(timeout=max(0.001, remaining))
    except concurrent.futures.TimeoutError:
        raise _Timeout from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    return same_value(res, x.expect)


def grade(run: Run, contract: Contract, *, registry: Mapping[str, ExecFn] | None = None, deadline_s: float = 30.0) -> Verdict:
    """The trusted grader (see the module docstring). Pure: reads ``run`` and ``contract`` only."""
    reg = EXEC if registry is None else registry
    stage = "setup"
    try:
        t_end = time.monotonic() + deadline_s
        have_state = run.start is not None and run.end is not None
        if not have_state and not contract.observation_only:
            return Verdict(passed=False, reasons=["GRADER_ERROR:missing_state"])
        reasons: list[str] = []
        s0 = run.start or CanonState()
        sT = run.end or CanonState()
        unobserved = set(s0.unobserved) | set(sT.unobserved)
        if any(not is_glob(p.app) and p.app in unobserved for p in contract.required):
            return Verdict(passed=False, reasons=["GRADER_ERROR:missing_state"])
        n = contract.normalizers
        ctx = _Ctx(n.apply(s0), n.apply(sT), run.answer, t_end)
        if have_state:
            stage = "diff"
            delta = diff(s0, sT, n)
            stage = "required"
            short, uncovered = assign(contract.required, contract.allowed, delta, ctx)
            for p in contract.required:
                ctx.tick()
                if p.kind == "state" and not state_ok(p, state_count(p, ctx)):
                    short.append(p.id)
            order = [p.id for p in contract.required]
            reasons += [f"MISSING_REQUIRED:{i}" for i in sorted(set(short), key=order.index)]
            stage = "frame"
            reasons += [f"FORBIDDEN_EFFECT:{c.addr}" for c in uncovered]
        stage = "answer"
        for q in contract.answer:
            ctx.tick()
            if not _answer_ok(q, run, ctx):
                reasons.append(f"ANSWER:{q.id}")
        stage = "trace"
        for m in contract.trace:
            ctx.tick()
            if not _trace_ok(m, run):
                reasons.append(f"TRACE:{m.id}")
        stage = "exec"
        for x in contract.exec:
            if not _exec_ok(x, run, reg, t_end - time.monotonic()):
                reasons.append(f"MISSING_REQUIRED:{x.id}")
        ctx.tick()
        return Verdict(passed=not reasons, reasons=reasons)
    except _Timeout:
        return Verdict(passed=False, reasons=["GRADER_ERROR:timeout"])
    except _MissingState:
        return Verdict(passed=False, reasons=["GRADER_ERROR:missing_state"])
    except Exception:  # noqa: BLE001 - fail closed on anything, with a fixed code (no message leaves the grader)
        return Verdict(passed=False, reasons=[f"GRADER_ERROR:{stage}"])
