"""The straight-line oracle program π* (part2-paper-core §3.3) and its interpreter with taint tracking (C3).

A program is a list of steps ``id: tool(args)``; an argument is ``lit(v)``, ``ref("step.path")`` (a value read from an
earlier observation; integer segments index lists) or ``fn(name, args)`` over the fixed pure library ``LIBRARY``. An
optional ``answer`` is any argument (usually a ref or a fn over refs). Straight-line is enough because π* is per concrete
start state: branching is resolved when the program is built, which is what makes mutation and taint tractable.

``run(world, program, instruction)`` executes from ``reset(seed)`` and returns a ``Run``: every call with its resolved
arguments and observation (and, for write calls, the entities it changed), the answer, the start and end canonical
states, and the **taint map**: for every literal, every place the agent could have learned it.

Literal sourcing (one ``TaintEntry`` per scalar literal; a list or object literal is split into its leaves):
- ``instruction``: a normalised token match in ι, with the span. Numbers match numeric tokens numerically (``1,250.00``
  is ``1250``); strings match case-insensitively with any run of whitespace, on word boundaries.
- ``observation``: any leaf of an **earlier** step's observation (every step's, for the answer): numbers numerically,
  strings case/whitespace-normalised equal, or (3+ characters) a token inside a text leaf such as a policy document.
- ``given``: an entry of the intent card's ``given`` list (information the task hands the agent outside ι, e.g. an actor's).
- ``tool_spec``: the tool catalogue the agent sees: an enum value or a default in a schema, or a token of a description.
- ``structural``: ``None``, booleans and the empty string carry no task information and are not searched; nor do the
  schema positions of the library (``get``'s path, ``pluck``'s field, ``round``'s digits, the field names of a
  ``select`` filter, whose values are still sourced).
An entry with no source is UNSOURCED (TAINT reports it as D5).

Execution semantics, chosen so mutants are always gradable: a tool error is an observation and execution continues; a ref
that cannot be resolved (a dropped step, an errored observation, a missing path), a library error or a World exception
stops the program there (``Run.error``), and the state reached is what gets graded.

Ref anchors (PERT's "refs re-resolved"): whenever a ref indexes a list, the run records the index's element by content
hash. A run given those anchors resolves the same ref to the element with the same content wherever the list now has it,
so a shuffled or padded start state does not silently point the oracle at another entity.
"""

from __future__ import annotations

import math
import re
from collections.abc import Awaitable, Callable, Iterable, Iterator, Sequence
from typing import Annotated, Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import CanonState, cjson, content_hash, diff, is_number, norm_text, same_value
from crucible.synth.iface import Obs, SynthWorld, ToolInfo

STEP_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# ---------------------------------------------------------------------------------------------------------- the DSL

class Lit(CrucibleModel):
    op: Literal["lit"] = "lit"
    v: Any = None


class Ref(CrucibleModel):
    op: Literal["ref"] = "ref"
    path: str


class Fn(CrucibleModel):
    op: Literal["fn"] = "fn"
    name: str
    args: list[Arg] = Field(default_factory=list)


Arg = Annotated[Lit | Ref | Fn, Field(discriminator="op")]
Fn.model_rebuild()


def lit(v: Any) -> Lit:
    return Lit(v=v)


def ref(path: str) -> Ref:
    return Ref(path=path)


def fn(name: str, *args: Any) -> Fn:
    return Fn(name=name, args=[a if isinstance(a, Lit | Ref | Fn) else Lit(v=a) for a in args])


class Step(CrucibleModel):
    id: str
    tool: str
    args: dict[str, Arg] = Field(default_factory=dict)


class Program(CrucibleModel):
    steps: list[Step] = Field(default_factory=list)
    answer: Arg | None = None
    answer_template: str | None = None  # display only, ``{}`` is the answer; checks read the answer value

    def problems(self) -> list[str]:
        """Static checks a built oracle must pass (mutants may break them on purpose): step ids, unique, refs backwards,
        known library functions."""
        out: list[str] = []
        seen: set[str] = set()
        for s in self.steps:
            if not STEP_ID.match(s.id):
                out.append(f"step id {s.id!r} is not an identifier")
            if s.id in seen:
                out.append(f"duplicate step id {s.id!r}")
            for name, a in s.args.items():
                out += [f"{s.id}.{name}: {p}" for p in _arg_problems(a, seen)]
            seen.add(s.id)
        if self.answer is not None:
            out += [f"answer: {p}" for p in _arg_problems(self.answer, seen)]
        return out

    def step(self, sid: str) -> Step | None:
        return next((s for s in self.steps if s.id == sid), None)

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 32)


def _arg_problems(a: Arg, seen: set[str]) -> list[str]:
    if isinstance(a, Ref):
        head = a.path.split(".", 1)[0]
        return [] if head in seen else [f"ref {a.path!r} names no earlier step"]
    if isinstance(a, Fn):
        out = [] if a.name in LIBRARY else [f"unknown function {a.name!r}"]
        for x in a.args:
            out += _arg_problems(x, seen)
        return out
    return []


def walk_args(a: Arg, loc: str) -> Iterator[tuple[str, Arg]]:
    """Every argument node with its location (``s3.amount``, ``s3.amount/fn:sum[0]``)."""
    yield loc, a
    if isinstance(a, Fn):
        for i, x in enumerate(a.args):
            yield from walk_args(x, f"{loc}/fn:{a.name}[{i}]")


def program_locs(p: Program) -> Iterator[tuple[str, Arg]]:
    for s in p.steps:
        for name in sorted(s.args):
            yield from walk_args(s.args[name], f"{s.id}.{name}")
    if p.answer is not None:
        yield from walk_args(p.answer, "answer")


# ---------------------------------------------------------------------------------------------------------- library

class LibraryError(ValueError):
    pass


def _nums(args: Sequence[Any]) -> list[float | int]:
    xs = list(args[0]) if len(args) == 1 and isinstance(args[0], list) else list(args)
    for x in xs:
        if not is_number(x):
            raise LibraryError(f"not a number: {x!r}")
    return xs


def _get(obj: Any, path: Any) -> Any:
    segs = path if isinstance(path, list) else ([path] if isinstance(path, int) else [p for p in str(path).split(".") if p != ""])
    for seg in segs:
        obj = _step_into(obj, seg)
    return obj


def _cmp(op: str, x: Any, v: Any) -> bool:
    if op == "eq":
        return _loose_eq(x, v)
    if op == "ne":
        return not _loose_eq(x, v)
    if op == "in":
        return any(_loose_eq(x, y) for y in v)
    if not (is_number(x) and is_number(v)):
        return False
    return {"lt": x < v, "le": x <= v, "gt": x > v, "ge": x >= v}[op]


def _select(lst: Any, where: Any) -> list[Any]:
    if not isinstance(lst, list) or not isinstance(where, dict):
        raise LibraryError("select(list, {field: value | {op, v}})")
    out = []
    for x in lst:
        if not isinstance(x, dict):
            continue
        ok = True
        for f, cond in where.items():
            got = x.get(f)
            if isinstance(cond, dict) and set(cond) == {"op", "v"}:
                ok = _cmp(cond["op"], got, cond["v"])
            else:
                ok = _loose_eq(got, cond)
            if not ok:
                break
        if ok:
            out.append(x)
    return out


def _concat(*xs: Any) -> Any:
    if all(isinstance(x, list) for x in xs):
        return [y for x in xs for y in x]
    return "".join(str(x) for x in xs)


def _div(a: Any, b: Any) -> Any:
    a, b = _nums([a, b])
    if b == 0:
        raise LibraryError("division by zero")
    return a / b


LIBRARY: dict[str, Callable[..., Any]] = {
    "sum": lambda *a: sum(_nums(a)),
    "min": lambda *a: min(_nums(a)),
    "max": lambda *a: max(_nums(a)),
    "round": lambda x, nd=0: round(_nums([x])[0], int(nd)),
    "len": lambda x: len(x),
    "concat": _concat,
    "select": _select,
    "get": _get,
    "pluck": lambda lst, f: [x.get(f) if isinstance(x, dict) else None for x in lst],
    "add": lambda a, b: _nums([a])[0] + _nums([b])[0],
    "sub": lambda a, b: _nums([a])[0] - _nums([b])[0],
    "mul": lambda a, b: _nums([a])[0] * _nums([b])[0],
    "div": _div,
}
# Library argument positions that name schema (a path, a field, a digit count) or a filter whose keys are fields: their
# literals are not task information, so taint records them as structural (a filter's values are still sourced).
STRUCTURAL_ARGS: dict[str, dict[int, str]] = {"get": {1: "schema"}, "pluck": {1: "schema"}, "round": {1: "schema"}, "select": {1: "where"}}


# ---------------------------------------------------------------------------------------------------------- matching

def _loose_eq(a: Any, b: Any) -> bool:
    """Literal-sourcing equality: numbers numerically, strings case/whitespace-normalised, else ``same_value``."""
    if is_number(a) and is_number(b):
        return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    if isinstance(a, str) and isinstance(b, str):
        return norm_text(a) == norm_text(b)
    return same_value(a, b)


_NUM_TOKEN = re.compile(r"(?<![\w.])[-+]?\$?(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?%?(?![\w])")


def number_tokens(text: str) -> Iterator[tuple[float, int, int]]:
    for m in _NUM_TOKEN.finditer(text):
        raw = m.group(0).replace("$", "").replace(",", "").rstrip("%")
        try:
            yield float(raw), m.start(), m.end()
        except ValueError:
            continue


def _as_number(v: Any) -> float | None:
    if is_number(v):
        return float(v)
    if isinstance(v, str) and re.fullmatch(r"\s*[-+]?\d+(\.\d+)?\s*", v):
        return float(v)
    return None


def text_spans(v: Any, text: str, *, min_len: int = 1) -> list[tuple[int, int]]:
    """Where literal ``v`` occurs in ``text`` (normalised token match; numbers numerically)."""
    spans: list[tuple[int, int]] = []
    num = _as_number(v)
    if num is not None:
        spans += [(a, b) for x, a, b in number_tokens(text) if math.isclose(x, num, rel_tol=1e-9, abs_tol=1e-12)]
    if isinstance(v, str) and len(v.strip()) >= min_len and num is None:
        parts = norm_text(v, fold_case=False).split(" ")
        body = r"\s+".join(re.escape(p) for p in parts)
        pre = r"(?<!\w)" if re.match(r"\w", parts[0]) else ""
        post = r"(?!\w)" if re.search(r"\w$", parts[-1]) else ""
        spans += [(m.start(), m.end()) for m in re.finditer(pre + body + post, text, re.IGNORECASE)]
    return sorted(set(spans))


def leaves(v: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    if isinstance(v, dict):
        for k in sorted(v, key=str):
            yield from leaves(v[k], f"{path}.{k}" if path else str(k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from leaves(x, f"{path}.{i}" if path else str(i))
    else:
        yield path, v


def value_paths(v: Any, value: Any) -> list[str]:
    """Paths of ``value`` in an observation: equal leaves, or (strings of 3+ characters, numbers) tokens in text leaves."""
    out = []
    for p, leaf in leaves(value):
        if isinstance(leaf, bool) or leaf is None:
            continue
        if _loose_eq(v, leaf) or (isinstance(v, str) and _as_number(v) is not None and is_number(leaf) and _loose_eq(_as_number(v), leaf)) or isinstance(leaf, str) and (not isinstance(v, str) or len(v.strip()) >= 3) and text_spans(v, leaf, min_len=3):
            out.append(p)
    return out


def is_structural(v: Any) -> bool:
    return v is None or isinstance(v, bool) or (isinstance(v, str) and not v.strip())


def scalar_leaves(v: Any, loc: str) -> Iterator[tuple[str, Any]]:
    if isinstance(v, dict):
        for k in sorted(v, key=str):
            yield from scalar_leaves(v[k], f"{loc}.{k}")
    elif isinstance(v, list):
        for i, x in enumerate(v):
            yield from scalar_leaves(x, f"{loc}.{i}")
    else:
        yield loc, v


# ---------------------------------------------------------------------------------------------------------- the run

class TaintSource(CrucibleModel):
    kind: Literal["instruction", "observation", "given", "tool_spec", "structural"]
    span: list[int] | None = None  # instruction: [start, end)
    step: str | None = None  # observation: the step whose observation holds it
    path: str | None = None  # observation: path in it; tool_spec: the tool
    given: int | None = None  # given: index in the intent card's list


class TaintEntry(CrucibleModel):
    loc: str
    value: Any = None
    sources: list[TaintSource] = Field(default_factory=list)

    @property
    def unsourced(self) -> bool:
        return not self.sources

    def has(self, kind: str) -> bool:
        return any(s.kind == kind for s in self.sources)


class CallRecord(CrucibleModel):
    step: str
    tool: str
    args: dict[str, Any] = Field(default_factory=dict)
    obs: Obs
    read_only: bool = False
    effects: list[list[str]] | None = None  # [app, coll, key] of every entity the call changed (write calls)


class Run(CrucibleModel):
    """One executed episode: what ``grade`` reads (``start``, ``end``, ``calls``, ``answer``) plus the taint map."""

    instruction: str = ""
    seed: int = 0
    calls: list[CallRecord] = Field(default_factory=list)
    answer: Any = None
    answer_text: str | None = None
    start: CanonState | None = None
    end: CanonState | None = None
    taint: list[TaintEntry] = Field(default_factory=list)
    anchors: dict[str, list[list[Any]]] = Field(default_factory=dict)
    read_only: dict[str, bool] = Field(default_factory=dict)  # tool -> read_only flag as the World declared it
    error: str | None = None

    def call(self, sid: str) -> CallRecord | None:
        return next((c for c in self.calls if c.step == sid), None)

    def values(self) -> list[Any]:
        """Every value π* produced or saw: resolved arguments, observation leaves and the answer (TAINT's "π* values")."""
        out: list[Any] = []
        for c in self.calls:
            out += [v for _, v in scalar_leaves(c.args, "")]
            out += [v for _, v in leaves(c.obs.value)]
        if self.answer is not None:
            out += [v for _, v in scalar_leaves(self.answer, "")]
        return out


class RefError(LookupError):
    pass


def _step_into(obj: Any, seg: Any) -> Any:
    if isinstance(obj, dict):
        if str(seg) in obj:
            return obj[str(seg)]
        raise RefError(f"no key {seg!r}")
    if isinstance(obj, list):
        try:
            i = int(seg)
        except (TypeError, ValueError):
            raise RefError(f"list index {seg!r} is not an integer") from None
        if -len(obj) <= i < len(obj):
            return obj[i]
        raise RefError(f"index {i} out of range")
    raise RefError(f"cannot index {type(obj).__name__} with {seg!r}")


class _Interp:
    def __init__(self, instruction: str, given: Sequence[Any], tools: list[ToolInfo], anchors: dict[str, list[list[Any]]] | None):
        self.instruction = instruction
        self.given = list(given)
        self.tools = tools
        self.obs: dict[str, Obs] = {}
        self.order: list[str] = []
        self.taint: list[TaintEntry] = []
        self.anchors_in = anchors
        self.anchors_out: dict[str, list[list[Any]]] = {}

    def resolve_ref(self, r: Ref, loc: str) -> Any:
        head, *segs = r.path.split(".")
        if head not in self.obs:
            raise RefError(f"{r.path}: no step {head!r} ran before")
        o = self.obs[head]
        if o.error is not None:
            raise RefError(f"{r.path}: step {head!r} errored")
        cur, rec = o.value, []
        want = {int(d): h for d, h in (self.anchors_in or {}).get(loc, [])} if self.anchors_in is not None else {}
        for depth, seg in enumerate(segs):
            if isinstance(cur, list) and depth in want:
                j = next((i for i, x in enumerate(cur) if content_hash(x) == want[depth]), None)
                if j is None:
                    raise RefError(f"{r.path}: the anchored element is gone")
                seg = j
            nxt = _step_into(cur, seg)
            if isinstance(cur, list):
                rec.append([depth, content_hash(nxt)])
            cur = nxt
        if rec:
            self.anchors_out[loc] = rec
        return cur

    def source(self, loc: str, v: Any, upto: int) -> None:
        for leaf_loc, x in scalar_leaves(v, loc):
            e = TaintEntry(loc=leaf_loc, value=x)
            if is_structural(x):
                e.sources = [TaintSource(kind="structural")]
                self.taint.append(e)
                continue
            src: list[TaintSource] = [TaintSource(kind="instruction", span=[a, b]) for a, b in text_spans(x, self.instruction)]
            src += [TaintSource(kind="given", given=i) for i, g in enumerate(self.given)
                    if _loose_eq(x, g) or (isinstance(g, str) and text_spans(x, g, min_len=3))]
            for t in self.tools:
                hit = bool(text_spans(x, t.description, min_len=3)) if t.description else False
                for p in (t.input_schema.get("properties") or {}).values():
                    if isinstance(p, dict):
                        hit = hit or any(_loose_eq(x, e2) for e2 in p.get("enum", [])) or ("default" in p and _loose_eq(x, p["default"]))
                if hit:
                    src.append(TaintSource(kind="tool_spec", path=t.name))
            n_obs = 0
            for sid in self.order[:upto]:
                o = self.obs[sid]
                for p in value_paths(x, o.value)[:1]:
                    src.append(TaintSource(kind="observation", step=sid, path=p))
                    n_obs += 1
                if n_obs >= 3:
                    break
            e.sources = src
            self.taint.append(e)

    def eval(self, a: Arg, loc: str, upto: int, record: bool) -> Any:
        if isinstance(a, Lit):
            if record:
                self.source(loc, a.v, upto)
            return a.v
        if isinstance(a, Ref):
            return self.resolve_ref(a, loc)
        f = LIBRARY.get(a.name)
        if f is None:
            raise LibraryError(f"unknown function {a.name!r}")
        vals = []
        for i, x in enumerate(a.args):
            sub = f"{loc}/fn:{a.name}[{i}]"
            role = STRUCTURAL_ARGS.get(a.name, {}).get(i)
            if isinstance(x, Lit) and role == "schema":  # a path, a field name, a digit count: schema, not task information
                if record:
                    self.taint.append(TaintEntry(loc=sub, value=x.v, sources=[TaintSource(kind="structural")]))
                vals.append(x.v)
            elif isinstance(x, Lit) and role == "where" and isinstance(x.v, dict):  # field names are schema; the values are taint
                if record:
                    for k in sorted(x.v, key=str):
                        cond = x.v[k]
                        self.source(f"{sub}.{k}", cond["v"] if isinstance(cond, dict) and set(cond) == {"op", "v"} else cond, upto)
                vals.append(x.v)
            else:
                vals.append(self.eval(x, sub, upto, record))
        try:
            return f(*vals)
        except (LibraryError, RefError):
            raise
        except (TypeError, ValueError, KeyError, IndexError, AttributeError) as e:
            raise LibraryError(f"{a.name}: {type(e).__name__}") from e


def _entities(changes: Iterable[Any]) -> list[list[str]]:
    return sorted({(c.addr.app, c.addr.coll, c.addr.key) for c in changes})  # type: ignore[return-value]


async def run(world: SynthWorld, program: Program, instruction: str = "", *, seed: int = 0, given: Sequence[Any] = (), reset: bool = True,
              prepare: Callable[[SynthWorld], Awaitable[None]] | None = None, anchors: dict[str, list[list[Any]]] | None = None,
              track_effects: bool = True, taint: bool = True) -> Run:
    """Execute ``program`` (see the module docstring). ``prepare`` edits the start state after the reset (PERT);
    ``anchors`` re-resolves list refs to the same elements (a previous run's ``Run.anchors``)."""
    if reset:
        await world.reset(seed)
    if prepare is not None:
        await prepare(world)
    tools = await world.tools()
    ro = {t.name: t.read_only for t in tools}
    start = await world.canon()
    ip = _Interp(instruction, given, tools, anchors)
    out = Run(instruction=instruction, seed=seed, start=start, read_only=ro)
    last = start
    for k, s in enumerate(program.steps):
        try:
            args = {name: ip.eval(s.args[name], f"{s.id}.{name}", k, taint) for name in sorted(s.args)}
        except (RefError, LibraryError) as e:
            out.error = f"step {s.id}: {type(e).__name__}: {e}"
            break
        try:
            obs = await world.call(s.tool, args)
        except Exception as e:  # noqa: BLE001 - a World exception ends the episode; the state reached is graded
            out.error = f"step {s.id}: world {type(e).__name__}"
            break
        rec = CallRecord(step=s.id, tool=s.tool, args=args, obs=obs, read_only=ro.get(s.tool, False))
        if track_effects and not rec.read_only and last is not None:
            now = await world.canon()
            rec.effects = _entities(diff(last, now)) if now is not None else None
            last = now
        out.calls.append(rec)
        ip.obs[s.id] = obs
        ip.order.append(s.id)
    if out.error is None and program.answer is not None:
        try:
            out.answer = ip.eval(program.answer, "answer", len(ip.order), taint)
            if program.answer_template:
                out.answer_text = program.answer_template.format(out.answer)
        except (RefError, LibraryError, IndexError, KeyError, ValueError) as e:
            out.error = f"answer: {type(e).__name__}: {e}"
            out.answer = None
    out.end = await world.canon()
    out.taint = ip.taint
    out.anchors = ip.anchors_out
    return out


def program_json(p: Program) -> str:
    return cjson(p.model_dump(mode="json"))
