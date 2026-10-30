"""The domain-generic World interface of ``crucible.synth`` (part2-paper-core §3.1) and two implementations.

``SynthWorld`` is the six operations every falsifier needs, async like the rest of Crucible's World code:

| op | returns | notes |
|---|---|---|
| ``reset(seed)`` | a state token | deterministic from the seed; REP checks it |
| ``tools()`` | ``list[ToolInfo]`` | qualified name, JSON schema, ``read_only``, ``writes`` (collections it may write; None = unknown) |
| ``call(tool, args)`` | ``Obs`` | a JSON value, or ``error`` set; a tool error is an observation, never an exception |
| ``snapshot()`` / ``restore(ref)`` | | exact |
| ``canon()`` | ``CanonState`` or None | None = black-box (observation-only contracts) |

A World that can also edit its start state implements ``perturb(Perturbation)`` (``Perturbable``): the PERT falsifier's
distractor clones, row shuffles and text edits. A World without it simply gets no PERT trials, and its certificate says so.

- ``PyWorld``: an in-process World of Apps over dict tables, for tests and synthesized Apps. An App is tables plus plain
  Python tool functions, each declaring ``read_only`` or the collections it ``writes``. Calls are atomic (a tool that
  raises leaves no partial write) and the declaration is enforced: a tool that changes a collection it did not declare
  raises ``UndeclaredWrite`` (a World defect, not an observation), which is how the risk "canonical state misses
  reward-bearing state" (§16) is held down for synthesized Apps. Randomness and time come from the context (a seeded
  ``rng``, a virtual clock that advances one second per call), so the same seed gives the same episode.
- ``CrucibleWorldAdapter``: wraps a started ``crucible.world.world.World``. Calls go through the public policy path
  (``World.step(actor, ToolCall)``, i.e. ``_call_tool`` with its ACL, schema checks, deadline and bounding); snapshot and
  restore are the World's own; ``canon()`` reads every App through the verifier view (the policy never sees it): a
  ``sql`` App by its tables (primary keys from ``pragma_table_info``), an ``fs``/``fs_git`` App by its manifest, a
  ``json_doc`` App by its document. Any other adapter kind is listed as ``unobserved``.
"""

from __future__ import annotations

import copy
import inspect
import random
import re
from collections.abc import Awaitable, Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import CanonState, canon_fs_manifest, canon_json, canon_sql_tables, canon_tables, content_hash

# The read verbs ``accept/mutation/actions.py`` uses (copied: synth sits below accept). A tool whose spec does not say
# ``read_only`` but whose name starts with one of these is treated as a read.
READ_VERB = re.compile(r"^(get|list|read|search|find|view|show|lookup|describe|query|fetch|count)(_|$)", re.IGNORECASE)


class ToolInfo(CrucibleModel):
    name: str  # qualified: "<app>.<tool>"
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    read_only: bool = False
    writes: list[str] | None = None  # "<app>.<coll>" the tool may write; None = not declared


class Obs(CrucibleModel):
    value: Any = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


class Perturbation(CrucibleModel):
    """One edit of a start state (PERT). Keys are canonical key strings (``CanonState`` keys).
    - ``clone``: copy entity ``key`` to ``new_key`` with ``fields`` overridden, inserted at ``position`` (None = append);
    - ``shuffle``: permute the rows of ``(app, coll)`` with ``seed``;
    - ``set``: overwrite ``fields`` of entity ``key``."""

    op: Literal["clone", "shuffle", "set"]
    app: str
    coll: str
    key: str | None = None
    new_key: str | None = None
    fields: dict[str, Any] = Field(default_factory=dict)
    position: int | None = None
    seed: int = 0


@runtime_checkable
class SynthWorld(Protocol):
    async def reset(self, seed: int) -> Any: ...

    async def tools(self) -> list[ToolInfo]: ...

    async def call(self, tool: str, args: dict[str, Any]) -> Obs: ...

    async def snapshot(self) -> Any: ...

    async def restore(self, ref: Any) -> None: ...

    async def canon(self) -> CanonState | None: ...


@runtime_checkable
class Perturbable(Protocol):
    async def perturb(self, p: Perturbation) -> None: ...


WorldFactory = Callable[[], SynthWorld | Awaitable[SynthWorld]]


# ---------------------------------------------------------------------------------------------------------- PyWorld

class ToolError(Exception):
    """A tool refusing a call (bad argument, wrong state): becomes an ``Obs`` with ``error``."""


class UndeclaredWrite(RuntimeError):
    """A tool changed a collection it did not declare (or a read-only tool changed anything): a World defect."""


_JSON_TYPES = {int: "integer", float: "number", str: "string", bool: "boolean", list: "array", dict: "object"}


@dataclass
class ToolDef:
    name: str
    fn: Callable[..., Any]
    read_only: bool
    writes: tuple[str, ...]
    description: str
    input_schema: dict[str, Any]


@dataclass
class ToolCtx:
    """What a PyWorld tool sees: its App's tables, every App's tables (cross-App tools), a seeded rng and a clock."""

    app: str
    world: dict[str, dict[str, dict[Any, dict[str, Any]]]]
    rng: random.Random
    step: int
    t0: datetime

    @property
    def tables(self) -> dict[str, dict[Any, dict[str, Any]]]:
        return self.world[self.app]

    def now(self) -> str:
        return (self.t0 + timedelta(seconds=self.step)).isoformat().replace("+00:00", "Z")


def _schema_of(fn: Callable[..., Any]) -> dict[str, Any]:
    props: dict[str, Any] = {}
    required: list[str] = []
    params = list(inspect.signature(fn).parameters.values())[1:]  # the first parameter is the ToolCtx
    hints = getattr(fn, "__annotations__", {})
    for p in params:
        t = hints.get(p.name)
        prop: dict[str, Any] = {}
        if isinstance(t, str):
            t = {"int": int, "float": float, "str": str, "bool": bool, "list": list, "dict": dict}.get(t.split("|")[0].strip())
        if t in _JSON_TYPES:
            prop["type"] = _JSON_TYPES[t]
        if p.default is inspect.Parameter.empty:
            required.append(p.name)
        else:
            prop["default"] = p.default
        props[p.name] = prop
    return {"type": "object", "properties": props, "required": required}


class App:
    """Tables plus tools. ``positional`` collections are canonicalized by row position (weak on purpose: fixtures that
    must be caught by PERT); ``hidden`` collections are World-internal and not in ``canon()``."""

    def __init__(self, name: str, tables: Mapping[str, Mapping[Any, Mapping[str, Any]]] | None = None, *,
                 positional: Iterable[str] = (), hidden: Iterable[str] = ()):
        self.name = name
        self.initial = {c: {k: dict(r) for k, r in rows.items()} for c, rows in (tables or {}).items()}
        self.positional = frozenset(positional)
        self.hidden = frozenset(hidden)
        self.tools: dict[str, ToolDef] = {}

    def tool(self, fn: Callable[..., Any] | None = None, *, read_only: bool = False, writes: Iterable[str] = (), name: str | None = None,
             description: str | None = None) -> Any:
        """Register a tool: ``@app.tool(read_only=True)`` or ``@app.tool(writes=["invoices", "audit.log"])``. A write
        collection without a dot is this App's."""

        def deco(f: Callable[..., Any]) -> Callable[..., Any]:
            w = tuple(x if "." in x else f"{self.name}.{x}" for x in writes)
            if read_only and w:
                raise ValueError(f"tool {f.__name__}: read_only tools declare no writes")
            if not read_only and not w:
                raise ValueError(f"tool {f.__name__}: a write tool must declare the collections it writes")
            n = name or f.__name__
            self.tools[n] = ToolDef(n, f, read_only, w, description or inspect.getdoc(f) or "", _schema_of(f))
            return f

        return deco(fn) if fn is not None else deco


def _coll_digest(rows: Mapping[Any, Any]) -> str:
    return content_hash([[repr(k), r] for k, r in rows.items()], 32)  # order-sensitive: a reorder is a write too


class PyWorld:
    """An in-process ``SynthWorld`` (see the module docstring). ``seed_fn(seed)`` returns ``{app: {coll: {key: row}}}``
    for a seed; without one, every seed starts from the Apps' own tables."""

    def __init__(self, apps: Iterable[App], *, seed_fn: Callable[[int], Mapping[str, Any]] | None = None, strict_writes: bool = True,
                 t0: datetime | None = None):
        self.apps = {a.name: a for a in apps}
        self.seed_fn = seed_fn
        self.strict_writes = strict_writes
        self.t0 = t0 or datetime(2026, 1, 1, tzinfo=UTC)
        self.tables: dict[str, dict[str, dict[Any, dict[str, Any]]]] = {}
        self.seed: int | None = None
        self.step = 0
        self.rng = random.Random(0)
        self._snaps: dict[str, tuple[Any, int, Any]] = {}

    # -- SynthWorld ------------------------------------------------------------------------------------------------
    async def reset(self, seed: int) -> str:
        base = self.seed_fn(seed) if self.seed_fn is not None else {n: a.initial for n, a in self.apps.items()}
        self.tables = {n: {c: {k: dict(r) for k, r in rows.items()} for c, rows in base.get(n, {}).items()} for n in self.apps}
        self.seed, self.step = seed, 0
        self.rng = random.Random(f"pyworld/{seed}")
        return self._fingerprint()

    async def tools(self) -> list[ToolInfo]:
        out = []
        for an in sorted(self.apps):
            for tn, t in sorted(self.apps[an].tools.items()):
                out.append(ToolInfo(name=f"{an}.{tn}", description=t.description, input_schema=t.input_schema, read_only=t.read_only,
                                    writes=list(t.writes)))
        return out

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        app, _, name = tool.partition(".")
        t = self.apps.get(app).tools.get(name) if app in self.apps else None
        if t is None:
            return Obs(error=f"unknown tool {tool!r}")
        if not isinstance(args, dict):
            return Obs(error="arguments must be an object")
        try:
            inspect.signature(t.fn).bind(None, **args)
        except TypeError as e:
            return Obs(error=f"bad arguments: {e}")
        self.step += 1
        ctx = ToolCtx(app, self.tables, self.rng, self.step, self.t0)
        before = self._digests()
        saved = copy.deepcopy(self.tables)
        rng_state = self.rng.getstate()
        try:
            value = t.fn(ctx, **args)
            value = _jsonable(value)
        except Exception as e:  # noqa: BLE001 - any tool failure is the caller's observation, and the call is undone
            self.tables = saved
            self.rng.setstate(rng_state)
            return Obs(error=f"{type(e).__name__}: {e}")
        if self.strict_writes:
            changed = {k for k, d in self._digests().items() if before.get(k) != d}
            undeclared = sorted(changed - set(t.writes))
            if undeclared:
                self.tables = saved
                raise UndeclaredWrite(f"tool {tool} changed {undeclared} (declared {list(t.writes)})")
        return Obs(value=value)

    async def snapshot(self) -> str:
        ref = f"snap-{len(self._snaps)}-{self._fingerprint()[:12]}"
        self._snaps[ref] = (copy.deepcopy(self.tables), self.step, self.rng.getstate())
        return ref

    async def restore(self, ref: Any) -> None:
        tables, step, rng = self._snaps[ref]
        self.tables, self.step = copy.deepcopy(tables), step
        self.rng.setstate(rng)

    async def canon(self) -> CanonState:
        out = CanonState()
        for n in sorted(self.apps):
            a = self.apps[n]
            visible = {c: rows for c, rows in self.tables.get(n, {}).items() if c not in a.hidden}
            out = out.merged(canon_tables(n, visible, positional=a.positional))
        return out

    # -- Perturbable -----------------------------------------------------------------------------------------------
    async def perturb(self, p: Perturbation) -> None:
        rows = self.tables[p.app][p.coll]
        keys = list(rows)
        if p.op == "shuffle":
            random.Random(f"perturb/{p.seed}").shuffle(keys)
            self.tables[p.app][p.coll] = {k: rows[k] for k in keys}
            return
        k = self._key(p.app, p.coll, p.key)
        if p.op == "set":
            rows[k].update(copy.deepcopy(p.fields))
            return
        if p.new_key is None:
            raise ValueError("clone needs new_key")
        nk: Any = int(p.new_key) if all(isinstance(x, int) for x in keys) and p.new_key.lstrip("-").isdigit() else p.new_key
        if nk in rows:
            raise ValueError(f"clone target {nk!r} exists")
        row = {f: (nk if v == k and type(v) is type(k) else copy.deepcopy(v)) for f, v in rows[k].items()}
        row.update(copy.deepcopy(p.fields))
        pos = len(keys) if p.position is None else max(0, min(p.position, len(keys)))
        items = list(rows.items())
        items.insert(pos, (nk, row))
        self.tables[p.app][p.coll] = dict(items)

    # -- helpers ---------------------------------------------------------------------------------------------------
    def _key(self, app: str, coll: str, key: str | None) -> Any:
        rows = self.tables[app][coll]
        if coll in self.apps[app].positional:
            return list(rows)[int(key or 0)]
        for k in rows:
            if str(k) == key:
                return k
        raise KeyError(f"{app}.{coll} has no key {key!r}")

    def _digests(self) -> dict[str, str]:
        return {f"{a}.{c}": _coll_digest(rows) for a, colls in self.tables.items() for c, rows in colls.items()}

    def _fingerprint(self) -> str:
        return content_hash([[a, c, [[repr(k), r] for k, r in rows.items()]] for a, colls in sorted(self.tables.items())
                             for c, rows in sorted(colls.items())], 64)


def _jsonable(v: Any) -> Any:
    """A tool's return value as JSON (tuples to lists, keys to strings, anything else as its text)."""
    import json

    return json.loads(json.dumps(v, default=str, allow_nan=False)) if v is not None else None


# ---------------------------------------------------------------------------------------------------------- Crucible World

class CanonError(RuntimeError):
    """A canonical dump could not be taken completely (a truncated query, a failed adapter op): fail closed."""


SQL_SCHEMA = ("SELECT m.name, p.name, p.pk FROM sqlite_master m, pragma_table_info(m.name) p "
              "WHERE m.type = 'table' AND m.name NOT LIKE 'sqlite_%' ORDER BY m.name, p.cid")
MAX_ROWS = 1_000_000


@dataclass
class CrucibleWorldAdapter:
    """``SynthWorld`` over a started Crucible ``World`` (see the module docstring). ``factory(seed)`` rebuilds the World
    for another seed (a Crucible World's seed is part of its spec); without one, ``reset`` of a different seed is
    refused. ``read_only`` globs mark reads the tool specs do not mark (the read-verb heuristic applies otherwise)."""

    world: Any
    actor: str | None = None
    factory: Callable[[int], Awaitable[Any]] | None = None
    seed: int | None = None
    read_only: tuple[str, ...] = ()
    natural_keys: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # app -> table -> columns
    _n: int = 0
    _pk: dict[str, dict[str, list[str]]] = field(default_factory=dict)  # app -> table -> pk columns (schema is fixed per World)

    def __post_init__(self) -> None:
        if self.actor is None:
            pol = [a.id for a in self.world.spec.actors if a.kind == "policy"]
            if not pol:
                raise ValueError("the World has no policy actor to call tools as")
            self.actor = pol[0]

    async def reset(self, seed: int) -> Any:
        if self.seed is not None and seed != self.seed:
            if self.factory is None:
                raise ValueError(f"this World was built for seed {self.seed}; pass a factory to reset to seed {seed}")
            await self.world.close()
            self.world, self._pk = await self.factory(seed), {}
        self.seed = seed
        await self.world.reset()
        self._n = 0
        return self.world.initial.fingerprint if self.world.initial is not None else None

    async def tools(self) -> list[ToolInfo]:
        import fnmatch

        out = []
        for t in self.world.tools_for(self.actor):
            ro = t.read_only or bool(READ_VERB.match(t.name)) or any(fnmatch.fnmatchcase(t.qualified, p) for p in self.read_only)
            out.append(ToolInfo(name=t.qualified, description=t.description, input_schema=t.input_schema, read_only=ro))
        return sorted(out, key=lambda t: t.name)

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        import json

        from crucible.core.schema.actions import ToolCall

        self._n += 1
        res = await self.world.step(self.actor, ToolCall(tool_call_id=f"synth-{self._n}", function_name=tool, arguments=dict(args)))
        results = res.observation.tool_results
        if not results:
            return Obs(error="no tool result")
        tr = results[0]
        if not tr.ok:
            return Obs(error=f"{tr.error or 'error'}: {tr.content}"[:2000])
        if tr.data is not None:
            return Obs(value=tr.data)
        try:
            return Obs(value=json.loads(tr.content))
        except (ValueError, TypeError):
            return Obs(value=tr.content)

    async def snapshot(self) -> Any:
        return await self.world.snapshot()

    async def restore(self, ref: Any) -> None:
        await self.world.restore(ref)

    async def canon(self) -> CanonState:
        from crucible.core.schema.state import Query

        view = self.world.verifier_view()
        out = CanonState()
        unobserved: list[str] = []
        for app_id in sorted(self.world.apps):
            adapter = view.adapter(app_id)
            kind = getattr(adapter, "kind", None)
            if kind == "sql":
                out = out.merged(await self._canon_sql(view, app_id, Query))
            elif kind in ("fs", "fs_git"):
                r = await view.query(app_id, Query(op="manifest", args={}))
                if not r.ok:
                    raise CanonError(f"{app_id}: manifest failed")
                out = out.merged(canon_fs_manifest(app_id, r.value["entries"]))
            elif kind == "json_doc":
                r = await view.query(app_id, Query(op="doc", args={}))
                if not r.ok:
                    raise CanonError(f"{app_id}: doc failed")
                out = out.merged(canon_json(app_id, "doc", r.value))
            else:
                unobserved.append(app_id)
        return CanonState.build(out.apps, [*out.unobserved, *unobserved])

    async def _canon_sql(self, view: Any, app_id: str, Query: Any) -> CanonState:
        async def q(sql: str) -> dict[str, Any]:
            r = await view.query(app_id, Query(op="sql", args={"sql": sql, "max_rows": MAX_ROWS}))
            if not r.ok or not isinstance(r.value, dict):
                raise CanonError(f"{app_id}: query failed")
            if r.value.get("truncated"):
                raise CanonError(f"{app_id}: a table has more than {MAX_ROWS} rows")
            return r.value

        if app_id not in self._pk:
            cols: dict[str, list[tuple[int, str]]] = {}
            for t, c, pk in (await q(SQL_SCHEMA))["rows"]:
                cols.setdefault(t, [])
                if pk:
                    cols[t].append((pk, c))
            self._pk[app_id] = {t: [c for _, c in sorted(v)] for t, v in cols.items()}
        tables = {}
        for t, pk in self._pk[app_id].items():
            r = await q(f'SELECT * FROM "{t}"')
            tables[t] = {"columns": r["columns"], "pk": pk, "rows": r["rows"]}
        return canon_sql_tables(app_id, tables, self.natural_keys.get(app_id))
