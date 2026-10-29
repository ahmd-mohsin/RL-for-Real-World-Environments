"""Canonical typed state and the exact diff (part2-paper-core §3.2): the one view of a World that rewards may read.

A ``CanonState`` is a finite map from an **address** ``(app, collection, key, field)`` to a JSON value. The key names an
entity, never a position: a SQL row by its primary key (else a declared natural key, else the hash of the row), a file by
its path, a message by its id, a JSON entity by its pointer prefix. Only the Gym canonicalizer (an observation index) and
JSON lists of objects without a declared key field are positional, and both say so: a contract keyed on a position is
exactly what the PERT falsifier exists to catch.

Stored as ``apps[app][coll][key] = {field: value}``, so a state is plain JSON, hashes stably and validates cheaply.

``diff(s, s2, normalizers)`` is the set of ``Change``s, deterministic and exact:
- an **update** is per field (``addr.field`` is the field, ``old``/``new`` are the two values), so "only ``status``
  changed" is expressible and two fields of one row are two pieces of evidence;
- an **insert** or **delete** is one change per entity (``addr.field == ""``; ``new``/``old`` is the whole row), so a
  pattern counts rows, not cells.

Normalizers (declared per App or per contract) are applied to both states before diffing, in this order: ``ignore_fields``,
``float_round``, ``text_norm``, ``order_insensitive``; then ``id_canon`` re-keys the *inserted* entities of a collection by
the hash of their content (their id fields dropped), so an autoincrement id an agent happened to get is not a different
effect. Deleted and updated entities keep their keys: they existed at the start and their identity is meaningful.

Value equality (``same_value``) is type-aware: numbers compare numerically (``1 == 1.0``), a bool is never a number
(``True != 1``), lists and objects compare element-wise. Everything here is pure and deterministic.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import math
import os
import re
import sqlite3
import stat
import unicodedata
from collections.abc import Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from pydantic import ConfigDict, Field

from crucible.core.schema.base import CrucibleModel


class _Missing:
    """An absent cell (distinct from a JSON ``null``). A singleton that survives copy and pickle."""

    _it: _Missing | None = None

    def __new__(cls) -> _Missing:  # noqa: PYI034 - a singleton
        if cls._it is None:
            cls._it = super().__new__(cls)
        return cls._it

    def __repr__(self) -> str:
        return "MISSING"

    def __bool__(self) -> bool:
        return False

    def __reduce__(self) -> str:
        return "MISSING"


MISSING: Any = _Missing()


# ---------------------------------------------------------------------------------------------------------- values

def is_number(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def same_value(a: Any, b: Any) -> bool:
    """Type-aware equality: numbers numerically, bools only with bools, containers element-wise, else ``==``."""
    if is_number(a) and is_number(b):
        return a == b
    if isinstance(a, bool) or isinstance(b, bool):
        return isinstance(a, bool) and isinstance(b, bool) and a == b
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(same_value(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(same_value(a[k], b[k]) for k in a)
    if a is MISSING or b is MISSING:
        return a is b
    return type(a) is type(b) and a == b


def cjson(v: Any) -> str:
    """Compact sorted JSON of a JSON-shaped value; numbers that are whole floats are written as ints so ``1.0`` and ``1``
    hash alike (they compare equal under ``same_value``)."""

    def fold(x: Any) -> Any:
        if isinstance(x, float) and x.is_integer() and abs(x) < 2**53:
            return int(x)
        if isinstance(x, dict):
            return {str(k): fold(y) for k, y in x.items()}
        if isinstance(x, list | tuple):
            return [fold(y) for y in x]
        return x

    return json.dumps(fold(v), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False, default=str)


def content_hash(v: Any, n: int = 16) -> str:
    return hashlib.sha256(cjson(v).encode()).hexdigest()[:n]


def key_str(values: Sequence[Any]) -> str:
    """An entity key as text: one int or str value as itself (``42``, ``PO-5000``), anything else as canonical JSON."""
    if len(values) == 1:
        v = values[0]
        if isinstance(v, str):
            return v
        if isinstance(v, int) and not isinstance(v, bool):
            return str(v)
        return cjson(v)
    return cjson(list(values))


# ---------------------------------------------------------------------------------------------------------- addresses

class Address(CrucibleModel):
    """``(app, coll, key, field)``. ``field == ""`` names the whole entity (insert and delete changes)."""

    model_config = ConfigDict(**{**CrucibleModel.model_config, "frozen": True})

    app: str
    coll: str
    key: str
    field: str = ""

    def __str__(self) -> str:
        return f"{self.app}.{self.coll}[{self.key}]" + (f".{self.field}" if self.field else "")

    @property
    def entity(self) -> tuple[str, str, str]:
        return (self.app, self.coll, self.key)


class Change(CrucibleModel):
    addr: Address
    kind: Literal["insert", "update", "delete"]
    old: Any = None
    new: Any = None

    def sort_key(self) -> tuple[str, str, str, str, str]:
        return (self.addr.app, self.addr.coll, self.addr.key, self.addr.field, self.kind)

    def same(self, other: Change) -> bool:
        return self.addr == other.addr and self.kind == other.kind and same_value(self.old, other.old) and same_value(self.new, other.new)


class CanonState(CrucibleModel):
    """Every reward-bearing value of a World. ``unobserved`` lists what the canonicalizer could not see (an App whose
    adapter has no typed dump): a contract over it fails closed, and a certificate records the gap."""

    apps: dict[str, dict[str, dict[str, dict[str, Any]]]] = Field(default_factory=dict)
    unobserved: list[str] = Field(default_factory=list)

    @classmethod
    def build(cls, apps: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]], unobserved: Iterable[str] = ()) -> CanonState:
        """Build without re-validating every cell (callers pass plain JSON); the nested dicts are copied."""
        data = {a: {c: {k: dict(r) for k, r in rows.items()} for c, rows in colls.items()} for a, colls in apps.items()}
        return cls.model_construct(apps=data, unobserved=sorted(set(unobserved)))

    def entity(self, app: str, coll: str, key: str) -> dict[str, Any] | None:
        return self.apps.get(app, {}).get(coll, {}).get(key)

    def get(self, addr: Address) -> Any:
        row = self.entity(addr.app, addr.coll, addr.key)
        if row is None:
            return MISSING
        if not addr.field:
            return row
        return row.get(addr.field, MISSING)

    def entities(self) -> Iterator[tuple[str, str, str, dict[str, Any]]]:
        for app in sorted(self.apps):
            for coll in sorted(self.apps[app]):
                rows = self.apps[app][coll]
                for key in sorted(rows):
                    yield app, coll, key, rows[key]

    def cells(self) -> Iterator[tuple[Address, Any]]:
        for app, coll, key, row in self.entities():
            for f in sorted(row):
                yield Address(app=app, coll=coll, key=key, field=f), row[f]

    def collection(self, app: str, coll: str) -> dict[str, dict[str, Any]]:
        return self.apps.get(app, {}).get(coll, {})

    def merged(self, other: CanonState) -> CanonState:
        """Two Apps' states side by side (a World's canon is the union of its Apps')."""
        apps = {**self.apps, **other.apps}
        return CanonState.build(apps, [*self.unobserved, *other.unobserved])

    def fingerprint(self) -> str:
        return hashlib.sha256(cjson({"apps": self.apps, "unobserved": self.unobserved}).encode()).hexdigest()


# ---------------------------------------------------------------------------------------------------------- normalizers

class FieldScope(CrucibleModel):
    """Which cells a normalizer applies to: globs over app, collection and field (keys are never scoped: a normalizer
    describes a schema, not an entity)."""

    app: str = "*"
    coll: str = "*"
    field: str = "*"

    def covers(self, app: str, coll: str, field: str) -> bool:
        return fnmatch.fnmatchcase(app, self.app) and fnmatch.fnmatchcase(coll, self.coll) and fnmatch.fnmatchcase(field, self.field)


class FloatRound(FieldScope):
    digits: int = 6


class TextNorm(FieldScope):
    fold_case: bool = True


class IdCanon(CrucibleModel):
    """Inserted entities of ``(app, coll)`` are keyed by the hash of their content without ``id_fields``."""

    app: str = "*"
    coll: str
    id_fields: list[str] = Field(default_factory=lambda: ["id"])

    def covers(self, app: str, coll: str) -> bool:
        return fnmatch.fnmatchcase(app, self.app) and fnmatch.fnmatchcase(coll, self.coll)


def norm_text(s: str, fold_case: bool = True) -> str:
    t = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", s)).strip()
    return t.casefold() if fold_case else t


class Normalizers(CrucibleModel):
    ignore_fields: list[FieldScope] = Field(default_factory=list)
    id_canon: list[IdCanon] = Field(default_factory=list)
    float_round: list[FloatRound] = Field(default_factory=list)
    order_insensitive: list[FieldScope] = Field(default_factory=list)
    text_norm: list[TextNorm] = Field(default_factory=list)

    def merged(self, other: Normalizers | None) -> Normalizers:
        if other is None:
            return self
        return Normalizers(**{f: [*getattr(self, f), *getattr(other, f)] for f in type(self).model_fields})

    @property
    def empty(self) -> bool:
        return not any(getattr(self, f) for f in type(self).model_fields)

    def value(self, app: str, coll: str, field: str, v: Any) -> Any:
        """One cell under the value normalizers (rounding, text, list order). Returns ``MISSING`` for an ignored cell."""
        if any(s.covers(app, coll, field) for s in self.ignore_fields):
            return MISSING
        for r in self.float_round:
            if r.covers(app, coll, field):
                v = _round_all(v, r.digits)
        for t in self.text_norm:
            if t.covers(app, coll, field):
                v = _text_all(v, t.fold_case)
        if any(s.covers(app, coll, field) for s in self.order_insensitive) and isinstance(v, list):
            v = sorted(v, key=cjson)
        return v

    def apply(self, s: CanonState) -> CanonState:
        """The per-state normalizers (everything but ``id_canon``, which needs the start state: see ``diff``)."""
        if self.empty:
            return s
        out: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for app, colls in s.apps.items():
            for coll, rows in colls.items():
                dst = out.setdefault(app, {}).setdefault(coll, {})
                for key, row in rows.items():
                    nrow = {}
                    for f, v in row.items():
                        nv = self.value(app, coll, f, v)
                        if nv is not MISSING:
                            nrow[f] = nv
                    dst[key] = nrow
        return CanonState.build(out, s.unobserved)

    def id_canon_for(self, app: str, coll: str) -> IdCanon | None:
        return next((c for c in self.id_canon if c.covers(app, coll)), None)


def _round_all(v: Any, digits: int) -> Any:
    if isinstance(v, float):
        r = round(v, digits)
        return 0.0 if r == 0 else r  # -0.0 folded
    if isinstance(v, list):
        return [_round_all(x, digits) for x in v]
    if isinstance(v, dict):
        return {k: _round_all(x, digits) for k, x in v.items()}
    return v


def _text_all(v: Any, fold: bool) -> Any:
    if isinstance(v, str):
        return norm_text(v, fold)
    if isinstance(v, list):
        return [_text_all(x, fold) for x in v]
    if isinstance(v, dict):
        return {k: _text_all(x, fold) for k, x in v.items()}
    return v


# ---------------------------------------------------------------------------------------------------------- diff

def _hashed_key(row: Mapping[str, Any], seen: dict[str, int]) -> str:
    h = "h:" + content_hash(row)
    n = seen.get(h, 0)
    seen[h] = n + 1
    return h if n == 0 else f"{h}~{n}"


def diff(s: CanonState, s2: CanonState, normalizers: Normalizers | None = None) -> list[Change]:
    """``Δ(N(s), N(s2))``: the changes, sorted by address. Exact and deterministic (see the module docstring)."""
    n = normalizers or Normalizers()
    a, b = n.apply(s), n.apply(s2)
    changes: list[Change] = []
    for app in sorted(set(a.apps) | set(b.apps)):
        ca, cb = a.apps.get(app, {}), b.apps.get(app, {})
        for coll in sorted(set(ca) | set(cb)):
            ra, rb = ca.get(coll, {}), cb.get(coll, {})
            ic = n.id_canon_for(app, coll)
            inserted = sorted(set(rb) - set(ra))
            if ic is not None:  # re-key inserts by content (sorted first, so equal rows get stable ~n suffixes)
                seen: dict[str, int] = {}
                stripped = [{f: v for f, v in rb[k].items() if f not in ic.id_fields} for k in inserted]
                for row in sorted(stripped, key=cjson):
                    changes.append(Change(addr=Address(app=app, coll=coll, key=_hashed_key(row, seen)), kind="insert", new=row))
            else:
                for k in inserted:
                    changes.append(Change(addr=Address(app=app, coll=coll, key=k), kind="insert", new=dict(rb[k])))
            for k in sorted(set(ra) - set(rb)):
                changes.append(Change(addr=Address(app=app, coll=coll, key=k), kind="delete", old=dict(ra[k])))
            for k in sorted(set(ra) & set(rb)):
                xa, xb = ra[k], rb[k]
                for f in sorted(set(xa) | set(xb)):
                    va, vb = xa.get(f, MISSING), xb.get(f, MISSING)
                    if not same_value(va, vb):
                        changes.append(Change(addr=Address(app=app, coll=coll, key=k, field=f), kind="update",
                                              old=None if va is MISSING else va, new=None if vb is MISSING else vb))
    changes.sort(key=Change.sort_key)
    return changes


def same_diff(d1: Sequence[Change], d2: Sequence[Change]) -> bool:
    """Equality of two diffs as sets of changes (the MUT equivalence test)."""
    if len(d1) != len(d2):
        return False
    x = sorted(d1, key=Change.sort_key)
    y = sorted(d2, key=Change.sort_key)
    return all(p.same(q) for p, q in zip(x, y, strict=True))


# ---------------------------------------------------------------------------------------------------------- canonicalizers

TableDump = dict[str, Any]  # {"columns": [...], "pk": [...], "rows": [[cell, ...], ...]}


def decode_cell(v: Any) -> Any:
    """A cell as the sql state snippet encodes it (``{"f": float.hex}``, ``{"b": hex}``) back to a JSON value."""
    if isinstance(v, dict) and len(v) == 1:
        if "f" in v:
            return float.fromhex(v["f"])
        if "b" in v:
            return "b:" + v["b"]
        if "x" in v:
            return v["x"]
    if isinstance(v, bytes | memoryview):
        return "b:" + bytes(v).hex()
    return v


def _match(name: str, pats: Iterable[str]) -> bool:
    return any(fnmatch.fnmatchcase(name, p) for p in pats)


def sqlite_tables(source: str | os.PathLike[str] | sqlite3.Connection, include: Sequence[str] = (), exclude: Sequence[str] = ()) -> dict[str, TableDump]:
    """Every user table of a SQLite database: columns, primary-key columns (``PRAGMA table_info``, in key order) and rows.
    A path is opened read-only."""
    own = not isinstance(source, sqlite3.Connection)
    con = sqlite3.connect(f"file:{os.fspath(source)}?mode=ro", uri=True) if own else source  # type: ignore[arg-type]
    try:
        names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name")]
        out: dict[str, TableDump] = {}
        for t in names:
            if (include and not _match(t, include)) or (exclude and _match(t, exclude)):
                continue
            info = con.execute(f'PRAGMA table_info("{t}")').fetchall()
            cols = [r[1] for r in info]
            pk = [r[1] for r in sorted((r for r in info if r[5]), key=lambda r: r[5])]
            rows = [[decode_cell(v) for v in r] for r in con.execute(f'SELECT * FROM "{t}"')]
            out[t] = {"columns": cols, "pk": pk, "rows": rows}
        return out
    finally:
        if own:
            con.close()


def canon_sql_tables(app: str, tables: Mapping[str, TableDump], natural_keys: Mapping[str, Sequence[str]] | None = None) -> CanonState:
    """Rows keyed by primary key, else the declared natural key, else the row's content hash (``#``-prefixed; identical
    rows get ``~n`` suffixes). A key that collides (a natural key that is not unique) also gets ``~n``, in content order."""
    natural_keys = natural_keys or {}
    colls: dict[str, dict[str, dict[str, Any]]] = {}
    for t in sorted(tables):
        d = tables[t]
        cols = list(d["columns"])
        key_cols = list(d.get("pk") or []) or list(natural_keys.get(t, []))
        rows = [dict(zip(cols, (decode_cell(v) for v in r), strict=True)) for r in d["rows"]]
        rows.sort(key=cjson)
        dst: dict[str, dict[str, Any]] = {}
        seen: dict[str, int] = {}
        for row in rows:
            if key_cols:
                k = key_str([row.get(c) for c in key_cols])
                n = seen.get(k, 0)
                seen[k] = n + 1
                k = k if n == 0 else f"{k}~{n}"
            else:
                k = "#" + _hashed_key(row, seen)[2:]
            dst[k] = row
        colls[t] = dst
    return CanonState.build({app: colls})


def canon_sql(app: str, source: str | os.PathLike[str] | sqlite3.Connection, *, natural_keys: Mapping[str, Sequence[str]] | None = None,
              include: Sequence[str] = (), exclude: Sequence[str] = ()) -> CanonState:
    return canon_sql_tables(app, sqlite_tables(source, include, exclude), natural_keys)


def _mode(m: int) -> str:
    return format(stat.S_IMODE(m), "o")


def canon_fs(app: str, root: str | os.PathLike[str], *, coll: str = "files", text_max: int = 4096,
             exclude: Sequence[str] = (".git", ".git/*")) -> CanonState:
    """A file tree: path (posix, relative) → ``sha256``, ``mode`` and, for a UTF-8 file of at most ``text_max`` bytes,
    ``text``. A symlink is ``link`` (its target), never followed. Directories are implied by their files."""
    base = Path(root)
    rows: dict[str, dict[str, Any]] = {}
    if base.exists():
        for p in sorted(base.rglob("*")):
            rel = p.relative_to(base).as_posix()
            if _match(rel, exclude):
                continue
            st = p.lstat()
            if stat.S_ISLNK(st.st_mode):
                rows[rel] = {"link": os.readlink(p), "mode": _mode(st.st_mode)}
            elif stat.S_ISREG(st.st_mode):
                data = p.read_bytes()
                row: dict[str, Any] = {"sha256": hashlib.sha256(data).hexdigest(), "mode": _mode(st.st_mode)}
                if len(data) <= text_max:
                    try:
                        row["text"] = data.decode("utf-8")
                    except UnicodeDecodeError:
                        pass
                rows[rel] = row
    return CanonState.build({app: {coll: rows}})


def canon_fs_manifest(app: str, entries: Iterable[Sequence[str]], *, coll: str = "files", texts: Mapping[str, str] | None = None) -> CanonState:
    """A file tree from a state adapter's manifest (``[rel, mode, hash]`` entries, ``world/state/fs.py``)."""
    rows: dict[str, dict[str, Any]] = {}
    for rel, mode, h in entries:
        rows[rel] = {"sha256": h, "mode": mode, **({"text": texts[rel]} if texts and rel in texts else {})}
    return CanonState.build({app: {coll: rows}})


def _esc(seg: str) -> str:
    return seg.replace("~", "~0").replace("/", "~1")


def _flatten(v: Any, segs: list[str], list_keys: Mapping[str, str], out: list[tuple[list[str], Any]]) -> None:
    ptr = "/" + "/".join(segs) if segs else ""
    if isinstance(v, dict) and v:
        for k in sorted(v):
            _flatten(v[k], [*segs, _esc(str(k))], list_keys, out)
        return
    if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
        kf = list_keys.get(ptr)
        if kf is not None and all(kf in x for x in v):
            for x in v:
                _flatten({k: y for k, y in x.items()}, [*segs, _esc(f"[{kf}={key_str([x[kf]])}]")], list_keys, out)
        else:  # positional: a list of objects without a declared key field
            for i, x in enumerate(v):
                _flatten(x, [*segs, str(i)], list_keys, out)
        return
    out.append((segs, v))  # a scalar, an empty container, or a list of scalars (one atomic value)


def canon_json(app: str, coll: str, doc: Any, *, depth: int = 1, list_keys: Mapping[str, str] | None = None) -> CanonState:
    """A JSON document flattened by pointer: the entity key is the pointer prefix of ``depth`` segments, the field is
    the rest (``""`` for a leaf at the prefix itself). A list of objects under a pointer named in ``list_keys`` is keyed
    by that field (``/tickets/[id=7]/status``), so its order is not identity; any other list of objects is positional
    (``/tickets/0/status``); a list of scalars is one atomic value (see ``Normalizers.order_insensitive``)."""
    leaves: list[tuple[list[str], Any]] = []
    _flatten(doc, [], list_keys or {}, leaves)
    rows: dict[str, dict[str, Any]] = {}
    for segs, v in leaves:
        k = "/" + "/".join(segs[:depth]) if segs else ""
        f = "/" + "/".join(segs[depth:]) if len(segs) > depth else ""
        rows.setdefault(k, {})[f] = v
    return CanonState.build({app: {coll: rows}})


def canon_mail(app: str, messages: Iterable[Mapping[str, Any]], *, coll: str = "messages", id_field: str = "id") -> CanonState:
    """Messages keyed by their id (a message without one by its content hash); every other field is a cell, lists and
    objects atomic."""
    rows: dict[str, dict[str, Any]] = {}
    seen: dict[str, int] = {}
    for m in sorted(messages, key=cjson):
        row = {k: v for k, v in m.items() if k != id_field}
        k = key_str([m[id_field]]) if id_field in m else "#" + _hashed_key(row, seen)[2:]
        rows[k] = row
    return CanonState.build({app: {coll: rows}})


def canon_gym(app: str, obs: Sequence[float], *, coll: str = "obs") -> CanonState:
    """A Gym observation vector: index → ``value`` (positional by nature; contracts use region predicates)."""
    rows = {}
    for i, x in enumerate(obs):
        v = float(x)
        if not math.isfinite(v):
            raise ValueError(f"observation {i} is not finite")
        rows[str(i)] = {"value": v}
    return CanonState.build({app: {coll: rows}})


def canon_tables(app: str, tables: Mapping[str, Mapping[Any, Mapping[str, Any]]], *, positional: Iterable[str] = ()) -> CanonState:
    """In-memory dict tables (``PyWorld``): ``{coll: {key: row}}``. A collection named in ``positional`` is keyed by row
    position in insertion order (a deliberately weak canonicalization, as a positional JSON list is)."""
    pos = set(positional)
    colls: dict[str, dict[str, dict[str, Any]]] = {}
    for coll, rows in tables.items():
        if coll in pos:
            colls[coll] = {str(i): dict(r) for i, r in enumerate(rows.values())}
        else:
            colls[coll] = {key_str([k]): dict(r) for k, r in rows.items()}
    return CanonState.build({app: colls})
