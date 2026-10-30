"""Deterministic start-state generation for World instances (part2-paper-core §6 density, §11 synthesized Apps).

A World **family** is generated once by the swarm and frozen: tool code, blind tests, and a ``DataSpec``, the
family's *data generator* written as data. An **instance** is the family plus a seed and a data profile, sampled here
with no model call, so one family yields as many distinct, dense start states as wanted at zero marginal cost.

The swarm writes the spec; this module only interprets it (a small, closed language), samples it, plants the noise a
real system has, and checks the result:

    {"tables": [{"table": "invoices", "rows": 60,
                 "fields": {"invoice_id": {"kind": "seq", "prefix": "INV-", "start": 1001},
                            "vendor_id":  {"kind": "ref", "table": "vendors", "where": {"status": "active"}},
                            "vendor_name": {"kind": "lookup", "ref": "vendor_id", "field": "name"},
                            "amount":     {"kind": "number", "dist": "lognormal", "mean": 2500, "sd": 1.0, "min": 50, "max": 90000, "round": 2},
                            "status":     {"kind": "choice", "values": ["pending", "approved", "on_hold"], "weights": [6, 3, 1]},
                            "invoice_date": {"kind": "date", "days_ago": [5, 200]},
                            "due_date":   {"kind": "date_offset", "from": "invoice_date", "min_days": 15, "max_days": 60},
                            "approved_at": {"kind": "date_offset", "from": "invoice_date", "min_days": 1, "max_days": 10,
                                            "when": "status == 'approved'", "else": null},
                            "total":      {"kind": "sum", "table": "invoice_lines", "fk": "invoice_id", "field": "line_total"}},
                 "rules": ["due_date >= invoice_date", "amount > 0"],
                 "near_duplicate": {"vary": ["vendor_name"]},
                 "stale": {"set": {"status": "superseded"}, "older": ["invoice_date", "due_date"]}}]}

Field kinds: ``seq`` (keys), ``choice``, ``int``, ``number`` (uniform, normal, lognormal), ``bool``, ``date`` (``days_ago``
relative to the World clock's today, or ``start``/``end``), ``date_offset``, ``ref`` (a key of another table, optionally
filtered), ``ref_list``, ``lookup`` (a field copied from the referenced row: the master copy a conflicting source
disagrees with), ``pool`` (one of a list of strings, ``unique`` draws without replacement), ``template`` (``{part}`` from
pools and ``{field}`` from the row), ``calc`` (an arithmetic expression over the row), ``const``, ``list``, and the
aggregates ``sum`` / ``count`` over a child table (computed after every table exists). Any field may carry ``when`` (an
expression; false gives ``else``) and ``null_p``. Expressions are a closed AST subset: names are the row's fields,
literals, arithmetic, comparisons, ``and``/``or``/``not``, ``in``, ``if``-``else``, and ``round``/``min``/``max``/``abs``/
``len``/``days_between``.

**Noise** (per ``Profile``): ``near_duplicate`` clones a row under a new key with one varied field (a typo, a suffix, a
digit swap); ``stale`` clones a row with older dates and the ``set`` fields (a superseded record); ``conflict`` makes a
``lookup`` copy disagree with its master. Every planted item is recorded in the noise manifest (never shown to the
agent). Aggregates are computed before noise, and noise rows are exempt from aggregate checks.

**Checks** (``integrity``): every row has every schema field with the declared type; every ``ref`` resolves; every
``choice`` value is declared; dates parse and ``date`` values are not after today; ``date_offset`` values follow their
source; every rule holds on every row that is not noise; ``lookup`` copies equal their master except the planted
conflicts; aggregates equal their children.
"""

from __future__ import annotations

import ast
import copy
import math
import random
import re
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

TODAY = datetime(2026, 1, 1, tzinfo=UTC)  # the PyWorld clock's t0: no generated history lies after it
KINDS = ("seq", "choice", "int", "number", "bool", "date", "date_offset", "ref", "ref_list", "lookup", "pool", "template", "calc", "const",
         "list", "sum", "count")
AGGREGATES = ("sum", "count")


class DataSpecError(ValueError):
    """The spec cannot be compiled or sampled (the text goes back to the swarm as feedback)."""


# ---------------------------------------------------------------------------------------------------------- the spec

class _M(BaseModel):
    model_config = ConfigDict(extra="ignore")


class NearDup(_M):
    vary: list[str] = Field(default_factory=list)


class Stale(_M):
    set: dict[str, Any] = Field(default_factory=dict)
    older: list[str] = Field(default_factory=list)


class TableGen(_M):
    table: str
    rows: int = 20
    fields: dict[str, dict[str, Any]]
    rules: list[str] = Field(default_factory=list)
    near_duplicate: NearDup | None = None
    stale: Stale | None = None

    @field_validator("rows")
    @classmethod
    def _rows(cls, v: int) -> int:
        if not 1 <= v <= 2000:
            raise ValueError(f"rows must be 1 to 2000, not {v}")
        return v


class DataSpec(_M):
    tables: list[TableGen]
    notes: str = ""


class Profile(_M):
    """An instance's data profile: row-count scale, per-table minimum (density), and noise rates (share of a table)."""

    name: str = "standard"
    scale: float = 1.0
    min_rows: dict[str, int] = Field(default_factory=dict)
    near_duplicate: float = 0.08
    stale: float = 0.06
    conflict: float = 0.06


PROFILES: dict[str, Profile] = {
    "standard": Profile(name="standard"),
    "dense": Profile(name="dense", scale=1.6, near_duplicate=0.08, stale=0.06, conflict=0.05),
    "noisy": Profile(name="noisy", scale=1.0, near_duplicate=0.15, stale=0.12, conflict=0.12),
    "clean": Profile(name="clean", scale=1.2, near_duplicate=0.03, stale=0.02, conflict=0.02),
}


class NoiseItem(_M):
    kind: Literal["near_duplicate", "stale", "conflict"]
    table: str
    key: Any
    of: Any = None  # the row it was cloned from (near_duplicate, stale)
    field: str | None = None
    master: str | None = None  # conflict: "<table>.<key>.<field>"
    value: Any = None
    original: Any = None


class Generated(_M):
    tables: dict[str, list[dict[str, Any]]]
    noise: list[NoiseItem] = Field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------- expressions

_FUNCS = {"round": round, "min": min, "max": max, "abs": abs, "len": len}
_ALLOWED = (ast.Expression, ast.BoolOp, ast.And, ast.Or, ast.UnaryOp, ast.Not, ast.USub, ast.UAdd, ast.BinOp, ast.Add, ast.Sub, ast.Mult,
            ast.Div, ast.FloorDiv, ast.Mod, ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn, ast.Is,
            ast.IsNot, ast.Name, ast.Load, ast.Constant, ast.List, ast.Tuple, ast.IfExp, ast.Call)


def parse_expr(src: str) -> ast.Expression:
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise DataSpecError(f"expression {src!r}: {e.msg}") from None
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED):
            raise DataSpecError(f"expression {src!r}: {type(node).__name__} is not allowed")
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in (*_FUNCS, "days_between")):
            raise DataSpecError(f"expression {src!r}: only round, min, max, abs, len, days_between may be called")
    return tree


def expr_names(src: str) -> set[str]:
    tree = parse_expr(src)
    calls = {n.func.id for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} - calls - {"None", "True", "False"}


def _days_between(a: Any, b: Any) -> int:
    return (parse_date(b) - parse_date(a)).days


def eval_expr(src: str, row: Mapping[str, Any]) -> Any:
    tree = parse_expr(src)
    env = {**_FUNCS, "days_between": _days_between, "None": None, "True": True, "False": False}

    def ev(n: ast.AST) -> Any:
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant):
            return n.value
        if isinstance(n, ast.Name):
            if n.id in row:
                return row[n.id]
            if n.id in env:
                return env[n.id]
            raise DataSpecError(f"expression {src!r}: unknown field {n.id!r}")
        if isinstance(n, ast.List | ast.Tuple):
            return [ev(x) for x in n.elts]
        if isinstance(n, ast.BoolOp):
            vals = (ev(x) for x in n.values)
            return all(vals) if isinstance(n.op, ast.And) else any(vals)
        if isinstance(n, ast.UnaryOp):
            v = ev(n.operand)
            return (not v) if isinstance(n.op, ast.Not) else (-v if isinstance(n.op, ast.USub) else +v)
        if isinstance(n, ast.BinOp):
            a, b = ev(n.left), ev(n.right)
            ops = {ast.Add: lambda: a + b, ast.Sub: lambda: a - b, ast.Mult: lambda: a * b, ast.Div: lambda: a / b, ast.FloorDiv: lambda: a // b,
                   ast.Mod: lambda: a % b}
            return ops[type(n.op)]()
        if isinstance(n, ast.Compare):
            left = ev(n.left)
            for op, right_n in zip(n.ops, n.comparators, strict=True):
                right = ev(right_n)
                ok = {ast.Eq: lambda x, y: x == y, ast.NotEq: lambda x, y: x != y, ast.Lt: lambda x, y: x < y, ast.LtE: lambda x, y: x <= y,
                      ast.Gt: lambda x, y: x > y, ast.GtE: lambda x, y: x >= y, ast.In: lambda x, y: x in y, ast.NotIn: lambda x, y: x not in y,
                      ast.Is: lambda x, y: x is y, ast.IsNot: lambda x, y: x is not y}[type(op)](left, right)
                if not ok:
                    return False
                left = right
            return True
        if isinstance(n, ast.IfExp):
            return ev(n.body) if ev(n.test) else ev(n.orelse)
        if isinstance(n, ast.Call):
            f = env[n.func.id]  # type: ignore[union-attr]
            return f(*[ev(a) for a in n.args])
        raise DataSpecError(f"expression {src!r}: unsupported")

    try:
        return ev(tree)
    except DataSpecError:
        raise
    except Exception as e:  # noqa: BLE001 - a type error in the spec's arithmetic is the spec's problem
        raise DataSpecError(f"expression {src!r} failed on a row: {type(e).__name__}: {e}") from None


# ---------------------------------------------------------------------------------------------------------- dates

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}([T ]\d{2}:\d{2}(:\d{2})?(\.\d+)?(Z|[+-]\d{2}:?\d{2})?)?$")


def parse_date(v: Any) -> datetime:
    if not isinstance(v, str) or not _DATE.match(v):
        raise ValueError(f"not an ISO date: {v!r}")
    s = v.replace("Z", "+00:00").replace(" ", "T")
    d = datetime.fromisoformat(s)
    return d if d.tzinfo else d.replace(tzinfo=UTC)


def is_date(v: Any) -> bool:
    try:
        parse_date(v)
        return True
    except ValueError:
        return False


def fmt_date(d: datetime, with_time: bool) -> str:
    return d.strftime("%Y-%m-%dT%H:%M:%SZ") if with_time else d.strftime("%Y-%m-%d")


def _shift(v: str, days: int) -> str:
    d = parse_date(v) - timedelta(days=days)
    return fmt_date(d, "T" in v)


# ---------------------------------------------------------------------------------------------------------- compile

def _deps(g: Mapping[str, Any]) -> set[str]:
    out: set[str] = set()
    k = g.get("kind")
    if k == "date_offset":
        out.add(str(g.get("from")))
    elif k == "lookup":
        out.add(str(g.get("ref")))
    elif k == "calc":
        out |= expr_names(str(g.get("expr", "")))
    elif k == "template":
        out |= {m for m in re.findall(r"\{(\w+)\}", str(g.get("template", ""))) if m not in (g.get("parts") or {})}
    elif k == "list" and isinstance(g.get("of"), dict):
        out |= _deps(g["of"])
    if g.get("when"):
        out |= expr_names(str(g["when"]))
    return out


def _check_gen(t: str, f: str, g: Mapping[str, Any], schema_fields: Mapping[str, str], tables: Mapping[str, TableGen]) -> None:
    where = f"{t}.{f}"
    k = g.get("kind")
    if k not in KINDS:
        raise DataSpecError(f"{where}: unknown kind {k!r} (one of {', '.join(KINDS)})")
    if k == "choice" and not (isinstance(g.get("values"), list) and g["values"]):
        raise DataSpecError(f"{where}: choice needs a non-empty values list")
    if k == "choice" and g.get("weights") is not None and len(g["weights"]) != len(g["values"]):
        raise DataSpecError(f"{where}: weights and values differ in length")
    if k == "pool" and not (isinstance(g.get("values"), list) and g["values"]):
        raise DataSpecError(f"{where}: pool needs a non-empty values list")
    if k in ("ref", "ref_list", "sum", "count") and g.get("table") not in tables:
        raise DataSpecError(f"{where}: {k} names table {g.get('table')!r}, which the spec does not generate")
    if k in ("sum", "count"):
        child = tables[g["table"]]
        if g.get("fk") not in child.fields:
            raise DataSpecError(f"{where}: {k} needs fk, a field of {g['table']} that refers to {t}")
        if k == "sum" and g.get("field") not in child.fields:
            raise DataSpecError(f"{where}: sum needs field, a field of {g['table']}")
    if k == "lookup":
        rf = g.get("ref")
        if rf not in schema_fields and rf not in tables[t].fields:
            raise DataSpecError(f"{where}: lookup ref {rf!r} is not a field of {t}")
        rg = tables[t].fields.get(str(rf)) or {}
        if rg.get("kind") != "ref":
            raise DataSpecError(f"{where}: lookup ref {rf!r} must be a ref field")
        if g.get("field") not in tables[rg["table"]].fields:
            raise DataSpecError(f"{where}: lookup field {g.get('field')!r} is not a field of {rg['table']}")
    if k == "date_offset" and g.get("from") not in tables[t].fields:
        raise DataSpecError(f"{where}: date_offset from {g.get('from')!r} is not a field of {t}")
    if k == "date" and not (g.get("days_ago") or (g.get("start") and g.get("end"))):
        raise DataSpecError(f"{where}: date needs days_ago [lo, hi] or start and end")
    if k == "calc":
        parse_expr(str(g.get("expr", "")))
    if k == "template" and not g.get("template"):
        raise DataSpecError(f"{where}: template needs a template text")
    if g.get("when"):
        parse_expr(str(g["when"]))
    for n in _deps(g):
        if n not in tables[t].fields:
            raise DataSpecError(f"{where}: refers to {n!r}, which is not a field of {t}")


def compile_spec(spec: DataSpec, schema: Mapping[str, Mapping[str, Any]]) -> list[tuple[TableGen, list[str]]]:
    """Check the spec against the schema (``{table: {"key": k, "fields": {f: type}}}``) and order it: tables by their
    ``ref`` dependencies, each table's fields by their own. ``DataSpecError`` names the first problem."""
    tables = {t.table: t for t in spec.tables}
    if len(tables) != len(spec.tables):
        raise DataSpecError("a table is generated twice")
    for t in spec.tables:
        if t.table not in schema:
            raise DataSpecError(f"table {t.table!r} is not in the schema")
        sf = schema[t.table]["fields"]
        missing = sorted(set(sf) - set(t.fields))
        if missing:
            raise DataSpecError(f"table {t.table}: no generator for schema fields {missing}")
        extra = sorted(set(t.fields) - set(sf))
        if extra:
            raise DataSpecError(f"table {t.table}: generators for fields the schema does not have {extra}")
        key = schema[t.table]["key"]
        kg = t.fields[key]
        if kg.get("kind") not in ("seq", "pool", "template"):
            raise DataSpecError(f"table {t.table}: the key {key!r} needs a seq, pool or template generator")
        if kg.get("kind") in ("pool", "template") and not kg.get("unique"):
            raise DataSpecError(f"table {t.table}: a pool or template key must be unique (\"unique\": true)")
        for f, g in t.fields.items():
            _check_gen(t.table, f, g, sf, tables)
        for r in t.rules:
            bad = expr_names(r) - set(t.fields)
            if bad:
                raise DataSpecError(f"table {t.table}: rule {r!r} names unknown fields {sorted(bad)}")
        for nf in (t.near_duplicate.vary if t.near_duplicate else []):
            if nf not in t.fields or nf == key:
                raise DataSpecError(f"table {t.table}: near_duplicate varies {nf!r}, which is not a non-key field")
        if t.stale:
            for f in [*t.stale.set, *t.stale.older]:
                if f not in t.fields or f == key:
                    raise DataSpecError(f"table {t.table}: stale names {f!r}, which is not a non-key field")
            for f, v in t.stale.set.items():
                g = t.fields[f]
                if g.get("kind") == "choice" and v not in g["values"]:
                    raise DataSpecError(f"table {t.table}: stale sets {f}={v!r}, which is not one of its choice values {g['values']}")
    order: list[str] = []
    state: dict[str, int] = {}

    def visit(n: str, path: tuple[str, ...]) -> None:
        if state.get(n) == 2:
            return
        if state.get(n) == 1:
            raise DataSpecError(f"tables refer to each other in a cycle: {' -> '.join((*path, n))}")
        state[n] = 1
        for g in tables[n].fields.values():
            if g.get("kind") in ("ref", "ref_list") and g["table"] != n:
                visit(g["table"], (*path, n))
        state[n] = 2
        order.append(n)

    for n in tables:
        visit(n, ())
    out = []
    for n in order:
        t = tables[n]
        fo: list[str] = []
        st: dict[str, int] = {}

        def fvisit(f: str, path: tuple[str, ...], t: TableGen = t, fo: list[str] = fo, st: dict[str, int] = st) -> None:
            if st.get(f) == 2:
                return
            if st.get(f) == 1:
                raise DataSpecError(f"table {t.table}: fields depend on each other in a cycle: {' -> '.join((*path, f))}")
            st[f] = 1
            for d in sorted(_deps(t.fields[f])):
                fvisit(d, (*path, f))
            st[f] = 2
            fo.append(f)

        for f in t.fields:
            fvisit(f, ())
        out.append((t, fo))
    return out


# ---------------------------------------------------------------------------------------------------------- sampling

def _num(g: Mapping[str, Any], rng: random.Random) -> float:
    lo, hi = float(g.get("min", 0)), float(g.get("max", 1000))
    dist = g.get("dist", "uniform")
    if dist == "lognormal":
        mean = float(g.get("mean", (lo + hi) / 2) or 1.0)
        sd = abs(float(g.get("sd", 0.75)))
        # pilot 1: a spec gave the spread in the value's own units (sd 4000 for a mean of 2500), which overflows as a log
        # sigma; a spread above 3 is read as an absolute standard deviation and converted
        sigma = sd if sd <= 3 else math.sqrt(math.log(1 + (sd / max(mean, 1e-9)) ** 2))
        v = rng.lognormvariate(math.log(max(mean, 1e-9)), min(sigma, 3.0))
    elif dist == "normal":
        v = rng.gauss(float(g.get("mean", (lo + hi) / 2)), float(g.get("sd", (hi - lo) / 6 or 1.0)))
    else:
        v = rng.uniform(lo, hi)
    v = min(hi, max(lo, v))
    return round(v, int(g.get("round", 2)))


class _Sampler:
    def __init__(self, compiled: list[tuple[TableGen, list[str]]], schema: Mapping[str, Mapping[str, Any]], profile: Profile, seed: int):
        self.compiled, self.schema, self.profile = compiled, schema, profile
        self.rng = random.Random(f"datagen/{seed}/{profile.name}")
        self.out: dict[str, list[dict[str, Any]]] = {}
        self.by_key: dict[str, dict[str, dict[str, Any]]] = {}
        self.used: dict[tuple[str, str], set[Any]] = {}
        self.seq: dict[str, int] = {}

    def n_rows(self, t: TableGen) -> int:
        return max(round(t.rows * self.profile.scale), int(self.profile.min_rows.get(t.table, 0)), 1)

    def key_of(self, table: str) -> str:
        return str(self.schema[table]["key"])

    def pick_pool(self, t: str, f: str, values: Sequence[Any], unique: bool) -> Any:
        if not unique:
            return self.rng.choice(list(values))
        used = self.used.setdefault((t, f), set())
        free = [v for v in values if v not in used]
        if not free:
            raise DataSpecError(f"{t}.{f}: a unique pool of {len(values)} values ran out (more rows than values)")
        v = self.rng.choice(free)
        used.add(v)
        return v

    def gen(self, t: TableGen, f: str, g: Mapping[str, Any], row: dict[str, Any]) -> Any:
        if g.get("when") and not eval_expr(str(g["when"]), row):
            return g.get("else")
        if g.get("null_p") and self.rng.random() < float(g["null_p"]):
            return None
        k = g["kind"]
        rng = self.rng
        if k == "seq":
            n = self.seq.get(t.table, int(g.get("start", 1)))
            self.seq[t.table] = n + int(g.get("step", 1))
            w = int(g.get("width", 0))
            s = str(n).zfill(w) if w else str(n)
            if not (g.get("prefix") or g.get("suffix")) and str(self.schema[t.table]["fields"].get(f, "string")).startswith("integer"):
                return n
            return f"{g.get('prefix', '')}{s}{g.get('suffix', '')}"
        if k == "choice":
            return rng.choices(list(g["values"]), weights=g.get("weights"), k=1)[0]
        if k == "int":
            return rng.randint(int(g.get("min", 0)), int(g.get("max", 100)))
        if k == "number":
            return _num(g, rng)
        if k == "bool":
            return rng.random() < float(g.get("p", 0.5))
        if k == "date":
            if g.get("days_ago"):
                lo, hi = (int(x) for x in g["days_ago"])
                d = TODAY - timedelta(days=rng.randint(min(lo, hi), max(lo, hi)), minutes=rng.randint(0, 600) if g.get("time") else 0)
            else:
                a, b = parse_date(g["start"]), parse_date(g["end"])
                d = a + timedelta(seconds=rng.randint(0, max(0, int((b - a).total_seconds()))))
            return fmt_date(d, bool(g.get("time")))
        if k == "date_offset":
            base = row.get(str(g["from"]))
            if base is None:
                return None
            d = parse_date(base) + timedelta(days=rng.randint(int(g.get("min_days", 0)), int(g.get("max_days", 30))))
            return fmt_date(d, "T" in str(base) or bool(g.get("time")))
        if k == "ref":
            return self.pick_ref(t.table, f, g)
        if k == "ref_list":
            n = rng.randint(int(g.get("min", 0)), int(g.get("max", 3)))
            keys = self.candidates(g)
            return sorted(rng.sample(keys, min(n, len(keys))), key=str)
        if k == "lookup":
            ref_g = t.fields[str(g["ref"])]
            target = self.by_key[ref_g["table"]].get(str(row.get(str(g["ref"]))))
            return None if target is None else copy.deepcopy(target.get(str(g["field"])))
        if k == "pool":
            return self.pick_pool(t.table, f, g["values"], bool(g.get("unique")))
        if k == "template":
            for _ in range(200):
                parts = {p: rng.choice(list(vs)) for p, vs in (g.get("parts") or {}).items()}
                try:
                    v = str(g["template"]).format(**{**{x: ("" if y is None else y) for x, y in row.items()}, **parts})
                except (KeyError, IndexError, ValueError) as e:
                    raise DataSpecError(f"{t.table}.{f}: template {g['template']!r}: {e}") from None
                if not g.get("unique") or v not in self.used.setdefault((t.table, f), set()):
                    if g.get("unique"):
                        self.used[(t.table, f)].add(v)
                    return v
            raise DataSpecError(f"{t.table}.{f}: a unique template ran out of combinations")
        if k == "calc":
            v = eval_expr(str(g["expr"]), row)
            return round(v, int(g["round"])) if g.get("round") is not None and isinstance(v, int | float) else v
        if k == "const":
            return copy.deepcopy(g.get("value"))
        if k == "list":
            n = rng.randint(int(g.get("min", 0)), int(g.get("max", 3)))
            vals = [self.gen(t, f, {**g["of"], "unique": False}, row) for _ in range(n)]
            return sorted(set(vals), key=str) if g.get("unique") else vals
        if k in AGGREGATES:
            return None  # filled after every table exists
        raise DataSpecError(f"{t.table}.{f}: unknown kind {k!r}")

    def candidates(self, g: Mapping[str, Any]) -> list[Any]:
        rows = self.out.get(str(g["table"]), [])
        kf = self.key_of(str(g["table"]))
        where = g.get("where") or {}
        keep = [r[kf] for r in rows if all((r.get(a) in b) if isinstance(b, list) else r.get(a) == b for a, b in where.items())]
        if not keep:
            raise DataSpecError(f"ref to {g['table']} where {where}: no row matches")
        return keep

    def pick_ref(self, table: str, f: str, g: Mapping[str, Any]) -> Any:
        keys = self.candidates(g)
        if g.get("unique"):
            return self.pick_pool(table, f, keys, True)
        return self.rng.choice(keys)

    def row(self, t: TableGen, order: list[str], tries: int = 40) -> dict[str, Any]:
        """One row, **rejection-sampled against the table's rules**: a draw that breaks a rule is redrawn (its unique
        values and sequence number released), up to ``tries`` times; the last draw is kept and ``integrity`` reports it.
        Pilot 1: a leave-balance spec drew ``used`` and ``entitled`` independently, so ``used <= entitled`` failed on a
        third of the rows and the swarm could not express the dependency."""
        seq0 = dict(self.seq)
        row: dict[str, Any] = {}
        for _ in range(tries):
            before = {k: set(v) for k, v in self.used.items()}
            self.seq = dict(seq0)
            row = {}
            try:
                for f in order:
                    row[f] = self.gen(t, f, t.fields[f], row)
            except (OverflowError, ZeroDivisionError, TypeError, ValueError) as e:
                if isinstance(e, DataSpecError):
                    raise
                raise DataSpecError(f"table {t.table}: a generator failed: {type(e).__name__}: {e}") from None
            ok = True
            for r in t.rules:
                try:
                    ok = bool(eval_expr(r, row))
                except DataSpecError:
                    ok = True  # an unevaluable rule is integrity's to report
                if not ok:
                    break
            if ok:
                return row
            self.used = before
        return row

    def run(self) -> Generated:
        for t, order in self.compiled:
            kf = self.key_of(t.table)
            rows: list[dict[str, Any]] = []
            self.out[t.table] = rows
            self.by_key[t.table] = {}
            for _ in range(self.n_rows(t)):
                row = self.row(t, order)
                if str(row[kf]) in self.by_key[t.table]:
                    raise DataSpecError(f"table {t.table}: key {row[kf]!r} repeats")
                rows.append({f: row[f] for f in t.fields})  # the schema's field order
                self.by_key[t.table][str(row[kf])] = rows[-1]
        aggregates(self.compiled, self.out, self.schema)
        noise = self.plant()
        return Generated(tables=self.out, noise=noise)

    # -- noise -----------------------------------------------------------------------------------------------------
    def fresh_key(self, t: TableGen) -> Any:
        kf = self.key_of(t.table)
        g = t.fields[kf]
        for _ in range(500):
            k = self.gen(t, kf, g, {})
            if str(k) not in self.by_key[t.table]:
                return k
        raise DataSpecError(f"table {t.table}: no fresh key for a noise row")

    def plant(self) -> list[NoiseItem]:
        out: list[NoiseItem] = []
        p = self.profile
        rng = self.rng
        for t, _order in self.compiled:
            rows = self.out[t.table]
            kf = self.key_of(t.table)
            base = list(rows)
            if not base:
                continue

            def count(rate: float, n: int = len(base)) -> int:
                return 0 if rate <= 0 else max(1, round(rate * n))

            lookups = [(f, g) for f, g in t.fields.items() if g.get("kind") == "lookup"]
            if lookups:
                for src in rng.sample(base, min(len(base), count(p.conflict))):
                    f, g = rng.choice(lookups)
                    old = src.get(f)
                    if not isinstance(old, str | int | float) or isinstance(old, bool):
                        continue
                    src[f] = near_variant(old, rng)
                    ref_g = t.fields[str(g["ref"])]
                    out.append(NoiseItem(kind="conflict", table=t.table, key=src[kf], field=f, value=src[f], original=old,
                                         master=f"{ref_g['table']}.{src.get(str(g['ref']))}.{g['field']}"))
            if t.near_duplicate and t.near_duplicate.vary:
                for src in rng.sample(base, min(len(base), count(p.near_duplicate))):
                    f = rng.choice(t.near_duplicate.vary)
                    new = copy.deepcopy(src)
                    new[kf] = self.fresh_key(t)
                    new[f] = near_variant(src.get(f), rng)
                    rows.append(new)
                    self.by_key[t.table][str(new[kf])] = new
                    out.append(NoiseItem(kind="near_duplicate", table=t.table, key=new[kf], of=src[kf], field=f, value=new[f], original=src.get(f)))
            if t.stale and (t.stale.set or t.stale.older):
                for src in rng.sample(base, min(len(base), count(p.stale))):
                    new = copy.deepcopy(src)
                    new[kf] = self.fresh_key(t)
                    days = rng.randint(45, 400)
                    for f in t.stale.older:
                        if isinstance(new.get(f), str) and is_date(new[f]):
                            new[f] = _shift(new[f], days)
                    new.update(copy.deepcopy(t.stale.set))
                    rows.append(new)
                    self.by_key[t.table][str(new[kf])] = new
                    out.append(NoiseItem(kind="stale", table=t.table, key=new[kf], of=src[kf], value=dict(t.stale.set)))
        return out


def aggregates(compiled: list[tuple[TableGen, list[str]]], out: dict[str, list[dict[str, Any]]], schema: Mapping[str, Mapping[str, Any]],
               skip: set[tuple[str, str]] | None = None) -> None:
    for t, _ in compiled:
        kf = str(schema[t.table]["key"])
        for f, g in t.fields.items():
            if g.get("kind") not in AGGREGATES:
                continue
            children = out.get(str(g["table"]), [])
            fk = str(g["fk"])
            for row in out[t.table]:
                if skip and (t.table, str(row[kf])) in skip:
                    continue
                mine = [c for c in children if str(c.get(fk)) == str(row[kf]) and (not skip or (str(g["table"]), str(c.get(str(schema[g["table"]]["key"])))) not in skip)]
                if g["kind"] == "count":
                    row[f] = len(mine)
                else:
                    tot = sum(float(c.get(str(g["field"])) or 0) for c in mine)
                    row[f] = round(tot, int(g.get("round", 2)))


_SUFFIXES = (("Inc.", "Inc"), ("Ltd", "Ltd."), ("LLC", "L.L.C."), ("Corp", "Corporation"), ("&", "and"), ("Co.", "Company"))


def near_variant(v: Any, rng: random.Random) -> Any:
    """A look-alike of ``v``: a suffix swap, a dropped or doubled letter, swapped adjacent letters, a case change, or a
    small numeric shift. Always different from ``v``."""
    if isinstance(v, bool) or v is None:
        return v if v is None else (not v)
    if isinstance(v, int):
        return v + rng.choice((-1, 1)) * max(1, abs(v) // 100)
    if isinstance(v, float):
        return round(v * rng.choice((0.99, 1.01)) + (0.01 if v == 0 else 0), 2)
    if isinstance(v, list):
        return [*v[:-1]] if v else ["?"]
    s = str(v)
    for a, b in _SUFFIXES:
        if a in s:
            return s.replace(a, b, 1)
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    ops = ["swap", "drop", "double", "case"] if len(letters) >= 3 else ["double", "case"]
    for _ in range(10):
        op = rng.choice(ops)
        if not letters:
            out = s + " "
        elif op == "swap":
            i = rng.choice(letters[:-1])
            out = s[:i] + s[i + 1] + s[i] + s[i + 2:]
        elif op == "drop":
            i = rng.choice(letters[1:])
            out = s[:i] + s[i + 1:]
        elif op == "double":
            i = rng.choice(letters)
            out = s[:i] + s[i] + s[i:]
        else:
            out = s.upper() if s != s.upper() else s.lower()
        if out != s:
            return out
    return s + "."


def generate(spec: DataSpec, schema: Mapping[str, Mapping[str, Any]], *, seed: int, profile: Profile | None = None) -> Generated:
    """Sample one start state (module docstring). Deterministic in (spec, schema, seed, profile)."""
    return _Sampler(compile_spec(spec, schema), schema, profile or PROFILES["standard"], seed).run()


# ---------------------------------------------------------------------------------------------------------- checks

_TYPES = {"string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict}


def _type_ok(v: Any, ty: str) -> bool:
    nullable = ty.endswith("?")
    base = ty.rstrip("?")
    if v is None:
        return nullable
    if base == "any":
        return True
    if base in ("integer", "number") and isinstance(v, bool):
        return False
    want = _TYPES.get(base)
    return want is None or isinstance(v, want)


def integrity(spec: DataSpec, schema: Mapping[str, Mapping[str, Any]], gen: Generated, *, max_problems: int = 40) -> list[str]:
    """Executed integrity and consistency checks of a generated state (module docstring): the problems, empty = clean."""
    probs: list[str] = []
    tables = {t.table: t for t in spec.tables}
    noise_keys = {(n.table, str(n.key)) for n in gen.noise if n.kind in ("near_duplicate", "stale")}
    conflicts = {(n.table, str(n.key), n.field) for n in gen.noise if n.kind == "conflict"}
    keys = {t: {str(r.get(schema[t]["key"])) for r in rows} for t, rows in gen.tables.items() if t in schema}
    by_key = {t: {str(r.get(schema[t]["key"])): r for r in rows} for t, rows in gen.tables.items() if t in schema}

    def add(p: str) -> None:
        if len(probs) < max_problems:
            probs.append(p)

    for tn, rows in gen.tables.items():
        if tn not in schema:
            add(f"table {tn} is not in the schema")
            continue
        sf = schema[tn]["fields"]
        kf = schema[tn]["key"]
        t = tables.get(tn)
        if len(keys[tn]) != len(rows):
            add(f"{tn}: keys repeat")
        for r in rows:
            k = str(r.get(kf))
            for f, ty in sf.items():
                if f not in r:
                    add(f"{tn}[{k}]: no field {f}")
                elif not _type_ok(r[f], str(ty)):
                    add(f"{tn}[{k}].{f} = {r[f]!r} is not a {ty}")
            if t is None:
                continue
            for f, g in t.fields.items():
                v = r.get(f)
                kind = g.get("kind")
                if v is None:
                    continue
                if kind == "ref" and str(v) not in keys.get(str(g["table"]), set()):
                    add(f"{tn}[{k}].{f} = {v!r} refers to no {g['table']} row")
                elif kind == "ref_list" and isinstance(v, list):
                    for x in v:
                        if str(x) not in keys.get(str(g["table"]), set()):
                            add(f"{tn}[{k}].{f} has {x!r}, which refers to no {g['table']} row")
                elif kind == "choice":
                    stale_ok = t.stale is not None and f in t.stale.set and v == t.stale.set[f] and (tn, k) in noise_keys
                    if v not in g["values"] and not stale_ok:
                        add(f"{tn}[{k}].{f} = {v!r} is not a declared value {g['values']}")
                elif kind == "date":
                    if not is_date(v):
                        add(f"{tn}[{k}].{f} = {v!r} is not an ISO date")
                    elif parse_date(v) > TODAY + timedelta(days=1):
                        add(f"{tn}[{k}].{f} = {v} is after today ({TODAY.date()}): history cannot lie in the future")
                elif kind == "date_offset" and (tn, k) not in noise_keys:
                    src = r.get(str(g["from"]))
                    if not is_date(v):
                        add(f"{tn}[{k}].{f} = {v!r} is not an ISO date")
                    elif src is not None and is_date(src) and parse_date(v) < parse_date(src) + timedelta(days=int(g.get("min_days", 0))) - timedelta(seconds=1):
                        add(f"{tn}[{k}].{f} = {v} is before {g['from']} = {src}")
                elif kind == "lookup" and (tn, k, f) not in conflicts and (tn, k) not in noise_keys:
                    ref_g = t.fields[str(g["ref"])]
                    master = by_key.get(str(ref_g["table"]), {}).get(str(r.get(str(g["ref"]))))
                    if master is not None and master.get(str(g["field"])) != v:
                        add(f"{tn}[{k}].{f} = {v!r} disagrees with its master {ref_g['table']}.{g['field']} = {master.get(str(g['field']))!r} "
                            "(not a planted conflict)")
            if (tn, k) in noise_keys:
                continue
            for rule in t.rules:
                try:
                    ok = eval_expr(rule, r)
                except DataSpecError as e:
                    add(f"{tn}[{k}]: rule {rule!r} cannot be evaluated: {e}")
                    continue
                if not ok:
                    add(f"{tn}[{k}]: rule {rule!r} does not hold")
    # aggregates against a recomputation over the non-noise rows
    if spec.tables:
        try:
            compiled = compile_spec(spec, schema)
        except DataSpecError as e:
            return [*probs, str(e)]
        redo = copy.deepcopy(gen.tables)
        aggregates(compiled, redo, schema, skip=noise_keys)
        for t, _ in compiled:
            kf = schema[t.table]["key"]
            for f, g in t.fields.items():
                if g.get("kind") not in AGGREGATES:
                    continue
                for r0, r1 in zip(gen.tables.get(t.table, []), redo.get(t.table, []), strict=False):
                    if (t.table, str(r0.get(kf))) in noise_keys:
                        continue
                    a, b = r0.get(f), r1.get(f)
                    if not (a == b or (isinstance(a, int | float) and isinstance(b, int | float) and abs(a - b) < 0.011)):
                        add(f"{t.table}[{r0.get(kf)}].{f} = {a!r} but its {g['table']} children give {b!r}")
    for n in gen.noise:
        if n.kind == "near_duplicate" and (n.table not in by_key or str(n.of) not in by_key[n.table]):
            add(f"noise: near-duplicate {n.table}[{n.key}] has no original {n.of!r}")
    return probs


def value_sets(gen: Generated) -> dict[str, dict[str, set[str]]]:
    """table -> field -> the distinct scalar values (as text) the state holds (for the enum and docs checks)."""
    out: dict[str, dict[str, set[str]]] = {}
    for t, rows in gen.tables.items():
        d: dict[str, set[str]] = {}
        for r in rows:
            for f, v in r.items():
                if isinstance(v, str | int | float | bool) and v is not None:
                    d.setdefault(f, set()).add(str(v))
        out[t] = d
    return out


DATAGEN_FORMAT = """DATA GENERATOR FORMAT (JSON; the framework samples it deterministically, many times, with different seeds and sizes)
{"tables": [{"table": "<schema table>", "rows": <base row count>,
             "fields": {"<every schema field>": <generator>},
             "rules": ["<expression that must hold on every row, e.g. due_date >= invoice_date>"],
             "near_duplicate": {"vary": ["<text field a look-alike copy differs in, e.g. a name>"]},
             "stale": {"set": {"<status field>": "<a declared value meaning outdated, e.g. superseded>"}, "older": ["<date fields>"]}}],
 "notes": "..."}
Generators ({"kind": ...}):
  seq       {"prefix": "INV-", "start": 1001}                      keys and document numbers
  choice    {"values": [...], "weights": [...]}                    statuses, categories, codes (realistic weights)
  int       {"min": a, "max": b}        number {"min", "max", "dist": "uniform|normal|lognormal", "mean", "sd", "round"}
  bool      {"p": 0.8}
  date      {"days_ago": [lo, hi], "time": false}                  history; today is 2026-01-01, never later
  date_offset {"from": "<date field>", "min_days": 0, "max_days": 30}  a later date (due dates may lie after today)
  ref       {"table": "<other table>", "where": {"<field>": <value or [values]>}}   a key of another table
  ref_list  {"table": ..., "min": 0, "max": 3}
  lookup    {"ref": "<a ref field of this table>", "field": "<field of the referenced row>"}  a copied master value
  pool      {"values": ["<realistic strings>", ...], "unique": true|false}   20 to 60 realistic values per pool
  template  {"template": "{first} {last}", "parts": {"first": [...], "last": [...]}, "unique": true}  ({field} = this row's field)
  calc      {"expr": "quantity * unit_price", "round": 2}
  const     {"value": ...}
  list      {"of": <generator>, "min": 0, "max": 3, "unique": true}
  sum       {"table": "<child table>", "fk": "<child field referring here>", "field": "<child number field>"}
  count     {"table": "<child table>", "fk": "<child field referring here>"}
  any generator may add "when": "<expression over this row>", "else": <value>, and "null_p": 0.1 (nullable fields only)
Expressions: field names, literals, + - * / // %, == != < <= > >=, in, not in, and, or, not, x if c else y,
round(), min(), max(), abs(), len(), days_between(a, b). Dates are ISO text and compare as text.
Keys use seq (or a unique pool/template). Every reference is a ref. Weights and ranges must look like a real company's data."""


__all__ = [
    "DATAGEN_FORMAT",
    "PROFILES",
    "TODAY",
    "DataSpec",
    "DataSpecError",
    "Generated",
    "NoiseItem",
    "Profile",
    "TableGen",
    "compile_spec",
    "eval_expr",
    "generate",
    "integrity",
    "near_variant",
    "value_sets",
]
