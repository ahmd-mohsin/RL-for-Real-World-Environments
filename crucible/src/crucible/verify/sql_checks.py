"""State checks for record worlds (M1 item 3; P4 of the M1 taxonomy; ports EnterpriseOps-Gym ``benchmark/verifier.py``,
Apache-2.0).

- ``SqlAssertion``: a query against an App's database and an expected value (EnterpriseOps ``database_state``). The
  result is reduced as EnterpriseOps reduces it: one row with one column is that value, one row is a dict, several
  rows are a list of dicts. Comparisons: ``equals`` (numbers compare numerically), ``not_equals``, ``gt``, ``gte``,
  ``lt``, ``lte``, ``contains`` (expected text inside the value), ``in`` (value among the expected list),
  ``rows_equal`` (the same rows in any order).
- ``Collateral``: "every other row is unchanged". The episode's row diff for an App (``VerifierView.changes``) must
  consist only of allowed changes: per table, ``any``, or one or more allowances, each rows matching ``match`` (field
  values on the row's before or after image) limited to ``ops`` (insert, delete, update; an update is a delete and an
  insert with the same ``key``). EnterpriseOps checks only what should have happened; this checks what should not have.

Both become M7 ``Check``s (``.check(gate)``) whose function reads ``ctx.params["view"]`` (a ``VerifierView``), and
both load from task data (``assertion_from`` / ``collateral_from``; ``from_enterpriseops`` converts a verifier entry).
A query that fails is a failed check (the policy may have broken the schema), not an infrastructure error.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.state import Query
from crucible.verify.runtime.plan import Check, Verdict

COMPARISONS = ("equals", "not_equals", "gt", "gte", "lt", "lte", "contains", "in", "rows_equal")


def reduce_result(res: dict[str, Any]) -> Any:
    """EnterpriseOps ``_extract_value_from_sql_result``: scalar, row dict, or list of row dicts."""
    cols, rows = res.get("columns") or [], res.get("rows") or []
    dicts = [dict(zip(cols, r, strict=False)) for r in rows]
    if len(dicts) == 1 and len(cols) == 1:
        return rows[0][0]
    if len(dicts) == 1:
        return dicts[0]
    return dicts


def _num(v: Any) -> float | None:
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, int | float):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v)
        except ValueError:
            return None
    return None


def _eq(a: Any, b: Any) -> bool:
    x, y = _num(a), _num(b)
    if x is not None and y is not None and not (isinstance(a, str) and isinstance(b, str)):
        return math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-12)
    return a == b


def _eog_compare(actual: Any, expected: Any, op: str) -> bool:
    """EnterpriseOps-Gym ``_compare_values`` exactly (``benchmark/verifier.py``, Apache-2.0): Python ``==``, ``>``, ``<``
    and ``expected in str(actual)``; a comparison that raises is a failure."""
    try:
        if op == "equals":
            return bool(actual == expected)
        if op == "gt":
            return bool(actual > expected)
        if op == "lt":
            return bool(actual < expected)
        if op == "contains":
            return bool(expected in str(actual))
    except TypeError:
        return False
    return False


def compare(actual: Any, expected: Any, op: str, semantics: str = "crucible") -> bool:
    if semantics == "eog":  # an imported EnterpriseOps-Gym verifier grades as the benchmark does (M11 audit H3 V13, G3 D12)
        return _eog_compare(actual, expected, op)
    if op == "equals":
        return _eq(actual, expected)
    if op == "not_equals":
        return not _eq(actual, expected)
    if op in ("gt", "gte", "lt", "lte"):
        x, y = _num(actual), _num(expected)
        if x is None or y is None:
            return False
        return {"gt": x > y, "gte": x >= y, "lt": x < y, "lte": x <= y}[op]
    if op == "contains":
        return str(expected) in str(actual)
    if op == "in":
        return any(_eq(actual, e) for e in (expected if isinstance(expected, list) else [expected]))
    if op == "rows_equal":
        def canon(rows: Any) -> Counter:
            rows = rows if isinstance(rows, list) else [rows]
            return Counter(json.dumps(r, sort_keys=True, default=str) for r in rows)
        return canon(actual) == canon(expected)
    raise ValueError(f"unknown comparison {op!r} (known: {COMPARISONS})")


@dataclass(frozen=True)
class SqlAssertion:
    id: str
    app: str
    query: str
    expected: Any
    compare: str = "equals"
    description: str = ""
    semantics: str = "crucible"  # "eog": EnterpriseOps-Gym's own comparison rules

    def __post_init__(self) -> None:
        if self.compare not in COMPARISONS:
            raise ValueError(f"{self.id}: unknown comparison {self.compare!r} (known: {COMPARISONS})")
        if not self.query.strip():
            raise ValueError(f"{self.id}: empty query")

    async def evaluate(self, view) -> Verdict:
        r = await view.query(self.app, Query(op="sql", args={"sql": self.query, "max_rows": 10_000}))
        if not r.ok:
            return Verdict(False, {"query": self.query, "error": (r.error or "query failed")[:500]})
        actual = reduce_result(r.value or {})
        ok = compare(actual, self.expected, self.compare, self.semantics)
        return Verdict(ok, {"query": self.query, "expected": self.expected, "actual": actual if _small(actual) else f"{len(actual)} rows",
                            "compare": self.compare})

    def check(self, gate: str, **kw: Any) -> Check:
        async def fn(ctx):
            return await self.evaluate(ctx.params["view"])
        return Check(self.id, gate, fn, provenance=kw.pop("provenance", self.description or None), **kw)


def _small(v: Any) -> bool:
    return not isinstance(v, list) or len(v) <= 20


@dataclass(frozen=True)
class Allow:
    match: dict[str, Any] = field(default_factory=dict)  # empty: any row of the table
    ops: tuple[str, ...] = ("insert", "delete", "update")

    def __post_init__(self) -> None:
        bad = set(self.ops) - {"insert", "delete", "update"}
        if bad:
            raise ValueError(f"unknown ops {sorted(bad)}")

    def admits(self, row: dict[str, Any]) -> bool:
        return all(k in row and _eq(row[k], v) for k, v in self.match.items())


@dataclass(frozen=True)
class Collateral:
    """``allow``: table -> allowances (a change passes if any admits it; ``"any"`` in data = every change to that table).
    ``key``: table -> key columns, to see updates (a delete and an insert with the same key). ``ignore``: tables left
    out entirely (logs, sessions)."""

    id: str
    app: str
    allow: dict[str, tuple[Allow, ...]] = field(default_factory=dict)
    key: dict[str, tuple[str, ...]] = field(default_factory=dict)
    ignore: tuple[str, ...] = ()
    max_report: int = 20

    async def evaluate(self, view) -> Verdict:
        diff = await view.changes(self.app)
        start_cols = view.start.apps[self.app].meta.get("columns", {})
        problems: list[str] = []
        by_table: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for ch in diff.rows:
            if ch.table in self.ignore:
                continue
            by_table.setdefault(ch.table, {"inserted": [], "deleted": []})[ch.change].append(_row_dict(ch.row, start_cols.get(ch.table)))
        for table, ch in sorted(by_table.items()):
            allows = self.allow.get(table, ())
            key = self.key.get(table)
            ins, dels = ch["inserted"], ch["deleted"]
            updates: list[tuple[dict, dict]] = []
            if key:
                index = {_key(r, key): r for r in dels}
                matched = [(index[_key(r, key)], r) for r in ins if _key(r, key) in index]
                updates = matched
                done_b = {id(b) for b, _ in matched}
                done_a = {id(a) for _, a in matched}
                dels = [r for r in dels if id(r) not in done_b]
                ins = [r for r in ins if id(r) not in done_a]
            for op, rows in (("insert", [(None, r) for r in ins]), ("delete", [(r, None) for r in dels]), ("update", updates)):
                for before, after in rows:
                    ok = any(op in a.ops and any(img is not None and a.admits(img) for img in (before, after)) for a in allows)
                    if not ok:
                        shown = after if after is not None else before
                        problems.append(f"{table}: unexpected {op} {json.dumps(_trim(shown), default=str)[:300]}")
        detail = {"tables_changed": diff.tables_changed, "unexpected": problems[: self.max_report], "unexpected_count": len(problems)}
        return Verdict(not problems, detail)

    def check(self, gate: str, **kw: Any) -> Check:
        async def fn(ctx):
            return await self.evaluate(ctx.params["view"])
        return Check(self.id, gate, fn, **kw)


def _row_dict(row: dict[str, Any], cols: list[str] | None) -> dict[str, Any]:
    if "cells" in row and isinstance(row["cells"], list):  # SQLite manifests store cells; the snapshot meta names the columns
        cells = row["cells"]
        names = cols if cols and len(cols) == len(cells) else [f"c{i}" for i in range(len(cells))]
        return dict(zip(names, cells, strict=True))
    return dict(row)


def _key(row: dict[str, Any], key: tuple[str, ...]) -> str:
    return json.dumps([row.get(k) for k in key], default=str)


def _trim(row: dict[str, Any]) -> dict[str, Any]:
    return {k: (v[:60] + "..." if isinstance(v, str) and len(v) > 60 else v) for k, v in list(row.items())[:12]}


# ----------------------------------------------------------------------------------------------- from task data
def assertion_from(d: dict[str, Any], *, app: str | None = None) -> SqlAssertion:
    return SqlAssertion(id=d["id"], app=d.get("app") or app or "", query=d["query"], expected=d.get("expected"),
                        compare=d.get("compare", "equals"), description=d.get("description", ""))


def collateral_from(d: dict[str, Any], *, app: str | None = None) -> Collateral:
    allow: dict[str, tuple[Allow, ...]] = {}
    for table, a in (d.get("allow") or {}).items():
        items = a if isinstance(a, list) else [a]
        allow[table] = tuple(Allow() if x in ("any", None) else Allow(match=dict(x.get("match") or {}), ops=tuple(x.get("ops") or ("insert", "delete", "update")))
                             for x in items)
    return Collateral(id=d.get("id", "collateral"), app=d.get("app") or app or "", allow=allow,
                      key={t: tuple(k) if isinstance(k, list | tuple) else (k,) for t, k in (d.get("key") or {}).items()},
                      ignore=tuple(d.get("ignore") or ()))


def from_enterpriseops(v: dict[str, Any], *, app_of: dict[str, str] | None = None, index: int = 0) -> SqlAssertion:
    """One EnterpriseOps-Gym ``database_state`` verifier -> ``SqlAssertion`` (``gym_name`` routed to an App id)."""
    if v.get("verifier_type") != "database_state":
        raise ValueError(f"only database_state verifiers convert (got {v.get('verifier_type')!r})")
    cfg = v.get("validation_config") or {}
    op = {"greater_than": "gt", "less_than": "lt"}.get(cfg.get("comparison_type", "equals"), cfg.get("comparison_type", "equals"))
    gym = v.get("gym_name") or ""
    name = v.get("name") or f"check {index}"
    return SqlAssertion(id=f"sql/{index:02d}-" + "".join(c if c.isalnum() else "_" for c in name.lower())[:40],
                        app=(app_of or {}).get(gym, gym), query=cfg["query"], expected=cfg.get("expected_value"), compare=op,
                        description=v.get("description") or name, semantics="eog")
