"""A witness action sequence derived from an EnterpriseOps-Gym task's expected database state (the dataset ships no gold
actions: its tasks are graded on the final state only, ``benchmark/verifier.py``).

Each ``database_state`` verifier is a SQL query and an expected value. The ones this module can invert are single-table
conjunctions: ``SELECT COUNT(*) [AS x] FROM t [a] WHERE p AND p ...`` expecting 1 (a row like this must exist) or 0 (no
row like this may exist), and ``SELECT col FROM t WHERE p ...`` expecting a scalar (that row exists with ``col`` set).
A predicate is ``col = lit``, ``col IS [NOT] NULL``, ``col <op> lit`` (``>``, ``>=``, ``<``, ``<=``), ``col > col``,
``col LIKE 'lit'`` (``%`` dropped) or ``col IN (lit, ...)``. Joins, subqueries, ``OR``, ``!=`` and ``BETWEEN`` are not
inverted (``Underivable``): such a task still imports and grades, it only has no derived witness.

Derivation: the rows the verifiers describe on one table are merged when their equalities agree (two checks on case
1233 are one row; case_sla rows for SLA 1 and SLA 6 stay two). A row whose primary key is fixed becomes an upsert
(``INSERT ... WHERE NOT EXISTS`` on the key, then ``UPDATE`` of every constrained column); a row without one becomes a
guarded insert (inserted only when no row already satisfies every predicate). A count-0 check becomes a ``DELETE``.
Inequalities get a value that satisfies them: the World's ``now`` for ``IS NOT NULL`` and dates, ``lit +/- 1`` for
numbers. The statements are self-deciding (no reads needed), so the witness is a static list of ``sql_execute`` calls.
It proves the imported World can reach reward 1 through the policy's own tools; it is not a reference policy (it inserts
where a careful agent might have updated, which the verifiers cannot tell apart).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


class Underivable(ValueError):
    """The verifier's query is outside the invertible subset."""


_STR = re.compile(r"'(?:[^']|'')*'")
_COL = r"(?:[A-Za-z_]\w*\.)?([A-Za-z_]\w*)"
_LITX = r"(?:\x00\d+\x00|-?\d+(?:\.\d+)?|NULL|TRUE|FALSE)"
_LIT = f"({_LITX})"
_SELECT = re.compile(r"^\s*SELECT\s+(?P<sel>.+?)\s+FROM\s+(?P<table>[A-Za-z_]\w*)(?:\s+(?:AS\s+)?(?P<alias>[A-Za-z_]\w*))?\s+WHERE\s+(?P<where>.+?)\s*;?\s*$",
                     re.IGNORECASE | re.DOTALL)
_COUNT = re.compile(r"^COUNT\s*\(\s*\*\s*\)(?:\s+AS\s+\w+)?$", re.IGNORECASE)
_P_NOTNULL = re.compile(rf"^{_COL}\s+IS\s+NOT\s+NULL$", re.IGNORECASE)
_P_NULL = re.compile(rf"^{_COL}\s+IS\s+NULL$", re.IGNORECASE)
_P_CMP = re.compile(rf"^{_COL}\s*(==|=|>=|<=|>|<)\s*{_LIT}$", re.IGNORECASE)
_P_COLCMP = re.compile(rf"^{_COL}\s*(>=|<=|>|<)\s*{_COL}$", re.IGNORECASE)
_P_LIKE = re.compile(rf"^{_COL}\s+LIKE\s+{_LIT}$", re.IGNORECASE)
_P_IN = re.compile(rf"^{_COL}\s+IN\s*\(\s*({_LITX}(?:\s*,\s*{_LITX})*)\s*\)$", re.IGNORECASE)
_BLOCK = re.compile(r"\bJOIN\b|\(\s*SELECT\b|\bOR\b|\bBETWEEN\b|\bGROUP\s+BY\b|\bUNION\b", re.IGNORECASE)


@dataclass
class Pred:
    col: str
    op: str  # "=" | "notnull" | ">" | ">=" | "<" | "<=" | "col>" ... (``other`` names the right-hand column)
    value: Any = None
    other: str | None = None


@dataclass
class Want:
    """What one verifier requires of a table: ``exists`` (one matching row) or ``absent`` (none)."""

    table: str
    kind: str
    preds: list[Pred]
    source: str = ""


def _lit(tok: str, strings: list[str]) -> Any:
    if tok.startswith("\x00"):
        return strings[int(tok.strip("\x00"))][1:-1].replace("''", "'")
    up = tok.upper()
    if up == "NULL":
        return None
    if up in ("TRUE", "FALSE"):
        return int(up == "TRUE")
    return float(tok) if "." in tok else int(tok)


def _split_and(where: str) -> list[str]:
    return [p.strip() for p in re.split(r"\bAND\b", where, flags=re.IGNORECASE) if p.strip()]


def parse_verifier(query: str, expected: Any, compare: str = "equals") -> Want:
    """One ``database_state`` verifier -> ``Want`` (raises ``Underivable``)."""
    if compare != "equals":
        raise Underivable(f"comparison {compare!r}")
    if _BLOCK.search(_STR.sub("''", query)):
        raise Underivable("join, subquery, OR, BETWEEN or grouping")
    strings: list[str] = []
    masked = _STR.sub(lambda m: (strings.append(m.group(0)), f"\x00{len(strings) - 1}\x00")[1], " ".join(query.split()))
    m = _SELECT.match(masked)
    if not m or (m["alias"] or "").upper() == "WHERE":
        raise Underivable("not a single-table SELECT ... WHERE")
    preds: list[Pred] = []
    for part in _split_and(m["where"]):
        if mm := _P_NOTNULL.match(part):
            preds.append(Pred(mm[1], "notnull"))
        elif mm := _P_NULL.match(part):
            preds.append(Pred(mm[1], "=", None))
        elif mm := _P_CMP.match(part):
            preds.append(Pred(mm[1], "=" if mm[2] in ("=", "==") else mm[2], _lit(mm[3], strings)))
        elif mm := _P_COLCMP.match(part):
            preds.append(Pred(mm[1], "col" + mm[2], other=mm[3]))
        elif mm := _P_LIKE.match(part):
            v = _lit(mm[2], strings)
            preds.append(Pred(mm[1], "=", v.replace("%", "") if isinstance(v, str) else v))
        elif mm := _P_IN.match(part):
            preds.append(Pred(mm[1], "=", _lit(re.split(r"\s*,\s*", mm[2])[0], strings)))
        else:
            raise Underivable(f"predicate {part.replace(chr(0), '')!r}")
    sel = m["sel"].strip()
    if _COUNT.match(sel):
        if expected in (1, 0) and not isinstance(expected, bool):
            return Want(m["table"], "exists" if expected == 1 else "absent", preds, query)
        raise Underivable(f"COUNT expected {expected!r} (only 0 or 1)")
    if mm := re.match(rf"^{_COL}$", sel):
        return Want(m["table"], "exists", [*preds, Pred(mm[1], "=", expected)], query)
    raise Underivable(f"select list {sel!r}")


def _same(a: Any, b: Any) -> bool:
    if a is None or b is None:
        return a is b
    try:
        return float(a) == float(b) and not (isinstance(a, str) and isinstance(b, str) and a != b)
    except (TypeError, ValueError):
        return str(a) == str(b)


@dataclass
class Row:
    table: str
    eq: dict[str, Any] = field(default_factory=dict)
    rel: list[Pred] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)

    def agrees(self, preds: list[Pred]) -> bool:
        return all(_same(self.eq[p.col], p.value) for p in preds if p.op == "=" and p.col in self.eq)

    def add(self, w: Want) -> None:
        for p in w.preds:
            if p.op == "=":
                self.eq[p.col] = p.value
            else:
                self.rel.append(p)
        self.sources.append(w.source)


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _bump(now: str, lit: Any, op: str) -> Any:
    """A value satisfying ``col <op> lit``: numbers step by one; text (dates) prefers ``now`` when it qualifies."""
    if isinstance(lit, int | float):
        return lit + 1 if op in (">", ">=") else lit - 1
    if op in (">", ">="):
        return now if now > str(lit) else str(lit) + " 23:59:59"
    return "" if op == "<" else str(lit)


def _choose(row: Row, now: str) -> dict[str, Any]:
    """Values for every column the row's inequalities constrain (and the columns they compare against)."""
    vals: dict[str, Any] = {}
    later = (datetime.fromisoformat(now) + timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S")
    for p in row.rel:
        if p.col in row.eq:
            continue
        if p.op == "notnull":
            vals.setdefault(p.col, now)
        elif p.op.startswith("col"):
            base = row.eq.get(p.other, vals.get(p.other))
            if base is None:
                vals[p.other] = now
                base = now
            vals[p.col] = later if p.op in ("col>", "col>=") and base == now else _bump(now, base, p.op[3:])
        else:
            vals[p.col] = _bump(now, p.value, p.op)
    return vals


def _cond(p: Pred) -> tuple[str, list[Any]]:
    if p.op == "=":
        return (f"{_q(p.col)} IS NULL", []) if p.value is None else (f"{_q(p.col)} = ?", [p.value])
    if p.op == "notnull":
        return f"{_q(p.col)} IS NOT NULL", []
    if p.op.startswith("col"):
        return f"{_q(p.col)} {p.op[3:]} {_q(p.other or '')}", []
    return f"{_q(p.col)} {p.op} ?", [p.value]


def _where(preds: list[Pred]) -> tuple[str, list[Any]]:
    parts, params = [], []
    for p in preds:
        s, a = _cond(p)
        parts.append(s)
        params += a
    return " AND ".join(parts) or "1", params


def derive(verifiers: list[tuple[str, Any, str]], *, pk: dict[str, str], now: str) -> list[tuple[str, list[Any]]]:
    """``verifiers``: (query, expected, comparison) for one App; ``pk``: table -> primary-key column; ``now``: the World's
    pinned ``now`` (``YYYY-MM-DD HH:MM:SS``). Returns ``(sql, params)`` statements (raises ``Underivable``)."""
    wants = [parse_verifier(q, e, c) for q, e, c in verifiers]
    stmts: list[tuple[str, list[Any]]] = []
    for w in wants:
        if w.kind == "absent":
            cond, params = _where(w.preds)
            stmts.append((f"DELETE FROM {_q(w.table)} WHERE {cond}", params))
    rows: list[Row] = []
    for w in wants:
        if w.kind != "exists":
            continue
        row = next((r for r in rows if r.table == w.table and r.agrees(w.preds)), None)
        if row is None:
            row = Row(w.table)
            rows.append(row)
        row.add(w)
    for row in rows:
        chosen = _choose(row, now)
        key = pk.get(row.table)
        if key and key in row.eq and row.eq[key] is not None:
            kv = row.eq[key]
            stmts.append((f"INSERT INTO {_q(row.table)} ({_q(key)}) SELECT ? WHERE NOT EXISTS (SELECT 1 FROM {_q(row.table)} WHERE {_q(key)} = ?)", [kv, kv]))
            sets = {**{c: v for c, v in row.eq.items() if c != key}, **chosen}
            if sets:
                stmts.append((f"UPDATE {_q(row.table)} SET {', '.join(f'{_q(c)} = ?' for c in sets)} WHERE {_q(key)} = ?", [*sets.values(), kv]))
            continue
        values = {**row.eq, **chosen}
        if not values:
            raise Underivable(f"{row.table}: no column values to insert")
        cond, params = _where([Pred(c, "=", v) for c, v in row.eq.items()] + row.rel)
        cols = ", ".join(_q(c) for c in values)
        sql = f"INSERT INTO {_q(row.table)} ({cols}) SELECT {', '.join('?' * len(values))} WHERE NOT EXISTS (SELECT 1 FROM {_q(row.table)} WHERE {cond})"
        stmts.append((sql, [*values.values(), *params]))
    return stmts
