"""``sql_audit_log``: a database's change log -> one Trace per business object (M3 generic connector).

Systems of record keep audit tables ("who changed which row, when, from what to what"): an ERP's invoice history,
a CRM's activity log, a Postgres trigger table. A query selects the rows; ``FieldMap`` says which columns are the
case key, time, actor, operation and object; each row becomes a state event that *writes* ``<table>:<row id>``,
which is what M4's generic attribution needs ("the event that last wrote the object a signal is about").
SQLite needs nothing; Postgres needs psycopg2 (``DATABASE_URL``-style DSN).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Iterator
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.connectors.file_drop import FieldMap, rows_to_trace
from crucible.core.schema.trace import Trace


def _single_select(sql: str) -> None:
    """The audit query reads: one SELECT / WITH statement (an UPDATE ... RETURNING or "SELECT 1; DELETE ..." ran and,
    on Postgres, was committed)."""
    body = re.sub(r"--[^\n]*|/\*.*?\*/|'(?:[^']|'')*'|\"(?:[^\"]|\"\")*\"", " ", sql, flags=re.DOTALL).strip().rstrip(";").strip()
    if ";" in body or not re.match(r"(?is)^(?:select|with)\b", body):
        raise ValueError("sql_audit_log runs one SELECT (or WITH ... SELECT) statement")


def _query(dsn: str, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    _single_select(sql)
    if dsn.startswith(("postgres://", "postgresql://", "postgresql+psycopg2://")):
        import psycopg2
        import psycopg2.extras

        # a read-only session, a server-side cursor (no whole-table fetch into memory at once), always closed: the old
        # ``with psycopg2.connect()`` committed and never closed the connection
        con = psycopg2.connect(dsn.replace("postgresql+psycopg2://", "postgresql://", 1))
        try:
            con.set_session(readonly=True, autocommit=False)
            with con.cursor(name="crucible_audit", cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.itersize = 5000
                cur.execute(sql, params)
                rows = [dict(r) for r in cur]
            con.rollback()
            return rows
        finally:
            con.close()
    from urllib.parse import quote

    # quoted: a '#' or '?' in the path ended the file name and silently opened another database
    con = sqlite3.connect(f"file:{quote(dsn.removeprefix('sqlite:///'))}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA query_only = ON")
        return [{k: (v.isoformat() if hasattr(v, "isoformat") else v) for k, v in dict(r).items()} for r in con.execute(sql, params)]
    finally:
        con.close()


class SqlAuditLogConnector:
    """scope: {"dsn": "path/to.db" | "postgresql://...", "query": "SELECT ... FROM audit_log"} (the query must
    return the columns named in the FieldMap; ``writes`` may be a column or derived from ``table`` + ``row_id``)."""

    name = "sql_audit_log"

    def __init__(self, fields: FieldMap | None = None, *, table_col: str = "table_name", row_col: str = "row_id"):
        self.fields = fields or FieldMap(kind="kind", type="op")
        self.table_col, self.row_col = table_col, row_col
        self._rows: dict[str, list[dict[str, Any]]] = {}
        self.skipped = 0  # rows with a NULL case key (they used to form one "None" trace of unrelated objects)

    def _load(self, dsn: str, query: str) -> dict[str, list[dict[str, Any]]]:
        f, out = self.fields, {}
        self.skipped = 0
        for r in _query(dsn, query):
            if r.get(f.case) in (None, ""):
                self.skipped += 1
                continue
            if not r.get(f.writes) and r.get(self.table_col) is not None and r.get(self.row_col) is not None:
                r[f.writes] = [f"{r[self.table_col]}:{r[self.row_col]}"]
            if not r.get(f.kind):
                r[f.kind] = "state"
            out.setdefault(str(r[f.case]), []).append(r)
        return out

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        self._rows = self._load(scope["dsn"], scope["query"])
        for case in sorted(self._rows):
            yield SourceRef(connector=self.name, key=case, meta={"dsn": scope["dsn"], "query": scope["query"]})

    def fetch(self, ref: SourceRef) -> RawRecord:
        if ref.key not in self._rows:  # a ref from an earlier run: query again
            self._rows = self._load(ref.meta["dsn"], ref.meta["query"])
        if ref.key not in self._rows:
            raise LookupError(f"case {ref.key!r} is no longer returned by the audit query")
        return RawRecord(ref=ref, payload={"rows": self._rows[ref.key]}, privacy=self.fields.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        return rows_to_trace(raw.payload["rows"], self.fields, case=raw.ref.key, connector=self.name, privacy=raw.privacy)
