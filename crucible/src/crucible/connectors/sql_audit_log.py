"""``sql_audit_log``: a database's change log -> one Trace per business object (M3 generic connector).

Systems of record keep audit tables ("who changed which row, when, from what to what"): an ERP's invoice history,
a CRM's activity log, a Postgres trigger table. A query selects the rows; ``FieldMap`` says which columns are the
case key, time, actor, operation and object; each row becomes a state event that *writes* ``<table>:<row id>``,
which is what M4's generic attribution needs ("the event that last wrote the object a signal is about").
SQLite needs nothing; Postgres needs psycopg2 (``DATABASE_URL``-style DSN).
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.connectors.file_drop import FieldMap, rows_to_trace
from crucible.core.schema.trace import Trace


def _query(dsn: str, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    if dsn.startswith(("postgres://", "postgresql://")):
        import psycopg2
        import psycopg2.extras

        with psycopg2.connect(dsn) as con, con.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]
    con = sqlite3.connect(f"file:{dsn.removeprefix('sqlite:///')}?mode=ro", uri=True)
    try:
        con.row_factory = sqlite3.Row
        return [dict(r) for r in con.execute(sql, params).fetchall()]
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

    def _load(self, dsn: str, query: str) -> dict[str, list[dict[str, Any]]]:
        f, out = self.fields, {}
        for r in _query(dsn, query):
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
        return RawRecord(ref=ref, payload={"rows": self._rows[ref.key]}, privacy=self.fields.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        return rows_to_trace(raw.payload["rows"], self.fields, case=raw.ref.key, connector=self.name, privacy=raw.privacy)
