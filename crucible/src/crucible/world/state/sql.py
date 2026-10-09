"""SQLite State Adapter (``sql``).

Params: ``path`` (db file in the exec), ``include`` / ``exclude`` (table globs).
Snapshots use SQLite's online backup API (consistent even while App servers hold connections);
restore copies *into* the live database so every open connection sees the restored state.
Postgres-backed Apps (agentenv ServiceDB) are reached through ``interop.agentenv``.
"""

from __future__ import annotations

import json
from typing import Any

from crucible.core.schema.state import QueryResult, RowChange, StateDiff, StateRef
from crucible.world.state.base import SnapshotMismatch, StateAdapter, snippet

SQL_SNIPPET = snippet("sql")


class SqliteAdapter(StateAdapter):
    kind = "sql"
    OPS = {
        "sql": "read-only SQL: {sql, params, max_rows}",
        "scalar": "first column of first row: {sql, params}",
        "count": "row count of a table, optional where: {table, where, params}",
        "tables": "table names with row counts",
        "dump": "canonical per-table hashes (no rows)",
    }

    def _args(self, **extra: Any) -> dict[str, Any]:
        path = self.params.get("path")
        if not path:
            raise ValueError(f"sql adapter for app {self.app!r} needs params.path")
        return {"path": path, "include": list(self.params.get("include", [])), "exclude": list(self.params.get("exclude", [])), **extra}

    async def _snip(self, op: str, **extra: Any) -> dict[str, Any]:
        return await self.exec.run_python(SQL_SNIPPET, self._args(op=op, **extra))

    async def fingerprint(self) -> str:
        return (await self._snip("dump"))["fingerprint"]

    async def snapshot(self) -> StateRef:
        out = self._tmp_remote(".sqlite")
        d = await self._snip("backup", out=out)
        if d.get("missing"):  # no database at all is a valid (empty) state
            return StateRef(app=self.app, adapter=self.kind, fingerprint=d["fingerprint"], blob=None, meta={"missing": True, "tables": {}})
        key, size = await self._pull_to_store(out)
        rows = await self._snip("dump", rows=True)
        if rows["fingerprint"] != d["fingerprint"]:
            # The db changed between backup and dump (a live writer): retry once, else fail loudly.
            d = await self._snip("backup", out=out)
            key, size = await self._pull_to_store(out)
            rows = await self._snip("dump", rows=True)
            if rows["fingerprint"] != d["fingerprint"]:
                raise SnapshotMismatch(f"{self.app}: database kept changing during snapshot")
        manifest_key = self.store.put_bytes(json.dumps({t: v["rows"] for t, v in rows["tables"].items()}, separators=(",", ":")).encode())
        return StateRef(
            app=self.app,
            adapter=self.kind,
            fingerprint=d["fingerprint"],
            blob=key,
            size_bytes=size,
            meta={"tables": {t: v["count"] for t, v in d["tables"].items()}, "manifest": manifest_key},
        )

    async def restore(self, ref: StateRef) -> None:
        if ref.adapter != self.kind:
            raise ValueError(f"cannot restore {ref.adapter} ref with sql adapter")
        if ref.blob is None:
            if not ref.meta.get("missing"):
                raise ValueError("sql ref has no payload")
            d = await self._snip("restore", src=None)
            return
        # Always restore from a private copy: SQLite may create -shm/-journal files next to the source.
        src = await self._push_from_store(ref.blob, ".sqlite", copy=True)
        try:
            d = await self._snip("restore", src=src)
        finally:
            await self.exec.run(f"rm -f {src} {src}-journal {src}-wal {src}-shm")
        if d["fingerprint"] != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: restored db {d['fingerprint'][:12]} != snapshot {ref.fingerprint[:12]}")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        ra, rb = self._rows_of(a), self._rows_of(b)
        changes: list[RowChange] = []
        tables_changed: list[str] = []
        for t in sorted(set(ra) | set(rb)):
            xa, xb = ra.get(t, []), rb.get(t, [])
            if xa == xb:
                continue
            tables_changed.append(t)
            sa, sb = _multiset(xa), _multiset(xb)
            for row, n in sorted(sa.items()):
                for _ in range(max(0, n - sb.get(row, 0))):
                    changes.append(RowChange(table=t, change="deleted", row={"cells": json.loads(row)}))
            for row, n in sorted(sb.items()):
                for _ in range(max(0, n - sa.get(row, 0))):
                    changes.append(RowChange(table=t, change="inserted", row={"cells": json.loads(row)}))
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint, rows=changes, tables_changed=tables_changed)

    def _rows_of(self, ref: StateRef) -> dict[str, list[str]]:
        if ref.meta.get("missing"):
            return {}
        key = ref.meta.get("manifest")
        if not key:
            raise ValueError("ref carries no row manifest; diff needs captured snapshots")
        return self.store.get_json(key)

    # ---- query ops ----
    async def q_sql(self, sql: str, params: list[Any] | None = None, max_rows: int = 1000) -> Any:
        return await self._snip("query", sql=sql, params=params or [], max_rows=max_rows)

    async def q_scalar(self, sql: str, params: list[Any] | None = None) -> Any:
        r = await self._snip("query", sql=sql, params=params or [], max_rows=1)
        if not r["rows"]:
            return QueryResult(ok=True, value=None)
        return r["rows"][0][0]

    async def q_count(self, table: str, where: str | None = None, params: list[Any] | None = None) -> Any:
        if not table.replace("_", "").isalnum():
            raise ValueError(f"bad table name {table!r}")
        sql = f'SELECT COUNT(*) FROM "{table}"' + (f" WHERE {where}" if where else "")
        r = await self._snip("query", sql=sql, params=params or [], max_rows=1)
        return r["rows"][0][0]

    async def q_tables(self) -> Any:
        d = await self._snip("dump")
        return {t: v["count"] for t, v in d["tables"].items()}

    async def q_dump(self) -> Any:
        return await self._snip("dump")


def _multiset(rows: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r] = out.get(r, 0) + 1
    return out
