"""Postgres State Adapter (``postgres``): the state of a database-server App.

Runs through an Exec into the Postgres container (``psql`` / ``pg_dump`` / ``pg_restore`` — no Python
needed there). Canonical form, computed inside Postgres:
- per table: sha256 over the table's rows as ``row_to_json`` text, sorted (row order and storage
  layout never matter; values do), plus the row count;
- schema: columns (name, type, nullability, default), constraints (``pg_get_constraintdef``),
  indexes (``indexdef``) and sequence values (they are state: the next id an INSERT gets).
(Inside an aggregate, ``ORDER BY 1`` orders by the constant 1, not by column position: every ordering
below names its expressions explicitly, or restored databases would fingerprint differently.)
Snapshots are ``pg_dump -Fc`` archives of the App's schema; restore drops the schema and replays the
archive with ``pg_restore``, then verifies the fingerprint. Params: ``database``, ``user``,
``schema`` (default ``public``).
"""

from __future__ import annotations

import json
import shlex
from decimal import Decimal
from typing import Any, ClassVar

from crucible.core.ids import sha256_text
from crucible.core.schema.state import QueryResult, RowChange, StateDiff, StateRef
from crucible.world.exec import ExecError
from crucible.world.state.base import SnapshotMismatch, StateAdapter


def _lit(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"


def _ident(s: str) -> str:
    return '"' + s.replace('"', '""') + '"'


class PostgresAdapter(StateAdapter):
    kind = "postgres"
    OPS: ClassVar[dict[str, str]] = {
        "sql": "read-only SQL (runs in a READ ONLY transaction): {sql, max_rows} -> {columns, rows}",
        "scalar": "first column of first row: {sql}",
        "count": "row count: {table, where}",
        "tables": "tables with row counts",
        "dump": "canonical per-table hashes and schema hash",
    }

    @property
    def db(self) -> str:
        return self.params.get("database", "crucible")

    @property
    def user(self) -> str:
        return self.params.get("user", "crucible")

    @property
    def schema(self) -> str:
        return self.params.get("schema", "public")

    async def psql_json(self, sql: str, *, read_only: bool = True, timeout_s: float = 300) -> Any:
        """Run SQL whose single result cell is JSON; return it parsed."""
        pre = "SET default_transaction_read_only = on; " if read_only else ""
        cmd = f"psql -X -A -t -q -v ON_ERROR_STOP=1 -U {shlex.quote(self.user)} -d {shlex.quote(self.db)} -c {shlex.quote(pre + sql)}"
        out = await self.exec.run(cmd, timeout_s=timeout_s)
        if not out.ok:
            raise ExecError(f"psql failed: {(out.stderr or out.stdout)[-1500:]}", out)
        text = out.stdout.strip()
        # NUMERIC must stay exact (money): parse JSON numbers with a fraction as Decimal, not float
        return json.loads(text, parse_float=Decimal) if text else None

    async def psql(self, sql: str, *, timeout_s: float = 300) -> str:
        cmd = f"psql -X -q -v ON_ERROR_STOP=1 -U {shlex.quote(self.user)} -d {shlex.quote(self.db)} -c {shlex.quote(sql)}"
        out = await self.exec.run(cmd, timeout_s=timeout_s)
        if not out.ok:
            raise ExecError(f"psql failed: {(out.stderr or out.stdout)[-1500:]}", out)
        return out.stdout

    async def _tables(self) -> list[str]:
        q = (f"SELECT coalesce(json_agg(c.relname ORDER BY c.relname), '[]') FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
             f"WHERE n.nspname = {_lit(self.schema)} AND c.relkind IN ('r','p')")
        return await self.psql_json(q)

    async def dump(self, *, rows: bool = False) -> dict[str, Any]:
        s = _lit(self.schema)
        schema_q = (
            "SELECT json_build_object("
            f"'columns', (SELECT coalesce(json_agg(json_build_array(table_name, column_name, data_type, is_nullable, column_default) ORDER BY table_name, ordinal_position), '[]') FROM information_schema.columns WHERE table_schema = {s}),"
            f"'constraints', (SELECT coalesce(json_agg(json_build_array(conrelid::regclass::text, conname, pg_get_constraintdef(oid)) ORDER BY conrelid::regclass::text, conname), '[]') FROM pg_constraint WHERE connamespace = {s}::regnamespace),"
            f"'indexes', (SELECT coalesce(json_agg(json_build_array(indexname, indexdef) ORDER BY indexname), '[]') FROM pg_indexes WHERE schemaname = {s}),"
            f"'sequences', (SELECT coalesce(json_agg(json_build_array(sequencename, last_value) ORDER BY sequencename), '[]') FROM pg_sequences WHERE schemaname = {s}))"
        )
        schema = await self.psql_json(schema_q)
        tables: dict[str, Any] = {}
        for t in await self._tables():
            fq = f"{_ident(self.schema)}.{_ident(t)}"
            agg = "json_agg(r ORDER BY r)" if rows else "NULL"
            q = (f"SELECT json_build_object('n', count(*), 'h', encode(sha256(convert_to(coalesce(string_agg(r, E'\\n' ORDER BY r), ''), 'UTF8')), 'hex'), "
                 f"'rows', {agg}) FROM (SELECT row_to_json(x)::text AS r FROM {fq} x) s")
            v = await self.psql_json(q)
            tables[t] = {"count": v["n"], "rows_hash": v["h"], **({"rows": v["rows"] or []} if rows else {})}
        schema_hash = sha256_text(json.dumps(schema, sort_keys=True, default=str))
        fp = sha256_text(json.dumps({"schema": schema_hash, "tables": {t: v["rows_hash"] for t, v in tables.items()}}, sort_keys=True))
        return {"fingerprint": fp, "schema_hash": schema_hash, "tables": tables}

    async def fingerprint(self) -> str:
        return (await self.dump())["fingerprint"]

    async def snapshot(self) -> StateRef:
        remote = self._tmp_remote(".pgdump")
        out = await self.exec.run(f"pg_dump -Fc -U {shlex.quote(self.user)} -d {shlex.quote(self.db)} -n {shlex.quote(self.schema)} -f {remote}", timeout_s=1800)
        if not out.ok:
            raise ExecError(f"pg_dump failed: {(out.stderr or out.stdout)[-800:]}", out)
        d = await self.dump(rows=True)
        key, size = await self._pull_to_store(remote)
        manifest = self.store.put_bytes(json.dumps({t: v["rows"] for t, v in d["tables"].items()}, separators=(",", ":")).encode())
        return StateRef(app=self.app, adapter=self.kind, fingerprint=d["fingerprint"], blob=key, size_bytes=size,
                        meta={"tables": {t: v["count"] for t, v in d["tables"].items()}, "manifest": manifest, "schema_hash": d["schema_hash"]})

    async def restore(self, ref: StateRef) -> None:
        if ref.adapter != self.kind or ref.blob is None:
            raise ValueError("cannot restore this ref with the postgres adapter")
        remote = await self._push_from_store(ref.blob, ".pgdump", copy=True)
        try:
            await self.psql(f"DROP SCHEMA IF EXISTS {_ident(self.schema)} CASCADE;")
            out = await self.exec.run(f"pg_restore -U {shlex.quote(self.user)} -d {shlex.quote(self.db)} --no-owner {remote}", timeout_s=1800)
            if not out.ok:
                raise ExecError(f"pg_restore failed: {(out.stderr or out.stdout)[-800:]}", out)
        finally:
            await self.exec.run(f"rm -f {remote}")
        fp = await self.fingerprint()
        if fp != ref.fingerprint:
            raise SnapshotMismatch(f"{self.app}: restored postgres {fp[:12]} != {ref.fingerprint[:12]}")

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        ra = self.store.get_json(a.meta["manifest"])
        rb = self.store.get_json(b.meta["manifest"])
        changes: list[RowChange] = []
        changed: list[str] = []
        for t in sorted(set(ra) | set(rb)):
            xa, xb = ra.get(t, []), rb.get(t, [])
            if xa == xb:
                continue
            changed.append(t)
            sa, sb = _count(xa), _count(xb)
            for r, n in sorted(sa.items()):
                changes += [RowChange(table=t, change="deleted", row=json.loads(r))] * max(0, n - sb.get(r, 0))
            for r, n in sorted(sb.items()):
                changes += [RowChange(table=t, change="inserted", row=json.loads(r))] * max(0, n - sa.get(r, 0))
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint, rows=changes, tables_changed=changed,
                         detail={"schema_changed": a.meta.get("schema_hash") != b.meta.get("schema_hash")})

    # ---- query ops ----
    async def q_sql(self, sql: str, max_rows: int = 1000) -> Any:
        body = sql.strip().rstrip(";")
        rows = await self.psql_json(f"SELECT coalesce(json_agg(t), '[]') FROM (SELECT * FROM ({body}) q LIMIT {int(max_rows) + 1}) t")
        cols = list(rows[0].keys()) if rows else []
        return {"columns": cols, "rows": [[r[c] for c in cols] for r in rows[:max_rows]], "truncated": len(rows) > max_rows}

    async def q_scalar(self, sql: str) -> Any:
        r = await self.q_sql(sql, max_rows=1)
        return r["rows"][0][0] if r["rows"] else QueryResult(ok=True, value=None)

    async def q_count(self, table: str, where: str | None = None) -> Any:
        return await self.psql_json(f"SELECT count(*) FROM {_ident(self.schema)}.{_ident(table)}" + (f" WHERE {where}" if where else ""))

    async def q_tables(self) -> Any:
        return {t: v["count"] for t, v in (await self.dump())["tables"].items()}

    async def q_dump(self) -> Any:
        return await self.dump()


def _count(rows: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in rows:
        out[r] = out.get(r, 0) + 1
    return out
