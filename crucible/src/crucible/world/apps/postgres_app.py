"""Postgres App: a database server as a system of record (ERP, CRM, ledger at enterprise scale).

Runs the stock ``postgres:16-alpine`` image (agentenv's ServiceDB default) in its own container on
the App's Docker daemon; state is the ``postgres`` adapter, executed with ``psql`` / ``pg_dump`` inside
that container. Tool surface:
- generic: ``list_tables``, ``describe_table``, ``sql_query`` (read-only transaction) and, with
  ``sql_write``, ``sql_execute``;
- or a pack's Python ``tools_module`` served over MCP with ``DATABASE_URL`` set (the convention
  agentenv's MCP servers use), so domain tools (``approve_invoice`` ...) sit on a real database.
Params: ``image``, ``database``, ``user``, ``password``, ``schema``, ``schema_sql`` / ``seed_sql``
(host files), ``sql_write``, ``tools_module``, ``docker_host``.
"""

from __future__ import annotations

import asyncio
import shlex
import time
import uuid
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.exec import DockerExec, _run_subprocess
from crucible.world.state.postgres import PostgresAdapter, _ident

_S = {"type": "string"}


class PostgresApp(App):
    template = "common/postgres_app"
    PATH_PARAMS = ()

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        if not isinstance(self.state, PostgresAdapter):
            raise ValueError(f"{self.id}: common/postgres_app needs state.adapter = postgres")
        p = spec.params
        self.container = f"crucible-pg-{spec.id}-{uuid.uuid4().hex[:8]}"
        self.docker_host = p.get("docker_host")
        self.image = p.get("image", "postgres:16-alpine")
        self.password = p.get("password", "crucible")
        self.host_port: int | None = None
        self.state.exec = DockerExec(self.container, docker_host=self.docker_host)
        self._server = None
        self._client = None
        self._specs: list[ToolSpec] | None = None

    def _docker_env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    @property
    def database_url(self) -> str:
        # libpq "options" puts the App's schema first on the search path for every client
        return (f"postgresql://{self.state.user}:{self.password}@127.0.0.1:{self.host_port}/{self.state.db}"
                f"?options=-csearch_path%3D{self.state.schema}")

    async def provision(self) -> None:
        argv = ["docker", "run", "-d", "--rm", "--name", self.container, "-e", f"POSTGRES_USER={self.state.user}",
                "-e", f"POSTGRES_PASSWORD={self.password}", "-e", f"POSTGRES_DB={self.state.db}", "-p", "127.0.0.1::5432",
                "--label", "crucible.app=postgres", self.image]
        out = await _run_subprocess(argv, cwd=None, env=self._docker_env(), timeout_s=300, stdin=None)
        if not out.ok:
            raise RuntimeError(f"cannot start {self.image}: {out.stderr[-500:]}")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            r = await self.state.exec.run(f"pg_isready -U {self.state.user} -d {self.state.db} -h 127.0.0.1 && psql -U {self.state.user} -d {self.state.db} -Atc 'select 1'", timeout_s=20)
            if r.ok and r.stdout.strip().endswith("1"):
                break
            await asyncio.sleep(0.5)
        else:
            raise TimeoutError(f"postgres in {self.container} not ready")
        port = await _run_subprocess(["docker", "port", self.container, "5432/tcp"], cwd=None, env=self._docker_env(), timeout_s=30, stdin=None)
        self.host_port = int(port.stdout.strip().splitlines()[0].rsplit(":", 1)[1])
        if self.state.schema != "public":
            await self.state.psql(f"CREATE SCHEMA IF NOT EXISTS {_ident(self.state.schema)};")
        for key in ("schema_sql", "seed_sql"):
            if self.spec.params.get(key):
                src = self.ctx.host_seed(self.spec.params[key])
                remote = f"/tmp/.crucible_{key}.sql"
                await self.state.exec.push_file(src, remote)
                path = f"SET search_path TO {_ident(self.state.schema)};"
                r = await self.state.exec.run(
                    f"psql -X -q -v ON_ERROR_STOP=1 -U {self.state.user} -d {self.state.db} -c {shlex.quote(path)} -f {remote}", timeout_s=600)
                if not r.ok:
                    raise RuntimeError(f"{key} failed in {self.id}: {(r.stderr or r.stdout)[-800:]}")

    async def start(self) -> None:
        mod = self.spec.params.get("tools_module")
        if not mod:
            return
        from crucible.world.packaging.sidecar import McpClient, McpServerProcess

        path = Path(self.ctx.resolve(mod)) if not Path(mod).is_absolute() else Path(mod)
        if not path.exists():
            path = self.ctx.host_seed(mod)
        self._server = McpServerProcess(name=path.stem, tools_path=path, log_path=self.ctx.host_workdir / f"mcp_{self.id}.log",
                                        env={"DATABASE_URL": self.database_url, "DB_SCHEMA": self.state.schema})
        await self._server.start()
        self._client = McpClient(self._server.url)
        self._specs = await self._client.list_tools(self.id)

    async def stop(self) -> None:
        if self._server is not None:
            await self._server.stop()
            self._server = None
        await _run_subprocess(["docker", "rm", "-f", self.container], cwd=None, env=self._docker_env(), timeout_s=60, stdin=None)

    async def list_tools(self) -> list[ToolSpec]:
        if self._specs is not None:
            return self._specs
        tools = [
            ToolSpec(app=self.id, name="list_tables", read_only=True, description="List tables with row counts.", input_schema={"type": "object", "properties": {}}),
            ToolSpec(app=self.id, name="describe_table", read_only=True, description="Columns and types of a table.",
                     input_schema={"type": "object", "properties": {"table": _S}, "required": ["table"]}),
            ToolSpec(app=self.id, name="sql_query", read_only=True, description="Run a read-only SQL query (PostgreSQL).",
                     input_schema={"type": "object", "properties": {"sql": _S, "max_rows": {"type": "integer"}}, "required": ["sql"]}),
        ]
        if self.spec.params.get("sql_write"):
            tools.append(ToolSpec(app=self.id, name="sql_execute", description="Execute a SQL statement that may modify data (PostgreSQL).",
                                  input_schema={"type": "object", "properties": {"sql": _S}, "required": ["sql"]}))
        return tools

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        if self._client is not None:
            return await self._client.call(self.id, name, args, call_id=call_id)
        return await super().call(name, args, actor=actor, call_id=call_id)

    async def tool_list_tables(self) -> Any:
        return await self.state.q_tables()

    async def tool_describe_table(self, table: str) -> Any:
        r = await self.state.q_sql(f"SELECT column_name, data_type, is_nullable FROM information_schema.columns "
                                   f"WHERE table_schema = '{self.state.schema}' AND table_name = '{table.replace(chr(39), '')}' ORDER BY ordinal_position")
        if not r["rows"]:
            raise ToolError(f"no table {table!r}")
        return r

    async def tool_sql_query(self, sql: str, max_rows: int = 200) -> Any:
        try:
            return await self.state.q_sql(sql, max_rows=min(max_rows, 2000))
        except Exception as e:
            raise ToolError(str(e).split("psql failed: ")[-1][:1500]) from e

    async def tool_sql_execute(self, sql: str) -> Any:
        if not self.spec.params.get("sql_write"):
            raise ToolError("writes are disabled")
        try:
            out = await self.state.psql(f"SET search_path TO {_ident(self.state.schema)}; {sql}")
        except Exception as e:
            raise ToolError(str(e).split("psql failed: ")[-1][:1500]) from e
        return out.strip() or "ok"
