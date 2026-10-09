"""SQLite-backed App: an enterprise system of record (ERP, ticketing, ledger, ...).

Tool surface, in order of preference:
1. ``tools_module``: a Python file of plain public functions (the MiMo ``tools/<app>.py`` format, also
   what AWM-style generated apps reduce to). Loaded with MiMo's own loader; transport ``inprocess``
   (default) or ``mcp`` (served by the vendored MiMo ``mcp_http`` over streamable HTTP).
2. ``sql_tools: true``: generic ``sql_query`` (read-only) and, if ``sql_write`` is true, ``sql_execute``.

Params: ``tools_module``, ``transport``, ``schema_sql`` / ``seed_db`` (host files copied in at
provision), ``sql_tools``, ``sql_write``, ``env`` (extra env for the server process).
State: ``sql`` adapter with ``path``.
"""

from __future__ import annotations

import shlex
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.state.base import snippet

SQL_SNIPPET = snippet("sql")


class SqliteApp(App):
    template = "common/sqlite_app"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self._module = None
        self._server = None
        self._client = None
        self._specs: list[ToolSpec] | None = None

    @property
    def db_path(self) -> str:
        return self.state_params["path"]

    async def provision(self) -> None:
        p = self.spec.params
        if p.get("seed_db"):
            src = self.ctx.host_seed(p["seed_db"])
            await self.ctx.exec.run(f"mkdir -p {shlex.quote(str(Path(self.db_path).parent))}")
            await self.ctx.exec.push_file(src, self.db_path)
        elif p.get("schema_sql"):
            sql = self.ctx.host_seed(p["schema_sql"]).read_text()
            with tempfile.TemporaryDirectory() as d:
                local = Path(d) / "seed.db"
                con = sqlite3.connect(local)
                con.executescript(sql)
                con.commit()
                con.close()
                await self.ctx.exec.run(f"mkdir -p {shlex.quote(str(Path(self.db_path).parent))}")
                await self.ctx.exec.push_file(local, self.db_path)

    def _module_path(self) -> str | None:
        m = self.spec.params.get("tools_module")
        return self.ctx.resolve(m) if m else None

    async def start(self) -> None:
        mod = self._module_path()
        if not mod:
            return
        transport = self.spec.params.get("transport", "inprocess")
        if not self.ctx.exec.is_local:
            raise RuntimeError("tools_module Apps need a local exec here; container worlds serve modules through interop backends")
        if transport == "inprocess":
            from crucible.world.packaging.mcp_tools import InProcessToolModule

            self._module = InProcessToolModule(self.id, mod)
            self._specs = self._module.specs()
        elif transport == "mcp":
            from crucible.world.packaging.sidecar import McpClient, McpServerProcess

            self._server = McpServerProcess(
                name=self.spec.params.get("server_name", Path(mod).stem), tools_path=Path(mod),
                log_path=self.ctx.host_workdir / f"mcp_{self.id}.log", env=dict(self.spec.params.get("env", {})),
            )
            await self._server.start()
            self._client = McpClient(self._server.url)
            self._specs = await self._client.list_tools(self.id)
        else:
            raise ValueError(f"unknown transport {transport!r}")

    @property
    def mcp_url(self) -> str | None:
        return self._server.url if self._server else None

    async def stop(self) -> None:
        if self._server:
            await self._server.stop()
            self._server = None
        if self._module:
            self._module.unload()
            self._module = None

    async def list_tools(self) -> list[ToolSpec]:
        if self._specs is not None:
            return self._specs
        tools: list[ToolSpec] = []
        if self.spec.params.get("sql_tools"):
            tools.append(ToolSpec(app=self.id, name="sql_query", read_only=True, description="Run a read-only SQL query (SQLite dialect).",
                                  input_schema={"type": "object", "properties": {"sql": {"type": "string"}, "max_rows": {"type": "integer"}}, "required": ["sql"]}))
            if self.spec.params.get("sql_write"):
                tools.append(ToolSpec(app=self.id, name="sql_execute", description="Execute a SQL statement that may modify data.",
                                      input_schema={"type": "object", "properties": {"sql": {"type": "string"}, "params": {"type": "array"}}, "required": ["sql"]}))
        return tools

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        if self._module is not None:
            return await self._module.call(name, args, call_id=call_id)
        if self._client is not None:
            return await self._client.call(self.id, name, args, call_id=call_id)
        return await super().call(name, args, actor=actor, call_id=call_id)

    async def tool_sql_query(self, sql: str, max_rows: int = 200) -> Any:
        try:
            r = await self.ctx.exec.run_python(SQL_SNIPPET, {"op": "query", "path": self.db_path, "sql": sql, "max_rows": min(max_rows, 2000)})
        except Exception as e:
            raise ToolError(str(e).split("python snippet failed: ")[-1]) from e
        return {"columns": r["columns"], "rows": r["rows"], "truncated": r["truncated"]}

    async def tool_sql_execute(self, sql: str, params: list[Any] | None = None) -> Any:
        if not self.spec.params.get("sql_write"):
            raise ToolError("writes are disabled")
        code = (
            "import sqlite3\n"
            "def main(a):\n"
            "    con = sqlite3.connect(a['path'])\n"
            "    try:\n"
            "        cur = con.execute(a['sql'], a.get('params') or [])\n"
            "        con.commit()\n"
            "        return {'rowcount': cur.rowcount, 'lastrowid': cur.lastrowid}\n"
            "    finally:\n"
            "        con.close()\n"
        )
        try:
            return await self.ctx.exec.run_python(code, {"path": self.db_path, "sql": sql, "params": params or []})
        except Exception as e:
            raise ToolError(str(e).split("python snippet failed: ")[-1]) from e
