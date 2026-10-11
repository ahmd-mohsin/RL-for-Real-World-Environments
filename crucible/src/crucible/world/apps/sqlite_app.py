"""SQLite-backed App: an enterprise system of record (ERP, ticketing, ledger, ...).

Tool surface, in order of preference:
1. ``tools_module``: a Python file of plain public functions (the MiMo ``tools/<app>.py`` format, also
   what AWM-style generated apps reduce to). Loaded with MiMo's own loader; transport ``inprocess``
   (default) or ``mcp`` (served by the vendored MiMo ``mcp_http`` over streamable HTTP).
2. ``sql_tools: true``: generic ``sql_query`` (read-only) and, if ``sql_write`` is true, ``sql_execute``.

Params: ``tools_module``, ``transport``, ``schema_sql`` / ``seed_db`` (host files copied in at
provision), ``seed_sql`` (one or more host SQL files of data statements run after the schema, e.g. EnterpriseOps-Gym
seed dumps), ``sql_tools``, ``sql_write``, ``env`` (extra env for the server process).
State: ``sql`` adapter with ``path``.

Seeding is deterministic: while the schema and seed scripts run, SQLite's ``now`` (``datetime('now')``, ``date()``,
``strftime(fmt)``, ...) is the World clock's ``t0`` (else ``virtual_epoch``), so two provisions of one spec hold the same
bytes (EnterpriseOps-Gym seed dumps call ``datetime('now')`` throughout).
"""

from __future__ import annotations

import shlex
import shutil
import sqlite3
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.state.base import snippet

SQL_SNIPPET = snippet("sql")


def pin_now(con: sqlite3.Connection, now: str) -> None:
    """Make SQLite's ``now`` a fixed instant on ``con``: the date/time functions are overridden by wrappers that swap a
    ``'now'`` time value (or a missing one) for ``now`` and evaluate the built-in on a private in-memory connection."""
    aux = sqlite3.connect(":memory:", check_same_thread=False)

    def wrap(name: str, at: int):
        def f(*args):
            a = list(args)
            if len(a) <= at:
                a.append(now)
            elif isinstance(a[at], str) and a[at].strip().lower() == "now":
                a[at] = now
            return aux.execute(f"SELECT {name}({','.join('?' * len(a))})", a).fetchone()[0]

        return f

    for name in ("datetime", "date", "time", "julianday", "unixepoch"):
        con.create_function(name, -1, wrap(name, 0), deterministic=True)
    con.create_function("strftime", -1, wrap("strftime", 1), deterministic=True)


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

    def _seed_now(self) -> str:
        from crucible.world.clock import parse_time

        clock = self.ctx.clock
        t0 = clock.spec.t0 if clock is not None and clock.spec is not None else self.spec.params.get("virtual_epoch", "2026-01-05T09:00:00Z")
        return parse_time(t0).strftime("%Y-%m-%d %H:%M:%S")

    async def provision(self) -> None:
        p = self.spec.params
        seeds = p.get("seed_sql") or []
        seeds = [seeds] if isinstance(seeds, str) else list(seeds)
        if p.get("seed_db") and not seeds:
            src = self.ctx.host_seed(p["seed_db"])
            await self.ctx.exec.run(f"mkdir -p {shlex.quote(str(Path(self.db_path).parent))}")
            await self.ctx.exec.push_file(src, self.db_path)
        elif p.get("schema_sql") or seeds:
            with tempfile.TemporaryDirectory() as d:
                local = Path(d) / "seed.db"
                if p.get("seed_db"):
                    shutil.copyfile(self.ctx.host_seed(p["seed_db"]), local)
                con = sqlite3.connect(local)
                try:
                    pin_now(con, self._seed_now())
                    if p.get("schema_sql"):
                        con.executescript(self.ctx.host_seed(p["schema_sql"]).read_text())
                    for f in seeds:
                        con.executescript(self.ctx.host_seed(f).read_text())
                    con.commit()
                finally:
                    con.close()
                await self.ctx.exec.run(f"mkdir -p {shlex.quote(str(Path(self.db_path).parent))}")
                await self.ctx.exec.push_file(local, self.db_path)

    async def after_restore(self) -> None:
        if getattr(self, "_venv", None) is not None:
            self._venv.reset()

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

            env = None
            if self.spec.params.get("virtual_time", True):
                from crucible.world.clock import parse_time
                from crucible.world.packaging.virtual_env import VirtualEnv

                clock = self.ctx.clock
                if clock is not None and clock.spec is not None:
                    now = clock.now
                else:  # an unclocked World still gets one fixed, plausible instant (deterministic, not 1970)
                    epoch = parse_time(self.spec.params.get("virtual_epoch", "2026-01-05T09:00:00Z"))
                    now = lambda: epoch
                env = VirtualEnv(now, int(self.ctx.extra.get("seed", 0)))
            self._venv = env
            self._module = InProcessToolModule(self.id, mod, env=env)
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
