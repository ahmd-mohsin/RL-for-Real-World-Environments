"""Website App: a web application (UI + backend) whose state lives in a database.

A pack supplies an ASGI app factory (``"path/to/site.py:create_app"`` taking the SQLite path); the App
runs it with uvicorn in a subprocess on a free local port and exposes ``url``. The policy normally
reaches it through a ``common/browser`` App (what a human clerk would do); ``api_tools: true`` adds a
plain ``http_request`` tool for API-level access. State: the ``sql`` adapter over the site's DB (the
real system of record), so verifiers check outcomes, not pixels.

Params: ``app`` (factory ref, pack-relative), ``schema_sql`` / ``seed_db``, ``api_tools``, ``host``.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import sys
import time
from typing import Any

import httpx

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import ToolError
from crucible.world.apps.sqlite_app import SqliteApp
from crucible.world.exec import spawn_logged
from crucible.world.packaging.sidecar import free_port, port_open


class WebsiteApp(SqliteApp):
    template = "common/website"

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.port = 0
        self.proc: asyncio.subprocess.Process | None = None

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def start(self) -> None:
        if not self.ctx.exec.is_local:
            raise RuntimeError("common/website runs a local server; use the local or agentenv backend")
        ref = self.spec.params["app"]
        path, _, factory = ref.partition(":")
        site = self.ctx.host_seed(path)
        os.makedirs(os.path.dirname(os.path.abspath(self.db_path)), exist_ok=True)
        self.port = free_port()
        self.proc = await spawn_logged(
            sys.executable, "-m", "crucible.world.apps.website_runner", "--site", str(site), "--factory", factory or "create_app",
            "--db", os.path.abspath(self.db_path), "--port", str(self.port), log=self.ctx.host_workdir / f"site_{self.id}.log",
        )
        deadline = time.monotonic() + 60
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                raise RuntimeError(f"website {self.id} exited: {(self.ctx.host_workdir / f'site_{self.id}.log').read_text()[-1500:]}")
            if port_open(self.port):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"website {self.id} did not start")

    async def stop(self) -> None:
        if self.proc is not None and self.proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, 15)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.proc.wait(), 5)

    async def list_tools(self) -> list[ToolSpec]:
        if not self.spec.params.get("api_tools"):
            return []
        return [ToolSpec(app=self.id, name="http_request", description="HTTP request to this web application (path relative to its root).",
                         input_schema={"type": "object", "properties": {"method": {"type": "string"}, "path": {"type": "string"}, "form": {"type": "object"}},
                                       "required": ["method", "path"]})]

    async def tool_http_request(self, method: str, path: str, form: dict[str, Any] | None = None) -> Any:
        if not path.startswith("/"):
            raise ToolError("path must start with /")
        async with httpx.AsyncClient(base_url=self.url, timeout=30, follow_redirects=True) as c:
            r = await c.request(method.upper(), path, data=form)
        return f"HTTP {r.status_code}\n{r.text[:20000]}"
