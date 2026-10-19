"""MCP sidecar: serve tool modules over streamable HTTP and call them (replaces MiMo's sidecar_entrypoint).

``McpServerProcess`` runs the vendored ``mcp_http`` script for one tools file on a free local port (socket
readiness instead of MiMo's Linux-only /proc probe). Nothing restarts a server that crashed: MiMo's supervisor loop
did, which hid the outage, so a dead server's calls fail as infrastructure (M1 audit R-42 removed the unused
``Sidecar`` supervisor). ``McpClient`` lists and calls tools on one server; it keeps
no long-lived session (each operation opens a session, like MiMo's one-shot bridge), which is
robust across event loops and cancellation at ~10-30 ms per call locally.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.exec import safe_environ, spawn_logged
from crucible.world.packaging.mcp_compat import mcp_field  # mcp 1.x / 2.x field names
from crucible.world.packaging.mcp_tools import _content_text


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_open(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.2)
        return s.connect_ex((host, port)) == 0


class McpClient:
    def __init__(self, url: str, *, timeout_s: float = 60, headers: dict[str, str] | None = None):
        self.url = url
        self.timeout_s = timeout_s
        self.headers = headers or {}

    @contextlib.asynccontextmanager
    async def _session(self):
        from mcp.client.session import ClientSession

        from crucible.world.packaging.mcp_compat import http_streams

        async with (
            http_streams(self.url, headers=self.headers, timeout_s=self.timeout_s, read_timeout_s=self.timeout_s * 5) as (r, w),
            ClientSession(r, w) as session,
        ):
            await session.initialize()
            yield session

    async def list_tools(self, app: str) -> list[ToolSpec]:
        async with self._session() as s:
            res = await s.list_tools()
        return sorted(
            (ToolSpec(app=app, name=t.name, description=t.description or t.name, input_schema=mcp_field(t, "inputSchema") or {"type": "object", "properties": {}}) for t in res.tools),
            key=lambda t: t.name,
        )

    async def call(self, app: str, name: str, args: dict[str, Any], *, call_id: str) -> ToolResult:
        from crucible.world.packaging.mcp_compat import call_timeout

        t0 = time.monotonic()
        try:
            async with self._session() as s:
                res = await s.call_tool(name, args, read_timeout_seconds=call_timeout(self.timeout_s))
        except Exception as e:  # noqa: BLE001 - httpx, anyio and MCP errors all mean the same thing to the caller: the server is unreachable
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{name}", ok=False, content=f"transport error: {e}", error="transport", duration_s=time.monotonic() - t0)
        return ToolResult(
            tool_call_id=call_id,
            function_name=f"{app}.{name}",
            ok=not mcp_field(res, "isError"),
            content=_content_text(res.content),
            data=mcp_field(res, "structuredContent"),
            error="tool_error" if mcp_field(res, "isError") else None,
            duration_s=time.monotonic() - t0,
        )


@dataclass
class McpServerProcess:
    name: str
    tools_path: Path
    port: int = 0
    host: str = "127.0.0.1"
    cwd: Path | None = None
    env: dict[str, str] = field(default_factory=dict)
    log_path: Path | None = None
    proc: asyncio.subprocess.Process | None = None

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    async def start(self, timeout_s: float = 60) -> None:
        if not self.port:
            self.port = free_port()
        self.proc = await spawn_logged(
            sys.executable, "-m", "crucible._vendor.mimo_sidecar.mcp_http",
            "--tools-path", str(self.tools_path), "--server-name", self.name, "--port", str(self.port), "--host", self.host,
            cwd=str(self.cwd) if self.cwd else None,
            env=safe_environ(self.env),
            log=self.log_path,
        )
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.proc.returncode is not None:
                raise RuntimeError(f"MCP server {self.name} exited with {self.proc.returncode} (log: {self.log_path})")
            if port_open(self.port, self.host):
                return
            await asyncio.sleep(0.05)
        raise TimeoutError(f"MCP server {self.name} not listening on {self.port} after {timeout_s}s")

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    async def stop(self) -> None:
        if self.proc is None or self.proc.returncode is not None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(self.proc.pid, 15)
        try:
            await asyncio.wait_for(self.proc.wait(), 5)
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.proc.pid, 9)
            await self.proc.wait()


def dump_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, default=str)
