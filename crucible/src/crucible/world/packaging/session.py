"""A persistent MCP client session owned by one background task.

Stateful MCP servers (a browser, anything with per-session memory) need every call on the same
session. The MCP SDK's clients are anyio context managers that must be entered and exited in one task,
so the session lives in a dedicated task and requests reach it through a queue — safe to use from any
task of the World's event loop. Supports stdio servers (spawned) and streamable-HTTP servers (by URL).
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.packaging.mcp_tools import _content_text


@dataclass
class _Req:
    method: str
    args: tuple
    fut: asyncio.Future = field(default_factory=lambda: asyncio.get_running_loop().create_future())


class PersistentMcpSession:
    def __init__(self, *, command: list[str] | None = None, env: dict[str, str] | None = None, cwd: str | None = None,
                 url: str | None = None, headers: dict[str, str] | None = None, timeout_s: float = 120):
        if bool(command) == bool(url):
            raise ValueError("give exactly one of command (stdio) or url (streamable http)")
        self.command, self.env, self.cwd, self.url, self.headers = command, env or {}, cwd, url, headers or {}
        self.timeout_s = timeout_s
        self._q: asyncio.Queue[_Req | None] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._ready = asyncio.Event()
        self._error: BaseException | None = None

    async def start(self, timeout_s: float = 120) -> None:
        self._task = asyncio.create_task(self._run())
        try:
            await asyncio.wait_for(self._ready.wait(), timeout_s)
        except TimeoutError:
            await self.stop()
            raise TimeoutError(f"MCP server did not initialize in {timeout_s}s: {self.command or self.url}") from None
        if self._error is not None:
            raise RuntimeError(f"MCP server failed to start: {self._error}") from self._error

    @contextlib.asynccontextmanager
    async def _transport(self):
        if self.command:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(command=self.command[0], args=self.command[1:], env={**os.environ, **self.env}, cwd=self.cwd)
            async with stdio_client(params) as (r, w):
                yield r, w
        else:
            import httpx
            from mcp.client.streamable_http import streamable_http_client

            async with httpx.AsyncClient(headers=self.headers, timeout=httpx.Timeout(self.timeout_s)) as http, streamable_http_client(self.url, http_client=http) as (r, w, _):
                yield r, w

    async def _run(self) -> None:
        from mcp.client.session import ClientSession

        try:
            async with self._transport() as (r, w), ClientSession(r, w) as s:
                await s.initialize()
                self._ready.set()
                while True:
                    req = await self._q.get()
                    if req is None:
                        return
                    try:
                        res = await getattr(s, req.method)(*req.args)
                        if not req.fut.done():
                            req.fut.set_result(res)
                    except Exception as e:  # noqa: BLE001 - delivered to the caller
                        if not req.fut.done():
                            req.fut.set_exception(e)
        except Exception as e:  # noqa: BLE001
            self._error = e
            self._ready.set()
            while not self._q.empty():
                req = self._q.get_nowait()
                if req is not None and not req.fut.done():
                    req.fut.set_exception(RuntimeError(f"MCP session closed: {e}"))

    async def _call(self, method: str, *args: Any) -> Any:
        if self._error is not None:
            raise RuntimeError(f"MCP session is down: {self._error}")
        req = _Req(method, args)
        await self._q.put(req)
        return await asyncio.wait_for(req.fut, self.timeout_s + 5)

    async def list_tools(self, app: str) -> list[ToolSpec]:
        res = await self._call("list_tools")
        return sorted((ToolSpec(app=app, name=t.name, description=t.description or t.name, input_schema=t.inputSchema or {"type": "object", "properties": {}},
                                read_only=bool(getattr(t, "annotations", None) and getattr(t.annotations, "readOnlyHint", False)))
                       for t in res.tools), key=lambda t: t.name)

    async def call(self, app: str, name: str, args: dict[str, Any], *, call_id: str) -> ToolResult:
        t0 = time.monotonic()
        try:
            res = await self._call("call_tool", name, args, timedelta(seconds=self.timeout_s))
        except Exception as e:  # noqa: BLE001
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{name}", ok=False, content=f"transport error: {e}", error="transport",
                              duration_s=time.monotonic() - t0)
        return ToolResult(tool_call_id=call_id, function_name=f"{app}.{name}", ok=not res.isError, content=_content_text(res.content),
                          data=res.structuredContent, error="tool_error" if res.isError else None, duration_s=time.monotonic() - t0)

    async def stop(self) -> None:
        if self._task is None:
            return
        await self._q.put(None)
        try:
            await asyncio.wait_for(self._task, 15)
        except (TimeoutError, Exception):  # noqa: BLE001
            self._task.cancel()
            with contextlib.suppress(BaseException):
                await self._task
        self._task = None
