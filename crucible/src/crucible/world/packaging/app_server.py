"""Serve Crucible Apps over MCP (streamable HTTP), in-process.

Any World App — SQLite tool modules, workspaces, mailboxes, doc libraries, JSON APIs — becomes an MCP
server with its exact tool schemas, so:

- the agentenv gateway can front it (each App is one internal MCP server; tool names are namespaced
  ``<app>__<tool>`` because the gateway requires unique names across servers);
- any MCP-capable agent harness (Claude Code, Codex, OpenHands via Harbor agents) can be pointed at
  a Crucible World directly.

Uses the MCP SDK's low-level ``Server`` (explicit JSON schemas, no signature inference) and its
``StreamableHTTPSessionManager``; runs under uvicorn as a task on the caller's event loop. A
``/ready`` route answers 200 once serving (the agentenv gateway's readiness probe).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Awaitable, Callable
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.packaging.sidecar import free_port

ToolCaller = Callable[[str, dict[str, Any]], Awaitable[ToolResult]]


class AppMcpServer:
    def __init__(self, name: str, tools: list[ToolSpec], caller: ToolCaller, *, prefix: bool = True, host: str = "127.0.0.1", port: int = 0):
        self.name = name
        self.tools = tools
        self.caller = caller
        self.prefix = prefix
        self.host = host
        self.port = port or free_port()
        self._server_task: asyncio.Task | None = None
        self._uvicorn = None

    def exposed_name(self, t: ToolSpec) -> str:
        return f"{t.app}__{t.name}" if self.prefix else t.name

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    def _build(self):
        from mcp import types
        from mcp.server.lowlevel import Server
        from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
        from starlette.applications import Starlette
        from starlette.responses import JSONResponse
        from starlette.routing import Mount, Route

        server = Server(self.name)
        by_name = {self.exposed_name(t): t for t in self.tools}

        @server.list_tools()
        async def _list() -> list[types.Tool]:
            return [types.Tool(name=n, description=t.description, inputSchema=t.input_schema) for n, t in by_name.items()]

        @server.call_tool(validate_input=False)
        async def _call(name: str, arguments: dict[str, Any]):
            t = by_name.get(name)
            if t is None:
                raise ValueError(f"Unknown tool: {name}")
            res = await self.caller(t.name, arguments or {})
            if not res.ok:
                raise ValueError(res.content)  # becomes isError=True with this text
            content = [types.TextContent(type="text", text=res.content)]
            if isinstance(res.data, dict):
                return content, res.data
            return content

        manager = StreamableHTTPSessionManager(app=server, stateless=True, json_response=True)

        async def handle(scope, receive, send):
            await manager.handle_request(scope, receive, send)

        async def ready(_request):
            return JSONResponse({"ok": True, "server": self.name, "tools": len(by_name)})

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            async with manager.run():
                yield

        return Starlette(routes=[Route("/ready", ready), Mount("/mcp", app=handle)], lifespan=lifespan)

    async def start(self) -> None:
        import uvicorn

        config = uvicorn.Config(self._build(), host=self.host, port=self.port, log_level="warning", lifespan="on")
        self._uvicorn = uvicorn.Server(config)
        self._server_task = asyncio.create_task(self._uvicorn.serve())
        for _ in range(400):
            if self._uvicorn.started:
                return
            if self._server_task.done():
                raise RuntimeError(f"MCP server {self.name} failed: {self._server_task.exception()}")
            await asyncio.sleep(0.025)
        raise TimeoutError(f"MCP server {self.name} did not start")

    async def stop(self) -> None:
        if self._uvicorn is not None:
            self._uvicorn.should_exit = True
        if self._server_task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._server_task, 5)


def describe(servers: list[AppMcpServer]) -> str:
    return json.dumps({s.name: s.url for s in servers})
