"""MCP SDK 1.x / 2.x compatibility for Crucible's MCP clients and servers.

mcp 2 renamed FastMCP to MCPServer, moved function tools to ``mcp.server.mcpserver.tools``, made the streamable-HTTP
client yield two streams instead of three, and switched its HTTP client to ``httpx2``. The vLLM image (where training
rollouts run) ships mcp 2 and the ``verifiers`` extra pins it, while agentenv needs mcp 1; found live when every
``common/sqlite_app`` World failed to start inside the vLLM image. Every Crucible MCP client opens its streams here.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator
from typing import Any


@contextlib.asynccontextmanager
async def http_streams(url: str, *, headers: dict[str, str] | None = None, timeout_s: float | None = 120.0,
                       read_timeout_s: float | None = None) -> AsyncIterator[tuple[Any, Any]]:
    """``(read, write)`` streams of a streamable-HTTP MCP connection, on either SDK major version."""
    import mcp.client.streamable_http as sh

    if hasattr(sh, "streamablehttp_client"):  # mcp 1.x: httpx client, three-value yield
        import httpx

        t = httpx.Timeout(timeout_s, read=read_timeout_s if read_timeout_s is not None else timeout_s) if timeout_s is not None else None
        async with httpx.AsyncClient(headers=headers or {}, timeout=t) as http, sh.streamable_http_client(url, http_client=http) as (r, w, _):
            yield r, w
        return
    import httpx2  # mcp 2.x ships it

    t2 = httpx2.Timeout(timeout_s, read=read_timeout_s if read_timeout_s is not None else timeout_s) if timeout_s is not None else None
    async with sh.create_mcp_http_client(headers=headers or {}, timeout=t2) as http, sh.streamable_http_client(url, http_client=http) as (r, w):
        yield r, w


_SNAKE = {"inputSchema": "input_schema", "isError": "is_error", "structuredContent": "structured_content", "outputSchema": "output_schema",
          "nextCursor": "next_cursor", "mimeType": "mime_type"}


def mcp_field(obj: Any, camel: str, default: Any = None) -> Any:
    """A protocol model's field by its 1.x camelCase name, or its 2.x snake_case one."""
    if hasattr(obj, camel):
        return getattr(obj, camel)
    return getattr(obj, _SNAKE.get(camel, camel), default)


def mcp_major() -> int:
    import mcp.client.streamable_http as sh

    return 1 if hasattr(sh, "streamablehttp_client") else 2


def call_timeout(seconds: float) -> Any:
    """``ClientSession.call_tool(read_timeout_seconds=...)``: a ``timedelta`` in mcp 1.x, plain seconds in 2.x."""
    if mcp_major() == 1:
        from datetime import timedelta

        return timedelta(seconds=seconds)
    return float(seconds)
