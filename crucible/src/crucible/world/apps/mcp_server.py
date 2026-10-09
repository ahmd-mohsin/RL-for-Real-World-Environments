"""Generic MCP-server App: any MCP server (stdio command or streamable-HTTP URL) becomes a World App.

The thousands of existing MCP servers (filesystems, ticketing, calendars, browsers ...) plug into a
World with no adapter code: tools are discovered from the server, calls go over one persistent session
(stateful servers keep their state), and ``allow_tools`` narrows what the policy may see (an
allowlist of fnmatch patterns over the server's tool names). State: whatever adapter the spec names
(``process`` for stateless servers; the server's backing DB via ``sql`` / ``postgres`` / ``fs``).

Params: ``command`` (list) or ``url``; ``env``; ``cwd``; ``allow_tools``; ``timeout_s``. In ``command`` and
``env``, ``{app:<id>}`` is replaced by that peer App's URL (e.g. a website the server should reach).
"""

from __future__ import annotations

import fnmatch
import re
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App
from crucible.world.packaging.session import PersistentMcpSession

_REF = re.compile(r"\{app:([A-Za-z0-9_-]+)\}")


class McpServerApp(App):
    template = "common/mcp_server"
    PATH_PARAMS = ()

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.session: PersistentMcpSession | None = None
        self._tools: list[ToolSpec] = []

    def _resolve_refs(self, value: str) -> str:
        peers = self.ctx.extra.get("peers", {})

        def sub(m: re.Match) -> str:
            peer = peers.get(m.group(1))
            url = getattr(peer, "url", None)
            if not url:
                raise RuntimeError(f"{self.id}: {m.group(0)} has no URL (start that App earlier in the spec)")
            return url

        return _REF.sub(sub, value)

    def server_command(self) -> list[str] | None:
        cmd = self.spec.params.get("command")
        return [self._resolve_refs(c) for c in cmd] if cmd else None

    async def start(self) -> None:
        p = self.spec.params
        if not self.ctx.exec.is_local:
            raise RuntimeError("common/mcp_server spawns local processes; use the local or agentenv backend")
        env = {k: self._resolve_refs(str(v)) for k, v in p.get("env", {}).items()}
        self.session = PersistentMcpSession(command=self.server_command(), url=self._resolve_refs(p["url"]) if p.get("url") else None,
                                            env=env, cwd=p.get("cwd"), timeout_s=float(p.get("timeout_s", 120)))
        await self.session.start(timeout_s=float(p.get("start_timeout_s", 180)))
        tools = await self.session.list_tools(self.id)
        allow = p.get("allow_tools")
        self._tools = [t for t in tools if not allow or any(fnmatch.fnmatchcase(t.name, pat) for pat in allow)]

    async def stop(self) -> None:
        if self.session is not None:
            await self.session.stop()
            self.session = None

    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        if name not in {t.name for t in self._tools}:
            return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.{name}", ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        assert self.session is not None
        return await self.session.call(self.id, name, args, call_id=call_id)

    async def call_unlisted(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Verifier/state-side calls to server tools hidden from the policy (e.g. a browser's evaluate)."""
        assert self.session is not None
        return await self.session.call(self.id, name, args, call_id=f"sys-{name}")
