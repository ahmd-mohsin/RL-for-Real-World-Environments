"""Generic MCP-server App: any MCP server (stdio command or streamable-HTTP URL) becomes a World App.

The thousands of existing MCP servers (filesystems, ticketing, calendars, browsers ...) plug into a
World with no adapter code: tools are discovered from the server, calls go over one persistent session
(stateful servers keep their state), and ``allow_tools`` narrows what the policy may see (an
allowlist of fnmatch patterns over the server's tool names). State: whatever adapter the spec names
(``process`` for stateless servers; the server's backing DB via ``sql`` / ``postgres`` / ``fs``).

Params: ``command`` (list) or ``url``; ``headers`` (HTTP only); ``env``; ``cwd``; ``allow_tools``; ``timeout_s``. In
``command``, ``url`` and ``env``, ``{app:<id>}`` is replaced by that peer App's URL (e.g. a website the server should reach).

Identity (M1 item 3, the EnterpriseOps-Gym context headers): over HTTP, a call by an actor with an ``IdentitySpec`` goes
through that identity's own session, opened on first use with ``x-user-*`` headers, so the server applies its own
permissions and row visibility for that user. Calls by actors without an identity, and system calls, use the App's
session. A stdio server cannot receive per-call identity; ``identity: required`` refuses that configuration.
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
        self._as: dict[str, PersistentMcpSession] = {}  # identity headers (canonical JSON) -> that user's session

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
        if p.get("identity") == "required" and not p.get("url") and self.ctx.extra.get("identities"):
            raise RuntimeError(f"{self.id}: identity is required but a stdio server cannot receive per-call identity headers; serve it over HTTP")
        self.session = PersistentMcpSession(command=self.server_command(), url=self._resolve_refs(p["url"]) if p.get("url") else None,
                                            env=env, cwd=p.get("cwd"), headers=dict(p.get("headers", {})), timeout_s=float(p.get("timeout_s", 120)))
        await self.session.start(timeout_s=float(p.get("start_timeout_s", 180)))
        tools = await self.session.list_tools(self.id)
        allow = p.get("allow_tools")
        self._tools = [t for t in tools if not allow or any(fnmatch.fnmatchcase(t.name, pat) for pat in allow)]

    async def stop(self) -> None:
        for sess in self._as.values():
            await sess.stop()
        self._as = {}
        if self.session is not None:
            await self.session.stop()
            self.session = None

    async def _session_for(self, actor: str) -> PersistentMcpSession:
        import json

        assert self.session is not None
        ident = self.identity(actor)
        if ident is None or not self.spec.params.get("url"):
            return self.session
        headers = {**self.spec.params.get("headers", {}), **ident.http_headers()}
        key = json.dumps(headers, sort_keys=True)
        sess = self._as.get(key)
        if sess is None:
            p = self.spec.params
            sess = PersistentMcpSession(url=self._resolve_refs(p["url"]), headers=headers, timeout_s=float(p.get("timeout_s", 120)))
            await sess.start(timeout_s=float(p.get("start_timeout_s", 180)))
            self._as[key] = sess
        return sess

    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        if name not in {t.name for t in self._tools}:
            return ToolResult(tool_call_id=call_id, function_name=f"{self.id}.{name}", ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        return await (await self._session_for(actor)).call(self.id, name, args, call_id=call_id)

    async def call_unlisted(self, name: str, args: dict[str, Any]) -> ToolResult:
        """Verifier/state-side calls to server tools hidden from the policy (e.g. a browser's evaluate)."""
        assert self.session is not None
        return await self.session.call(self.id, name, args, call_id=f"sys-{name}")
