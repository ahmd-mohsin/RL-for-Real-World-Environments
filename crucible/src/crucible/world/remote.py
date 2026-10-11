"""A WebSocket client for environments served elsewhere (M1 item 9; wire-compatible with OpenEnv's ``/ws`` endpoint,
``openenv.core.env_client.EnvClient`` and ``env_server/http_server.py``, BSD-3-Clause; written against the protocol, no
OpenEnv import, so a remote environment needs nothing installed on the Crucible side but ``websockets``).

Protocol (one persistent socket = one server session; JSON text frames, one response per request, in order):

    -> {"type": "reset", "data": {kwargs}}     <- {"type": "observation", "data": {"observation", "reward", "done"}}
    -> {"type": "step",  "data": {action}}     <- {"type": "observation", "data": {...}}
    -> {"type": "state"}                       <- {"type": "state", "data": {...}}
    -> {"type": "mcp", "data": {JSON-RPC}}      <- {"type": "mcp", "data": {JSON-RPC response}}   (tools/list, tools/call)
    -> {"type": "close"}
    <- {"type": "error", "data": {"message", "code"}} for any request

``?session_id=`` attaches the socket to an existing server session (OpenEnv's HTTP MCP sessions).

Failure semantics (the World decides blame from these):

- ``RemoteActionError``: the server refused the action itself (``VALIDATION_ERROR``, ``INVALID_JSON``,
  ``UNKNOWN_TYPE``): the policy's malformed call;
- ``RemoteEnvError``: the environment failed (``EXECUTION_ERROR``, ``CAPACITY_REACHED``, ``FACTORY_ERROR``, a lost
  connection, a timeout): infrastructure, masked;
- a request that times out or is cancelled drops the socket (OpenEnv's desync guard: a late reply must never be read as
  the answer to the next request). The next request opens a new socket, which is a new server session, so
  ``session_lost`` is set and the caller restores its state (``OpenEnvApp`` replays its journal and checks every digest).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from typing import Any

ACTION_CODES = frozenset({"VALIDATION_ERROR", "INVALID_JSON", "UNKNOWN_TYPE"})


class RemoteEnvError(RuntimeError):
    """The remote environment or the connection failed (infrastructure)."""

    def __init__(self, message: str, code: str | None = None):
        super().__init__(f"{message} ({code})" if code else message)
        self.code = code


class RemoteActionError(RemoteEnvError):
    """The remote environment rejected the action as malformed (the policy's error)."""


def ws_url(url: str) -> str:
    """``http(s)://host[:port][/ws]`` or ``ws(s)://...`` -> the session socket URL (``.../ws``)."""
    u = url.rstrip("/")
    if u.startswith("http://"):
        u = "ws://" + u[7:]
    elif u.startswith("https://"):
        u = "wss://" + u[8:]
    elif not u.startswith(("ws://", "wss://")):
        raise ValueError(f"not an http(s) or ws(s) URL: {url!r}")
    head, _, query = u.partition("?")
    if not head.endswith("/ws"):
        head += "/ws"
    return head + ("?" + query if query else "")


class RemoteEnvClient:
    def __init__(self, url: str, *, session_id: str | None = None, message_timeout_s: float = 300.0, connect_timeout_s: float = 30.0,
                 ping_interval_s: float | None = 20.0, max_message_bytes: int = 100 * 1024 * 1024, headers: dict[str, str] | None = None):
        self.url = ws_url(url) + (("&" if "?" in url else "?") + f"session_id={session_id}" if session_id else "")
        self.message_timeout_s, self.connect_timeout_s = message_timeout_s, connect_timeout_s
        self.ping_interval_s, self.max_message_bytes, self.headers = ping_interval_s, max_message_bytes, dict(headers or {})
        self._ws: Any = None
        self._lock = asyncio.Lock()
        self._rpc_id = 0
        self.sessions_opened = 0
        self.session_lost = False  # a socket was dropped after the first session: the server-side state is gone

    async def connect(self) -> None:
        if self._ws is not None:
            return
        from websockets.asyncio.client import connect

        try:
            self._ws = await asyncio.wait_for(
                connect(self.url, max_size=self.max_message_bytes, ping_interval=self.ping_interval_s, ping_timeout=self.ping_interval_s,
                        additional_headers=self.headers or None, open_timeout=self.connect_timeout_s),
                timeout=self.connect_timeout_s + 5)
        except (OSError, TimeoutError) as e:
            raise RemoteEnvError(f"cannot connect to {self.url}: {type(e).__name__}: {e}", "UNREACHABLE") from e
        except Exception as e:
            raise RemoteEnvError(f"cannot connect to {self.url}: {type(e).__name__}: {e}", "HANDSHAKE") from e
        if self.sessions_opened:
            self.session_lost = True
        self.sessions_opened += 1

    def _drop(self) -> None:
        ws, self._ws = self._ws, None
        if ws is not None:
            task = asyncio.ensure_future(ws.close())  # never wait out a close handshake on the error path
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)

    async def request(self, message: dict[str, Any]) -> dict[str, Any]:
        async with self._lock:  # one request in flight per socket: responses pair with requests by order
            await self.connect()
            try:
                await self._ws.send(json.dumps(message))
                raw = await asyncio.wait_for(self._ws.recv(), timeout=self.message_timeout_s)
            except (TimeoutError, asyncio.CancelledError):
                self._drop()
                raise
            except Exception as e:
                self._drop()
                raise RemoteEnvError(f"connection lost during {message.get('type')}: {type(e).__name__}: {e}", "CONNECTION_LOST") from e
        try:
            resp = json.loads(raw)
        except (TypeError, ValueError) as e:
            raise RemoteEnvError(f"non-JSON reply to {message.get('type')}", "BAD_REPLY") from e
        if resp.get("type") == "error":
            data = resp.get("data") or {}
            code = str(data.get("code") or "")
            exc = RemoteActionError if code.upper() in ACTION_CODES else RemoteEnvError
            raise exc(str(data.get("message") or "remote error")[:2000], code or None)
        return resp

    async def reset(self, **kwargs: Any) -> dict[str, Any]:
        return (await self.request({"type": "reset", "data": {k: v for k, v in kwargs.items() if v is not None}})).get("data") or {}

    async def step(self, action: dict[str, Any]) -> dict[str, Any]:
        return (await self.request({"type": "step", "data": dict(action)})).get("data") or {}

    async def state(self) -> dict[str, Any]:
        return (await self.request({"type": "state"})).get("data") or {}

    async def rpc(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self._rpc_id += 1
        resp = (await self.request({"type": "mcp", "data": {"jsonrpc": "2.0", "id": self._rpc_id, "method": method, "params": params or {}}})).get("data") or {}
        if resp.get("error"):
            err = resp["error"]
            raise RemoteEnvError(f"{method}: {err.get('message', err)}", str(err.get("code")))
        return resp.get("result")

    async def list_tools(self) -> list[dict[str, Any]]:
        return list((await self.rpc("tools/list")).get("tools") or [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return await self.rpc("tools/call", {"name": name, "arguments": dict(arguments)}) or {}

    async def close(self) -> None:
        if self._ws is None:
            return
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._ws.send(json.dumps({"type": "close"})), timeout=2)
        with contextlib.suppress(Exception):
            await asyncio.wait_for(self._ws.close(), timeout=5)
        self._ws = None
