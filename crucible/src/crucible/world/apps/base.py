"""App base: one system in a World = state (via a State Adapter) + a tool surface + optional servers.

Apps never see actors' identities beyond the ``actor`` argument of ``call`` (for audit/ownership),
and the policy never sees an App's State Adapter: the World hands the adapter to the verifier view.
"""

from __future__ import annotations

import posixpath
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crucible.core.registry import get_registry
from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.core.schema.world import AppSpec
from crucible.core.store import Store
from crucible.world.exec import Exec

if TYPE_CHECKING:
    from crucible.world.clock import VirtualClock
    from crucible.world.state.base import StateAdapter


@dataclass
class AppContext:
    exec: Exec
    store: Store
    base_path: str  # where relative app paths live in the exec (local: world dir; container: e.g. "/")
    host_workdir: Path  # host-side scratch for this world (logs, server sockets)
    clock: VirtualClock | None = None
    pack_root: Path | None = None  # seeds referenced by the spec resolve against this (host side)
    extra: dict[str, Any] = field(default_factory=dict)
    seed: int = 0  # this App's own seed: derive_seed(WorldSpec.seed, app id); reproducible from the spec hash in every trace

    def resolve(self, p: str) -> str:
        return p if p.startswith("/") else posixpath.normpath(posixpath.join(self.base_path, p))

    def host_seed(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        if self.pack_root is None:
            raise ValueError(f"relative seed path {p!r} but no pack_root to resolve it against")
        return (self.pack_root / path).resolve()


class ToolError(Exception):
    """Raised by App tool implementations; becomes ok=False with the message as content."""


class App(ABC):
    template: ClassVar[str]
    PATH_PARAMS: ClassVar[tuple[str, ...]] = ("root", "path")

    def __init__(self, spec: AppSpec, ctx: AppContext):
        self.spec = spec
        self.ctx = ctx
        self.id = spec.id
        params = dict(spec.state.params)
        for key in self.PATH_PARAMS:
            if isinstance(params.get(key), str):
                params[key] = ctx.resolve(params[key])
        adapter_cls = get_registry().get("state_adapter", spec.state.adapter)
        self.state: StateAdapter = adapter_cls(spec.id, ctx.exec, params, ctx.store)
        self.state_params = params

    # ---- lifecycle ----
    async def provision(self) -> None:
        pass

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    # ---- tools ----
    @abstractmethod
    async def list_tools(self) -> list[ToolSpec]: ...

    async def after_restore(self) -> None:
        """Hook: the World just restored this App's state (and every other App's) from a WorldRef."""

    def image(self, data: bytes, mime: str = "image/png"):
        """An image tool output: stored once (content-addressed) and attached to the live observation."""
        import base64

        from crucible.core.schema.actions import ImageRef

        return ImageRef(mime=mime, ref=self.ctx.store.put_bytes(data), data_b64=base64.b64encode(data).decode())

    @property
    def policy_exec(self):
        """Exec for tools that touch policy-owned files: the guard's demoted Exec in isolated worlds, else ctx.exec.
        Tools over App-owned state (an SQLite file, a mailbox) run as the App, i.e. ctx.exec, like a real server."""
        return self.ctx.extra.get("policy_exec") or self.ctx.exec

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        fn = getattr(self, f"tool_{name}", None)
        qualified = f"{self.id}.{name}"
        if fn is None or name not in {t.name for t in await self.list_tools()}:
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        try:
            out = await fn(**args)
        except ToolError as e:
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, content=str(e), error="tool_error")
        except TypeError as e:
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, content=f"bad arguments: {e}", error="bad_arguments")
        if isinstance(out, ToolResult):
            return out
        content = out if isinstance(out, str) else _json(out)
        return ToolResult(tool_call_id=call_id, function_name=qualified, ok=True, content=content, data=None if isinstance(out, str) else out)


def _json(x: Any) -> str:
    import json

    return json.dumps(x, ensure_ascii=False, indent=1, default=str)  # Decimal / datetime render exactly as text


def confine(root: str, path: str) -> str:
    """Resolve ``path`` under ``root``; refuse escapes (``..``, absolute paths elsewhere)."""
    if path.startswith("/"):
        full = posixpath.normpath(path)
    else:
        full = posixpath.normpath(posixpath.join(root, path))
    root_n = posixpath.normpath(root)
    if full != root_n and not full.startswith(root_n.rstrip("/") + "/"):
        raise ToolError(f"path {path!r} is outside the workspace")
    return full
