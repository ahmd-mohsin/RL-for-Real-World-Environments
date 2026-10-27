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
from crucible.world.exec import EnvContainerLost, Exec, container_lost_cause

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
    tool_timeout_s: float | None = None  # the World's limits.tool_timeout_s: no tool's own deadline exceeds it (App.deadline)

    def resolve(self, p: str) -> str:
        return p if p.startswith("/") else posixpath.normpath(posixpath.join(self.base_path, p))

    def host_seed(self, p: str) -> Path:
        path = Path(p)
        if path.is_absolute():
            return path
        if self.pack_root is None:
            raise ValueError(f"relative seed path {p!r} but no pack_root to resolve it against")
        return (self.pack_root / path).resolve()


# how long one statement of a policy's SQL may run (an App's ``sql_deadline_s`` param; M1 audit R-12)
SQL_DEADLINE_S = 30.0


class ToolError(Exception):
    """Raised by App tool implementations; becomes ok=False with the message as content."""


class PolicyToolError(ToolError):
    """A ``ToolError`` the policy's own well-formed call caused (an ``old_string`` the file does not hold, a regex that does
    not compile, a path that does not exist): it stays ``tool_error`` (not a malformed call, so no format penalty) with
    fault ``policy``. It was labelled ``tool`` and read as a World fault (mvdan-sh dc951aeb-e1, 2026-10-26)."""


class ArgumentError(ToolError):
    """A ``ToolError`` the caller's arguments caused (a path outside the workspace, an id that does not exist): the
    policy's malformed call, ``invalid_arguments`` with policy fault (M14 campaign B13)."""


class WorldSeedError(RuntimeError):
    """Copying an App's seed into the World failed: infrastructure (``crucible:backend_error``), never unknown blame
    (M14 campaign B3)."""


async def push_seed_tree(app: App, src: Path, dest: str) -> None:
    """Copy the host directory ``src`` into ``dest`` in the App's exec as one tarball under a name unique to this call
    (M14 campaign B3: a fixed ``/tmp/.crucible_seed_<app>.tar`` was shared by every World on the host, so concurrent
    Worlds truncated or deleted each other's seed). The tarball is removed whatever happens."""
    import shlex
    import tarfile
    import tempfile
    import uuid

    ex = app.ctx.exec
    with tempfile.TemporaryDirectory(prefix="crucible-seed-") as d:
        tar = Path(d) / "seed.tar"
        try:
            with tarfile.open(tar, "w") as tf:
                for p in sorted(src.rglob("*")):
                    tf.add(p, arcname=str(p.relative_to(src)), recursive=False)
        except OSError as e:
            raise WorldSeedError(f"seeding {app.id}: cannot pack {src}: {e}") from e
        remote = str(tar) if ex.is_local else f"{ex.staging_dir}/.crucible_seed_{app.id}_{uuid.uuid4().hex}.tar"
        try:
            if not ex.is_local:
                await ex.push_file(tar, remote)
            out = await ex.run(f"mkdir -p {shlex.quote(dest)} && tar -xf {shlex.quote(remote)} -C {shlex.quote(dest)}")
        except Exception as e:
            raise WorldSeedError(f"seeding {app.id} failed: {type(e).__name__}: {e}"[:600]) from e
        finally:
            if not ex.is_local:
                try:
                    await ex.run(f"rm -f {shlex.quote(remote)}")
                except Exception:  # noqa: BLE001, S110 - best effort; the staging dir dies with the World
                    pass
        if not out.ok:
            raise WorldSeedError(f"seeding {app.id} failed: {out.stderr[-500:]}")


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

    async def before_snapshot(self) -> None:
        """Hook: flush state held in memory to where the state adapter reads it (a snapshot, a fingerprint or a verifier
        query follows). Apps that write their state on every change need nothing here."""

    def image(self, data: bytes, mime: str = "image/png"):
        """An image tool output: stored once (content-addressed) and attached to the live observation."""
        import base64

        from crucible.core.schema.actions import ImageRef

        return ImageRef(mime=mime, ref=self.ctx.store.put_bytes(data), data_b64=base64.b64encode(data).decode())

    def identity(self, actor: str):
        """The calling actor's ``IdentitySpec`` (None for actors without one, and for system calls)."""
        return self.ctx.extra.get("identities", {}).get(actor)

    def deadline(self, requested: float, cap: float | None = None) -> float:
        """A tool's own time limit: what the caller asked for (at least 1 s), at most ``cap`` and the World's
        ``tool_timeout_s``. The tool's deadline then fires before the World's, so a policy's hung command comes back as its
        own scored ``[TIMEOUT]``, never a masked ``transport_timeout`` (NeMo-Gym's deadline layering; M1 audit R-4)."""
        return max(1.0, min([float(requested), *(float(c) for c in (cap, self.ctx.tool_timeout_s) if c is not None)]))

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
        except ArgumentError as e:  # the caller's arguments: the policy's error (M14 campaign B13)
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, content=str(e), error="invalid_arguments", fault="policy")
        except PolicyToolError as e:
            return ToolResult(tool_call_id=call_id, function_name=qualified, ok=False, content=str(e), error="tool_error", fault="policy")
        except ToolError as e:
            if (lost := container_lost_cause(e)) is not None:  # the host confirmed the container is gone: the World's, not the policy's (M1 R-1)
                raise EnvContainerLost(str(lost)) from e
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
        raise ArgumentError(f"path {path!r} is outside the workspace")
    return full
