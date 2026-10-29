"""World backends: where a World's Apps run.

The core ships ``local``. Framework backends live in ``crucible.interop`` (``harbor``: Harbor sandbox
environments, incl. GPU and remote Docker hosts; ``agentenv``: agentenv gateway with clock,
triggers and role-based tool access) and are looked up by name in the registry, so the world
package never imports a framework.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from crucible.core.schema.world import AppSpec, WorldSpec
from crucible.world.exec import Exec

if TYPE_CHECKING:
    from crucible.core.schema.actions import ToolResult
    from crucible.world.world import World


@dataclass
class BackendCapabilities:
    isolation: str = "none"  # none | container | gvisor | vm
    gpus: bool = False
    network_enforced: bool = False
    native_clock: bool = False  # backend keeps virtual time itself (agentenv gateway)
    native_acl: bool = False  # backend enforces per-actor tool access itself
    pausable: bool = False  # pause()/resume() freeze the World's processes (m13 S6)
    notes: list[str] = field(default_factory=list)


class ToolRouter(ABC):
    """Optional: a backend that routes tool calls itself (e.g. through agentenv's gateway)."""

    @abstractmethod
    async def call(self, actor: str, app: str, tool: str, args: dict[str, Any], call_id: str, *, system: bool = False) -> ToolResult:
        """``system=True``: an environment-dynamics action (event), not subject to the actor's tool access."""


class Backend(ABC):
    name: ClassVar[str]

    def __init__(self, spec: WorldSpec, params: dict[str, Any], *, workdir: Path):
        self.spec = spec
        self.params = dict(params)
        self.workdir = workdir
        self.world: World | None = None

    @property
    @abstractmethod
    def capabilities(self) -> BackendCapabilities: ...

    @abstractmethod
    async def provision(self) -> None:
        """Bring up containers / processes. Called once before Apps are created."""

    @abstractmethod
    def exec_for(self, app: AppSpec) -> Exec: ...

    async def pause(self) -> None:
        """Freeze the World's processes (a backend with ``capabilities.pausable``)."""
        raise NotImplementedError(f"{self.name} backend cannot pause")

    async def resume(self) -> None:
        raise NotImplementedError(f"{self.name} backend cannot pause")

    @abstractmethod
    def base_path(self, app: AppSpec) -> str:
        """Directory that relative App paths resolve against, inside the App's exec."""

    async def after_apps_started(self, world: World) -> None:
        """Called once Apps are up (e.g. to start a gateway in front of their MCP servers)."""

    def router(self) -> ToolRouter | None:
        return None

    def delegated_events(self) -> set[str]:
        """Event ids the backend executes itself (the World's scheduler skips them)."""
        return set()

    async def snapshot_state(self) -> dict[str, Any]:
        return {}

    async def restore_state(self, state: dict[str, Any]) -> None:
        pass

    async def after_step(self, world: World) -> None:
        pass

    async def apply_permission(self, world: World, actor: str, *, grant: list[str] | None = None, revoke: list[str] | None = None) -> None:
        """A permission event the World's own scheduler fired (not delegated to the backend), on a backend that enforces
        tool access itself (``capabilities.native_acl``): the grant or revocation is made where access is checked. The
        World's ACL records it either way; a backend without native ACL needs nothing here."""

    async def on_reset(self, world: World) -> None:
        pass

    # Set by container backends that harden the World (``crucible.world.guard.isolate.isolate_world``).
    isolation: Any = None

    @property
    def policy_uid(self) -> int | None:
        return self.isolation.uid if self.isolation else None

    @property
    def egress_report(self) -> dict | None:
        return self.isolation.egress if self.isolation else None

    async def quiesce(self) -> dict | None:
        """Kill everything the policy left running (before verification). Backends without isolation: no-op."""
        if self.isolation is None:
            return None
        return await self.isolation.guard.call("quiesce", {"uid": self.isolation.uid})

    async def before_restore(self, world: World) -> None:
        """Nothing of the policy may run while state is rewritten under it."""
        if self.isolation is not None:
            from crucible.world.guard.isolate import before_restore

            await before_restore(self.isolation)

    async def after_restore(self, world: World) -> None:
        """Episode hygiene: the policy's .git back to the trusted copy; /tmp, home, /dev/shm and SysV IPC emptied."""
        if self.isolation is not None:
            from crucible.world.guard.isolate import after_restore

            world._log("state", None, "reclaim", data=await after_restore(self.isolation))

    @abstractmethod
    async def teardown(self) -> None: ...

    def validate(self) -> list[str]:
        """Problems that make this backend unsuitable for the spec (empty = fine)."""
        return []
