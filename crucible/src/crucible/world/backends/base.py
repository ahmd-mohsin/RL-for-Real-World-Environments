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
    isolation: str = "none"  # none | container | vm
    gpus: bool = False
    network_enforced: bool = False
    native_clock: bool = False  # backend keeps virtual time itself (agentenv gateway)
    native_acl: bool = False  # backend enforces per-actor tool access itself
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

    @abstractmethod
    def base_path(self, app: AppSpec) -> str:
        """Directory that relative App paths resolve against, inside the App's exec."""

    async def after_apps_started(self, world: World) -> None:  # noqa: B027 - optional hook
        """Called once Apps are up (e.g. to start a gateway in front of their MCP servers)."""

    def router(self) -> ToolRouter | None:
        return None

    def delegated_events(self) -> set[str]:
        """Event ids the backend executes itself (the World's scheduler skips them)."""
        return set()

    async def snapshot_state(self) -> dict[str, Any]:
        return {}

    async def restore_state(self, state: dict[str, Any]) -> None:  # noqa: B027 - optional
        pass

    async def after_step(self, world: World) -> None:  # noqa: B027 - optional: import backend-side happenings
        pass

    async def on_reset(self, world: World) -> None:  # noqa: B027 - optional: fresh-episode hook (after state restore)
        pass

    @abstractmethod
    async def teardown(self) -> None: ...

    def validate(self) -> list[str]:
        """Problems that make this backend unsuitable for the spec (empty = fine)."""
        return []
