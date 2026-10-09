"""Plugin registry.

Pattern from agentenv (``agent_env/config/runtime.py`` ``load_impl`` and the per-kind registries):
every pluggable thing is identified by a ``type`` string and constructed from a ``module:Class``
pointer, resolved lazily so the core never imports a backend, a pack or a framework until it is
used. Duplicate registrations fail loudly at registration time, as in agentenv.

Kinds used so far: ``state_adapter``, ``app_template``, ``world_backend``, ``actor``,
``record_source``, ``transport``, ``pack``. Packs and interop modules register through the
``crucible.plugins`` entry-point group: each entry point is a callable ``register(registry)``.
"""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from importlib.metadata import entry_points
from typing import Any

logger = logging.getLogger(__name__)

ENTRY_POINT_GROUP = "crucible.plugins"


class RegistryError(RuntimeError):
    pass


def load_impl(pointer: str) -> Any:
    """Resolve ``"package.module:Attr"`` to the attribute."""
    if ":" not in pointer:
        raise RegistryError(f"impl pointer must look like 'module:Attr', got {pointer!r}")
    mod_name, _, attr = pointer.partition(":")
    try:
        mod = importlib.import_module(mod_name)
    except ImportError as e:
        raise RegistryError(f"cannot import {mod_name!r} for {pointer!r}: {e}") from e
    obj: Any = mod
    for part in attr.split("."):
        try:
            obj = getattr(obj, part)
        except AttributeError as e:
            raise RegistryError(f"{mod_name!r} has no attribute {attr!r}") from e
    return obj


@dataclass
class _Entry:
    pointer: str | None
    obj: Any = None
    source: str = "builtin"

    def resolve(self) -> Any:
        if self.obj is None:
            assert self.pointer is not None
            self.obj = load_impl(self.pointer)
        return self.obj


@dataclass
class Registry:
    _kinds: dict[str, dict[str, _Entry]] = field(default_factory=dict)
    _plugins_loaded: bool = False

    def register(self, kind: str, name: str, impl: str | Any, *, source: str = "builtin", replace: bool = False) -> None:
        table = self._kinds.setdefault(kind, {})
        if name in table and not replace:
            prev = table[name]
            raise RegistryError(f"{kind} {name!r} already registered by {prev.source}; refusing {source}")
        table[name] = _Entry(pointer=impl if isinstance(impl, str) else None, obj=None if isinstance(impl, str) else impl, source=source)

    def get(self, kind: str, name: str) -> Any:
        self.load_plugins()
        table = self._kinds.get(kind, {})
        if name not in table:
            known = ", ".join(sorted(table)) or "none"
            raise RegistryError(f"no {kind} named {name!r} (known: {known})")
        return table[name].resolve()

    def has(self, kind: str, name: str) -> bool:
        self.load_plugins()
        return name in self._kinds.get(kind, {})

    def names(self, kind: str) -> list[str]:
        self.load_plugins()
        return sorted(self._kinds.get(kind, {}))

    def describe(self) -> dict[str, dict[str, str]]:
        self.load_plugins()
        return {k: {n: (e.pointer or repr(e.obj)) + f"  [{e.source}]" for n, e in sorted(t.items())} for k, t in sorted(self._kinds.items())}

    def load_plugins(self) -> None:
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        for ep in entry_points(group=ENTRY_POINT_GROUP):
            try:
                fn: Callable[[Registry], None] = ep.load()
                fn(self)
            except RegistryError:
                raise
            except Exception as e:  # a broken plugin must not take the core down
                logger.warning("crucible plugin %s failed to load: %s", ep.name, e)


_REGISTRY: Registry | None = None


def get_registry() -> Registry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = Registry()
        _register_builtins(_REGISTRY)
    return _REGISTRY


def reset_registry() -> None:
    global _REGISTRY
    _REGISTRY = None


def _register_builtins(r: Registry) -> None:
    # State adapters
    r.register("state_adapter", "fs", "crucible.world.state.fs:FsAdapter")
    r.register("state_adapter", "fs_git", "crucible.world.state.fs:FsGitAdapter")
    r.register("state_adapter", "sql", "crucible.world.state.sql:SqliteAdapter")
    r.register("state_adapter", "json_doc", "crucible.world.state.json_doc:JsonDocAdapter")
    r.register("state_adapter", "process", "crucible.world.state.process:ProcessAdapter")
    # App templates (generic; packs add their own)
    r.register("app_template", "common/workspace", "crucible.world.apps.workspace:WorkspaceApp")
    r.register("app_template", "common/git_repo", "crucible.world.apps.workspace:GitRepoApp")
    r.register("app_template", "common/sqlite_app", "crucible.world.apps.sqlite_app:SqliteApp")
    r.register("app_template", "common/mailbox", "crucible.world.apps.mailbox:MailboxApp")
    r.register("app_template", "common/doc_library", "crucible.world.apps.doc_library:DocLibraryApp")
    r.register("app_template", "common/json_api", "crucible.world.apps.json_api:JsonApiApp")
    r.register("app_template", "common/toolchain", "crucible.world.apps.toolchain:ToolchainApp")
    r.register("app_template", "openenv/awm_service", "crucible.interop.openenv.awm:AwmServiceApp")
    # World backends: core ships `local`; framework backends live in interop and load lazily.
    r.register("world_backend", "local", "crucible.world.backends.local:LocalBackend")
    r.register("world_backend", "harbor", "crucible.interop.harbor.world_backend:HarborBackend")
    r.register("world_backend", "agentenv", "crucible.interop.agentenv.world_backend:AgentEnvBackend")
    # Actors
    r.register("actor", "policy/scripted", "crucible.actors.policy:ScriptedPolicy")
    r.register("actor", "policy/llm", "crucible.actors.policy:LLMToolPolicy")
    r.register("actor", "simulated/grounded", "crucible.actors.grounded:GroundedResponder")
    r.register("actor", "simulated/scenario", "crucible.actors.simulated:SimulatedActor")
    r.register("actor", "system/ci", "crucible.actors.scripted:CheckRunnerActor")
    r.register("actor", "system/relay", "crucible.actors.scripted:RelayActor")
    r.register("actor", "human", "crucible.actors.human:HumanActor")
    r.register("actor", "peer", "crucible.actors.human:PeerActor")
    # Transports for human / peer actors
    r.register("transport", "file_queue", "crucible.actors.transports:FileQueueTransport")
    r.register("transport", "callable", "crucible.actors.transports:CallableTransport")
    r.register("transport", "a2a", "crucible.interop.agentenv.a2a:A2ATransport")
