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
    # entry point name -> why it did not load (agentenv ``plugins/_registration.py:merge`` records every skip the same way)
    failures: dict[str, str] = field(default_factory=dict)

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
            why = "".join(f"; plugin {ep!r} failed to load: {reason}" for ep, reason in sorted(self.failures.items()))
            raise RegistryError(f"no {kind} named {name!r} (known: {known}){why}")
        return table[name].resolve()

    def has(self, kind: str, name: str) -> bool:
        self.load_plugins()
        return name in self._kinds.get(kind, {})

    def names(self, kind: str) -> list[str]:
        self.load_plugins()
        return sorted(self._kinds.get(kind, {}))

    def describe(self) -> dict[str, dict[str, str]]:
        self.load_plugins()
        out = {k: {n: (e.pointer or repr(e.obj)) + f"  [{e.source}]" for n, e in sorted(t.items())} for k, t in sorted(self._kinds.items())}
        if self.failures:
            out["failed_plugins"] = dict(sorted(self.failures.items()))
        return out

    def load_plugins(self) -> None:
        """Plugins are staged, then merged (R-13; PATTERN: agentenv ``plugins/_registration.py:merge``, Apache-2.0), so
        the result does not depend on install order. Each ``register(registry)`` runs against a ``_Staged`` recorder: one
        that raises adds nothing. A name two plugins claim is withheld from both, and a built-in name from any plugin;
        each is recorded in ``failures`` (the plugin's other names still load). Merged entries carry the installed
        distribution as ``source`` (``name==version``). A later lookup names every failure instead of "known: none"."""
        if self._plugins_loaded:
            return
        self._plugins_loaded = True
        staged: dict[str, _Staged] = {}
        for ep in sorted(entry_points(group=ENTRY_POINT_GROUP), key=lambda e: (e.name, _dist_label(e) or "")):
            rec = _Staged(source=_dist_label(ep) or f"plugin:{ep.name}")
            try:
                fn: Callable[[Any], None] = ep.load()
                fn(rec)
            except Exception as e:  # noqa: BLE001 - a broken plugin must not take the core down
                self.failures[ep.name] = f"{type(e).__name__}: {e}"[:500]
                logger.warning("crucible plugin %s failed to load: %s", ep.name, e)
                continue
            staged[ep.name] = rec
        claimants: dict[tuple[str, str], list[str]] = {}
        for ep_name, rec in staged.items():
            for kind, name, _ in rec.claims:
                claimants.setdefault((kind, name), []).append(ep_name)
        for ep_name, rec in staged.items():
            refused = []
            for kind, name, impl in rec.claims:
                if len(owners := claimants[(kind, name)]) > 1:
                    refused.append(f"{kind} {name!r} withheld: also claimed by {', '.join(staged[o].source for o in owners if o != ep_name)}")
                elif name in self._kinds.get(kind, {}):
                    refused.append(f"{kind} {name!r} is built in ({self._kinds[kind][name].source})")
                else:
                    self.register(kind, name, impl, source=rec.source)
            if refused:
                self.failures[ep_name] = "; ".join(refused)[:500]
                logger.warning("crucible plugin %s: %s", ep_name, self.failures[ep_name])


@dataclass
class _Staged:
    """What one plugin's ``register(registry)`` asks for, held until every plugin has run. It has the registry's
    ``register`` signature; a name registered twice by the same plugin is that plugin's bug and fails it."""

    source: str
    claims: list[tuple[str, str, str | Any]] = field(default_factory=list)

    def register(self, kind: str, name: str, impl: str | Any, *, source: str = "", replace: bool = False) -> None:
        if any((k, n) == (kind, name) for k, n, _ in self.claims):
            raise RegistryError(f"{kind} {name!r} registered twice by {self.source}")
        self.claims.append((kind, name, impl))


def _dist_label(ep: Any) -> str | None:
    """``name==version`` of the distribution that installed ``ep``, when known."""
    dist = getattr(ep, "dist", None)
    if dist is None:
        return None
    try:
        return f"{dist.name}=={dist.version}"
    except Exception:  # noqa: BLE001 - broken metadata leaves the label unknown
        return None


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
    r.register("state_adapter", "postgres", "crucible.world.state.postgres:PostgresAdapter")
    r.register("state_adapter", "json_doc", "crucible.world.state.json_doc:JsonDocAdapter")
    r.register("state_adapter", "browser", "crucible.world.apps.browser:BrowserStateAdapter")
    r.register("state_adapter", "process", "crucible.world.state.process:ProcessAdapter")
    r.register("state_adapter", "agentenv_data", "crucible.interop.agentenv.mcp_env:AgentEnvDataAdapter")
    # App templates (generic; packs add their own)
    r.register("app_template", "common/workspace", "crucible.world.apps.workspace:WorkspaceApp")
    r.register("app_template", "common/git_repo", "crucible.world.apps.workspace:GitRepoApp")
    r.register("app_template", "common/sqlite_app", "crucible.world.apps.sqlite_app:SqliteApp")
    r.register("app_template", "common/mailbox", "crucible.world.apps.mailbox:MailboxApp")
    r.register("app_template", "common/doc_library", "crucible.world.apps.doc_library:DocLibraryApp")
    r.register("app_template", "common/json_api", "crucible.world.apps.json_api:JsonApiApp")
    r.register("app_template", "common/toolchain", "crucible.world.apps.toolchain:ToolchainApp")
    r.register("app_template", "common/postgres_app", "crucible.world.apps.postgres_app:PostgresApp")
    r.register("app_template", "common/control", "crucible.world.apps.control:ControlApp")
    r.register("app_template", "common/desktop", "crucible.world.apps.desktop:DesktopApp")
    r.register("app_template", "common/vm", "crucible.world.apps.vm:VMApp")
    r.register("app_template", "common/mcp_server", "crucible.world.apps.mcp_server:McpServerApp")
    r.register("app_template", "common/website", "crucible.world.apps.website:WebsiteApp")
    r.register("app_template", "common/browser", "crucible.world.apps.browser:BrowserApp")
    r.register("app_template", "common/ci", "crucible.world.apps.ci:CiApp")
    # CI runners (where hidden checks execute: never the policy's sandbox)
    r.register("ci_runner", "local", "crucible.world.apps.ci:LocalCiRunner")
    r.register("ci_runner", "docker", "crucible.world.apps.ci:DockerCiRunner")
    r.register("app_template", "openenv/awm_service", "crucible.interop.openenv.awm:AwmServiceApp")
    r.register("app_template", "openenv/env", "crucible.interop.openenv.import_env:OpenEnvApp")
    r.register("app_template", "agentenv/mcp_env", "crucible.interop.agentenv.mcp_env:AgentEnvMcpApp")
    r.register("app_template", "harbor/terminal", "crucible.interop.harbor.inplace:TerminalApp")
    r.register("app_template", "nemo_gym/resources", "crucible.interop.nemo_gym.import_server:NemoGymApp")
    # World backends: core ships `local`; framework backends live in interop and load lazily.
    r.register("world_backend", "local", "crucible.world.backends.local:LocalBackend")
    r.register("world_backend", "ssh_exec", "crucible.world.backends.ssh:SSHExecBackend")
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
    # Connectors (M3); domain connectors register from packs
    r.register("connector", "file_drop", "crucible.connectors.file_drop:FileDropConnector")
    r.register("connector", "sql_audit_log", "crucible.connectors.sql_audit_log:SqlAuditLogConnector")
    r.register("connector", "otel_import", "crucible.connectors.otel_import:OtelConnector")
    r.register("connector", "capture_proxy", "crucible.connectors.capture_proxy:CaptureLogConnector")
    r.register("connector", "chat_messages", "crucible.connectors.chat_messages:ChatMessagesConnector")
    r.register("connector", "atif", "crucible.interop.harbor.atif_import:AtifConnector")
    # Transports for human / peer actors
    r.register("transport", "file_queue", "crucible.actors.transports:FileQueueTransport")
    r.register("transport", "callable", "crucible.actors.transports:CallableTransport")
    r.register("transport", "a2a", "crucible.interop.agentenv.a2a:A2ATransport")
