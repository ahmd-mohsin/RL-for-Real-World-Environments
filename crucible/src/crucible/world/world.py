"""The World runtime (ARCHITECTURE M1): a gym-like environment over Apps, Actors, a Clock and Events.

    world = World(spec, instructions={"agent": "..."})
    await world.start()                      # provision backend + apps, capture the initial state
    obs = await world.reset()                # {policy actor: Observation(instruction, tools)}
    res = await world.step("agent", ToolCall(...))
    ref = await world.snapshot()             # WorldRef: every App + clock + events + ACL + inboxes + actors
    await world.restore(ref)                 # exact; verified by fingerprints
    view = world.verifier_view()             # State Adapters — never handed to a policy

Non-policy actors (simulated people, system actors) are part of the environment's dynamics: a
``Responder`` attached with ``world.attach`` is run *inside* ``step`` when it receives a message or
an event, under its own tool permissions, and its replies land in the sender's inbox. A policy only
ever calls ``reset`` / ``step``.
"""

from __future__ import annotations

import asyncio
import contextlib
import difflib
import fnmatch
import json
import logging
import shutil
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

from crucible.core.agent_events import annotate_probes, app_writes, writes_of
from crucible.core.ids import derive_seed, new_id, sha256_text
from crucible.core.registry import get_registry
from crucible.core.schema.actions import Action, Done, Inbound, MediaRef, Message, Observation, StepResult, ToolCall, ToolResult, ToolSpec
from crucible.core.schema.base import matches_content_id
from crucible.core.schema.run import Failure
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.state import Query, QueryResult, StateDiff, StateRef, WorldRef
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource
from crucible.core.schema.world import ActorSpec, EventSpec, WorldSpec
from crucible.core.store import Store, get_store
from crucible.core.text import safe_json, safe_text, unencodable
from crucible.world import lifecycle
from crucible.world.apps.base import App, AppContext
from crucible.world.backends.base import Backend
from crucible.world.clock import AccessControl, EventScheduler, Fired, VirtualClock, parse_duration
from crucible.world.exec import container_lost_cause

MAX_INNER_TURNS = 8  # tool calls a responder may make while handling one inbound batch

logger = logging.getLogger(__name__)
INFRA_ERRORS = frozenset({"transport", "gateway_error", "internal_error", "timeout", "session_lost"})
# how long past limits.tool_timeout_s the World waits for a tool: its own deadline result (an in-container kill, then the
# transport backstops of +15 s) arrives first and is scored (NeMo-Gym: outer = tool timeout + cleanup + 30 s)
TOOL_GRACE_S = 60.0
# World.close bounds each App's stop (M1 audit R-9; CI's own container removal allows 120 s); the whole close is bounded by
# ``World.close_timeout_s`` (M10 lab audit R-6)
APP_STOP_S = 150.0


def _argument_problem(args: Any, schema: dict[str, Any] | None) -> str | None:
    """Why ``args`` do not fit the tool's JSON schema (None when they do). verifiers' wording for non-objects."""
    if not isinstance(args, dict):
        return f"tool arguments must be a JSON object, got {type(args).__name__}"
    if "__unparseable__" in args:
        return f"tool arguments must be a JSON object, got {args['__type__']}" if "__type__" in args else "the arguments were not valid JSON"
    if not schema:
        return None
    try:
        from jsonschema import Draft202012Validator
    except ImportError:  # pragma: no cover - jsonschema ships with the MCP SDK
        return None
    errors = sorted(Draft202012Validator(schema).iter_errors(args), key=lambda e: list(e.path))
    if not errors:
        return None
    return "; ".join(f"{'/'.join(map(str, e.path)) or '(arguments)'}: {e.message}" for e in errors[:5])


def _event_json(ev: TraceEvent) -> str:
    """One trace line. Events are escaped when logged; one that still cannot be encoded (a prefix from elsewhere) is
    escaped here rather than failing the episode."""
    try:
        return ev.model_dump_json(exclude_none=True)
    except Exception:  # noqa: BLE001 - PydanticSerializationError on an unencodable string
        return json.dumps(safe_json(ev.model_dump(mode="python", exclude_none=True)), ensure_ascii=False, default=str)


def _inbound_json(m: Inbound) -> dict[str, Any]:
    d = m.model_dump(mode="json", by_alias=True)
    if d.get("attachments") is None:
        d.pop("attachments", None)  # no media: the snapshot (and its fingerprint) is what it was before attachments existed
    return d


class WorldError(RuntimeError):
    pass


@dataclass
class ActorTurn:
    """What a non-policy actor sees when the World asks it to act."""

    actor: str
    time: str | None
    inbox: list[Inbound]
    tool_results: list[ToolResult] = field(default_factory=list)
    tools: list[ToolSpec] = field(default_factory=list)


@runtime_checkable
class Responder(Protocol):
    actor_id: str

    async def respond(self, turn: ActorTurn) -> list[Action]: ...

    def state_dict(self) -> dict[str, Any]: ...

    def load_state_dict(self, state: dict[str, Any]) -> None: ...


class VerifierView:
    """Read-only, verifier-side access to every App's state."""

    def __init__(self, world: World):
        self._world = world

    @property
    def apps(self) -> list[str]:
        return list(self._world.apps)

    def adapter(self, app: str):
        self._world._touched += 1  # a query may run a command (op run): a later restore never trusts the old state (R-5)
        return self._world.apps[app].state

    async def query(self, app: str, q: Query | dict[str, Any]) -> QueryResult:
        await self._world.apps[app].before_snapshot()
        return await self.adapter(app).query(q if isinstance(q, Query) else Query.model_validate(q))

    async def fingerprints(self) -> dict[str, str]:
        await self._world.flush()
        apps = self._world.apps
        fps = await asyncio.gather(*(a.state.fingerprint() for a in apps.values()))
        return dict(zip(apps, fps, strict=True))

    def seat_rewards(self) -> dict[str, float]:
        """Per-seat rewards of a turn-based World (the rules App's, or empty)."""
        return dict(self._world.turns.rewards) if self._world.turns is not None else {}

    @property
    def start(self) -> WorldRef:
        """The WorldRef the episode started from (the reset target; the initial snapshot before any reset)."""
        ref = self._world.episode_start or self._world.initial
        if ref is None:
            raise WorldError("no episode-start WorldRef: start() the world first")
        return ref

    async def changes(self, app: str) -> StateDiff:
        """What the episode changed in one App: its episode-start state against a snapshot taken now (rows inserted and
        deleted per table for databases, files for trees); collateral-damage checks read this."""
        adapter = self.adapter(app)
        return await adapter.diff(self.start.apps[app], await adapter.snapshot())

    async def unchanged(self, *apps: str) -> dict[str, bool]:
        """Whether each App (default: all) still has its episode-start state: the check for infeasible requests, where
        the right outcome is to refuse and touch nothing (EnterpriseOps' refusal subset, MiMo ``act: none``)."""
        start = self._world.initial
        if start is None:
            raise WorldError("no initial WorldRef: start() the world first")
        await self._world.flush()
        names = apps or tuple(self._world.apps)
        fps = await asyncio.gather(*(self._world.apps[a].state.fingerprint() for a in names))
        return {a: fp == start.apps[a].fingerprint for a, fp in zip(names, fps, strict=True)}


class World:
    # M10 lab audit R-6: what ``close`` may take before it is given up on (recorded in the attempt scope, never raised): a backend
    # teardown that hangs (a Harbor compose ``down``) must not hold an attempt, or the lease behind it, for ever
    close_timeout_s: float = lifecycle.WORLD_CLOSE_TIMEOUT_S

    def __init__(
        self,
        spec: WorldSpec,
        *,
        instructions: dict[str, str] | None = None,
        store: Store | None = None,
        workdir: Path | None = None,
        pack_root: Path | None = None,
        backend: str | None = None,
        backend_params: dict[str, Any] | None = None,
        interceptors: list | None = None,
        instruction_media: dict[str, list[MediaRef]] | None = None,
        noise: Any = None,
    ):
        """``noise``: a ``world.noise.WorldNoise`` (M12-infra D): perturbs the policy's view of instructions and tool
        results (never state or the verifier's view); every perturbation is a ``noise`` trace event."""
        self.spec = spec
        self.noise = noise
        if interceptors is None:  # m13 C-4: the spec's rules (default: remote code fetch, verifier paths); its allowlist passes
            from crucible.world.intercept import DEFAULT_RULES, RuleInterceptor

            names = [r.id for r in DEFAULT_RULES] if spec.intercept is None else list(spec.intercept)
            unknown = sorted(set(names) - {r.id for r in DEFAULT_RULES})
            if unknown:
                raise WorldError(f"unknown intercept rules {unknown} (known: {[r.id for r in DEFAULT_RULES]})")
            rules = tuple(r for r in DEFAULT_RULES if r.id in names)
            interceptors = [RuleInterceptor(rules, allow_hosts=tuple(spec.allowed_hosts))] if rules else []
        self.interceptors = list(interceptors)
        self.spec_hash = spec.content_hash()
        # the spec as it was hashed: a ref's world id is checked against this, never the live (since mutated) spec (R-21)
        self._spec_doc = spec.model_dump(mode="json", exclude_none=True, by_alias=True)
        self.instructions = dict(instructions or {})
        self.instruction_media = {a: list(m) for a, m in (instruction_media or {}).items()}  # actor -> media with its instruction
        self.store = store or get_store()
        self._own_workdir = workdir is None
        self.workdir = (workdir or Path(tempfile.mkdtemp(prefix="crucible-world-"))).resolve()
        self.workdir.mkdir(parents=True, exist_ok=True)
        self.pack_root = pack_root
        name = backend or spec.backend
        params = {**spec.backend_params, **(backend_params or {})}
        backend_cls = get_registry().get("world_backend", name)
        self.backend: Backend = backend_cls(spec, params, workdir=self.workdir)
        self.backend.world = self
        self.clock = VirtualClock(spec.clock)
        self.scheduler = EventScheduler(spec.events, self.clock)
        self.acl = AccessControl({a.id: a.tools for a in spec.actors})
        self.identities = {a.id: a.identity for a in spec.actors if a.identity is not None}  # M1 item 3: who each actor is
        self._exposed: dict[str, frozenset[str]] = {}  # actor -> tools its catalog shows (actors with a CatalogSpec only)
        self.apps: dict[str, App] = {}
        self.responders: dict[str, Responder] = {}
        self.inboxes: dict[str, list[Inbound]] = {a.id: [] for a in spec.actors}
        self.done: set[str] = set()
        # the pre-grade cut (M11 lab audit R-2, ``cut_policy``): the actors cut, the ones the cut itself made done, and every
        # step one of them tried after it (refused, a ``late_call`` trace event)
        self._cut: set[str] = set()
        self._cut_added: set[str] = set()
        self.late_calls: list[dict[str, Any]] = []
        self.tool_counts: dict[str, int] = {}  # LimitsSpec.tool_quota pattern -> calls used this episode
        self.step_index = 0
        self.log: list[TraceEvent] = []
        self.trace_sink: Path | None = None  # stream every trace event here as it happens (long episodes: a crash loses nothing)
        self.episode_id = new_id("ep")
        self.initial: WorldRef | None = None
        self.episode_start: WorldRef | None = None  # the WorldRef the current episode was reset to (protected-path baseline)
        self._tools: dict[str, list[ToolSpec]] = {}
        self._step_lock = asyncio.Lock()
        self.finished: dict[str, str] = {}  # non-policy actors that ended their part (user_stop, out_of_scope, ...)
        self.failures: list[Failure] = []  # infrastructure faults this episode (cleared by reset and restore)
        self._acl_changed: set[str] = set()
        self._started = False
        self._closed = False
        self._closing: asyncio.Future | None = None  # M10 lab audit R-6: the one shielded close every caller shares
        self._close_exc: BaseException | None = None  # what ``_close`` raised, for the first caller to see
        self._scope: lifecycle.Scope | None = None  # the lifecycle scope this World was started in: its close is recorded there
        # restores a WorldRef already in place skips (M1 audit R-5): tool calls and verifier queries so far, and the ref the
        # state was last set to with that count; ``restored_apps``: the Apps the last restore actually restored
        self._touched = 0
        self._settled: tuple[str, int] | None = None
        self.restored_apps: list[str] = []
        self.health: dict[str, Any] = {}  # app id -> last health check report (M1 item 6)
        self.turns = None  # TurnManager when the spec has turns (M1 item 7)
        if spec.turns is not None:
            from crucible.world.turns import TurnManager

            self.turns = TurnManager(self, spec.turns)
        self._serial = None  # SerialLock while a serial world runs

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> WorldRef:
        problems = self.backend.validate()
        if problems:
            raise WorldError("; ".join(problems))
        self._scope = lifecycle.current()  # M10 lab audit R-6: an attempt scope closes (and records) every World started under it
        self._scope.track_world(self)
        try:
            if self.spec.serial:  # one episode of this world at a time on this host (shared live services)
                from crucible.world.health import SerialLock

                self._serial = SerialLock(self.spec.world)
                waited = await self._serial.acquire()
                if waited > 1:
                    logger.info("world %s: waited %.1fs for the serial lock", self.spec.world, waited)
            return await self._start()
        except BaseException:
            # A failed start must not leak containers, gateways or servers (found: a crashed App.provision
            # left a Harbor sandbox running). M10 lab audit R-6: a failing close never replaces the start's own error (a
            # HealthcheckError is retryable infra; a teardown error would have masked it)
            with contextlib.suppress(Exception):
                await self.close()
            raise

    async def _start(self) -> WorldRef:
        await self.backend.provision()
        await self._seed_bundles()
        reg = get_registry()
        for app_spec in self.spec.apps:
            ctx = AppContext(
                exec=self.backend.exec_for(app_spec),
                store=self.store,
                base_path=self.backend.base_path(app_spec),
                host_workdir=self.workdir,
                clock=self.clock,
                pack_root=self.pack_root,
                extra={"peers": self.apps, "seed": self.spec.seed, "identities": self.identities},  # peers: live dict (a browser reaches a website's URL)
                seed=derive_seed(self.spec.seed, app_spec.id),
                tool_timeout_s=self.spec.limits.tool_timeout_s,
            )
            self.apps[app_spec.id] = reg.get("app_template", app_spec.template)(app_spec, ctx)
        for app in self.apps.values():
            await app.provision()
        for app in self.apps.values():  # spec order: dependencies first (a browser after the sites it opens)
            await app.start()
            if app.spec.healthcheck is not None:  # ready before anything depends on it (Harbor: before agent setup)
                await self.check_health(app.id)
        for app_id, app in self.apps.items():
            self._tools[app_id] = await app.list_tools()
        self._exposed = self._catalogs()
        await self.backend.after_apps_started(self)
        delegated = self.backend.delegated_events()
        if delegated:
            self.scheduler = EventScheduler([e for e in self.spec.events if e.id not in delegated], self.clock)
        self._started = True
        if self.turns is not None:
            await self.turns.begin()
        self.initial = await self.snapshot()
        self._settled = (self.initial.fingerprint, self._touched)
        return self.initial

    async def _seed_bundles(self) -> None:
        """Copy ``backend_params.bundles`` ([{src, dest}]) into the world before Apps exist.

        Used for imported environments whose files must sit together (MiMo: tools/ next to system/).
        ``src`` is a host directory (relative paths resolve against pack_root); ``dest`` resolves like
        App paths (relative to the backend's base path).
        """
        bundles = self.backend.params.get("bundles", [])
        if not bundles:
            return
        import tarfile
        import uuid

        exec_ = self.backend.exec_for(self.spec.apps[0])
        base = self.backend.base_path(self.spec.apps[0])
        for b in bundles:
            src = Path(b["src"])
            if not src.is_absolute():
                if self.pack_root is None:
                    raise WorldError(f"relative bundle src {b['src']!r} needs pack_root")
                src = self.pack_root / src
            dest = b["dest"] if b["dest"].startswith("/") else f"{base.rstrip('/')}/{b['dest']}"
            if exec_.is_local:
                shutil.copytree(src, dest, symlinks=True, dirs_exist_ok=True,
                                ignore=shutil.ignore_patterns(*b.get("ignore", ["__pycache__", ".cache"])))
                continue
            with tempfile.TemporaryDirectory() as d:
                tar = Path(d) / "bundle.tar"
                skip = set(b.get("ignore", ["__pycache__", ".cache"]))
                with tarfile.open(tar, "w") as tf:
                    for p in sorted(src.rglob("*")):
                        if not skip.intersection(p.relative_to(src).parts):
                            tf.add(p, arcname=str(p.relative_to(src)), recursive=False)
                remote = f"{exec_.staging_dir}/.crucible_bundle_{uuid.uuid4().hex}.tar"
                await exec_.push_file(tar, remote)
                out = await exec_.run(f"mkdir -p '{dest}' && tar -xf {remote} -C '{dest}' && rm -f {remote}")
                if not out.ok:
                    raise WorldError(f"bundle {src} -> {dest} failed: {out.stderr[-400:]}")

    async def pause(self) -> None:
        """Freeze the World while the trainer holds the GPU (m13 S6): its processes stop (cgroup freezer), its state is
        untouched, and its virtual clock does not advance until ``resume``."""
        if not self.backend.capabilities.pausable:
            raise WorldError(f"{self.backend.name} backend cannot pause this World")
        self.clock.pause()
        await self.backend.pause()

    async def resume(self) -> None:
        await self.backend.resume()
        self.clock.resume()

    async def close(self) -> None:
        """Stop the Apps, tear the backend down, release the serial lock. M10 lab audit R-6: shielded (a cancelled caller lets the
        close run to its end, bounded by ``close_timeout_s``, and is cancelled after it), idempotent (a second or concurrent caller
        awaits the same close; a separate verifier closes the policy's World before the harness's cleanup does) and bounded. It
        raises only what ``_close`` raised before R-6 (a backend teardown error), to the first caller; a close that ran past
        ``close_timeout_s`` is recorded in the attempt scope it started in (``Scope.world_closed``) and logged, never raised: a
        verifier's grade that was already computed must not become a failure because a teardown was slow."""
        if self._closing is None:
            self._closed = True
            self._closing = lifecycle.hold(self._close_bounded())
        closing = self._closing
        if not closing.done():
            try:
                await asyncio.shield(closing)
            except asyncio.CancelledError:
                await lifecycle.finish(closing, self.close_timeout_s)
                raise
        exc, self._close_exc = self._close_exc, None  # the first caller to look takes it
        if exc is not None:
            raise exc

    async def _close_bounded(self) -> None:
        """The one close, in its own task: it never raises (``close`` re-raises ``_close_exc``) so a shielded task nobody awaits any
        more cannot log 'exception never retrieved'."""
        error: str | None = None
        bound = asyncio.timeout(self.close_timeout_s)
        try:
            async with bound:
                await self._close()
        except Exception as e:  # noqa: BLE001 - recorded for the attempt's teardown.json
            if isinstance(e, TimeoutError) and bound.expired():  # the bound fired (a TimeoutError a teardown raised itself is its own error)
                error = f"closing world {self.spec.world} ran past {self.close_timeout_s:g}s"
                logger.warning("%s", error)
            else:
                error = f"{type(e).__name__}: {e}"
                logger.warning("world %s: close failed: %s", self.spec.world, error)
                self._close_exc = e  # re-raised to the first caller, as before R-6
        finally:
            if self._serial is not None:
                self._serial.release()
                self._serial = None
            if self._scope is not None:
                self._scope.world_closed(self, error)

    async def _close(self) -> None:
        try:
            for app in reversed(list(self.apps.values())):  # dependents first: a browser before the sites it opens
                try:
                    # bounded each (M1 audit R-9): one hung stop must not use up ``close_timeout_s`` before the backend's teardown
                    await asyncio.wait_for(app.stop(), APP_STOP_S)
                except Exception as e:  # noqa: BLE001 - teardown is best effort, but never silent (a timeout included)
                    logger.warning("world %s: stopping app %s failed: %r", self.spec.world, app.id, e)
            await self.backend.teardown()
        finally:
            if self._own_workdir:  # M10 lab audit R-6: removed even when the teardown raised or ran past its bound
                shutil.rmtree(self.workdir, ignore_errors=True)

    async def __aenter__(self) -> Self:
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    def attach(self, responder: Responder) -> None:
        actor = self.spec.actor(responder.actor_id)
        if actor.kind == "policy":
            raise WorldError(f"{actor.id!r} is a policy actor; policies act through step(), not attach()")
        if self._started:  # its memory would be missing from the initial snapshot, so reset() could not rewind it
            raise WorldError(f"attach {actor.id!r} before start(): the initial WorldRef must hold every responder's state")
        self.responders[actor.id] = responder

    # ------------------------------------------------------------------ tools
    def all_tools(self) -> list[ToolSpec]:
        return [t for app in self.spec.apps for t in self._tools.get(app.id, [])]

    def tools_for(self, actor: str) -> list[ToolSpec]:
        if self.backend.capabilities.native_acl:
            tools = self.backend.allowed_tools(actor)  # type: ignore[attr-defined]
        else:
            tools = [t for t in self.all_tools() if self.acl.allows(actor, t.qualified)]
        shown = self._exposed.get(actor)
        return tools if shown is None else [t for t in tools if t.qualified in shown]

    def _catalogs(self) -> dict[str, frozenset[str]]:
        """Each catalogued actor's shown tools: its oracle tools plus ``distractors`` drawn from the rest of the World's
        tools with a seed derived from the World seed and the actor (EnterpriseOps-Gym ``plus_N_tools``), minus ``hide``.
        Drawn over every tool, not the ACL's current grants, so a later grant cannot change which distractors exist."""
        import random

        out: dict[str, frozenset[str]] = {}
        names = sorted(t.qualified for t in self.all_tools())
        for a in self.spec.actors:
            c = a.catalog
            if c is None:
                continue
            hidden = {n for n in names if any(fnmatch.fnmatchcase(n, p) for p in c.hide)}
            oracle = {n for n in names if n not in hidden and any(fnmatch.fnmatchcase(n, p) for p in c.oracle)}
            pool = [n for n in names if n not in hidden and n not in oracle]
            k = len(pool) if c.distractors < 0 else min(c.distractors, len(pool))
            out[a.id] = frozenset(oracle | set(random.Random(derive_seed(self.spec.seed, f"catalog:{a.id}")).sample(pool, k)))
        return out

    def _rbac_denies(self, actor: str, qualified: str) -> list[str] | None:
        """The actor's roles when RBAC refuses this call, else None (no identity or no rbac: not RBAC's concern)."""
        ident = self.identities.get(actor)
        if self.spec.rbac is None or ident is None:
            return None
        roles = [self.spec.rbac[r] for r in ident.roles]
        allowed = any(any(fnmatch.fnmatchcase(qualified, p) for p in r.allow) for r in roles)
        denied = any(any(fnmatch.fnmatchcase(qualified, p) for p in r.deny) for r in roles)
        return None if allowed and not denied else list(ident.roles)

    @staticmethod
    def _split(function_name: str) -> tuple[str, str]:
        name = function_name
        if "." not in name and "__" in name:
            name = name.replace("__", ".", 1)
        app, sep, tool = name.partition(".")
        if not sep:
            raise WorldError(f"tool name {function_name!r} must be '<app>.<tool>'")
        return app, tool

    async def _call_tool(self, actor: str, call: ToolCall, *, system: bool = False) -> ToolResult:
        """The single choke point for every tool call (policy, simulated people, events): name and ACL checks,
        argument validation against the tool's schema, a deadline, crash containment, output bounding, and the
        fault label that decides whether a failure is the policy's (scored) or the infrastructure's (masked)."""
        self._touched += 1  # any tool call may change state a fingerprint cannot see (R-5)

        def fail(name: str, content: str, error: str, fault: str = "policy") -> ToolResult:
            return ToolResult(tool_call_id=safe_text(call.tool_call_id), function_name=safe_text(name), ok=False, content=safe_text(content), error=error, fault=fault)

        # M14 campaign B1: a string UTF-8 cannot encode (a lone surrogate) is the caller's malformed call, refused before
        # anything runs; the episode continues and is scored (it raised while the trace was written and masked the episode)
        bad = "function name" if unencodable(call.function_name) is not None else (
            "tool_call_id" if unencodable(call.tool_call_id) is not None else unencodable(call.arguments))
        if bad is not None:
            name = safe_text(call.function_name)
            return fail(name, f"Invalid arguments for {name}: {bad} holds a string that is not valid UTF-8 (a lone surrogate). "
                              "Resend the call with valid text.", "invalid_arguments")
        try:
            app_id, tool = self._split(call.function_name)
        except WorldError as e:
            # the valid names and the nearest one (``shell`` -> ``repo.shell``): a hint, never an alias, so the policy acts
            valid = sorted(t.qualified for t in self.tools_for(actor))
            bare = safe_text(call.function_name).lower()
            near = [q for q in valid if q.rsplit(".", 1)[-1] == bare] or difflib.get_close_matches(bare, valid, n=1, cutoff=0.4) or difflib.get_close_matches(
                bare, [q.rsplit(".", 1)[-1] for q in valid], n=1, cutoff=0.4)
            near = [next((q for q in valid if q.endswith("." + n)), n) if "." not in n else n for n in near]
            hint = (f" Did you mean {near[0]}?" if near else "") + (f" Valid tools: {', '.join(valid)}." if valid else "")
            return fail(call.function_name, str(e) + "." + hint, "bad_tool_name")
        qualified = f"{app_id}.{tool}"
        if app_id not in self.apps:
            return fail(qualified, f"Unknown app: {app_id}", "unknown_app")
        spec = next((t for t in self._tools.get(app_id, []) if t.name == tool), None)
        if spec is not None and not system and actor in self._exposed and qualified not in self._exposed[actor]:
            spec = None  # outside the actor's catalog: the tool does not exist for it
        if spec is None:
            near = difflib.get_close_matches(qualified, [t.qualified for t in self.tools_for(actor)], n=3, cutoff=0.6)
            return fail(qualified, f"Unknown tool: {qualified}" + (f". Did you mean: {', '.join(near)}?" if near else ""), "unknown_tool")
        if not system and not self.backend.capabilities.native_acl and not self.acl.allows(actor, qualified):
            return fail(qualified, f"Permission denied: {qualified}", "permission_denied")
        if not system and (roles := self._rbac_denies(actor, qualified)) is not None:
            return fail(qualified, f"Permission denied: role(s) {', '.join(roles) or '(none)'} may not call {qualified}", "permission_denied")
        problem = _argument_problem(call.arguments, spec.input_schema)
        if problem:
            return fail(qualified, f"Invalid arguments for {qualified}: {problem}. Resend the call with a JSON object matching the tool's schema.", "invalid_arguments")
        if not system and self.interceptors:  # m13 C-4: an intercepted call is never executed; the episode continues
            hit = next((h for ic in self.interceptors if (h := ic.check(actor, qualified, call.arguments)) is not None), None)
            if hit is not None:
                self._log("state", actor, "intercept", text=hit.message,
                          data={"rule": hit.rule, "category": hit.category, "tool": qualified, "action": hit.action, "evidence": hit.evidence})
                if hit.action == "dummy":
                    return ToolResult(tool_call_id=call.tool_call_id, function_name=qualified, ok=True, content="(no output)")
                return fail(qualified, hit.message, "intercepted")
        if not system and self.noise is not None and (nz := self.noise.before(qualified, call.tool_call_id)) is not None:
            self._log("state", actor, "noise", data={"ops": ["tool_transient_error"], "tool": qualified})
            return nz  # a retryable tool error, not executed (M12-infra D)
        if not system and self.spec.limits.tool_quota:
            hit = [pat for pat in self.spec.limits.tool_quota if fnmatch.fnmatchcase(qualified, pat)]
            spent = [pat for pat in hit if self.tool_counts.get(pat, 0) >= self.spec.limits.tool_quota[pat]]
            if spent:
                return fail(qualified, f"{qualified}: no calls left this episode (budget {self.spec.limits.tool_quota[spent[0]]})", "quota_exhausted")
            for pat in hit:
                self.tool_counts[pat] = self.tool_counts.get(pat, 0) + 1
        router = self.backend.router()
        t0 = time.monotonic()
        try:
            coro = (router.call(actor, app_id, tool, call.arguments, call.tool_call_id, system=system) if router is not None
                    else self.apps[app_id].call(tool, call.arguments, actor=actor, call_id=call.tool_call_id))
            # a tool's own deadline is at most tool_timeout_s (App.deadline); the World waits TOOL_GRACE_S longer, so it ends
            # only a tool hung past its own deadline (infrastructure), never a policy's slow command (M1 audit R-4)
            result = await asyncio.wait_for(coro, timeout=self.spec.limits.tool_timeout_s + TOOL_GRACE_S)
        except TimeoutError:
            self._infra("transport_timeout", f"{qualified} hung past its own deadline ({self.spec.limits.tool_timeout_s}s + {TOOL_GRACE_S:g}s)")
            return fail(qualified, f"{qualified} timed out", "timeout", "infra")
        except Exception as e:  # noqa: BLE001 - an App crashing is an environment bug: contain it, record it, mask the episode
            if (lost := container_lost_cause(e)) is not None:  # the host confirmed the container is gone: an outage, not a tool bug (M1 R-1)
                self._infra("crucible:env_container_lost", f"{qualified}: {lost}")
                return fail(qualified, f"{qualified}: the environment's container was lost", "transport", "infra")
            import traceback

            self._infra("crucible:tool_crashed", f"{qualified}: {type(e).__name__}: {e}", traceback.format_exc()[-4000:])
            return fail(qualified, f"internal error in {qualified}", "internal_error", "infra")
        if result.duration_s is None:
            result.duration_s = time.monotonic() - t0
        if not result.ok and result.fault is None:
            result.fault = "infra" if result.error in INFRA_ERRORS else ("tool" if result.error in ("tool_error",) else "policy")
        if result.fault == "infra":
            self._infra("transport_unreachable" if result.error in ("transport", "gateway_error") else "crucible:tool_crashed", f"{qualified}: {result.content[:300]}")
        if not system and self.noise is not None:
            result, ops = self.noise.after(qualified, result)
            if ops:
                self._log("state", actor, "noise", data={"ops": ops, "tool": qualified})
        return self._bound(result)

    def _infra(self, kind: str, reason: str, detail: str | None = None) -> None:
        """An infrastructure fault during the episode: run_episode masks the sample (DESIGN P16)."""
        self.failures.append(Failure(kind=kind, stage="agent", reason=reason[:500]))
        self._log("state", None, "infra_fault", text=reason[:2000], data={"kind": kind, **({"traceback": detail} if detail else {})})

    def _bound(self, r: ToolResult) -> ToolResult:
        """One head+tail cut with an explicit banner (errors live at the tail); the full text goes to the store."""
        cap = self.spec.limits.max_tool_output_chars
        if len(r.content) <= cap:
            return r
        try:
            raw = r.content.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError:  # a surrogate outside the escape range (\ud800): escaped, never a crash (M14 campaign B1)
            raw = r.content.encode("utf-8", "backslashreplace")
        key = self.store.put_bytes(raw)
        head, tail = r.content[: cap * 2 // 3], r.content[-(cap // 3):]
        cut = len(r.content) - len(head) - len(tail)
        banner = f"\n\n[... {cut} characters truncated; narrow the command or read the output in parts ...]\n\n"
        return r.model_copy(update={"content": head + banner + tail, "truncated": True, "content_ref": key})

    # ------------------------------------------------------------------ episode
    async def reset(self, *, start: WorldRef | None = None, instructions: dict[str, str] | None = None) -> dict[str, Observation]:
        if not self._started:
            raise WorldError("call start() first")
        if instructions is not None:
            self.instructions = dict(instructions)
        ref = start or self.initial
        assert ref is not None
        await self.restore(ref)
        self.episode_start = ref
        if start is None:
            # A fresh episode, not a mid-episode restore: wall clocks restart at t0 (a wall-clock
            # snapshot is necessarily taken a little after t0) and backends reset what they hold.
            if self.clock.mode == "wall":
                self.clock.reset()
            await self.backend.on_reset(self)
            await self._reset_hooks_and_health()
            if self.turns is not None:
                await self.turns.begin()
        self.log = []
        if self.trace_sink is not None:
            self.trace_sink.write_text("")
        self.failures = []
        self.finished = {}
        self.episode_id = new_id("ep")
        self._log("state", None, "reset", data={"world_ref": ref.fingerprint})
        if start is None:  # a fresh episode: recorded conversations and opening lines (responders' episode_start hooks)
            prefilled = 0
            policies = [p.id for p in self.spec.policy_actors]
            for aid, r in self.responders.items():
                for frm, to, text in (r.episode_start(policies) if hasattr(r, "episode_start") else []):
                    self.inboxes.setdefault(to, []).append(Inbound(**{"from": frm, "text": text, "at": self._now(), "prefill": True}))
                    self._log("message", frm, "message", text=text, data={"to": to, "prefill": True, "via": aid})
                    prefilled += 1
            if prefilled:
                self._log("state", None, "continuation_start", data={"prefilled_messages": prefilled})
        # Events due at t0 (e.g. an opening email) fire before the first observation.
        await self._fire_due()
        return {a.id: self._observe(a.id, [], first=True) for a in self.spec.policy_actors}

    async def begin_task_step(self, name: str, index: int, instruction: str | dict[str, str]) -> dict[str, Observation]:
        """Open the next step of a multi-step task in the running episode (M1 item 4; Harbor multi-step trials): policy
        actors leave ``done``, the step boundary goes into the trace, and each policy actor's next observation carries
        the step's instruction (and its tools again). State, clock, inboxes and the trace continue."""
        if not self._started:
            raise WorldError("call start() first")
        texts = instruction if isinstance(instruction, dict) else {a.id: instruction for a in self.spec.policy_actors}
        for a in self.spec.policy_actors:
            self.done.discard(a.id)
        self._cut.clear()  # a new task step is the policy's again (M11 lab audit R-2)
        self._cut_added.clear()
        self.instructions.update(texts)
        self._log("state", None, "task_step", data={"name": name, "index": index})
        return {a.id: self._observe(a.id, [], first=True) for a in self.spec.policy_actors}

    async def check_health(self, app_id: str, spec: Any = None) -> dict[str, Any]:
        """Run an App's health check (or ``spec``, e.g. a task step's); raises ``HealthcheckError`` when it fails."""
        from crucible.world.health import run_healthcheck

        app = self.apps[app_id]
        hc = spec or app.spec.healthcheck
        if hc is None:
            return {"healthy": True, "attempts": 0}
        rep = await run_healthcheck(getattr(app, "health_exec", None) or app.ctx.exec, hc, what=f"app {app_id!r}")  # an App's own container, if it has one
        self.health[app_id] = rep
        return rep

    async def _reset_hooks_and_health(self) -> None:
        """Fresh episodes only: reset hooks of live services no adapter restores, then health checks, in spec order."""
        from crucible.world.health import ResetHookError, run_probe

        for app in self.apps.values():
            if app.spec.reset_hook is not None:
                ok, detail = await run_probe(app.ctx.exec, app.spec.reset_hook)
                if not ok:
                    raise ResetHookError(f"app {app.id!r}: reset hook failed: {detail}")
            if app.spec.healthcheck is not None:
                await self.check_health(app.id)

    async def step(self, actor: str, action: Action | dict[str, Any]) -> StepResult:
        async with self._step_lock:  # concurrent callers (several harnesses on one World) are serialized
            return await self._step(actor, action)

    async def _step(self, actor: str, action: Action | dict[str, Any]) -> StepResult:
        if isinstance(action, dict):
            from pydantic import TypeAdapter

            action = TypeAdapter(Action).validate_python(action)
        if isinstance(action, Message | Done) and unencodable(d := action.model_dump(mode="python", by_alias=True)) is not None:
            action = type(action).model_validate(safe_json(d))  # M14 campaign B1: the policy's unencodable text is escaped, never a crash
        spec = self.spec.actor(actor)
        if spec.kind != "policy":
            raise WorldError(f"{actor!r} is not a policy actor")
        if actor in self.done:
            if actor in self._cut:  # M11 lab audit R-2: a policy call after the pre-grade cut, refused and recorded
                what = action.function_name if isinstance(action, ToolCall) else type(action).__name__.lower()
                self.late_call(actor, what)
                raise WorldError(f"{actor!r} was cut before grading: late call {what!r} refused")
            raise WorldError(f"{actor!r} already declared done")
        self.step_index += 1
        results: list[ToolResult] = []
        if self.turns is not None and isinstance(action, ToolCall | Message):
            refused = self.turns.refusal(actor, action)
            if refused is not None:  # not this seat's move, or not a legal one: the policy's error, never executed
                self._log("observation", actor, "tool_result", text=refused.content,
                          data={"function": refused.function_name, "ok": False, "error": refused.error, "id": refused.tool_call_id, "fault": "policy"})
                return StepResult(observation=self._observe(actor, [refused]), done=False, info={"step": self.step_index, "refused": refused.error})
            if self.turns.mode == "simultaneous" and actor in self.turns.seats:
                return await self._simultaneous(actor, action)
        if isinstance(action, ToolCall):
            self._log("action", actor, "tool_call", data={"function": action.function_name, "arguments": action.arguments, "id": action.tool_call_id})
            res = await self._call_tool(actor, action)
            results.append(res)
            data = {"function": res.function_name, "ok": res.ok, "error": res.error, "id": res.tool_call_id, "fault": res.fault, "truncated": res.truncated,
                    "content_ref": res.content_ref}
            if res.images:  # store refs only: pixels never enter the trace
                data["images"] = [{"mime": i.mime, "ref": i.ref} for i in res.images]
            self._log("observation", actor, "tool_result", text=res.content, data=data)
            if res.ok:  # an errored call provokes nothing (agentenv's action-trigger semantics)
                from crucible.core.triggers import result_data

                await self._run_fired(self.scheduler.on_tool(res.function_name, actor, action.arguments, result_data(res.content)))
            await self._eval_state_events()
            if res.ends_episode:  # the environment terminated the episode (imported OpenEnv / Gym environments)
                for a in res.ends_for or [actor]:
                    if a not in self.done:
                        self.done.add(a)
                        self._log("action", a, "done", text="environment_terminal")
        elif isinstance(action, Message):
            await self._send(actor, action)
        elif isinstance(action, Done):
            self.done.add(actor)
            self._log("action", actor, "done", text=action.reason, data={"answer": action.answer} if action.answer is not None else None)
        if self.turns is not None:
            await (self.turns.on_done(actor) if isinstance(action, Done) else self.turns.after_action(actor))
            self._end_if_terminal()
        self.clock.on_step()
        await self._fire_due()
        await self.backend.after_step(self)
        if self.backend.capabilities.native_acl:
            self._acl_changed |= self.backend.pop_acl_changes()  # type: ignore[attr-defined]
        obs = self._observe(actor, results)
        all_done = all(a.id in self.done for a in self.spec.policy_actors)
        return StepResult(observation=obs, done=all_done, info={"step": self.step_index, "time": obs.time, "finished": dict(self.finished)})

    def _end_if_terminal(self) -> None:
        if self.turns is not None and self.turns.terminal:
            for seat in self.turns.seats:
                if seat not in self.done:
                    self.done.add(seat)
                    self._log("action", seat, "done", text="game_over", data={"rewards": dict(self.turns.rewards)})

    async def _simultaneous(self, actor: str, action: ToolCall | Message) -> StepResult:
        """A simultaneous round: buffer this seat's move; when every seat to move has moved, resolve the round (the
        rules App's ``resolve_round`` or the calls in seat order) and hand each seat its results."""
        assert self.turns is not None
        self._log("action", actor, "move_submitted", data={"round": self.turns.round, "kind": action.kind})
        if not self.turns.submit(actor, action):
            return StepResult(observation=self._observe(actor, []), done=False, info={"step": self.step_index, "waiting": True})
        moves = self.turns.take_round()
        app = self.apps.get(self.spec.turns.app or "") if self.spec.turns else None
        by_seat: dict[str, list[ToolResult]] = {}
        if app is not None and hasattr(app, "resolve_round"):
            calls = {s: m for s, m in moves.items() if m.get("kind") == "tool_call"}
            for s, m in moves.items():
                if m.get("kind") == "tool_call":
                    self._log("action", s, "tool_call", data={"function": m["function_name"], "arguments": m.get("arguments", {}), "id": m["tool_call_id"]})
            out = await app.resolve_round(calls)
            for s, r in out.items():
                by_seat[s] = [r]
                self._log("observation", s, "tool_result", text=r.content, data={"function": r.function_name, "ok": r.ok, "error": r.error, "id": r.tool_call_id})
            for s, m in moves.items():
                if m.get("kind") == "message":
                    await self._send(s, Message.model_validate(m))
        else:
            for s, m in moves.items():
                if m.get("kind") == "tool_call":
                    call = ToolCall.model_validate(m)
                    self._log("action", s, "tool_call", data={"function": call.function_name, "arguments": call.arguments, "id": call.tool_call_id})
                    r = await self._call_tool(s, call)
                    self._log("observation", s, "tool_result", text=r.content, data={"function": r.function_name, "ok": r.ok, "error": r.error, "id": r.tool_call_id})
                    by_seat[s] = [r]
                else:
                    await self._send(s, Message.model_validate(m))
        self.turns.end_round()
        if self.turns.mode == "simultaneous" and self.spec.turns and self.spec.turns.app:
            await self.turns.pull()
        for s, rs in by_seat.items():
            if s != actor:
                self.turns.defer(s, rs)
        self._end_if_terminal()
        self.clock.on_step()
        await self._fire_due()
        await self.backend.after_step(self)
        obs = self._observe(actor, by_seat.get(actor, []))
        all_done = all(a.id in self.done for a in self.spec.policy_actors)
        return StepResult(observation=obs, done=all_done, info={"step": self.step_index, "round": self.turns.round})

    def _instruction(self, actor: str) -> str | None:
        text = self.instructions.get(actor)
        if self.noise is None:
            return text
        noisy, ops = self.noise.instruction(text)  # the stored instruction stays clean: the next reset perturbs afresh
        if ops:
            self._log("state", actor, "noise", data={"ops": ops, "instruction": True})
        return noisy

    def _observe(self, actor: str, results: list[ToolResult], *, first: bool = False) -> Observation:
        inbox, self.inboxes[actor] = self.inboxes.get(actor, []), []
        if self.spec.isolation.hide_citations:  # citations (record ids name repos / PRs) stay in the trace only
            inbox = [m.model_copy(update={"citations": []}) if m.citations else m for m in inbox]
        inbox = [m.model_copy(update={"attachments": self.hydrate(m.attachments)}) if m.attachments else m for m in inbox]
        media = self.hydrate(self.instruction_media.get(actor)) if first else None
        send_tools = first or actor in self._acl_changed
        self._acl_changed.discard(actor)
        return Observation(
            actor=actor,
            step=self.step_index,
            time=self.clock.now_iso() if self.spec.clock else None,
            tool_results=results,
            inbox=inbox,
            instruction=self._instruction(actor) if first else None,
            instruction_media=media or None,
            tools=self.tools_for(actor) if send_tools else [],
            to_move=list(self.turns.to_move) if self.turns is not None else None,
            legal=self.turns.legal.get(actor) if self.turns is not None else None,
        )

    async def _send(self, frm: str, msg: Message) -> None:
        try:
            self.spec.actor(msg.to)
        except KeyError:
            self._log("message", frm, "message_undeliverable", text=msg.text, data={"to": msg.to})
            self.inboxes.setdefault(frm, []).append(Inbound(**{"from": "world", "text": f"No such recipient: {msg.to}", "at": self._now()}))
            return
        att = [a.model_copy(update={"data_b64": None}) for a in msg.attachments] if msg.attachments else None  # refs only from here on
        self._log("message", frm, "message", text=msg.text,
                  data={"to": msg.to, "citations": msg.citations, **({"attachments": [a.model_dump(mode="json") for a in att]} if att else {})})
        self.inboxes.setdefault(msg.to, []).append(Inbound(**{"from": frm, "text": msg.text, "at": self._now(), "citations": msg.citations,
                                                               **({"attachments": att} if att else {})}))
        if msg.to in self.responders:
            await self._run_responder(msg.to)

    async def _run_responder(self, actor_id: str) -> None:
        responder = self.responders[actor_id]
        inbox, self.inboxes[actor_id] = self.inboxes.get(actor_id, []), []
        results: list[ToolResult] = []
        for _ in range(MAX_INNER_TURNS):
            turn = ActorTurn(actor=actor_id, time=self._now(), inbox=inbox, tool_results=results, tools=self.tools_for(actor_id))
            try:
                actions = await responder.respond(turn)
            except Exception as e:  # noqa: BLE001 - a broken simulator masks the sample; it crashed World.step and run_episode (M2 audit)
                import traceback

                self._infra("crucible:sim_failure", f"{actor_id} raised {type(e).__name__}: {e}", traceback.format_exc()[-4000:])
                return
            inbox, results = [], []
            calls = [a for a in actions if isinstance(a, ToolCall)]
            for a in actions:
                if isinstance(a, ToolCall):
                    self._log("action", actor_id, "tool_call", data={"function": a.function_name, "arguments": a.arguments, "id": a.tool_call_id})
                    r = await self._call_tool(actor_id, a)
                    self._log("observation", actor_id, "tool_result", text=r.content, data={"function": r.function_name, "ok": r.ok, "error": r.error, "fault": r.fault})
                    results.append(r)
                elif isinstance(a, Message):
                    latency = getattr(responder, "reply_latency", None)
                    if latency and self.clock.mode == "step":
                        self.clock.advance(parse_duration(latency) if isinstance(latency, str) else timedelta(seconds=latency))
                    text = a.text
                    if self.spec.isolation.scrub_messages and a.to in {p.id for p in self.spec.policy_actors}:
                        from crucible.world.leaks import scrub_text

                        text = scrub_text(a.text) or "(no comment)"
                    self._log("message", actor_id, "message", text=text, data={"to": a.to, "citations": a.citations, **({"unscrubbed": a.text} if text != a.text else {})})
                    # Replies are not re-dispatched to other responders here (no unbounded chains).
                    self.inboxes.setdefault(a.to, []).append(Inbound(**{"from": actor_id, "text": text, "at": self._now(), "citations": a.citations}))
                elif isinstance(a, Done):
                    self.finished[actor_id] = a.reason or "done"
                    if hasattr(responder, "finished_reason"):  # kept in the responder's saved state: a restore reads it (M2 R-3)
                        responder.finished_reason = self.finished[actor_id]
                    self._log("action", actor_id, "done", text=a.reason)
            if not calls:
                return
        self._log("action", actor_id, "responder_turn_limit", data={"max": MAX_INNER_TURNS})
        self.failures.append(Failure(kind="crucible:sim_failure", stage="agent", reason=f"{actor_id} exceeded {MAX_INNER_TURNS} tool turns"))

    async def _fire_due(self) -> None:
        for _ in range(1000):  # bounded catch-up
            fired = self.scheduler.pop_due()
            if not fired:
                return
            await self._run_fired(fired)

    async def _run_check(self, check: dict, ctx: dict | None, actor: str) -> tuple[bool, dict]:
        """A trigger check pipeline (agentenv semantics): each step calls a tool as the system, extracts a value, binds
        or tests it; the last step's predicate decides."""
        from crucible.core.triggers import apply_extract, eval_predicate, result_data, template

        bindings: dict[str, Any] = {}
        for k, step in enumerate(check["steps"]):
            args = template(step.get("args", {}), bindings, ctx)
            res = await self._call_tool(actor, ToolCall(tool_call_id=f"check-{self.step_index}-{k}-{time.monotonic_ns()}", function_name=step["tool"],
                                                        arguments=args), system=True)
            if not res.ok:
                return False, bindings
            data = result_data(res.content)
            value = apply_extract(step.get("extract"), data if not isinstance(data, str) else None, res.content or "")
            if step.get("bind") is not None:
                if value in (None, "", []):
                    return False, bindings
                bindings[step["bind"]] = value
            if "predicate" in step and not eval_predicate(step["predicate"], value):
                return False, bindings
        return True, bindings

    async def _eval_state_events(self) -> None:
        for i, ev in self.scheduler.state_events():
            ok, bindings = await self._run_check(ev.when.state or {}, None, ev.actor)
            if ok:
                self.scheduler.mark_fired(ev.id)
                await self._run_fired([Fired(ev, i, self.clock.now(), bindings=bindings)])

    async def _run_fired(self, fired: list[Fired]) -> None:
        from crucible.core.triggers import TemplateError, template

        for f in fired:
            ev: EventSpec = f.event
            act = ev.action
            self._log("action", ev.actor, f"event:{act.type}", text=act.text, data={"event": ev.id, "due": f.at.isoformat()})
            ok = True
            if act.type == "nl":
                targets = [act.to] if act.to else [a.id for a in self.spec.policy_actors]
                for t in targets:
                    self.inboxes.setdefault(t, []).append(Inbound(**{"from": ev.actor, "text": act.text or "", "at": self._now()}))
                    if t in self.responders:
                        await self._run_responder(t)
            elif act.type == "tool":
                try:
                    args = template(act.args, f.bindings, f.ctx)
                except TemplateError as e:
                    self._log("state", ev.actor, "event_failed", text=str(e), data={"event": ev.id})
                    continue
                res = await self._call_tool(ev.actor, ToolCall(tool_call_id=f"event-{ev.id}-{self.step_index}", function_name=act.tool or "", arguments=args), system=True)
                self._log("observation", ev.actor, "tool_result", text=res.content[:4000], data={"function": res.function_name, "ok": res.ok, "event": ev.id})
                ok = res.ok
            elif act.type == "permission":
                target = act.to or ev.actor
                if act.grant:
                    self.acl.grant(target, act.grant)
                if act.revoke:
                    self.acl.revoke(target, act.revoke)
                self._acl_changed.add(target)
            if ok and act.verify is not None:
                vok, _ = await self._run_check(act.verify, f.ctx, ev.actor)
                self._log("state", ev.actor, "event_verify", data={"event": ev.id, "ok": vok})
                ok = vok
            if ok:
                self.scheduler.anchor(ev.id)  # events anchored on this one get their mark (after_event + offset)

    def firing_log(self) -> list[dict[str, Any]]:
        """Every event firing in order: id, the virtual instant it was due at, its action type (the conformance view)."""
        return [{"event": e.data["event"], "at": e.data.get("due"), "type": e.type.split(":", 1)[1]} for e in self.log
                if e.kind == "action" and e.type.startswith("event:")]

    def _now(self) -> str | None:
        return self.clock.now_iso() if self.spec.clock else None

    def _log(self, kind: str, actor: str | None, typ: str, *, text: str | None = None, data: dict[str, Any] | None = None) -> None:
        # every string the trace holds is UTF-8 writable (M14 campaign B1): a policy's lone surrogate is kept as escaped text
        ev = TraceEvent(seq=len(self.log), time=self._now(), kind=kind, actor=actor, type=typ, text=safe_text(text) if isinstance(text, str) else text,
                        data=safe_json(data or {}))
        self.log.append(ev)
        if self.trace_sink is not None:
            with self.trace_sink.open("a") as f:
                f.write(_event_json(ev) + "\n")

    def stream_trace(self, path: Path, *, prefix: list[TraceEvent] | None = None) -> None:
        """Stream trace events to ``path`` (JSONL). ``prefix``: the events of an episode resumed from a checkpoint, which
        become this World's log (and the head of the stream) so the trace and every event-reading check stay whole."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if prefix is not None:
            self.log = list(prefix)
        path.write_text("".join(_event_json(e) + "\n" for e in self.log))
        self.trace_sink = path

    @staticmethod
    def read_trace(path: Path, upto: int | None = None) -> list[TraceEvent]:
        evs = [TraceEvent.model_validate_json(x) for x in Path(path).read_text().splitlines() if x.strip()]
        return evs[:upto] if upto is not None else evs

    # ------------------------------------------------------------------ state
    async def cut_policy(self, actors: Iterable[str] | None = None) -> list[str]:
        """The pre-grade cut (M11 lab audit R-2): ``actors`` (default: every policy actor) are done from here, so ``step``
        refuses them, and each refused call is recorded in ``late_calls`` (a ``late_call`` trace event, never replayed).
        Taken under the step lock, so a step already in flight ends before the grade reads state. ``lift_cut`` reopens the
        World for another episode. An outside agent (an A2A harness, an agentenv run) has no turn loop that stops at the
        episode's end: without the cut it kept calling tools during its grade (grading takes no step lock)."""
        async with self._step_lock:
            ids = list(actors) if actors is not None else [a.id for a in self.spec.policy_actors]
            self._cut_added.update(a for a in ids if a not in self.done)
            self.done.update(ids)
            self._cut.update(ids)
            self._log("state", None, "policy_cut", data={"actors": ids})
        return ids

    def lift_cut(self) -> None:
        """Undo ``cut_policy`` for a World that serves another episode: an actor that declared done itself stays done."""
        self.done.difference_update(self._cut_added)
        self._cut.clear()
        self._cut_added.clear()

    @property
    def cut(self) -> frozenset[str]:
        """The actors ``cut_policy`` cut (empty when no cut is in force)."""
        return frozenset(self._cut)

    def late_call(self, actor: str, what: str) -> None:
        """Record a policy call refused after the cut (``late_calls`` and a ``late_call`` trace event)."""
        late = {"actor": actor, "action": what, "at": self._now()}
        self.late_calls.append(late)
        self._log("state", actor, "late_call", data=late)

    async def quiesce(self) -> dict | None:
        """Verification boundary: stop every process the policy left behind (no-op without isolation)."""
        rep = await self.backend.quiesce()
        if rep is not None:
            self._log("state", None, "quiesce", data=rep)
            if not rep.get("ok", True):
                raise WorldError(f"policy processes survived quiesce: {rep}")
        return rep

    async def flush(self) -> None:
        """Every App's in-memory state to its adapter's storage (``App.before_snapshot``)."""
        for app in self.apps.values():
            await app.before_snapshot()

    async def snapshot(self, *, quiesce: bool = False) -> WorldRef:
        if quiesce:
            await self.quiesce()
        await self.flush()
        names = list(self.apps)
        refs: list[StateRef] = list(await asyncio.gather(*(self.apps[n].state.snapshot() for n in names)))
        ref = WorldRef(
            world=self.spec_hash,
            step=self.step_index,
            apps=dict(zip(names, refs, strict=True)),
            clock=self.clock.state(),
            pending_events=self.scheduler.state(),
            actors={a: r.state_dict() for a, r in sorted(self.responders.items())},
            fired_counts=dict(sorted(self.scheduler.fired_counts.items())),
            acl_rules=self.acl.state(),
            inboxes={a: [_inbound_json(m) for m in msgs] for a, msgs in sorted(self.inboxes.items()) if msgs},
            done=sorted(self.done),
            tool_counts=dict(sorted(self.tool_counts.items())),
            backend=await self.backend.snapshot_state(),
            turns=self.turns.state() if self.turns is not None else None,
        )
        ref.fingerprint = self._fingerprint(ref)
        return ref

    def _fingerprint(self, ref: WorldRef) -> str:
        body = ref.model_dump(mode="json", exclude={"fingerprint", "step"})
        body["apps"] = {k: v["fingerprint"] for k, v in body["apps"].items()}
        if not body.get("tool_counts"):
            body.pop("tool_counts", None)  # no quotas: fingerprints stay identical to refs captured before tool_count existed
        if body.get("turns") is None:
            body.pop("turns", None)  # not turn-based: fingerprints of earlier refs are unchanged
        if ref.clock.mode == "wall":
            body["clock"] = {k: v for k, v in body["clock"].items() if k != "now"}  # wall time always moves
        return sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))

    async def restore(self, ref: WorldRef) -> None:
        try:
            # by the scheme the ref's id was made with (R-21), not today's, over the spec as this World was built from it
            same = ref.world == self.spec_hash or matches_content_id(self._spec_doc, ref.world)
        except ValueError as e:
            raise WorldError(f"WorldRef names its world by {ref.world[:20]!r}: {e}") from e
        if not same:
            raise WorldError(f"WorldRef belongs to world {ref.world[:12]}, not {self.spec_hash[:12]}")
        missing = set(self.apps) ^ set(ref.apps)
        if missing:
            raise WorldError(f"WorldRef apps mismatch: {sorted(missing)}")
        unchanged = await self._unchanged_apps(ref)
        await self.backend.before_restore(self)
        self.restored_apps = [n for n in ref.apps if n not in unchanged]
        await asyncio.gather(*(self.apps[n].state.restore(ref.apps[n]) for n in self.restored_apps))
        await self.backend.after_restore(self)
        for app in self.apps.values():
            await app.after_restore()
        self.clock.restore(ref.clock)
        self.scheduler.restore(ref.pending_events, ref.fired_counts)
        self.acl.restore(ref.acl_rules)
        self.inboxes = {a.id: [] for a in self.spec.actors}
        for a, msgs in ref.inboxes.items():
            self.inboxes[a] = [Inbound.model_validate(m) for m in msgs]
        self.done = set(ref.done)
        self._cut.clear()  # a restored state is another episode's: the cut and its late calls were this one's
        self._cut_added.clear()
        self.tool_counts = dict(ref.tool_counts)
        for a, st in ref.actors.items():
            if a in self.responders:
                self.responders[a].load_state_dict(st)
        # M2 R-3: who had ended, and the faults seen, are the restored branch's. Both were kept from before the restore, so
        # in M6 branch_value a user's DONE in one continuation cut the next ones short and one responder fault masked all
        # of them, and a resumed checkpoint forgot that the user had ended. ``finished`` follows the responders' saved
        # state; ``failures`` starts empty (never carried in a ref: a retry resumes from its masked attempt's checkpoint).
        self.finished = {a: str(st["finished_reason"]) for a, st in ref.actors.items() if a in self.responders and st.get("finished_reason")}
        self.failures = []
        await self.backend.restore_state(ref.backend)
        if self.turns is not None:
            self.turns.restore(ref.turns)
        self.step_index = ref.step
        self._acl_changed = set()
        self._settled = (ref.fingerprint, self._touched)

    async def _unchanged_apps(self, ref: WorldRef) -> set[str]:
        """The Apps a restore of ``ref`` may skip (M1 audit R-5): the World's state was last set to ``ref`` (by start or a
        restore), no tool call or verifier query ran since, and the App's live fingerprint still equals ``ref``'s. Before
        step 1 a World is restored up to three times (start, reset, the harness's reset), tens of seconds each for a large
        seed. A fingerprint cannot see every bit of state (file times, owners, what an App keeps outside its adapter), so it
        is trusted only when nothing ran; a command run underneath (a setup hook) still shows in it."""
        if self._settled != (ref.fingerprint, self._touched):
            return set()
        await self.flush()
        names = list(ref.apps)
        live = await asyncio.gather(*(self.apps[n].state.fingerprint() for n in names))
        return {n for n, fp in zip(names, live, strict=True) if fp == ref.apps[n].fingerprint}

    def verifier_view(self) -> VerifierView:
        return VerifierView(self)

    def put_media(self, kind: str, mime: str, data: bytes) -> MediaRef:
        """Store an image or audio clip (content-addressed) and return its reference (M1 item 8)."""
        return MediaRef(kind=kind, mime=mime, ref=self.store.put_bytes(data))  # type: ignore[arg-type]

    def hydrate(self, media: list[MediaRef] | None) -> list[MediaRef] | None:
        """The live copy of stored media: bytes attached for the policy (never serialized: ``data_b64`` is excluded)."""
        import base64

        if not media:
            return media
        return [m.model_copy(update={"data_b64": base64.b64encode(self.store.get_bytes(m.ref)).decode()}) if m.ref and m.data_b64 is None else m
                for m in media]

    @property
    def closed(self) -> bool:
        return self._closed

    # ------------------------------------------------------------------ export
    def to_trace(self) -> Trace:
        actors = [TraceActor(id=a.id, role=a.kind, kind="agent" if a.kind in ("policy", "peer") else ("human" if a.kind == "human" else "system")) for a in self.spec.actors]
        actor_ids = {a.id for a in actors}
        events = [e.model_copy(deep=True, update={} if e.actor is None or e.actor in actor_ids else {"actor": None}) for e in self.log]
        try:
            read_only = {t.qualified: t.read_only for t in self.all_tools()}
        except Exception:  # noqa: BLE001 - a backend that cannot list tools after close: fall back to the verb rule
            read_only = {}
        failed = {e.data.get("id") for e in events if e.type == "tool_result" and e.data.get("ok") is False and e.data.get("id") is not None}  # a refused or failed call wrote nothing
        for e in events:  # what each call wrote: M4 grades a later signal about that file or record to this step, whichever agent made it
            fn = str(e.data.get("function", ""))
            if e.type == "tool_call" and e.data.get("id") not in failed and (w := writes_of(fn, e.data.get("arguments")) or app_writes(fn, e.data.get("arguments"), read_only=read_only.get(fn))):
                e.data["writes"] = w
                e.target = SignalTarget(app=str(e.data["function"]).split(".")[0], object=w[0])
        probes = annotate_probes(events)  # answer-key attempts are recorded, never scored (P12); an exposure masks the episode
        return Trace(
            trace_id=self.episode_id,
            family=self.spec.world,
            source=TraceSource(connector="crucible.world", ref=f"world:{self.spec_hash[:16]}", privacy="internal"),
            actors=actors,
            events=events,
            outcome=TraceOutcome(details={**({"probes": probes} if probes["attempted"] else {}), **({"intercepts": self.intercepts} if self.intercepts else {})}),
        )

    @property
    def intercepts(self) -> list[dict[str, Any]]:
        """This episode's intercepted calls (``intercept`` events in the log: they follow resets and snapshots)."""
        return [{"seq": e.seq, "actor": e.actor, **e.data} for e in self.log if e.type == "intercept"]

    def actor_spec(self, actor: str) -> ActorSpec:
        return self.spec.actor(actor)
