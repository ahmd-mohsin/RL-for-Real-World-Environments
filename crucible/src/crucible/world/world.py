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
import json
import shutil
import tempfile
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from crucible.core.ids import new_id, sha256_text
from crucible.core.registry import get_registry
from crucible.core.schema.actions import Action, Done, Inbound, Message, Observation, StepResult, ToolCall, ToolResult, ToolSpec
from crucible.core.schema.state import Query, QueryResult, StateRef, WorldRef
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource
from crucible.core.schema.world import ActorSpec, EventSpec, WorldSpec
from crucible.core.store import Store, get_store
from crucible.world.apps.base import App, AppContext
from crucible.world.backends.base import Backend
from crucible.world.clock import AccessControl, EventScheduler, Fired, VirtualClock, parse_duration

MAX_INNER_TURNS = 8  # tool calls a responder may make while handling one inbound batch


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
        return self._world.apps[app].state

    async def query(self, app: str, q: Query | dict[str, Any]) -> QueryResult:
        return await self.adapter(app).query(q if isinstance(q, Query) else Query.model_validate(q))

    async def fingerprints(self) -> dict[str, str]:
        apps = self._world.apps
        fps = await asyncio.gather(*(a.state.fingerprint() for a in apps.values()))
        return dict(zip(apps, fps, strict=True))


class World:
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
    ):
        self.spec = spec
        self.spec_hash = spec.content_hash()
        self.instructions = dict(instructions or {})
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
        self.apps: dict[str, App] = {}
        self.responders: dict[str, Responder] = {}
        self.inboxes: dict[str, list[Inbound]] = {a.id: [] for a in spec.actors}
        self.done: set[str] = set()
        self.step_index = 0
        self.log: list[TraceEvent] = []
        self.episode_id = new_id("ep")
        self.initial: WorldRef | None = None
        self._tools: dict[str, list[ToolSpec]] = {}
        self._acl_changed: set[str] = set()
        self._started = False

    # ------------------------------------------------------------------ lifecycle
    async def start(self) -> WorldRef:
        problems = self.backend.validate()
        if problems:
            raise WorldError("; ".join(problems))
        try:
            return await self._start()
        except BaseException:
            # A failed start must not leak containers, gateways or servers (found: a crashed App.provision
            # left a Harbor sandbox running).
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
                extra={"peers": self.apps},  # live dict: an App can reach peers started before it (e.g. a browser -> a website's URL)
            )
            self.apps[app_spec.id] = reg.get("app_template", app_spec.template)(app_spec, ctx)
        for app in self.apps.values():
            await app.provision()
        for app in self.apps.values():  # spec order: dependencies first (a browser after the sites it opens)
            await app.start()
        for app_id, app in self.apps.items():
            self._tools[app_id] = await app.list_tools()
        await self.backend.after_apps_started(self)
        delegated = self.backend.delegated_events()
        if delegated:
            self.scheduler = EventScheduler([e for e in self.spec.events if e.id not in delegated], self.clock)
        self._started = True
        self.initial = await self.snapshot()
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
                remote = f"/tmp/.crucible_bundle_{abs(hash(dest))}.tar"
                await exec_.push_file(tar, remote)
                out = await exec_.run(f"mkdir -p '{dest}' && tar -xf {remote} -C '{dest}' && rm -f {remote}")
                if not out.ok:
                    raise WorldError(f"bundle {src} -> {dest} failed: {out.stderr[-400:]}")

    async def close(self) -> None:
        for app in self.apps.values():
            try:
                await app.stop()
            except Exception:  # noqa: BLE001 - best effort on teardown
                pass
        await self.backend.teardown()
        if self._own_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)

    async def __aenter__(self) -> World:
        await self.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()

    def attach(self, responder: Responder) -> None:
        actor = self.spec.actor(responder.actor_id)
        if actor.kind == "policy":
            raise WorldError(f"{actor.id!r} is a policy actor; policies act through step(), not attach()")
        self.responders[actor.id] = responder

    # ------------------------------------------------------------------ tools
    def all_tools(self) -> list[ToolSpec]:
        return [t for app in self.spec.apps for t in self._tools.get(app.id, [])]

    def tools_for(self, actor: str) -> list[ToolSpec]:
        if self.backend.capabilities.native_acl:
            return self.backend.allowed_tools(actor)  # type: ignore[attr-defined]
        return [t for t in self.all_tools() if self.acl.allows(actor, t.qualified)]

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
        try:
            app_id, tool = self._split(call.function_name)
        except WorldError as e:
            return ToolResult(tool_call_id=call.tool_call_id, function_name=call.function_name, ok=False, content=str(e), error="bad_tool_name")
        qualified = f"{app_id}.{tool}"
        if app_id not in self.apps:
            return ToolResult(tool_call_id=call.tool_call_id, function_name=qualified, ok=False, content=f"Unknown app: {app_id}", error="unknown_app")
        if not system and not self.backend.capabilities.native_acl and not self.acl.allows(actor, qualified):
            return ToolResult(tool_call_id=call.tool_call_id, function_name=qualified, ok=False, content=f"Permission denied: {qualified}", error="permission_denied")
        router = self.backend.router()
        if router is not None:
            result = await router.call(actor, app_id, tool, call.arguments, call.tool_call_id, system=system)
        else:
            result = await self.apps[app_id].call(tool, call.arguments, actor=actor, call_id=call.tool_call_id)
        return result

    # ------------------------------------------------------------------ episode
    async def reset(self, *, start: WorldRef | None = None, instructions: dict[str, str] | None = None) -> dict[str, Observation]:
        if not self._started:
            raise WorldError("call start() first")
        if instructions is not None:
            self.instructions = dict(instructions)
        ref = start or self.initial
        assert ref is not None
        await self.restore(ref)
        if start is None:
            # A fresh episode, not a mid-episode restore: wall clocks restart at t0 (a wall-clock
            # snapshot is necessarily taken a little after t0) and backends reset what they hold.
            if self.clock.mode == "wall":
                self.clock.reset()
            await self.backend.on_reset(self)
        self.log = []
        self.episode_id = new_id("ep")
        self._log("state", None, "reset", data={"world_ref": ref.fingerprint})
        # Events due at t0 (e.g. an opening email) fire before the first observation.
        await self._fire_due()
        return {a.id: self._observe(a.id, [], first=True) for a in self.spec.policy_actors}

    async def step(self, actor: str, action: Action | dict[str, Any]) -> StepResult:
        if isinstance(action, dict):
            from pydantic import TypeAdapter

            action = TypeAdapter(Action).validate_python(action)
        spec = self.spec.actor(actor)
        if spec.kind != "policy":
            raise WorldError(f"{actor!r} is not a policy actor")
        if actor in self.done:
            raise WorldError(f"{actor!r} already declared done")
        self.step_index += 1
        results: list[ToolResult] = []
        if isinstance(action, ToolCall):
            self._log("action", actor, "tool_call", data={"function": action.function_name, "arguments": action.arguments, "id": action.tool_call_id})
            res = await self._call_tool(actor, action)
            results.append(res)
            self._log("observation", actor, "tool_result", text=res.content[:4000], data={"function": res.function_name, "ok": res.ok, "error": res.error, "id": res.tool_call_id})
            await self._run_fired(self.scheduler.on_tool(res.function_name, actor))
        elif isinstance(action, Message):
            await self._send(actor, action)
        elif isinstance(action, Done):
            self.done.add(actor)
            self._log("action", actor, "done", text=action.reason)
        self.clock.on_step()
        await self._fire_due()
        await self.backend.after_step(self)
        if self.backend.capabilities.native_acl:
            self._acl_changed |= self.backend.pop_acl_changes()  # type: ignore[attr-defined]
        obs = self._observe(actor, results)
        all_done = all(a.id in self.done for a in self.spec.policy_actors)
        return StepResult(observation=obs, done=all_done, info={"step": self.step_index, "time": obs.time})

    def _observe(self, actor: str, results: list[ToolResult], *, first: bool = False) -> Observation:
        inbox, self.inboxes[actor] = self.inboxes.get(actor, []), []
        if self.spec.isolation.hide_citations:  # citations (record ids name repos / PRs) stay in the trace only
            inbox = [m.model_copy(update={"citations": []}) if m.citations else m for m in inbox]
        send_tools = first or actor in self._acl_changed
        self._acl_changed.discard(actor)
        return Observation(
            actor=actor,
            step=self.step_index,
            time=self.clock.now_iso() if self.spec.clock else None,
            tool_results=results,
            inbox=inbox,
            instruction=self.instructions.get(actor) if first else None,
            tools=self.tools_for(actor) if send_tools else [],
        )

    async def _send(self, frm: str, msg: Message) -> None:
        try:
            self.spec.actor(msg.to)
        except KeyError:
            self._log("message", frm, "message_undeliverable", text=msg.text, data={"to": msg.to})
            self.inboxes.setdefault(frm, []).append(Inbound(**{"from": "world", "text": f"No such recipient: {msg.to}", "at": self._now()}))
            return
        self._log("message", frm, "message", text=msg.text, data={"to": msg.to, "citations": msg.citations})
        self.inboxes.setdefault(msg.to, []).append(Inbound(**{"from": frm, "text": msg.text, "at": self._now(), "citations": msg.citations}))
        if msg.to in self.responders:
            await self._run_responder(msg.to)

    async def _run_responder(self, actor_id: str) -> None:
        responder = self.responders[actor_id]
        inbox, self.inboxes[actor_id] = self.inboxes.get(actor_id, []), []
        results: list[ToolResult] = []
        for _ in range(MAX_INNER_TURNS):
            turn = ActorTurn(actor=actor_id, time=self._now(), inbox=inbox, tool_results=results, tools=self.tools_for(actor_id))
            actions = await responder.respond(turn)
            inbox, results = [], []
            calls = [a for a in actions if isinstance(a, ToolCall)]
            for a in actions:
                if isinstance(a, ToolCall):
                    self._log("action", actor_id, "tool_call", data={"function": a.function_name, "arguments": a.arguments, "id": a.tool_call_id})
                    r = await self._call_tool(actor_id, a)
                    self._log("observation", actor_id, "tool_result", text=r.content[:4000], data={"function": r.function_name, "ok": r.ok, "error": r.error})
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
                    self._log("action", actor_id, "done", text=a.reason)
            if not calls:
                return
        self._log("action", actor_id, "responder_turn_limit", data={"max": MAX_INNER_TURNS})

    async def _fire_due(self) -> None:
        for _ in range(1000):  # bounded catch-up
            fired = self.scheduler.pop_due()
            if not fired:
                return
            await self._run_fired(fired)

    async def _run_fired(self, fired: list[Fired]) -> None:
        for f in fired:
            ev: EventSpec = f.event
            act = ev.action
            self._log("action", ev.actor, f"event:{act.type}", text=act.text, data={"event": ev.id, "due": f.at.isoformat()})
            if act.type == "nl":
                targets = [act.to] if act.to else [a.id for a in self.spec.policy_actors]
                for t in targets:
                    self.inboxes.setdefault(t, []).append(Inbound(**{"from": ev.actor, "text": act.text or "", "at": self._now()}))
                    if t in self.responders:
                        await self._run_responder(t)
            elif act.type == "tool":
                res = await self._call_tool(ev.actor, ToolCall(tool_call_id=f"event-{ev.id}-{self.step_index}", function_name=act.tool or "", arguments=act.args), system=True)
                self._log("observation", ev.actor, "tool_result", text=res.content[:4000], data={"function": res.function_name, "ok": res.ok, "event": ev.id})
            elif act.type == "permission":
                target = act.to or ev.actor
                if act.grant:
                    self.acl.grant(target, act.grant)
                if act.revoke:
                    self.acl.revoke(target, act.revoke)
                self._acl_changed.add(target)

    def _now(self) -> str | None:
        return self.clock.now_iso() if self.spec.clock else None

    def _log(self, kind: str, actor: str | None, typ: str, *, text: str | None = None, data: dict[str, Any] | None = None) -> None:
        self.log.append(TraceEvent(seq=len(self.log), time=self._now(), kind=kind, actor=actor, type=typ, text=text, data=data or {}))

    # ------------------------------------------------------------------ state
    async def snapshot(self) -> WorldRef:
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
            inboxes={a: [m.model_dump(mode="json", by_alias=True) for m in msgs] for a, msgs in sorted(self.inboxes.items()) if msgs},
            done=sorted(self.done),
            backend=await self.backend.snapshot_state(),
        )
        ref.fingerprint = self._fingerprint(ref)
        return ref

    def _fingerprint(self, ref: WorldRef) -> str:
        body = ref.model_dump(mode="json", exclude={"fingerprint", "step"})
        body["apps"] = {k: v["fingerprint"] for k, v in body["apps"].items()}
        if ref.clock.mode == "wall":
            body["clock"] = {k: v for k, v in body["clock"].items() if k != "now"}  # wall time always moves
        return sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))

    async def restore(self, ref: WorldRef) -> None:
        if ref.world != self.spec_hash:
            raise WorldError(f"WorldRef belongs to world {ref.world[:12]}, not {self.spec_hash[:12]}")
        missing = set(self.apps) ^ set(ref.apps)
        if missing:
            raise WorldError(f"WorldRef apps mismatch: {sorted(missing)}")
        await asyncio.gather(*(self.apps[n].state.restore(r) for n, r in ref.apps.items()))
        self.clock.restore(ref.clock)
        self.scheduler.restore(ref.pending_events, ref.fired_counts)
        self.acl.restore(ref.acl_rules)
        self.inboxes = {a.id: [] for a in self.spec.actors}
        for a, msgs in ref.inboxes.items():
            self.inboxes[a] = [Inbound.model_validate(m) for m in msgs]
        self.done = set(ref.done)
        for a, st in ref.actors.items():
            if a in self.responders:
                self.responders[a].load_state_dict(st)
        await self.backend.restore_state(ref.backend)
        self.step_index = ref.step
        self._acl_changed = set()

    def verifier_view(self) -> VerifierView:
        return VerifierView(self)

    # ------------------------------------------------------------------ export
    def to_trace(self) -> Trace:
        actors = [TraceActor(id=a.id, role=a.kind, kind="agent" if a.kind in ("policy", "peer") else ("human" if a.kind == "human" else "system")) for a in self.spec.actors]
        actor_ids = {a.id for a in actors}
        events = [e if e.actor is None or e.actor in actor_ids else e.model_copy(update={"actor": None}) for e in self.log]
        return Trace(
            trace_id=self.episode_id,
            family=self.spec.world,
            source=TraceSource(connector="crucible.world", ref=f"world:{self.spec_hash[:16]}", privacy="internal"),
            actors=actors,
            events=events,
        )

    def actor_spec(self, actor: str) -> ActorSpec:
        return self.spec.actor(actor)
