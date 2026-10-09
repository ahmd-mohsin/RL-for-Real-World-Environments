"""agentenv backend: Crucible Apps behind a real agentenv gateway (DEPEND: ``agent_env.env.gateway``).

Topology (all local processes / tasks, no Docker needed):

    policy / harness --MCP or REST /step, header AgentEnv-Role: <actor>-->  agentenv Gateway (subprocess)
                                                                              |-- internal MCP server per App
                                                                              |   (crucible AppMcpServer, tools "<app>__<tool>")
                                                                              |-- clock (urn:agentenv:clock/v1)
                                                                              `-- triggers (urn:agentenv:triggers/v1) --nl--> RelayAgent (A2A) -> World inbox

What agentenv provides here, unchanged: tool aggregation, **per-role tool access** (one role per
actor), the **virtual clock** and its ``get_time`` tool, the **trigger engine** (time / action triggers
with tool, permission and nl actions), and the gateway trajectory log. Crucible keeps state
(State Adapters), snapshots, inboxes and simulated actors.

Params: ``gateway_mode`` (``consistent`` default: serialized tool calls, deterministic),
``delegate_events`` (default: true when the world clock is ``wall``), ``default_actor`` (actor id used
for MCP clients that do not identify themselves; default: first policy actor).
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
from typing import TYPE_CHECKING, Any

import httpx

from crucible.core.schema.actions import Inbound, ToolResult, ToolSpec
from crucible.core.schema.world import AppSpec, EventSpec, SandboxSpec
from crucible.world.backends.base import Backend, BackendCapabilities, ToolRouter
from crucible.world.clock import parse_duration
from crucible.world.exec import Exec, LocalExec, spawn_logged
from crucible.world.packaging.app_server import AppMcpServer
from crucible.world.packaging.sidecar import free_port

if TYPE_CHECKING:
    from crucible.world.world import World

ROLE_HEADER = "AgentEnv-Role"
ACTOR_ARG = "__actor"
EXECUTOR_ROLE = "crucible-executor"
SYSTEM_ROLE = "crucible-system"  # kept disabled: environment dynamics call Apps directly, never through a header
HOST_PROCESS_TEMPLATES = ("common/mcp_server", "common/website", "common/browser", "openenv/awm")


def gw_name(app: str, tool: str) -> str:
    return f"{app}__{tool}"


class GatewayRouter(ToolRouter):
    def __init__(self, backend: AgentEnvBackend):
        self.b = backend

    async def call(self, actor: str, app: str, tool: str, args: dict[str, Any], call_id: str, *, system: bool = False) -> ToolResult:
        if system:  # environment dynamics: straight to the App; no role header that a client could also send
            assert self.b._world is not None
            return await self.b._world.apps[app].call(tool, args, actor=actor, call_id=call_id)
        name = gw_name(app, tool)
        t0 = time.monotonic()
        try:
            r = await self.b.http.post("/step", headers={ROLE_HEADER: actor},
                                       json={"action": "call_tool", "tool_name": name, "arguments": {**args, ACTOR_ARG: self.b.sign_actor(actor)}})
        except httpx.HTTPError as e:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=f"gateway unreachable: {e}", error="transport")
        dt = time.monotonic() - t0
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}
        if r.status_code == 403:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=f"Permission denied: {app}.{tool}", error="permission_denied", duration_s=dt)
        if r.status_code != 200:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=str(body.get("error", body)), error="gateway_error", duration_s=dt)
        text = "\n".join(c.get("text", "") for c in body.get("content", []) if c.get("type") == "text")
        return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=not body.get("isError", False), content=text,
                          data=body.get("structuredContent"), error="tool_error" if body.get("isError") else None, duration_s=dt)


class AgentEnvBackend(Backend):
    name = "agentenv"

    def __init__(self, spec, params, *, workdir):
        super().__init__(spec, params, workdir=workdir)
        self._exec = LocalExec()
        self.apps_dir = (workdir / "apps").resolve()
        self.servers: dict[str, AppMcpServer] = {}
        self.gateway: asyncio.subprocess.Process | None = None
        self.gateway_port = free_port()
        self.http = httpx.AsyncClient(base_url=f"http://127.0.0.1:{self.gateway_port}", timeout=120)
        self.relay = None
        self._router = GatewayRouter(self)
        self._allowed: dict[str, set[str]] = {}
        self._acl_changed: set[str] = set()
        self._world: World | None = None
        self._pending_responders: set[str] = set()
        clock_wall = spec.clock is not None and spec.clock.mode == "wall"
        self.delegate = bool(params.get("delegate_events", clock_wall))
        self._delegated: set[str] = set()
        self._trigger_body: dict[str, Any] | None = None
        self._keepalive: asyncio.Task | None = None
        self._keepalive_stop = asyncio.Event()
        self._secret = secrets.token_bytes(32)  # signs the actor identity carried to App servers (spoof-proof __actor)
        self.sandbox_spec = SandboxSpec.model_validate(params["sandbox"]) if params.get("sandbox") else None
        self.sandbox = None

    def sign_actor(self, actor: str) -> str:
        return f"{actor}:{hmac.new(self._secret, actor.encode(), hashlib.sha256).hexdigest()[:32]}"

    def verify_actor(self, value: Any) -> str | None:
        if not isinstance(value, str) or ":" not in value:
            return None
        actor, _, sig = value.rpartition(":")
        return actor if hmac.compare_digest(self.sign_actor(actor), value) and sig else None

    # ---------------------------------------------------------------- Backend API
    @property
    def capabilities(self) -> BackendCapabilities:
        sbx = self.sandbox_spec is not None
        return BackendCapabilities(isolation="container" if sbx else "none", gpus=sbx and any(a.resources.gpus for a in self.spec.apps),
                                   native_acl=True, native_clock=self.spec.clock is not None and self.spec.clock.mode == "wall",
                                   network_enforced=sbx and self.spec.network != "public",
                                   notes=["agentenv gateway: role-based tool access, clock, triggers"]
                                   + ([f"state and commands in a sandbox: {self.sandbox_spec.provider} {self.sandbox_spec.image}"] if sbx else []))

    def validate(self) -> list[str]:
        problems = []
        if self.sandbox_spec is None:
            problems += [f"app {a.id!r} needs GPUs: give the backend a sandbox (backend_params.sandbox: docker on a GPU host, or modal)"
                         for a in self.spec.apps if a.resources.gpus]
            if self.spec.isolation.policy_user:
                problems.append("isolation.policy_user needs a container: set backend_params.sandbox")
        else:
            for a in self.spec.apps:
                if a.template in HOST_PROCESS_TEMPLATES or (a.template == "common/sqlite_app" and a.params.get("tools_module")):
                    problems.append(f"app {a.id!r} runs a host-side server/tool module; in a sandboxed agentenv world it would run outside the sandbox")
        return problems

    async def provision(self) -> None:
        if self.sandbox_spec is None:
            self.apps_dir.mkdir(parents=True, exist_ok=True)
            return
        from crucible.interop.agentenv.sandbox import open_sandbox

        iso = self.spec.isolation
        self.sandbox = open_sandbox(self.sandbox_spec, self.spec, pids_limit=iso.pids_limit if iso.policy_user else None, memory=iso.memory)
        await self.sandbox.start()
        self._exec = self.sandbox.exec
        probe = await self._exec.run(f"mkdir -p {self.sandbox_spec.workdir} && command -v python3 >/dev/null && command -v git >/dev/null && echo ok", timeout_s=120)
        if "ok" not in probe.stdout:
            raise RuntimeError(f"sandbox image needs python3 and git for Crucible's state adapters ({self.sandbox.description}): {probe.stderr[-300:]}")

    def exec_for(self, app: AppSpec) -> Exec:
        return self._exec

    def base_path(self, app: AppSpec) -> str:
        return self.sandbox_spec.workdir if self.sandbox_spec else str(self.apps_dir)

    def router(self) -> ToolRouter:
        return self._router

    def delegated_events(self) -> set[str]:
        return set(self._delegated)

    @property
    def gateway_url(self) -> str:
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def mcp_url(self) -> str:
        """Aggregate MCP endpoint for external harnesses (send header AgentEnv-Role: <actor>)."""
        return f"{self.gateway_url}/mcp"

    async def after_apps_started(self, world: World) -> None:
        self._world = world
        if self.sandbox_spec is not None and self.spec.isolation.policy_user:
            from crucible.world.guard.isolate import isolate_world

            # the gateway's host port must be unreachable from the policy (it serves every tool to any role header)
            self.isolation = await isolate_world(world, self._exec, base_path=self.sandbox_spec.workdir, root_only=self.params.get("root_only", []),
                                                 egress_targets=[["host.docker.internal", self.gateway_port], ["172.17.0.1", self.gateway_port]])
        default_actor = self.params.get("default_actor") or world.spec.policy_actors[0].id
        for app_id, app in world.apps.items():
            async def caller(name: str, args: dict[str, Any], _app=app) -> ToolResult:
                # only the World's own router can name an actor (HMAC); any other MCP client acts as the default actor
                actor = self.verify_actor(args.pop(ACTOR_ARG, None)) or default_actor
                return await _app.call(name, args, actor=actor, call_id=f"gw-{time.monotonic_ns()}")

            srv = AppMcpServer(app_id, world._tools[app_id], caller, prefix=True)
            await srv.start()
            self.servers[app_id] = srv
        await self._start_gateway()
        await self._start_keepalive()
        await self._program_acl(world)
        if world.spec.clock is not None and world.spec.clock.mode == "wall":
            await self._set_clock(world.spec.clock.t0, world.spec.clock.rate)
        if self.delegate and world.spec.events:
            await self._register_triggers(world)
        await self._refresh_allowed(world)
        self._acl_changed.clear()

    async def _start_gateway(self) -> None:
        env = {
            **os.environ,
            "INTERNAL_MCP_SERVERS": ",".join(f"{n}={s.url}" for n, s in self.servers.items()),
            "MCP_HOST": "127.0.0.1",
            "MCP_PORT": str(self.gateway_port),
            "MCP_SERVER_NAME": f"crucible-{self.spec.world.replace('/', '-')}"[:60],
            "GATEWAY_MODE": self.params.get("gateway_mode", "consistent"),
            "GATEWAY_TRAJECTORY_FILE": str(self.workdir / "gateway_trajectory.jsonl"),
        }
        self.gateway = await spawn_logged(sys.executable, "-m", "agent_env.env.gateway.entrypoint", env=env, log=self.workdir / "gateway.log")
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            if self.gateway.returncode is not None:
                raise RuntimeError(f"agentenv gateway exited ({self.gateway.returncode}); see {self.workdir / 'gateway.log'}")
            with contextlib.suppress(httpx.HTTPError):
                r = await self.http.get("/state")
                if r.status_code == 200 and r.json().get("mcp_servers"):
                    return
            await asyncio.sleep(0.1)
        raise TimeoutError("agentenv gateway did not become ready")

    async def _start_keepalive(self) -> None:
        """Hold one MCP session open on the gateway for the world's lifetime.

        agentenv quirk: the gateway starts its time-trigger driver inside the FastMCP server lifespan,
        which only runs while an MCP client session is open (``gateway.py`` lifespan ->
        ``start_driver``; ``stop_driver`` on exit). REST-only use (``/step``) never starts it, so time
        triggers would silently never fire.
        """
        ready = asyncio.Event()

        async def hold() -> None:
            from mcp.client.session import ClientSession
            from mcp.client.streamable_http import streamable_http_client

            async with (
                httpx.AsyncClient(headers={ROLE_HEADER: EXECUTOR_ROLE}, timeout=None) as http,
                streamable_http_client(f"{self.gateway_url}/mcp", http_client=http) as (r, w, _),
                ClientSession(r, w) as session,
            ):
                await session.initialize()
                ready.set()
                await self._keepalive_stop.wait()

        self._keepalive = asyncio.create_task(hold())
        await asyncio.wait_for(ready.wait(), 30)

    def _expand(self, world: World, patterns: list[str]) -> list[str]:
        import fnmatch

        return sorted(gw_name(t.app, t.name) for t in world.all_tools() if any(fnmatch.fnmatchcase(t.qualified, p) for p in patterns))

    async def _toggle(self, action: str, role: str, tools: list[str] | str) -> None:
        if tools == []:
            return
        r = await self.http.post(f"/tools/{action}", json={"role": role, "tools": tools})
        r.raise_for_status()

    async def _program_acl(self, world: World) -> None:
        # Nobody gets anything by default (anonymous clients use role "default").
        await self._toggle("disable", "default", "*")
        for a in world.spec.actors:
            await self._toggle("disable", a.id, "*")
            await self._toggle("enable", a.id, self._expand(world, a.tools))
        await self._toggle("disable", EXECUTOR_ROLE, "*")
        await self._toggle("disable", SYSTEM_ROLE, "*")

    async def _refresh_allowed(self, world: World) -> None:
        for a in world.spec.actors:
            r = await self.http.post("/step", headers={ROLE_HEADER: a.id}, json={"action": "list_tools"})
            r.raise_for_status()
            names = {t["name"] for t in r.json()["tools"]}
            if self._allowed.get(a.id) is not None and self._allowed[a.id] != names:
                self._acl_changed.add(a.id)
            self._allowed[a.id] = names

    def allowed_tools(self, actor: str) -> list[ToolSpec]:
        assert self._world is not None
        allowed = self._allowed.get(actor, set())
        return [t for t in self._world.all_tools() if gw_name(t.app, t.name) in allowed]

    def pop_acl_changes(self) -> set[str]:
        out, self._acl_changed = self._acl_changed, set()
        return out

    async def _set_clock(self, t: str, rate: float) -> None:
        r = await self.http.put("/clock/set-time", json={"virtual_time": t, "virtual_seconds_per_real_second": rate})
        r.raise_for_status()

    async def gateway_time(self) -> str | None:
        r = await self.http.get("/clock/time")
        return r.json().get("virtual_time") if r.status_code == 200 else None

    def _trigger(self, world: World, ev: EventSpec) -> dict[str, Any] | None:
        w, a = ev.when, ev.action
        if w.on_actor:  # agentenv watches roles globally, not per trigger: keep it in Crucible's scheduler
            return None
        if w.on_tool:
            app, _, tool = w.on_tool.partition(".")
            if "*" in w.on_tool or "?" in w.on_tool:
                return None
            when: dict[str, Any] = {"type": "action", "tool": gw_name(app, tool), "repeat": bool(w.repeat or w.every)}
        else:
            if world.spec.clock is None or world.spec.clock.mode != "wall":
                return None  # time triggers need the gateway clock armed
            when = {"type": "time"}
            if w.at:
                when["at"] = w.at
            else:
                when["at"] = f"PT{int(parse_duration(w.after or '0s').total_seconds())}S"
            if w.every:
                when["every"] = f"PT{int(parse_duration(w.every).total_seconds())}S"
            if w.count:
                when["count"] = w.count
        if a.type == "tool":
            app, _, tool = (a.tool or "").partition(".")
            action = {"type": "tool", "tool": gw_name(app, tool), "args": a.args}
        elif a.type == "permission":
            target = a.to or ev.actor
            actions = []
            if a.grant:
                actions.append({"type": "permission", "action": "enable", "role": target, "tools": self._expand(world, a.grant)})
            if a.revoke:
                actions.append({"type": "permission", "action": "disable", "role": target, "tools": self._expand(world, a.revoke)})
            return {"id": ev.id, "when": when, "actions": actions}
        else:
            action = {"type": "nl", "instruction": json.dumps({"to": a.to, "from": ev.actor, "text": a.text, "event": ev.id})}
        return {"id": ev.id, "when": when, "actions": [action]}

    async def _register_triggers(self, world: World) -> None:
        from crucible.interop.agentenv.a2a import ServedAgent, make_relay_agent

        triggers = []
        for ev in world.spec.events:
            t = self._trigger(world, ev)
            if t is not None:
                triggers.append(t)
                self._delegated.add(ev.id)
        if not triggers:
            return
        body: dict[str, Any] = {"watch_roles": [a.id for a in world.spec.policy_actors], "triggers": triggers}
        if any(act["type"] == "nl" for t in triggers for act in t["actions"]):
            self.relay = await ServedAgent(make_relay_agent(self._deliver)).start()
            body["executor"] = {"a2a_url": self.relay.url, "role": EXECUTOR_ROLE, "timeout_seconds": 60}
        self._trigger_body = body
        r = await self.http.post("/triggers/register", json=body)
        if r.status_code != 200:
            raise RuntimeError(f"trigger registration failed: {r.text[:500]}")

    async def on_reset(self, world: World) -> None:
        """A fresh episode: clock back to the spec's t0, base role rules, triggers re-registered (fire counts reset)."""
        await self._program_acl(world)
        if world.spec.clock is not None and world.spec.clock.mode == "wall":
            await self._set_clock(world.spec.clock.t0, world.spec.clock.rate)
        if self._trigger_body is not None:
            (await self.http.post("/triggers/clear")).raise_for_status()
            r = await self.http.post("/triggers/register", json=self._trigger_body)
            if r.status_code != 200:
                raise RuntimeError(f"trigger re-registration failed: {r.text[:500]}")
        self._pending_responders.clear()
        await self._refresh_allowed(world)
        self._acl_changed.clear()

    async def _deliver(self, payload: dict[str, Any]) -> None:
        world = self._world
        assert world is not None
        targets = [payload["to"]] if payload.get("to") else [a.id for a in world.spec.policy_actors]
        at = await self.gateway_time()
        for t in targets:
            world.inboxes.setdefault(t, []).append(Inbound(**{"from": payload.get("from") or "world", "text": payload.get("text") or "", "at": at}))
            world._log("action", payload.get("from"), "event:nl", text=payload.get("text"), data={"event": payload.get("event"), "via": "agentenv-trigger"})
            if t in world.responders:
                self._pending_responders.add(t)

    async def after_step(self, world: World) -> None:
        await asyncio.sleep(0)  # let in-flight relay deliveries land
        for t in sorted(self._pending_responders):
            await world._run_responder(t)
        self._pending_responders.clear()
        await self._refresh_allowed(world)

    async def snapshot_state(self) -> dict[str, Any]:
        st: dict[str, Any] = {}
        r = await self.http.get("/clock/state")
        if r.status_code == 200:
            st["clock"] = r.json()
        r = await self.http.get("/triggers/state")
        if r.status_code == 200:
            st["triggers"] = r.json()
        for a in self._allowed:
            st.setdefault("allowed", {})[a] = sorted(self._allowed[a])
        return st

    async def restore_state(self, state: dict[str, Any]) -> None:
        world = self._world
        if world is None or not state:
            return
        clock = state.get("clock") or {}
        if clock.get("armed") and clock.get("virtual_time"):
            await self._set_clock(clock["virtual_time"], clock.get("virtual_seconds_per_real_second", 1))
        # Role rules: restore exactly the allowed tool sets captured with the snapshot.
        for actor, names in (state.get("allowed") or {}).items():
            await self._toggle("disable", actor, "*")
            await self._toggle("enable", actor, list(names))
        await self._refresh_allowed(world)
        self._acl_changed.clear()

    async def teardown(self) -> None:
        self._keepalive_stop.set()
        if self._keepalive is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._keepalive, 5)
        if self.gateway is not None and self.gateway.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(self.gateway.pid, 15)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self.gateway.wait(), 5)
        if self.relay is not None:
            await self.relay.stop()
        for s in self.servers.values():
            await s.stop()
        await self.http.aclose()
        if self.sandbox is not None:
            await self.sandbox.close()
            self.sandbox = None
        if not self.params.get("keep_workdir"):
            import shutil

            shutil.rmtree(self.apps_dir, ignore_errors=True)
