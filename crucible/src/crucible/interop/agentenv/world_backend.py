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
for MCP clients that do not identify themselves; default: first policy actor), ``gateway_bind`` (the host each
actor's ``AgentProxy`` listens on, e.g. ``172.17.0.1`` for agent containers on a Linux daemon; default loopback).

M11 lab audit R-2 (grader isolation, P21/P23):

- **The gateway listens on loopback only.** Its control routes (``/tools/*``, ``/clock/*``, ``/triggers/*``,
  ``/trajectory``, ``/state``) have no authentication and take the role from a header (agentenv ``gateway.py:189-341``).
  ``gateway_bind`` used to bind the gateway itself, so an agent container on docker0 could re-enable a revoked tool or
  move the clock through the port; now only the proxies (``/mcp`` and ``call_tool``/``list_tools``, role pinned) listen
  there (SkyCap's two-listener shape). A host process with a shell still reaches loopback: that needs the World
  sandboxed (``isolation.policy_user``). ``gateway_isolation`` names which holds (``sandboxed`` or ``bound_loopback``),
  recorded with every grade.
- **Signed actors cannot be replayed.** The ``__actor`` a call carries to an App server is an HMAC over the actor, a
  nonce and the tool. The World's own calls (``GatewayRouter``) carry a single-use nonce and the MAC covers their
  arguments; a trigger's tool action and state checks carry a reusable one (the gateway runs them again with templated
  arguments; their arguments are not logged, ``triggers.py:1243``), still bound to their one tool. A token that fails, one
  spent already, or one for an actor the pre-grade cut has cut is refused (it used to be accepted as long as it matched: a static HMAC, logged in the
  gateway's trajectory, let a reader call ``App.call`` directly on the unauthenticated App-server port, skipping ACL,
  quotas, the log and the clock).
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import datetime
import hashlib
import hmac
import json
import os
import secrets
import sys
import time
import zlib
from typing import TYPE_CHECKING, Any

import httpx

from crucible.core.schema.actions import Inbound, ToolResult, ToolSpec
from crucible.core.schema.world import AppSpec, EventSpec, SandboxSpec
from crucible.core.triggers import exp_interval, exp_stream
from crucible.world.backends.base import Backend, BackendCapabilities, ToolRouter
from crucible.world.clock import parse_duration, parse_time
from crucible.world.exec import Exec, LocalExec, safe_environ, spawn_logged
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


# ---- restoring delegated triggers (harness-reward audit §5, 2026-10-28) -------------------------------------------------------
# agentenv has no trigger import API: a restore clears the engine and registers the triggers again, and a fresh registration
# starts from nothing (no fires, no mark, a fresh RNG, marks resolved against the restored clock's t0). Each trigger of the
# original body is therefore re-planned from its *progress* (fires so far, its next mark, the draw its seeded stream is at),
# and the gateway triggers standing in for it ("parts") are remembered, so a later snapshot reads its progress whatever
# registration it is on (a restore of a restore). The mark arithmetic is agentenv's (``triggers.py`` ``_resolve_first_mark``,
# ``_advance_marks``, ``_next_interval_seconds``): the first mark of an ``every`` with no ``at`` is t0 + one interval, each
# fire advances the mark by the next draw of ``random.Random(seed)``.
MAX_EXPAND = 512  # arrivals of a seeded exponential recurrence registered one by one; past them it continues on a derived seed


def _iso(dt: Any) -> str:
    """agentenv's RFC 3339 rendering (``clock._iso``: microseconds kept, ``Z``)."""
    return dt.astimezone(datetime.UTC).isoformat().replace("+00:00", "Z")


def _instant(text: str, t0: Any) -> Any:
    """An ``at`` / ``until`` value as an instant: RFC 3339, or a P-duration from ``t0`` (agentenv ``_time_mark_spec``)."""
    return t0 + parse_duration(text) if text.startswith("P") else parse_time(text)


def _mean_s(every: dict[str, Any]) -> float:
    return parse_duration(every["mean"]).total_seconds()


def _first_mark(when: dict[str, Any], t0: Any) -> tuple[str | None, int | None]:
    """The first mark of a registered, not yet evaluated time trigger and the draw that follows it (None: not stochastic)."""
    stoch = isinstance(when.get("every"), dict)
    if "at" in when:
        return (_iso(_instant(when["at"], t0)) if t0 is not None or not when["at"].startswith("P") else None), (0 if stoch else None)
    if t0 is None or "every" not in when:
        return None, None
    if stoch:
        return _iso(t0 + datetime.timedelta(seconds=exp_interval(int(when["seed"]), _mean_s(when["every"]), 0))), 1
    return _iso(t0 + parse_duration(when["every"])), None


def native_plan(t: dict[str, Any]) -> dict[str, Any]:
    """A trigger registered as itself (a fresh registration of the original body)."""
    return {"fired": 0, "parts": [{"gid": t["id"], "kind": "native", "when": t["when"]}]}


def trigger_progress(body: list[dict[str, Any]], plan: dict[str, dict[str, Any]], states: list[dict[str, Any]], t0: Any) -> dict[str, dict[str, Any]]:
    """Per trigger of the original ``body`` (by id), its progress over every registration so far, from the engine's settled
    state (``/triggers/state`` rows) and ``plan`` (id -> {"fired": fires before this registration, "parts": the gateway
    triggers standing in for it}; absent: registered as itself). A row: ``fired`` and ``done``, and while it can still fire
    ``next_mark`` (with ``draw``, the index of the stream draw that follows it, and ``seed`` for a seeded recurrence), or
    ``after`` (an anchored trigger whose anchor has not fired)."""
    st = {s.get("id"): s for s in states}
    out = {}
    for t in body:
        p = plan.get(t["id"]) or native_plan(t)
        parts = p["parts"]
        fired = int(p.get("fired") or 0) + sum(int((st.get(x["gid"]) or {}).get("fire_count") or 0) for x in parts)
        out[t["id"]] = {"fired": fired, "done": True}
        if any((st.get(x["gid"]) or {}).get("status") == "failed" for x in parts):
            continue  # a failed fire retires the trigger (``_fire`` / ``_fire_time``): it never fires again
        for x in parts:
            s = st.get(x["gid"]) or {}
            n, status = int(s.get("fire_count") or 0), s.get("status")
            if x["kind"] == "arrival":
                if status == "fired" or n:
                    continue
                out[t["id"]] = {"fired": fired, "done": False, "next_mark": x["mark"], "draw": x["draw"], "seed": x["seed"]}
                break
            w = x["when"]
            if status == "fired" or (n and "every" not in w and not w.get("repeat")):
                continue  # retired: a one-shot that fired, a recurrence past its count or until
            if w["type"] != "time":
                out[t["id"]] = {"fired": fired, "done": False}
                break
            stoch = isinstance(w.get("every"), dict)
            mark, draw = s.get("next_mark"), ((n + (0 if "at" in w else 1)) if stoch else None)
            if mark is None and status in ("firing", "queued"):
                continue  # its last arrivals are firing: it retires when they land
            if mark is None and "after" in w:
                out[t["id"]] = {"fired": fired, "done": False, "after": t["when"].get("after", w["after"]), "offset": w["offset"]}
                break
            if mark is None:  # registered, not evaluated yet (no clock tick since): its first mark, as agentenv resolves it
                mark, draw = _first_mark(w, t0)
            if mark is None:
                continue
            out[t["id"]] = {"fired": fired, "done": False, "next_mark": mark, "draw": draw, "seed": w.get("seed")}
            break
    return out


def plan_restore(body: list[dict[str, Any]], progress: dict[str, dict[str, Any]], t0: Any) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """The gateway triggers that continue every trigger of ``body`` from ``progress`` (``trigger_progress``), and the plan
    that maps them back. A one-shot or anchored trigger waiting on its mark is registered at that mark (an anchored one
    used to wait for an anchor that had already fired, and was lost); a fixed recurrence restarts at its next mark with
    the rest of its ``count`` (it used to restart one interval after the restore); a seeded exponential recurrence is
    expanded into its next arrivals, each at the mark its own stream gives (``core.triggers.exp_stream``; it used to draw
    from the start of the stream again), up to ``MAX_EXPAND``, then continues on a seed derived from the stream's position.
    A trigger that fired its last time, failed, or waits on an anchor that can no longer fire is not registered."""
    specs: dict[str, list[dict[str, Any]]] = {}
    plan: dict[str, dict[str, Any]] = {}
    todo = list(body)
    while todo:  # anchors before the triggers anchored on them (an anchored trigger names its anchor's first part)
        later = [t for t in todo if (progress.get(t["id"]) or {}).get("after") in {x["id"] for x in todo if x is not t}]
        if len(later) == len(todo):
            later = []  # an anchor cycle: agentenv never resolves it either
        for t in (t for t in todo if t not in later):
            tid, w0 = t["id"], t["when"]
            pr = progress.get(tid) or {"fired": 0, "done": False}
            fired = int(pr.get("fired") or 0)
            plan[tid] = {"fired": fired, "parts": []}
            specs[tid] = []
            if pr.get("done"):
                continue
            if w0["type"] != "time":
                specs[tid] = [t]
                plan[tid]["parts"] = [{"gid": tid, "kind": "native", "when": w0}]
                continue
            if pr.get("after") is not None:
                anchor = (plan.get(pr["after"]) or {}).get("parts") or []
                if not anchor:
                    continue  # its anchor will never fire again
                when = {**w0, "after": anchor[0]["gid"]}
                specs[tid] = [{**t, "when": when}]
                plan[tid]["parts"] = [{"gid": tid, "kind": "native", "when": when}]
                continue
            if pr.get("next_mark") is None:
                continue
            base = {"type": "time", "at": pr["next_mark"]}
            if "every" not in w0:  # a one-shot (``at``, or an anchored trigger whose anchor fired): at its mark
                specs[tid] = [{**t, "when": base}]
                plan[tid]["parts"] = [{"gid": tid, "kind": "native", "when": base}]
                continue
            left = int(w0["count"]) - fired if "count" in w0 else None
            if left is not None and left <= 0:
                continue
            until = _instant(w0["until"], t0) if "until" in w0 and (t0 is not None or not w0["until"].startswith("P")) else None
            bound = {**({"count": left} if left is not None else {}), **({"until": _iso(until)} if until is not None else {})}
            if not isinstance(w0["every"], dict):
                when = {**base, "every": w0["every"], **bound}
                specs[tid] = [{**t, "when": when}]
                plan[tid]["parts"] = [{"gid": tid, "kind": "native", "when": when}]
                continue
            seed, k = int(pr["seed"]), int(pr["draw"])
            m, draws, n = parse_time(pr["next_mark"]), exp_stream(seed, _mean_s(w0["every"]), k), 0
            while (left is None or n < left) and (until is None or m <= until) and n < MAX_EXPAND:
                gid = f"{tid}~{fired + n}"
                specs[tid].append({**t, "id": gid, "when": {"type": "time", "at": _iso(m)}})
                plan[tid]["parts"].append({"gid": gid, "kind": "arrival", "mark": _iso(m), "draw": k + n, "seed": seed})
                m, n = m + datetime.timedelta(seconds=next(draws)), n + 1
            if n == MAX_EXPAND and (left is None or n < left) and (until is None or m <= until):
                tail_seed = zlib.crc32(f"{seed}:{k + n}".encode())
                when = {"type": "time", "at": _iso(m), "every": w0["every"], "seed": tail_seed,
                        **({"count": left - n} if left is not None else {}), **({"until": _iso(until)} if until is not None else {})}
                gid = f"{tid}~{fired + n}+"
                specs[tid].append({**t, "id": gid, "when": when})
                plan[tid]["parts"].append({"gid": gid, "kind": "native", "when": when})
        todo = later
    return [s for t in body for s in specs.get(t["id"], [])], plan


# set while the World steps for an outside agent's call: the gateway already holds that call (and its consistent-mode tool
# lock), so the World's own tool calls in that step (the call itself, responders, events) go straight to the Apps
_DIRECT: contextvars.ContextVar[bool] = contextvars.ContextVar("crucible_agentenv_direct", default=False)


class GatewayRouter(ToolRouter):
    def __init__(self, backend: AgentEnvBackend):
        self.b = backend

    async def call(self, actor: str, app: str, tool: str, args: dict[str, Any], call_id: str, *, system: bool = False) -> ToolResult:
        if system:  # environment dynamics: straight to the App; no role header that a client could also send
            assert self.b._world is not None
            return await self.b._world.apps[app].call(tool, args, actor=actor, call_id=call_id)
        name = gw_name(app, tool)
        if _DIRECT.get():
            # inside an outside agent's gateway call: the gateway's role rules, read back, decide (they were applied to
            # that call); the App raises into the World's choke point, which records the fault
            if name not in self.b._allowed.get(actor, set()):
                return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=f"Permission denied: {app}.{tool}", error="permission_denied")
            assert self.b._world is not None
            return await self.b._world.apps[app].call(tool, args, actor=actor, call_id=call_id)
        t0 = time.monotonic()
        try:
            r = await self.b.http.post("/step", headers={ROLE_HEADER: actor},
                                       json={"action": "call_tool", "tool_name": name, "arguments": {**args, ACTOR_ARG: self.b.sign_actor(actor, name, args=args)}})
        except httpx.HTTPError as e:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=f"gateway unreachable: {e}", error="transport")
        dt = time.monotonic() - t0
        body = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text}
        if r.status_code == 403:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=f"Permission denied: {app}.{tool}", error="permission_denied", duration_s=dt)
        if r.status_code != 200:
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=str(body.get("error", body)), error="gateway_error", duration_s=dt)
        text = "\n".join(c.get("text", "") for c in body.get("content", []) if c.get("type") == "text")
        sc = body.get("structuredContent") if isinstance(body.get("structuredContent"), dict) else None
        if body.get("isError"):  # the App server's fault label survives the hop (H2 V2)
            sc = sc or {}
            return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=False, content=text, error=sc.get("crucible_error") or "tool_error",
                              fault=sc.get("crucible_fault"), duration_s=dt)
        return ToolResult(tool_call_id=call_id, function_name=f"{app}.{tool}", ok=True, content=text, data=sc, duration_s=dt)


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
        # the gateway triggers standing in for each delegated trigger since the last restore (``plan_restore``); empty: each
        # is registered as itself
        self._trigger_plan: dict[str, dict[str, Any]] = {}
        self._keepalive: asyncio.Task | None = None
        self._keepalive_stop = asyncio.Event()
        self._secret = secrets.token_bytes(32)  # signs the actor identity carried to App servers (spoof-proof __actor)
        self._spent: set[str] = set()  # single-use nonces already presented (M11 lab audit R-2: a replay is refused)
        self.sandbox_spec = SandboxSpec.model_validate(params["sandbox"]) if params.get("sandbox") else None
        self.sandbox = None
        self.proxies: dict[str, AgentProxy] = {}

    @staticmethod
    def _args_digest(args: dict[str, Any] | None) -> str:
        """The call's arguments (``__actor`` aside) as canonical JSON: the gateway forwards them unchanged
        (``gateway.py:928-953``; the App server does not validate, ``app_server.py``), so both sides see the same."""
        body = {k: v for k, v in (args or {}).items() if k != ACTOR_ARG}
        return hashlib.sha256(json.dumps(json.loads(json.dumps(body, default=str)), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _mac(self, actor: str, nonce: str, tool: str, digest: str) -> str:
        return hmac.new(self._secret, f"{actor}\0{nonce}\0{tool}\0{digest}".encode(), hashlib.sha256).hexdigest()[:32]

    def sign_actor(self, actor: str, tool: str, *, once: bool = True, args: dict[str, Any] | None = None) -> str:
        """``actor`` signed for a call of the gateway tool ``tool`` (``<app>__<tool>``): ``actor:nonce:mac`` (module doc,
        M11 lab audit R-2). ``once``: the nonce is spent by its first use and the MAC covers ``args`` too (the World's own
        calls: the gateway logs the token before it forwards the call, so a reader racing it can only repeat that exact
        call, once); a trigger's action or check, which the gateway runs again with templated arguments, gets a reusable
        nonce, still valid for that one tool only."""
        nonce = ("1" if once else "0") + secrets.token_hex(8)
        return f"{actor}:{nonce}:{self._mac(actor, nonce, tool, self._args_digest(args) if once else '')}"

    def verify_actor(self, value: Any, tool: str, args: dict[str, Any] | None = None) -> str | None:
        """The actor of a token signed for ``tool`` (and, single-use, for exactly ``args``), or None (unsigned, forged,
        signed for another tool or other arguments, or a spent single-use nonce: a replay)."""
        if not isinstance(value, str):
            return None
        rest, _, mac = value.rpartition(":")
        actor, _, nonce = rest.rpartition(":")
        if not (actor and nonce and mac):
            return None
        digest = self._args_digest(args) if nonce.startswith("1") else ""
        if not hmac.compare_digest(self._mac(actor, nonce, tool, digest), mac):
            return None
        if nonce.startswith("1"):
            if nonce in self._spent:
                return None
            self._spent.add(nonce)
        return actor

    @property
    def gateway_isolation(self) -> str:
        """What keeps policy processes off the gateway port (M11 lab audit R-2):

        - ``sandboxed``: the World's sandbox ran the egress probe for the policy's uid against the gateway port and it was
          unreachable (``isolate.py``: no-network or allowlist worlds with ``egress_probe``);
        - ``bound_loopback``: the gateway listens on 127.0.0.1 only, on a Linux host: unreachable from containers on the
          docker0 bridge and from other machines, reachable by a host process with a shell;
        - ``exposed``: not Linux. Docker Desktop (macOS, Windows) routes ``host.docker.internal`` to the host's loopback, so
          an agent container reaches the gateway's control routes and the App servers. (Docker Desktop on Linux and WSL2
          mirrored networking do the same and are not detected.)"""
        egress = getattr(getattr(self, "isolation", None), "egress", None) or {}
        if egress.get("isolated") and any(str(self.gateway_port) in str(t) for t in egress.get("tested") or []):
            return "sandboxed"
        return "bound_loopback" if sys.platform.startswith("linux") else "exposed"

    # ---------------------------------------------------------------- Backend API
    @property
    def capabilities(self) -> BackendCapabilities:
        sbx = self.sandbox_spec is not None
        gv = sbx and self.sandbox_spec.runtime in ("runsc", "runsc-in-container")
        return BackendCapabilities(isolation=("gvisor" if gv else "container") if sbx else "none", gpus=sbx and any(a.resources.gpus for a in self.spec.apps),
                                   native_acl=True, native_clock=self.spec.clock is not None and self.spec.clock.mode == "wall",
                                   network_enforced=sbx and self.spec.network != "public", pausable=sbx and hasattr(self.sandbox_cls_hint(), "pause"),
                                   notes=["agentenv gateway: role-based tool access, clock, triggers"]
                                   + ([f"state and commands in a sandbox: {self.sandbox_spec.provider} {self.sandbox_spec.image}"] if sbx else []))

    def sandbox_cls_hint(self):
        if self.sandbox_spec is None:
            return object
        if self.sandbox_spec.provider == "docker":
            from crucible.world.sandbox import DockerSandbox

            return DockerSandbox
        return object

    async def pause(self) -> None:
        if self.sandbox is None or not hasattr(self.sandbox, "pause"):
            raise NotImplementedError("this agentenv world has no pausable sandbox")
        await self.sandbox.pause()

    async def resume(self) -> None:
        await self.sandbox.resume()
        w = self._world
        if w is not None and w.spec.clock is not None and w.spec.clock.mode == "wall":
            await self._set_clock(w.clock.now_iso(), w.spec.clock.rate)  # the gateway's clock continues from the frozen time

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
        """The gateway, on loopback whatever ``gateway_bind`` says (M11 lab audit R-2): for Crucible and the orchestration
        steps on this host only."""
        return f"http://127.0.0.1:{self.gateway_port}"

    @property
    def mcp_url(self) -> str:
        """The gateway's aggregate MCP endpoint: for Crucible itself and trusted orchestration. Never hand it to a policy:
        the same port serves the unauthenticated control plane (use ``agent_mcp_url``)."""
        return f"{self.gateway_url}/mcp"

    async def agent_mcp_url(self, actor: str | None = None) -> str:
        """The MCP endpoint for an outside agent acting as ``actor``: a proxy that forwards only ``/mcp`` and REST
        ``call_tool`` / ``list_tools``, with the role pinned (M11 audit H2 V6, G4 D5: the agent re-enabled a revoked tool
        through ``/tools/enable`` and could set the clock and read the trajectory, which carries actor tokens)."""
        assert self._world is not None
        actor = actor or self.params.get("default_actor") or self._world.spec.policy_actors[0].id
        if actor not in self.proxies:
            px = AgentProxy(self.gateway_url, actor, host=self.params.get("gateway_bind", "127.0.0.1"))
            await px.start()
            self.proxies[actor] = px
        return self.proxies[actor].mcp_url

    async def agent_url(self, actor: str | None = None) -> str:
        """The agent proxy's base URL (``agent_mcp_url`` without ``/mcp``)."""
        return (await self.agent_mcp_url(actor)).removesuffix("/mcp")

    async def after_apps_started(self, world: World) -> None:
        self._world = world
        if self.sandbox_spec is not None and self.spec.isolation.policy_user:
            from crucible.world.guard.isolate import isolate_world

            # the gateway's host port must be unreachable from the policy (it serves every tool to any role header)
            self.isolation = await isolate_world(world, self._exec, base_path=self.sandbox_spec.workdir, root_only=self.params.get("root_only", []),
                                                 egress_targets=[["host.docker.internal", self.gateway_port], ["172.17.0.1", self.gateway_port]])
        default_actor = self.params.get("default_actor") or world.spec.policy_actors[0].id
        for app_id, app in world.apps.items():
            async def caller(name: str, args: dict[str, Any], _app=app, _app_id=app_id) -> ToolResult:
                token = args.pop(ACTOR_ARG, None)
                signed = self.verify_actor(token, gw_name(_app_id, name), args)
                if token is not None and signed is None:
                    # M11 lab audit R-2: a forged, replayed or misdirected token is refused; only Crucible signs, and every
                    # path it signs for presents a fresh or a tool-bound token, so this is never the World's own call
                    world._log("state", None, "actor_token_refused", data={"tool": f"{_app_id}.{name}"})
                    if world.cut:  # after the pre-grade cut it is a late call too (agentenv.cut)
                        world.late_call(default_actor, f"{_app_id}.{name} (invalid {ACTOR_ARG})")
                    return ToolResult(tool_call_id="gw", function_name=f"{_app_id}.{name}", ok=False, error="refused", fault="policy",
                                      content=f"refused: {ACTOR_ARG} is not a valid token for {_app_id}.{name}")
                if signed is not None and signed in world.cut:
                    # the World never signs for a cut actor (its step refuses first): a token taken before the cut, used after it
                    world.late_call(signed, f"{_app_id}.{name}")
                    return ToolResult(tool_call_id="gw", function_name=f"{_app_id}.{name}", ok=False, error="refused", fault="policy",
                                      content=f"refused: {signed!r} was cut before grading")
                if signed is not None:
                    # the World's own router: the World already ran its choke point for this call. A crash is labelled
                    # infra so the label crosses the MCP hop (H2 V2)
                    try:
                        return await _app.call(name, args, actor=signed, call_id=f"gw-{time.monotonic_ns()}")
                    except Exception as e:  # noqa: BLE001
                        return ToolResult(tool_call_id="gw", function_name=f"{_app_id}.{name}", ok=False, content=f"internal error in {_app_id}.{name}: {type(e).__name__}",
                                          error="internal_error", fault="infra")
                # an outside agent through the gateway (MCP or REST): the call is a World step, so the choke point (schema,
                # catalog, RBAC, interceptors, quotas, fault labels), the log, the clock, events and responders all run
                # (H2 V3, G4 D3: these calls reached App.call directly; a crash scored 0 and no event fired). Unsigned
                # clients act as the default actor.
                return await self._external_step(world, default_actor, _app_id, name, args)

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

    async def _external_step(self, world: World, actor: str, app_id: str, tool: str, args: dict[str, Any]) -> ToolResult:
        from crucible.core.schema.actions import ToolCall
        from crucible.world.world import WorldError

        cid = f"ext-{time.monotonic_ns()}"
        token = _DIRECT.set(True)
        try:
            res = await world.step(actor, ToolCall(tool_call_id=cid, function_name=f"{app_id}.{tool}", arguments=args))
        except WorldError as e:  # e.g. the actor already declared done: the agent's own call, refused
            return ToolResult(tool_call_id=cid, function_name=f"{app_id}.{tool}", ok=False, content=str(e), error="refused", fault="policy")
        finally:
            _DIRECT.reset(token)
        tr = res.observation.tool_results
        return tr[0] if tr else ToolResult(tool_call_id=cid, function_name=f"{app_id}.{tool}", ok=False, content="no result", error="refused", fault="policy")

    async def _start_gateway(self) -> None:
        env = {
            **safe_environ(),
            "INTERNAL_MCP_SERVERS": ",".join(f"{n}={s.url}" for n, s in self.servers.items()),
            # loopback only: the gateway serves every tool to any role header and its control routes are unauthenticated.
            # Agent containers reach the World through their AgentProxy, which binds gateway_bind (m13 audit 02, C2; M11
            # lab audit R-2: gateway_bind used to bind the gateway itself, exposing its control plane on docker0)
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

            from crucible.world.packaging.mcp_compat import http_streams

            async with (
                http_streams(f"{self.gateway_url}/mcp", headers={ROLE_HEADER: EXECUTOR_ROLE}, timeout_s=None) as (r, w),
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

    async def apply_permission(self, world: World, actor: str, *, grant: list[str] | None = None, revoke: list[str] | None = None) -> None:
        """A permission event Crucible's scheduler fired (a step clock: events are not delegated): the gateway's role rule
        changes, since the gateway is what checks access (``native_acl``). Smoke 2026-10-29: only ``World.acl`` changed, so
        the toy helpdesk's month-end freeze revoked ``tickets.issue_refund`` on the local backend and left it callable here."""
        if grant:
            await self._toggle("enable", actor, self._expand(world, list(grant)))
        if revoke:
            await self._toggle("disable", actor, self._expand(world, list(revoke)))
        await self._refresh_allowed(world)

    def pop_acl_changes(self) -> set[str]:
        out, self._acl_changed = self._acl_changed, set()
        return out

    async def _set_clock(self, t: str, rate: float) -> None:
        r = await self.http.put("/clock/set-time", json={"virtual_time": t, "virtual_seconds_per_real_second": rate})
        r.raise_for_status()

    async def gateway_time(self) -> str | None:
        r = await self.http.get("/clock/time")
        return r.json().get("virtual_time") if r.status_code == 200 else None

    @staticmethod
    def _iso_dur(text: str) -> str:
        return f"PT{int(parse_duration(text).total_seconds())}S"

    @staticmethod
    def _gw_tool(qualified: str) -> str:
        app, _, tool = qualified.partition(".")
        return gw_name(app, tool)

    def _gw_check(self, check: dict, actor: str | None = None) -> dict:
        # the gateway's own reads: environment dynamics, not an agent's step. Reusable (the gateway checks again), one tool each
        return {"steps": [{**st, "tool": self._gw_tool(st["tool"]),
                           "args": {**(st.get("args") or {}), ACTOR_ARG: self.sign_actor(actor or SYSTEM_ROLE, self._gw_tool(st["tool"]), once=False)}}
                          for st in check["steps"]]}

    def _trigger(self, world: World, ev: EventSpec) -> dict[str, Any] | None:
        """An EventSpec as an agentenv trigger (the whole urn:agentenv:triggers/v1 language, m13 C8). None keeps the event
        in Crucible's scheduler (per-actor watches, glob tools, time triggers without a wall clock)."""
        from crucible.core.schema.world import EveryDist
        from crucible.world.clock import parse_time

        w, a = ev.when, ev.action
        if w.on_actor:  # agentenv watches roles globally, not per trigger: keep it in Crucible's scheduler
            return None
        if w.on_tool:
            if "*" in w.on_tool or "?" in w.on_tool:
                return None
            when: dict[str, Any] = {"type": "action", "tool": self._gw_tool(w.on_tool), "repeat": bool(w.repeat or w.every)}
            if w.where:
                when["where"] = dict(w.where)
        elif w.state:
            when = {"type": "state", "check": self._gw_check(w.state, ev.actor)}
        else:
            if world.spec.clock is None or world.spec.clock.mode != "wall":
                return None  # time triggers need the gateway clock armed
            t0 = parse_time(world.spec.clock.t0)
            when = {"type": "time"}
            if w.after_event:
                when["after"], when["offset"] = w.after_event, self._iso_dur(w.offset or "0s")
            elif w.at:
                when["at"] = w.at
            elif w.after:  # absolute: a relative mark would re-anchor to whatever time a restore sets the clock to
                when["at"] = (t0 + parse_duration(w.after)).isoformat().replace("+00:00", "Z")
            if isinstance(w.every, EveryDist):
                when["every"], when["seed"] = {"dist": "exp", "mean": self._iso_dur(w.every.mean)}, w.seed
            elif w.every:
                when["every"] = self._iso_dur(w.every)
            if w.count:
                when["count"] = w.count
            if w.until:
                when["until"] = w.until if not w.until.startswith("P") else (t0 + parse_duration(w.until)).isoformat().replace("+00:00", "Z")
        extra: dict[str, Any] = {"verify": self._gw_check(a.verify, ev.actor)} if a.verify else {}
        if a.type == "tool":
            # signed as the event's actor: the gateway executes it, and an unsigned call is an outside agent's World step
            # (M11 audit), which would wait on the step that provoked this trigger
            tool = self._gw_tool(a.tool or "")
            actions = [{"type": "tool", "tool": tool, "args": {**(a.args or {}), ACTOR_ARG: self.sign_actor(ev.actor or SYSTEM_ROLE, tool, once=False)}, **extra}]
        elif a.type == "permission":
            target = a.to or ev.actor
            actions = []
            if a.grant:
                actions.append({"type": "permission", "action": "enable", "role": target, "tools": self._expand(world, a.grant)})
            if a.revoke:
                actions.append({"type": "permission", "action": "disable", "role": target, "tools": self._expand(world, a.revoke)})
            if actions and extra:
                actions[-1].update(extra)
        else:
            actions = [{"type": "nl", "instruction": json.dumps({"to": a.to, "from": ev.actor, "text": a.text, "event": ev.id}), **extra}]
        spec: dict[str, Any] = {"id": ev.id, "when": when, "actions": actions}
        if ev.barrier:
            spec["barrier"] = {"at": ev.barrier}
        return spec

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
            self._trigger_plan = {}  # registered as itself again
            r = await self.http.post("/triggers/register", json=self._trigger_body)
            if r.status_code != 200:
                raise RuntimeError(f"trigger re-registration failed: {r.text[:500]}")
        self._pending_responders.clear()
        await self._refresh_allowed(world)
        self._acl_changed.clear()

    def _clock_t0(self) -> Any:
        """The spec clock's t0: what the original registration's marks resolve against (``on_reset`` arms the clock there)."""
        c = self._world.spec.clock if self._world is not None else None
        return parse_time(c.t0) if c is not None and c.mode == "wall" else None

    def trigger_progress(self, snap: dict[str, Any]) -> dict[str, dict[str, Any]]:
        """Every delegated trigger's progress in an engine state (``/triggers/state``), across the registrations so far."""
        assert self._trigger_body is not None
        return trigger_progress(self._trigger_body["triggers"], self._trigger_plan, snap.get("triggers", []), self._clock_t0())

    async def _restore_triggers(self, snap: dict[str, Any], progress: dict[str, dict[str, Any]] | None = None) -> None:
        """Triggers as they stood in the snapshot (``plan_restore``): what fired for the last time stays fired, what waits
        on a mark fires at that mark, a recurrence continues its count and its seeded stream. agentenv has no import API,
        so the set is re-registered. ``progress``: the snapshot's own (``snapshot_state``); a snapshot taken before it was
        recorded is read as a registration of the original body."""
        assert self._trigger_body is not None
        if progress is None:
            progress = trigger_progress(self._trigger_body["triggers"], {}, snap.get("triggers", []), self._clock_t0())
        keep, plan = plan_restore(self._trigger_body["triggers"], progress, self._clock_t0())
        (await self.http.post("/triggers/clear")).raise_for_status()
        self._trigger_plan = plan
        if keep:
            r = await self.http.post("/triggers/register", json={**self._trigger_body, "triggers": keep})
            if r.status_code != 200:
                raise RuntimeError(f"trigger restore failed: {r.text[:500]}")

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

    async def _settled_trigger_state(self, settle_s: float = 15.0) -> dict[str, Any] | None:
        """/triggers/state once no trigger is mid-fire (agentenv prompt_agent._fetch_settled_trigger_state): a snapshot
        taken while an action is queued would restore a world in which it neither happened nor will happen."""
        deadline = time.monotonic() + settle_s
        while True:
            r = await self.http.get("/triggers/state")
            if r.status_code != 200:
                return None
            st = r.json()
            busy = [t for t in st.get("triggers", []) if (t.get("status") in ("firing", "queued") or t.get("pending"))
                    and not ((t.get("when") or {}).get("every") and not (t.get("when") or {}).get("count"))]  # unbounded recurrences never settle
            if not busy or time.monotonic() > deadline:
                return st
            await asyncio.sleep(0.25)

    async def snapshot_state(self) -> dict[str, Any]:
        st: dict[str, Any] = {}
        if getattr(self.sandbox, "image_id", None):
            st["image"] = self.sandbox.image_id
        r = await self.http.get("/clock/state")
        if r.status_code == 200:
            st["clock"] = r.json()
        trig = await self._settled_trigger_state()
        if trig is not None:
            st["triggers"] = trig
            if self._trigger_body is not None:  # harness-reward audit §5: what a restore continues from, whatever is registered now
                st["trigger_progress"] = self.trigger_progress(trig)
        for a in self._allowed:
            st.setdefault("allowed", {})[a] = sorted(self._allowed[a])
        return st

    async def restore_state(self, state: dict[str, Any]) -> None:
        world = self._world
        if world is None or not state:
            return
        from crucible.world.sandbox import check_image

        check_image(state, getattr(self.sandbox, "image_id", None))
        clock = state.get("clock") or {}
        if clock.get("armed") and clock.get("virtual_time"):
            await self._set_clock(clock["virtual_time"], clock.get("virtual_seconds_per_real_second", 1))
        if self._trigger_body is not None and state.get("triggers") is not None:
            await self._restore_triggers(state["triggers"], state.get("trigger_progress"))
        # Role rules: restore exactly the allowed tool sets captured with the snapshot.
        for actor, names in (state.get("allowed") or {}).items():
            await self._toggle("disable", actor, "*")
            await self._toggle("enable", actor, list(names))
        await self._refresh_allowed(world)
        self._acl_changed.clear()

    async def teardown(self) -> None:
        """Each stage contained, the sandbox closed last whatever failed before it (H2 V16: one failing stop skipped a paid
        sandbox's close), and the gateway killed if SIGTERM does not stop it."""
        try:
            self._keepalive_stop.set()
            if self._keepalive is not None:
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._keepalive, 5)
            if self.gateway is not None and self.gateway.returncode is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(self.gateway.pid, 15)
                try:
                    await asyncio.wait_for(self.gateway.wait(), 5)
                except Exception:  # noqa: BLE001
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(self.gateway.pid, 9)
            for stop in [*(p.stop for p in self.proxies.values()), *([self.relay.stop] if self.relay is not None else []), *(s.stop for s in self.servers.values()),
                         self.http.aclose]:
                with contextlib.suppress(Exception):
                    await stop()
        finally:
            if self.sandbox is not None:
                try:
                    await self.sandbox.close()
                finally:
                    self.sandbox = None
            if not self.params.get("keep_workdir"):
                import shutil

                shutil.rmtree(self.apps_dir, ignore_errors=True)


class AgentProxy:
    """An outside agent's only door into the gateway: ``/mcp`` (streamed both ways) and REST ``/step`` limited to
    ``call_tool`` / ``list_tools``, with ``AgentEnv-Role`` forced to one actor. The control plane (``/tools/*``,
    ``/clock/*``, ``/triggers/*``, ``/trajectory``, ``/state``) is not reachable through it."""

    def __init__(self, upstream: str, actor: str, *, host: str = "127.0.0.1"):
        self.upstream, self.actor, self.host, self.port = upstream.rstrip("/"), actor, host, free_port()
        self._server: Any = None
        self._task: asyncio.Task | None = None
        self._client: httpx.AsyncClient | None = None

    @property
    def url(self) -> str:
        return f"http://{'127.0.0.1' if self.host in ('0.0.0.0', '') else self.host}:{self.port}"

    @property
    def mcp_url(self) -> str:
        return f"{self.url}/mcp"

    def app(self):
        from starlette.applications import Starlette
        from starlette.background import BackgroundTask
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route

        drop = {"host", "content-length", ROLE_HEADER.lower(), "connection", "transfer-encoding"}

        async def forward(request):
            assert self._client is not None
            body = await request.body()
            if request.url.path == "/step":
                try:
                    action = json.loads(body or b"{}").get("action")
                except (ValueError, AttributeError):
                    action = None
                if action not in ("call_tool", "list_tools"):
                    return JSONResponse({"error": "only call_tool and list_tools are served to agents"}, status_code=403)
            headers = {k: v for k, v in request.headers.items() if k.lower() not in drop}
            headers[ROLE_HEADER] = self.actor
            url = self.upstream + request.url.path + (f"?{request.url.query}" if request.url.query else "")
            r = await self._client.send(self._client.build_request(request.method, url, headers=headers, content=body), stream=True)
            out = {k: v for k, v in r.headers.items() if k.lower() not in ("connection", "transfer-encoding")}
            return StreamingResponse(r.aiter_raw(), status_code=r.status_code, headers=out, background=BackgroundTask(r.aclose))

        async def refuse(request):
            return JSONResponse({"error": "not served to agents"}, status_code=404)

        from agent_env.env.gateway.constants import WELL_KNOWN_PATH

        return Starlette(routes=[Route("/mcp", forward, methods=["GET", "POST", "DELETE"]), Route("/mcp/{rest:path}", forward, methods=["GET", "POST", "DELETE"]),
                                 Route("/step", forward, methods=["POST"]), Route(WELL_KNOWN_PATH, forward, methods=["GET"]),
                                 Route("/{rest:path}", refuse, methods=["GET", "POST", "PUT", "DELETE"])])

    async def start(self) -> None:
        import uvicorn

        self._client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        self._server = uvicorn.Server(uvicorn.Config(self.app(), host=self.host, port=self.port, log_level="warning", lifespan="off"))
        self._task = asyncio.create_task(self._server.serve())
        for _ in range(400):
            if self._server.started:
                return
            if self._task.done():
                raise RuntimeError(f"agent proxy failed: {self._task.exception()}")
            await asyncio.sleep(0.025)
        raise TimeoutError("agent proxy did not start")

    async def stop(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._task is not None:
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._task, 5)
        if self._client is not None:
            await self._client.aclose()
