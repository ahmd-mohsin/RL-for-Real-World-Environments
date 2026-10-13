"""Any OpenEnv ``Environment`` as a Crucible World (import; research m11-m12/05 §3.5, N1–N4). DEPENDs on OpenEnv's own
``Environment`` / MCP action and observation types; runs where OpenEnv is installed (``crucible/.envs/openenv``), so
OpenEnv is imported lazily.

The export side (``export.py``) turns a Crucible task into an OpenEnv Environment; this is the reverse, so every
environment of the OpenEnv / FineEnvs catalogue (simulations, games, planners, MCP services) gets Crucible's contract:
snapshot / restore / verifier view, the single reward producer, the dense channel, the M10 runtime and the M12 trainer.

**App** (template ``openenv/env``), params:

- ``factory`` ``"module:attr"`` (a class or callable returning the Environment) + ``factory_kwargs``;
- ``reset``: ``reset()`` kwargs (task selection: ``seed`` / ``split`` / ``index`` / ``task_id``; ``episode_id`` is pinned to
  ``"crucible"`` unless given, so environments that put the id in observations stay deterministic);
- MCP environments (``MCPEnvironment``): the tools come from ``ListToolsAction``; ``allow_tools`` (fnmatch) narrows them;
  each call is a ``CallToolAction``; MCP ``ImageContent`` parts become ``ToolResult.images`` (stored, digested). Other environments: one tool ``step`` whose schema is ``action_cls``'s JSON schema;
- ``show_metadata``: also show the observation's ``metadata`` to the policy (off by default: environments put grades
  and hidden fields there).
- ``reset.seed = "world"``: the App's derived seed (``derive_seed(WorldSpec.seed, app id)``);
- ``clock_from``: a dotted metadata path holding the simulation's time; with a ``sim`` clock the World follows it, so
  time-triggered World events (disruptions, deadlines) fire on simulation time.

**State = the replay journal** (``json_doc`` adapter over ``path``): reset kwargs, every call's arguments, the digest
of its result (and the text, capped), its reward and ``done``. OpenEnv environments are opaque Python objects with no snapshot of their own;
an Environment is deterministic given reset + actions (OpenEnv's own contract for reproducible ``reset(seed)``), so
``restore`` = rebuild + reset + re-issue the journal, and **verify** every re-issued result's digest: a mismatch is a
non-deterministic environment and raises ``SnapshotMismatch`` (never silently a different state).

**Termination**: an observation with ``done`` ends the episode (``ToolResult.ends_episode``: the World marks the actor
done). OpenEnv environments end themselves (a final answer, a step budget); Crucible Worlds otherwise end on the
policy's ``Done``.

**Multi-agent** (two-player games: connect4, chess, textarena; m13 E pick 8): ``players`` (seat order) makes every seat a
policy actor of the World. ``turn_from`` (a dotted metadata path, default ``next_player``) and ``turn_map`` ({value:
actor}) say whose move it is after each observation; a call from any other seat is refused (``not_your_turn``, a policy
fault, never executed or journaled), and the journal records each step's mover. A terminal observation ends every seat
(``ToolResult.ends_for``). ``OpenEnvTask.actor_rewards`` attributes the outcome per seat: the terminal step's reward to its
mover and, with ``zero_sum`` (default), its negation to the others.

**Reward** (``OpenEnvTask.grade``): one score gate over the journal. The environment's reward (its rubric's value on
the terminal step, ``reward_mode="last"``, or the sum of step rewards, ``"sum"``) normalized by ``reward_range``. An
episode that never terminated (the policy stopped without a final answer) scores 0: a policy outcome, never a mask
(N4). An environment exception inside ``step`` reaches the World's choke point and masks the episode (tool crashed).
"""

from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import importlib
import json
import math
import posixpath
import shlex
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.world.apps.base import App
from crucible.world.state.base import SnapshotMismatch

STEP_TOOL = "step"
TEXT_CAP = 16384  # result text kept in the journal for progress checks (the digest covers the full text)


def load(ref: str) -> Any:
    mod, _, attr = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split(".") if attr else []:
        obj = getattr(obj, part)
    return obj


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def result_text(result: Any) -> str:
    """An MCP ``CallToolResult`` (fastmcp / mcp) or any value, as the text the policy reads."""
    if result is None:
        return ""
    if isinstance(result, str):
        return result
    content = getattr(result, "content", None)
    if isinstance(content, list):
        parts = [getattr(c, "text", None) for c in content]
        if any(p is not None for p in parts):
            return "\n".join(p for p in parts if p is not None)
    if isinstance(result, dict) and isinstance(result.get("content"), list):
        return "\n".join(c.get("text", "") for c in result["content"] if isinstance(c, dict))
    data = getattr(result, "structured_content", None) or getattr(result, "data", None)
    if data is not None:
        return json.dumps(data, ensure_ascii=False, sort_keys=True, default=str)
    if hasattr(result, "model_dump"):
        return json.dumps(result.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
    return json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)


def result_images(result: Any) -> list[tuple[bytes, str]]:
    """MCP ``ImageContent`` parts of a tool result, as (bytes, mime)."""
    import base64

    out = []
    for c in getattr(result, "content", None) or []:
        if getattr(c, "type", None) == "image" and getattr(c, "data", None):
            out.append((base64.b64decode(c.data), getattr(c, "mimeType", None) or "image/png"))
    return out


def _accepted(fn: Any, kw: dict) -> dict:
    """The keyword arguments ``fn``'s own signature accepts (many OpenEnv environments declare a bare ``reset()``)."""
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return kw
    if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return kw
    return {k: v for k, v in kw.items() if k in params}


async def _call(env: Any, name: str, *args: Any, **kw: Any) -> Any:
    """``reset`` / ``step`` on an OpenEnv Environment. The base class's ``*_async`` forwards ``seed`` / ``episode_id`` /
    ``timeout_s`` to the sync method, which breaks every environment whose ``reset()`` / ``step(action)`` declares fewer
    parameters (grid_world, connect4, wildfire); so the async method is used only when the class overrides it, and only
    the keyword arguments the target's signature accepts are passed."""
    try:
        from openenv.core.env_server.interfaces import Environment

        base_async = getattr(Environment, f"{name}_async", None)
    except ImportError:
        base_async = None
    a = getattr(type(env), f"{name}_async", None)
    if a is not None and a is not base_async:
        return await getattr(env, f"{name}_async")(*args, **_accepted(getattr(env, f"{name}_async"), kw))
    fn = getattr(env, name)
    return fn(*args, **_accepted(fn, kw))


def _decode_image(v: str) -> tuple[bytes, str]:
    import base64

    if v.startswith("data:"):
        head, _, data = v.partition(",")
        return base64.b64decode(data), head[5:].split(";", 1)[0] or "image/png"
    return base64.b64decode(v), "image/png"


def _is_error(result: Any) -> bool:
    return bool(getattr(result, "is_error", False) or getattr(result, "isError", False) or (isinstance(result, dict) and result.get("isError")))


class _ObsView:
    """An OpenEnv client ``StepResult`` (observation dict + reward + done) seen through the attributes the App reads from
    in-process observations (``reward``, ``done``, ``metadata``, ``model_dump``, and any observation key)."""

    def __init__(self, obs: dict, reward: Any, done: bool):
        self._obs, self.reward, self.done = dict(obs or {}), reward, bool(done)
        self.metadata = self._obs.get("metadata") or {}

    def model_dump(self, mode: str = "python") -> dict:
        return {**self._obs, "reward": self.reward, "done": self.done}

    def __getattr__(self, name: str) -> Any:
        try:
            return self.__dict__["_obs"][name]
        except KeyError:
            raise AttributeError(name) from None


class ClientEnv:
    """A remote OpenEnv environment (its own server, a Docker image, a Hugging Face Space) behind the in-process
    Environment surface, over Crucible's native WebSocket client (``world/remote.py``, OpenEnv's ``/ws`` protocol; no
    OpenEnv import on this side). ``reset`` / ``step`` return observation views; actions are dicts. ``docker_image``
    starts the image with its port published on 127.0.0.1 and waits for ``/health``."""

    def __init__(self, base_url: str | None = None, *, docker_image: str | None = None, environment: dict | None = None,
                 session_id: str | None = None, port: int = 8000, message_timeout_s: float = 300.0):
        from crucible.world.remote import RemoteEnvClient

        self.container: str | None = None
        if docker_image:
            base_url = self._start_container(docker_image, environment or {}, port)
        if not base_url:
            raise ValueError("client mode needs base_url or docker_image")
        self.base_url = base_url
        self.client = RemoteEnvClient(base_url, session_id=session_id, message_timeout_s=message_timeout_s)

    def _start_container(self, image: str, env: dict, port: int) -> str:
        import subprocess
        import time
        import urllib.request
        import uuid

        name = f"crucible-openenv-{uuid.uuid4().hex[:8]}"
        argv = ["docker", "run", "-d", "--rm", "--name", name, "-p", f"127.0.0.1::{port}", *[x for k, v in env.items() for x in ("-e", f"{k}={v}")], image]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=600, check=False)
        if r.returncode != 0:
            raise RuntimeError(f"docker run {image} failed: {r.stderr[-500:]}")
        self.container = name
        out = subprocess.run(["docker", "port", name, f"{port}/tcp"], capture_output=True, text=True, timeout=60, check=False).stdout.split()
        url = f"http://{out[0]}" if out else None
        deadline = time.monotonic() + 300
        while url and time.monotonic() < deadline:
            try:
                with urllib.request.urlopen(f"{url}/health", timeout=2) as h:
                    if h.status == 200:
                        return url
            except OSError:
                time.sleep(1)
        self._stop_container()
        raise RuntimeError(f"{image} did not become healthy on {url}")

    def _stop_container(self) -> None:
        import subprocess

        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=120, check=False)
            self.container = None

    async def reset_async(self, **kw: Any) -> _ObsView:
        r = await self.client.reset(**kw)
        return _ObsView(r.get("observation") or {}, r.get("reward"), bool(r.get("done")))

    async def step_async(self, action: dict, **kw: Any) -> _ObsView:
        r = await self.client.step(action)
        return _ObsView(r.get("observation") or {}, r.get("reward"), bool(r.get("done")))

    async def aclose(self) -> None:
        await self.client.close()
        self._stop_container()

    def close(self) -> None:
        import asyncio as _a

        try:
            loop = _a.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            task = loop.create_task(self.aclose())
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        else:
            _a.run(self.aclose())


def _err_type(err: Any) -> str:
    """An MCP call's error type: in-process an object with an enum, over the wire a dict."""
    t = err.get("error_type") if isinstance(err, dict) else getattr(err, "error_type", None)
    return str(getattr(t, "value", t) or "")


class OpenEnvApp(App):
    template = "openenv/env"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.env: Any = None
        self.mcp = False
        self._tools: list[ToolSpec] = []
        self.journal: dict = {}
        self.reset_observation: Any = None
        self.turn: str | None = None  # multi-agent: the seat whose move it is

    # ---- lifecycle --------------------------------------------------------------------------------------------
    def _reset_kwargs(self) -> dict:
        kw = dict(self.spec.params.get("reset") or {})
        kw.setdefault("episode_id", "crucible")
        if kw.get("seed") == "world":  # this App's derived seed (WorldSpec.seed → derive_seed(seed, app id))
            kw["seed"] = self.ctx.seed
        return kw

    def _sim_time(self, obs: Any) -> None:
        """``clock_from``: a dotted path into the observation's metadata holding the simulation's time (seconds since the
        World clock's t0); the World's sim clock follows it, so time-triggered events fire on simulation time."""
        path = self.spec.params.get("clock_from")
        clock = self.ctx.clock
        if not path or clock is None or clock.mode != "sim":
            return
        node: Any = getattr(obs, "metadata", None) or {}
        for part in path.split("."):
            if not isinstance(node, dict) or part not in node:
                return
            node = node[part]
        if isinstance(node, (int, float)) and clock.t0.timestamp() + float(node) > clock.now().timestamp():
            clock.advance_sim_seconds(float(node))

    async def _fresh(self) -> Any:
        """A new Environment, reset; returns the reset observation."""
        if self.env is not None:
            self._close_env()
        p = self.spec.params
        if p.get("client"):  # client mode (M1 item 9): the environment's own server / Docker image / Space, over a WebSocket
            c = dict(p["client"])
            self.env = ClientEnv(c.get("base_url"), docker_image=c.get("docker_image"), environment=c.get("environment"),
                                 session_id=c.get("session_id"), port=int(c.get("port", 8000)), message_timeout_s=float(c.get("message_timeout_s", 300)))
            self.mcp = bool(c.get("mcp"))  # an MCPEnvironment server: tools come from list_tools, calls are call_tool steps
        else:
            self.env = load(p["factory"])(**(p.get("factory_kwargs") or {}))
            from openenv.core.env_server.mcp_environment import MCPEnvironment

            self.mcp = isinstance(self.env, MCPEnvironment)
        obs = await _call(self.env, "reset", **self._reset_kwargs())
        self.reset_observation = obs
        self._update_turn(obs)
        return obs

    def _players(self) -> list[str]:
        return list(self.spec.params.get("players") or [])

    def _update_turn(self, obs: Any) -> None:
        players = self._players()
        if not players:
            return
        node: Any = getattr(obs, "metadata", None) or {}
        for part in str(self.spec.params.get("turn_from") or "next_player").split("."):
            node = node.get(part) if isinstance(node, dict) else None
        tmap = {str(k): v for k, v in (self.spec.params.get("turn_map") or {}).items()} or {"1": players[0], "-1": players[-1]}
        if node is not None and str(node) in tmap:
            self.turn = tmap[str(node)]

    async def start(self) -> None:
        obs = await self._fresh()
        self._tools = await self._discover()
        self.journal = {"reset": self._reset_kwargs(), "reset_digest": _digest(self._obs_text(obs, reset=True)), "steps": [], "done": False,
                        "final_reward": None, "rewards": []}
        await self._write()

    async def stop(self) -> None:
        self._close_env()

    def _close_env(self) -> None:
        if self.env is not None and hasattr(self.env, "close"):
            with contextlib.suppress(Exception):  # closing a finished environment is best effort
                self.env.close()
        self.env = None

    async def _discover(self) -> list[ToolSpec]:
        p = self.spec.params
        if self.mcp and isinstance(self.env, ClientEnv):
            obs = await self._step({"type": "list_tools"})
            tools = [ToolSpec(app=self.id, name=t["name"], description=t.get("description") or "",
                              input_schema=t.get("input_schema") or t.get("inputSchema") or {"type": "object"}) for t in obs.tools]
        elif self.mcp:
            from openenv.core.env_server.mcp_types import ListToolsAction

            obs = await self._step(ListToolsAction())
            tools = [ToolSpec(app=self.id, name=t.name, description=t.description or "", input_schema=t.input_schema or {"type": "object"}) for t in obs.tools]
        elif p.get("client") and not p.get("action_cls"):  # client mode: the action schema is declared (the client sends dicts)
            tools = [ToolSpec(app=self.id, name=STEP_TOOL, description=p.get("step_description") or "Take one action.",
                              input_schema=p.get("action_schema") or {"type": "object"})]
        else:
            if not p.get("action_cls"):
                raise ValueError(f"{self.id}: a non-MCP OpenEnv environment needs params.action_cls (its Action class)")
            schema = load(p["action_cls"]).model_json_schema()
            schema.get("properties", {}).pop("metadata", None)
            tools = [ToolSpec(app=self.id, name=STEP_TOOL, description=p.get("step_description") or (load(p["action_cls"]).__doc__ or "Take one action.").strip(),
                              input_schema=schema)]
        allow = p.get("allow_tools")
        return [t for t in tools if not allow or any(fnmatch.fnmatchcase(t.name, pat) for pat in allow)]

    async def _step(self, action: Any) -> Any:
        return await _call(self.env, "step", action)

    async def _write(self) -> None:
        path = self.state_params["path"]
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(posixpath.dirname(path))}")
        await self.ctx.exec.write_text(path, json.dumps(self.journal, sort_keys=True, separators=(",", ":"), ensure_ascii=False))

    async def after_restore(self) -> None:
        """Rebuild the live Environment to the journal the adapter just restored: reset, then re-issue every call,
        verifying each result digest (a deterministic environment reproduces them exactly)."""
        await self._replay(json.loads(await self.ctx.exec.read_text(self.state_params["path"])))

    async def _replay(self, doc: dict) -> None:
        obs = await self._fresh()
        if _digest(self._obs_text(obs, reset=True)) != doc["reset_digest"]:
            raise SnapshotMismatch(f"{self.id}: the environment's reset observation changed (non-deterministic reset)")
        for i, s in enumerate(doc["steps"]):
            obs, text, _, _ = await self._issue(s["tool"], s["args"])
            self._update_turn(obs)
            if self._step_digest(obs, text) != s["digest"]:
                raise SnapshotMismatch(f"{self.id}: replayed step {i} ({s['tool']}) gave a different result: the environment is not deterministic")
        self.journal = doc

    # ---- tools ------------------------------------------------------------------------------------------------
    async def list_tools(self) -> list[ToolSpec]:
        return self._tools

    async def _healthy(self) -> bool:
        """Rebuild the environment to the journal (every digest verified): True when it replays exactly."""
        try:
            await self._replay(json.loads(json.dumps(self.journal)))
            return True
        except Exception:  # noqa: BLE001 -- any failure to reproduce the journal: the environment is not healthy
            return False

    def _obs_text(self, obs: Any, *, reset: bool = False) -> str:
        if not reset and hasattr(obs, "tool_name"):  # CallToolObservation
            if obs.error is not None:
                msg = obs.error.get("message") if isinstance(obs.error, dict) else obs.error.message
                return f"{_err_type(obs.error)}: {msg}"
            return result_text(obs.result)
        d = obs.model_dump(mode="json") if hasattr(obs, "model_dump") else dict(obs)
        for k in ("reward", "done"):
            d.pop(k, None)
        for f in self.spec.params.get("image_fields") or []:  # pixels go to ToolResult.images, never into the text
            if f in d:
                d[f] = f"<image:{f}>"
        if not (self.spec.params.get("show_metadata") or reset):
            d.pop("metadata", None)
        return json.dumps(d, ensure_ascii=False, sort_keys=True, default=str)

    async def _issue(self, tool: str, args: dict) -> tuple[Any, str, float | None, bool]:
        if self.mcp and isinstance(self.env, ClientEnv):
            obs = await self._step({"type": "call_tool", "tool_name": tool, "arguments": dict(args)})
        elif self.mcp:
            from openenv.core.env_server.mcp_types import CallToolAction

            obs = await self._step(CallToolAction(tool_name=tool, arguments=dict(args)))
        elif isinstance(self.env, ClientEnv):
            obs = await self._step(dict(args))
        else:
            obs = await self._step(load(self.spec.params["action_cls"]).model_validate(args))
        r = getattr(obs, "reward", None)
        return obs, self._obs_text(obs), (float(r) if r is not None else None), bool(getattr(obs, "done", False))

    def _images(self, obs: Any) -> list[tuple[bytes, str]]:
        if hasattr(obs, "tool_name"):
            return result_images(obs.result) if obs.error is None else []
        out = []  # ``image_fields``: observation keys holding images (base64 or data URLs; pixels of non-MCP environments)
        for f in self.spec.params.get("image_fields") or []:
            v = (obs.model_dump(mode="json") if hasattr(obs, "model_dump") else dict(obs)).get(f)
            for item in v if isinstance(v, list) else [v]:
                if isinstance(item, str) and item:
                    out.append(_decode_image(item))
        return out

    def _step_digest(self, obs: Any, text: str) -> str:
        """The step's whole observation: the policy's text, images, and the hidden parts a restore must reproduce too
        (``metadata``, ``reward``, ``done``; M11 audit H3 V1: a restore that drew a different hidden card was accepted)."""
        hidden = {"metadata": getattr(obs, "metadata", None), "reward": getattr(obs, "reward", None), "done": bool(getattr(obs, "done", False))}
        return _digest(text + "".join(f"|image:{hashlib.sha256(b).hexdigest()}" for b, _ in self._images(obs))
                       + "|" + json.dumps(hidden, sort_keys=True, default=str, ensure_ascii=False))

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        q = f"{self.id}.{name}"
        if name not in {t.name for t in self._tools}:
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        ends_for = self._players() or None
        if self.journal.get("done"):
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content="The episode is over.", error="episode_over", fault="policy",
                              ends_episode=True, ends_for=ends_for)
        if ends_for and self.turn is not None and actor != self.turn:  # not this seat's move: refused, never executed
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"It is not your turn ({self.turn} to move).", error="not_your_turn",
                              fault="policy")
        from crucible.world.remote import RemoteActionError, RemoteEnvError

        try:
            obs, text, reward, done = await self._issue(name, args)
        except RemoteActionError as e:  # the server rejected the action itself: the policy's malformed call, not journaled
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"Invalid action: {e}", error="invalid_action", fault="policy")
        except RemoteEnvError as e:
            if e.code != "CONNECTION_LOST":
                raise  # the environment failed: the World's gate records an infra fault and masks the episode
            # The socket died, and with it the server session. Rebuild the state in a new session by replaying the
            # journal (every digest verified), then issue the call once more.
            await self._replay(json.loads(json.dumps(self.journal)))
            obs, text, reward, done = await self._issue(name, args)
        except Exception as e:
            # OpenEnv environments raise on actions they reject (Atari's out-of-range action_id; H3 V2: one bad action
            # masked the policy's own episode). Rebuild the environment from the journal: if every digest replays, the
            # environment is healthy and the action was the policy's error (not journaled); if not, it is infra.
            if not await self._healthy():
                raise
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"Invalid action: {type(e).__name__}: {e}"[:2000],
                              error="invalid_action", fault="policy")
        self._update_turn(obs)
        self._sim_time(obs)
        err = getattr(obs, "error", None)
        images = [self.image(b, mime) for b, mime in self._images(obs)]
        self.journal["steps"].append({"tool": name, "args": args, "actor": actor, "digest": self._step_digest(obs, text), "text": text[:TEXT_CAP], "reward": reward, "done": done,
                                      **({"images": [i.ref for i in images]} if images else {})})
        self.journal["rewards"].append(reward)
        if done:
            self.journal["done"] = True
            self.journal["final_reward"] = reward
            self.journal["final_metadata"] = json.loads(json.dumps(getattr(obs, "metadata", None) or {}, default=str))
        await self._write()
        if err is not None:
            et = _err_type(err)
            fault = {"invalid_args": "policy", "tool_not_found": "policy", "execution_error": "tool"}.get(et, "infra")
            if et == "timeout" and await self._healthy():  # a call the policy made slow (OpenEnv's MCP deadline), on a healthy env
                fault = "policy"
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=text, error=et or "tool_error", fault=fault, ends_episode=done,
                              ends_for=ends_for)
        if self.mcp and _is_error(obs.result):
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=text, error="tool_error", fault="tool", ends_episode=done, ends_for=ends_for)
        return ToolResult(tool_call_id=call_id, function_name=q, ok=True, content=text, data={"reward": reward, "done": done}, ends_episode=done, images=images,
                          ends_for=ends_for)


# ---- the RunTask ----------------------------------------------------------------------------------------------
class OpenEnvTask:
    """An M10 ``RunTask`` over one OpenEnv environment configuration (one task of its catalogue).

    ``instruction``: the policy's instruction; a callable gets the reset observation (e.g. the environment's rules in
    ``metadata["instructions"]``). ``progress``: extra M7 checks over the journal for the dense channel (each
    ``fn(journal) -> bool``); the defaults are ``terminated`` and ``positive_reward``. ``tool_quota`` / ``clock`` /
    ``seed`` / ``events`` go to the WorldSpec: Crucible's own budgets, sim clock and timed events layered over the
    imported environment.
    """

    def __init__(self, id: str, factory: str, *, reset: dict | None = None, factory_kwargs: dict | None = None, app: str = "env",
                 instruction: str | Callable[[Any], str] = "", action_cls: str | None = None, allow_tools: Sequence[str] | None = None,
                 show_metadata: bool = False, reward_range: tuple[float, float] = (0.0, 1.0), reward_mode: str = "last",
                 progress: dict[str, Callable[[dict], bool]] | None = None, tool_timeout_s: float = 120.0, tool_quota: dict[str, int] | None = None,
                 clock: dict | None = None, clock_from: str | None = None, seed: int = 0, events: Sequence[dict] | None = None,
                 players: Sequence[str] | None = None, turn_from: str | None = None, turn_map: dict | None = None, zero_sum: bool = True,
                 client: dict | None = None, action_schema: dict | None = None, image_fields: Sequence[str] | None = None):
        if reward_mode not in ("last", "sum"):
            raise ValueError("reward_mode: 'last' | 'sum'")
        self.id, self.world_id, self.app = id, f"openenv/{id.split('/')[1] if '/' in id else id}", app
        self.params = {"factory": factory, "factory_kwargs": factory_kwargs or {}, "reset": reset or {}, "show_metadata": show_metadata}
        if action_cls:
            self.params["action_cls"] = action_cls
        if allow_tools:
            self.params["allow_tools"] = list(allow_tools)
        self.instruction, self.reward_range, self.reward_mode = instruction, reward_range, reward_mode
        if clock_from:
            self.params["clock_from"] = clock_from
        if client:  # client mode: {"base_url": ...} or {"docker_image": ..., "environment": {...}}
            self.params["client"] = dict(client)
        if action_schema:
            self.params["action_schema"] = action_schema
        if image_fields:
            self.params["image_fields"] = list(image_fields)
        self.players, self.zero_sum = list(players or []), zero_sum
        if self.players:
            self.params["players"] = self.players
            if turn_from:
                self.params["turn_from"] = turn_from
            if turn_map:
                self.params["turn_map"] = {str(k): v for k, v in turn_map.items()}
        self.progress = progress or {}
        self.tool_timeout_s, self.tool_quota, self.clock, self.seed = tool_timeout_s, dict(tool_quota or {}), clock, seed
        self.events = list(events or [])  # World events over the imported environment (disruptions, deadlines, messages)

    def spec(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec

        extra = {"clock": self.clock} if self.clock else {}
        if self.events:
            extra["events"] = self.events
        return WorldSpec.model_validate({"world": self.world_id, "seed": self.seed, **extra, "limits": {"tool_timeout_s": self.tool_timeout_s, "tool_quota": self.tool_quota or None},
                                         "apps": [{"id": self.app, "template": "openenv/env",
                                                   "state": {"adapter": "json_doc", "params": {"path": str(Path(workdir).resolve() / "openenv" / "journal.json")}},
                                                   "params": self.params}],
                                         "actors": [{"id": a, "kind": "policy", "tools": [f"{self.app}.*"]} for a in (self.players or ["agent"])]})

    def actor_rewards(self, j: dict) -> dict[str, float]:
        """Per-seat outcome: the terminal step's reward to its mover, its negation to the other seats (``zero_sum``), or
        each seat's own step rewards summed (not zero-sum). Seats get 0 while the game is unfinished."""
        players = self.players or ["agent"]
        out = {a: 0.0 for a in players}
        steps = j.get("steps") or []
        if not j.get("done") or not steps:
            return out
        if self.zero_sum:
            mover, r = steps[-1].get("actor") or players[0], float(j.get("final_reward") or 0.0)
            return {a: (r if a == mover else -r) for a in players}
        for s_ in steps:
            if s_.get("actor") in out:
                out[s_["actor"]] += float(s_.get("reward") or 0.0)
        return out

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec(workdir), store=Store(workdir / "s"), workdir=workdir / "w")
        await w.start()
        app = w.apps[self.app]
        text = self.instruction(app.reset_observation) if callable(self.instruction) else self.instruction
        await w.reset(instructions={a: (text if isinstance(text, str) else text[a]) for a in (self.players or ["agent"])})
        return w

    # ---- grading ----------------------------------------------------------------------------------------------
    @staticmethod
    async def journal(view, app: str) -> dict:
        r = await view.query(app, {"op": "doc", "args": {}})
        if not r.ok:
            raise RuntimeError(r.error)
        return r.value or {}

    def episode_reward(self, j: dict) -> float | None:
        if not j.get("done"):
            return None
        if self.reward_mode == "last":
            return j.get("final_reward")
        rs = j.get("rewards", [])
        return None if any(r is not None and not math.isfinite(r) for r in rs) else sum(r or 0.0 for r in rs)

    def plan(self):
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        lo, hi = self.reward_range

        async def env_reward(ctx) -> Verdict:
            from crucible.verify.runtime.trusted import InfraError

            j = await self.journal(ctx.params["view"], self.app)
            if not j.get("done"):
                return Verdict(False, {"reason": "the episode never terminated (no final answer): scored 0", "steps": len(j.get("steps", []))})
            r = self.episode_reward(j)
            # a terminal step without a reward, or a non-finite one, is the environment's bug: masked (H3 V3: NaN scored 0,
            # inf 1, None "never terminated")
            if r is None or not math.isfinite(r):
                raise InfraError(f"crucible:invalid_reward: the environment ended the episode with reward {r!r}")
            norm = min(1.0, max(0.0, (r - lo) / (hi - lo))) if hi > lo else float(r > lo)
            # ``success``: full credit only (G5: every terminated episode was an outcome pass, so a 0 read as solved)
            return Verdict(True, {"score": r, "normalized": norm, "normalized_reward": norm, "success": norm >= 1.0, "steps": len(j.get("steps", [])),
                                  **({"actor_rewards": self.actor_rewards(j)} if self.players else {})})

        return Plan(self.id, [Gate("env_reward", kind="score", description="the OpenEnv environment's own reward")],
                    [Check("env_reward", "env_reward", env_reward, tier="fast", provenance="openenv:rubric")])

    async def grade(self, world, workdir: Path):
        from crucible.verify.runtime.plan import Context
        from crucible.verify.runtime.trusted import Trusted

        return await self.plan().arun(Context(Trusted(Path(workdir)), params={"view": world.verifier_view()}))

    def plan_for_progress(self):
        """Dense-progress subgoals over the journal (``interop.replay.PlanSubgoals``)."""
        from crucible.verify.runtime.plan import Check, Gate, Plan, Verdict

        goals: dict[str, Callable[[dict], bool]] = {"terminated": lambda j: bool(j.get("done")),
                                                    "positive_reward": lambda j: any((r or 0.0) > 0 for r in j.get("rewards", [])),
                                                    **self.progress}

        def check(fn):
            async def run(ctx) -> Verdict:
                return Verdict(bool(fn(await self.journal(ctx.params["view"], self.app))))

            return run

        return Plan(self.id, [Gate("progress", kind="outcome")], [Check(k, "progress", check(fn)) for k, fn in goals.items()])
