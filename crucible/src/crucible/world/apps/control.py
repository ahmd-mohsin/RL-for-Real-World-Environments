"""Control App: a simulator or control environment as a World App (M1 item 10; Gymnasium's ``reset``/``step`` API as
dm_control, MuJoCo, Unity ML-Agents, FinRL and the classic-control suite expose it through OpenEnv and Gymnasium; CARLA's
``ticks_after_tool`` for language-model policies).

**Source** (``params``): ``env: "gymnasium:<id>"`` (``gymnasium.make``) with ``env_kwargs``, or ``factory: "module:attr"``
returning an object with ``reset(seed=) -> (obs, info)``, ``step(action) -> (obs, reward, terminated, truncated, info)``,
``action_space`` and ``observation_space`` (Gymnasium spaces, or ``Space`` data in ``params.action_space`` /
``params.observation_space``), or ``client: {base_url, action_key, observation_key, done_is, reset_kwargs}``: a simulator served
elsewhere over WebSocket (``RemoteControlEnv``; OpenEnv's dm_control, Unity, FinRL servers), spaces from its state.

**Time belongs to the simulator.** One agent action holds for ``frame_skip`` ticks (rewards summed; the episode can end
mid-skip); ``sim_dt_s`` seconds of simulation pass per tick (default: the environment's ``dt``), and a World in ``sim``
clock mode follows simulation time, so time-triggered events fire on it. ``max_episode_ticks`` truncates.

**Two kinds of policy, one environment.**

- A neural policy (``world/gym.py``'s ``WorldEnv``, M1 item 10c) calls ``step`` with raw actions every decision.
- A language model gets the same ``step`` tool, its schema derived from the action space, plus ``wait`` (let ``ticks``
  pass under ``idle_action``) and ``macros`` (``{name: {action, ticks, description}}``: a named action held for a number of
  ticks, e.g. "brake for 1 s"). It acts at a few decisions per episode while the simulator runs many ticks: CARLA's
  tool-call to simulation-time policy.

Results: ``content`` is a compact JSON (observation rounded to ``round`` digits, reward, terminated, truncated, tick) for
language models; ``data`` holds the exact values. ``terminated`` / ``truncated`` end the episode.

**State = the journal** (``json_doc`` adapter at ``path``): seed, reset digest, every action with the digest of its
observation, rewards, ticks. ``journal: lazy`` (high-rate control) keeps it in memory and writes it when a snapshot,
a fingerprint or a verifier query needs it (``App.before_snapshot``), not on every tick. ``restore: replay`` (default) rebuilds the simulator from the seed and replays the actions,
checking every digest (a non-deterministic simulator raises ``SnapshotMismatch``, never silently a different state);
``restore: none`` restores the episode start only (a fresh reset with the seed; a stochastic simulator's different
start is accepted) and refuses mid-episode restores, so branching and resume are off for that environment. A remote
simulator keeps one session for the App's life (servers cap sessions). ``reseed(seed)`` starts a new episode on another seed.
"""

from __future__ import annotations

import hashlib
import json
import posixpath
import shlex
from typing import Any

from crucible.core.schema.actions import ToolResult, ToolSpec
from crucible.core.schema.spaces import Space, from_gymnasium
from crucible.world.apps.base import App, ToolError
from crucible.world.state.base import SnapshotMismatch


def to_jsonable(v: Any) -> Any:
    """numpy arrays and scalars -> lists and Python numbers; dicts and tuples recursively."""
    if hasattr(v, "tolist"):
        return v.tolist()
    if isinstance(v, dict):
        return {str(k): to_jsonable(x) for k, x in v.items()}
    if isinstance(v, list | tuple):
        return [to_jsonable(x) for x in v]
    if hasattr(v, "item"):
        return v.item()
    return v


def _round(v: Any, digits: int) -> Any:
    if isinstance(v, float):
        return round(v, digits)
    if isinstance(v, list):
        return [_round(x, digits) for x in v]
    if isinstance(v, dict):
        return {k: _round(x, digits) for k, x in v.items()}
    return v


def obs_digest(obs: Any) -> str:
    return hashlib.sha256(json.dumps(obs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:24]


def _load(ref: str) -> Any:
    import importlib

    mod, _, attr = ref.partition(":")
    obj = importlib.import_module(mod)
    for part in attr.split(".") if attr else []:
        obj = getattr(obj, part)
    return obj


class RemoteControlEnv:
    """A control environment served elsewhere (OpenEnv's dm_control / Unity / FinRL servers; M1 item 10d), over the
    native WebSocket client (``world/remote.py``). Actions go out as ``{action_key: action}`` (dm_control: ``values``),
    observations come back under ``observation_key`` (dm_control: ``observations``, a dict of named float lists). Spaces
    come from the server's state (``action_spec`` / ``observation_spec``, the dm_env spec dicts OpenEnv exports) unless
    the App declares them. ``done_is``: a server ``done`` is a ``truncated`` episode (dm_control ends at a time limit) or
    ``terminated``; an observation's own ``terminated`` / ``truncated`` keys win."""

    def __init__(self, base_url: str, *, action_key: str | None = "values", observation_key: str | None = "observations", done_is: str = "truncated",
                 session_id: str | None = None, message_timeout_s: float = 300.0, reset_kwargs: dict[str, Any] | None = None,
                 ping_interval_s: float | None = 20.0):
        """``ping_interval_s``: WebSocket keepalive (None: off). A single-worker simulator server that blocks while it loads a
        scene (Unity downloading and starting a binary at reset) cannot answer pings; found live on the lab with Unity."""
        from crucible.world.remote import RemoteEnvClient

        self.client = RemoteEnvClient(base_url, session_id=session_id, message_timeout_s=message_timeout_s, ping_interval_s=ping_interval_s)
        self.action_key, self.observation_key, self.done_is = action_key, observation_key, done_is
        self.reset_kwargs = dict(reset_kwargs or {})  # sent with every reset (OpenEnv Unity: {"env_id": "3DBall"})
        self.state: dict[str, Any] = {}

    def _obs(self, data: dict[str, Any]) -> Any:
        o = data.get("observation") or {}
        return o.get(self.observation_key, o) if self.observation_key else o

    async def areset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        data = await self.client.reset(seed=seed, **self.reset_kwargs)
        self.state = await self.client.state()
        return self._obs(data), {}

    async def astep(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        data = await self.client.step({self.action_key: action} if self.action_key else action)
        o = data.get("observation") or {}
        done = bool(data.get("done"))
        term = bool(o.get("terminated", done and self.done_is == "terminated"))
        trunc = bool(o.get("truncated", done and self.done_is == "truncated"))
        return self._obs(data), float(data.get("reward") or 0.0), term, trunc, {}

    def close(self) -> None:
        import asyncio

        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.client.close())
            return
        t = loop.create_task(self.client.close())
        t.add_done_callback(lambda x: x.exception() if not x.cancelled() else None)


class ControlApp(App):
    template = "common/control"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        self.env: Any = None
        self.action_space: Space | None = None
        self.observation_space: Space | None = None
        self.journal: dict[str, Any] = {}
        self.obs: Any = None
        self.done = False
        self._dirty = False

    # ---- the simulator -------------------------------------------------------------------------------------------
    def _make(self) -> Any:
        p = self.spec.params
        if p.get("client"):
            c = dict(p["client"])
            return RemoteControlEnv(c["base_url"], action_key=c.get("action_key", "values"), observation_key=c.get("observation_key", "observations"),
                                    done_is=c.get("done_is", "truncated"), session_id=c.get("session_id"), message_timeout_s=float(c.get("message_timeout_s", 300)),
                                    reset_kwargs=c.get("reset_kwargs"), ping_interval_s=c.get("ping_interval_s", 20.0))
        if p.get("env"):
            src = str(p["env"])
            if not src.startswith("gymnasium:"):
                raise ValueError(f"{self.id}: env must be 'gymnasium:<id>' (or use factory)")
            import gymnasium

            return gymnasium.make(src.split(":", 1)[1], **(p.get("env_kwargs") or {}))
        if p.get("factory"):
            return _load(p["factory"])(**(p.get("env_kwargs") or {}))
        raise ValueError(f"{self.id}: a control App needs params.env, params.factory or params.client")

    def _spaces(self) -> None:
        from crucible.core.schema.spaces import from_spec_dict

        p = self.spec.params
        st = getattr(self.env, "state", None) if isinstance(self.env, RemoteControlEnv) else None
        if p.get("action_space"):
            self.action_space = Space.model_validate(p["action_space"])
        elif st is not None:
            if not st.get("action_spec"):
                raise ValueError(f"{self.id}: the remote environment exports no action_spec; declare params.action_space")
            self.action_space = from_spec_dict(st["action_spec"])
        else:
            self.action_space = from_gymnasium(self.env.action_space)
        if p.get("observation_space"):
            self.observation_space = Space.model_validate(p["observation_space"])
        elif st is not None:
            self.observation_space = from_spec_dict(st["observation_spec"]) if st.get("observation_spec") else None
        else:
            self.observation_space = from_gymnasium(self.env.observation_space) if hasattr(self.env, "observation_space") else None

    @property
    def frame_skip(self) -> int:
        return max(1, int(self.spec.params.get("frame_skip", 1)))

    @property
    def sim_dt(self) -> float:
        p = self.spec.params
        if p.get("sim_dt_s") is not None:
            return float(p["sim_dt_s"])
        if isinstance(self.env, RemoteControlEnv):
            dt = self.env.state.get("control_timestep")
            return float(dt) if isinstance(dt, int | float) else 0.05
        inner = getattr(self.env, "unwrapped", self.env)
        dt = getattr(inner, "dt", None) or getattr(inner, "tau", None)
        return float(dt) if isinstance(dt, int | float) else 0.05

    def _seed(self) -> int:
        s = self.spec.params.get("seed", "world")
        return int(self.ctx.seed) if s == "world" else int(s)

    def _native(self, action: Any) -> Any:
        sp = self.action_space
        assert sp is not None
        if isinstance(self.env, RemoteControlEnv):
            return to_jsonable(action)  # JSON on the wire
        if sp.kind == "box":
            import numpy as np

            return np.asarray(action, dtype=np.float64 if sp.dtype == "float64" else np.float32).reshape(sp.shape)
        if sp.kind in ("multi_discrete", "multi_binary"):
            import numpy as np

            return np.asarray(action, dtype=np.int64)
        return action

    async def _fresh(self, seed: int) -> Any:
        if isinstance(self.env, RemoteControlEnv):
            pass  # one session for the App's life: a server with a session cap (OpenEnv Unity: 1) refuses a second one
        else:  # (found live: closing asynchronously and reconnecting at every reset hit CAPACITY_REACHED)
            if self.env is not None and hasattr(self.env, "close"):
                self.env.close()
            self.env = self._make()
        if hasattr(self.env, "areset"):
            obs, _info = await self.env.areset(seed=seed)
        else:
            obs, _info = self.env.reset(seed=seed)
        self._spaces()
        self.obs, self.done = to_jsonable(obs), False
        return self.obs

    async def _tick(self, action: Any, ticks: int) -> dict[str, Any]:
        """Hold ``action`` for ``ticks`` simulator ticks (stopping at the end of the episode)."""
        native = self._native(action)
        total, n, term, trunc, info = 0.0, 0, False, False, {}
        cap = self.spec.params.get("max_episode_ticks")
        remote = hasattr(self.env, "astep")
        for _ in range(ticks):
            obs, r, term, trunc, info = (await self.env.astep(native)) if remote else self.env.step(native)
            total += float(r)
            n += 1
            self.journal["tick"] = self.journal.get("tick", 0) + 1
            if cap and self.journal["tick"] >= int(cap):
                trunc = True
            if term or trunc:
                break
        self.obs, self.done = to_jsonable(obs), bool(term or trunc)
        return {"reward": total, "ticks": n, "terminated": bool(term), "truncated": bool(trunc), "info": to_jsonable(info) if isinstance(info, dict) else {}}

    def _clock(self) -> None:
        clock = self.ctx.clock
        if clock is not None and clock.mode == "sim":
            secs = self.journal.get("tick", 0) * self.sim_dt
            if clock.t0.timestamp() + secs > clock.now().timestamp():
                clock.advance_sim_seconds(secs)

    # ---- lifecycle -------------------------------------------------------------------------------------------------
    async def start(self) -> None:
        await self.reseed(self._seed())

    async def reseed(self, seed: int) -> Any:
        """A new episode on ``seed``: the journal starts again (callers that branch take a snapshot afterwards)."""
        obs = await self._fresh(seed)
        self.journal = {"seed": seed, "reset_digest": obs_digest(obs), "steps": [], "tick": 0, "return": 0.0, "done": False}
        await self._write()
        return obs

    async def stop(self) -> None:
        if isinstance(self.env, RemoteControlEnv):
            await self.env.client.close()
        elif self.env is not None and hasattr(self.env, "close"):
            self.env.close()
        self.env = None

    async def _write(self) -> None:
        self._dirty = False
        path = self.state_params["path"]
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(posixpath.dirname(path))}")
        await self.ctx.exec.write_text(path, json.dumps(self.journal, sort_keys=True, separators=(",", ":")))

    async def after_restore(self) -> None:
        doc = json.loads(await self.ctx.exec.read_text(self.state_params["path"]))
        if doc.get("steps") and self.spec.params.get("restore", "replay") == "none":
            raise SnapshotMismatch(f"{self.id}: this simulator restores the episode start only (restore: none); a mid-episode state cannot be rebuilt")
        obs = await self._fresh(int(doc["seed"]))
        if obs_digest(obs) != doc["reset_digest"]:
            if self.spec.params.get("restore", "replay") != "none":
                raise SnapshotMismatch(f"{self.id}: reset on seed {doc['seed']} gave a different observation (non-deterministic reset)")
            doc = {**doc, "reset_digest": obs_digest(obs), "reset_nondeterministic": True}  # restore: none: a fresh start is the start
        self.journal = {**doc, "tick": 0}
        for i, s in enumerate(doc["steps"]):
            await self._tick(s["action"], int(s["ticks_requested"]))
            if obs_digest(self.obs) != s["digest"]:
                raise SnapshotMismatch(f"{self.id}: replayed step {i} gave a different observation: the simulator is not deterministic")
        self.journal = doc
        self.done = bool(doc.get("done"))

    # ---- tools -----------------------------------------------------------------------------------------------------
    async def list_tools(self) -> list[ToolSpec]:
        assert self.action_space is not None
        tools = [ToolSpec(app=self.id, name="step", description=self.spec.params.get("step_description")
                          or f"Apply one action; it holds for {self.frame_skip} simulator tick(s) of {self.sim_dt:g} s.",
                          input_schema={"type": "object", "properties": {"action": self.action_space.json_schema()}, "required": ["action"],
                                        "additionalProperties": False})]
        if self.spec.params.get("idle_action") is not None:
            tools.append(ToolSpec(app=self.id, name="wait", description="Let simulation time pass under the idle action.",
                                  input_schema={"type": "object", "properties": {"ticks": {"type": "integer", "minimum": 1, "maximum": int(self.spec.params.get("max_wait_ticks", 1000))}},
                                                "required": ["ticks"], "additionalProperties": False}))
        for name, m in (self.spec.params.get("macros") or {}).items():
            tools.append(ToolSpec(app=self.id, name=name, description=m.get("description") or f"Hold a preset action for {m.get('ticks', 1)} tick(s).",
                                  input_schema={"type": "object", "properties": {}, "additionalProperties": False}))
        tools.append(ToolSpec(app=self.id, name="observe", read_only=True, description="The current observation, without acting.",
                              input_schema={"type": "object", "properties": {}, "additionalProperties": False}))
        return tools

    def _content(self, out: dict[str, Any] | None) -> str:
        digits = int(self.spec.params.get("round", 4))
        body: dict[str, Any] = {"observation": _round(self.obs, digits), "tick": self.journal.get("tick", 0),
                                "time_s": round(self.journal.get("tick", 0) * self.sim_dt, 6)}
        if out is not None:
            body.update({"reward": round(out["reward"], digits), "terminated": out["terminated"], "truncated": out["truncated"]})
        return json.dumps(body, separators=(",", ":"))

    async def act(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        """Run one tool call against the simulator and journal it (also the entry point of ``WorldEnv``)."""
        assert self.action_space is not None
        if self.done:
            raise ToolError("The episode is over.")
        p = self.spec.params
        if tool == "step":
            action = args.get("action")
            if (why := self.action_space.problem(action)) is not None:
                raise ToolError(f"invalid action: {why}")
            ticks = self.frame_skip
        elif tool == "wait":
            action, ticks = p["idle_action"], int(args["ticks"])
        else:
            m = (p.get("macros") or {})[tool]
            action, ticks = m["action"], int(m.get("ticks", 1))
        out = await self._tick(action, ticks)
        self.journal["steps"].append({"tool": tool, "action": to_jsonable(action), "ticks_requested": ticks, "ticks": out["ticks"],
                                      "reward": out["reward"], "terminated": out["terminated"], "truncated": out["truncated"], "digest": obs_digest(self.obs)})
        self.journal["return"] = self.journal.get("return", 0.0) + out["reward"]
        self.journal["done"] = self.done
        self._clock()
        if self.spec.params.get("journal", "eager") == "lazy":  # high-rate control: written when a snapshot or a verifier needs it
            self._dirty = True
        else:
            await self._write()
        return out

    async def before_snapshot(self) -> None:
        if self._dirty:
            await self._write()

    async def call(self, name: str, args: dict[str, Any], *, actor: str, call_id: str) -> ToolResult:
        q = f"{self.id}.{name}"
        if name not in {t.name for t in await self.list_tools()}:
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=f"Unknown tool: {name}", error="unknown_tool")
        if name == "observe":
            return ToolResult(tool_call_id=call_id, function_name=q, ok=True, content=self._content(None), data={"observation": self.obs})
        try:
            out = await self.act(name, args)
        except ToolError as e:
            over = self.done
            return ToolResult(tool_call_id=call_id, function_name=q, ok=False, content=str(e), error="episode_over" if over else "invalid_action",
                              fault="policy", ends_episode=over)
        return ToolResult(tool_call_id=call_id, function_name=q, ok=True, content=self._content(out), ends_episode=self.done,
                          data={"observation": self.obs, "reward": out["reward"], "terminated": out["terminated"], "truncated": out["truncated"],
                                "ticks": out["ticks"], "tick": self.journal["tick"]})
