"""Gymnasium-API environments over Worlds, for neural-network policies (M1 item 10c).

``WorldEnv`` wraps a World that holds one control App (``world/apps/control.py``) as ``reset(seed) -> (obs, info)`` and
``step(action) -> (obs, reward, terminated, truncated, info)``. Every step still goes through ``World.step`` (the tool
gate, the trace, events on simulation time, quotas), so a neural policy and a language model act on the same
environment under the same rules; only the policy differs. ``obs`` is the flat float vector of the observation space
(``Space.flatten``) unless ``flatten=False``. Async methods (``areset`` / ``astep``) let a trainer step many Worlds
concurrently; the sync methods run them on a private event loop (for Gymnasium-style libraries).

``VectorWorldEnv`` steps N Worlds together and resets each one when its episode ends (the auto-reset convention of
Gymnasium's vector API): ``step`` returns the next observations, and the final observation of an ended episode is in
``info["final_obs"]``.
"""

from __future__ import annotations

import asyncio
import itertools
import tempfile
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.spaces import Space


class WorldEnv:
    def __init__(self, task: Any, *, workdir: Path | None = None, app: str | None = None, actor: str = "agent", flatten: bool = True):
        """``task``: a ``ControlTask`` (or anything with ``open(workdir) -> World`` and ``app``)."""
        self.task, self.actor, self.flatten = task, actor, flatten
        self.app_id = app or getattr(task, "app", "sim")
        self.workdir = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="crucible-worldenv-"))
        self.world: Any = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ids = itertools.count()
        self.episode_return = 0.0
        self.episode_ticks = 0

    # ---- spaces ----------------------------------------------------------------------------------------------------
    @property
    def app(self) -> Any:
        return self.world.apps[self.app_id]

    @property
    def action_space(self) -> Space:
        return self.app.action_space

    @property
    def observation_space(self) -> Space:
        return self.app.observation_space

    @property
    def obs_dim(self) -> int:
        return self.observation_space.flat_dim()

    def _obs(self, raw: Any) -> Any:
        return self.observation_space.flatten(raw) if self.flatten and self.observation_space is not None else raw

    # ---- async API -------------------------------------------------------------------------------------------------
    async def aopen(self) -> None:
        if self.world is None:
            self.world = await self.task.open(self.workdir)

    async def areset(self, seed: int | None = None) -> tuple[Any, dict[str, Any]]:
        await self.aopen()
        await self.world.reset()
        if seed is not None:
            await self.app.reseed(int(seed))
        self.episode_return, self.episode_ticks = 0.0, 0
        return self._obs(self.app.obs), {"seed": self.app.journal.get("seed")}

    async def astep(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        sp = self.action_space
        if sp.kind == "box":
            action = sp.clip([float(x) for x in (action.tolist() if hasattr(action, "tolist") else action)])
        elif sp.kind == "discrete":
            action = int(action)
        res = await self.world.step(self.actor, ToolCall(tool_call_id=f"s{next(self._ids)}", function_name=f"{self.app_id}.step", arguments={"action": action}))
        r = res.observation.tool_results[0]
        if not r.ok:
            raise RuntimeError(f"step refused: {r.error}: {r.content[:300]}")  # a neural policy's actions are clipped: this is a bug, not data
        d = r.data
        self.episode_return += d["reward"]
        self.episode_ticks += d["ticks"]
        info = {"ticks": d["ticks"], "tick": d["tick"]}
        if d["terminated"] or d["truncated"]:
            info["episode"] = {"return": self.episode_return, "ticks": self.episode_ticks}
        return self._obs(d["observation"]), float(d["reward"]), bool(d["terminated"]), bool(d["truncated"]), info

    async def aclose(self) -> None:
        if self.world is not None:
            await self.world.close()
            self.world = None

    # ---- sync API (Gymnasium-style libraries) ----------------------------------------------------------------------
    def _run(self, coro: Any) -> Any:
        if self._loop is None:
            self._loop = asyncio.new_event_loop()
        return self._loop.run_until_complete(coro)

    def reset(self, seed: int | None = None, options: dict | None = None) -> tuple[Any, dict[str, Any]]:
        return self._run(self.areset(seed))

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        return self._run(self.astep(action))

    def close(self) -> None:
        if self._loop is not None:
            self._loop.run_until_complete(self.aclose())
            self._loop.close()
            self._loop = None

    def gym_spaces(self) -> tuple[Any, Any]:
        """(action_space, observation_space) as Gymnasium objects, for libraries that need them (needs ``gymnasium``)."""
        import gymnasium as g
        import numpy as np

        def conv(sp: Space, flat: bool = False) -> Any:
            if flat or sp.kind == "box":
                if sp.kind == "box":
                    lo, hi = sp.bounds()
                    shape = [sp.flat_dim()] if flat else sp.shape
                    return g.spaces.Box(np.array(lo, dtype=np.float32).reshape(shape), np.array(hi, dtype=np.float32).reshape(shape))
                return g.spaces.Box(-np.inf, np.inf, (sp.flat_dim(),), np.float32)
            if sp.kind == "discrete":
                return g.spaces.Discrete(int(sp.n), start=sp.start)
            if sp.kind == "multi_discrete":
                return g.spaces.MultiDiscrete(sp.nvec)
            if sp.kind == "multi_binary":
                return g.spaces.MultiBinary(int(sp.n))
            raise ValueError(f"no Gymnasium form for {sp.kind}")

        return conv(self.action_space), conv(self.observation_space, flat=self.flatten)


class VectorWorldEnv:
    """N ``WorldEnv``s stepped concurrently, each reset (on a fresh seed) when its episode ends."""

    def __init__(self, make_env: Any, n: int, *, seed: int = 0):
        self.envs = [make_env(i) for i in range(n)]
        self.n, self.seed = n, seed
        self._episodes = 0

    def _next_seed(self) -> int:
        self._episodes += 1
        return self.seed * 1_000_003 + self._episodes

    @property
    def action_space(self) -> Space:
        return self.envs[0].action_space

    @property
    def observation_space(self) -> Space:
        return self.envs[0].observation_space

    async def areset(self) -> list[Any]:
        out = await asyncio.gather(*(e.areset(self._next_seed()) for e in self.envs))
        return [o for o, _ in out]

    async def astep(self, actions: list[Any]) -> tuple[list[Any], list[float], list[bool], list[bool], list[dict[str, Any]]]:
        res = await asyncio.gather(*(e.astep(a) for e, a in zip(self.envs, actions, strict=True)))
        obs, rew, term, trunc, infos = [list(x) for x in zip(*res, strict=True)]
        for i, e in enumerate(self.envs):
            if term[i] or trunc[i]:
                infos[i] = {**infos[i], "final_obs": obs[i]}
                obs[i], _ = await e.areset(self._next_seed())
        return obs, rew, term, trunc, infos

    async def aclose(self) -> None:
        await asyncio.gather(*(e.aclose() for e in self.envs))
