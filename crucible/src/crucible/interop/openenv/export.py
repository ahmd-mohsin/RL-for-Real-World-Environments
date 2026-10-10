"""A Crucible task as an OpenEnv ``Environment`` (export; research m11-m12/05 §3.5). DEPENDs on OpenEnv's own base class
and MCP action/observation types (``openenv.core.env_server``); runs in an environment that has OpenEnv installed
(``crucible/.envs/openenv``), so this module imports OpenEnv lazily via ``make_environment``.

- ``ListToolsAction`` → the World's policy tools as MCP tools (``<app>__<tool>``), plus ``crucible__finish``;
- ``CallToolAction(tool_name, arguments)`` → ``World.step``; ``crucible__finish`` ends the episode and grades it; a call
  after which the World is done (an environment-terminated episode: ``ToolResult.ends_episode``) grades on that step;
- **per-step reward = Ψ-telescoping**: Ψ_t = Φ_t (dense progress over the task's subgoals or its plan's checks) for
  t < T, Ψ_T = the final reward; ``Observation.reward_t = Ψ_t − Ψ_{t−1}``. A consumer that sums step rewards
  (verifiers' OpenEnv bridge) and one that reads the last value get the same total: the final reward;
- masked grade → ``reward = None`` and ``metadata["masked"] = True`` (TRL treats None as unscorable, OpenEnv's
  TrainingTrace as unscored), never a 0;
- ``metadata["crucible"]``: φ, Δφ, subgoals gained, the derived ``call_id``; at done, the full ``crucible_reward.json``.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

FINISH = "crucible__finish"


class _Loop:
    """A private event loop on a thread, so OpenEnv's sync ``reset``/``step`` can drive the async World."""

    def __init__(self) -> None:
        self.loop = asyncio.new_event_loop()
        threading.Thread(target=self.loop.run_forever, daemon=True).start()

    def run(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result()


def make_environment(task_factory, *, episode_prefix: str = "openenv"):
    from openenv.core.env_server.interfaces import Environment
    from openenv.core.env_server.mcp_types import CallToolAction, CallToolObservation, ListToolsAction, ListToolsObservation, Tool
    from openenv.core.env_server.types import Observation, State

    from crucible.interop.canonical import call_id, canonical_tool
    from crucible.interop.replay import PlanSubgoals
    from crucible.verify.progress import ProgressTracker
    from crucible.verify.runtime.plan import write_reward_files

    class CrucibleEnvironment(Environment):
        SUPPORTS_CONCURRENT_SESSIONS = True

        def __init__(self) -> None:
            super().__init__()
            self._loop = _Loop()
            self._world = None
            self._task = None
            self._state = State()
            self._psi = 0.0
            self._ordinal = 0
            self._episode = ""
            self._tracker = None

        # ---- lifecycle -------------------------------------------------------------------------------------------
        async def _reset(self, seed, episode_id, **kw) -> Observation:
            if self._world is not None:
                await self._world.close()
            self._task = task_factory(**kw) if kw else task_factory()
            self._dir = Path(tempfile.mkdtemp(prefix="openenv-"))
            self._world = await self._task.open(self._dir / "work")
            sg = getattr(self._task, "subgoals", None) or PlanSubgoals(self._task.plan_for_progress(), workdir=self._dir / "p")
            self._tracker = ProgressTracker(sg, full_every=1)
            await self._tracker.on_start(self._world)
            # Ψ_{-1} := 0, not Φ_0: a plan's invariants can hold at the start (Φ_0 > 0), and the step rewards must sum to the
            # final reward itself; the first step carries Φ_0 (found by the M11 gate: Σ was 0.6 for a final 1.0)
            self._psi, self._ordinal = 0.0, 0
            self._episode = episode_id or f"{episode_prefix}|{seed}"
            self._state = State(episode_id=self._episode, step_count=0)
            return Observation(done=False, reward=None, metadata={"instruction": self._world.instructions.get("agent", ""), "crucible": {"phi": self._tracker.phi}})

        def reset(self, seed=None, episode_id=None, **kw) -> Observation:
            return self._loop.run(self._reset(seed, episode_id, **kw))

        async def reset_async(self, seed=None, episode_id=None, **kw) -> Observation:
            return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._reset(seed, episode_id, **kw), self._loop.loop))

        def _tools(self) -> list:
            ts = [Tool(name=t.qualified.replace(".", "__", 1), description=t.description or "", input_schema=t.input_schema or {"type": "object"})
                  for t in self._world.tools_for("agent")]
            return [*ts, Tool(name=FINISH, description="End the episode and grade it.", input_schema={"type": "object", "properties": {}})]

        async def _step(self, action) -> Observation:
            from crucible.core.schema.actions import Done, ToolCall

            if isinstance(action, ListToolsAction):
                return ListToolsObservation(tools=self._tools(), done=False, reward=None)
            assert isinstance(action, CallToolAction)
            self._state.step_count += 1
            cid = call_id(self._episode, self._ordinal)
            self._ordinal += 1
            if action.tool_name == FINISH:
                if "agent" not in self._world.done:
                    res = await self._world.step("agent", Done(reason="finish"))
                    await self._tracker.on_step(self._world, self._state.step_count, "agent", Done(reason="finish"), res)
                return await self._final(FINISH, "graded", cid)
            known = [t.qualified for t in self._world.tools_for("agent")]
            act = ToolCall(tool_call_id=f"oe{self._ordinal}", function_name=canonical_tool(action.tool_name, known=known), arguments=dict(action.arguments))
            res = await self._world.step("agent", act)
            row = await self._tracker.on_step(self._world, self._state.step_count, "agent", act, res)
            tr = res.observation.tool_results[0] if res.observation.tool_results else None
            if res.done:  # the World's environment ended the episode with this call (an imported OpenEnv env's final answer)
                return await self._final(action.tool_name, tr.content if tr else "", cid)
            delta = row.phi - self._psi
            self._psi = row.phi
            return CallToolObservation(tool_name=action.tool_name, result=tr.content if tr else "", done=False, reward=delta,
                                       metadata={"crucible": {"phi": row.phi, "delta_phi": delta, "gained": row.gained, "call_id": cid, "ok": tr.ok if tr else False}})

        async def _final(self, tool_name: str, result: str, cid: str) -> Observation:
            rep = await self._task.grade(self._world, self._dir / "grade")
            rj = rep.reward_json()
            doc = write_reward_files(self._dir / "out", rj, rep.to_dict())
            masked = (rj.get("infra") or {}).get("status") == "masked"
            reward = None if masked else float(rj["reward"]) - self._psi
            return CallToolObservation(tool_name=tool_name, result=result, done=True, reward=reward,
                                       metadata={"masked": masked, "crucible_reward": doc, "crucible": {"call_id": cid, "psi_final": None if masked else float(rj["reward"])}})

        def step(self, action, timeout_s=None, **kw) -> Observation:
            return self._loop.run(self._step(action))

        async def step_async(self, action, timeout_s=None, **kw) -> Observation:
            return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(self._step(action), self._loop.loop))

        @property
        def state(self) -> State:
            return self._state

        def close(self) -> None:
            if self._world is not None:
                self._loop.run(self._world.close())
                self._world = None

    return CrucibleEnvironment
