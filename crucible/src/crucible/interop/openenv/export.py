"""A Crucible task as an OpenEnv ``Environment`` (export; research m11-m12/05 §3.5). DEPENDs on OpenEnv's own base class
and MCP action/observation types (``openenv.core.env_server``); runs in an environment that has OpenEnv installed
(``crucible/.envs/openenv``), so this module imports OpenEnv lazily via ``make_environment``.

- ``ListToolsAction`` → the World's policy tools as MCP tools (``<app>__<tool>``), plus ``crucible__finish``;
- ``CallToolAction(tool_name, arguments)`` → ``World.step``; ``crucible__finish`` ends the episode and grades it; a call
  after which the World is done (an environment-terminated episode: ``ToolResult.ends_episode``) grades on that step;
- **per-step reward = Ψ-telescoping**: Ψ_t = Φ_t (dense progress over the task's subgoals or its plan's checks) for
  t < T, Ψ_T = the final reward; ``Observation.reward_t = Ψ_t − Ψ_{t−1}``. A consumer that sums step rewards
  (verifiers' OpenEnv bridge) and one that reads the last value get the same total: the final reward;
- **a masked episode raises** ``CrucibleMaskedEpisode`` at its final step (a masked grade, or an infrastructure fault
  recorded in the World during the episode, checked on every step): the Ψ steps already paid sum to Φ, and a consumer
  that sums step rewards and reads a ``None`` final as 0 (verifiers' OpenEnv taskset) scored a masked episode as its
  progress (M11 audit G1 D1, H3 V4). An error is dropped by every consumer;
- after the episode ends, ``crucible__finish`` returns the latched final observation with reward 0 (H3 V5: it re-paid
  the final reward each time) and any other call a ``done`` error observation; episode ids default to a fresh uuid;
- ``metadata["crucible"]``: φ, Δφ, subgoals gained, the derived ``call_id``; at done, the full ``crucible_reward.json``.
"""

from __future__ import annotations

import asyncio
import tempfile
import threading
from pathlib import Path

FINISH = "crucible__finish"


class CrucibleMaskedEpisode(RuntimeError):
    """The episode is masked (infrastructure): never a reward, never a 0."""


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
            self._latched = None  # the final observation, once the episode ended

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
            self._psi, self._ordinal, self._latched = 0.0, 0, None
            import uuid

            self._episode = episode_id or f"{episode_prefix}|{seed if seed is not None else uuid.uuid4().hex}"
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
            if self._latched is not None:  # the episode is over: nothing more is paid or executed
                if action.tool_name == FINISH:
                    return CallToolObservation(tool_name=FINISH, result=self._latched.result, done=True, reward=0.0, metadata=dict(self._latched.metadata or {}))
                return CallToolObservation(tool_name=action.tool_name, result="the episode is over", done=True, reward=0.0,
                                           metadata={"crucible": {"after_done": True}})
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
            if self._fault() is not None:  # an infrastructure fault ends the episode as masked, at once
                return await self._final(action.tool_name, tr.content if tr else "", cid)
            if res.done:  # the World's environment ended the episode with this call (an imported OpenEnv env's final answer)
                return await self._final(action.tool_name, tr.content if tr else "", cid)
            delta = row.phi - self._psi
            self._psi = row.phi
            return CallToolObservation(tool_name=action.tool_name, result=tr.content if tr else "", done=False, reward=delta,
                                       metadata={"crucible": {"phi": row.phi, "delta_phi": delta, "gained": row.gained, "call_id": cid, "ok": tr.ok if tr else False}})

        def _fault(self) -> str | None:
            from crucible.actors.policy import _masking_failure

            return _masking_failure(self._world)

        async def _final(self, tool_name: str, result: str, cid: str) -> Observation:
            rep = await self._task.grade(self._world, self._dir / "grade")
            rj = rep.reward_json()
            fault = self._fault()
            if fault is not None and (rj.get("infra") or {}).get("status") != "masked":
                rj = {**rj, "infra": {"status": "masked", "failure_kind": fault, "reason": "an infrastructure fault during the episode"}}
            doc = write_reward_files(self._dir / "out", rj, rep.to_dict())
            masked = (rj.get("infra") or {}).get("status") == "masked"
            reward = None if masked else float(rj["reward"]) - self._psi
            obs = CallToolObservation(tool_name=tool_name, result=result, done=True, reward=reward,
                                      metadata={"masked": masked, "crucible_reward": doc, "crucible": {"call_id": cid, "psi_final": None if masked else float(rj["reward"])}})
            self._latched = obs
            if not masked:
                self._psi = float(rj["reward"])
            if masked:
                raise CrucibleMaskedEpisode(f"masked: {(rj.get('infra') or {}).get('failure_kind')}: {(rj.get('infra') or {}).get('reason', '')}"[:300])
            return obs

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
