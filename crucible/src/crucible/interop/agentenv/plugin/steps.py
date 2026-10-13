"""The ``crucible_world`` and ``crucible_verify`` task steps (research m11-m12/05 §4.2–4.3).

``crucible_world`` (design A): builds the task's World on Crucible's ``AgentEnvBackend`` — a real agentenv gateway with one
MCP server per App (tools ``<app>__<tool>``), per-role ACL, clock and triggers — and appends a ``DeployedGatewayEnv`` to
the context, so agentenv's stock ``deploy_agent`` / ``prompt_agent`` / clock / trigger steps work unchanged. The record's
``mcp_url`` is the agent proxy (``/mcp`` only, role pinned); ``gateway_url`` stays for orchestration steps. The World
object lives in this process (``WORLDS``), keyed by a per-run ``instance_id`` (M11 audit G4 D1: keyed by ``env_id``,
concurrent runs of one task graded each other's World); the persisted context holds ids only. ``sandbox_id`` is empty, so
agentenv's ``teardown_run`` skips it (G4 D6: an unknown sandbox type failed teardown).

``crucible_verify``: grades the World from its state with the task's own grader and Crucible's single reward producer
(never agentenv's ``reward.txt`` path: that clamps to [0, 1] and scores a missing reward as 0). An infrastructure fault
recorded in the World during the episode masks it (H2 V3). ``steps`` come from replaying the World's own trace (every
outside call is a World step now), and the replay must reproduce the live reward, else the grade is masked
``crucible:replay_divergence`` (G4 D4). It records ``context.metadata["verifications"][verifier_id]`` with one
``reward`` row, agentenv's reward-mode shape (G4 D7: one row per gate made ``aggregate_verifiers`` score the gate mean),
the gates under ``crucible.gates``, and ``metadata["score"]``. **A masked grade records the row with ``skipped: True``
and then raises**, so the run is FAILED (infrastructure), never scored 0. It closes the World unless ``close=False``.
"""

from __future__ import annotations

import importlib
import json
import tempfile
import uuid
from pathlib import Path
from typing import Any, ClassVar

from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.task_step import TaskStep

WORLDS: dict[str, dict[str, Any]] = {}


def _load(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


def instance_for(context, env_id: str) -> str:
    """The ``WORLDS`` key of this run's World for ``env_id`` (the last one this context deployed)."""
    rec = next((e for e in reversed(context.deployed_envs) if e.env_id == env_id and getattr(e, "instance_id", None) in WORLDS), None)
    if rec is None:
        raise CrucibleWorldMissing(f"no live crucible_world for env {env_id!r} in this run (a resumed run cannot reuse a World from another process)")
    return rec.instance_id


async def close_world(context, env_id: str) -> None:
    """Close and forget this run's World (for runs that end before ``crucible_verify``)."""
    s = WORLDS.pop(instance_for(context, env_id), None)
    if s is not None:
        await s["world"].close()


class CrucibleMaskedGrade(RuntimeError):
    """The grade was masked (infrastructure): the run fails, it is never scored 0."""


class CrucibleWorldMissing(RuntimeError):
    """The step's World is not live in this process."""


class CrucibleWorldStep(TaskStep):
    type: ClassVar[str] = "crucible_world"

    def __init__(self, id: str, version: int | None = None, *, env_id: str, task: str, task_kwargs: dict | None = None, **kw: Any):
        super().__init__(id, version, **kw)
        self.env_id, self.task, self.task_kwargs = env_id, task, dict(task_kwargs or {})

    async def execute(self, context):
        import datetime

        import httpx
        from agent_env.env.gateway.constants import WELL_KNOWN_PATH

        task = _load(self.task)(**{"backend": "agentenv", **self.task_kwargs})
        d = Path(tempfile.mkdtemp(prefix="ae-crucible-"))
        world = await task.open(d / "work")
        gw = world.backend.gateway_url
        async with httpx.AsyncClient(timeout=60) as c:
            card = (await c.get(f"{gw}{WELL_KNOWN_PATH}")).raise_for_status().json()
        instance = uuid.uuid4().hex
        WORLDS[instance] = {"task": task, "world": world, "dir": d, "trace_offset": len(world.log), "env_id": self.env_id}
        # the full record (m13 audit 02, C2): stock steps (deploy_agent, sync_env_clock, modify_env_tool_access,
        # register_env_triggers, collect_artifacts) read the card and the MCP URL from it; agents get the proxy (H2 V6)
        rec = DeployedGatewayEnv(
            env_id=self.env_id, env_version=0, gateway_url=gw, gateway_mode="consistent", sandbox_id="",
            sandbox_type="crucible_world", environment_card_url=f"{gw}{WELL_KNOWN_PATH}", environment_card=card,
            environment_card_read_at_utc=datetime.datetime.now(datetime.UTC).isoformat(), mcp_server_name=card.get("name"), instance_id=instance)
        # the card URL stays the gateway's (orchestration steps invoke its extensions); the MCP URL a deploy_agent hands the
        # agent is the proxy's. A record rebuilt from a persisted context derives the gateway's again, but such a run
        # cannot reach this World anyway (``CrucibleWorldMissing``)
        rec.mcp_url = await world.backend.agent_mcp_url()
        context.deployed_envs.append(rec)
        context.metadata.setdefault("crucible", {})[self.env_id] = {"task": self.task, "task_kwargs": self.task_kwargs, "env_id": self.env_id,
                                                                     "world": getattr(task, "world_id", ""), "instance_id": instance}
        return context

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "env_id": self.env_id, "task": self.task, "task_kwargs": self.task_kwargs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrucibleWorldStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], task=data["task"], task_kwargs=data.get("task_kwargs"))


class CrucibleVerifyStep(TaskStep):
    type: ClassVar[str] = "crucible_verify"

    def __init__(self, id: str, version: int | None = None, *, env_id: str, verifier_id: str = "crucible", episode_key: str = "e0", close: bool = True, **kw: Any):
        super().__init__(id, version, **kw)
        self.env_id, self.verifier_id, self.episode_key, self.close = env_id, verifier_id, episode_key, close

    async def execute(self, context):
        from crucible.actors.policy import _masking_failure
        from crucible.interop.replay import from_world_trace, replay
        from crucible.verify.runtime.plan import write_reward_files

        instance = instance_for(context, self.env_id)
        s = WORLDS[instance]
        task, world, d = s["task"], s["world"], s["dir"]
        try:
            rep = await task.grade(world, d / "grade")
            rj = rep.reward_json()
            infra = dict(rj.get("infra") or {})
            fault = _masking_failure(world) if infra.get("status") != "masked" else None
            if fault is not None:  # an infra fault during the episode (H2 V3: a crashed tool behind the gateway scored 0)
                infra = {"status": "masked", "failure_kind": fault, "reason": "an infrastructure fault during the episode"}
            # this episode's actions: the World's own trace since the last verify on this World (C3: reused Worlds)
            events = list(world.log)
            start = int(s.get("trace_offset", 0))
            s["trace_offset"] = len(events)
            trace = world.to_trace()
            actions = from_world_trace(trace.model_copy(update={"events": trace.events[start:]}) if start else trace)
            steps_dir = d / f"steps-{self.verifier_id}"
            step_cfg = next((st for st in context.metadata.get("crucible", {}).values() if isinstance(st, dict) and st.get("instance_id") == instance), None)
            kwargs = dict((step_cfg or {}).get("task_kwargs") or {})
            local_task = _load(step_cfg["task"])(**{**kwargs, "backend": "local"}) if step_cfg else type(task)()  # replay on the local backend, same task
            rp = await replay(local_task, actions, steps_dir, episode_key=f"{getattr(task, 'id', self.env_id)}|{self.episode_key}", harness="agentenv@1")
            if infra.get("status") != "masked" and not start and rp.get("reward") != rep.reward:
                infra = {"status": "masked", "failure_kind": "crucible:replay_divergence",
                         "reason": f"the replayed episode grades {rp.get('reward')}, the live World {rep.reward}"}
            if infra.get("status") == "masked":
                rj = {**rj, "infra": infra}
            doc = write_reward_files(d / f"verify-{self.verifier_id}", rj, rep.to_dict())
            masked = infra.get("status") == "masked"
            rows = [json.loads(x) for x in (steps_dir / "steps.jsonl").read_text().splitlines() if x.strip()]
            reward = None if masked else float(rj["reward"])
            # agentenv's reward-mode verifier row (G4 D7); agentenv's aggregators test ``result is True`` (C3)
            results = [{"id": "reward", "score": 0.0 if masked else reward, "result": bool(not masked and reward >= 1.0), "weight": 1.0,
                        **({"skipped": True} if masked else {})}]
            context.metadata.setdefault("verifications", {})[self.verifier_id] = {
                "results": results, "score": reward, "actions_replayed": len(actions),
                "crucible": {"reward_json": doc, "masked": masked, "failure_kind": infra.get("failure_kind") if masked else None, "steps": rows,
                             "gates": {g: bool(v) for g, v in sorted((rj.get("gates") or {}).items())}, "out_dir": str(d / f"verify-{self.verifier_id}")}}
            context.metadata["score"] = reward  # what ``agentenv eval run`` prints (G4 D16)
        finally:
            if self.close:
                WORLDS.pop(instance, None)
                await world.close()
        if masked:
            raise CrucibleMaskedGrade(f"crucible grade masked: {infra.get('failure_kind')}")
        return context

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "env_id": self.env_id, "verifier_id": self.verifier_id, "episode_key": self.episode_key, "close": self.close}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrucibleVerifyStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], verifier_id=data.get("verifier_id", "crucible"),
                   episode_key=data.get("episode_key", "e0"), close=data.get("close", True))
