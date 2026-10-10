"""The ``crucible_world`` and ``crucible_verify`` task steps (research m11-m12/05 §4.2–4.3).

``crucible_world`` (design A): builds the task's World on Crucible's ``AgentEnvBackend`` — a real agentenv gateway with one
MCP server per App (tools ``<app>__<tool>``), per-role ACL, clock and triggers — and appends a ``DeployedGatewayEnv`` to
the context, so agentenv's stock ``deploy_agent`` / ``prompt_agent`` / clock / trigger steps work unchanged. The World
object lives in this process (``WORLDS``); the persisted context holds ids only.

``crucible_verify``: grades the World from its state with the task's own grader and Crucible's single reward producer
(never agentenv's ``reward.txt`` path: that clamps to [0, 1] and scores a missing reward as 0). It records
``context.metadata["verifications"][verifier_id]`` in the aggregate shape (one row per gate, ``score`` = the reward,
unclamped, plus ``crucible: {reward_json, masked, failure_kind, steps}``); ``steps`` come from replaying the gateway's
own trajectory (``GET /trajectory``). **A masked grade records rows with ``skipped: True`` and then raises**, so the run
is FAILED (infrastructure), never scored 0.
"""

from __future__ import annotations

import importlib
import json
import tempfile
from pathlib import Path
from typing import Any, ClassVar

from agent_env.env.env import DeployedGatewayEnv
from agent_env.task_step.task_step import TaskStep

WORLDS: dict[str, dict[str, Any]] = {}


def _load(spec: str):
    mod, _, attr = spec.partition(":")
    return getattr(importlib.import_module(mod), attr)


class CrucibleMaskedGrade(RuntimeError):
    """The grade was masked (infrastructure): the run fails, it is never scored 0."""


class CrucibleWorldStep(TaskStep):
    type: ClassVar[str] = "crucible_world"

    def __init__(self, id: str, version: int | None = None, *, env_id: str, task: str, task_kwargs: dict | None = None, **kw: Any):
        super().__init__(id, version, **kw)
        self.env_id, self.task, self.task_kwargs = env_id, task, dict(task_kwargs or {})

    async def execute(self, context):
        task = _load(self.task)(**{"backend": "agentenv", **self.task_kwargs})
        d = Path(tempfile.mkdtemp(prefix="ae-crucible-"))
        world = await task.open(d / "work")
        WORLDS[self.env_id] = {"task": task, "world": world, "dir": d}
        context.deployed_envs.append(DeployedGatewayEnv(env_id=self.env_id, env_version=0, gateway_url=world.backend.gateway_url, gateway_mode="consistent",
                                                        sandbox_id=f"crucible-local:{self.env_id}", sandbox_type="crucible_world"))
        context.metadata.setdefault("crucible", {})[self.env_id] = {"task": self.task, "world": getattr(task, "world_id", "")}
        return context

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "env_id": self.env_id, "task": self.task, "task_kwargs": self.task_kwargs}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrucibleWorldStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], task=data["task"], task_kwargs=data.get("task_kwargs"))


class CrucibleVerifyStep(TaskStep):
    type: ClassVar[str] = "crucible_verify"

    def __init__(self, id: str, version: int | None = None, *, env_id: str, verifier_id: str = "crucible", episode_key: str = "e0", **kw: Any):
        super().__init__(id, version, **kw)
        self.env_id, self.verifier_id, self.episode_key = env_id, verifier_id, episode_key

    async def execute(self, context):
        import httpx

        from crucible.interop.replay import from_gateway_jsonl, replay
        from crucible.verify.runtime.plan import write_reward_files

        s = WORLDS[self.env_id]
        task, world, d = s["task"], s["world"], s["dir"]
        rep = await task.grade(world, d / "grade")
        rj = rep.reward_json()
        doc = write_reward_files(d / "verify", rj, rep.to_dict())
        masked = (rj.get("infra") or {}).get("status") == "masked"
        async with httpx.AsyncClient(timeout=60) as c:
            traj = (await c.get(f"{world.backend.gateway_url}/trajectory")).text
        events = [json.loads(x) for x in traj.splitlines() if x.strip()]
        known = [t.qualified for t in world.tools_for("agent")]
        steps_dir = d / "steps"
        local_task = type(task)(**{k: v for k, v in getattr(task, "__dict__", {}).items() if k == "instruction"})  # replay on the local backend
        await replay(local_task, from_gateway_jsonl(events, known=known), steps_dir, episode_key=f"{getattr(task, 'id', self.env_id)}|{self.episode_key}",
                     harness="agentenv@1")
        rows = [json.loads(x) for x in (steps_dir / "steps.jsonl").read_text().splitlines() if x.strip()]
        gates = rj.get("gates") or {}
        results = [{"id": f"gate/{g}", "score": float(v), "result": "pass" if v else "fail", "weight": 1.0, **({"skipped": True} if masked else {})}
                   for g, v in sorted(gates.items())]
        context.metadata.setdefault("verifications", {})[self.verifier_id] = {
            "results": results, "score": None if masked else float(rj["reward"]),
            "crucible": {"reward_json": doc, "masked": masked, "failure_kind": (rj.get("infra") or {}).get("failure_kind"), "steps": rows,
                         "out_dir": str(d / "verify")}}
        if masked:
            raise CrucibleMaskedGrade(f"crucible grade masked: {(rj.get('infra') or {}).get('failure_kind')}")
        return context

    def to_dict(self) -> dict[str, Any]:
        return {**super().to_dict(), "env_id": self.env_id, "verifier_id": self.verifier_id, "episode_key": self.episode_key}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CrucibleVerifyStep:
        return cls(**cls._base_from_dict(data), env_id=data["env_id"], verifier_id=data.get("verifier_id", "crucible"),
                   episode_key=data.get("episode_key", "e0"))
