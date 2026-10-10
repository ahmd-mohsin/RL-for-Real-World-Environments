"""Harbor task directories executed **in place**, with Harbor's own semantics (DEPEND: ``harbor.verifier``,
``harbor.agents.oracle``, ``harbor.trial.network_policy``; m13 audit 03, gap G1–G4).

``verify_dir.HarborDirTask`` grades a *patch* in a fresh container and leaves applying it to the task's ``test.sh``.
That holds for Crucible-authored tasks (the verifier kit applies ``/logs/artifacts/model.patch``) but not for foreign
Harbor tasks: terminal-bench pytest-checks ``/app``, BeyondSWE ``/workspace/<repo>``, MiMo-harbor-code diffs
``/testbed``. Their verifiers grade the **container state the agent left**. So through the patch path, an oracle and a
no-op get the same (pristine-image) grade: a silent wrong reward.

``HarborInPlaceTask`` is an M10 ``RunTask``:

- ``open``: a World on the Harbor backend (Harbor's own environment factory: Docker / GPU Docker / any provider) built
  from the task directory. The policy gets one App, ``harbor/terminal`` (``exec``, ``read_file``, ``write_file``),
  running in the task container at the task's working directory. Images need neither python3 nor git (the App's state
  adapter is ``process``: in-place Harbor state is the container itself, so a mid-episode restore is not available).
- ``oracle(world)``: Harbor's ``OracleAgent`` runs ``solution/solve.sh`` in the container (acceptance: oracle → the
  native reward).
- ``grade(world)``: Harbor's ``Verifier`` uploads ``tests/`` and runs the verifier command **in the same container**,
  with ``[verifier].env`` and ``user``, under Harbor's own phase network plan (``resolve_trial_network_plan``: the
  task's ``[environment].network_mode`` / deprecated ``allow_internet`` baseline and the ``[verifier]`` override). A
  phase policy the environment cannot apply masks the grade (``check_infra``) instead of scoring 0.
- **reward**: Harbor's ``reward.json`` is a flat dict of named numbers (``reward.txt`` one number). The scalar is
  ``reward`` when present, else the single key, else ``reward_key``; with several keys and no rule the grade masks
  (``ambiguous_reward``) rather than invent a scalar. Every key is kept in ``components`` (``harbor/<key>``). No
  reward file, an unparseable one, or a verifier timeout masks the grade.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, ToolError


class TerminalApp(App):
    """A shell in the task container (Harbor terminal-agent surface): ``exec``, ``read_file``, ``write_file``."""

    template = "harbor/terminal"
    PATH_PARAMS = ()

    async def list_tools(self) -> list[ToolSpec]:
        return [
            ToolSpec(app=self.id, name="exec", description="Run a shell command in the task container; returns exit code and output.",
                     input_schema={"type": "object", "properties": {"command": {"type": "string"}, "timeout_s": {"type": "number"}}, "required": ["command"]}),
            ToolSpec(app=self.id, name="read_file", description="Read a text file.",
                     input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}),
            ToolSpec(app=self.id, name="write_file", description="Write a text file (parents created).",
                     input_schema={"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]}),
        ]

    @property
    def _cwd(self) -> str | None:
        return self.spec.params.get("workdir")

    async def tool_exec(self, command: str, timeout_s: float = 600.0):
        r = await self.ctx.exec.run(command, cwd=self._cwd, timeout_s=min(float(timeout_s), float(self.spec.params.get("max_timeout_s", 3600))))
        return {"exit_code": r.rc, "output": (r.stdout + r.stderr)[-20000:], "timed_out": r.timed_out}

    async def tool_read_file(self, path: str):
        import shlex

        r = await self.ctx.exec.run(f"cat -- {shlex.quote(path)}", cwd=self._cwd, timeout_s=60)
        if not r.ok:
            raise ToolError(f"cannot read {path}: {(r.stdout + r.stderr)[-500:]}")
        return r.stdout

    async def tool_write_file(self, path: str, content: str):
        import shlex

        r = await self.ctx.exec.run(f"mkdir -p \"$(dirname -- {shlex.quote(path)})\" && cat > {shlex.quote(path)}", cwd=self._cwd, timeout_s=60,
                                    stdin=content.encode())
        if not r.ok:
            raise ToolError(f"cannot write {path}: {(r.stdout + r.stderr)[-500:]}")
        return f"wrote {len(content)} characters to {path}"


def scalar_reward(rewards: dict[str, float] | None, reward_key: str | None = None) -> tuple[float | None, str | None]:
    """Harbor reward dict → (scalar, problem). ``reward`` key, else the single key, else ``reward_key``."""
    if not rewards:
        return None, "no_reward"
    if "reward" in rewards:
        return float(rewards["reward"]), None
    if len(rewards) == 1:
        return float(next(iter(rewards.values()))), None
    if reward_key and reward_key in rewards:
        return float(rewards[reward_key]), None
    return None, f"ambiguous_reward: keys {sorted(rewards)} and no reward / reward_key"


class HarborInPlaceTask:
    def __init__(self, task_dir: Path, *, docker_host: str | None = None, gpu_host: str | None = None, reward_key: str | None = None,
                 environment_type: str = "docker", environment_kwargs: dict | None = None, user: str | None = None, keep_containers: bool = False):
        from harbor.models.task.task import Task

        self.dir = Path(task_dir).resolve()
        self.task = Task(self.dir)
        cfg = self.task.config
        name = self.task.short_name if hasattr(self.task, "short_name") else self.dir.name
        self.id, self.world_id = f"harbor/{name}", f"harbor:{name}"
        self.instruction = self.task.instruction if hasattr(self.task, "instruction") else (self.dir / "instruction.md").read_text()
        self.reward_key = reward_key
        self.workdir_in_container = cfg.environment.workdir
        self.verifier_timeout = float(cfg.verifier.timeout_sec or 600)
        self.backend_params = {"task_dir": str(self.dir), "require_state_tools": False, "environment_type": environment_type,
                               "environment_kwargs": dict(environment_kwargs or {}), "keep_containers": keep_containers}
        if docker_host:
            self.backend_params["docker_host"] = docker_host
        if gpu_host:
            self.backend_params["gpu_host"] = gpu_host
        if user:
            self.backend_params["user"] = user
        self.plan = self._network_plan()

    def _network_plan(self):
        from harbor.models.task.verifier_mode import resolve_task_verifier_mode
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.trial.network_policy import resolve_trial_network_plan

        return resolve_trial_network_plan(self.task.config, TrialAgentConfig(), TrialEnvironmentConfig(), None,
                                          verifier_mode=resolve_task_verifier_mode(self.task.config))

    def _world_network(self) -> tuple[str, list[str]]:
        base = self.plan.agent_env_baseline
        mode = getattr(base.network_mode, "value", str(base.network_mode))
        return {"no-network": "no-network", "allowlist": "allowlist"}.get(mode, "public"), list(base.allowed_hosts or [])

    def spec(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec

        net, hosts = self._world_network()
        app = {"id": "term", "template": "harbor/terminal", "state": {"adapter": "process", "params": {}},
               "params": {"workdir": self.workdir_in_container} if self.workdir_in_container else {}}
        gpus = int(self.task.config.environment.gpus or 0)
        if gpus:
            app["resources"] = {"gpus": gpus}
        doc = {"world": self.world_id, "backend": "harbor", "backend_params": self.backend_params, "network": net, "apps": [app],
               "actors": [{"id": "agent", "kind": "policy", "tools": ["term.*"]}],
               "limits": {"tool_timeout_s": 3600}}
        if net == "allowlist":
            doc["allowed_hosts"] = hosts
        return WorldSpec.model_validate(doc)

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec(workdir), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", backend="harbor",
                  backend_params=self.backend_params)
        await w.start()
        await w.reset()
        await self._healthcheck(w)
        return w

    async def _healthcheck(self, world) -> None:
        env = world.backend.env
        if self.task.config.environment.healthcheck is not None and hasattr(env, "run_healthcheck"):
            await env.run_healthcheck()

    @staticmethod
    def _trial_paths(world):
        """The environment's own trial paths: Harbor mounts ``/logs/{agent,verifier,artifacts}`` from them, so the verifier
        must read the reward where the container wrote it."""
        tp = world.backend.env.trial_paths
        tp.mkdir()
        return tp

    async def oracle(self, world, workdir: Path) -> None:
        """Harbor's OracleAgent: ``solution/solve.sh`` in the task container (the reference solution)."""
        from harbor.agents.oracle import OracleAgent
        from harbor.models.agent.context import AgentContext

        tp = self._trial_paths(world)
        agent = OracleAgent(logs_dir=tp.agent_dir, task_dir=self.dir, trial_paths=tp)
        env = world.backend.env
        with env.with_default_user(self.task.config.agent.user):
            await agent.run(self.instruction, env, AgentContext())

    async def grade(self, world, workdir: Path):
        from crucible.verify.runtime.plan import GradeReport

        t0 = time.monotonic()

        def masked(kind: str, reason: str) -> GradeReport:
            return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0,
                               infra={"status": "masked", "failure_kind": kind, "reason": reason[:500]}, trusted=True, elapsed_s=round(time.monotonic() - t0, 2))

        from harbor.verifier.verifier import Verifier

        env = world.backend.env
        tp = self._trial_paths(world)
        plan = self.plan
        phase, base = plan.verifier_phase, plan.agent_env_baseline
        try:
            if phase != base:
                caps = getattr(env, "capabilities", None)
                if not getattr(caps, "dynamic_network_policy", False):
                    return masked("check_infra", f"the verifier needs network {phase} but this environment cannot change its policy ({base})")
                await env.set_network_policy(phase)
            with env.with_default_user(self.task.config.verifier.user):
                verifier = Verifier(task=self.task, trial_paths=tp, environment=env)
                result = await asyncio.wait_for(verifier.verify(), timeout=self.verifier_timeout + 60)
        except TimeoutError:
            return masked("verifier_timeout", f"verifier exceeded {self.verifier_timeout:.0f}s")
        except Exception as e:  # noqa: BLE001 -- Harbor's typed verifier errors (no / empty / unparseable reward file): never a 0
            return masked(type(e).__name__, str(e))
        finally:
            if phase != base and getattr(getattr(env, "capabilities", None), "dynamic_network_policy", False):
                await env.set_network_policy(base)
        rewards = dict(result.rewards or {})
        reward, problem = scalar_reward(rewards, self.reward_key)
        if reward is None:
            return masked(problem.split(":")[0], problem)
        comps: dict[str, Any] = {f"harbor/{k}": float(v) for k, v in rewards.items()}
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"rewards": rewards, "components": comps, "verifier": json.loads(result.model_dump_json())})
