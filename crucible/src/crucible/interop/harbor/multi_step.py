"""Harbor multi-step task directories as Crucible multi-step tasks (DEPEND: ``harbor.models.task``, ``harbor.verifier``,
``harbor.trial.multi_step``, ``harbor.trial.network_policy``, ``harbor.agents.oracle``; M1 item 4 ``TaskStep``).

Harbor's layout (``harbor/models/task/paths.py``): ``task.toml`` declares ``[[steps]]`` (``StepConfig``: ``name``,
``agent``, ``verifier``, ``min_reward``, ``healthcheck``, ``artifacts``) and ``multi_step_reward_strategy``; each step
has ``steps/<name>/instruction.md`` and optionally ``tests/``, ``solution/`` and ``workdir/`` (with ``setup.sh``); a
top-level ``tests/`` is shared by every step. ``HarborMultiStepTask`` extends the in-place single-step import
(``inplace.HarborInPlaceTask``: one task container built by Harbor's own environment factory, the ``harbor/terminal``
App) with one ``TaskStep`` per ``[[steps]]`` entry, in declared order, so ``NativeHarness`` runs it like Harbor's
``MultiStepTrial``:

- **instruction**: Harbor's ``Task.step_instruction`` (canary stripped), opened in the running episode.
- **setup** (Harbor ``_prepare_step``): the previous step's shared-verifier leftovers (``/logs/verifier``, ``/tests``)
  are emptied so the policy never sees a grading harness, ``workdir/`` is uploaded into the container's working
  directory, its ``setup.sh`` runs there, then the step's ``healthcheck`` (Docker HEALTHCHECK semantics). The step's
  ``agent.user`` (else the task's) becomes the terminal's user for the step. A failure raises: the attempt masks (Harbor
  records the step as errored and stops; Crucible never scores an environment failure as the policy's 0).
- **grade** (Harbor ``_run_step_verifier``, shared mode): Harbor's ``Verifier`` with ``step_name`` (it merges the shared
  ``tests/`` and ``steps/<name>/tests/`` into ``/tests`` and runs the step's ``test.sh``, else the shared one), the
  step's ``verifier.env``, ``user`` and ``timeout_sec``, under the step's verifier network phase. The reward follows
  ``inplace.scalar_reward``; every key is kept as ``harbor/<key>``. A timeout or a crash masks; after the shared dirs
  are reset a missing reward is the agent's. The grader's toolchain and the container's liveness are checked host-side
  as in place (``HarborInPlaceTask._verify``, M13 audit B1, B2).
- **min_reward**: Harbor's own ``MultiStepTrial._min_reward_failure`` judges the step's reward dict (a float gates
  ``reward``; a dict gates each key, a missing key counting as -inf) and reports it as the outcome gate
  ``harbor_min_reward``; the ``TaskStep`` gates on ``check/harbor_min_reward``, so a failing step ends the task exactly
  where Harbor would.
- **reward**: Harbor ``mean`` (the default: per-key mean over the steps that produced a result) -> ``mean_executed``;
  ``final`` -> ``final``.
- **network**: each step's agent and verifier phases come from ``resolve_trial_network_plan(..., step_cfg)``; phases
  other than the baseline are registered with the backend (Harbor's egress control) and applied around each phase.
- **oracle**: ``oracle_actions()`` replays every step's ``solution/`` through the terminal tools (a scripted policy can
  drive the harness with it); ``oracle_step`` runs Harbor's ``OracleAgent`` for one step directly.

Refused at import (``ValueError``): a task without ``[[steps]]``, and steps whose verifier runs in a separate
environment (``verifier.environment`` / ``environment_mode = "separate"``): the in-place World has one container.
Not mapped (kept in ``step_meta``): per-step ``agent.timeout_sec`` (Crucible budgets a step in actions, ``max_steps``),
per-step ``artifacts`` (Crucible keeps its own per-step evidence dirs), ``trajectory.json`` prior context and agent
resume. Divergences: a verifier failure masks the attempt (Harbor drops the step from the mean); under ``final`` a task
stopped early scores 0 (Harbor reports the last executed step's reward).
"""

from __future__ import annotations

import json
import time
from functools import partial
from pathlib import Path
from typing import Any

from crucible.interop.harbor.inplace import HarborInPlaceTask, scalar_reward
from crucible.runtime.harness.native import TaskStep

MIN_REWARD_GATE = "harbor_min_reward"


def min_reward_failure(rewards: dict[str, float] | None, min_reward: float | dict[str, float]) -> str | None:
    """Harbor's own rule (``MultiStepTrial._min_reward_failure``): why a step's rewards fail its threshold, or None."""
    from harbor.trial.multi_step import MultiStepTrial

    return MultiStepTrial._min_reward_failure(rewards, min_reward)


class HarborMultiStepTask(HarborInPlaceTask):
    def __init__(self, task_dir: Path, **kw: Any):
        super().__init__(task_dir, **kw)
        from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_step_verifier_mode

        cfg = self.task.config
        if not cfg.steps:
            raise ValueError(f"{self.dir}: not a multi-step task (task.toml declares no [[steps]]); use HarborInPlaceTask")
        separate = [s.name for s in cfg.steps if resolve_step_verifier_mode(cfg, s) == VerifierEnvironmentMode.SEPARATE]
        if separate:
            raise ValueError(f"{self.dir}: steps {separate} verify in a separate environment, which the in-place World (one container) cannot host")
        strategy = getattr(cfg.multi_step_reward_strategy, "value", cfg.multi_step_reward_strategy) or "mean"
        self.step_reward = "final" if strategy == "final" else "mean_executed"
        self.step_plans = {s.name: self._step_network_plan(s) for s in cfg.steps}
        base = self.plan.agent_env_baseline
        phases = []
        for p in self.step_plans.values():
            phases += [x for x in (p.agent_phase, p.verifier_phase) if x != base and x not in phases]
        if phases:
            self.backend_params["phase_network_policies"] = [x.model_dump(mode="json") for x in phases]
        else:
            self.backend_params.pop("phase_network_policies", None)
        self.steps = [TaskStep(name=s.name, instruction=self.task.step_instruction(s.name), grade=partial(self._grade_step, s),
                               min_reward={f"check/{MIN_REWARD_GATE}": 1.0} if s.min_reward is not None else None, setup=partial(self._setup_step, s))
                      for s in cfg.steps]
        self.instruction = self.steps[0].instruction  # the trajectory exporter's task text; each step opens its own
        self.step_meta = {s.name: {"agent_timeout_sec": s.agent.timeout_sec if s.agent.timeout_sec is not None else cfg.agent.timeout_sec,
                                   "verifier_timeout_sec": s.verifier.timeout_sec, "min_reward": s.min_reward,
                                   "artifacts": [a if isinstance(a, str) else a.source for a in s.artifacts],
                                   "tests": self._tests_for(s.name), "has_workdir": self.task.paths.step_workdir_dir(s.name).is_dir()}
                          for s in cfg.steps}

    def _step_network_plan(self, step):
        from harbor.models.task.verifier_mode import VerifierEnvironmentMode
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.trial.network_policy import resolve_trial_network_plan

        return resolve_trial_network_plan(self.task.config, TrialAgentConfig(), TrialEnvironmentConfig(), step, verifier_mode=VerifierEnvironmentMode.SHARED)

    def _tests_for(self, name: str) -> str:
        """Which test script Harbor's ``Verifier._resolve_tests`` will run for the step (host path, relative)."""
        os_ = self.task.config.environment.os
        p = self.task.paths.discovered_step_test_path_for(name, os_) or self.task.paths.discovered_test_path_for(os_)
        return str(p.relative_to(self.dir)) if p else ""

    @staticmethod
    def _state(world) -> dict[str, Any]:
        """Per-World step state (the task object is shared by concurrent attempts): the network phase in force and whether
        a shared verifier left files in the agent container."""
        return world.__dict__.setdefault("_harbor_multistep", {"phase": None, "dirty": False})

    async def _set_phase(self, world, policy) -> None:
        st = self._state(world)
        if policy == (st["phase"] or self.plan.agent_env_baseline):
            return
        env = world.backend.env
        if not getattr(getattr(env, "capabilities", None), "dynamic_network_policy", False):
            raise RuntimeError(f"network phase {policy} needs dynamic network policy, which this environment lacks")
        await env.set_network_policy(policy)
        st["phase"] = policy

    def _user(self, step) -> Any:
        return step.agent.user if step.agent.user is not None else (self.task.config.agent.user if self.task.config.agent.user is not None else self.backend_params.get("user"))

    async def _setup_step(self, step, world) -> None:
        env = world.backend.env
        st = self._state(world)
        if st["dirty"]:
            await self._reset_shared_dirs(env)
            st["dirty"] = False
        await self._set_phase(world, self.step_plans[step.name].agent_phase)
        user = self._user(step)
        world.backend.exec_for(world.spec.apps[0]).default_user = user  # the policy's terminal runs as the step's agent user
        with env.with_default_user(user):
            wd = (await env.exec("pwd")).stdout
            workdir = (wd or "/").strip() or "/"
            src = self.task.paths.step_workdir_dir(step.name)
            self.task.paths.validate_input_tree(src)
            if src.is_dir():
                await env.upload_dir(source_dir=src, target_dir=workdir)
            if self.task.paths.step_setup_path(step.name).exists():
                import shlex

                r = await env.exec(f"bash {shlex.quote(workdir.rstrip('/') + '/setup.sh')}")
                if r.return_code != 0:
                    raise RuntimeError(f"step {step.name!r} setup.sh exited {r.return_code}: {(r.stderr or r.stdout or '')[-500:]}")
            if step.healthcheck is not None:
                await env.run_healthcheck(step.healthcheck)
        surf = world.__dict__.get("_grader_surface")
        if surf is not None:  # the step's own setup is the task's, not the policy's: a new baseline for the grader check
            from crucible.interop.harbor import grader_guard

            surf.advance(await grader_guard.diff(world.backend, surf.container))

    async def _grade_step(self, step, world, workdir: Path):
        from crucible.verify.runtime.plan import CheckResult, GradeReport

        t0 = time.monotonic()
        tid = f"{self.id}/{step.name}"
        timeout = float(step.verifier.timeout_sec)
        user = step.verifier.user if step.verifier.user is not None else self.task.config.verifier.user
        plan = self.step_plans[step.name]

        async def phase_in():
            self._state(world)["dirty"] = True  # set before verifying: a failed verifier may still leave files behind
            await self._set_phase(world, plan.verifier_phase)

        async def phase_out():
            await self._set_phase(world, self.plan.agent_env_baseline)

        # Harbor's Verifier with ``step_name`` under the shared guard (liveness, the grader's toolchain, the network:
        # M13 audit B1, B2); a reward-file error after the reset is the agent's (H1 V4)
        result, ended, flags = await self._verify(world, tid=tid, user=user, timeout=timeout, t0=t0, phase=plan.verifier_phase,
                                                  phase_in=phase_in, phase_out=phase_out,
                                                  verifier_kw={"verifier_env": dict(step.verifier.env) or None, "step_name": step.name})
        if ended is not None:
            return ended
        rewards = dict(result.rewards or {})
        reward, problem = scalar_reward(rewards, self.reward_key)
        if reward is None:
            return self._masked(tid, t0, problem.split(":")[0], f"step {step.name!r}: {problem}", flags=flags)
        checks, gates, kinds = [], {}, {}
        if step.min_reward is not None:
            why = min_reward_failure(rewards, step.min_reward)
            gates[MIN_REWARD_GATE] = "pass" if why is None else "fail"
            kinds[MIN_REWARD_GATE] = "outcome"
            checks.append(CheckResult(MIN_REWARD_GATE, MIN_REWARD_GATE, gates[MIN_REWARD_GATE], {"min_reward": step.min_reward, "rewards": rewards, "harbor": why},
                                      provenance="harbor:MultiStepTrial._min_reward_failure"))
        return GradeReport(task=tid, tier="full", checks=checks, gates=gates, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2),
                           meta={"step": step.name, "rewards": rewards, "kinds": kinds, "components": {f"harbor/{k}": float(v) for k, v in rewards.items()},
                                 "flags": flags, "verifier": json.loads(result.model_dump_json())})

    async def grade(self, world, workdir: Path):
        """A multi-step task is graded step by step (``steps``); this grades the last step alone, for callers that
        drive the World without ``NativeHarness``."""
        return await self._grade_step(self.task.config.steps[-1], world, workdir)

    def oracle_actions(self, *, done: str = "done") -> list:
        """Every step's ``solution/`` replayed through the terminal tools: files written under ``/tmp/crucible-oracle/<step>``
        (the World refuses policy access to ``/solution`` and ``/logs/verifier``, ``world/intercept.py``, so a solve script
        that names those paths is refused too), ``solve.sh`` run from the task's working directory
        (``DEBIAN_FRONTEND=noninteractive`` as Harbor's ``OracleAgent``), the directory removed, then ``Done``. One ``Done``
        per step, so a scripted policy drives ``NativeHarness`` with it."""
        from harbor.models.task.config import TaskOS

        from crucible.core.schema.actions import Done, ToolCall

        if self.task.config.environment.os == TaskOS.WINDOWS:
            raise ValueError("oracle_actions replays POSIX solve.sh scripts; use oracle_step for Windows tasks")
        acts: list = []
        for s in self.task.config.steps:
            root = f"/tmp/crucible-oracle/{s.name}"
            sol = self.task.paths.step_solution_dir(s.name)
            solve = self.task.paths.discovered_step_solve_path_for(s.name, self.task.config.environment.os)
            if solve is None:
                raise FileNotFoundError(f"step {s.name!r} has no solution/solve script")
            for f in sorted(p for p in sol.rglob("*") if p.is_file()):
                acts.append(ToolCall(tool_call_id=f"o{len(acts) + 1}", function_name="term.write_file",
                                     arguments={"path": f"{root}/{f.relative_to(sol).as_posix()}", "content": f.read_text()}))
            cmd = f"DEBIAN_FRONTEND=noninteractive bash {root}/{solve.relative_to(sol).as_posix()}; rc=$?; rm -rf {root}; exit $rc"
            acts += [ToolCall(tool_call_id=f"o{len(acts) + 1}", function_name="term.exec", arguments={"command": cmd}), Done(reason=done)]
        return acts

    async def oracle_step(self, world, index: int) -> None:
        """Harbor's ``OracleAgent`` for step ``index`` (uploads ``steps/<name>/solution`` to ``/solution`` and runs it)."""
        from harbor.agents.oracle import OracleAgent
        from harbor.models.agent.context import AgentContext

        tp = self._trial_paths(world)
        agent = OracleAgent(logs_dir=tp.agent_dir, task_dir=self.dir, trial_paths=tp)
        agent._step_index = index
        env = world.backend.env
        with env.with_default_user(self._user(self.task.config.steps[index])):
            await agent.run(self.steps[index].instruction if isinstance(self.steps[index].instruction, str) else "", env, AgentContext())
