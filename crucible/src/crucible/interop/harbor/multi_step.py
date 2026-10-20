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
  records the step as errored and stops; Crucible never scores an environment failure as the policy's 0). M10 lab audit
  R-2: after the first step's setup (before the policy acts) the process baseline is taken again, so what that setup
  started is the task's and spared; a later step's ``setup.sh`` runs in a container a root policy controlled, so the
  processes it starts are never made pristine: the host lists the container before and after it
  (``quiesce.started_since``), and what it started is killed before that step's grade, which then ends as a masked,
  non-retryable ``crucible:harness_unsupported`` of unknown blame (``setup_service``) rather than the policy's 0 (review
  round 1, P16; review round 2: unknown, never the task's, since a root policy can make that setup start a process: a
  tool it runs that the policy replaced, or a process of the policy forking during it). Review round 3: so that an
  honest task whose later setup starts a service never reaches that unknown-blame stop, ``open`` first runs every step's
  setup once in a pristine container (``_preflight_setups``: no policy, once per task object and process) and refuses
  the task (``crucible:harness_unsupported``, harness blame, stage ``seed``) when a later setup leaves a process running
  there; ``setup_service`` then remains only where that preflight left nothing running or was inconclusive (a later
  setup that needs the agent's earlier work fails there). After a clean step grade the listing before the next step's
  setup is taken whether or not that step has a ``setup.sh``: what it shows beyond the baseline is the grade's leftovers
  (its ``test.sh`` left a server running), killed at the next grade and never flagged as the policy's, nor are their
  descendants (``quiesce`` receipt ``leftovers``; review rounds 2 and 3).
- **grade** (Harbor ``_run_step_verifier``, shared mode), after a host-side quiesce (M10 lab audit R-2,
  ``HarborInPlaceTask._quiesce``: every process started after the baseline is killed and confirmed gone, receipt
  ``steps/NN-<name>/verifier/quiesce.json``; survivors are ``crucible:policy_misbehavior``; the step report's
  ``meta.flags``, ``policy_process_killed`` included, reach the rollout through ``NativeHarness._run_steps``): Harbor's ``Verifier`` with
  ``step_name`` (it merges the shared ``tests/`` and ``steps/<name>/tests/`` into ``/tests`` and runs the step's ``test.sh``, else the shared one), the
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

import asyncio
import contextlib
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
    # M10 lab audit R-3: a step's grade uses up its R-2 per-step state (the setup listing, the previous grade's leftovers),
    # so NativeHarness grades each step once and settles a step failure without a control (``grade_control`` is None)
    step_grades_once = True

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

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the in-place declaration (the task tree's digests, which cover ``steps/<name>/``) plus the step
        reward rule and the step names in order. ``self.steps`` holds ``partial`` graders: never described, only named."""
        return {**super().identity_config(), "step_reward": self.step_reward, "steps": [s.name for s in self.steps]}

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
        a shared verifier left files in the agent container; ``graded``: a step grade ran (the policy has acted: M10 lab
        audit R-2 takes no further process baseline); ``setup_procs``: what the current step's ``setup.sh`` started, when
        it is not the first step's (``quiesce.started_since``, M10 lab audit R-2 review round 1), else None; ``swept``: the
        last step grade's quiesce was clean, so every process of the policy's was killed before its Verifier ran (review
        round 2); ``leftovers``: what was beyond the baseline before the current step's setup after such a grade (that
        grade's leftovers: killed at this step's grade, never flagged as the policy's), else None."""
        return world.__dict__.setdefault("_harbor_multistep", {"phase": None, "dirty": False, "graded": False, "setup_procs": None,
                                                               "swept": False, "leftovers": None})

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
        # M10 lab audit R-2: the first step's setup (before any grade) re-takes the baseline below; a later step's setup.sh
        # or healthcheck is listed host-side before and after (review round 1), so what it started is known at its grade.
        # Review round 2: after a clean grade the listing before the setup is taken on every later step (setup.sh or
        # not): that grade's quiesce killed every process of the policy's before its Verifier ran, so what is beyond the
        # baseline now came from the grade (a server its test.sh left running): its ``leftovers``, killed at this step's
        # grade and never flagged as the policy's
        quiesced = self.QUIESCE and "_quiesce_baseline" in world.__dict__
        first = step.name == self.task.config.steps[0].name and not st.get("graded")
        swept, st["swept"] = bool(st.get("swept")), False
        st["setup_procs"] = st["leftovers"] = None
        has_setup = self.task.paths.step_setup_path(step.name).exists() or step.healthcheck is not None
        pre = await self._setup_listing(world) if quiesced and not first and (swept or has_setup) else None
        if swept and pre is not None:
            st["leftovers"] = pre
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
        # M10 lab audit R-3 review round 3: a step grade whose Verifier started and raised (a reward-file error scores the
        # step 0 and the task goes on) left its window pending for a re-grade of the same work (HarborInPlaceTask._guard_window).
        # A step is never graded again (step_grades_once) and the advance below absorbs that window, so it is closed here, its
        # moved-aside tools kept suspect as the normal path keeps them. Judged at the next step's grade instead, against the
        # previous step's pre-Verifier diff, its after-grade relaxations covered this step's own agent phase (a ~/.local/bin
        # shadow, a pre-placed grader-only tool) and the baseline moved past them: an unearned 1.0, never flagged
        pending = world.__dict__.pop("_guard_pending", None)
        if surf is not None:  # the step's own setup is the task's, not the policy's: a new baseline for the grader check
            from crucible.interop.harbor import grader_guard

            surf.advance(await grader_guard.diff(world.backend, surf.container), rejudge=pending[1] if pending is not None else ())
        if quiesced and first:
            # M10 lab audit R-2: the first step's setup ran before the policy acted, from the task's own tree: what it
            # started (a service its tests query) is the task's, so the process baseline is taken again here. Not after a
            # later step's setup (nor the first step's after a grade, for a caller driving the World by hand): ``bash
            # setup.sh`` then runs in a container whose binaries a root policy may have replaced (nothing guards setup's
            # tools), so a writer it spawned would become pristine and grade the step 1.0
            await self._record_baseline(world, stage="agent", close=False)
        elif pre is not None and has_setup:
            # M10 lab audit R-2 review round 1: never added to the baseline. What this setup started (listed now, not
            # before it: the previous verifier's leftovers and their children drop out) is killed at the step's grade like
            # any target, and if it still ran then, the grade ends as a mask of unknown blame (``quiesce`` verdict
            # ``setup_service``; review round 2: a root policy can make the setup start one)
            st["setup_procs"] = await self._setup_listing(world, pre)

    async def _setup_listing(self, world, before=None):
        """``quiesce.started_since`` against the World's baseline (host-only); a host that cannot list the container
        raises ``StageFailure`` (``crucible:backend_error``, as ``_record_baseline``) (M10 lab audit R-2 review round 1)."""
        from crucible.core.schema.run import Failure
        from crucible.interop.harbor import quiesce
        from crucible.runtime.infra import StageFailure

        try:
            return await quiesce.started_since(world.backend, world.__dict__["_quiesce_baseline"], before)
        except quiesce.QuiesceError as e:
            raise StageFailure(Failure(kind="crucible:backend_error", stage="agent", detail="quiesce_setup_listing",
                                       reason=f"{self.id}: the container could not be listed from the host around a step's setup "
                                              f"(M10 lab audit R-2): {e}")) from e

    async def open(self, workdir: Path, *, before_agent=None):
        """``HarborInPlaceTask.open``, after the task's pristine setup preflight (M10 lab audit R-2 review round 3,
        ``_preflight_setups``): a task whose later step's ``setup.sh`` (or healthcheck) leaves a process running in a
        container no policy touched is refused before any container for the policy starts and before any spend:
        ``StageFailure`` of ``crucible:harness_unsupported`` (harness blame, stage ``seed``, never retried), counted only
        toward the run's mask stop, never its 5% unknown-blame stop. The task's own setup service is the task's property,
        while a setup process at a grade (``setup_service``, unknown blame) stays reserved for what a policy can induce.
        An installed-agent harness's ``before_agent`` on a ``[[steps]]`` task is refused by ``HarborInPlaceTask.open``
        before any container starts, so no preflight runs for it."""
        if before_agent is None and self.QUIESCE:
            pre = await self._preflight_setups(Path(workdir).parent / "setup_preflight")
            bad = {name: procs for name, procs in pre.items() if procs}
            if bad:
                from crucible.core.schema.run import Failure
                from crucible.runtime.infra import StageFailure

                what = "; ".join(f"step {name!r}: " + ", ".join(f"{e['pid']} {e['args'][:80]}" for e in procs[:4]) for name, procs in bad.items())
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False, detail="quiesce_setup_service_task",
                                           reason=f"{self.id}: a later step's setup.sh left process(es) running in a pristine container before any "
                                                  "policy ran; the in-place quiesce must kill them before that step's grade (a root policy "
                                                  "controlled the container before that setup ran, so they cannot be spared), so no grade of that "
                                                  "step would be the policy's: the task cannot be graded in place (M10 lab audit R-2 review round 3): "
                                                  + what))
        return await super().open(workdir, before_agent=before_agent)

    async def _preflight_setups(self, workdir: Path) -> dict[str, list[dict] | None]:
        """M10 lab audit R-2 review round 3: what each later step's setup leaves running in a pristine container (no
        policy ever ran in it), as ``{step name: [host listing entries] | None}``; ``{}`` when no later step has a
        ``setup.sh`` or a healthcheck (no container is started). Every step's setup runs in declared order through
        ``_setup_step`` (the first one's re-takes the baseline; each later one is listed host-side before and after:
        ``setup_procs``), in a World ``HarborInPlaceTask.open`` builds and that is closed afterwards. A step whose setup or
        healthcheck raises makes it and every later step ``None`` (inconclusive: a later setup may need the agent's earlier
        work); a ``StageFailure`` (the host could not list the container) propagates and nothing is kept. Run once per
        task object, under a lock: the task is shared by concurrent attempts. Created lazily, so a task built without
        ``__init__`` (the unit tests' fakes) has it too."""
        from crucible.runtime.infra import StageFailure

        lock = self.__dict__.setdefault("_preflight_lock", asyncio.Lock())
        async with lock:
            done = self.__dict__.get("_setup_preflight")
            if done is not None:
                return done
            steps = list(self.task.config.steps)
            if not any(self.task.paths.step_setup_path(s.name).exists() or s.healthcheck is not None for s in steps[1:]):
                self.__dict__["_setup_preflight"] = {}
                return {}
            out: dict[str, list[dict] | None] = {}
            w = await HarborInPlaceTask.open(self, Path(workdir))
            try:
                broken = False
                for i, step in enumerate(steps):
                    if not broken:
                        try:
                            await self._setup_step(step, w)
                        except StageFailure:
                            raise
                        except Exception:  # noqa: BLE001 -- a setup or healthcheck that fails without the agent's work: inconclusive
                            broken = True
                    if i:
                        procs = self._state(w)["setup_procs"]  # None for a later step without setup.sh or healthcheck
                        out[step.name] = None if broken else ([*procs.pristine.values()] if procs is not None else [])
            finally:
                with contextlib.suppress(Exception):
                    await w.close()
            self.__dict__["_setup_preflight"] = out
            return out

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
        # M10 lab audit R-2: a grade, whatever its outcome: the policy has acted on this World (``_setup_step`` takes no
        # further baseline). The step's quiesce receipt goes next to its grade dir (``steps/NN-<name>/verifier/quiesce.json``);
        # what a later step's setup.sh started is passed on (review round 1: ``setup_service``, never the policy's 0), and
        # so is what the previous step's grade left running (review round 2: ``leftovers``, killed, never flagged)
        st = self._state(world)
        st["graded"] = True
        result, ended, flags, q = await self._verify(world, tid=tid, user=user, timeout=timeout, t0=t0, phase=plan.verifier_phase,
                                                     phase_in=phase_in, phase_out=phase_out,
                                                     verifier_kw={"verifier_env": dict(step.verifier.env) or None, "step_name": step.name},
                                                     receipt_dir=Path(workdir).parent / "verifier", setup=st.get("setup_procs"),
                                                     leftovers=st.get("leftovers"))
        # M10 lab audit R-2 review round 2: a clean quiesce here killed every process of the policy's before the Verifier
        # ran, so whatever is beyond the baseline at the next step's setup came from this grade (``_setup_step``)
        st["swept"] = (q or {}).get("verdict") == "clean"
        if ended is not None:
            if q is not None:
                ended.meta.setdefault("quiesce", q)
            if flags:  # M10 lab audit R-2 review round 2: a _liveness report replacing the grade carries no flags; the quiesce's policy_process_killed rides it too
                ended.meta.setdefault("flags", flags)
            return ended
        rewards = dict(result.rewards or {})
        reward, problem = scalar_reward(rewards, self.reward_key)
        if reward is None:
            return self._masked(tid, t0, problem.split(":")[0], f"step {step.name!r}: {problem}", flags=flags, quiesce=q)
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
                                 "flags": flags, "verifier": json.loads(result.model_dump_json()), "quiesce": q})

    async def grade(self, world, workdir: Path):
        """A multi-step task is graded step by step (``steps``); this grades the last step alone, for callers that
        drive the World without ``NativeHarness``."""
        return await self._grade_step(self.task.config.steps[-1], world, workdir)

    async def grade_control(self, workdir: Path):
        """M10 lab audit R-3: no control. A step's grade uses up its per-step state (R-2's setup listing, the previous
        grade's leftovers; ``step_grades_once``), so ``NativeHarness`` grades each step once and masks an answer-sensitive
        step failure with flag ``no_control``; the in-place control this class would inherit
        (``HarborInPlaceTask.grade_control``) grades only the last step and is never run."""
        return None

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
