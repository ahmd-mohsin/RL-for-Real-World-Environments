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
  (``ambiguous_reward``) rather than invent a scalar. Every key is kept in ``components`` (``harbor/<key>``).
- **the agent shares the grader's container** (M11 audit): the policy's processes are killed first (M10 lab audit R-2,
  below), then ``/logs/verifier`` and ``/tests`` are emptied before the grade (Harbor's multi-step reset); the agent's
  own network phase is applied after ``open``, and tasks whose verifier runs in a separate environment are refused.
  After the agent ran, a container in which the verifier user can no longer write its reward (``agent_broke_env``) or a
  verifier that writes no valid reward is the agent's outcome, scored 0; a timeout, an upload failure or a lost
  container masks.
- **the grader's own toolchain** (M13 audit B1, ``grader_guard``): captured from the pristine container at ``open`` and
  checked host-side (``docker diff`` / ``docker cp``) before and after the verifier. A policy that shadowed, replaced or
  hooked a tool the verifier runs is ``crucible:grader_tampered`` (scored 0, flagged ``grader_tampered``, never 1.0);
  the verifier runs with the pristine ``PATH`` and a fresh ``PYTHONPYCACHEPREFIX``. Environments the host cannot inspect
  (non-Docker providers) are refused in place outright (M10 lab audit R-2: ``grader_integrity="off"`` no longer admits
  them; grade them in patch mode).
- **no grade while a policy process can run** (M10 lab audit R-2, ``quiesce``; P23): ``open`` records a process baseline
  after start, reset and healthcheck and before ``before_agent``, the grader capture or the policy (PID 1's host pid, a
  start mark, the host's ``docker top``); every grade (``grade``, each ``HarborMultiStepTask`` step, and so R-1's
  shared-verifier seam, which calls ``grade``) first kills every process started after it from the host, confirms with
  the host's frozen listings (``docker pause``, ``docker top``, ``docker unpause``: a fork-and-die chain is listed too),
  and writes ``<workdir>/../verifier/quiesce.json`` (next to the grade dir: ``<attempt or step>/verifier``). Only a clean
  receipt reaches ``_reset_shared_dirs``, the toolchain check and Harbor's ``Verifier``. A process still listed after
  the bounded rounds is ``crucible:policy_misbehavior`` (scored 0, flag ``policy_process_survived``) once the host shows
  the container alive; one killed and confirmed gone is not blamed but flagged ``policy_process_killed`` (non-scoring:
  the reward is kept, but a passing attempt counts in bench's hack ``rate`` as a ``flagged_passes`` and studio skips it;
  the R-2 Acceptance 1 reading in the ``quiesce`` docstring, chosen by the user on 2026-10-17); a host that cannot list or
  drive the container, or a helper exec the daemon never started, masks ``crucible:backend_error`` (``_liveness``
  first); only children a pristine service respawned, or only processes in uninterruptible sleep (never paused for: a
  task in state D cannot be frozen, review round 2), end the attempt as a non-retryable ``crucible:backend_error`` of
  unknown blame (a policy can induce a respawn; the 5% unknown-blame stop bounds it; a masked grade would be retried,
  re-running the agent: P16); a later multi-step step's ``setup.sh`` that left a process running ends it as a
  non-retryable ``crucible:harness_unsupported`` of unknown blame (review round 2: a root policy can make that setup
  start one; review round 3: a task whose later setup leaves a process running in a pristine preflight, before any
  policy, is refused at ``HarborMultiStepTask.open`` with harness blame instead). A World without a
  baseline is ``crucible:harness_unsupported``; so is an environment the host cannot list (``environment_type`` other
  than ``docker``), refused at ``open`` before any container starts, whatever ``grader_integrity`` is. Accepted
  trade-off: a task whose tests need a process the agent left running fails. Residual: a process the policy's code
  under test starts during the Verifier itself (the receipt covers what runs when the grade starts).
- **infrastructure is told apart from the policy by host-side evidence** (M13 audit B2): a container that is gone,
  stopped through the daemon API, or whose daemon cannot be reached masks ``crucible:env_container_lost`` (mid-episode
  too: the terminal raises instead of returning compose's "service is not running" as the policy's failed command); a
  container that exited from inside, or one alive whose verifier user cannot work, is ``agent_broke_env``. A grade whose
  test script needs the network fails while the host shows the container detached or the network unreachable: masked
  ``crucible:grader_network_error``.
- **installed agents** (M10 lab audit R-1, ``interop/harbor/agent_harness.py``): the capture proxy's host reaches the
  agent phase only through Harbor's ``TrialAgentConfig.extra_allowed_hosts`` (``agent_allowed_hosts=``); the plan was
  built from a bare ``TrialAgentConfig()``, so no proxy host could pass a restricted agent phase (inplace.py:148-155).
  ``open(before_agent=...)`` runs the harness's agent setup (the install) under the World's baseline network, before the
  grader surface is captured and before the agent phase is applied; on a ``[[steps]]`` task.toml (``multi_step``, whose
  agent phases ``open`` never applies) it is refused before any container starts. ``admits_agent_host``, ``multi_step``,
  ``agent_user``, ``agent_timeout_s`` and ``agent_logs`` are the harness's view of the task. Separate-verifier tasks stay refused here;
  ``patch_task.HarborPatchTask`` grades them from a collected patch in a fresh container.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
import uuid
from collections.abc import Awaitable, Callable, Sequence
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


from crucible.interop.harbor.rewards import scalar_reward

# reward-file errors after the agent ran in the same container: its outcome, never a mask (M11 audit H1 V4). The grader
# alone wrote there: the policy's processes were killed and confirmed gone and the dirs emptied first (M10 lab audit R-2;
# before it a detached writer of the policy re-created reward.json after the reset)
REWARD_FILE_ERRORS = ("RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError")


class HarborInPlaceTask:
    SEPARATE_OK = False  # a separate-verifier task is refused in place (``patch_task.HarborPatchTask`` sets True)
    # M10 lab audit R-2: every grade in the agent's container is preceded by a host-side quiesce against the baseline
    # ``open`` records, and an environment the host cannot list is refused at ``open``. ``HarborPatchTask`` sets False:
    # it never grades in the agent's container (its ``grade`` is refused), and a non-Docker provider may host its agent
    QUIESCE = True

    def __init__(self, task_dir: Path, *, docker_host: str | None = None, gpu_host: str | None = None, reward_key: str | None = None,
                 environment_type: str = "docker", environment_kwargs: dict | None = None, user: str | None = None, keep_containers: bool = False,
                 grader_integrity: str = "verify", agent_allowed_hosts: Sequence[str] = ()):
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode

        # hosts the agent phase must admit besides the task's own (the capture proxy of an installed agent); read by
        # ``_network_plan``, so assigned before it runs
        self.agent_allowed_hosts = tuple(str(h) for h in agent_allowed_hosts)
        if grader_integrity not in ("verify", "off"):
            raise ValueError("grader_integrity: 'verify' | 'off'")
        if grader_integrity == "verify" and environment_type != "docker":
            # the grader runs in the agent's container; only a host-side Docker CLI can tell whether the policy changed the
            # grader's toolchain (M13 audit B1). Without it a stubbed pytest would grade 1.0 unseen. M10 lab audit R-2: nor
            # can the host kill the policy's processes there, so ``open`` refuses such an environment in place whatever
            # grader_integrity is; patch mode is the only way to grade it
            raise ValueError(f"{task_dir}: in-place grading on {environment_type!r} cannot verify the grader's toolchain nor quiesce the "
                             "policy's processes (needs a host-side Docker CLI), so it is refused in place outright (M10 lab audit R-2); "
                             "grade it in patch mode (verify_dir / HarborPatchTask)")
        self.grader_integrity = grader_integrity
        self.dir = Path(task_dir).resolve()
        self.task = Task(self.dir)
        cfg = self.task.config
        if not cfg.steps and not self.SEPARATE_OK and resolve_task_verifier_mode(cfg) == VerifierEnvironmentMode.SEPARATE:
            # Harbor grades it in a fresh container after the collect hooks, with only the declared artifacts; grading it in
            # the agent's container lets a plant score and a real solve fail (M11 audit H1 V2, G1 D4, G5)
            raise ValueError(f"{self.dir}: the verifier runs in a separate environment ([verifier].environment_mode = \"separate\"), "
                             "which the in-place World (one container) cannot host; grade it in patch mode (verify_dir) or on Harbor")
        # the full name keeps two orgs' same-named tasks apart (H1 V11: ``short_name`` drops the org)
        name = getattr(self.task, "name", None) or self.dir.name
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
        phases = [x for x in self._declared_phases() if x != self.plan.agent_env_baseline]
        if phases:  # the environment must be able to switch policy per phase (Harbor egress control)
            self.backend_params["phase_network_policies"] = [x.model_dump(mode="json") for x in phases]

    def _declared_phases(self) -> tuple[Any, ...]:
        """The network phases this container must be able to switch to (the agent's, and the verifier's in place)."""
        return (self.plan.agent_phase, self.plan.verifier_phase)

    def _network_plan(self):
        from harbor.models.task.verifier_mode import resolve_task_verifier_mode
        from harbor.models.trial.config import AgentConfig as TrialAgentConfig
        from harbor.models.trial.config import EnvironmentConfig as TrialEnvironmentConfig
        from harbor.trial.network_policy import resolve_trial_network_plan

        # no extra hosts: a bare TrialAgentConfig(), exactly as before. Extra hosts are the 0.24 API (trial/network_policy.py:
        # 105-118); an older Harbor that refuses the field, or silently drops it, fails here rather than run the agent phase
        # without the proxy host (M10 lab audit R-1)
        try:
            agent_cfg = TrialAgentConfig(**({"extra_allowed_hosts": list(self.agent_allowed_hosts)} if self.agent_allowed_hosts else {}))
            if self.agent_allowed_hosts and getattr(agent_cfg, "extra_allowed_hosts", None) is None:
                raise TypeError("TrialAgentConfig has no extra_allowed_hosts")
        except (TypeError, ValueError) as e:
            raise ValueError(f"{self.dir}: this Harbor ({_harbor_version()}) cannot admit agent-phase hosts {list(self.agent_allowed_hosts)}: "
                             f"extra_allowed_hosts is the 0.24 API (trial/network_policy.py:105-118): {e}") from e
        return resolve_trial_network_plan(self.task.config, agent_cfg, TrialEnvironmentConfig(), None,
                                          verifier_mode=resolve_task_verifier_mode(self.task.config))

    # ---- the installed-agent harness's view of the task (M10 lab audit R-1) ---------------------------------------
    def admits_agent_host(self, host: str | None) -> bool:
        """Whether the agent phase lets the container reach ``host``: a public phase, or an allowlist naming it exactly
        (a wildcard or CIDR entry is not matched here: the harness refuses rather than guess)."""
        if not host:
            return False
        phase = self.plan.agent_phase
        mode = getattr(phase.network_mode, "value", str(phase.network_mode))
        if mode == "public":
            return True
        if mode != "allowlist":
            return False
        return host.strip().casefold() in {str(h).strip().casefold() for h in (phase.allowed_hosts or [])}

    @property
    def multi_step(self) -> bool:
        """Whether task.toml declares ``[[steps]]``, whichever adapter loaded it. ``open`` then leaves every agent phase to
        the per-step setup (``HarborMultiStepTask``), so a plain adapter on such a dir applies none; the harness refuses
        it rather than run an installed agent under the baseline network (M10 lab audit R-1)."""
        return bool(self.task.config.steps)

    @property
    def agent_user(self) -> str | int | None:
        return self.task.config.agent.user

    @property
    def agent_timeout_s(self) -> float | None:
        t = self.task.config.agent.timeout_sec
        return float(t) if t else None

    @staticmethod
    def _logs_bound(world) -> bool:
        """``/logs/{agent,verifier,artifacts}`` are bind-mounted to the trial dir only for a local Docker daemon
        (``world_backend.provision``: not on a remote ``DOCKER_HOST`` or a non-Docker provider). Decided from Crucible's
        own backend params, never from ``env.capabilities.mounted``: Harbor's Docker environment always declares True, so
        on the lab nothing would ever be downloaded (M10 lab audit R-1). The backend copies a ``gpu_host``'s daemon into
        ``params["docker_host"]`` at provision."""
        params = getattr(world.backend, "params", None) or {}
        return not (params.get("docker_host") or params.get("environment_type", "docker") != "docker")

    async def agent_logs(self, world, *, fetch: bool = False) -> Path:
        """The host directory of the agent's ``/logs/agent`` (an installed agent's ``logs_dir``). ``fetch``: bring it up to
        date after the agent ran: downloaded when the dirs are not bound (the lab), as Harbor's trial does for non-mounted
        environments (trial.py:669-690); when bound, Harbor's ``prepare_logs_for_host`` (chown) is run, best effort."""
        tp = self._trial_paths(world)
        env = world.backend.env
        if fetch:
            if self._logs_bound(world):
                prep = getattr(env, "prepare_logs_for_host", None)
                if prep is not None:
                    with contextlib.suppress(Exception):
                        await prep()
            else:
                await env.download_dir(str(self._env_paths().agent_dir), tp.agent_dir)
        return tp.agent_dir

    def _world_network(self) -> tuple[str, list[str]]:
        base = self.plan.agent_env_baseline
        mode = getattr(base.network_mode, "value", str(base.network_mode))
        return {"no-network": "no-network", "allowlist": "allowlist"}.get(mode, "public"), list(base.allowed_hosts or [])

    def _agent_public(self) -> bool:
        return getattr(self.plan.agent_phase.network_mode, "value", str(self.plan.agent_phase.network_mode)) not in ("no-network", "allowlist")

    def spec(self, workdir: Path):
        from crucible.core.schema.world import WorldSpec

        net, hosts = self._world_network()
        app = {"id": "term", "template": "harbor/terminal", "state": {"adapter": "process", "params": {}},
               "params": {"workdir": self.workdir_in_container} if self.workdir_in_container else {}}
        gpus = int(self.task.config.environment.gpus or 0)
        if gpus:
            app["resources"] = {"gpus": gpus}
        doc = {"world": self.world_id, "backend": "harbor", "backend_params": self.backend_params, "network": net, "apps": [app],
               # internet is part of a task whose *agent* phase is public: only the verifier-path rule applies (m13 C-4); a
               # public baseline with a restricted agent phase keeps the fetch rule (H1 V3)
               **({"intercept": ["grader_paths"]} if net == "public" and self._agent_public() else {}),
               "actors": [{"id": "agent", "kind": "policy", "tools": ["term.*"]}],
               "limits": {"tool_timeout_s": 3600}}
        if net == "allowlist":
            doc["allowed_hosts"] = hosts
        return WorldSpec.model_validate(doc)

    async def open(self, workdir: Path, *, before_agent: Callable[[Any], Awaitable[None]] | None = None):
        """A started, reset, healthy World in the agent's network phase. ``before_agent(world)``: the installed-agent
        harness's setup (M10 lab audit R-1). It runs under the World's baseline network (installs need it), before the
        grader surface is captured (an install must not read as grader tampering) and before the restricted agent phase;
        if it raises, the World is closed and the error propagates. Refused on a ``[[steps]]`` task before any container
        starts: its agent phases are applied per step, never here, so the install would be followed by an agent run under
        the baseline network (M10 lab audit R-1). M10 lab audit R-2: an environment the host cannot list is refused
        (``crucible:harness_unsupported``) before any container starts, and the process baseline every grade quiesces
        against is recorded after the healthcheck and before ``before_agent``, so nothing an installed agent or the policy
        starts is ever spared."""
        from crucible.core.store import Store
        from crucible.world.world import World

        if self.QUIESCE and self.backend_params.get("environment_type", "docker") != "docker":
            from crucible.core.schema.run import Failure
            from crucible.runtime.infra import StageFailure

            # M10 lab audit R-2 (P23): only a host-side Docker CLI can list and kill the processes in the agent's container,
            # and Harbor's verifier runs there; on any other provider a detached writer of the policy would grade itself.
            # Refused whatever grader_integrity is (it was admitted with grader_integrity="off")
            raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="seed", retryable=False, detail="quiesce_unsupported",
                                       reason=f"{self.dir}: in-place grading on {self.backend_params.get('environment_type')!r} cannot kill the "
                                              "policy's processes before the grade (M10 lab audit R-2, P23: needs a host-side Docker CLI); "
                                              "grade it in patch mode (verify_dir / HarborPatchTask)"))
        if before_agent is not None and self.multi_step:
            raise ValueError(f"{self.dir}: task.toml declares [[steps]], whose agent phases are applied per step "
                             "(HarborMultiStepTask), which no installed-agent harness drives")
        workdir = Path(workdir)
        w = World(self.spec(workdir), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", backend="harbor",
                  backend_params=self.backend_params)
        await w.start()
        await w.reset()
        await self._healthcheck(w)
        if self.QUIESCE:  # M10 lab audit R-2: before an install, the grader capture or the policy; the container is trusted now
            await self._record_baseline(w)
        if before_agent is not None:
            try:
                await before_agent(w)
            except BaseException:
                with contextlib.suppress(Exception):
                    await w.close()
                raise
        if self.grader_integrity == "verify":
            await self._capture_grader(w, workdir)
        if not self.task.config.steps:
            await self._apply_agent_phase(w)
        return w

    def grader_scripts(self) -> list[Path]:
        """The task's test scripts (shared and per step): the commands they run are the grader's toolchain."""
        out = []
        for p in sorted(self.dir.rglob("*")):
            rel = p.relative_to(self.dir).parts
            if p.is_file() and "tests" in rel[:-1] and (p.suffix in (".sh", ".bash") or (not p.suffix and p.read_bytes()[:2] == b"#!")):
                out.append(p)
        return out

    async def _capture_grader(self, world, workdir: Path) -> None:
        """The grader surface of the pristine container (before the policy acts): ``grader_guard.capture``."""
        from crucible.interop.harbor import grader_guard

        try:
            surf = await grader_guard.capture(self, world)
            if surf is None:
                raise RuntimeError("this environment has no host-side Docker CLI to verify the grader's toolchain")
        except BaseException:
            await world.close()
            raise
        world.__dict__["_grader_surface"] = surf
        grader_guard.save(surf, Path(workdir))

    # ---- the host-side quiesce (M10 lab audit R-2) -------------------------------------------------------------------
    async def _record_baseline(self, world, *, stage: str = "seed", close: bool = True) -> None:
        """The process baseline every grade quiesces against (``quiesce.record_baseline``), kept on the World. Taken only
        while the container is trusted: in ``open`` before the policy, an installed agent or the grader capture runs
        (``HarborMultiStepTask`` takes it again after its first step's setup). A host that cannot find, inspect, exec into
        or list the container raises ``StageFailure``: ``crucible:harness_unsupported`` (not retried) when runc cannot
        start ``sh`` in the image, else ``crucible:backend_error``; ``close``: the World is closed first (``open``'s
        pattern for ``_capture_grader``), as is on any other exception, which propagates."""
        from crucible.core.schema.run import Failure
        from crucible.interop.harbor import quiesce
        from crucible.runtime.infra import StageFailure

        try:
            base = await quiesce.record_baseline(world.backend)
        except quiesce.QuiesceError as e:
            if close:
                with contextlib.suppress(Exception):
                    await world.close()
            if e.unsupported:
                raise StageFailure(Failure(kind="crucible:harness_unsupported", stage=stage, retryable=False, detail="quiesce_baseline",
                                           reason=f"{self.id}: the image has no shell the host can run to quiesce the policy's processes "
                                                  f"before an in-place grade (M10 lab audit R-2): {e}")) from e
            raise StageFailure(Failure(kind="crucible:backend_error", stage=stage, detail="quiesce_baseline",
                                       reason=f"{self.id}: the process baseline could not be recorded from the host (M10 lab audit R-2): {e}")) from e
        except BaseException:
            if close:
                with contextlib.suppress(Exception):
                    await world.close()
            raise
        world.__dict__["_quiesce_baseline"] = base

    async def _quiesce(self, world, *, tid: str, t0: float, receipt_dir: Path | None, setup=None, leftovers=None):
        """M10 lab audit R-2 (P23): kill every process started after the baseline, confirmed by the host, before anything
        of the grade runs. Returns ``(None, summary, flags)`` for a clean receipt (``flags``: ``policy_process_killed``
        when the quiesce killed processes the policy left running: non-scoring, the grade runs), else ``(GradeReport,
        summary, [])``: the grade ends there and nothing was reset or verified. The receipt goes to
        ``receipt_dir/quiesce.json`` (``<attempt or step>/verifier``; a failed write never changes the verdict). ``setup``:
        what a later multi-step step's ``setup.sh`` started (``quiesce.started_since``); ``leftovers`` (review round 2):
        what the previous step's clean grade left running, listed before this step's setup (killed, never flagged).
        Verdicts (``quiesce`` module docstring; only the host's listing decides, the in-container helper's output is
        recorded only):

        - ``survivors``: ``crucible:policy_misbehavior``, scored 0, flag ``policy_process_survived``, once ``_liveness``
          shows the container alive (paused, stopped from outside or removed while the rounds ran masks instead; review
          round 3: a report ``_liveness`` put in its place keeps the flag);
        - ``error`` (the host could not list or drive the container, or no helper exec reached it): ``_liveness``, else
          masked ``crucible:backend_error``;
        - ``setup_service`` (a later step's ``setup.sh`` left a process running, killed now): ``_liveness``, else
          ``StageFailure`` of ``crucible:harness_unsupported`` with unknown blame (review round 2: a root policy can make
          that setup start one, so the 5% unknown-blame stop bounds it; review round 3: reached only where the task's
          pristine setup preflight left nothing running or was inconclusive, ``HarborMultiStepTask.open`` refusing the
          rest with harness blame), not retried, decided before the Verifier so it never depends on the outcome (masking
          only failed grades would inflate pass rates);
        - ``respawned`` / ``uninterruptible`` (blame undecidable from the host: a pristine service or the policy can start
          the chain; review round 3: a pause refused while a target read D, ``Receipt.pause_refused``, is named in the
          reason): ``_liveness``, else ``StageFailure`` of ``crucible:backend_error`` with unknown blame, not retried:
          a masked grade goes through ``classify_verifier``, which keeps it retryable, and a retry re-runs the agent, so a
          policy that set a pristine service crash-looping would turn its 0 into a fresh sample (P16; R-1's precedent,
          agent_harness.py:898-904);
        - no baseline on the World (``open`` did not run): masked ``crucible:harness_unsupported``, never unguarded."""
        from crucible.core.schema.run import Failure
        from crucible.interop.harbor import quiesce
        from crucible.runtime.infra import StageFailure

        base = world.__dict__.get("_quiesce_baseline")
        if base is None:
            return self._masked(tid, t0, "crucible:harness_unsupported", "no process baseline was recorded (open() did not run on this World): "
                                "an in-place grade without the quiesce is never run (M10 lab audit R-2)"), None, []
        receipt = await quiesce.quiesce(world.backend, base, setup=setup, leftovers=leftovers)
        path = None
        if receipt_dir is not None:
            try:
                path = quiesce.save(receipt, Path(receipt_dir))
            except OSError:  # the receipt is evidence; failing to write it never changes the verdict
                path = None
        summary = receipt.summary(path)
        if receipt.verdict == "clean":
            # M10 lab audit R-2 review round 1: a process the policy left running, killed and confirmed gone, is never
            # dropped: the non-scoring flag ``policy_process_killed`` (quiesce module docstring, "Acceptance 1")
            return None, summary, [f] if (f := quiesce.killed_flag(receipt)) else []

        async def alive_or(rep):
            dead = await self._liveness(world, tid, t0)
            rep = dead or rep
            if rep is not None:
                rep.meta.setdefault("quiesce", summary)
            return rep

        if receipt.verdict == "survivors":
            left = receipt.left
            what = "; ".join(f"{s['pid']} {s['args'][:120]}" for s in left[:8])
            sflags = [quiesce.flag(receipt)]
            rep = await alive_or(self._zero(tid, t0, "crucible:policy_misbehavior",
                                            f"{len(left)} process(es) the policy started still ran after {receipt.rounds} host-side kill "
                                            f"rounds, so the grade did not start (M10 lab audit R-2, P23): {what}",
                                            flags=sflags, quiesce=summary))
            # M10 lab audit R-2 review round 3: a report _liveness put in its place (the container exited from inside, or
            # was lost, after the rounds) keeps the host-listed survivor's flag, as grade / _grade_step keep the clean
            # path's policy_process_killed (review round 2)
            rep.meta.setdefault("flags", sflags)
            return rep, summary, []
        if receipt.verdict == "error":
            return await alive_or(self._masked(tid, t0, "crucible:backend_error",
                                               f"{tid}: the container could not be quiesced from the host (M10 lab audit R-2): {receipt.error}",
                                               quiesce=summary)), summary, []
        dead = await alive_or(None)
        if dead is not None:
            return dead, summary, []
        if receipt.verdict == "setup_service":
            # M10 lab audit R-2 review round 1 (P16): the step's own setup.sh is task code; its processes cannot be spared
            # (a root policy controlled the container before that setup ran), so the grade would not be the policy's.
            # M10 lab audit R-2 review round 2: unknown blame, never the task's, as ``respawned`` below: a root policy can
            # make that setup start a process (a tool it runs that the policy replaced, or a process of the policy forking
            # during it), so the run's 5% unknown-blame stop (runtime/infra.py:196-205) bounds it like any mask a policy
            # can choose to cause. M10 lab audit R-2 review round 3: reached only where the task's pristine setup preflight
            # (``HarborMultiStepTask.open``: every setup once before any policy) left nothing running or was inconclusive;
            # a task whose later setup leaves a service there is refused at open with harness blame, so an honest task
            # never counts toward that stop
            entries = "; ".join(f"{s['pid']} {s['args'][:80]}" for s in receipt.setup[:6])
            pre = self.__dict__.get("_setup_preflight")
            preflight = ("not run" if pre is None else ", ".join(f"{n}: {'clean' if v == [] else 'inconclusive'}" for n, v in pre.items())
                         or "no later setup")
            raise StageFailure(Failure(kind="crucible:harness_unsupported", stage="verification", retryable=False, blame="unknown", detail="quiesce_setup_service",
                                       reason=f"{tid}: step setup.sh left {len(receipt.setup)} process(es) running that the in-place quiesce must "
                                              "kill before the grade (they cannot be spared: a root policy controlled the container before that "
                                              "setup ran), so the grade would not be the policy's; what a later step's setup.sh started cannot be "
                                              "told from the host from what a root policy made it start (a tool it runs that the policy replaced: "
                                              "only the test scripts' tools are guarded; or a process of the policy forking during the setup), so "
                                              f"the blame is unknown (M10 lab audit R-2, P16; the task's pristine setup preflight: {preflight}; "
                                              f"receipt {path}): " + entries))
        if receipt.pause_refused:
            # M10 lab audit R-2 review round 3: the quiesce ended at a refused pause (quiesce.Receipt.pause_refused)
            what = ("a docker pause the freezer refused while a target was in state D may have left the container's cgroup "
                    f"freezing: {receipt.pause_refused}")
        elif receipt.verdict == "respawned":
            what = "only children a pristine service started again after each kill remain"
        else:
            what = "only processes in uninterruptible sleep (state D) remain after the extra wait"
        raise StageFailure(Failure(kind="crucible:backend_error", stage="verification", retryable=False, blame="unknown", detail=f"quiesce_{receipt.verdict}",
                                   reason=f"{tid}: the container could not be quiesced before the grade: {what} (M10 lab audit R-2; receipt "
                                          f"{path}): " + "; ".join(f"{s['pid']} {s['args'][:80]}" for s in receipt.survivors[:6])))

    async def infra_cause(self, world):
        """Host-side: why an episode the World masked was the infrastructure's (the container is gone, was stopped from
        outside, or its daemon is unreachable): a typed ``Failure``, or None (M13 audit B2)."""
        from crucible.core.schema.run import Failure

        lost = getattr(world.backend, "container_lost", None)
        why = await lost() if lost is not None else None
        return Failure(kind="crucible:env_container_lost", stage="agent", reason=why) if why else None

    async def _apply_agent_phase(self, world) -> None:
        """The agent's own network phase (H1 V3: only the verifier phase was ever applied, so ``[agent] network_mode =
        "no-network"`` on a public baseline ran the agent with the internet). Fails closed."""
        plan = self.plan
        if plan.agent_phase == plan.agent_env_baseline:
            return
        env = world.backend.env
        if not getattr(getattr(env, "capabilities", None), "dynamic_network_policy", False):
            await world.close()
            raise RuntimeError(f"the agent phase needs network {plan.agent_phase} but this environment cannot change its policy")
        await env.set_network_policy(plan.agent_phase)

    def _env_paths(self):
        from harbor.models.trial.paths import EnvironmentPaths

        return EnvironmentPaths.for_os(self.task.config.environment.os)

    async def _reset_shared_dirs(self, env) -> None:
        """Harbor ``_reset_shared_step_verifier_dirs``: empty ``/logs/verifier`` and ``/tests`` in the agent container, so
        the grader reads only what it writes (H1 V1: a planted ``reward.json`` or ``/tests/conftest.py`` scored 1). In a
        grade it runs only after a clean quiesce receipt (M10 lab audit R-2): a detached writer of the policy still
        running re-created ``reward.json`` after it, and Harbor reads that before ``reward.txt``."""
        paths = self._env_paths()
        await env.empty_dirs([paths.verifier_dir], chmod=True)
        await env.empty_dirs([paths.tests_dir], chmod=False)

    async def _probe(self, env, user) -> str | None:
        """Can the verifier user still run a shell and write its reward? None when yes, else what is broken (the agent
        broke its own container: scored 0). An exception (the container is gone) propagates: infrastructure."""
        paths = self._env_paths()
        f = f"{paths.verifier_dir}/.crucible_probe"
        with env.with_default_user(user):
            r = await env.exec(f"sh -c 'echo ok > {f} && cat {f} && rm -f {f}'")
        if r.return_code != 0 or "ok" not in (r.stdout or ""):
            return f"the verifier user cannot write {paths.verifier_dir} or run a shell after the agent ran: {(r.stderr or r.stdout or '')[-200:]}"
        return None

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

    # ---- grading -------------------------------------------------------------------------------------------------
    @staticmethod
    def _masked(tid: str, t0: float, kind: str, reason: str, **meta):
        from crucible.verify.runtime.plan import GradeReport

        return GradeReport(task=tid, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0,
                           infra={"status": "masked", "failure_kind": kind, "reason": reason[:1800]}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta=meta)

    @staticmethod
    def _zero(tid: str, t0: float, kind: str, reason: str, **meta):
        """The policy's own outcome in its container: scored 0 (never masked); ``meta.failure`` names it."""
        from crucible.verify.runtime.plan import GradeReport

        return GradeReport(task=tid, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"failure": {"kind": kind, "reason": reason[:1800]}, **meta})

    async def _liveness(self, world, tid: str, t0: float):
        """Host-side container facts before blaming the policy (M13 audit B2): gone / stopped through the daemon API /
        daemon unreachable masks; exited from inside is the policy's. None when the container runs (or cannot be asked)."""
        state = getattr(world.backend, "container_state", None)
        if state is None:
            return None
        st = await state()
        if not st.get("supported"):
            return None
        lost = await world.backend.container_lost()
        if lost:
            return self._masked(tid, t0, "crucible:env_container_lost", lost, container=st)
        if st.get("exists") and not st.get("running"):
            how = "killed for memory (OOM)" if st.get("oom") else f"its main process exited (code {st.get('exit_code')})"
            return self._zero(tid, t0, "crucible:agent_broke_env", f"the task container stopped from inside during the episode: {how}", container=st)
        return None

    def _verifier_needs_network(self, phase) -> bool:
        return getattr(phase.network_mode, "value", str(phase.network_mode)) != "no-network"

    async def _network_lost(self, world, surf, phase, *, host: bool = True) -> str | None:
        """Host-side evidence that the grader's own grade-time setup had no network (M13 audit B2): the container is
        detached from every network, or the host itself cannot reach the hosts the test script names. Only for test
        scripts that use the network in a verifier phase that allows it; in-container symptoms alone never mask (a
        policy can forge those)."""
        if surf is None or not surf.needs_network or not self._verifier_needs_network(phase):
            return None
        nets = await world.backend.networks()
        if nets is not None and not nets:
            return "the task container is attached to no network (detached from outside) while its test script needs the network"
        hosts = list(surf.hosts) or ["deb.debian.org", "archive.ubuntu.com", "pypi.org"]
        if host and not await _host_reaches(hosts):
            return f"the host cannot reach {', '.join(hosts[:4])}: no network for the grader's own setup"
        return None

    async def _verify(self, world, *, tid: str, user, timeout: float, t0: float, phase, phase_in, phase_out, verifier_kw: dict | None = None,
                      receipt_dir: Path | None = None, setup=None, leftovers=None):
        """Harbor's ``Verifier`` in the agent's container, guarded (M11 audit H1, M13 audit B1, B2, M10 lab audit R-2):
        returns ``(VerifierResult, None, flags, quiesce)`` or ``(None, GradeReport, flags, quiesce)`` when the grade ends
        without a verifier result. ``quiesce``: the receipt's summary (``Receipt.summary``) once the host-side quiesce ran
        (``receipt_dir``: where ``quiesce.json`` goes), else None; the callers put it in the report's ``meta.quiesce``.
        ``flags`` start with the quiesce's own (``policy_process_killed``) and keep them through the toolchain check; a
        report ``_liveness`` replaced carries none of its own, so the callers attach them (review round 2). ``setup``: a
        later multi-step step's setup processes, ``leftovers``: the previous step's grade leftovers (``_quiesce``). Order:
        the host's liveness, the quiesce, and
        only after a clean receipt the reset, the toolchain check and Harbor's ``Verifier`` (R-2: nothing of the grade runs
        while a process of the policy can)."""
        from crucible.interop.harbor import grader_guard

        env = world.backend.env
        surf = world.__dict__.get("_grader_surface")
        if surf is None and self.grader_integrity == "verify":
            return None, self._masked(tid, t0, "crucible:harness_unsupported", "the grader's toolchain was never captured (open() did not run on this World)"), [], None
        tag = uuid.uuid4().hex[:12]
        moved: list[tuple[str, str]] = []
        flags: list[dict] = []
        dead = await self._liveness(world, tid, t0)
        if dead is not None:
            return None, dead, flags, None
        # M10 lab audit R-2 (P23): before the reset, so a detached writer of the policy cannot re-create reward.json after
        # it; a StageFailure (respawned / uninterruptible) propagates to the caller's boundary, never into the except below
        ended, q, qflags = await self._quiesce(world, tid=tid, t0=t0, receipt_dir=receipt_dir, setup=setup, leftovers=leftovers)
        if ended is not None:
            return None, ended, flags, q
        flags = [*qflags]  # M10 lab audit R-2 review round 1: the quiesce's own flag rides every report of this grade
        from harbor.verifier.verifier import Verifier  # after the quiesce: an ended grade needs no Harbor

        try:
            tp = self._trial_paths(world)
            await self._reset_shared_dirs(env)
            broken = await self._probe(env, user)
            if broken:
                return None, (await self._liveness(world, tid, t0)) or self._zero(tid, t0, "crucible:agent_broke_env", broken, flags=flags), flags, q
            if surf is not None:
                findings, before = await grader_guard.check(world.backend, surf)
                flags = [*flags, *findings.flags()]
                if findings.verdicts:
                    return None, self._zero(tid, t0, "crucible:grader_tampered", "the policy changed the grader's toolchain: " + "; ".join(
                        f"{v['path']} ({v['why']})" for v in findings.verdicts[:8]), flags=flags, integrity={"verdicts": findings.verdicts,
                                                                                                            "suspicions": findings.suspicions}), flags, q
                moved = await grader_guard.hide(env, findings.hide, tag)
            net = await self._network_lost(world, surf, phase, host=False)  # cheap host-side fact before a long install
            if net:
                return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags, q
            await phase_in()
            pins = grader_guard.pinned_env(surf, tag) if surf is not None else {}
            with env.with_default_user(user):
                verifier = Verifier(task=self.task, trial_paths=tp, environment=env, override_env=pins, **(verifier_kw or {}))
                result = await asyncio.wait_for(verifier.verify(), timeout=timeout + 60)
            if surf is not None:  # a tool swapped while the verifier ran (a background process of the policy)
                post, after = await grader_guard.check(world.backend, surf, after_grade=True, since=before)
                if post.verdicts:
                    flags = [*flags, *post.flags()]
                    return None, self._zero(tid, t0, "crucible:grader_tampered", "the grader's toolchain changed during the grade: " + "; ".join(
                        f"{v['path']} ({v['why']})" for v in post.verdicts[:8]), flags=flags, integrity={"verdicts": post.verdicts}), flags, q
                surf.advance(after, rejudge=[p for p, _ in moved])
        except TimeoutError:
            return None, (await self._liveness(world, tid, t0)) or self._masked(tid, t0, "verifier_timeout", f"{tid}: verifier exceeded {timeout:.0f}s", flags=flags), flags, q
        except Exception as e:  # noqa: BLE001 -- typed infra errors mask; reward-file errors after a reset are the agent's (H1 V4)
            dead = await self._liveness(world, tid, t0)
            if dead is not None:
                return None, dead, flags, q
            if type(e).__name__ in REWARD_FILE_ERRORS:
                net = await self._network_lost(world, surf, phase)
                if net:
                    return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags, q
                return None, self._zero(tid, t0, f"crucible:{type(e).__name__}", f"{tid}: the verifier wrote no valid reward after the agent ran: {e}", flags=flags), flags, q
            if isinstance(e, grader_guard.GuardError):  # the host could not inspect the container: never an unverified grade
                return None, self._masked(tid, t0, "crucible:backend_error", f"{tid}: the grader's toolchain could not be verified: {e}", flags=flags), flags, q
            return None, self._masked(tid, t0, type(e).__name__, f"{tid}: {e}", flags=flags), flags, q
        finally:
            try:
                await phase_out()
            except Exception:  # noqa: BLE001, S110 -- restoring the phase never overrides the grade (H1 V19)
                pass
            if moved:
                await grader_guard.unhide(env, moved)
        dead = await self._liveness(world, tid, t0)
        if dead is not None:
            return None, dead, flags, q
        reward, _ = scalar_reward(dict(result.rewards or {}), self.reward_key)
        if reward is None or reward < 1.0:  # a failed grade: was the grader's own network gone?
            net = await self._network_lost(world, surf, phase)
            if net:
                return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags, q
        return result, None, flags, q

    async def grade(self, world, workdir: Path):
        from crucible.verify.runtime.plan import GradeReport

        t0 = time.monotonic()
        env = world.backend.env
        plan = self.plan
        phase, current = plan.verifier_phase, plan.agent_phase
        dynamic = getattr(getattr(env, "capabilities", None), "dynamic_network_policy", False)
        if phase != current and not dynamic:
            return self._masked(self.id, t0, "check_infra", f"the verifier needs network {phase} but this environment cannot change its policy ({current})")

        async def phase_in():
            if phase != current:
                await env.set_network_policy(phase)

        async def phase_out():
            if phase != current:
                await env.set_network_policy(current)

        # M10 lab audit R-2: the quiesce receipt goes next to the grade dir, ``<workdir>/../verifier/quiesce.json`` (every
        # harness grades into ``<attempt>/grade`` and reports into ``<attempt>/verifier``: native.py:361,
        # agent_harness.py:948-951)
        result, ended, flags, q = await self._verify(world, tid=self.id, user=self.task.config.verifier.user, timeout=self.verifier_timeout, t0=t0,
                                                     phase=phase, phase_in=phase_in, phase_out=phase_out, receipt_dir=Path(workdir).parent / "verifier")
        if ended is not None:
            if q is not None:
                ended.meta.setdefault("quiesce", q)
            if flags:  # M10 lab audit R-2 review round 2: a _liveness report replacing the grade carries no flags; the quiesce's policy_process_killed rides it too
                ended.meta.setdefault("flags", flags)
            return ended
        rewards = dict(result.rewards or {})
        reward, problem = scalar_reward(rewards, self.reward_key)
        if reward is None:
            return self._masked(self.id, t0, problem.split(":")[0], problem, flags=flags, quiesce=q)
        comps: dict[str, Any] = {f"harbor/{k}": float(v) for k, v in rewards.items()}
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"rewards": rewards, "components": comps, "flags": flags,
                                                                            "verifier": json.loads(result.model_dump_json()), "quiesce": q})


def _harbor_version() -> str:
    """The installed Harbor's version, named in errors about API differences (Crucible codes against 0.24)."""
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("harbor")
    except PackageNotFoundError:
        return "absent"


async def _host_reaches(hosts: list[str], timeout: float = 5.0) -> bool:
    """Can the host itself open a TCP connection to any of these hosts (443, then 80)?"""
    for h in hosts[:4]:
        for port in (443, 80):
            try:
                _, w = await asyncio.wait_for(asyncio.open_connection(h, port), timeout)
                w.close()
                return True
            except (OSError, TimeoutError):
                continue
    return False
