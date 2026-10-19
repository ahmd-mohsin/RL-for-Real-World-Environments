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
- **the agent shares the grader's container** (M11 audit): ``/logs/verifier`` and ``/tests`` are emptied before the grade
  (Harbor's multi-step reset), the agent's own network phase is applied after ``open``, and tasks whose verifier runs
  in a separate environment are refused. After the agent ran, a container in which the verifier user can no longer
  write its reward (``agent_broke_env``) or a verifier that writes no valid reward is the agent's outcome, scored 0;
  a timeout, an upload failure or a lost container masks.
- **the grader's own toolchain** (M13 audit B1, ``grader_guard``): captured from the pristine container at ``open`` and
  checked host-side (``docker diff`` / ``docker cp``) before and after the verifier. A policy that shadowed, replaced or
  hooked a tool the verifier runs is ``crucible:grader_tampered`` (scored 0, flagged ``grader_tampered``, never 1.0);
  the verifier runs with the pristine ``PATH`` and a fresh ``PYTHONPYCACHEPREFIX``. Environments the host cannot inspect
  (non-Docker providers) are refused in place unless ``grader_integrity="off"``.
- **infrastructure is told apart from the policy by host-side evidence** (M13 audit B2): a container that is gone,
  stopped through the daemon API, or whose daemon cannot be reached masks ``crucible:env_container_lost`` (mid-episode
  too: the terminal raises instead of returning compose's "service is not running" as the policy's failed command); a
  container that exited from inside, or one alive whose verifier user cannot work, is ``agent_broke_env``. A grade whose
  test script needs the network fails while the host shows the container detached or the network unreachable: masked
  ``crucible:grader_network_error``.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
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
        r = await self.ctx.exec.run(command, cwd=self._cwd, timeout_s=self.deadline(timeout_s, self.spec.params.get("max_timeout_s", 3600)))
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

# reward-file errors after the agent ran in the same container: its outcome (the dirs were emptied first, so the grader
# alone wrote there), never a mask (M11 audit H1 V4)
REWARD_FILE_ERRORS = ("RewardFileNotFoundError", "RewardFileEmptyError", "VerifierOutputParseError")


class HarborInPlaceTask:
    def __init__(self, task_dir: Path, *, docker_host: str | None = None, gpu_host: str | None = None, reward_key: str | None = None,
                 environment_type: str = "docker", environment_kwargs: dict | None = None, user: str | None = None, keep_containers: bool = False,
                 grader_integrity: str = "verify"):
        from harbor.models.task.task import Task
        from harbor.models.task.verifier_mode import VerifierEnvironmentMode, resolve_task_verifier_mode

        if grader_integrity not in ("verify", "off"):
            raise ValueError("grader_integrity: 'verify' | 'off'")
        if grader_integrity == "verify" and environment_type != "docker":
            # the grader runs in the agent's container; only a host-side Docker CLI can tell whether the policy changed the
            # grader's toolchain (M13 audit B1). Without it a stubbed pytest would grade 1.0 unseen
            raise ValueError(f"{task_dir}: in-place grading on {environment_type!r} cannot verify the grader's toolchain (needs a host-side "
                             "Docker CLI); grade it in patch mode (verify_dir), or pass grader_integrity='off' to accept unverified grades")
        self.grader_integrity = grader_integrity
        self.dir = Path(task_dir).resolve()
        self.task = Task(self.dir)
        cfg = self.task.config
        if not cfg.steps and resolve_task_verifier_mode(cfg) == VerifierEnvironmentMode.SEPARATE:
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
        phases = [x for x in (self.plan.agent_phase, self.plan.verifier_phase) if x != self.plan.agent_env_baseline]
        if phases:  # the environment must be able to switch policy per phase (Harbor egress control)
            self.backend_params["phase_network_policies"] = [x.model_dump(mode="json") for x in phases]

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

    async def open(self, workdir: Path):
        from crucible.core.store import Store
        from crucible.world.world import World

        workdir = Path(workdir)
        w = World(self.spec(workdir), instructions={"agent": self.instruction}, store=Store(workdir / "s"), workdir=workdir / "w", backend="harbor",
                  backend_params=self.backend_params)
        await w.start()
        await w.reset()
        await self._healthcheck(w)
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
        the grader reads only what it writes (H1 V1: a planted ``reward.json`` or ``/tests/conftest.py`` scored 1)."""
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

    def digest(self) -> str:
        """Harbor's own content hash of the task directory (R-2, P18): the id Harbor publishes it under, over the files
        Harbor selects (task.toml, instruction, environment/, tests/, solution/, steps/, .gitignore applied)."""
        from harbor.publisher.packager import Packager

        return "sha256:" + Packager.compute_content_hash(self.dir)[0]

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

    async def _verify(self, world, *, tid: str, user, timeout: float, t0: float, phase, phase_in, phase_out, verifier_kw: dict | None = None):
        """Harbor's ``Verifier`` in the agent's container, guarded (M11 audit H1, M13 audit B1, B2): returns
        ``(VerifierResult, None, flags)`` or ``(None, GradeReport, flags)`` when the grade ends without a verifier result."""
        from harbor.verifier.verifier import Verifier

        from crucible.interop.harbor import grader_guard

        env = world.backend.env
        surf = world.__dict__.get("_grader_surface")
        if surf is None and self.grader_integrity == "verify":
            return None, self._masked(tid, t0, "crucible:harness_unsupported", "the grader's toolchain was never captured (open() did not run on this World)"), []
        tag = uuid.uuid4().hex[:12]
        moved: list[tuple[str, str]] = []
        flags: list[dict] = []
        dead = await self._liveness(world, tid, t0)
        if dead is not None:
            return None, dead, flags
        try:
            tp = self._trial_paths(world)
            await self._reset_shared_dirs(env)
            broken = await self._probe(env, user)
            if broken:
                return None, (await self._liveness(world, tid, t0)) or self._zero(tid, t0, "crucible:agent_broke_env", broken), flags
            if surf is not None:
                findings, before = await grader_guard.check(world.backend, surf)
                flags = findings.flags()
                if findings.verdicts:
                    return None, self._zero(tid, t0, "crucible:grader_tampered", "the policy changed the grader's toolchain: " + "; ".join(
                        f"{v['path']} ({v['why']})" for v in findings.verdicts[:8]), flags=flags, integrity={"verdicts": findings.verdicts,
                                                                                                            "suspicions": findings.suspicions}), flags
                moved = await grader_guard.hide(env, findings.hide, tag)
            net = await self._network_lost(world, surf, phase, host=False)  # cheap host-side fact before a long install
            if net:
                return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags
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
                        f"{v['path']} ({v['why']})" for v in post.verdicts[:8]), flags=flags, integrity={"verdicts": post.verdicts}), flags
                surf.advance(after, rejudge=[p for p, _ in moved])
        except TimeoutError:
            return None, (await self._liveness(world, tid, t0)) or self._masked(tid, t0, "verifier_timeout", f"{tid}: verifier exceeded {timeout:.0f}s", flags=flags), flags
        except Exception as e:  # noqa: BLE001 -- typed infra errors mask; reward-file errors after a reset are the agent's (H1 V4)
            dead = await self._liveness(world, tid, t0)
            if dead is not None:
                return None, dead, flags
            if type(e).__name__ in REWARD_FILE_ERRORS:
                net = await self._network_lost(world, surf, phase)
                if net:
                    return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags
                return None, self._zero(tid, t0, f"crucible:{type(e).__name__}", f"{tid}: the verifier wrote no valid reward after the agent ran: {e}", flags=flags), flags
            if isinstance(e, grader_guard.GuardError):  # the host could not inspect the container: never an unverified grade
                return None, self._masked(tid, t0, "crucible:backend_error", f"{tid}: the grader's toolchain could not be verified: {e}", flags=flags), flags
            return None, self._masked(tid, t0, type(e).__name__, f"{tid}: {e}", flags=flags), flags
        finally:
            try:
                await phase_out()
            except Exception:  # noqa: BLE001, S110 -- restoring the phase never overrides the grade (H1 V19)
                pass
            if moved:
                await grader_guard.unhide(env, moved)
        dead = await self._liveness(world, tid, t0)
        if dead is not None:
            return None, dead, flags
        reward, _ = scalar_reward(dict(result.rewards or {}), self.reward_key)
        if reward is None or reward < 1.0:  # a failed grade: was the grader's own network gone?
            net = await self._network_lost(world, surf, phase)
            if net:
                return None, self._masked(tid, t0, "crucible:grader_network_error", net, flags=flags), flags
        return result, None, flags

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

        result, ended, flags = await self._verify(world, tid=self.id, user=self.task.config.verifier.user, timeout=self.verifier_timeout, t0=t0,
                                                  phase=phase, phase_in=phase_in, phase_out=phase_out)
        if ended is not None:
            return ended
        rewards = dict(result.rewards or {})
        reward, problem = scalar_reward(rewards, self.reward_key)
        if reward is None:
            return self._masked(self.id, t0, problem.split(":")[0], problem, flags=flags)
        comps: dict[str, Any] = {f"harbor/{k}": float(v) for k, v in rewards.items()}
        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"rewards": rewards, "components": comps, "flags": flags,
                                                                            "verifier": json.loads(result.model_dump_json())})


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
