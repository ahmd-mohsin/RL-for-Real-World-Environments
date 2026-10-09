"""CI App: the policy runs hidden checks without ever seeing them (CI-as-a-tool).

How a run works (``run_ci``):
1. The candidate is the current state of the ``source`` App (an ``fs`` / ``fs_git`` State Adapter): it is
   snapshotted (managed files only; FIFOs, devices, symlink escapes are never followed) ...
2. ... and restored into a **separate runner sandbox** at ``target_root``. The runner holds the hidden
   checks (``hidden.src`` copied to ``hidden.dest``, root-only); they never enter the policy's world.
3. Each check runs in the runner; the policy gets **only** ``name: PASS|FAIL|TIMEOUT (seconds)``. Output
   is returned only for checks marked ``show_output`` (truncated, scrubbed of leak patterns).
4. Runs cost virtual time (``latency``) and are budgeted (``max_runs``); the run counter and history are
   App state (``json_doc``), so a WorldRef snapshot/restore includes the CI budget.

Runners (registry kind ``ci_runner``): ``local`` (a directory outside the world), ``docker`` (a container from
any image — e.g. a task's verifier image — on the local daemon or a remote one via ``docker_host``, with
``--network none``, ``--init`` and optional GPUs).

Params: ``source``, ``target_root``, ``runner`` {kind, ...}, ``hidden`` {src, dest}, ``checks`` [{name, cmd,
timeout_s, show_output, user}], ``max_runs``, ``latency``, ``setup`` (runner commands run once, as root).
"""

from __future__ import annotations

import json
import shlex
import shutil
import tarfile
import tempfile
import time
import uuid
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from crucible.core.registry import get_registry
from crucible.core.schema.actions import ToolSpec
from crucible.world.apps.base import App, ToolError
from crucible.world.clock import parse_duration
from crucible.world.exec import DockerExec, Exec, LocalExec, _run_subprocess


class CiRunner(ABC):
    exec: Exec

    def __init__(self, params: dict[str, Any], *, workdir: Path):
        self.params = params
        self.workdir = workdir

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    def root(self, target_root: str) -> str:
        return target_root


class LocalCiRunner(CiRunner):
    """Checks run on this machine in a private directory (CI for local worlds; no isolation from the host)."""

    async def start(self) -> None:
        self.dir = Path(tempfile.mkdtemp(prefix="crucible-ci-", dir=self.params.get("tmp")))
        self.exec = LocalExec()

    def root(self, target_root: str) -> str:
        return str(self.dir / target_root.lstrip("/"))

    async def stop(self) -> None:
        shutil.rmtree(self.dir, ignore_errors=True)


class DockerCiRunner(CiRunner):
    """A container from ``image`` (e.g. a task's verifier image), no network, PID 1 = init, optional GPUs."""

    async def start(self) -> None:
        self.name = f"crucible-ci-{uuid.uuid4().hex[:10]}"
        env = {"DOCKER_HOST": self.params["docker_host"]} if self.params.get("docker_host") else None
        argv = ["docker", "run", "-d", "--rm", "--init", "--network", "none", "--name", self.name, "--label", "crucible.app=ci",
                "--pids-limit", str(self.params.get("pids_limit", 4096))]
        if self.params.get("gpus"):
            argv += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"] if self.params.get("gpu_mode", "runtime") == "runtime" else ["--gpus", "all"]
        if self.params.get("memory"):
            argv += ["--memory", str(self.params["memory"])]
        argv += [self.params["image"], "sleep", "infinity"]
        out = await _run_subprocess(argv, cwd=None, env=env, timeout_s=600, stdin=None)
        if not out.ok:
            raise RuntimeError(f"CI runner failed to start: {out.stderr[-600:]}")
        self.exec = DockerExec(self.name, docker_host=self.params.get("docker_host"))
        self._env = env

    async def stop(self) -> None:
        await _run_subprocess(["docker", "rm", "-f", self.name], cwd=None, env=self._env, timeout_s=120, stdin=None)


class CiApp(App):
    template = "common/ci"
    PATH_PARAMS = ("path",)

    def __init__(self, spec, ctx):
        super().__init__(spec, ctx)
        p = spec.params
        for key in ("source", "target_root", "checks"):
            if key not in p:
                raise ValueError(f"{self.id}: common/ci needs params.{key}")
        if self.spec.state.adapter != "json_doc":
            raise ValueError(f"{self.id}: common/ci keeps its run history in a json_doc state")
        self.runner: CiRunner | None = None
        self._runner_ready = False

    async def provision(self) -> None:
        await self.ctx.exec.run(f"mkdir -p {shlex.quote(str(Path(self.state_params['path']).parent))}")
        await self._save({"runs": 0, "history": []})

    async def _load(self) -> dict[str, Any]:
        return json.loads(await self.ctx.exec.read_text(self.state_params["path"]))

    async def _save(self, st: dict[str, Any]) -> None:
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(st, f, sort_keys=True)
        try:
            await self.ctx.exec.push_file(Path(f.name), self.state_params["path"])
        finally:
            Path(f.name).unlink(missing_ok=True)

    async def _ensure_runner(self) -> CiRunner:
        if self.runner is not None and self._runner_ready:
            return self.runner
        rp = dict(self.spec.params.get("runner", {"kind": "local"}))
        cls = get_registry().get("ci_runner", rp.get("kind", "local"))
        self.runner = cls(rp, workdir=self.ctx.host_workdir)
        await self.runner.start()
        hidden = self.spec.params.get("hidden")
        if hidden:
            src = self.ctx.host_seed(hidden["src"])
            dest = self.runner.root(hidden.get("dest", "/opt/crucible-hidden"))
            with tempfile.TemporaryDirectory() as d:
                tar = Path(d) / "hidden.tar"
                with tarfile.open(tar, "w") as tf:
                    for f in sorted(src.rglob("*")):
                        if "__pycache__" not in f.parts:
                            tf.add(f, arcname=str(f.relative_to(src)), recursive=False)
                remote = f"/tmp/.crucible_hidden_{uuid.uuid4().hex[:8]}.tar"
                await self.runner.exec.push_file(tar, remote)
                out = await self.runner.exec.run(f"mkdir -p {shlex.quote(dest)} && tar -xf {remote} -C {shlex.quote(dest)} && rm -f {remote} && chmod -R go-rwx {shlex.quote(dest)}")
                if not out.ok:
                    raise RuntimeError(f"cannot stage hidden checks: {out.stderr[-400:]}")
        for cmd in self.spec.params.get("setup", []):
            out = await self.runner.exec.run(cmd, timeout_s=1800)
            if not out.ok:
                raise RuntimeError(f"CI runner setup failed ({cmd!r}): {(out.stderr or out.stdout)[-600:]}")
        self._runner_ready = True
        return self.runner

    async def stop(self) -> None:
        if self.runner is not None:
            await self.runner.stop()
            self.runner = None
            self._runner_ready = False

    async def list_tools(self) -> list[ToolSpec]:
        names = [c["name"] for c in self.spec.params["checks"]]
        return [
            ToolSpec(app=self.id, name="run_ci", description=f"Run the CI checks ({', '.join(names)}) on the current code. Reports pass/fail per check.",
                     input_schema={"type": "object", "properties": {"checks": {"type": "array", "items": {"type": "string"}}}}),
            ToolSpec(app=self.id, name="ci_history", read_only=True, description="Results of previous CI runs.", input_schema={"type": "object", "properties": {}}),
        ]

    async def tool_ci_history(self) -> Any:
        st = await self._load()
        return {"runs": st["runs"], "max_runs": self.spec.params.get("max_runs"), "history": st["history"][-10:]}

    async def tool_run_ci(self, checks: list[str] | None = None) -> Any:
        from crucible.world.leaks import scrub_text
        from crucible.world.state.base import SnapshotMismatch
        from crucible.world.state.fs import FsAdapter

        st = await self._load()
        budget = self.spec.params.get("max_runs")
        if budget is not None and st["runs"] >= budget:
            raise ToolError(f"CI budget exhausted ({budget} runs)")
        all_checks = {c["name"]: c for c in self.spec.params["checks"]}
        wanted = checks or list(all_checks)
        unknown = [c for c in wanted if c not in all_checks]
        if unknown:
            raise ToolError(f"unknown checks {unknown}; available: {sorted(all_checks)}")
        peers = self.ctx.extra.get("peers", {})
        source = peers.get(self.spec.params["source"])
        if source is None:
            raise ToolError(f"CI source App {self.spec.params['source']!r} not found")
        runner = await self._ensure_runner()
        t0 = time.monotonic()
        candidate = await source.state.snapshot()
        root = runner.root(self.spec.params["target_root"])
        await runner.exec.run(f"mkdir -p {shlex.quote(root)}")
        target = FsAdapter(f"{self.id}-runner", runner.exec, {"root": root, "exclude": self.spec.params.get("runner_exclude", [])}, self.ctx.store)
        try:
            await target.restore(candidate.model_copy(update={"adapter": "fs"}))
        except SnapshotMismatch as e:  # sockets/devices/oversize files cannot be shipped to CI: the policy's problem
            raise ToolError(f"cannot send the code to CI: {str(e).split(': ', 1)[-1]}") from e
        lines, results = [], {}
        hidden = runner.root(self.spec.params.get("hidden", {}).get("dest", "/opt/crucible-hidden"))
        for name in wanted:
            c = all_checks[name]
            cmd = c["cmd"].replace("{hidden}", hidden).replace("{root}", root)
            out = await runner.exec.run(cmd, cwd=root, timeout_s=float(c.get("timeout_s", 1800)), user=c.get("user"))
            status = "TIMEOUT" if out.timed_out else ("PASS" if out.rc == 0 else "FAIL")
            results[name] = status
            line = f"{name}: {status} ({out.duration_s:.1f}s)"
            if c.get("show_output") and status != "PASS":
                tail = "\n".join((out.stdout + out.stderr).splitlines()[-int(c.get("output_lines", 30)):])
                line += "\n" + scrub_text(tail)
            lines.append(line)
        st["runs"] += 1
        st["history"].append({"run": st["runs"], "candidate": candidate.fingerprint[:16], "results": results})
        await self._save(st)
        latency = self.spec.params.get("latency")
        if latency and self.ctx.clock is not None and self.ctx.clock.mode == "step":
            self.ctx.clock.advance(parse_duration(latency))
        header = f"CI run #{st['runs']}" + (f" of {budget}" if budget else "") + f" ({len(wanted)} checks, {time.monotonic() - t0:.0f}s)"
        return header + "\n" + "\n".join(lines)
