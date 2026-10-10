"""Grade a stored patch with a Harbor task directory's *own* verifier (DEPEND on Harbor's layout and conventions).

``HarborDirTask`` is a ``RunTask`` for the workspace seam: the agent's work arrives as a patch and ``grade_patch`` runs
``[verifier].command`` in a fresh container of the task image with Harbor's layout (``/tests`` read-only,
``/logs/artifacts/model.patch``, ``/logs/verifier``), the declared network mode, limits and GPU flags, on any Docker
host (``GpuHost``: local, or the lab over ``DOCKER_HOST=ssh://``). The reward is Harbor's: ``reward.json`` (flat) or
``reward.txt``. No reward file, a timeout, or a stale image is masked, never 0 (``verify.stage`` conventions).

The task can be pinned to a git ref (``ref=``): a historical run is re-graded against the task *as it was* when the
run was graded (research m9-m10/02 §6.1: task02's suite was tightened after its frontier run).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tomllib
import uuid
from pathlib import Path

from crucible.runtime.gpu import GpuHost
from crucible.verify.runtime.plan import GradeReport


def export_ref(repo: Path, task_dir: Path, ref: str, dest: Path) -> Path:
    """The task directory as of ``ref`` (``git archive``), cached under ``dest``."""
    rel = Path(task_dir).resolve().relative_to(Path(repo).resolve())
    out = Path(dest) / f"{rel.name}@{ref}"
    if not (out / "task.toml").exists():
        out.mkdir(parents=True, exist_ok=True)
        arch = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", ref, str(rel)], capture_output=True, check=True).stdout
        subprocess.run(["tar", "-x", f"--strip-components={len(rel.parts)}", "-C", str(out)], input=arch, check=True)
    return out


def _gates(report: dict) -> dict[str, str]:
    out = {}
    for g, v in (report.get("gates") or {}).items():
        if isinstance(v, dict) and "pass" in v:
            out[g] = "pass" if v["pass"] else "fail"
        elif isinstance(v, dict) and v.get("skipped"):
            out[g] = "skip"
        elif isinstance(v, bool):
            out[g] = "pass" if v else "fail"
    return out


class HarborDirTask:
    def __init__(self, task_dir: Path, *, image: str, host: GpuHost | None = None, expected_image: str | None = None, ref: str | None = None,
                 perf: bool = False, timeout: float | None = None):
        self.dir = Path(task_dir)
        self.cfg = tomllib.loads((self.dir / "task.toml").read_text())
        name = (self.cfg.get("task") or {}).get("name") or self.dir.name
        self.id = f"{name}@{ref}" if ref else name
        self.world_id = f"harbor:{name}"
        self.instruction = (self.dir / "instruction.md").read_text() if (self.dir / "instruction.md").exists() else ""
        self.image, self.host, self.expected_image, self.perf = image, host or GpuHost(name="local", docker_host=None), expected_image, perf
        v = self.cfg.get("verifier") or {}
        self.command = v.get("command") or "bash /tests/test.sh"
        self.timeout = timeout or float(v.get("timeout_sec") or 3600)
        venv = v.get("environment") or {}
        env = self.cfg.get("environment") or {}
        self.gpus = int(venv.get("gpus") or env.get("gpus") or 0)
        self.cpus, self.memory_mb = venv.get("cpus") or env.get("cpus"), venv.get("memory_mb") or env.get("memory_mb")
        self.network = v.get("network_mode") or "no-network"

    def _docker(self, *a: str, timeout: float = 120) -> subprocess.CompletedProcess:
        env = None
        if self.host.docker_host:
            import os

            env = {**os.environ, "DOCKER_HOST": self.host.docker_host}
        return subprocess.run(["docker", *a], capture_output=True, text=True, timeout=timeout, env=env, check=False)

    def _masked(self, kind: str, reason: str, t0: float) -> GradeReport:
        import time

        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0,
                           infra={"status": "masked", "failure_kind": kind, "reason": reason}, trusted=True, elapsed_s=round(time.monotonic() - t0, 2))

    async def grade_patch(self, patch: str, gdir: Path) -> GradeReport:
        import asyncio
        import time

        t0 = time.monotonic()
        gdir = Path(gdir)
        if self.expected_image:
            got = self._docker("image", "inspect", "--format", "{{.Id}}", self.image).stdout.strip() or None
            if got != self.expected_image:
                return self._masked("check_infra", f"stale image: {self.image} is {got}, the task pins {self.expected_image}", t0)
        key = f"grades/{uuid.uuid4().hex[:12]}"
        art = gdir / "stage" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "model.patch").write_text(patch)
        (gdir / "stage" / "verifier").mkdir(exist_ok=True)
        tests = self.host.put(self.dir / "tests", f"{key}/tests")
        arts = self.host.put(art, f"{key}/artifacts")
        out = self.host.put(gdir / "stage" / "verifier", f"{key}/verifier")
        name = f"crucible-hgrade-{uuid.uuid4().hex[:10]}"
        argv = ["run", "--rm", "--init", "--name", name, "--label", "crucible.sandbox=1", "--shm-size", "8g",
                "-v", f"{tests}:/tests:ro", "-v", f"{arts}:/logs/artifacts:ro", "-v", f"{out}:/logs/verifier"]
        if self.network != "public":
            argv += ["--network", "none"]
        if self.gpus:
            argv += self.host.docker_gpu_flags()
        if self.cpus:
            argv += ["--cpus", str(self.cpus)]
        if self.memory_mb:
            argv += ["--memory", f"{int(self.memory_mb)}m"]
        argv += ["--entrypoint", "bash", self.image, "-c", self.command]
        try:
            p = await asyncio.to_thread(self._docker, *argv, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            self._docker("rm", "-f", name)
            return self._masked("verifier_timeout", f"grade exceeded {self.timeout:.0f}s; container {name} removed", t0)
        local = gdir / "verifier"
        self.host.get(f"{key}/verifier", local)
        self.host.exec(f"rm -rf {self.host.workroot}/{key}")
        (gdir / "console.log").write_text((p.stdout + p.stderr)[-50000:])
        report = json.loads((local / "report.json").read_text()) if (local / "report.json").exists() else {}
        reward = None
        if (local / "reward.json").exists():
            reward = float(json.loads((local / "reward.json").read_text()).get("reward"))
        elif (local / "reward.txt").exists() and (local / "reward.txt").read_text().strip():
            try:
                reward = float((local / "reward.txt").read_text().strip().splitlines()[-1])
            except ValueError:
                reward = None
        if reward is None:
            return self._masked("no_reward_json", f"the verifier left no reward (rc {p.returncode})", t0)
        gates = _gates(report)
        shutil.rmtree(gdir / "stage", ignore_errors=True)
        return GradeReport(task=self.id, tier="full", checks=[], gates=gates, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"report": report, "rc": p.returncode})
