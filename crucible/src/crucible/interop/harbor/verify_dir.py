"""Grade a stored patch with a Harbor task directory's *own* verifier (DEPEND on Harbor's layout and conventions).

``HarborDirTask`` is a ``RunTask`` for the workspace seam: the agent's work arrives as a patch and ``grade_patch`` runs
``[verifier].command`` in a fresh container of the task image with Harbor's layout (``/tests`` read-only,
``/logs/artifacts/model.patch``, ``/logs/verifier``), the declared network mode, limits and GPU flags, on any Docker
host (``GpuHost``: local, or the lab over ``DOCKER_HOST=ssh://``), with ``[verifier].env`` (an env file), ``user`` and
Harbor's verifier network phase (the environment baseline, default public, unless ``[verifier].network_mode``). The
reward follows Harbor's rules (``rewards.parse_reward_dir``): ``reward.json`` (flat, finite numbers) or ``reward.txt``
(one number); no reward, a non-finite or unparseable one, a timeout or a stale image is masked, never 0 (``verify.stage``
conventions). ``mask.json`` is honoured only from Crucible's grading kit. ``grade_control`` (M10 lab audit R-3) grades the
task's ``solution/model.patch`` (or ``oracle_patch``) the same way: the control a repeated hang or checker failure on a
policy's patch is decided by. A non-finite reward (a NaN or infinite reward file, or the kit's own non-finite mask) is
still masked first, as ``non_finite_reward``: answer-sensitive (a patch can make the tests write it), so a repeated one
is decided by the control. A grade cancelled by its caller (a scoring budget) removes its container before the staged
mounts go and before the control runs in the same slot.

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
from crucible.runtime.grading import unbudgeted
from crucible.verify.runtime.plan import GradeReport


def export_ref(repo: Path, task_dir: Path, ref: str, dest: Path) -> Path:
    """The task directory as of ``ref`` (``git archive``), cached under ``dest`` by the commit the ref names, extracted to
    a temporary directory and renamed into place (M11 audit H1 V20: a moved branch reused the old grader, and a crash
    mid-extract left a partial tree that was reused)."""
    import tempfile

    rel = Path(task_dir).resolve().relative_to(Path(repo).resolve())
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "--verify", f"{ref}^{{commit}}"], capture_output=True, text=True, check=True).stdout.strip()
    immutable = len(ref) >= 7 and sha.startswith(ref)  # a commit id: the name is already the content
    out = Path(dest) / f"{rel.name}@{ref if immutable else sha[:12]}"
    if not (out / "task.toml").exists():
        Path(dest).mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=dest, prefix=".extract-") as tmp:
            arch = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", sha, str(rel)], capture_output=True, check=True).stdout
            subprocess.run(["tar", "-x", f"--strip-components={len(rel.parts)}", "-C", tmp], input=arch, check=True)
            if out.exists():
                shutil.rmtree(out)
            Path(tmp).rename(out)
            Path(tmp).mkdir()  # the context manager removes it
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


def _resolve_env(env: dict) -> dict[str, str]:
    """Harbor ``resolve_env_vars``: ``${VAR}`` and ``${VAR:-default}`` from the host environment."""
    import os
    import re

    def sub(m: re.Match) -> str:
        name, _, default = m.group(1).partition(":-")
        return os.environ.get(name, default if m.group(1).find(":-") >= 0 else "")

    return {str(k): re.sub(r"\$\{([^}]+)\}", sub, str(v)) for k, v in (env or {}).items()}


def _memory_mb(text: object) -> int | None:
    """Harbor's legacy ``memory = "2G"`` (migrated to ``memory_mb``)."""
    if text is None:
        return None
    t = str(text).strip().upper().removesuffix("B")
    mult = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}.get(t[-1:], None)
    try:
        return int(float(t[:-1]) * mult) if mult else int(float(t) / (1024 * 1024))
    except ValueError:
        return None


class HarborDirTask:
    def __init__(self, task_dir: Path, *, image: str, host: GpuHost | None = None, expected_image: str | None = None, ref: str | None = None,
                 perf: bool = False, timeout: float | None = None, trust_mask: bool | None = None, oracle_patch: Path | None = None):
        """``oracle_patch``: the known-good patch the control grade scores (M10 lab audit R-3, ``grade_control``);
        default ``solution/model.patch`` in the task dir."""
        self.dir = Path(task_dir)
        self.oracle_patch = Path(oracle_patch) if oracle_patch is not None else self.dir / "solution" / "model.patch"
        self.cfg = tomllib.loads((self.dir / "task.toml").read_text())
        name = (self.cfg.get("task") or {}).get("name") or self.dir.name
        self.id = f"{name}@{ref}" if ref else name
        self.world_id = f"harbor:{name}"
        self.instruction = (self.dir / "instruction.md").read_text() if (self.dir / "instruction.md").exists() else ""
        self.image, self.host, self.expected_image, self.perf = image, host or GpuHost(name="local", docker_host=None), expected_image, perf
        v = self.cfg.get("verifier") or {}
        self.command = v.get("command") or "bash /tests/test.sh"
        self.timeout = timeout or float(v.get("timeout_sec") or 3600)
        # M10 lab audit R-3: what one control grade may take (``grading.control_budget`` raises the harness's budget to it):
        # grade_patch's run, plus the image inspect and the host put/get
        self.control_timeout_s = self.timeout + 300.0
        venv = v.get("environment") or {}
        env = self.cfg.get("environment") or {}
        self.gpus = int(venv.get("gpus") or env.get("gpus") or 0)
        self.cpus = venv.get("cpus") or env.get("cpus")
        self.memory_mb = venv.get("memory_mb") or env.get("memory_mb") or _memory_mb(venv.get("memory") or env.get("memory"))
        # Harbor's verifier phase: ``[verifier].network_mode``, else the environment baseline (``network_mode``, or the
        # deprecated ``allow_internet``), whose default is public (H1 V6: this defaulted to no-network)
        base = env.get("network_mode") or ("no-network" if env.get("allow_internet") is False else "public")
        self.network = v.get("network_mode") or base
        if self.network == "allowlist":
            raise ValueError(f"{self.dir}: an allowlist verifier network cannot be enforced in a plain container; grade it on Harbor")
        self.verifier_env = _resolve_env(v.get("env") or {})  # merged into the test run, as Harbor does
        self.user = v.get("user")
        # ``mask.json`` comes only from Crucible's grading kit; a foreign test.sh runs the code under test with
        # ``/logs/verifier`` writable, so its mask could be the candidate's own (H1 V7)
        test_sh = self.dir / "tests" / "test.sh"
        kit = (self.dir / "tests" / "crucible" / "verify" / "runtime").is_dir() or (test_sh.exists() and "crucible_verify" in test_sh.read_text(errors="replace"))
        self.trust_mask = kit if trust_mask is None else trust_mask

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
            # M10 lab audit R-3 review round 2: the image inspect and the host put/get are infrastructure, not the patch's
            # grade: outside the scoring budget (runtime.grading.unbudgeted holds it while they block), so a budget expiry
            # is only ever the test run's own (it was delivered at the docker run's first await, labelled verifier_timeout,
            # and a control whose transfers happened to be fast scored the answer 0)
            with unbudgeted():
                got = self._docker("image", "inspect", "--format", "{{.Id}}", self.image).stdout.strip() or None
            if got != self.expected_image:
                return self._masked("check_infra", f"stale image: {self.image} is {got}, the task pins {self.expected_image}", t0)
        key = f"grades/{uuid.uuid4().hex[:12]}"
        art = gdir / "stage" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "model.patch").write_text(patch)
        (gdir / "stage" / "verifier").mkdir(exist_ok=True)
        with unbudgeted():
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
        if self.user:
            argv += ["--user", str(self.user)]
        envfile = None
        if self.verifier_env:  # an env file, never ``-e K=V`` (values would sit on the command line)
            import os
            import tempfile

            fd, envfile = tempfile.mkstemp(prefix="crucible-venv-", suffix=".env")
            with os.fdopen(fd, "w") as f:
                f.write("".join(f"{k}={v}\n" for k, v in self.verifier_env.items()))
            argv += ["--env-file", envfile]
        argv += ["--entrypoint", "bash", self.image, "-c", self.command]
        try:
            try:
                p = await asyncio.to_thread(self._docker, *argv, timeout=self.timeout)
            except subprocess.TimeoutExpired:
                self._docker("rm", "-f", name)
                return self._masked("verifier_timeout", f"grade exceeded {self.timeout:.0f}s; container {name} removed", t0)
            except asyncio.CancelledError:
                # M10 lab audit R-3: a scoring budget (grading.grade_in_place) cancels this await but not the thread or the
                # container: remove it before the finally deletes its staged mounts, and before grading.settle grades the
                # control in the same PerfGuard slot (an orphan still running the policy's hang contaminates the control,
                # masks it, and blames the task)
                try:
                    self._docker("rm", "-f", name)  # the thread's docker run then returns
                except Exception:  # noqa: BLE001, S110 -- removal is best effort; the cancellation still propagates
                    pass
                raise
            local = gdir / "verifier"
            with unbudgeted():
                self.host.get(f"{key}/verifier", local)
        finally:  # staged files never stay on the host (H1 V21: a timeout returned before the cleanup)
            if envfile:
                Path(envfile).unlink(missing_ok=True)
            try:
                self.host.exec(f"rm -rf {self.host.workroot}/{key}")
            except Exception:  # noqa: BLE001, S110 -- cleanup never overrides the grade
                pass
        (gdir / "console.log").write_text((p.stdout + p.stderr)[-50000:])
        try:
            report = json.loads((local / "report.json").read_text()) if (local / "report.json").exists() else {}
        except ValueError:
            report = {}
        if self.trust_mask and (local / "mask.json").exists():  # Crucible's kit masked the grade: keep its failure kind
            try:
                mk = json.loads((local / "mask.json").read_text())
            except ValueError:
                mk = {}
            kind, why = mk.get("failure_kind") or "check_infra", mk.get("reason") or "masked by the grader"
            if kind == "verifier_error" and str(why).startswith("non-finite reward"):
                kind = "non_finite_reward"  # plan.write_reward_files' own NaN/inf mask: answer-sensitive (M10 lab audit R-3 item 2)
            return self._masked(kind, why, t0)
        from crucible.interop.harbor.rewards import parse_reward_dir, scalar_reward

        rewards, problem = parse_reward_dir(local)
        reward = None
        if rewards is not None:
            reward, problem = scalar_reward(rewards)
        if reward is None:  # Harbor's rules: no, empty, unparseable, non-finite or ambiguous rewards are verifier errors
            kind = (problem or "no_reward").split(":")[0]
            if kind == "VerifierOutputParseError" and "non-finite" in (problem or ""):
                # M10 lab audit R-3 item 2: a NaN or infinite value (rewards.parse_reward_dir) is a reward the grader could not
                # turn into a number, which a patch can cause (the tests compute it, or the code under test writes it):
                # graded again, then decided by the control. Harbor's name stays in the reason
                kind = "non_finite_reward"
            return self._masked("no_reward_json" if kind == "RewardFileNotFoundError" else kind, f"{problem} (rc {p.returncode})", t0)
        gates = _gates(report if isinstance(report, dict) else {})
        shutil.rmtree(gdir / "stage", ignore_errors=True)
        return GradeReport(task=self.id, tier="full", checks=[], gates=gates, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"report": report, "rc": p.returncode})

    async def grade_control(self, workdir: Path) -> GradeReport | None:
        """M10 lab audit R-3: the task's control (``runtime.grading.settle``): the oracle patch graded exactly as a
        collected patch is (``grade_patch``, a fresh container of the same image). None without one."""
        if not self.oracle_patch.is_file():
            return None
        return await self.grade_patch(self.oracle_patch.read_text(), Path(workdir))
