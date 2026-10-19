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

M10 lab audit R-5 item 4: every host call (docker, the staging rsync and ssh) is a coroutine on
``world.exec._run_subprocess`` (off the event loop, bounded, its session killed past the bound or on a cancel), so a stalled
transfer or daemon freezes no other attempt. A cancelled or timed-out grade runs ``docker rm -f <name>`` first, and the staged
mounts (``<workroot>/grades/<key>`` on the host) are deleted only once the container is known gone: ``docker run`` returned,
or ``rm -f`` returned 0 or "No such container". If the removal cannot be confirmed the staging stays and
``<gdir>/staging_kept.json`` names the host, the path and the container (evidence for R-6's sweep): deleting the mounts
under a container that may still be running the policy's code is worse than a leaked directory. A host call past its
timeout raises ``HostTimeout`` (``transport_unreachable``, infrastructure); only ``docker run`` itself running past the
verifier's timeout is the answer's ``verifier_timeout``. Review round 1: and so is the fetch of ``<key>/verifier`` running past
its bound after ``docker run`` returned (``check_infra``, answer-sensitive: the directory is mounted read-write where the
patch's code ran, so its size is the answer's, not the host's).

M10 lab audit R-6: the grade container is named and labelled like every container Crucible starts (``world.lifecycle``:
``crucible.owner``, ``crucible.expires`` = the grade's timeout plus ``GRADE_MARGIN_S``, and ``crucible.run`` / ``crucible.attempt`` /
``crucible.lease`` inside a Runner attempt) and recorded in the attempt's scope before ``docker run`` (``Scope.started``; its outcome
is ``Scope.closed``, so ``teardown.json`` lists it). Review round 1: the grade command (the task's own, unchanged) runs under an
in-container bound of the timeout plus ``GRADE_MARGIN_S`` (``lifecycle.BOUND_SH``: ``timeout -k 30`` where the image has one), so a
grade a killed runner left ends by itself even on a host no later Runner sweeps; in a live run ``docker run``'s own timeout kills the
CLI and its ``rm -f`` runs long before that, and after a crash the next Runner sweep (this task names its host: ``docker_hosts``)
removes it through its ``crucible.expires`` label. A removal that follows a cancelled or failed ``docker run`` looks again ``RECHECK_S`` later
(a daemon can create the container after the client was killed, after the first ``rm -f`` answered "No such container"), and a
``BaseException`` the branches above do not catch still reaches a shielded removal in the ``finally``, before the staging is
deleted or ``staging_kept.json`` is written. The attempt scope's reclaim removes anything the attempt's label still lists.

The task can be pinned to a git ref (``ref=``): a historical run is re-graded against the task *as it was* when the
run was graded (research m9-m10/02 §6.1: task02's suite was tightened after its frontier run).
"""

from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import tomllib
import uuid
from collections.abc import Awaitable
from pathlib import Path
from typing import Any, TypeVar

from crucible.runtime.gpu import GpuHost, HostTimeout
from crucible.runtime.grading import unbudgeted
from crucible.runtime.identity import file_digest, tree_digests
from crucible.verify.runtime.plan import GradeReport
from crucible.world import lifecycle
from crucible.world.exec import ExecOut, _run_subprocess

_T = TypeVar("_T")
_BACKGROUND: set[asyncio.Future] = set()  # removals a second cancel left running: held so they are not collected mid-flight


async def _shielded(aw: Awaitable[_T], default: _T, *, unwinding: bool) -> _T:
    """Run ``aw`` to its end even if this task is cancelled meanwhile (M10 lab audit R-5 item 4: the removal of a grading
    container must finish before the staging goes, whatever cancelled the grade). A cancel that arrives while it runs
    leaves it to finish in the background; when this task is already unwinding from an earlier cancel or error
    (``unwinding``) that second cancel is absorbed and ``default`` returned (the first propagates), otherwise it is raised."""
    fut = asyncio.ensure_future(aw)
    _BACKGROUND.add(fut)
    fut.add_done_callback(_BACKGROUND.discard)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        if not unwinding:
            raise
        return default


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

    def identity_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: task content digests and declared image refs; a change refuses the resume. ``files`` is one digest
        per top-level entry of the task directory (``tests``, ``solution``, ``environment``, ``task.toml``, ``instruction.md``), so a
        refusal names the part that changed (``tasks.<id>.files.tests``); the image the grade runs in (and the digest it must
        be), the test command, the network mode, the verifier's env and user, the mask trust, the resource limits and the control's
        patch are what decides the grade. ``timeout`` (operational) is left out, and so is the host of a non-perf task
        (placement); a perf task's grade is a speedup measured on its host's hardware, so ``host`` (its name) is part of what it
        measures and moving it is refused (M10 lab audit R-4 review). The files include hidden ones (``environment/.dockerignore``,
        ``tests/.expected_output.json``): ``identity.tree_digests`` leaves out only version control and junk, as Harbor's task
        hash does. Stdlib only (through ``runtime.identity``): this module still needs no Harbor."""
        return {"files": tree_digests(self.dir), "image": self.image, "expected_image": self.expected_image, "command": self.command, "network": self.network,
                "verifier_env": self.verifier_env, "user": self.user, "trust_mask": self.trust_mask, "perf": self.perf, "gpus": self.gpus, "cpus": self.cpus,
                "memory_mb": self.memory_mb, "oracle_patch": file_digest(self.oracle_patch), "host": self.host.name if self.perf else None}

    def docker_hosts(self) -> set[str]:
        """The docker host the grading container runs on (``''`` is the local daemon, which every start sweep covers anyway): what
        ``Runner._plan_hosts`` adds to the start sweep, so a grade a killed run left on a lab host is found by any later run (M10 lab
        audit R-6 review round 1)."""
        return {self.host.docker_host} if self.host.docker_host else set()

    async def _docker(self, *a: str, timeout: float = 120) -> ExecOut:
        """One docker CLI call on the grading host, off the event loop (M10 lab audit R-5 item 4). Past ``timeout`` the CLI's
        session is killed; the result is ``timed_out`` for ``docker run`` (the grade's own ``verifier_timeout``, the caller
        removes the container), and ``HostTimeout`` for anything else (an inspect or ``rm`` that did not answer is the
        host's, never the answer's)."""
        env = {"DOCKER_HOST": self.host.docker_host} if self.host.docker_host else None
        out = await _run_subprocess(["docker", *a], cwd=None, env=env, timeout_s=timeout, stdin=None)
        if out.timed_out and a[:1] != ("run",):
            raise HostTimeout(f"docker {a[0] if a else ''} on {self.host.name} ran past {timeout:g}s")
        return out

    async def _remove(self, name: str, *, recheck: bool = False) -> bool:
        """``docker rm -f <name>``: True when the container is known gone (removed, or "No such container"), False when that
        could not be confirmed (an error, a daemon that does not answer): the staging then stays (M10 lab audit R-5 item 4).
        M10 lab audit R-6, ``recheck`` (after a ``docker run`` that was cancelled or failed): once the first removal is confirmed,
        look again ``lifecycle.RECHECK_S`` later: a daemon that received the run request before its client was killed creates the
        container after the first ``rm -f`` answered "No such container". Gone only if both are confirmed."""
        for round_ in range(2 if recheck else 1):
            if round_:
                await asyncio.sleep(lifecycle.RECHECK_S)
            try:
                out = await self._docker("rm", "-f", name)
            except Exception:  # noqa: BLE001 -- removal is best effort; the caller keeps the staging when it is False
                return False
            if not (out.rc == 0 or "no such container" in out.stderr.lower()):
                return False
        return True

    async def _removed(self, name: str) -> tuple[bool, str]:
        """``_remove`` in the form ``Scope`` retries a container's removal in (its reclaim, when the grade could not confirm it)."""
        ok = await self._remove(name)
        return ok, "" if ok else "not confirmed removed"

    def _masked(self, kind: str, reason: str, t0: float) -> GradeReport:
        import time

        return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0,
                           infra={"status": "masked", "failure_kind": kind, "reason": reason}, trusted=True, elapsed_s=round(time.monotonic() - t0, 2))

    async def grade_patch(self, patch: str, gdir: Path) -> GradeReport:
        import time

        t0 = time.monotonic()
        gdir = Path(gdir)
        if self.expected_image:
            # M10 lab audit R-3 review round 2: the image inspect and the host put/get are infrastructure, not the patch's
            # grade: outside the scoring budget (runtime.grading.unbudgeted holds it while they run), so a budget expiry
            # is only ever the test run's own (it was delivered at the docker run's first await, labelled verifier_timeout,
            # and a control whose transfers happened to be fast scored the answer 0). M10 lab audit R-5 item 4: they are
            # awaited now (off the loop), and one that runs past its own timeout is a HostTimeout, never the answer's
            with unbudgeted():
                got = (await self._docker("image", "inspect", "--format", "{{.Id}}", self.image)).stdout.strip() or None
            if got != self.expected_image:
                return self._masked("check_infra", f"stale image: {self.image} is {got}, the task pins {self.expected_image}", t0)
        key = f"grades/{uuid.uuid4().hex[:12]}"
        art = gdir / "stage" / "artifacts"
        art.mkdir(parents=True, exist_ok=True)
        (art / "model.patch").write_text(patch)
        (gdir / "stage" / "verifier").mkdir(exist_ok=True)
        name = f"crucible-hgrade-{uuid.uuid4().hex[:10]}"
        envfile = None
        root: str | None = None  # the host path of this grade's staging (known once ``apath`` answered)
        started = gone = unwinding = False  # ``docker run`` was issued; its container is known gone; an exception is propagating
        tried = False  # a removal was attempted by one of the branches around ``docker run``
        scope = lifecycle.current()  # M10 lab audit R-6: who this container belongs to (the attempt's scope under a Runner)
        # M10 lab audit R-5 item 4 (critic amendment 4): the try opens before the first transfer, so a cancel or a HostTimeout
        # during an rsync still reaches the cleanup below (the transfers are awaits now, where they used to be uninterruptible)
        try:
            with unbudgeted():
                root = await self.host.apath(key)  # one absolute path: the cleanup runs from any cwd (the relative one was the local host's $HOME)
                tests = await self.host.aput(self.dir / "tests", f"{key}/tests")
                arts = await self.host.aput(art, f"{key}/artifacts")
                out = await self.host.aput(gdir / "stage" / "verifier", f"{key}/verifier")
            bound = int(self.timeout + lifecycle.GRADE_MARGIN_S)  # the in-container bound (``BOUND_SH``) and the expiry label are one number
            expires = time.time() + bound
            argv = ["run", "--rm", "--init", "--name", name, "--label", "crucible.sandbox=1", *scope.labels(expires=expires), "--shm-size", "8g",
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
            if self.verifier_env:  # an env file, never ``-e K=V`` (values would sit on the command line)
                import os
                import tempfile

                fd, envfile = tempfile.mkstemp(prefix="crucible-venv-", suffix=".env")
                with os.fdopen(fd, "w") as f:
                    f.write("".join(f"{k}={v}\n" for k, v in self.verifier_env.items()))
                argv += ["--env-file", envfile]
            # M10 lab audit R-6 review round 1: ``bash -c <command>`` as before, under ``timeout -k 30 <bound>`` where the image has one
            argv += ["--entrypoint", "bash", self.image, "-c", lifecycle.BOUND_SH, "crucible-bound", str(bound), "bash", "-c", self.command]
            scope.started(name, self.host.docker_host, renewable=False, deadline=expires, remove=lambda: self._removed(name))  # before the run
            started = True  # from here a container may exist, whatever happens to the call that created it
            try:
                p = await self._docker(*argv, timeout=self.timeout)
            except asyncio.CancelledError:
                # M10 lab audit R-3: a scoring budget (grading.grade_in_place) cancels this await; M10 lab audit R-5 item 4:
                # the cancelled call killed the docker CLI's session, which leaves the container running on the daemon.
                # Remove it before the finally deletes its staged mounts, and before grading.settle grades the control in
                # the same PerfGuard slot (an orphan still running the policy's hang contaminates the control, masks it, and
                # blames the task). Shielded: a second cancel does not abandon the removal. M10 lab audit R-6: and looked at again
                # after ``RECHECK_S``, for a daemon that creates the container after its client was killed
                tried = True
                gone = await _shielded(self._remove(name, recheck=True), False, unwinding=True)
                raise
            except Exception:  # noqa: BLE001 -- the docker client failed to start or died: the container may still exist
                tried = True
                gone = await _shielded(self._remove(name, recheck=True), False, unwinding=True)
                raise
            if p.timed_out:
                # the grade ran past its own timeout: the answer's verifier_timeout (the CLI's session is dead, the container
                # is not: the CLI ran to its bound, so the daemon had the request long ago and no recheck is needed); the staging
                # goes only once the container does
                tried = True
                gone = await _shielded(self._remove(name), False, unwinding=False)
                return self._masked("verifier_timeout", f"grade exceeded {self.timeout:.0f}s; container {name} "
                                    + ("removed" if gone else "NOT removed: staging kept"), t0)
            gone = True  # ``docker run --rm`` returned: the container exited and was removed
            local = gdir / "verifier"
            try:
                with unbudgeted():
                    await self.host.aget(f"{key}/verifier", local)
            except HostTimeout:
                # M10 lab audit R-5 review round 1: ``<key>/verifier`` is bind-mounted read-write into the container the patch's own
                # code ran in (``-v {out}:/logs/verifier``), so its size is the answer's: a patch whose tests write gigabytes there
                # makes this transfer outlive its bound. A HostTimeout is host infrastructure (``transport_unreachable``, counted on
                # the host's breaker, never controlled), which would let one answer open ``host:<name>`` for every sample and
                # keep its own 0. The container already ran to its end (``gone``): an answer-sensitive grade failure
                # (``check_infra``), graded again and decided by the task's control. Transfers whose size the answer does not set
                # (tests, artifacts, ``apath``, ``mkdir``) stay HostTimeout
                return self._masked("check_infra", f"the verifier output of container {name} could not be fetched within its bound "
                                    "(its size is set by the graded patch)", t0)
        except BaseException:
            unwinding = True  # a cancel or an error is propagating: a second cancel during the cleanup is absorbed
            raise
        finally:  # staged files never stay on the host (H1 V21: a timeout returned before the cleanup)
            if envfile:
                Path(envfile).unlink(missing_ok=True)
            if started and not gone and not tried:
                # M10 lab audit R-6: a BaseException the branches above do not catch (they cover CancelledError and Exception) left
                # the container unremoved: remove it here, shielded, before the staging decision below
                gone = await _shielded(self._remove(name, recheck=True), False, unwinding=True)
            if started:
                scope.closed(name, ok=gone, error=None if gone else "not confirmed removed")  # teardown.json, and the scope's reclaim retries it
            if root is not None and root.rstrip("/").endswith(key):
                if not started or gone:
                    try:
                        await _shielded(self.host.aexec(f"rm -rf {shlex.quote(root)}"), None, unwinding=unwinding)
                    except Exception:  # noqa: BLE001, S110 -- cleanup never overrides the grade
                        pass
                else:  # M10 lab audit R-5 item 4: never delete the mounts under a container that may still be running
                    try:
                        (gdir / "staging_kept.json").write_text(json.dumps({
                            "host": self.host.name, "path": root, "container": name,
                            "reason": "the grading container could not be confirmed removed (M10 lab audit R-5)"}))
                    except OSError:
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
            return self._masked("no_reward_json" if kind == "RewardFileNotFoundError" else kind, f"{problem} (rc {p.rc})", t0)
        gates = _gates(report if isinstance(report, dict) else {})
        shutil.rmtree(gdir / "stage", ignore_errors=True)
        return GradeReport(task=self.id, tier="full", checks=[], gates=gates, reward=reward, outcome=reward, infra={"status": "ok"}, trusted=True,
                           elapsed_s=round(time.monotonic() - t0, 2), meta={"report": report, "rc": p.rc})

    async def grade_control(self, workdir: Path) -> GradeReport | None:
        """M10 lab audit R-3: the task's control (``runtime.grading.settle``): the oracle patch graded exactly as a
        collected patch is (``grade_patch``, a fresh container of the same image). None without one."""
        if not self.oracle_patch.is_file():
            return None
        return await self.grade_patch(self.oracle_patch.read_text(), Path(workdir))
