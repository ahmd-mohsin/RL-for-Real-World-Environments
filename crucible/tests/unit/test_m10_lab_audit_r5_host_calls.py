"""M10 lab audit R-5, slice 1 (item 4, Acceptance 4 and 5): host calls never block the event loop, and a grade's container
is gone before its staging.

- every git, docker, rsync and ssh call an attempt makes runs through ``world.exec._run_subprocess``: a sibling attempt keeps
  stepping while one git call blocks (Acceptance 4), the call's session is killed on a cancel and past its timeout, and a
  timeout is infrastructure (``crucible:empty_collection`` for a collect, ``HostTimeout`` / ``transport_unreachable`` for a
  host transfer), never the agent's "changed nothing" or the answer's ``verifier_timeout``;
- ``HarborDirTask`` runs ``docker rm -f <name>`` when a grade is cancelled or times out and deletes its staging only once
  the container is gone; otherwise the staging stays and ``staging_kept.json`` names it (Acceptance 5); a cancel during a
  transfer still reaches the cleanup;
- ``HarborBackend.docker`` returns the CLI's stdout as bytes (the grader guard's tar streams), bounded and off the loop.

POSIX only where a fake ``git`` / ``rsync`` / ``docker`` shell script is run (sessions, ``killpg``). No Docker, no network.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from crucible.runtime.infra import classify_exception
from crucible.world.exec import ExecOut

posix = pytest.mark.skipif(sys.platform == "win32", reason="POSIX sessions (killpg) and shell-script fakes")


def _fake_bin(tmp_path: Path, name: str, body: str) -> Path:
    b = tmp_path / "bin"
    b.mkdir(exist_ok=True)
    (b / name).write_text(f"#!/bin/sh\n{body}\n")
    (b / name).chmod(0o755)
    return b


def _use(monkeypatch, b: Path) -> None:
    monkeypatch.setenv("PATH", f"{b}:{os.environ['PATH']}")
    monkeypatch.delenv("DOCKER_HOST", raising=False)


async def _ticks(stop: asyncio.Event) -> int:
    """How often the event loop got to run a 10 ms sleep: a blocked loop gets almost none."""
    n = 0
    while not stop.is_set():
        await asyncio.sleep(0.01)
        n += 1
    return n


# ---------------------------------------------------------------- collect: git off the loop, bounded


@posix
async def test_sibling_attempts_keep_stepping_while_one_git_call_blocks(tmp_path, monkeypatch):
    """Acceptance 4: a git that never returns blocks one collection, not the loop; cancelling it kills git's session."""
    from crucible.runtime.collect import collect

    pidfile = tmp_path / "git.pid"
    _use(monkeypatch, _fake_bin(tmp_path, "git", 'echo $$ > "$FAKE_PIDFILE"; exec sleep 600'))
    monkeypatch.setenv("FAKE_PIDFILE", str(pidfile))
    ws = tmp_path / "ws"
    ws.mkdir()
    t = asyncio.ensure_future(collect(ws))
    stop = asyncio.Event()
    ticker = asyncio.ensure_future(_ticks(stop))
    await asyncio.sleep(1.0)
    stop.set()
    ticks = await ticker
    assert ticks >= 50 and not t.done(), ticks
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    pid = int(pidfile.read_text().strip())
    for _ in range(100):  # the session was killed with the cancelled attempt: nothing outlives it
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            break
        await asyncio.sleep(0.05)
    else:
        pytest.fail(f"git (pid {pid}) outlived the cancelled collection")


@posix
async def test_a_git_call_past_its_timeout_is_a_masked_collect_failure_never_a_scored_zero(tmp_path, monkeypatch):
    from crucible.runtime.collect import classify, collect, repo_broken
    from crucible.verify import collect as vcollect

    _use(monkeypatch, _fake_bin(tmp_path, "git", "exec sleep 600"))
    monkeypatch.setattr(vcollect, "GIT_TIMEOUT_S", 0.3)
    ws = tmp_path / "ws"
    (ws / ".git" / "info").mkdir(parents=True)  # a repository directory, so ".git deleted" is not what fires
    c = await collect(ws)
    assert c.patch == "" and c.tree_changed is True, c  # unknown counts as changed: never "the agent did nothing"
    assert any("timed out" in e for e in c.errors), c.errors
    assert c.stalled is True  # review round 1: what the re-collect of the preserved workspace reads (``WorkspaceHarness``)
    f = classify(c)
    assert f is not None and f.kind == "crucible:empty_collection" and f.masks and f.retryable, f
    assert await repo_broken(ws, "abc") == []  # a git that did not answer is not the policy breaking its repository


@posix
async def test_a_timed_out_add_never_grades_a_half_built_index(tmp_path, monkeypatch):
    """Critic amendment 3: ``git add`` killed mid-way, ``git diff`` would still print a hunk: it is never the patch."""
    from crucible.runtime.collect import classify, collect
    from crucible.verify import collect as vcollect

    # the subcommand is the first word that names one: ``git -C <ws> -c k=v ... <subcommand> ...`` (GIT_FLAGS sit before it, review round 1)
    body = ('sub=; for a in "$@"; do case "$a" in add|diff|status|ls-files|reset) sub=$a; break;; esac; done\n'
            'case "$sub" in add) exec sleep 600;; diff) printf "diff --git a/x b/x\\n+x\\n";; esac\nexit 0')
    _use(monkeypatch, _fake_bin(tmp_path, "git", body))
    monkeypatch.setattr(vcollect, "GIT_TIMEOUT_S", 0.3)
    ws = tmp_path / "ws"
    (ws / ".git" / "info").mkdir(parents=True)
    lock = ws / ".git" / "index.lock"
    lock.write_text("")  # what the killed git leaves behind
    c = await collect(ws)
    assert c.patch == "" and "b/x" not in c.patch, c
    assert any("git add timed out" in e for e in c.errors), c.errors
    f = classify(c)
    assert f is not None and f.kind == "crucible:empty_collection" and f.masks, f  # masked and re-collected, never graded
    assert not lock.exists()  # the re-collect and the final git reset can run


@posix
async def test_every_collection_git_call_runs_without_fsmonitor_or_hooks(tmp_path, monkeypatch):
    """Review round 1: the repository is the agent's. A ``core.fsmonitor`` command or a hooks path in its own ``.git/config`` is a
    command host git would run for a plain ``status`` / ``add``; every call of the hook now overrides both per call, and ``git diff``
    runs no textconv and no external diff driver."""
    from crucible.runtime.collect import collect
    from crucible.verify import collect as vcollect

    log = tmp_path / "git.log"
    _use(monkeypatch, _fake_bin(tmp_path, "git", f'echo "$@" >> {log}\nexit 0'))
    ws = tmp_path / "ws"
    (ws / ".git" / "info").mkdir(parents=True)
    await collect(ws, base="abc")
    await collect(ws, base="abc", mode="legacy")
    calls = log.read_text().splitlines()
    assert calls and all("core.fsmonitor=false" in c and "core.hooksPath=/dev/null" in c for c in calls), calls  # legacy chain included
    assert vcollect.GIT_FLAGS == ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=/dev/null"]
    diffs = [c for c in calls if " diff " in c]
    assert diffs and all("--no-textconv" in c and "--no-ext-diff" in c for c in diffs), diffs


@posix
async def test_a_planted_fsmonitor_command_does_not_stall_the_collection(tmp_path, monkeypatch):
    """With a real git: ``core.fsmonitor`` in the workspace's config names a script that sleeps. Collection overrides it, so it neither
    stalls nor turns the agent's work into a masked ``empty_collection``."""
    import shutil
    import subprocess

    from crucible.runtime.collect import classify, collect
    from crucible.verify import collect as vcollect

    if not shutil.which("git"):
        pytest.skip("needs git")
    ws = tmp_path / "ws"
    ws.mkdir()
    g = lambda *a: subprocess.run(["git", "-C", str(ws), *a], capture_output=True, check=True)  # noqa: E731
    g("init", "-q")
    g("-c", "user.email=a@b", "-c", "user.name=a", "commit", "-q", "--allow-empty", "-m", "base")
    base = g("rev-parse", "HEAD").stdout.decode().strip()
    hook = tmp_path / "fsm.sh"
    hook.write_text("#!/bin/sh\nsleep 30\n")
    hook.chmod(0o755)
    g("config", "core.fsmonitor", str(hook))
    (ws / "x.txt").write_text("the agent's work\n")
    monkeypatch.setattr(vcollect, "GIT_TIMEOUT_S", 10.0)
    t0 = asyncio.get_running_loop().time()
    c = await collect(ws, base=base)
    assert asyncio.get_running_loop().time() - t0 < 8 and c.stalled is False, c  # the hook never ran
    assert "x.txt" in c.patch and classify(c) is None


async def test_a_recollect_that_stalls_again_is_the_policys_agent_broke_env(tmp_path, monkeypatch):
    """The first collection of the workspace stalled (masked ``empty_collection``, the host's: the workspace is preserved). The
    re-collect of the same tree stalling too means the repository's own config or attributes decide whether git answers (a filter
    driver the policy planted): its outcome, scored 0, never a second harness mask. A re-collect that merely failed stays
    ``empty_collection``."""
    from types import SimpleNamespace

    from crucible.core.schema.run import Failure, RunRecord
    from crucible.runtime.collect import Collected
    from crucible.runtime.harness import workspace as W
    from crucible.runtime.harness.base import Attempt

    a1 = tmp_path / "samples" / "s" / "a1"
    (a1 / "preserved_ws").mkdir(parents=True)
    (a1 / "base_sha").write_text("abc\n")
    rec = RunRecord(run_id="r", world="w", policy="p", started_at="2026-10-19T00:00:00Z", attempt="s-a1", outcome="completed", reward=1.0)
    rec.mark_failure(Failure(kind="crucible:empty_collection", stage="collect", reason="git add timed out after 300s", retryable=True))
    (a1 / "record.json").write_text(rec.model_dump_json())

    def attempt() -> Attempt:
        return Attempt(sample=None, id="s-a2", n=2, dir=tmp_path / "samples" / "s" / "a2", task=None,  # type: ignore[arg-type]
                       policy=SimpleNamespace(spend=lambda: (0.0, 0.0)))

    async def stalled(ws, *, base="HEAD", mode="tolerant"):
        return Collected(patch="", mode=mode, tree_changed=True, errors=["git add timed out after 300s"], stalled=True)

    monkeypatch.setattr(W, "collect", stalled)
    res = await W.WorkspaceHarness().run(attempt())
    assert res.failure is not None and res.failure.kind == "crucible:agent_broke_env" and not res.failure.masks and "does not answer" in res.failure.reason, res.failure

    async def failed(ws, *, base="HEAD", mode="tolerant"):
        return Collected(patch="", mode=mode, tree_changed=True, errors=["fatal: unable to write new index file"])

    monkeypatch.setattr(W, "collect", failed)
    res = await W.WorkspaceHarness().run(attempt())
    assert res.failure is not None and res.failure.kind == "crucible:empty_collection" and res.failure.masks, res.failure


# ---------------------------------------------------------------- GpuHost: rsync and ssh off the loop, a timeout is infrastructure


@posix
async def test_gpu_host_transfers_run_off_the_loop_and_time_out_as_infrastructure(tmp_path, monkeypatch):
    from crucible.runtime.gpu import GpuHost, HostTimeout

    _use(monkeypatch, _fake_bin(tmp_path, "rsync", "exec sleep 600"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "x.txt").write_text("x")
    h = GpuHost(name="h")
    stop = asyncio.Event()
    ticker = asyncio.ensure_future(_ticks(stop))
    try:
        with pytest.raises(HostTimeout) as ei:
            await h.aput(tmp_path / "x.txt", "k/x", timeout=0.5)
    finally:
        stop.set()
    assert await ticker >= 20
    ex = ei.value
    assert isinstance(ex, ConnectionError) and not isinstance(ex, TimeoutError)  # never the answer's verifier_timeout
    assert classify_exception(ex, "verification").kind == "transport_unreachable"


@posix
async def test_gpu_host_aexec_reads_the_local_shell_and_bounds_it(tmp_path, monkeypatch):
    from crucible.runtime.gpu import GpuHost, HostTimeout

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    h = GpuHost(name="h")
    assert await h.aexec("echo hi") == (0, "hi\n")
    assert (await h.apath("k/x")).endswith("/crucible_runtime/k/x")
    with pytest.raises(HostTimeout):
        await h.aexec("sleep 5", timeout=0.2)


# ---------------------------------------------------------------- HarborBackend.docker


@posix
async def test_harbor_backend_docker_is_bounded_and_off_the_loop(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from crucible.interop.harbor.world_backend import HarborBackend

    b = object.__new__(HarborBackend)
    _use(monkeypatch, _fake_bin(tmp_path, "docker", "exec sleep 600"))
    stop = asyncio.Event()
    ticker = asyncio.ensure_future(_ticks(stop))
    try:
        rc, out, err = await b.docker("top", "cid", timeout=0.5)
    finally:
        stop.set()
    assert (rc, out) == (-9, b"") and "timed out" in err, (rc, out, err)
    assert await ticker >= 20
    _use(monkeypatch, _fake_bin(tmp_path, "docker", "echo hello; exit 3"))
    assert await b.docker("top", "cid") == (3, b"hello\n", "")


@posix
async def test_harbor_backend_docker_keeps_binary_stdout_untouched(tmp_path, monkeypatch):
    """Critic amendment 1: the grader guard reads ``docker cp <cid>:<path> -`` tar streams (and a ``surrogateescape``
    ``docker diff``) from stdout; decoding it with ``errors="replace"`` would corrupt every non-UTF-8 byte."""
    pytest.importorskip("harbor")
    from crucible.interop.harbor.world_backend import HarborBackend

    b = object.__new__(HarborBackend)
    _use(monkeypatch, _fake_bin(tmp_path, "docker", r"printf '\377\376'"))
    assert await b.docker("cp", "cid:/x", "-") == (0, b"\xff\xfe", "")


# ---------------------------------------------------------------- HarborDirTask: the container goes before the staging


class _AsyncHost:
    """A ``GpuHost`` stand-in: ``calls`` records every host call and docker call, in order."""

    name = "local"
    docker_host = None
    workroot = "w"

    def __init__(self):
        self.calls: list[tuple] = []

    async def apath(self, rel):
        return f"/h/{self.workroot}/{rel}"

    async def aput(self, local, rel, *, timeout=None):
        self.calls.append(("put", rel))
        return str(local)

    async def aget(self, rel, local, *, timeout=None):
        self.calls.append(("get", rel))

    async def aexec(self, cmd, *, timeout=120):
        self.calls.append(("exec", cmd))
        return 0, ""

    def docker_gpu_flags(self):
        return []


def _dir_task(tmp_path: Path) -> Path:
    td = tmp_path / "dirtask"
    (td / "tests").mkdir(parents=True)
    (td / "task.toml").write_text('[task]\nname = "t"\n')
    (td / "tests" / "test.sh").write_text("exit 0\n")
    return td


def _index(host: _AsyncHost, pred) -> int:
    return next(i for i, c in enumerate(host.calls) if pred(c))


def _is_rm_container(c: tuple) -> bool:
    return c[:2] == ("docker", "rm")


def _is_rm_staging(c: tuple) -> bool:
    return c[0] == "exec" and "rm -rf" in c[1]


async def _cancel_once_running(host: _AsyncHost, dt, tmp_path: Path) -> asyncio.Future:
    t = asyncio.ensure_future(dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g"))
    for _ in range(500):
        if any(c[:2] == ("docker", "run") for c in host.calls):
            break
        await asyncio.sleep(0.01)
    else:
        pytest.fail("docker run was never issued")
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    return t


async def test_a_cancelled_dir_grade_removes_its_container_before_its_staging(tmp_path):
    """Acceptance 5: the cancelled grade's ``docker run`` client is dead but its container is not; ``docker rm -f`` runs and
    only then are the staged mounts deleted."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host, removed = _AsyncHost(), asyncio.Event()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            await removed.wait()  # the policy's hang: the container runs until it is removed
            return ExecOut(rc=137, stdout="", stderr="")
        if a[0] == "rm":
            removed.set()
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    t = await _cancel_once_running(host, dt, tmp_path)
    assert t.cancelled()
    rm, cleanup = _index(host, _is_rm_container), _index(host, _is_rm_staging)
    assert rm < cleanup and str(host.calls[rm][2]).startswith("crucible-hgrade-"), host.calls
    assert host.calls[cleanup][1].startswith("rm -rf /h/w/grades/")  # an absolute path: it runs from any cwd
    assert not (tmp_path / "g" / "staging_kept.json").exists()


async def test_staging_is_kept_when_the_grading_container_cannot_be_removed(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host = _AsyncHost()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            await asyncio.sleep(3600)
        return ExecOut(rc=1, stdout="", stderr="Cannot connect to the Docker daemon")

    dt._docker = docker  # type: ignore[method-assign]
    t = await _cancel_once_running(host, dt, tmp_path)
    assert t.cancelled()
    assert not any(_is_rm_staging(c) for c in host.calls), host.calls  # the mounts stay under a container that may still run
    kept = json.loads((tmp_path / "g" / "staging_kept.json").read_text())
    assert kept["container"].startswith("crucible-hgrade-") and kept["host"] == "local" and kept["path"].startswith("/h/w/grades/"), kept


async def test_a_timed_out_dir_grade_removes_its_container_then_its_staging_and_is_verifier_timeout(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host = _AsyncHost()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            return ExecOut(rc=-9, stdout="", stderr="", timed_out=True)  # the CLI ran past the verifier's timeout
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    rep = await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")
    assert rep.infra["failure_kind"] == "verifier_timeout" and "removed" in rep.infra["reason"] and "NOT" not in rep.infra["reason"], rep.infra
    assert _index(host, _is_rm_container) < _index(host, _is_rm_staging)


async def test_a_timed_out_dir_grade_whose_container_stays_keeps_its_staging(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host = _AsyncHost()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            return ExecOut(rc=-9, stdout="", stderr="", timed_out=True)
        return ExecOut(rc=1, stdout="", stderr="Error response from daemon: unreachable")

    dt._docker = docker  # type: ignore[method-assign]
    rep = await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")
    assert rep.infra["failure_kind"] == "verifier_timeout" and "NOT removed" in rep.infra["reason"], rep.infra
    assert not any(_is_rm_staging(c) for c in host.calls) and (tmp_path / "g" / "staging_kept.json").exists()


async def test_a_container_that_is_already_gone_counts_as_removed(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host = _AsyncHost()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            await asyncio.sleep(3600)
        return ExecOut(rc=1, stdout="", stderr="Error response from daemon: No such container: x")

    dt._docker = docker  # type: ignore[method-assign]
    await _cancel_once_running(host, dt, tmp_path)
    assert _index(host, _is_rm_container) < _index(host, _is_rm_staging)


async def test_a_cancel_during_a_transfer_still_deletes_the_staging(tmp_path):
    """Critic amendment 4: the transfers are awaits now; a cancel inside one reaches the cleanup (no container yet, so no
    ``docker rm`` is needed and the staging goes at once)."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    class Slow(_AsyncHost):
        async def aput(self, local, rel, *, timeout=None):
            self.calls.append(("put", rel))
            await asyncio.sleep(3600)

    host = Slow()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]
    t = asyncio.ensure_future(dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g"))
    for _ in range(500):
        if any(c[0] == "put" for c in host.calls):
            break
        await asyncio.sleep(0.01)
    t.cancel()
    await asyncio.gather(t, return_exceptions=True)
    assert t.cancelled()
    assert any(_is_rm_staging(c) and "/h/w/grades/" in c[1] for c in host.calls), host.calls
    assert not any(c[0] == "docker" for c in host.calls)


async def test_a_dir_grade_image_inspect_past_its_timeout_is_host_infrastructure_not_the_answers(tmp_path, monkeypatch):
    from crucible.interop.harbor import verify_dir
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.gpu import HostTimeout
    from crucible.runtime.grading import grade_in_place

    async def fake(argv, **kw):
        return ExecOut(rc=-9, stdout="", stderr="", timed_out=True)

    monkeypatch.setattr(verify_dir, "_run_subprocess", fake)
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=_AsyncHost(), expected_image="sha256:x")  # type: ignore[arg-type]
    with pytest.raises(HostTimeout):
        await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")
    reps, f = await grade_in_place(lambda d: dt.grade_patch("diff --git a/x b/x\n", d), tmp_path / "gr", retries=0)
    assert reps == [] and f is not None and f.kind == "transport_unreachable" and f.who == "infra", f  # never verifier_timeout


async def test_a_verifier_output_fetch_past_its_bound_is_the_answers_not_the_hosts(tmp_path):
    """Review round 1: ``<key>/verifier`` is mounted read-write where the graded patch's own code ran, so a patch that floods it makes
    the fetch outlive its bound. That is an answer-sensitive grade failure (``check_infra``: graded again, decided by the control), not
    a ``HostTimeout`` (``transport_unreachable``), which would count against ``host:<name>`` and let one answer pause every sample on
    the host. The container ran to its end, so its staging goes."""
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.breaker import SampleKeys, counted
    from crucible.runtime.gpu import HostTimeout
    from crucible.runtime.grading import answer_sensitive, grade_failure

    class Flood(_AsyncHost):
        async def aget(self, rel, local, *, timeout=None):
            self.calls.append(("get", rel))
            raise HostTimeout(f"fetching {rel} ran past 1800s")

    host = Flood()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    rep = await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")  # a masked report, not a raised HostTimeout
    assert rep.infra["status"] == "masked" and rep.infra["failure_kind"] == "check_infra", rep.infra
    f = grade_failure(rep)
    assert f is not None and f.kind == "verifier_error" and f.detail == "check_infra" and answer_sensitive(f), f
    keys = SampleKeys(provider="provider:p", host="host:local", task="task:t|h@1")
    assert counted(f, keys) == (keys.task,)  # the task's key, never the host's
    assert any(c[0] == "get" and c[1].endswith("/verifier") for c in host.calls)
    assert any(_is_rm_staging(c) for c in host.calls), host.calls  # the container is gone (docker run returned): its staging is removed


# ---------------------------------------------------------------- the RLM kernel's docker calls: the same bounded path


async def test_the_rlm_kernel_launchers_call_docker_through_the_bounded_host_path(monkeypatch):
    """Review round 1: ``DockerLauncher._run`` (``docker run -d``, ``rm -f``, the kernel push) and ``ExecLauncher.run`` were asyncio's
    child watcher with no timeout, so a wedged daemon held the attempt, its slot and the run's lease. They go through
    ``world.exec._run_subprocess_raw`` now (patched here): bounded, off the loop, the session killed past the bound."""
    from crucible.actors.rlm import kernel as K

    calls: list[tuple] = []
    answer: list = [(0, b"", b"", False, 0.0)]

    async def fake(argv, **kw):
        calls.append((list(argv), kw["timeout_s"], kw["stdin"]))
        if isinstance(answer[0], BaseException):
            raise answer[0]
        return answer[0]

    monkeypatch.setattr(K, "_run_subprocess_raw", fake)
    launcher = K.DockerLauncher(docker="dkr")
    await launcher._run("run", "-d", "img")
    await launcher._run("rm", "-f", "c")
    await launcher._run("exec", "-i", "c", "sh", "-c", "cat > /tmp/x", timeout=60.0, stdin=b"payload")  # the kernel push
    assert [(c[0][:2], c[1], c[2]) for c in calls] == [(["dkr", "run"], 900.0, None), (["dkr", "rm"], 120.0, None), (["dkr", "exec"], 60.0, b"payload")]

    answer[0] = (-9, b"", b"", True, 900.0)  # the CLI ran past its bound: its session was killed, the container may exist
    with pytest.raises(K.KernelStartError) as ei:
        await launcher._run("run", "-d", "img")
    assert ei.value.failure_kind == "crucible:env_container_lost" and "ran past" in str(ei.value)
    answer[0] = (1, b"", b"Error response from daemon: No such container: c", False, 0.1)
    with pytest.raises(K.KernelStartError) as ei:
        await launcher._run("exec", "c", "true")
    assert ei.value.failure_kind == "crucible:env_container_lost"  # classified as before
    answer[0] = FileNotFoundError("dkr")
    with pytest.raises(K.KernelStartError) as ei:
        await launcher._run("run", "img")
    assert ei.value.failure_kind == "crucible:env_container_lost" and "not available" in str(ei.value)

    el = K.ExecLauncher(["docker", "exec", "-i", "c"])
    answer[0] = (0, b"hi\n", b"warn\n", False, 0.1)
    calls.clear()
    assert await el.run("echo", "hi", timeout=5) == (0, "hi\nwarn\n")
    assert calls == [(["docker", "exec", "-i", "c", "echo", "hi"], 5, None)]
    answer[0] = (-9, b"", b"", True, 5.0)
    assert await el.run("sleep", "99", timeout=5) == (-1, "timeout")


# ---------------------------------------------------------------- OpenEnv client mode: the Docker start is awaited


@posix
async def test_an_openenv_docker_client_starts_off_the_loop(tmp_path, monkeypatch):
    """Critic amendment 10: ``ClientEnv`` started its container with ``subprocess.run`` and a ``time.sleep`` poll on the loop
    (up to 15 minutes in an attempt's seed phase); ``astart`` awaits the CLI and the health poll."""
    import threading
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    from crucible.interop.openenv.import_env import ClientEnv

    class Health(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            self.send_response(200)
            self.end_headers()

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), Health)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        log = tmp_path / "docker.log"
        body = (f'echo "$@" >> {log}\ncase "$1" in run) echo cid; sleep 0.5;; port) echo 127.0.0.1:{srv.server_port};; esac\nexit 0')
        _use(monkeypatch, _fake_bin(tmp_path, "docker", body))
        for k in ("NO_PROXY", "no_proxy"):
            monkeypatch.setenv(k, "127.0.0.1")  # the health poll goes straight to the local stub
        env = ClientEnv(docker_image="img", environment={"A": "1"})
        assert env.client is None and env.container is None  # nothing started by the constructor
        stop = asyncio.Event()
        ticker = asyncio.ensure_future(_ticks(stop))
        try:
            await env.astart()
        finally:
            stop.set()
        assert await ticker >= 20  # the 0.5 s ``docker run`` did not block the loop
        assert env.base_url == f"http://127.0.0.1:{srv.server_port}" and env.container.startswith("crucible-openenv-")
        await env.aclose()
        assert env.container is None
        calls = log.read_text().splitlines()
        assert calls[0].startswith("run -d --rm --name crucible-openenv-") and "-e A=1" in calls[0] and calls[-1].startswith("rm -f crucible-openenv-")
    finally:
        srv.shutdown()
