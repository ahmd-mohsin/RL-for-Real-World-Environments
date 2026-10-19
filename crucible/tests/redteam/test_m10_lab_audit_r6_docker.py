"""M10 lab audit R-6 (2026-10-19), Docker: sandbox hygiene against a real daemon (Acceptance 1-5 of the recommendation).

Every container Crucible starts names its owner and an expiry, ends by itself at that expiry, is removed on every exit path of
the code that started it, and is swept after a crash (``crucible.world.lifecycle``). The unit tests (``test_m10_lab_audit_r6_*``)
hold the same contracts against a fake daemon; this module holds them where the fake could be wrong about what Docker does with
a flag, a label filter or a killed client.

- Acceptance 1, TTL half: a runner that is SIGKILLed mid-attempt leaves a container that ends by itself at its TTL, with no sweep
  and no other Runner involved (the TTL is in the container's own command, the one backstop that survives a kill on every OS).
- Acceptance 1, sweep half: the next ``Runner.run`` of the killed run (a resume) removes the container it left, as ``own_run``;
  a run of another id on the same root removes it as ``lease_free`` (the dead run's ``run.lock`` is free). Never by pid.
- Acceptance 1, grade half (review round 1): a runner SIGKILLed during a ``HarborDirTask`` grade leaves a container that ends by
  itself at its in-container bound (``lifecycle.BOUND_SH``), with no sweep and no later Runner on that host.
- Acceptance 2: an ``open()`` that fails after the sandbox started (``HealthcheckError``: retryable infra) three times leaves no
  container, and each attempt's ``teardown.json`` lists the one it removed.
- Acceptance 3: a cancel during ``DockerSandbox.start`` and during a ``HarborDirTask`` grade leaves no container, and the grade's
  host staging is gone (it is deleted only once the container is).
- Acceptance 4 and 5: a run with ``pool > 0`` exits with no warm container, and every dispatched attempt dir has ``teardown.json``.

Needs a Docker daemon (the suite's conftest skips without one) and ``python:3.13-slim`` (pulled when absent). Containers are found
by ``docker ps -a --filter label=crucible.run=<id>`` (run ids are unique per test) and removed in a ``finally`` whatever the test
found. A Runner start sweeps the local daemon (``RunSpec.hygiene.sweep``): it removes only orphans (this run's lease, a free
lease, past expiry).
"""

from __future__ import annotations

import asyncio
import json
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.run import Runner, RunSpec
from crucible.world import lifecycle
from crucible.world.health import HealthcheckError
from crucible.world.pool import PooledSandbox, pool_for
from crucible.world.sandbox import DockerSandbox

pytestmark = [pytest.mark.docker]
IMAGE = "python:3.13-slim"
TEARDOWN_KEYS = {"terminated", "failed", "still_up", "released"}


@pytest.fixture(scope="module", autouse=True)
def _image():
    if not shutil.which("docker"):
        return
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode != 0:
        subprocess.run(["docker", "pull", "-q", IMAGE], capture_output=True, check=False, timeout=600)


def world_spec() -> WorldSpec:
    """The World a sandbox is built for (no Apps, no network)."""
    return WorldSpec.model_validate({"world": "test/r6", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})


def run_id() -> str:
    return f"r6-{uuid.uuid4().hex[:8]}"


# ---------------------------------------------------------------- docker helpers (synchronous: a test is not a Runner)


def _docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout, check=False)


def ps(label: str) -> list[str]:
    """Names of every container (running or not) carrying ``label`` (``key=value``)."""
    return sorted(_docker("ps", "-a", "--filter", f"label={label}", "--format", "{{.Names}}").stdout.split())


def purge(label: str) -> None:
    ids = _docker("ps", "-a", "-q", "--filter", f"label={label}").stdout.split()
    if ids:
        _docker("rm", "-f", *ids)


def inspect(name: str, fmt: str):
    out = _docker("inspect", "-f", fmt, name)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def wait_until(pred, timeout: float, step: float = 0.5) -> bool:
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(step)
    return pred()


def events_of(root: Path, rid: str) -> list[dict]:
    return [json.loads(x) for x in (root / rid / "journal.jsonl").read_text().splitlines() if x.strip()]


def teardowns(root: Path, rid: str) -> list[dict]:
    paths = sorted((root / rid / "samples").glob("*/a*/teardown.json"))
    return [json.loads(p.read_text()) for p in paths]


# ---------------------------------------------------------------- a runner in another process, killed mid-attempt

# The child is a real ``Runner.run`` whose one attempt starts a docker sandbox, says so, and waits for ever. Its harness is
# declared exactly as ``Resumed`` below (same id, same ``identity_config``): a run's identity is the resolved document, so the
# resume of its run id by ``Resumed`` is not drift.
CHILD = '''
import asyncio, sys
from types import SimpleNamespace

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.runtime.run import Runner, RunSpec
from crucible.world.lifecycle import Hygiene
from crucible.world.sandbox import DockerSandbox

root, run_id, attempt_s, default_s, image = sys.argv[1:6]


class Hold:
    id = "hold@1"

    def identity_config(self):
        return {}

    async def run(self, attempt):
        spec = WorldSpec.model_validate({"world": "test/r6", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})
        sbx = DockerSandbox(SandboxSpec(image=image), spec)
        await sbx.start()
        print("ready", sbx.name, flush=True)
        await asyncio.Event().wait()


run = RunSpec(run_id=run_id, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[Hold()], k=1,
              attempt_timeout_s=float(attempt_s) or None, hygiene=Hygiene(default_s=int(default_s)), breaker=None)
asyncio.run(Runner(root).run(run))
'''


def spawn(root: Path, rid: str, *, attempt_s: float = 0, default_s: int = 21600) -> subprocess.Popen:
    """The child runner. ``attempt_s`` 0 is no attempt backstop, so the container lifetime is ``Hygiene.default_s``."""
    return subprocess.Popen([sys.executable, "-c", CHILD, str(root), rid, str(attempt_s), str(default_s), IMAGE], stdin=subprocess.DEVNULL,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def ready(child: subprocess.Popen, timeout: float = 180.0) -> str:
    """The name of the container the child started, within ``timeout`` (a pipe cannot be select()ed on Windows)."""
    lines: list[str] = []

    def pump() -> None:
        for line in child.stdout:  # type: ignore[union-attr]
            lines.append(line.rstrip())
            if line.startswith("ready "):
                return

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    t.join(timeout)
    name = next((ln.split(" ", 1)[1] for ln in lines if ln.startswith("ready ")), None)
    assert name, "the child runner never reported a started container:\n" + "\n".join(lines[-30:])
    return name


def kill(child: subprocess.Popen) -> None:
    """SIGKILL: no atexit, no finally, no reclaim. Only the TTL and the next sweep can end what the child left."""
    if child.poll() is None:
        child.kill()
    child.wait(timeout=30)


class Resumed:
    """The child's harness as the run identity sees it (same id and ``identity_config``), but it returns at once: the resume of a
    killed run owes nothing but the sweep that its start runs."""

    id = "hold@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0, cost_usd=0.0)


def resumed_spec(rid: str) -> RunSpec:
    return RunSpec(run_id=rid, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[Resumed()], k=1,
                   breaker=None)


# ---------------------------------------------------------------- Acceptance 1: kill -9 the runner mid-attempt


def test_a_killed_runners_containers_end_at_their_ttl(tmp_path):
    """No attempt backstop and ``Hygiene(default_s=8)``: the container's life is 8 s from its start (the kernel's life loop, which
    only a renewing holder extends). The runner is killed right after it reports the container up; nothing sweeps and nothing
    renews, and the container is gone within the poll below: the TTL is in the container's own command."""
    rid = run_id()
    child = spawn(tmp_path, rid, attempt_s=0, default_s=8)
    label = f"crucible.run={rid}"
    try:
        name = ready(child)
        t_ready = time.time()
        assert name in ps(label)  # labelled with the run (the sweep and PerfGuard find it by that) and alive: its deadline is seconds away
        cfg = inspect(name, "{{json .Config}}")  # one call: the container's own life is 8 s
        labels, cmd = cfg["Labels"], cfg["Cmd"]
        assert len(labels["crucible.owner"].split(":")) == 3 and labels["crucible.run"] == rid
        assert labels["crucible.lease"] == str((tmp_path / rid / "run.lock").resolve()) and labels["crucible.attempt"].endswith("-a1")
        assert abs(int(labels["crucible.expires"]) - (t_ready + 8 + lifecycle.RENEW_SLACK_S)) < 120  # a renewable keeper's label: its deadline + slack
        assert labels["crucible.renewable"] == "1" and labels["crucible.sandbox"] == "1"
        assert "infinity" not in json.dumps(cmd) and cmd[:2] == ["sh", "-c"], cmd  # the life loop, never `sleep infinity`
        kill(child)
        assert child.poll() is not None
        gone = wait_until(lambda: ps(label) == [], 40)
        assert gone, f"the container outlived its 8 s TTL by 40 s: {ps(label)}"
    finally:
        kill(child)
        purge(label)


@pytest.mark.parametrize(("same_run", "why"), [(True, "own_run"), (False, "lease_free")])
async def test_the_next_run_sweeps_a_killed_runners_containers(tmp_path, same_run, why):
    """The container's life is 6 h (no backstop, the default): only the sweep can remove it now. A resume of the killed run takes
    the lease, finds the container carrying this run's lease path (``own_run``: no attempt of this run is live yet); a run of
    another id on the same root finds it by the dead run's lock, which is free (``lease_free``). Keyed on the lock, never the pid."""
    killed = run_id()
    child = spawn(tmp_path, killed, attempt_s=0, default_s=21600)
    label = f"crucible.run={killed}"
    try:
        name = ready(child)
        kill(child)
        assert name in ps(label)  # a 6 h life: still there after the kill
        rid = killed if same_run else run_id()
        summary = await Runner(tmp_path).run(resumed_spec(rid))
        assert summary.terminal == 1
        assert ps(label) == [], "the next run's sweep left the killed run's container"
        sweeps = [e for e in events_of(tmp_path, rid) if e["event"] == "sweep"]
        assert len(sweeps) == 1 and sweeps[0]["errors"] == {} and sweeps[0]["failed"] == [], sweeps
        assert {r["name"]: r["why"] for r in sweeps[0]["removed"]}.get(name) == why, sweeps[0]
    finally:
        kill(child)
        purge(label)


# ---------------------------------------------------------------- Acceptance 2: open() fails after start, with retries


class OpenFails:
    """A task whose ``open`` starts a docker sandbox and then raises a ``HealthcheckError`` (retryable infra: each retry used to leak
    one more container). It does not use ``lifecycle.opening``: only the attempt scope's reclaim can remove what it started."""

    id = "t"
    world_id = "test/r6"

    def identity_config(self) -> dict:
        return {}

    async def open(self, workdir):
        await DockerSandbox(SandboxSpec(image=IMAGE), world_spec()).start()
        raise HealthcheckError("not healthy")


async def test_an_open_that_fails_after_start_three_times_leaves_no_container(tmp_path):
    rid = run_id()
    label = f"crucible.run={rid}"
    spec = RunSpec(run_id=rid, tasks=[OpenFails()], policies=[PolicyRef("p", lambda a: {}, config={})], harnesses=[NativeHarness()], breaker=None,
                   retry_backoff_s=0.0)  # the default breaker would defer the third infra failure
    try:
        summary = await Runner(tmp_path).run(spec)
        assert summary.attempts == 3 and summary.masked_terminal == 1
        assert ps(label) == []
        tds = teardowns(tmp_path, rid)
        assert len(tds) == 3
        for td in tds:
            assert set(td) == TEARDOWN_KEYS and td["still_up"] == [] and td["failed"] == []
            assert len(td["terminated"]) == 1 and td["terminated"][0].startswith("crucible-sbx-")
    finally:
        purge(label)


# ---------------------------------------------------------------- Acceptance 3: a cancel during start, and during a grade


@pytest.mark.parametrize("delay", [0.05, 0.4, 1.0])
async def test_a_cancel_during_start_leaves_no_container(delay):
    """The cancel lands before, during or after the ``docker run`` (an image already cached takes a few hundred ms): wherever it
    lands the attempt scope's exit leaves nothing carrying its attempt label. A cancelled start lets the shielded run finish and
    removes what it made; a start that had already finished is removed by the scope's reclaim."""
    rid, att = run_id(), f"x{uuid.uuid4().hex[:6]}-a1"
    label = f"crucible.attempt={att}"
    scope = lifecycle.Scope(run=rid, attempt=att, ttl_s=600, renewable=True)
    try:
        async with lifecycle.attempt(scope):
            sbx = DockerSandbox(SandboxSpec(image=IMAGE), world_spec())
            t = asyncio.ensure_future(sbx.start())
            await asyncio.sleep(delay)
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
        assert await asyncio.to_thread(ps, label) == []
        rep = scope.report()
        assert sbx.name in rep["terminated"] and rep["still_up"] == [] and rep["failed"] == []
    finally:
        purge(label)


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("rsync"), reason="a grade stages through rsync and a POSIX bind mount")
async def test_a_cancelled_dir_grade_leaves_no_container_and_no_staging(tmp_path):
    """A ``HarborDirTask`` grade whose tests hang, cancelled once its container is listed by its attempt label (a scoring budget
    does this): the container is removed (looked at again ``RECHECK_S`` later, for a daemon that creates it after its client was
    killed) and only then the host staging under ``<workroot>/grades`` goes. A local ``GpuHost`` with an absolute workroot
    stages through local paths, so a real grade needs no lab."""
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.gpu import GpuHost

    td = tmp_path / "task"
    (td / "tests").mkdir(parents=True)
    (td / "task.toml").write_text('[task]\nname = "r6-hang"\n\n[verifier]\ntimeout_sec = 300\n')
    (td / "tests" / "test.sh").write_text("#!/bin/bash\nsleep 600\n")
    wr = tmp_path / "wr"
    dt = HarborDirTask(td, image=IMAGE, host=GpuHost(name="local", workroot=str(wr)))
    rid, att = run_id(), f"g{uuid.uuid4().hex[:6]}-a1"
    label = f"crucible.attempt={att}"
    scope = lifecycle.Scope(run=rid, attempt=att, ttl_s=600, renewable=True)
    try:
        async with lifecycle.attempt(scope):
            t = asyncio.ensure_future(dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g"))
            for _ in range(240):  # up to 60 s: the staging rsync, then the container
                if await asyncio.to_thread(ps, label):
                    break
                await asyncio.sleep(0.25)
            else:
                pytest.fail("the grade container never appeared")
            t.cancel()
            await asyncio.gather(t, return_exceptions=True)
            assert t.cancelled()
        assert await asyncio.to_thread(ps, label) == []
        grades = wr / "grades"
        assert not grades.exists() or list(grades.iterdir()) == [], list(grades.iterdir())  # staging deleted, after the container
        assert not (tmp_path / "g" / "staging_kept.json").exists()
        rep = scope.report()
        assert any(n.startswith("crucible-hgrade-") for n in rep["terminated"]) and rep["still_up"] == [] and rep["failed"] == []
    finally:
        purge(label)


# ---------------------------------------------------------------- Acceptance 1 for a grade container (review round 1)

# A grade container's command is the task's own, so it has no ``sleep <ttl>``; it runs under ``lifecycle.BOUND_SH`` instead (``timeout -k 30
# <bound>``, bound = the grade's timeout plus ``GRADE_MARGIN_S``). The child grades a task whose tests hang, with a 300 s grade timeout
# (the child's own ``docker run`` bound must not fire first) and a margin of -292 s, so the in-container bound is 8 s.
GRADE_CHILD = '''
import asyncio, sys
from pathlib import Path

from crucible.interop.harbor.verify_dir import HarborDirTask
from crucible.runtime.gpu import GpuHost
from crucible.world import lifecycle

task_dir, workroot, gdir, image = sys.argv[1:5]
lifecycle.GRADE_MARGIN_S = -292.0
dt = HarborDirTask(Path(task_dir), image=image, host=GpuHost(name="local", workroot=workroot), timeout=300.0)
print("owner", lifecycle.owner_id(), flush=True)
asyncio.run(dt.grade_patch("diff --git a/x b/x\\n", Path(gdir)))
'''


def reported(child: subprocess.Popen, prefix: str, timeout: float = 120.0) -> str:
    """The rest of the first line the child prints that starts with ``prefix``."""
    lines: list[str] = []

    def pump() -> None:
        for line in child.stdout:  # type: ignore[union-attr]
            lines.append(line.rstrip())
            if line.startswith(prefix):
                return

    t = threading.Thread(target=pump, daemon=True)
    t.start()
    t.join(timeout)
    got = next((ln[len(prefix):].strip() for ln in lines if ln.startswith(prefix)), None)
    assert got, f"the child never printed {prefix!r}:\n" + "\n".join(lines[-30:])
    return got


@pytest.mark.skipif(sys.platform == "win32" or not shutil.which("rsync"), reason="a grade stages through rsync and a POSIX bind mount")
def test_a_grade_a_killed_runner_left_ends_at_its_in_container_bound(tmp_path):
    """A runner SIGKILLed during a ``HarborDirTask`` grade whose tests hang (the incident: a perf grade on the lab GPU host that nothing
    ever ended, holding gate R1 for 12 h): no ``docker rm -f`` runs, and no later Runner need ever visit that host, yet the container
    ends by itself at its bound."""
    td = tmp_path / "task"
    (td / "tests").mkdir(parents=True)
    (td / "task.toml").write_text('[task]\nname = "r6-kill"\n')
    (td / "tests" / "test.sh").write_text("#!/bin/bash\nsleep 600\n")
    child = subprocess.Popen([sys.executable, "-c", GRADE_CHILD, str(td), str(tmp_path / "wr"), str(tmp_path / "g"), IMAGE], stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    label = ""
    try:
        label = f"crucible.owner={reported(child, 'owner ')}"
        assert wait_until(lambda: ps(label) != [], 120), "the grade container never appeared"
        (name,) = ps(label)
        assert name.startswith("crucible-hgrade-")
        cfg = inspect(name, "{{json .Config}}")
        assert cfg["Entrypoint"] == ["bash"] and cfg["Cmd"][:3] == ["-c", lifecycle.BOUND_SH, "crucible-bound"], cfg
        assert cfg["Labels"]["crucible.owner"] == label.split("=", 1)[1] and int(cfg["Labels"]["crucible.expires"]) > 0
        kill(child)
        assert wait_until(lambda: ps(label) == [], 60), f"the grade container outlived its 8 s in-container bound by 60 s: {ps(label)}"
    finally:
        kill(child)
        if label:
            purge(label)


# ---------------------------------------------------------------- Acceptance 4 and 5: a pooled run, and a teardown per attempt


class PooledUse:
    """Acquires a pooled sandbox, runs a command in it and releases it. The pool (and its warm containers) belongs to the run."""

    id = "pooled@1"

    def __init__(self) -> None:
        self.labels: list[dict] = []

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        sb = PooledSandbox(pool_for(SandboxSpec(image=IMAGE, pool=2), world_spec()))
        await sb.start()
        try:
            assert (await sb.exec.run("echo hi", timeout_s=30)).stdout.strip() == "hi"
            self.labels.append(inspect(sb._sbx.name, "{{json .Config.Labels}}"))
        finally:
            await sb.close()
        return HarnessResult(reward=1.0, cost_usd=0.0)


async def test_a_pooled_run_exits_with_no_warm_container(tmp_path):
    """Three attempts share one pool of two. ``close_pools(owner=lease)`` at the end of ``Runner.run`` removes the warm containers (and any
    still starting) before the lease is released; the pooled containers carried the run's labels and no attempt label."""
    from crucible.world import pool as pool_mod

    rid = run_id()
    label = f"crucible.run={rid}"
    h = PooledUse()
    spec = RunSpec(run_id=rid, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[h], k=3, breaker=None)
    try:
        summary = await Runner(tmp_path).run(spec)
        assert summary.scored == 3
        assert ps(label) == [], "a warm pooled container outlived the run"
        assert not [k for k in pool_mod._POOLS if k[0] == str((tmp_path / rid / "run.lock").resolve())]  # pools are keyed by the run's lease
        assert any(e["event"] == "pools_closed" for e in events_of(tmp_path, rid))
        assert len(h.labels) == 3
        for lb in h.labels:  # the pool is the run's: run labels and the lease, no attempt
            assert lb["crucible.run"] == rid and lb["crucible.lease"] == str((tmp_path / rid / "run.lock").resolve()) and "crucible.attempt" not in lb
        tds = teardowns(tmp_path, rid)  # Acceptance 5: every dispatched attempt dir has one
        assert len(tds) == 3 and all(set(td) == TEARDOWN_KEYS and td["still_up"] == [] for td in tds)
    finally:
        purge(label)
