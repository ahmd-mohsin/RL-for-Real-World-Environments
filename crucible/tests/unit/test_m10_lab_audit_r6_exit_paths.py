"""M10 lab audit R-6, slice 2: every exit path reclaims (``crucible.world.lifecycle`` in ``World``, ``RunTask.open``, the Runner's
attempt scope and ``teardown.json``, ``close_pools``, grade containers).

- ``lifecycle.opening(world)``: an ``open()`` that fails after ``World.start`` closes the World, and under a Runner one that does not
  use it is closed by the attempt scope's reclaim, so each retry of a ``HealthcheckError`` leaves nothing behind (Acceptance 2);
- ``World.close`` is shielded, idempotent and bounded, records its outcome in the scope it started in, and raises nothing new;
- every dispatched attempt (a grade-only retry, a cancelled one, a ``Runner.regrade`` attempt) has ``teardown.json``, a refused
  reservation has none (Acceptance 5); a cancelled attempt still removes its containers (Acceptance 3);
- the warm container pools end with the run (``close_pools``, Acceptance 4);
- ``HarborDirTask`` and ``verify.stage.docker_grade`` label their container, and a cancelled grade looks again before the staging goes.

No Docker: ``FakeDocker`` (``test_m10_lab_audit_r6_lifecycle``) stands in for ``world.exec._run_subprocess``; Worlds use the local
backend with no Apps.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core.schema.run import Failure
from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.infra import RetryRule
from crucible.runtime.run import Runner, RunSpec
from crucible.world import lifecycle
from crucible.world.exec import ExecOut
from crucible.world.health import HealthcheckError
from crucible.world.lifecycle import Hygiene, Scope
from crucible.world.pool import PooledSandbox, pool_for
from crucible.world.sandbox import DockerSandbox
from crucible.world.world import World
from tests.unit.test_m10_lab_audit_r6_lifecycle import FakeDocker, argv_labels

posix = pytest.mark.skipif(sys.platform == "win32", reason="a POSIX shell-script fake docker")
TEARDOWN_KEYS = {"terminated", "failed", "still_up", "released"}


def _spec() -> WorldSpec:
    """The World a sandbox is built for (no Apps; ``no-network``)."""
    return WorldSpec.model_validate({"world": "test/r6x", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})


def _world_spec() -> WorldSpec:
    """A World that starts on the local backend with nothing in it (the single-turn tasks' shape)."""
    return WorldSpec.model_validate({"world": "test/r6x", "apps": [], "actors": [{"id": "agent", "kind": "policy", "tools": []}],
                                     "intercept": [], "isolation": {"quiesce": False, "egress_probe": False}})


def _world(root: Path, *, own_workdir: bool = False) -> World:
    return World(_world_spec(), instructions={"agent": "go"}, store=Store(root / "s"), workdir=None if own_workdir else root / "w")


def _attempt_scope(**kw) -> Scope:
    return Scope(**{"run": "r", "attempt": "s-a1", "lock": "/x/run.lock", "ttl_s": 600, **kw})


@pytest.fixture(autouse=True)
def fake(monkeypatch) -> FakeDocker:
    from crucible.world import pool as pool_mod
    from crucible.world import sandbox as sandbox_mod

    f = FakeDocker()
    monkeypatch.setattr(sandbox_mod, "_run_subprocess", f)
    monkeypatch.setattr(lifecycle, "_run_subprocess", f)
    monkeypatch.setattr(lifecycle, "_install_reaper", lambda: None)
    monkeypatch.setattr(lifecycle, "_HOSTS", set())
    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)
    monkeypatch.setattr(pool_mod, "_POOLS", {})
    return f


# ---------------------------------------------------------------- lifecycle.opening and World.close


async def test_an_open_that_fails_after_start_closes_its_world(tmp_path):
    w = _world(tmp_path)
    with pytest.raises(HealthcheckError):
        async with lifecycle.opening(w):
            assert w._started and not w.closed
            raise HealthcheckError("x")
    assert w.closed
    await asyncio.wait_for(w.close(), 1)  # a second close returns at once


async def test_a_successful_opening_leaves_the_world_to_its_caller(tmp_path):
    w = _world(tmp_path)
    async with lifecycle.opening(w) as got:
        await got.reset()
    assert got is w and not w.closed
    await w.close()
    assert w.closed


async def test_a_cancel_inside_opening_closes_the_world_too(tmp_path):
    w = _world(tmp_path)
    reached = asyncio.Event()

    async def body():
        async with lifecycle.opening(w):
            reached.set()
            await asyncio.Event().wait()

    t = asyncio.create_task(body())
    await asyncio.wait_for(reached.wait(), 2)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert w.closed


async def test_world_close_is_shielded_idempotent_and_bounded(tmp_path):
    w = _world(tmp_path)
    await w.start()
    done: list[int] = []

    async def slow() -> None:
        await asyncio.sleep(0.3)
        done.append(1)

    w.backend.teardown = slow  # type: ignore[method-assign]
    t = asyncio.create_task(w.close())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert done == [1] and w.closed  # the cancelled caller let the close run to its end
    await asyncio.wait_for(w.close(), 1)  # a second close returns at once; the teardown ran once
    assert done == [1]

    # bounded: a teardown that runs past ``close_timeout_s`` is given up on, recorded in the scope, and raises nothing
    scope = _attempt_scope()
    async with lifecycle.attempt(scope):
        w2 = _world(tmp_path / "two")
        w2.close_timeout_s = 0.05
        await w2.start()

        async def hang() -> None:
            await asyncio.sleep(5)

        w2.backend.teardown = hang  # type: ignore[method-assign]
        t0 = time.monotonic()
        await w2.close()
        assert time.monotonic() - t0 < 1.5 and w2.closed
        assert any(f.get("world") == w2.spec.world and "ran past" in f["error"] for f in scope.failed), scope.failed


async def test_a_failed_world_close_is_recorded_not_lost(tmp_path):
    scope = _attempt_scope()
    async with lifecycle.attempt(scope):
        w = _world(tmp_path, own_workdir=True)
        await w.start()
        assert w.workdir.exists()

        async def boom() -> None:
            raise RuntimeError("boom")

        w.backend.teardown = boom  # type: ignore[method-assign]
        with pytest.raises(RuntimeError, match="boom"):  # what ``_close`` raised before R-6, to the first caller
            await w.close()
        await w.close()  # not to the second
        assert [f for f in scope.report()["failed"] if f.get("world") == w.spec.world and "boom" in f["error"]]
        assert not w.workdir.exists()  # the owned workdir goes even when the teardown raised


async def test_concurrent_closes_share_one_teardown_and_one_error(tmp_path):
    w = _world(tmp_path)
    await w.start()
    calls: list[int] = []

    async def boom() -> None:
        calls.append(1)
        await asyncio.sleep(0.05)
        raise RuntimeError("boom")

    w.backend.teardown = boom  # type: ignore[method-assign]
    got = await asyncio.gather(w.close(), w.close(), return_exceptions=True)
    assert calls == [1] and sum(isinstance(g, RuntimeError) for g in got) == 1 and sum(g is None for g in got) == 1


async def test_a_failed_start_still_surfaces_its_own_error_when_the_close_fails(tmp_path):
    """``World.start``'s failure path closes with errors suppressed: a teardown error never replaces the start's (retryable) one."""
    w = _world(tmp_path)

    async def provision() -> None:
        raise HealthcheckError("not healthy")

    async def boom() -> None:
        raise RuntimeError("teardown boom")

    w.backend.provision = provision  # type: ignore[method-assign]
    w.backend.teardown = boom  # type: ignore[method-assign]
    with pytest.raises(HealthcheckError):
        await w.start()
    assert w.closed


async def test_a_failing_sandbox_removal_never_replaces_a_grade_that_was_already_computed(tmp_path, fake):
    """A verifier World closes in a ``finally`` after its grade: an ``rm -f`` that fails (rc-checked now) is recorded, not raised."""
    scope = _attempt_scope(renewable=False)
    async with lifecycle.attempt(scope):
        w = _world(tmp_path)
        await w.start()
        sbx = DockerSandbox(SandboxSpec(image="img"), _spec())
        await sbx.start()

        async def teardown() -> None:
            await sbx.close()

        w.backend.teardown = teardown  # type: ignore[method-assign]
        fake.fail_rm.add(sbx.name)
        grade = {"reward": 1.0}
        await w.close()  # does not raise: the grade stands
        assert grade["reward"] == 1.0 and w.closed
        assert [f["name"] for f in scope.failed] == [sbx.name]
        fake.fail_rm.clear()  # the scope's reclaim retries it
    rep = scope.report()
    assert sbx.name in rep["terminated"] and rep["still_up"] == [] and rep["failed"] == [] and fake.containers() == []


async def test_inplace_open_closes_its_world_when_reset_fails(tmp_path, monkeypatch):
    from crucible.interop.harbor.inplace import HarborInPlaceTask

    calls: list[str] = []

    class FakeWorld:
        def __init__(self, spec, **kw):
            pass

        async def start(self):
            calls.append("world.start")

        async def reset(self):
            calls.append("world.reset")
            raise HealthcheckError("not healthy")

        async def close(self):
            calls.append("world.close")

    monkeypatch.setattr("crucible.world.world.World", FakeWorld)
    monkeypatch.setattr("crucible.core.store.Store", lambda root: None)
    t = object.__new__(HarborInPlaceTask)
    t.id, t.grader_integrity, t.QUIESCE = "harbor/t", "off", False
    t.backend_params, t.instruction, t.dir = {"environment_type": "docker"}, "do it", tmp_path
    t.task = SimpleNamespace(config=SimpleNamespace(steps=[], environment=SimpleNamespace(healthcheck=None)))
    t.spec = lambda workdir: None  # type: ignore[method-assign]
    with pytest.raises(HealthcheckError):
        await t.open(tmp_path / "work")
    assert calls == ["world.start", "world.reset", "world.close"]


# ---------------------------------------------------------------- the Runner's attempt scope and teardown.json


class LeakyOpen:
    """A task whose ``open`` builds a World, starts it and fails (a ``HealthcheckError``: retryable infra), without
    ``lifecycle.opening``: only the attempt scope's reclaim can close what it started."""

    id = "t"
    world_id = "test/r6x"

    def __init__(self, worlds: list) -> None:
        self.worlds = worlds

    def identity_config(self) -> dict:
        return {}

    async def open(self, workdir):
        w = World(_world_spec(), store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        self.worlds.append(w)
        await w.start()
        raise HealthcheckError("not healthy")


class SandboxLeakyOpen:
    """The same with a docker sandbox started and left behind."""

    id = "t"
    world_id = "test/r6x"

    def identity_config(self) -> dict:
        return {}

    async def open(self, workdir):
        await DockerSandbox(SandboxSpec(image="img"), _spec()).start()
        raise HealthcheckError("not healthy")


def native_spec(task, run_id: str = "r", **kw) -> RunSpec:
    kw = {"breaker": None, "retry_backoff_s": 0.0, **kw}  # the default breaker would defer the third infra failure
    return RunSpec(run_id=run_id, tasks=[task], policies=[PolicyRef("p", lambda a: {}, config={})], harnesses=[NativeHarness()], **kw)


def sdir_of(root: Path, spec: RunSpec) -> Path:
    return root / spec.run_id / "samples" / Runner(root).plan(spec)[0][0].id()


def teardown_of(sdir: Path, n: int) -> dict:
    return json.loads((sdir / f"a{n}" / "teardown.json").read_text())


def events_of(root: Path, run_id: str = "r") -> list[dict]:
    return [json.loads(x) for x in (root / run_id / "journal.jsonl").read_text().splitlines() if x.strip()]


async def test_every_retry_of_an_open_that_fails_after_start_leaves_no_world(tmp_path):
    """Acceptance 2 (unit): three attempts, three Worlds, none left started."""
    worlds: list[World] = []
    spec = native_spec(LeakyOpen(worlds))
    summ = await Runner(tmp_path).run(spec)
    assert summ.attempts == 3 and summ.masked_terminal == 1 and summ.by_failure == {"crucible:backend_error": 1}
    assert len(worlds) == 3 and all(w.closed for w in worlds)
    sdir = sdir_of(tmp_path, spec)
    for n in (1, 2, 3):
        td = teardown_of(sdir, n)
        assert set(td) == TEARDOWN_KEYS and td["still_up"] == [] and td["failed"] == []


async def test_a_sandbox_an_open_left_started_is_removed_by_the_attempt_scope(tmp_path, fake):
    """Acceptance 2 with containers, and the labels a Runner puts on them (the run, its lease, the attempt, the hygiene lifetime)."""
    spec = native_spec(SandboxLeakyOpen(), hygiene=Hygiene(default_s=1000))
    summ = await Runner(tmp_path).run(spec)
    assert summ.attempts == 3 and fake.containers() == []
    sdir = sdir_of(tmp_path, spec)
    sid = sdir.name
    runs = {a[a.index("--name") + 1]: argv_labels(a) for _, a in fake.verbs("run")}
    assert len(runs) == 3
    lock = str((tmp_path / "r" / "run.lock").resolve())
    for n in (1, 2, 3):
        td = teardown_of(sdir, n)
        assert len(td["terminated"]) == 1 and td["terminated"][0].startswith("crucible-sbx-") and td["still_up"] == [] and td["failed"] == []
        labels = runs[td["terminated"][0]]
        assert labels["crucible.attempt"] == f"{sid}-a{n}" and labels["crucible.run"] == "r" and labels["crucible.lease"] == lock
        assert abs(int(labels["crucible.expires"]) - (time.time() + 1000 + lifecycle.RENEW_SLACK_S)) < 30  # Hygiene.default_s, no backstop


class Ok:
    id = "ok@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0, cost_usd=0.1)


class Failing:
    """Fails (a masked transport timeout, retryable) on its first ``fail`` calls, then scores."""

    id = "failing@1"

    def __init__(self, fail: int) -> None:
        self.fail, self.calls = fail, 0

    def identity_config(self) -> dict:
        return {"fail": self.fail}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.calls += 1
        if self.calls <= self.fail:
            return HarnessResult(failure=Failure(kind="transport_timeout", stage="agent", reason="x"), cost_usd=0.1)
        return HarnessResult(reward=1.0, cost_usd=0.1)


class Regrading:
    """Persists a patch and fails its grade (a non-retryable infra mask at ``verification``), then re-grades it clean."""

    id = "regrading@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        (attempt.dir / "artifacts").mkdir(parents=True, exist_ok=True)
        (attempt.dir / "artifacts" / "model.patch").write_text("diff --git a/x b/x\n")
        return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", reason="x", retryable=False, blame="infra"), cost_usd=0.1)

    async def regrade(self, attempt: Attempt, prior: Path) -> HarnessResult:
        return HarnessResult(reward=1.0, cost_usd=0.0)


class GradeOnlyOnce:
    """Its first attempt fails at stage ``verification`` with its work persisted for a re-grade (``regradable``), so the Runner
    dispatches attempt 2 as an R-3 grade-only retry (``dispatched.grade_only``); the second attempt scores."""

    id = "gradeonly@1"

    def __init__(self) -> None:
        self.calls = 0

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.calls += 1
        if self.calls == 1:
            return HarnessResult(failure=Failure(kind="transport_timeout", stage="verification", reason="x"), cost_usd=0.1, regradable=True)
        return HarnessResult(reward=1.0, cost_usd=0.0)


def spec_of(harness, run_id: str = "r", **kw) -> RunSpec:
    kw = {"breaker": None, **kw}  # the default breaker would pause or defer an infra mask: not what these tests are about
    return RunSpec(run_id=run_id, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.5)], harnesses=[harness], k=1, **kw)


async def test_every_dispatched_attempt_dir_has_a_teardown_record(tmp_path):
    """Acceptance 5: a plain retry has it (the grade-only one has its own test below), a refused reservation (no attempt) has none, and
    so does a regrade."""
    h = Failing(fail=2)
    spec = spec_of(h, "a", retry_backoff_s=0.0)
    s = await Runner(tmp_path).run(spec)
    assert (s.terminal, s.scored, h.calls) == (1, 1, 3)
    sdir = sdir_of(tmp_path, spec)
    assert all(set(teardown_of(sdir, n)) == TEARDOWN_KEYS for n in (1, 2, 3))
    assert not [e for e in events_of(tmp_path, "a") if e["event"] == "teardown"]  # nothing failed, nothing still up: no event

    # a budget that refuses a2 after a1 spent 0.1: refused.json and no teardown.json in a2
    h2 = Failing(fail=2)
    spec2 = spec_of(h2, "b", concurrency=1, budget_usd=0.55, retry_backoff_s=0.0)
    s2 = await Runner(tmp_path).run(spec2)
    sdir2 = sdir_of(tmp_path, spec2)
    assert s2.stopped and s2.stopped.startswith("budget") and (sdir2 / "a2" / "refused.json").exists()
    assert (sdir2 / "a1" / "teardown.json").exists() and not (sdir2 / "a2" / "teardown.json").exists()

    # a regrade attempt dir has one too
    rspec = spec_of(Regrading(), "c")
    s3 = await Runner(tmp_path).run(rspec)
    assert s3.masked_terminal == 1
    sdir3 = sdir_of(tmp_path, rspec)
    s4 = await Runner(tmp_path).regrade(rspec)
    assert s4.attempts == 1 and s4.scored == 1
    assert set(teardown_of(sdir3, 1)) == TEARDOWN_KEYS and set(teardown_of(sdir3, 2)) == TEARDOWN_KEYS


async def test_a_grade_only_retry_has_a_teardown_record_too(tmp_path):
    """Acceptance 5, the R-3 path (review round 1): attempt 2 is dispatched with ``grade_only`` (the harness ran no agent in it), and
    still runs in its own lifecycle scope and leaves its own ``teardown.json``; so does attempt 1."""
    h = GradeOnlyOnce()
    spec = spec_of(h, "g", retry_backoff_s=0.0)
    s = await Runner(tmp_path).run(spec)
    assert (s.terminal, s.scored, h.calls) == (1, 1, 2)
    dispatched = [e for e in events_of(tmp_path, "g") if e["event"] == "dispatched"]
    assert [bool(e.get("grade_only")) for e in dispatched] == [False, True]  # the Runner itself marked a2 a grade-only retry
    sdir = sdir_of(tmp_path, spec)
    assert all(set(teardown_of(sdir, n)) == TEARDOWN_KEYS for n in (1, 2))
    assert (sdir / "a2" / "teardown.json").stat().st_mtime_ns <= (sdir / "a2" / "record.json").stat().st_mtime_ns  # written before the record


class Hold:
    """Starts a docker sandbox and blocks (the attempt is live while the test cancels the run)."""

    id = "hold@1"

    def __init__(self) -> None:
        self.sbx: DockerSandbox | None = None
        self.up = asyncio.Event()

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.sbx = DockerSandbox(SandboxSpec(image="img"), _spec())
        await self.sbx.start()
        self.up.set()
        await asyncio.Event().wait()
        return HarnessResult(reward=1.0)


async def test_a_cancelled_attempt_still_reclaims_and_writes_teardown(tmp_path, fake):
    h = Hold()
    spec = spec_of(h, "r")
    t = asyncio.create_task(Runner(tmp_path).run(spec))
    await asyncio.wait_for(h.up.wait(), 5)
    assert fake.containers() == [h.sbx.name]
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert fake.containers() == []
    td = teardown_of(sdir_of(tmp_path, spec), 1)
    assert td["terminated"] == [h.sbx.name] and td["still_up"] == [] and td["failed"] == []


class LeavesOne:
    """Starts a sandbox, never closes it, and finishes: its removal is what the attempt's teardown has to report."""

    id = "leaves@1"

    def __init__(self) -> None:
        self.sbx: DockerSandbox | None = None

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.sbx = DockerSandbox(SandboxSpec(image="img"), _spec())
        await self.sbx.start()
        return HarnessResult(reward=1.0, cost_usd=0.1)


class EveryName:
    """A ``FakeDocker.fail_rm`` that matches every container: the daemon cannot remove anything."""

    def __contains__(self, name: object) -> bool:
        return True

    def clear(self) -> None:
        pass

    def add(self, name: str) -> None:
        pass


async def test_a_container_that_cannot_be_removed_is_reported_still_up_and_journaled(tmp_path, fake):
    fake.fail_rm = EveryName()  # type: ignore[assignment]
    h = LeavesOne()
    spec = spec_of(h, "r")
    s = await Runner(tmp_path).run(spec)
    assert s.scored == 1  # cleanup never changes the result
    td = teardown_of(sdir_of(tmp_path, spec), 1)
    assert td["still_up"] == [h.sbx.name] and td["terminated"] == [] and td["failed"]
    ev = [e for e in events_of(tmp_path) if e["event"] == "teardown"]
    assert len(ev) == 1 and ev[0]["still_up"] == [h.sbx.name] and ev[0]["failed"] >= 1


# ---------------------------------------------------------------- close_pools at the end of every run


class Pooled:
    """Acquires a pooled sandbox, runs, releases it; optionally raises afterwards. The pool (and its warm containers) is the run's."""

    id = "pooled@1"

    def __init__(self, fail: bool = False) -> None:
        self.fail = fail

    def identity_config(self) -> dict:
        return {"fail": self.fail}

    async def run(self, attempt: Attempt) -> HarnessResult:
        sb = PooledSandbox(pool_for(SandboxSpec(image="img", pool=2), _spec()))
        await sb.start()
        await sb.close()
        if self.fail:
            raise RuntimeError("boom")
        return HarnessResult(reward=1.0, cost_usd=0.1)


async def test_close_pools_runs_at_the_end_of_every_run(tmp_path, fake):
    """Acceptance 4 (unit): no warm container, no pool of this run, and a ``pools_closed`` event."""
    from crucible.world import pool as pool_mod

    spec = RunSpec(run_id="r", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[Pooled()], k=2)
    s = await Runner(tmp_path).run(spec)
    assert s.scored == 2
    assert fake.verbs("run")  # a pool was built
    assert fake.containers() == []
    assert pool_mod._POOLS == {}  # keyed by the run's lease (review round 1): none of them is left
    ev = [e for e in events_of(tmp_path) if e["event"] == "pools_closed"]
    assert len(ev) == 1 and len(ev[0]["terminated"]) >= 2 and ev[0]["failed"] == [] and ev[0]["unfinished"] == 0
    runs = [argv_labels(a) for _, a in fake.verbs("run")]
    assert all(lb["crucible.run"] == "r" and "crucible.attempt" not in lb for lb in runs)  # the pool belongs to the run, not an attempt
    assert {lb["crucible.lease"] for lb in runs} == {str((tmp_path / "r" / "run.lock").resolve())}

    # a harness that raises leaves no warm container either
    bad = RunSpec(run_id="r2", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[Pooled(fail=True)],
                  k=1, retry=RetryRule(max_attempts=1), breaker=None)
    s2 = await Runner(tmp_path).run(bad)
    assert s2.masked_terminal == 1 and fake.containers() == []
    assert pool_mod._POOLS == {}


async def test_one_roots_run_end_leaves_a_same_id_run_in_another_root_its_pool(tmp_path, fake):
    """Review round 1: ``close_pools`` was keyed by run id, which is unique only within a root; a Runner closing its pool force-removed
    the containers a same-id run in another root had handed to its live attempts. The Runner closes by its lease."""
    from crucible.world import pool as pool_mod

    class Keeps(Pooled):
        """Holds a pooled container for the whole call (the other run's live attempt)."""

        gate = asyncio.Event()
        up = asyncio.Event()

        async def run(self, attempt: Attempt) -> HarnessResult:
            sb = PooledSandbox(pool_for(SandboxSpec(image="img", pool=1), _spec()))
            await sb.start()
            self.up.set()
            await self.gate.wait()
            await sb.close()
            return HarnessResult(reward=1.0, cost_usd=0.1)

    def spec(h) -> RunSpec:
        return RunSpec(run_id="same", tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[h], k=1, breaker=None)

    for root in ("one", "two"):
        (tmp_path / root).mkdir()
    long = asyncio.create_task(Runner(tmp_path / "one").run(spec(Keeps())))
    await asyncio.wait_for(Keeps.up.wait(), 5)
    held = set(fake.containers())  # root one's live pooled container (and its refill)
    assert held and {k[0] for k in pool_mod._POOLS} == {str((tmp_path / "one" / "same" / "run.lock").resolve())}
    s = await Runner(tmp_path / "two").run(spec(Pooled()))  # same run id, another root: runs, closes its own pools
    assert s.scored == 1
    assert held <= set(fake.containers())  # ... and leaves root one's alone
    assert {k[0] for k in pool_mod._POOLS} == {str((tmp_path / "one" / "same" / "run.lock").resolve())}
    Keeps.gate.set()
    await asyncio.wait_for(long, 5)
    assert fake.containers() == [] and pool_mod._POOLS == {}


async def test_a_pool_refill_stuck_in_docker_run_does_not_hold_the_runs_lease(tmp_path, fake, monkeypatch):
    """Review round 1: ``shielded`` bounds only a cancelled caller, so ``Runner.run`` waited for every in-flight refill (a ``docker run``
    of up to 30 min on a slow daemon) with the run's lease held. The normal path is bounded too, and the wait is journaled."""
    monkeypatch.setattr(lifecycle, "POOLS_CLOSE_S", 0.05)
    gate = asyncio.Event()

    async def slow_close(owner=None, *, run=None):
        await gate.wait()
        return {"terminated": [], "failed": [], "unfinished": 0}

    monkeypatch.setattr("crucible.runtime.run.close_pools", slow_close)
    t0 = time.monotonic()
    s = await asyncio.wait_for(Runner(tmp_path).run(spec_of(Ok(), "r")), 10)
    assert s.scored == 1 and time.monotonic() - t0 < 5
    ev = [e for e in events_of(tmp_path) if e["event"] == "pools_closed"]
    assert len(ev) == 1 and ev[0]["timed_out"] is True
    Runner(tmp_path)._open(spec_of(Ok(), "r")).close()  # the lease was released: another opener is not refused
    gate.set()
    await asyncio.sleep(0)


# ---------------------------------------------------------------- hygiene is operational


def test_hygiene_is_operational():
    from crucible.runtime import identity as ident

    a = spec_of(Ok())
    b = spec_of(Ok(), hygiene=Hygiene(margin_s=1.0, sweep=False, default_s=7))
    assert ident.fingerprint(a.resolved_doc()) == ident.fingerprint(b.resolved_doc())
    assert a.resolved_doc()["hygiene"]["margin_s"] == 900.0 and b.resolved_doc()["hygiene"]["margin_s"] == 1.0
    assert "hygiene" in ident.DENY and "hygiene" not in ident.compared(a.resolved_doc())


async def test_a_resume_with_different_hygiene_is_not_identity_drift(tmp_path):
    await Runner(tmp_path).run(spec_of(Ok(), "r"))
    s = await Runner(tmp_path).run(spec_of(Ok(), "r", hygiene=Hygiene(margin_s=5.0, floor_s=1.0, default_s=100, sweep=False)))
    assert s.terminal == 1  # resumed (nothing owed), not refused


# ---------------------------------------------------------------- grade containers


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


async def test_a_dir_grade_container_carries_the_attempt_labels(tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    host, runs = _AsyncHost(), []
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            runs.append(list(a))
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    scope = Scope(run="r", attempt="s-a1", lock="/l")
    t0 = time.time()
    async with lifecycle.attempt(scope):
        await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")  # no reward files: a masked report, which is not the point here
    (argv,) = runs
    labels = argv_labels(["docker", *argv])
    assert labels["crucible.run"] == "r" and labels["crucible.attempt"] == "s-a1" and labels["crucible.lease"] == "/l"
    assert labels["crucible.owner"] == lifecycle.owner_id() and labels["crucible.sandbox"] == "1"
    assert abs(int(labels["crucible.expires"]) - (t0 + dt.timeout + lifecycle.GRADE_MARGIN_S)) < 30
    name = argv[argv.index("--name") + 1]
    rep = scope.report()
    assert name.startswith("crucible-hgrade-") and name in rep["terminated"] and rep["failed"] == [] and rep["still_up"] == []
    # review round 1: ``bash -c <command>`` runs under an in-container bound (the same number as the expiry label), so a grade a killed
    # runner left ends by itself on a host no later Runner sweeps
    bound = int(dt.timeout + lifecycle.GRADE_MARGIN_S)
    i = argv.index("--entrypoint")
    assert argv[i:i + 7] == ["--entrypoint", "bash", "img", "-c", lifecycle.BOUND_SH, "crucible-bound", str(bound)] and argv[-3:] == ["bash", "-c", dt.command]
    assert abs(int(labels["crucible.expires"]) - (t0 + bound)) < 30


async def test_a_cancelled_dir_grade_rechecks_before_its_staging_goes(tmp_path, monkeypatch):
    """Acceptance 3: the first removal answers "No such container" (the daemon had not created it yet); the second finds it."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)
    host, rms = _AsyncHost(), []
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            await asyncio.sleep(3600)
        if a[0] == "rm":
            rms.append(a[-1])
            if len(rms) == 1:
                return ExecOut(rc=1, stdout="", stderr="Error response from daemon: No such container: x")
        return ExecOut(rc=0, stdout="", stderr="")  # the second removal finds the container the daemon created late

    dt._docker = docker  # type: ignore[method-assign]
    t = await _cancel_once_running(host, dt, tmp_path)
    assert t.cancelled()
    rm_idx = [i for i, c in enumerate(host.calls) if _is_rm_container(c)]
    cleanup = next(i for i, c in enumerate(host.calls) if _is_rm_staging(c))
    assert len(rm_idx) == 2 and max(rm_idx) < cleanup and rms[0] == rms[1] and rms[0].startswith("crucible-hgrade-"), host.calls
    assert not (tmp_path / "g" / "staging_kept.json").exists()


async def test_a_cancelled_dir_grade_whose_second_look_fails_keeps_its_staging(tmp_path, monkeypatch):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)
    host, rms = _AsyncHost(), []
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            await asyncio.sleep(3600)
        if a[0] == "rm":
            rms.append(a[-1])
            if len(rms) == 1:
                return ExecOut(rc=1, stdout="", stderr="Error response from daemon: No such container: x")
            return ExecOut(rc=1, stdout="", stderr="Cannot connect to the Docker daemon")
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    scope = Scope(run="r", attempt="s-a1", lock="/l")
    async with lifecycle.attempt(scope):
        t = await _cancel_once_running(host, dt, tmp_path)
    assert t.cancelled() and len(rms) >= 2
    assert not any(_is_rm_staging(c) for c in host.calls), host.calls  # the mounts stay under a container that may still run
    kept = json.loads((tmp_path / "g" / "staging_kept.json").read_text())
    assert kept["container"].startswith("crucible-hgrade-") and kept["host"] == "local"
    assert any(f.get("name") == kept["container"] for f in scope.report()["failed"])  # and the attempt's teardown says so


async def test_a_dir_grade_that_dies_of_a_base_exception_still_removes_its_container(tmp_path, monkeypatch):
    """The ``finally``'s own removal: a ``BaseException`` that is neither ``CancelledError`` nor ``Exception`` leaves the container
    unremoved by the branches around ``docker run``; it is removed before the staging goes."""
    from crucible.interop.harbor.verify_dir import HarborDirTask

    class Stop(BaseException):
        pass

    host = _AsyncHost()
    dt = HarborDirTask(_dir_task(tmp_path), image="img", host=host)  # type: ignore[arg-type]

    async def docker(*a, timeout=120):
        host.calls.append(("docker", a[0], a[-1]))
        if a[0] == "run":
            raise Stop()
        return ExecOut(rc=0, stdout="", stderr="")

    dt._docker = docker  # type: ignore[method-assign]
    with pytest.raises(Stop):
        await dt.grade_patch("diff --git a/x b/x\n", tmp_path / "g")
    rm_idx = [i for i, c in enumerate(host.calls) if _is_rm_container(c)]
    cleanup = next(i for i, c in enumerate(host.calls) if _is_rm_staging(c))
    assert rm_idx and max(rm_idx) < cleanup


@posix
def test_docker_grade_labels_its_container(tmp_path, monkeypatch):
    from crucible.verify.stage import docker_grade

    monkeypatch.delenv("DOCKER_HOST", raising=False)
    b = tmp_path / "bin"
    b.mkdir()
    log = tmp_path / "docker.log"
    (b / "docker").write_text(f'#!/bin/sh\necho "$@" >> {log}\ncase "$1" in run) echo "invalid mount config" >&2; exit 125;; esac\nexit 0\n')
    (b / "docker").chmod(0o755)
    monkeypatch.setenv("PATH", f"{b}{os.pathsep}{os.environ['PATH']}")
    out = tmp_path / "out"
    out.mkdir()
    docker_grade(image="img", bundle_dir=tmp_path, plan="p", out_dir=out)
    run_line = next(ln for ln in log.read_text().splitlines() if ln.startswith("run "))
    assert "--label crucible.owner=" in run_line and "--label crucible.expires=" in run_line and "--label crucible.sandbox=1" in run_line
    # review round 1: the same python3 entrypoint, with an alarm that survives the exec armed for ``timeout + GRADE_MARGIN_S``
    assert "--entrypoint python3 img -I -c " in run_line and "signal.alarm" in run_line and f" {7200 + int(lifecycle.GRADE_MARGIN_S)} -I " in run_line
    assert run_line.endswith("--tier full --require-root")


@posix
def test_the_shell_bound_runs_the_command_unchanged_and_ends_a_hung_one():
    """``lifecycle.BOUND_SH`` as ``docker run --entrypoint bash <image> -c BOUND_SH crucible-bound <secs> bash -c <command>`` runs it:
    the command's own status and output where it finishes, ``timeout -k 30``'s 124 where it hangs, and as it was where the image has
    no ``timeout`` (a PATH with nothing on it)."""
    import shutil
    import subprocess

    sh = shutil.which("sh")
    assert sh
    ok = subprocess.run([sh, "-c", lifecycle.BOUND_SH, "crucible-bound", "30", sh, "-c", "echo ran; exit 3"], capture_output=True, text=True, timeout=30, check=False)
    assert (ok.returncode, ok.stdout.strip()) == (3, "ran")
    if shutil.which("timeout"):
        hung = subprocess.run([sh, "-c", lifecycle.BOUND_SH, "crucible-bound", "1", sh, "-c", "sleep 5"], capture_output=True, text=True, timeout=25, check=False)
        assert hung.returncode == 124
    bare = subprocess.run([sh, "-c", lifecycle.BOUND_SH, "crucible-bound", "30", sh, "-c", "echo bare"], capture_output=True, text=True, timeout=30, check=False,
                          env={"PATH": ""})
    assert (bare.returncode, bare.stdout.strip()) == (0, "bare")


@posix
def test_the_python_bound_arms_an_alarm_that_survives_the_exec():
    import shutil
    import signal
    import subprocess

    if not shutil.which("python3"):
        pytest.skip("BOUND_PY execs python3 by name, as the grade container's entrypoint does")
    py = shutil.which("python3")
    ok = subprocess.run([py, "-I", "-c", lifecycle.BOUND_PY, "30", "-I", "-c", "import sys; print('ran', sys.argv[1:]); sys.exit(4)", "x"],
                        capture_output=True, text=True, timeout=30, check=False)
    assert (ok.returncode, ok.stdout.strip()) == (4, "ran ['x']")
    hung = subprocess.run([py, "-I", "-c", lifecycle.BOUND_PY, "1", "-I", "-c", "import time; time.sleep(30)"], capture_output=True, text=True, timeout=25, check=False)
    assert hung.returncode == -signal.SIGALRM
