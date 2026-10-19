"""M10 lab audit R-4, slice 1: one writer per run directory, and attempt indices and ledger op ids that are never reused.

The lease is ``run.lock`` (``core.locks.FileLock``) taken by ``RunJournal`` before ``run.json`` is read, before the ledger
opens and before any attempt index is chosen, and held for the whole of ``Runner.run`` / ``Runner.regrade``. Fakes here are
plain data (SimpleNamespace tasks and policies) and harnesses that declare ``identity_config`` (stateful ones), so the
tests hold unchanged under the fail-closed run identity of the later slices.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import gc
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core import locks
from crucible.core.locks import FileLock
from crucible.core.schema.run import Failure
from crucible.runtime.budget import BudgetLedger, Exhausted, OpReused
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.journal import LOCK, OWNER, REFUSED, RunBusy, RunJournal, assert_free
from crucible.runtime.keys import SampleKey
from crucible.runtime.run import Runner, RunSpec

CHILD = (
    "import os, sys\n"
    "from crucible.runtime.journal import RunJournal\n"
    "j = RunJournal(sys.argv[1], 'x', 'id', {})\n"
    "print('ready', os.getpid(), flush=True)\n"  # the interpreter's own pid: under a Windows venv launcher Popen.pid is the launcher's
    "sys.stdin.read()\n"
)


class Ok:
    id = "ok@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0, cost_usd=0.1)


class Blocking:
    """Starts, then waits for the test to open the gate: the run is live while the test probes it."""

    id = "block@1"

    def __init__(self) -> None:
        self.started, self.gate = asyncio.Event(), asyncio.Event()

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        self.started.set()
        await self.gate.wait()
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


class Siblings:
    """Task ``a`` waits until task ``b`` is in its agent phase, then returns; ``b`` blocks, and when it is cancelled it unwinds
    slowly (it waits for the test), as an agent shutting down its container would."""

    id = "siblings@1"

    def __init__(self) -> None:
        self.b_started, self.cancelled, self.unwind = asyncio.Event(), asyncio.Event(), asyncio.Event()

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        if attempt.task.id == "a":
            await self.b_started.wait()
            return HarnessResult(reward=1.0)
        self.b_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            self.cancelled.set()
            await self.unwind.wait()
            raise
        return HarnessResult(reward=1.0)


def spec_of(harness, run_id: str = "r", **kw) -> RunSpec:
    return RunSpec(run_id=run_id, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.5)], harnesses=[harness], k=1, **kw)


def ledger_rows(path: Path) -> dict[str, tuple[str, float | None]]:
    con = sqlite3.connect(path)
    try:
        return {op: (status, actual) for op, status, actual in con.execute("SELECT op, status, actual FROM ops")}
    finally:
        con.close()


def snapshot(run: Path) -> dict:
    """Every byte of the run directory but the lock file (a Windows byte-range lock is mandatory: reading the locked byte
    through another handle can fail) and the ledger's own file (compared through its rows, read-only)."""
    out: dict = {}
    for p in sorted(run.rglob("*")):
        if p.is_file() and p.name != LOCK and not p.name.startswith("ledger.sqlite"):
            out[p.relative_to(run).as_posix()] = p.read_bytes()
        elif p.is_dir():
            out[p.relative_to(run).as_posix() + "/"] = None
    con = sqlite3.connect(f"{(run / 'ledger.sqlite').as_uri()}?mode=ro", uri=True)
    try:
        out["ledger"] = con.execute("SELECT * FROM ops ORDER BY op").fetchall()
    finally:
        con.close()
    return out


def read_ready(child: subprocess.Popen, timeout: float = 60.0) -> str | None:
    """The child's first line, within ``timeout`` (a pipe cannot be select()ed on Windows)."""
    box: list[str] = []
    t = threading.Thread(target=lambda: box.append(child.stdout.readline()), daemon=True)  # type: ignore[union-attr]
    t.start()
    t.join(timeout)
    return box[0].strip() if box else None


# ---------------------------------------------------------------- the primitive


def test_file_lock_is_exclusive_and_released(tmp_path):
    p = tmp_path / "sub" / "x.lock"
    a, b = FileLock(p), FileLock(p)
    a.acquire()
    assert a.held
    with pytest.raises(BlockingIOError):
        b.acquire()  # the same process too: the lock belongs to the open file description, not to the process
    assert not b.held
    a.release()
    a.release()  # idempotent
    assert not a.held
    b.acquire()
    b.release()
    c = FileLock(p)
    c.acquire()
    del c  # dropped without release: freed by the finalizer
    gc.collect()
    d = FileLock(p)
    d.acquire()
    d.release()
    assert p.exists()  # never unlinked


def test_a_lock_held_by_another_process_names_it_and_dies_with_it(tmp_path):
    child = subprocess.Popen([sys.executable, "-c", CHILD, str(tmp_path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        tag, _, pid = (read_ready(child) or "").partition(" ")
        assert tag == "ready" and pid.isdigit()  # the pid the child reports is the process that holds the lock (not necessarily Popen.pid)
        with pytest.raises(RunBusy) as busy:
            RunJournal(tmp_path, "x", "id", {})
        assert pid in str(busy.value)
        owner = json.loads((tmp_path / "x" / OWNER).read_text())
        assert {"host", "pid", "process_started", "started_at"} <= set(owner) and owner["pid"] == int(pid)
    finally:
        with contextlib.suppress(OSError):
            child.stdin.close()  # type: ignore[union-attr]  # the child reads it to EOF and exits; kill only if it does not (a launcher's kill spares its child)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    RunJournal(tmp_path, "x", "id", {}).close()  # the OS dropped the dead holder's lock: nothing to reclaim


@pytest.mark.skipif(sys.platform == "win32", reason="fork (POSIX): a Windows child never inherits the handle")
def test_a_forked_child_does_not_keep_a_dead_holders_lease(tmp_path):
    """``flock`` belongs to the open file description a fork shares: a child that outlives a killed holder (``verify.answers``
    grades a math answer in a forked process that can run for hours) used to keep the run locked, with no stale-lock reclaim."""
    path = tmp_path / "run.lock"
    hold_r, hold_w = os.pipe()  # the orphan blocks reading it until the test closes its write end
    up_r, up_w = os.pipe()  # the orphan says it is alive
    holder = os.fork()
    if holder == 0:  # the holder: takes the lock, forks the orphan, then dies without releasing
        try:
            lock = FileLock(path)
            lock.acquire()
            if os.fork() == 0:
                try:
                    os.close(hold_w)
                    os.write(up_w, b"x")
                    os.read(hold_r, 1)
                finally:
                    os._exit(0)
            os.kill(os.getpid(), signal.SIGKILL)
        finally:
            os._exit(1)
    os.close(hold_r)
    os.close(up_w)
    try:
        assert os.read(up_r, 1) == b"x"  # the orphan is running
        _, status = os.waitpid(holder, 0)
        assert os.WIFSIGNALED(status)  # the holder died without releasing
        again = FileLock(path)
        again.acquire()  # BlockingIOError here: the orphan still held the shared description
        again.release()
    finally:
        os.close(hold_w)  # the orphan reads EOF and exits
        os.close(up_r)


def test_the_held_descriptors_are_tracked_and_a_release_is_idempotent(tmp_path):
    lock = FileLock(tmp_path / "x.lock")
    lock.acquire()
    fd = lock._fd
    assert fd in locks._HELD
    lock.release()
    assert fd not in locks._HELD
    lock.release()
    locks._release(fd)  # a finalizer that runs after the descriptor was dropped closes nothing: the number may belong to another file now
    again = FileLock(tmp_path / "x.lock")
    again.acquire()  # the path is free
    again.release()


# ---------------------------------------------------------------- the lease on a run directory


def test_the_journal_names_its_owner_and_close_frees_it(tmp_path):
    with RunJournal(tmp_path, "r", "id", {}) as j:
        owner = json.loads((j.dir / OWNER).read_text())
        assert owner["pid"] == os.getpid() and isinstance(owner["started_at"], float)
        with pytest.raises(RunBusy, match=str(os.getpid())):
            RunJournal(tmp_path, "r", "id", {})
        with pytest.raises(RunBusy, match=str(os.getpid())):
            assert_free(j.dir)
    assert_free(tmp_path / "r")
    assert (tmp_path / "r" / LOCK).exists()
    RunJournal(tmp_path, "r", "id", {}).close()


def test_assert_free_creates_nothing(tmp_path):
    assert_free(tmp_path / "never-opened")
    assert not (tmp_path / "never-opened").exists()


def test_a_refused_resume_releases_the_lease(tmp_path):
    RunJournal(tmp_path, "r", "id-1", {}).close()
    with pytest.raises(ValueError, match="different identity"):
        RunJournal(tmp_path, "r", "id-2", {})
    RunJournal(tmp_path, "r", "id-1", {}).close()  # the refused opener left nothing locked


async def test_a_second_runner_or_regrade_on_a_live_run_is_refused_and_changes_nothing(tmp_path):
    h = Blocking()
    spec = spec_of(h, "live", budget_usd=5.0)
    sid = Runner(tmp_path).plan(spec)[0][0].id()
    first = asyncio.create_task(Runner(tmp_path).run(spec))
    await asyncio.wait_for(h.started.wait(), 30)
    run = tmp_path / "live"
    before = snapshot(run)
    assert (run / LOCK).exists()
    with pytest.raises(RunBusy, match=str(os.getpid())):
        await Runner(tmp_path).run(spec)
    with pytest.raises(RunBusy, match=str(os.getpid())):
        await Runner(tmp_path).regrade(spec)
    assert snapshot(run) == before
    assert (run / "samples" / sid / "a1").is_dir() and not (run / "samples" / sid / "aborted").exists()
    h.gate.set()
    assert (await first).terminal == 1
    again = await Runner(tmp_path).run(spec)  # the lease was released: a third run resumes
    assert again.attempts == 0 and again.terminal == 1


async def test_a_second_bench_run_on_the_same_name_is_refused(tmp_path):
    from crucible.bench.run import run_bench

    h = Blocking()
    kw = {"root": tmp_path, "tasks": [SimpleNamespace(id="t")], "policies": [SimpleNamespace(id="p", estimate_usd=0.0)], "harnesses": [h], "k": 1}
    first = asyncio.create_task(run_bench("demo", **kw))
    await asyncio.wait_for(h.started.wait(), 30)
    tree = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
    with pytest.raises(RunBusy, match=str(os.getpid())):
        await run_bench("demo", **kw)
    assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == tree  # the second call created no file
    h.gate.set()
    assert await first


@pytest.mark.parametrize("held", ["bench-demo", "bench-demo-controls"])
async def test_a_bench_run_with_controls_is_refused_before_anything_is_written(tmp_path, held):
    """With controls, the preflight covers the ``-controls`` run id too: the controls run (its own run dir, its episodes) and
    ``validity.json`` are written before the main run takes its lease, so the second bench is refused first."""
    from crucible.bench.run import run_bench

    first = RunJournal(tmp_path, held, "id", {})  # another process's open run (the main one, or its controls)
    try:
        tree = sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*"))
        kw = {"root": tmp_path, "tasks": [SimpleNamespace(id="t")], "policies": [SimpleNamespace(id="p", estimate_usd=0.0)], "harnesses": [Ok()], "k": 1}
        with pytest.raises(RunBusy, match=str(os.getpid())):
            await run_bench("demo", controls=True, **kw)
        assert sorted(p.relative_to(tmp_path).as_posix() for p in tmp_path.rglob("*")) == tree  # no controls run, no validity.json
    finally:
        first.close()


async def test_the_lease_is_held_until_every_attempt_has_ended(tmp_path, monkeypatch):
    """One attempt raises (a full disk under ``journal.finish``) while a sibling is still running: ``gather`` alone would re-raise at
    once and let ``Runner.run`` release the lease under the sibling, which a second ``Runner.run`` then takes, moving its live dir
    to ``aborted/`` and dispatching the same sample again. The sibling is cancelled and awaited first."""
    h = Siblings()
    spec = RunSpec(run_id="r", tasks=[SimpleNamespace(id="a"), SimpleNamespace(id="b")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[h],
                   k=1, concurrency=2)
    real = RunJournal.finish

    def finish(self, rec, *, terminal):
        if rec.task == "a":
            raise OSError("disk full")
        return real(self, rec, terminal=terminal)

    monkeypatch.setattr(RunJournal, "finish", finish)
    first = asyncio.create_task(Runner(tmp_path).run(spec))
    await asyncio.wait_for(h.cancelled.wait(), 30)  # a raised, b was cancelled and is still unwinding
    assert not first.done()
    with pytest.raises(RunBusy):
        await Runner(tmp_path).run(spec)  # the lease is still the first run's
    assert not [p for p in (tmp_path / "r" / "samples").rglob("aborted")]  # and b's live dir was not moved
    h.unwind.set()
    with pytest.raises(OSError, match="disk full"):
        await asyncio.wait_for(first, 30)
    assert_free(tmp_path / "r")  # only now is the lease released


# ---------------------------------------------------------------- attempt indices and ledger ops are never reused


async def test_a_crash_then_a_refused_reservation_never_reuses_an_attempt_or_a_ledger_row(tmp_path):
    spec = spec_of(Ok(), "r", concurrency=1, budget_usd=10.2, retry_backoff_s=0.0)
    sid = Runner(tmp_path).plan(spec)[0][0].id()
    sdir = tmp_path / "r" / "samples" / sid
    (sdir / "a1").mkdir(parents=True)  # a crash: a1 reserved and dispatched, no record
    led = BudgetLedger(tmp_path / "r" / "ledger.sqlite", {})
    led.reserve("seed", ["run"], 9.0)
    led.settle("seed", 9.0)
    led.reserve(f"{sid}-a1", ["run"], 1.0)
    led.db.close()  # the uncertain share is exactly 10%: not over tolerance

    s = await Runner(tmp_path).run(spec)  # a2's 0.5 on top of 10.0 spent is above 10.2
    assert s.stopped and s.stopped.startswith("budget") and s.terminal == 0 and s.attempts == 0
    assert (sdir / "aborted" / "a1").is_dir()
    assert (sdir / "a2" / REFUSED).exists() and not (sdir / "a2" / "record.json").exists()
    rows = ledger_rows(tmp_path / "r" / "ledger.sqlite")
    assert rows[f"{sid}-a1"] == ("uncertain", 1.0) and f"{sid}-a2" not in rows

    s2 = await Runner(tmp_path).run(dataclasses.replace(spec, budget_usd=20.0))
    assert s2.terminal == 1 and s2.attempts == 1
    last = json.loads((tmp_path / "r" / "results.jsonl").read_text().splitlines()[-1])
    assert last["attempt"].endswith("-a3")
    rows = ledger_rows(tmp_path / "r" / "ledger.sqlite")
    assert rows[f"{sid}-a1"] == ("uncertain", 1.0) and f"{sid}-a2" not in rows and rows[f"{sid}-a3"][0] == "settled"
    assert (sdir / "a2" / REFUSED).exists()  # still in place, still holding its index
    assert sorted(p.name for p in (sdir / "aborted").iterdir()) == ["a1"]


async def test_a_refused_reservation_is_no_retry(tmp_path):
    h = Failing(fail=2)
    spec = spec_of(h, "r", concurrency=1, budget_usd=0.55, retry_backoff_s=0.0)  # a1 spends 0.1; a2's 0.5 then crosses the cap
    sid = Runner(tmp_path).plan(spec)[0][0].id()
    s = await Runner(tmp_path).run(spec)
    sdir = tmp_path / "r" / "samples" / sid
    assert s.stopped and s.stopped.startswith("budget") and s.terminal == 0 and (sdir / "a2" / REFUSED).exists()

    s2 = await Runner(tmp_path).run(dataclasses.replace(spec, budget_usd=20.0))
    # a3 fails again: the retry rule sees 2 dispatched attempts (a1, a3), not the index 3, so it is retried and a4 scores
    assert (s2.terminal, s2.scored, s2.attempts, h.calls) == (1, 1, 2, 3)
    last = json.loads((tmp_path / "r" / "results.jsonl").read_text().splitlines()[-1])
    assert last["attempt"].endswith("-a4") and last["reward"] == 1.0 and not last["mask_sample"]


async def test_an_op_id_that_already_exists_stops_the_run_and_raises_nothing(tmp_path):
    spec = spec_of(Ok(), "r", retry_backoff_s=0.0)
    sid = Runner(tmp_path).plan(spec)[0][0].id()
    led = BudgetLedger(tmp_path / "r" / "ledger.sqlite", {})
    led.reserve(f"{sid}-a1", ["run"], 0.1)  # an op named like the next attempt (settled: the uncertain-spend rule stays out of it)
    led.settle(f"{sid}-a1", 0.1)
    led.db.close()
    s = await Runner(tmp_path).run(spec)
    assert s.stopped and s.stopped.startswith("ledger") and s.terminal == 0 and s.attempts == 0
    assert (tmp_path / "r" / "samples" / sid / "a1" / REFUSED).exists()
    assert not (tmp_path / "r" / "results.jsonl").exists()
    events = [json.loads(x) for x in (tmp_path / "r" / "journal.jsonl").read_text().splitlines()]
    assert any(e["event"] == "stop_rule" and e.get("attempt") == f"{sid}-a1" for e in events)
    s2 = await Runner(tmp_path).run(spec)  # the next run takes a2: a fresh op, nothing is reused
    assert s2.terminal == 1 and ledger_rows(tmp_path / "r" / "ledger.sqlite")[f"{sid}-a2"][0] == "settled"


def test_ledger_op_ids_are_never_reused(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", {"run": 1.0})
    led.reserve("x", ["run"], 0.1)
    with pytest.raises(OpReused, match="already exists"):
        led.reserve("x", ["run"], 0.2)
    assert led.db.execute("SELECT op, status, reserved FROM ops").fetchall() == [("x", "reserved", 0.1)]
    with pytest.raises(Exhausted):
        led.reserve("y", ["run"], 5.0)
    assert [r[0] for r in led.db.execute("SELECT op FROM ops")] == ["x"]  # a refusal writes no row


def test_an_aborted_move_never_nests(tmp_path):
    with RunJournal(tmp_path, "r", "id", {}) as j:
        sdir = j.dir / "samples" / "sid"
        (sdir / "aborted" / "a1").mkdir(parents=True)  # a pre-R-4 leftover
        (sdir / "a1").mkdir()  # no record: a crash
        assert j.next_attempt("sid") == 2
        assert sorted(p.name for p in (sdir / "aborted").iterdir()) == ["a1", "a1~1"]
        assert not (sdir / "aborted" / "a1" / "a1").exists()
        (sdir / "a2").mkdir()
        assert j.next_attempt("sid") == 3 and (sdir / "aborted" / "a2").is_dir()
        assert sorted(p.name for p in (sdir / "aborted").iterdir()) == ["a1", "a1~1", "a2"]


def test_a_refused_reservation_keeps_its_index_and_stays_in_place(tmp_path):
    with RunJournal(tmp_path, "r", "id", {}) as j:
        j.attempt_dir("sid", 1).mkdir(parents=True)
        j.refuse(j.attempt_dir("sid", 1), "sid-a1", "budget: x")
        assert json.loads((j.attempt_dir("sid", 1) / REFUSED).read_text())["attempt"] == "sid-a1"
        assert j.next_attempt("sid") == 2
        assert j.attempt_dir("sid", 1).is_dir() and not (j.dir / "samples" / "sid" / "aborted").exists()
        assert j.dispatched("sid", 2) == 1  # the refused index is not a try
        j.attempt_dir("sid", 2).mkdir()
        assert j.dispatched("sid", 3) == 2 and j.dispatched("sid", 1) == 1


def test_resume_looks_past_a_refused_attempt(tmp_path):
    sdir = tmp_path / "s"
    (sdir / "aborted" / "a1").mkdir(parents=True)
    (sdir / "aborted" / "a1" / "checkpoint.json").write_text(json.dumps({"step": 5, "work_attempt": "a1"}))
    (sdir / "a2").mkdir()
    (sdir / "a2" / REFUSED).write_text("{}")
    att = Attempt(sample=SampleKey(task="t", policy="p", harness="h", sample_idx=0, seed=1), id="s-a3", n=3, dir=sdir / "a3",
                  task=SimpleNamespace(id="t"), policy=None)
    got = NativeHarness(checkpoint_every=1)._resumable(att)
    assert got is not None and got[0] == sdir / "aborted" / "a1" and got[1]["step"] == 5
