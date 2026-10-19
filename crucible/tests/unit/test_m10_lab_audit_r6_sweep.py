"""M10 lab audit R-6, slice 3: the lease-keyed sweep at Runner start (``lifecycle.sweep``, ``Runner._sweep``) and PerfGuard's expiry rule.

After a crash, kill -9 or OOM kill the next start removes this run's orphans, the orphans of runs whose ``run.lock`` is free, and any
container past its expiry. It is keyed on R-4's lock (free or held), never on pid liveness (``os.kill(pid, 0)`` has no meaning on
Windows), never touches a container without ``crucible.owner``, never creates a lock file, and a failed sweep never fails the run.
An expired orphan is never PerfGuard contention (a renewable keeper always is).

No Docker: ``FakeDocker`` (``test_m10_lab_audit_r6_lifecycle``) stands in for ``world.exec._run_subprocess`` and renders the ``ps
--filter label=K --format PS_FORMAT`` listing the sweep parses.
"""

from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core.locks import FileLock
from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.perf_guard import DOCKER_PS, PerfGuard, crucible_containers, sample
from crucible.runtime.run import Runner, RunSpec
from crucible.world import lifecycle
from crucible.world.exec import ExecOut
from crucible.world.lifecycle import Hygiene, Row
from crucible.world.sandbox import DockerSandbox
from tests.unit.test_m10_lab_audit_r6_lifecycle import FakeDocker

ME = socket.gethostname()


def _spec() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/r6s", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})


class DownDocker(FakeDocker):
    """A ``FakeDocker`` some hosts of which cannot be reached (every call answers rc 255)."""

    def __init__(self, down: set[str] | None = None) -> None:
        super().__init__()
        self.down = down if down is not None else set()

    async def __call__(self, argv, **kw) -> ExecOut:
        host = (kw.get("env") or {}).get("DOCKER_HOST", "")
        if host in self.down:
            self.calls.append((host, list(argv)))
            return ExecOut(255, "", "Cannot connect to the Docker daemon")
        return await super().__call__(argv, **kw)


class LeaseChecking(FakeDocker):
    """Asserts, on every ``ps``, that the run's lease is held: the sweep runs only under it."""

    def __init__(self, lock: Path) -> None:
        super().__init__()
        self.lock, self.checked = lock, 0

    async def __call__(self, argv, **kw) -> ExecOut:
        if argv[1] == "ps":
            probe = FileLock(self.lock)
            with pytest.raises(BlockingIOError):
                probe.acquire()
            self.checked += 1
        return await super().__call__(argv, **kw)


def install(monkeypatch, f: FakeDocker) -> FakeDocker:
    from crucible.world import sandbox as sandbox_mod

    monkeypatch.setattr(sandbox_mod, "_run_subprocess", f)
    monkeypatch.setattr(lifecycle, "_run_subprocess", f)
    return f


@pytest.fixture(autouse=True)
def fake(monkeypatch) -> FakeDocker:
    from crucible.world import pool as pool_mod

    monkeypatch.setattr(lifecycle, "docker_available", lambda: True)  # the unit conftest turns it off: the sweep is what is under test
    monkeypatch.setattr(lifecycle, "_install_reaper", lambda: None)
    monkeypatch.setattr(lifecycle, "_HOSTS", set())
    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)
    monkeypatch.setattr(pool_mod, "_POOLS", {})
    return install(monkeypatch, FakeDocker())


def labels(*, owner: str | None = f"{ME}:1:1", expires: float | None = None, lease: str = "", run: str = "", renewable: str = "") -> dict[str, str]:
    out: dict[str, str] = {}
    if owner is not None:
        out["crucible.owner"] = owner
    if expires is not None:
        out["crucible.expires"] = str(int(expires))
    if lease:
        out["crucible.lease"] = lease
    if run:
        out["crucible.run"] = run
    if renewable:
        out["crucible.renewable"] = renewable
    return out


def events_of(root: Path, run_id: str = "r", kind: str | None = None) -> list[dict]:
    got = [json.loads(x) for x in (root / run_id / "journal.jsonl").read_text().splitlines() if x.strip()]
    return [e for e in got if kind is None or e["event"] == kind]


class Ok:
    id = "ok@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        return HarnessResult(reward=1.0, cost_usd=0.1)


class UsesLab:
    """Starts a sandbox on a lab docker host and closes it: the run journals the host (``sandbox_host``)."""

    id = "lab@1"

    def identity_config(self) -> dict:
        return {}

    async def run(self, attempt: Attempt) -> HarnessResult:
        sbx = DockerSandbox(SandboxSpec(image="img", docker_host="ssh://lab"), _spec())
        await sbx.start()
        await sbx.close()
        return HarnessResult(reward=1.0, cost_usd=0.1)


def spec_of(harness, run_id: str = "r", **kw) -> RunSpec:
    kw = {"breaker": None, **kw}
    return RunSpec(run_id=run_id, tasks=[SimpleNamespace(id="t")], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[harness], k=1, **kw)


# ---------------------------------------------------------------- the rules


def test_rows_parse_the_ps_listing_and_ignore_anything_else():
    text = ("crucible-sbx-a\tr\ts-a1\th:1:2\t1700000000\t/x/run.lock\t1\n"
            "crucible-hgrade-b\t\t\th:1:2\t1700000001\t\t\r\n"
            "WARNING: something the CLI printed\n"
            "\n"
            "too\tfew\tfields\n"
            "crucible-sbx-old\tr\ts-a1\th:1:2\t1700000000\t/x/run.lock\n")  # six fields: a format from before the renewable column
    got = lifecycle.rows(text)
    assert got == [Row("crucible-sbx-a", "r", "s-a1", "h:1:2", "1700000000", "/x/run.lock", "1"), Row("crucible-hgrade-b", "", "", "h:1:2", "1700000001", "", "")]
    assert lifecycle.PS_FORMAT.count("\t") == 6 and lifecycle.PS_FORMAT.startswith("{{.Names}}") and 'crucible.lease' in lifecycle.PS_FORMAT
    assert lifecycle.PS_FORMAT.endswith('{{.Label "crucible.renewable"}}')  # review round 1: a renewable keeper's label is a floor, not a promise


def test_stale_rules_are_keyed_on_the_lock_never_on_the_pid():
    now, own = 1000.0, "/r/run.lock"
    free: dict[str, bool | None] = {"/f/run.lock": True, "/h/run.lock": False, "/u/run.lock": None}

    def lease_free(path: str) -> bool | None:
        return free[path]  # a KeyError means the sweep probed a lease a rule says it must not probe (None: cannot tell)

    def row(name: str, *, owner: str = f"{ME}:1:1", expires: float | str = 2000, lease: str = "", renewable: str = "") -> Row:
        return Row(name, "", "", owner, str(expires), lease, renewable)

    found = [
        row("own", lease=own, expires=2000),                       # this run's lease
        row("own-expired", lease=own, expires=1),                  # ... even expired
        row("held-expired", lease="/h/run.lock", expires=1),       # same host, another run's lock is held: renewed past its label, kept
        row("free", lease="/f/run.lock", expires=2000),            # same host, the owning run's lock is free
        row("far-new", owner="other:9:9", lease="/f/run.lock", expires=2000),   # another host: the lock says nothing, the label does
        row("far-old", owner="other:9:9", lease="/f/run.lock", expires=1),
        row("lone-old", expires=1),                                # no lease at all: the label decides
        row("lone-new", expires=2000),
        row("garbled", expires="soon"),                            # an unparseable expiry is not expired
        row("nobody", owner="", expires=1),                        # no crucible.owner: never touched
        row("forged-old", lease="/etc/passwd", expires=1),         # a lease that is not a run.lock is never probed: the label decides
        row("forged-new", lease="/etc/passwd", expires=2000),
        row("relative", lease="run.lock", expires=2000),           # not absolute: never probed
        # review round 1: a lease this host cannot resolve (the other OS flavour, an unmounted runs root) is "cannot tell": the label decides
        row("cannot-tell-old", lease="/u/run.lock", expires=1),
        row("cannot-tell-new", lease="/u/run.lock", expires=2000),
        # review round 1: two machines with one checkout path and a fixed run id carry the same lease string: only this host's is "own"
        row("own-far-new", owner="other:9:9", lease=own, expires=2000),
        row("own-far-old", owner="other:9:9", lease=own, expires=1),
        # review round 1: a renewable keeper's label is a floor: label expiry alone never removes one
        row("renewable-old", lease="/h/run.lock", expires=1, renewable="1"),               # same host, held: kept
        row("renewable-lone-old", expires=1, renewable="1"),                               # no lease: its real deadline decides (none read: kept)
        row("renewable-far-old", owner="other:9:9", lease="/f/run.lock", expires=1, renewable="1"),
        row("renewable-new", expires=2000, renewable="1"),
        row("renewable-own", lease=own, expires=1, renewable="1"),                         # this run's, at its start: still own_run
    ]
    got = dict(lifecycle.stale(found, own_lock=own, reclaim_own=True, owner_host=ME, now=now, lease_free=lease_free))
    assert got == {"own": "own_run", "own-expired": "own_run", "free": "lease_free", "far-old": "expired", "lone-old": "expired", "forged-old": "expired",
                   "cannot-tell-old": "expired", "own-far-old": "expired", "renewable-own": "own_run"}

    # not at a Runner start (reclaim_own False): the run's own containers belong to its live attempts, expired label or not
    got = dict(lifecycle.stale(found, own_lock=own, reclaim_own=False, owner_host=ME, now=now, lease_free=lease_free))
    assert got == {"free": "lease_free", "far-old": "expired", "lone-old": "expired", "forged-old": "expired", "cannot-tell-old": "expired",
                   "own-far-old": "expired"}

    # without an own lock nothing is "own"
    got = dict(lifecycle.stale([row("x", lease=own, expires=2000)], own_lock=None, reclaim_own=True, owner_host=ME, now=now, lease_free=lambda p: False))
    assert got == {}


def test_a_renewable_container_past_its_label_is_judged_by_its_real_deadline():
    now, day = 1_000_000.0, lifecycle.UNREADABLE_GRACE_S

    def row(name: str, *, expires: float, renewable: str = "1", owner: str = "other:9:9", lease: str = "") -> Row:
        return Row(name, "", "", owner, str(expires), lease, renewable)

    found = [
        row("alive", expires=now - 10),                    # another host's live holder renewed it: the real deadline is ahead
        row("dead", expires=now - 10),                     # its real deadline has passed too
        row("unreadable-fresh", expires=now - 10),         # cannot be read (paused, unreachable): kept until a day past its label
        row("unreadable-stale", expires=now - day - 10),
        row("plain-old", expires=now - 10, renewable=""),  # a fixed keeper's label is exact: removed
        row("lease-less-pooled", expires=now - 10, owner=f"{ME}:1:1"),  # a pooled container used outside a Runner (lease ''): same rule
    ]
    kw = {"own_lock": None, "reclaim_own": False, "owner_host": ME, "now": now, "lease_free": lambda p: None}
    pend = lifecycle.pending_deadlines(found, **kw)
    assert [r.name for r in pend] == ["alive", "dead", "unreadable-fresh", "unreadable-stale", "lease-less-pooled"]  # the ones a label cannot judge
    deadlines = {"alive": now + 3600, "dead": now - 5, "lease-less-pooled": now + 60}  # the others: unreadable
    got = dict(lifecycle.stale(found, **kw, deadlines=deadlines))
    assert got == {"dead": "expired", "unreadable-stale": "expired", "plain-old": "expired"}
    # no probe results at all (a caller that never read any): nothing renewable goes before its grace
    got = dict(lifecycle.stale(found, **kw))
    assert got == {"unreadable-stale": "expired", "plain-old": "expired"}


def test_is_run_lock_accepts_only_an_absolute_run_lock_path():
    assert lifecycle.is_run_lock("/data/runs/r/run.lock") and lifecycle.is_run_lock("C:\\data\\runs\\r\\run.lock")
    assert not lifecycle.is_run_lock("") and not lifecycle.is_run_lock("run.lock") and not lifecycle.is_run_lock("/data/runs/r/owner.json")
    assert not lifecycle.is_run_lock("/data/runs/r/run.lock.bak") and not lifecycle.is_run_lock("/etc/passwd")


def test_lease_free_reads_a_real_lock_and_never_creates_one(tmp_path):
    held_path, idle_path, gone_path, other_path = (tmp_path / "a" / "run.lock", tmp_path / "b" / "run.lock", tmp_path / "c" / "run.lock",
                                                    tmp_path / "d" / "other.lock")
    idle_path.parent.mkdir()
    idle_path.write_text("")
    other_path.parent.mkdir()
    other_path.write_text("")
    held = FileLock(held_path)
    held.acquire()
    try:
        assert lifecycle.lease_free(str(held_path)) is False  # a second opener of a held lock is refused, even in this process
    finally:
        held.release()
    assert lifecycle.lease_free(str(held_path)) is True  # released: free
    assert lifecycle.lease_free(str(idle_path)) is True
    probe = FileLock(idle_path)
    probe.acquire()  # the probe gave the lock back
    probe.release()
    assert lifecycle.lease_free(str(gone_path)) is True and not gone_path.exists() and not gone_path.parent.exists()  # the run dir is gone: its owner is gone
    assert lifecycle.lease_free("") is False and lifecycle.lease_free(str(other_path)) is False  # not a run.lock: never opened
    assert lifecycle.lease_free("run.lock") is False


def test_lease_state_says_cannot_tell_for_a_lease_this_host_cannot_resolve(tmp_path):
    """Review round 1: ``lease_free`` took "the file does not exist" as "the run is gone" for any run.lock-named path. A WSL2 runner and
    a Windows runner sharing Docker Desktop's engine (the hostname can match) carry each other's paths, which never resolve here, and a
    runs root on a share this host has not mounted is just as unreadable: all of those are ``None``, and the expiry label decides."""
    held_path, idle_path = tmp_path / "root" / "a" / "run.lock", tmp_path / "root" / "b" / "run.lock"
    idle_path.parent.mkdir(parents=True)
    idle_path.write_text("")
    held = FileLock(held_path)  # creates its directory
    held.acquire()
    try:
        assert lifecycle.lease_state(str(held_path)) is False
    finally:
        held.release()
    assert lifecycle.lease_state(str(idle_path)) is True
    assert lifecycle.lease_state(str(tmp_path / "root" / "gone" / "run.lock")) is True  # the run directory is gone under a runs root that exists
    assert lifecycle.lease_state(str(tmp_path / "unmounted" / "r" / "run.lock")) is None  # ... but not when the runs root itself is missing
    foreign = "/home/u/crucible/data/runs/R1/run.lock" if os.name == "nt" else "C:\\data\\runs\\R1\\run.lock"  # the other OS's flavour
    assert lifecycle.is_run_lock(foreign) and lifecycle.lease_state(foreign) is None and lifecycle.lease_free(foreign) is False
    assert lifecycle.lease_state("run.lock") is None and lifecycle.lease_state("/etc/passwd") is None and lifecycle.lease_state("") is None


# ---------------------------------------------------------------- lifecycle.sweep on a fake daemon


async def test_a_sweep_covers_exactly_the_hosts_it_is_given(fake):
    old = time.time() - 60
    fake.add("", "crucible-sbx-local", labels(owner="other:1:1", expires=old))
    fake.add("ssh://lab", "crucible-sbx-lab", labels(owner="other:1:1", expires=old))
    rep = await lifecycle.sweep(["ssh://lab"], own_lock=None, reclaim_own=False)
    assert rep.hosts == ["ssh://lab"] and rep.removed == [{"name": "crucible-sbx-lab", "host": "ssh://lab", "why": "expired"}] and rep.failed == [] and rep.errors == {}
    assert fake.containers() == ["crucible-sbx-local"]  # the local daemon was not asked: sweep() adds no host of its own
    assert {h for h, _ in fake.calls} == {"ssh://lab"}


async def test_a_sweep_reports_what_it_could_not_remove_and_a_host_it_could_not_list(monkeypatch):
    f = install(monkeypatch, DownDocker(down={"ssh://down"}))
    old = time.time() - 60
    f.add("", "crucible-sbx-a", labels(owner="other:1:1", expires=old))
    f.add("", "crucible-sbx-b", labels(owner="other:1:1", expires=old))
    f.add("", "crucible-sbx-keep", labels(owner="other:1:1", expires=time.time() + 3600))
    f.fail_rm.add("crucible-sbx-b")
    rep = await lifecycle.sweep(["", "ssh://down"], own_lock=None, reclaim_own=False)
    assert rep.hosts == ["", "ssh://down"]
    assert [r["name"] for r in rep.removed] == ["crucible-sbx-a"] and [x["name"] for x in rep.failed] == ["crucible-sbx-b"]
    assert "unreachable" in rep.failed[0]["error"] and rep.failed[0]["host"] == ""
    assert "Cannot connect" in rep.errors["ssh://down"] and rep.kept == 1
    assert f.containers() == ["crucible-sbx-b", "crucible-sbx-keep"]
    assert rep.as_dict()["kept"] == 1 and set(rep.as_dict()) == {"hosts", "removed", "failed", "kept", "errors"}


async def test_a_sweep_spares_a_live_renewed_container_of_another_host_and_reads_its_real_deadline(fake):
    """Review round 1: a renewed keeper outlives its expiry label. Runner Y on host B sweeping the shared lab daemon saw runner X's
    live, renewed container (owner host A, past its label) as ``expired`` and removed it, a pooled one used with no lease the same.
    Now the label alone never removes a renewable container: its real deadline (``/crucible-ctl/deadline``) is read first."""
    now = time.time()
    old = now - 60
    fake.add("", "crucible-sbx-live", labels(owner="other:1:1", expires=old, renewable="1"))
    fake.add("", "crucible-sbx-dead", labels(owner="other:1:1", expires=old, renewable="1"))
    fake.add("", "crucible-sbx-paused", labels(owner="other:1:1", expires=old, renewable="1"))  # its deadline cannot be read
    fake.add("", "crucible-sbx-pooled", labels(owner=f"{ME}:1:1", expires=old, renewable="1"))  # a pooled container, used outside a Runner
    fake.add("", "crucible-sbx-fixed", labels(owner="other:1:1", expires=old))  # a fixed keeper: its label is exact
    fake.deadlines = {"crucible-sbx-live": int(now) + 7200, "crucible-sbx-dead": int(now) - 5, "crucible-sbx-pooled": int(now) + 600}
    rep = await lifecycle.sweep([""], own_lock=None, reclaim_own=False)
    assert {r["name"]: r["why"] for r in rep.removed} == {"crucible-sbx-dead": "expired", "crucible-sbx-fixed": "expired"} and rep.failed == []
    assert fake.containers() == ["crucible-sbx-live", "crucible-sbx-paused", "crucible-sbx-pooled"] and rep.kept == 3
    probes = [a for _, a in fake.verbs("exec")]
    assert sorted(a[-3] for a in probes) == ["crucible-sbx-dead", "crucible-sbx-live", "crucible-sbx-paused", "crucible-sbx-pooled"]  # never the fixed one
    assert all(a[:4] == ["docker", "exec", "--user", "0"] and a[-2:] == ["cat", "/crucible-ctl/deadline"] for a in probes)


async def test_a_sweep_probes_nothing_when_every_label_decides(fake):
    now = time.time()
    fake.add("", "crucible-sbx-new", labels(owner="other:1:1", expires=now + 3600, renewable="1"))  # not past its label: no probe
    fake.add("", "crucible-sbx-fixed", labels(owner="other:1:1", expires=now - 60))
    rep = await lifecycle.sweep([""], own_lock=None, reclaim_own=False)
    assert [r["name"] for r in rep.removed] == ["crucible-sbx-fixed"] and fake.verbs("exec") == []


async def test_a_sweep_that_cannot_read_a_deadline_keeps_the_container(monkeypatch):
    class NoExec(FakeDocker):
        async def __call__(self, argv, **kw) -> ExecOut:
            if argv[1] == "exec":
                raise OSError("docker exec: not found")
            return await super().__call__(argv, **kw)

    f = install(monkeypatch, NoExec())
    f.add("", "crucible-sbx-renewed", labels(owner="other:1:1", expires=time.time() - 60, renewable="1"))
    rep = await lifecycle.sweep([""], own_lock=None, reclaim_own=False)
    assert rep.removed == [] and rep.failed == [] and rep.errors == {} and f.containers() == ["crucible-sbx-renewed"]


async def test_a_sweep_never_raises(monkeypatch):
    async def boom(*a, **kw):
        raise OSError("docker: not found")

    monkeypatch.setattr(lifecycle, "_run_subprocess", boom)
    rep = await lifecycle.sweep(["", "ssh://lab"], own_lock=None, reclaim_own=True)
    assert set(rep.errors) == {"", "ssh://lab"} and "OSError" in rep.errors[""] and rep.removed == [] and rep.failed == []


async def test_a_sweep_goes_through_its_docker_seam(fake):
    seen: list[tuple[list[str], str | None, float]] = []

    async def seam(args, host, *, timeout_s):
        seen.append((args, host, timeout_s))
        return ExecOut(0, "")

    rep = await lifecycle.sweep(["ssh://lab"], own_lock=None, reclaim_own=False, docker=seam)
    assert rep.errors == {} and fake.calls == []
    (args, host, timeout_s), = seen
    assert args[:5] == ["ps", "-a", "--filter", "label=crucible.owner", "--format"] and args[5] == lifecycle.PS_FORMAT
    assert host == "ssh://lab" and timeout_s == lifecycle.LIST_TIMEOUT_S


# ---------------------------------------------------------------- the Runner's sweep


async def test_the_runner_sweeps_after_taking_its_lease_and_journals_it(tmp_path, monkeypatch):
    run_dir = tmp_path / "r"
    lock = str((run_dir / "run.lock").resolve())
    dead_lock, live_lock = tmp_path / "dead" / "run.lock", tmp_path / "live" / "run.lock"
    dead_lock.parent.mkdir()
    dead_lock.write_text("")  # a run that is gone: its lock file is there and nobody holds it
    f = install(monkeypatch, LeaseChecking(run_dir / "run.lock"))
    now = time.time()
    f.add("", "crucible-sbx-own", labels(lease=lock, run="r", expires=now + 3600))
    f.add("", "crucible-sbx-dead", labels(lease=str(dead_lock), run="gone", expires=now + 3600))
    f.add("", "crucible-sbx-live", labels(lease=str(live_lock), run="live", expires=now - 10))  # held: renewed past its label
    f.add("", "crucible-sbx-foreign-old", labels(owner="otherhost:1:1", expires=now - 10))
    f.add("", "crucible-sbx-foreign-new", labels(owner="otherhost:1:1", expires=now + 3600))
    f.add("", "plain-rlm-kernel", {"crucible.rlm.owner": f"{ME}:1", "crucible.rlm.expires": str(int(now - 10))})  # no crucible.owner: never touched
    held = FileLock(live_lock)
    held.acquire()
    try:
        s = await Runner(tmp_path).run(spec_of(Ok(), "r"))
    finally:
        held.release()
    assert s.scored == 1 and f.checked >= 2  # the listing and the check afterwards both ran with the run's lease held
    assert f.containers() == ["crucible-sbx-foreign-new", "crucible-sbx-live", "plain-rlm-kernel"]
    (ev,) = events_of(tmp_path, kind="sweep")
    assert {r["name"]: r["why"] for r in ev["removed"]} == {"crucible-sbx-own": "own_run", "crucible-sbx-dead": "lease_free", "crucible-sbx-foreign-old": "expired"}
    assert ev["failed"] == [] and ev["errors"] == {} and ev["kept"] == 2 and ev["hosts"] == [""]
    assert not (tmp_path / "live" / "run.lock").read_text()  # and no lock file was written or changed by the probe
    # the sweep ran before the run planned anything
    kinds = [e["event"] for e in events_of(tmp_path)]
    assert kinds.index("sweep") < kinds.index("planned")


async def test_a_resume_sweeps_the_docker_hosts_the_run_used(tmp_path, fake):
    spec = spec_of(UsesLab(), "r")
    await Runner(tmp_path).run(spec)
    assert [e["host"] for e in events_of(tmp_path, kind="sandbox_host")] == ["ssh://lab"]
    with Runner(tmp_path)._open(spec) as j:  # the journal of that run (it takes the lease the first call released)
        assert j.sandbox_hosts() == {"ssh://lab"}
    lock = str((tmp_path / "r" / "run.lock").resolve())
    fake.add("ssh://lab", "crucible-sbx-orphan", labels(lease=lock, run="r", expires=time.time() + 3600))
    calls = len(fake.calls)
    s = await Runner(tmp_path).run(spec)  # a resume: nothing is owed, but the start sweeps what the killed run left on the lab host
    assert s.terminal == 1
    assert any(h == "ssh://lab" and a[1] == "ps" for h, a in fake.calls[calls:]) and fake.containers("ssh://lab") == []
    ev = events_of(tmp_path, kind="sweep")[-1]
    assert ev["removed"] == [{"name": "crucible-sbx-orphan", "host": "ssh://lab", "why": "own_run"}] and set(ev["hosts"]) == {"", "ssh://lab"}
    assert len(events_of(tmp_path, kind="sandbox_host")) == 1  # journaled once per host per run, not again by the resume


class NamesHosts:
    """A task adapter that says which docker hosts it uses (``docker_hosts()``: ``HarborDirTask``, ``HarborInPlaceTask``)."""

    id = "t"

    def __init__(self, hosts) -> None:
        self.hosts = hosts

    def identity_config(self) -> dict:
        return {}

    def docker_hosts(self):
        if isinstance(self.hosts, Exception):
            raise self.hosts
        return self.hosts


def test_plan_hosts_collects_what_the_adapters_say_and_ignores_the_rest():
    spec = SimpleNamespace(tasks=[NamesHosts({"ssh://a", ""}), SimpleNamespace(id="x"), NamesHosts(RuntimeError("no")), NamesHosts(["ssh://b"])])
    assert Runner._plan_hosts(spec) == {"ssh://a", "ssh://b"}  # '' (the local daemon) is always swept anyway; a raising adapter names none


async def test_a_fresh_runs_start_sweep_reaches_a_lab_host_the_plan_names(tmp_path, fake):
    """Review round 1: a runner killed mid-grade on the lab host leaves its grade container there. The next run has a new run id, so its
    journal names no host: before this, only a resume of the killed run ever swept that host."""
    fake.add("ssh://lab", "crucible-hgrade-orphan", labels(owner="other:1:1", expires=time.time() - 60))
    fake.add("ssh://other", "crucible-hgrade-elsewhere", labels(owner="other:1:1", expires=time.time() - 60))  # a host nobody names: not visited
    spec = RunSpec(run_id="fresh", tasks=[NamesHosts({"ssh://lab"})], policies=[SimpleNamespace(id="p", estimate_usd=0.0)], harnesses=[Ok()], k=1, breaker=None)
    s = await Runner(tmp_path).run(spec)
    assert s.scored == 1 and fake.containers("ssh://lab") == [] and fake.containers("ssh://other") == ["crucible-hgrade-elsewhere"]
    (ev,) = events_of(tmp_path, "fresh", kind="sweep")
    assert set(ev["hosts"]) == {"", "ssh://lab"}
    assert ev["removed"] == [{"name": "crucible-hgrade-orphan", "host": "ssh://lab", "why": "expired"}]
    assert not events_of(tmp_path, "fresh", kind="sandbox_host")  # the run itself used no host: it came from the plan


def test_the_harbor_adapters_name_their_docker_hosts(tmp_path):
    from crucible.interop.harbor.inplace import HarborInPlaceTask
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.gpu import GpuHost

    td = tmp_path / "dirtask"
    (td / "tests").mkdir(parents=True)
    (td / "task.toml").write_text('[task]\nname = "t"\n')
    (td / "tests" / "test.sh").write_text("exit 0\n")
    assert HarborDirTask(td, image="i", host=GpuHost(name="lab", docker_host="ssh://lab")).docker_hosts() == {"ssh://lab"}
    assert HarborDirTask(td, image="i").docker_hosts() == set()  # the local daemon: always swept

    def inplace(**bp) -> HarborInPlaceTask:
        t = object.__new__(HarborInPlaceTask)
        t.backend_params = {"environment_type": "docker", **bp}
        return t

    assert inplace(docker_host="ssh://lab").docker_hosts() == {"ssh://lab"}
    assert inplace().docker_hosts() == set()
    assert inplace(docker_host="ssh://lab", environment_type="modal").docker_hosts() == set()  # not a docker daemon
    assert inplace(gpu_host="no-such-host-in-any-crucible-toml").docker_hosts() == set()  # cannot be resolved: names none


async def test_a_regrade_is_a_start_too(tmp_path, fake):
    spec = spec_of(Ok(), "r")
    await Runner(tmp_path).run(spec)
    fake.add("", "crucible-sbx-orphan", labels(lease=str((tmp_path / "r" / "run.lock").resolve()), run="r", expires=time.time() + 3600))
    await Runner(tmp_path).regrade(spec)
    assert fake.containers() == [] and len(events_of(tmp_path, kind="sweep")) == 2


async def test_no_docker_means_no_sweep_and_no_event(tmp_path, fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "docker_available", lambda: False)
    s = await Runner(tmp_path).run(spec_of(Ok(), "r"))
    assert s.scored == 1 and events_of(tmp_path, kind="sweep") == [] and fake.calls == []
    # and with the sweep off in the RunSpec, a docker CLI on PATH changes nothing either
    monkeypatch.setattr(lifecycle, "docker_available", lambda: True)
    s2 = await Runner(tmp_path).run(spec_of(Ok(), "r2", hygiene=Hygiene(sweep=False)))
    assert s2.scored == 1 and events_of(tmp_path, "r2", kind="sweep") == [] and fake.verbs("ps") == []


async def test_a_failed_sweep_never_fails_the_run(tmp_path, monkeypatch):
    f = install(monkeypatch, DownDocker(down={""}))
    s = await Runner(tmp_path).run(spec_of(Ok(), "r"))
    assert s.terminal == 1 and s.scored == 1
    (ev,) = events_of(tmp_path, kind="sweep")
    assert "Cannot connect" in ev["errors"][""] and ev["removed"] == [] and f.verbs("rm") == []


async def test_a_sweep_that_raises_is_journaled_not_raised(tmp_path, monkeypatch):
    async def boom(*a, **kw):
        raise RuntimeError("the sweep itself broke")

    monkeypatch.setattr(lifecycle, "sweep", boom)
    s = await Runner(tmp_path).run(spec_of(Ok(), "r"))
    assert s.scored == 1
    (ev,) = events_of(tmp_path, kind="sweep")
    assert "the sweep itself broke" in ev["errors"]["*"]


async def test_a_sweep_never_runs_without_the_lease(tmp_path, fake):
    """A second Runner on a held run is refused (R-4) before anything else: no sweep, so no container of the live run is touched."""
    from crucible.runtime.journal import RunBusy

    spec = spec_of(Ok(), "r")
    fake.add("", "crucible-sbx-own", labels(lease=str((tmp_path / "r" / "run.lock").resolve()), run="r", expires=time.time() + 3600))
    with Runner(tmp_path)._open(spec):  # another writer holds the run
        with pytest.raises(RunBusy):
            await Runner(tmp_path).run(spec)
    assert fake.containers() == ["crucible-sbx-own"] and fake.calls == []


# ---------------------------------------------------------------- PerfGuard ignores the expired


def _probe(ps: str):
    def probe(cmd: str):
        return (0, ps) if cmd == DOCKER_PS else (0, "")

    return probe


def test_perf_guard_ignores_containers_past_their_expiry():
    now = 1_000_000.0
    ps = (f"crucible-sbx-a\t{now - 10:.0f}\t\n"       # past its expiry, not renewable: an orphan
          "crucible-sbx-b\t\t\n"                      # no label: nothing says it is dead
          f"crucible-hgrade-c\t{now + 600:.0f}\t\n"   # not yet
          "other\t\t\n")                              # not ours
    s = sample(_probe(ps), now)
    assert s["crucible_containers"] == 2
    why = PerfGuard().busy(s)
    assert any(w.startswith("2 other crucible") for w in why)
    # only the expired one: the host is quiet
    quiet = sample(_probe(f"crucible-sbx-a\t{now - 10:.0f}\t\n"), now)
    assert quiet["crucible_containers"] == 0 and PerfGuard().busy(quiet) == []


def test_a_running_renewable_keeper_past_its_label_still_counts():
    now = 1_000_000.0
    assert crucible_containers(f"crucible-sbx-live\t{now - 10:.0f}\t1\n", now) == 1  # a live holder outruns the label (R-3 grading window)
    assert crucible_containers(f"crucible-sbx-live\t{now - 10:.0f}\t\n", now) == 0
    assert crucible_containers("crucible-sbx-x\tsoon\t\ncrucible-sbx-y\n", now) == 2  # an unparseable expiry, a line with no label columns
    assert crucible_containers("crucible-sbx-z\t1\t\r\n", now) == 0  # CRLF line ends (an ssh probe from Windows) do not break the columns
    assert crucible_containers("", now) == 0


def test_the_perf_guard_probe_asks_docker_for_the_expiry_and_renewable_labels():
    assert "crucible.expires" in DOCKER_PS and "crucible.renewable" in DOCKER_PS and "grep" not in DOCKER_PS
    seen: list[str] = []

    def probe(cmd: str):
        seen.append(cmd)
        return 0, ""

    sample(probe)
    assert DOCKER_PS in seen


async def test_an_expired_orphan_does_not_hold_a_perf_slot():
    """The incident: one crashed run's orphan held gate R1 for 12 h. Now the slot opens at once."""
    ps = f"crucible-sbx-orphan\t{time.time() - 3600:.0f}\t\n"

    async def no_sleep(_s):
        raise AssertionError("the host was quiet: the guard must not wait")

    g = PerfGuard(probe=_probe(ps), wait_s=0, poll_s=0, sleep=no_sleep)
    async with g.slot("lab") as info:
        assert info["samples"][-1]["crucible_containers"] == 0
