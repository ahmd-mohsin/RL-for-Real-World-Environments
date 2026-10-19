"""M10 lab audit R-6, slice 1: the container lifecycle primitives (``crucible.world.lifecycle``) on World sandboxes and pools.

Every container Crucible starts names its owner and expiry, ends by itself (``sleep <ttl>`` or the RLM kernel's renewable life
loop), and is removed on a cancelled or failed start and on every close, recorded in a scope that reports ``{terminated, failed,
still_up, released}``. No Docker here: ``FakeDocker`` stands in for ``world.exec._run_subprocess`` and renders the exact argv
``DockerSandbox`` and ``lifecycle`` emit (value-taking flags, ``--label k=v``, ``ps --filter label=K[=V] --format``).
"""

from __future__ import annotations

import asyncio
import re
import subprocess
import time
from types import SimpleNamespace

import pytest

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world import lifecycle
from crucible.world.exec import ExecOut
from crucible.world.lifecycle import Scope
from crucible.world.pool import PooledSandbox, SandboxPool, close_pools, pool_for, signature
from crucible.world.sandbox import DockerSandbox

VALUE_FLAGS = {"--name", "--label", "--network", "--pids-limit", "--cap-add", "--cpus", "--memory", "--tmpfs", "-e", "--gpus", "--entrypoint",
               "--volumes-from", "--runtime"}
EXEC_VALUE_FLAGS = {"--user", "-u", "-w", "-e"}


def _spec() -> WorldSpec:
    return WorldSpec.model_validate({"world": "test/r6", "network": "no-network", "apps": [], "actors": [{"id": "agent", "kind": "policy"}]})


def _sbx(**kw) -> DockerSandbox:
    return DockerSandbox(SandboxSpec(image="img", **kw), _spec())


def argv_labels(a: list[str]) -> dict[str, str]:
    return dict(a[i + 1].partition("=")[::2] for i, x in enumerate(a) if x == "--label")


class FakeDocker:
    """A docker daemon per ``DOCKER_HOST`` (``''`` is the local one): containers are ``name -> {labels, cmd, image, argv}``."""

    def __init__(self) -> None:
        self.hosts: dict[str, dict[str, dict]] = {}
        self.calls: list[tuple[str, list[str]]] = []
        self.events: list[tuple[str, str]] = []  # ("created", name) / ("rm", name), in the order they happened
        self.execs: list[tuple[str, str, list[str]]] = []
        self.renewals: list[tuple[str, int]] = []
        self.run_gate: asyncio.Event | None = None  # a ``run`` waits for it before it creates its container
        self.exec_gate: asyncio.Event | None = None  # an ``exec`` waits for it
        self.fail_rm: set[str] = set()  # names whose ``rm`` answers rc 1 'daemon unreachable'
        self.dead: set[str] = set()  # names whose ``exec`` answers 'No such container' (their keeper ended)
        self.rm_delay = 0.0
        self.run_rc = 0  # rc of ``docker run`` after it created the container
        self.deadlines: dict[str, int] = {}  # name -> what ``cat /crucible-ctl/deadline`` answers (a name not here: no such file)
        self.inspect_gate: asyncio.Event | None = None  # an ``inspect`` (the image digest after ``docker run``) waits for it

    # -- inspection
    def containers(self, host: str | None = None) -> list[str]:
        if host is not None:
            return sorted(self.hosts.get(host, {}))
        return sorted(n for d in self.hosts.values() for n in d)

    def labels(self, name: str) -> dict[str, str]:
        return next(d[name]["labels"] for d in self.hosts.values() if name in d)

    def container(self, name: str) -> dict:
        return next(d[name] for d in self.hosts.values() if name in d)

    def add(self, host: str, name: str, labels: dict[str, str]) -> None:
        self.hosts.setdefault(host, {})[name] = {"labels": dict(labels), "cmd": [], "image": "img", "argv": []}

    def verbs(self, verb: str) -> list[tuple[str, list[str]]]:
        return [(h, a) for h, a in self.calls if a[1] == verb]

    # -- the daemon
    async def __call__(self, argv, *, cwd=None, env=None, timeout_s=None, stdin=None, sessions=None) -> ExecOut:
        host = (env or {}).get("DOCKER_HOST", "")
        argv = list(argv)
        self.calls.append((host, argv))
        d = self.hosts.setdefault(host, {})
        verb = argv[1]
        if verb == "run":
            i, name, labels = 2, None, {}
            while i < len(argv) and argv[i].startswith("-"):
                f = argv[i]
                if f in VALUE_FLAGS:
                    if f == "--name":
                        name = argv[i + 1]
                    elif f == "--label":
                        k, _, v = argv[i + 1].partition("=")
                        labels[k] = v
                    i += 2
                else:
                    i += 1
            if self.run_gate is not None:
                await self.run_gate.wait()
            d[name] = {"labels": labels, "cmd": argv[i + 1:], "image": argv[i], "argv": argv}
            self.events.append(("created", name))
            return ExecOut(self.run_rc, "cid\n" if self.run_rc == 0 else "", "" if self.run_rc == 0 else "docker: run failed")
        if verb == "rm":
            if self.rm_delay:
                await asyncio.sleep(self.rm_delay)
            errs = []
            for n in [a for a in argv[2:] if not a.startswith("-")]:
                self.events.append(("rm", n))
                if n in self.fail_rm:
                    errs.append("Cannot connect to the Docker daemon: daemon unreachable")
                elif n in d:
                    del d[n]
                else:
                    errs.append(f"Error response from daemon: No such container: {n}")
            return ExecOut(1 if errs else 0, "", "\n".join(errs))
        if verb == "ps":
            fmt = argv[argv.index("--format") + 1]
            want = [argv[k + 1][len("label="):] for k, x in enumerate(argv) if x == "--filter"]
            rows = []
            for n, c in d.items():
                if all((w in c["labels"]) if "=" not in w else (c["labels"].get(w.split("=", 1)[0]) == w.split("=", 1)[1]) for w in want):
                    rows.append(re.sub(r'\{\{\.Label "([^"]+)"\}\}', lambda m: c["labels"].get(m.group(1), ""), fmt.replace("{{.Names}}", n)))
            return ExecOut(0, "\n".join(rows) + ("\n" if rows else ""))
        if verb == "exec":
            i = 2
            while argv[i].startswith("-"):
                i += 2 if argv[i] in EXEC_VALUE_FLAGS else 1
            name = argv[i]
            self.execs.append((host, name, argv))
            if self.exec_gate is not None:
                await self.exec_gate.wait()
            if name not in d or name in self.dead:
                return ExecOut(1, "", f"Error response from daemon: No such container: {name}")
            if argv[-2:-1] == ["cat"] and argv[-1].endswith("/deadline"):  # the sweep's probe of a renewable container's real deadline
                if name in self.deadlines:
                    return ExecOut(0, f"{self.deadlines[name]}\n")
                return ExecOut(1, "", "cat: /crucible-ctl/deadline: No such file or directory")
            m = re.search(r"echo (\d+) >", argv[-1])
            if m:
                self.renewals.append((name, int(m.group(1))))
            return ExecOut(0, "")
        if verb in ("inspect", "image"):
            if self.inspect_gate is not None and verb == "inspect":
                await self.inspect_gate.wait()
            return ExecOut(0, "sha256:x\n")
        return ExecOut(0, "")


@pytest.fixture(autouse=True)
def fake(monkeypatch) -> FakeDocker:
    from crucible.world import pool as pool_mod
    from crucible.world import sandbox as sandbox_mod

    f = FakeDocker()
    monkeypatch.setattr(sandbox_mod, "_run_subprocess", f)
    monkeypatch.setattr(lifecycle, "_run_subprocess", f)
    monkeypatch.setattr(lifecycle, "_install_reaper", lambda: None)
    monkeypatch.setattr(lifecycle, "_HOSTS", set())
    monkeypatch.setattr(lifecycle, "RECHECK_S", 0.01)  # a raced reclaim looks twice, 2 s apart in production
    monkeypatch.setattr(pool_mod, "_POOLS", {})
    return f


async def until(cond, timeout: float = 2.0) -> None:
    end = time.monotonic() + timeout
    while not cond():
        assert time.monotonic() < end, "condition not reached"
        await asyncio.sleep(0.005)


def _attempt(**kw) -> Scope:
    return Scope(**{"run": "r", "attempt": "s-a1", "lock": "/x/run.lock", "ttl_s": 600, **kw})


# ---------------------------------------------------------------- labels and the keeper


async def test_a_sandbox_in_an_attempt_names_its_run_attempt_owner_lease_and_expiry():
    scope = _attempt(ttl_s=1000, renewable=False)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        before = int(time.time())
        a = sbx.argv()
    labels = argv_labels(a)
    assert labels["crucible.run"] == "r" and labels["crucible.attempt"] == "s-a1" and labels["crucible.lease"] == "/x/run.lock"
    assert labels["crucible.owner"] == lifecycle.owner_id() and len(labels["crucible.owner"].rsplit(":", 2)) == 3
    assert before + 1000 <= int(labels["crucible.expires"]) <= int(time.time()) + 1002
    assert labels["crucible.world"] == "test/r6" and labels["crucible.sandbox"] == "1"  # the old labels stay
    assert a[-3:] == ["img", "sleep", "1000"] and "--rm" in a and "infinity" not in a
    assert "crucible.renewable" not in labels and "--tmpfs" not in a


async def test_a_renewable_sandbox_runs_the_kernels_life_loop():
    from crucible.actors.rlm import kernel

    scope = _attempt(renewable=True)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        a = sbx.argv()
    assert a[-5:] == ["sh", "-c", lifecycle.LIFE_LOOP, "crucible-life", str(int(sbx.deadline))]
    k = a.index("--tmpfs")
    assert a[k + 1] == "/crucible-ctl:rw,size=64k,mode=0755" and k < a.index("--entrypoint")
    labels = argv_labels(a)
    assert sbx.expires == int(sbx.deadline) + lifecycle.RENEW_SLACK_S and int(labels["crucible.expires"]) == sbx.expires
    assert labels["crucible.renewable"] == "1"
    # one lifecycle design in the repository: the kernel imports what World sandboxes use
    assert kernel.LIFE_LOOP is lifecycle.LIFE_LOOP and kernel.lifetime_for_episode is lifecycle.lifetime_for
    assert kernel.CTL_DIR == lifecycle.CTL_DIR and kernel.RENEW_SLACK_S == lifecycle.RENEW_SLACK_S and kernel.DEFAULT_LIFETIME_S == lifecycle.DEFAULT_LIFETIME_S
    assert kernel._finish is lifecycle.finish


def test_outside_a_runner_a_sandbox_still_has_an_owner_and_a_bounded_renewable_life():
    """Review round 1: a fixed ``sleep 21600`` capped a live World at 6 h; outside a Runner the keeper is the life loop (the sandbox
    renews itself while it is open, ``test_outside_a_runner_a_sandbox_renews_itself_while_it_is_open``)."""
    sbx = _sbx()
    a = sbx.argv()
    labels = argv_labels(a)
    assert labels["crucible.owner"] == lifecycle.owner_id() and int(labels["crucible.expires"]) > time.time()
    assert not {"crucible.run", "crucible.attempt", "crucible.lease"} & set(labels)
    assert a[-5:] == ["sh", "-c", lifecycle.LIFE_LOOP, "crucible-life", str(int(sbx.deadline))] and "infinity" not in a
    assert labels["crucible.renewable"] == "1" and int(labels["crucible.expires"]) == int(sbx.deadline) + lifecycle.RENEW_SLACK_S


def test_lifetime_follows_the_backstop():
    assert lifecycle.lifetime_for(1800) == 2700
    assert lifecycle.lifetime_for(None) == lifecycle.DEFAULT_LIFETIME_S
    assert lifecycle.lifetime_for(10) == 910 and lifecycle.lifetime_for(10, margin_s=0) == 600  # the floor binds only below it
    assert lifecycle.lifetime_for(None, default_s=7) == 7 and lifecycle.lifetime_for(0, default_s=7) == 7


def test_owner_host_drops_the_pid_and_the_process_start():
    assert lifecycle.owner_host("h:1:2") == "h" and lifecycle.owner_host("h:1") == "h" and lifecycle.owner_host("plain") == "plain"
    assert lifecycle.owner_host(lifecycle.owner_id()) == lifecycle.owner_host(f"{lifecycle.owner_id().rsplit(':', 2)[0]}:1:1")


def test_the_pool_signature_ignores_names_labels_and_the_keeper_but_not_hardening(monkeypatch):
    """``signature`` builds its argv with a fresh ``lifecycle.Scope()``, so entering an attempt scope proves nothing (review round 1):
    the scopes it sees are patched in, populated, with a different run, attempt, lease and ttl (hence expiry) each. A control shows the
    argv of such a scope does carry every volatile label, so the equality below cannot hold vacuously."""
    sb, spec = SandboxSpec(image="img"), _spec()
    scopes = {"a": Scope(run="a", attempt="a-a1", lock="/a/run.lock", ttl_s=100, renewable=True),
              "b": Scope(run="b", attempt="b-a7", lock="/b/run.lock", ttl_s=90000, renewable=True)}
    raw = {k: DockerSandbox(sb, spec, pids_limit=64, scope=s, renewable=True).argv() for k, s in scopes.items()}
    for k, a in raw.items():
        labels = argv_labels(a)
        assert {"crucible.run", "crucible.attempt", "crucible.lease", "crucible.expires", "crucible.owner", "crucible.renewable"} <= set(labels)
        assert "--name" in a and "--tmpfs" in a and "crucible-life" in a
    assert argv_labels(raw["a"])["crucible.expires"] != argv_labels(raw["b"])["crucible.expires"]  # the expiry really differs between the scopes
    outside = signature(sb, spec, pids_limit=64, memory=None)
    seen = {}
    for k, s in scopes.items():
        with monkeypatch.context() as m:
            m.setattr(lifecycle, "Scope", lambda s=s: s)  # what ``signature`` builds its argv under
            seen[k] = signature(sb, spec, pids_limit=64, memory=None)
    assert outside == seen["a"] == seen["b"]
    flat = " ".join(str(x) for x in outside[1])
    for gone in ("crucible.owner", "crucible.run=", "crucible.attempt=", "crucible.lease=", "crucible.expires=", "crucible.renewable=",
                 "crucible.world=", "crucible-ctl", "crucible-life", "sleep", "--name"):
        assert gone not in flat, gone
    assert "img" in flat and "--pids-limit" in flat and "crucible.sandbox=1" in flat
    assert outside != signature(sb, spec, pids_limit=128, memory=None)
    assert outside != signature(sb, spec, pids_limit=64, memory="1g")


# ---------------------------------------------------------------- a start that is cancelled or fails


async def test_a_cancelled_start_removes_the_container_its_docker_run_made(fake):
    fake.run_gate = asyncio.Event()
    scope = _attempt(renewable=True)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        t = asyncio.create_task(sbx.start())
        await until(lambda: fake.verbs("run"))
        t.cancel()
        await asyncio.sleep(0.01)
        fake.run_gate.set()  # the daemon creates the container after its client was cancelled
        with pytest.raises(asyncio.CancelledError):
            await t
        assert fake.containers() == []
        i = fake.events.index(("created", sbx.name))
        assert ("rm", sbx.name) in fake.events[i + 1:]  # removed after the run made it, not before
    rep = scope.report()
    assert rep["terminated"] == [sbx.name] and rep["still_up"] == [] and rep["failed"] == []


async def test_a_late_docker_run_is_removed_when_it_finishes(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "START_GRACE_S", 0.05)
    fake.run_gate = asyncio.Event()
    async with lifecycle.attempt(_attempt(renewable=True)):
        sbx = _sbx()
        t = asyncio.create_task(sbx.start())
        await until(lambda: fake.verbs("run"))
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t  # returns after the grace: the run is still pending
        assert ("created", sbx.name) not in fake.events
        fake.run_gate.set()
        await until(lambda: ("created", sbx.name) in fake.events and ("rm", sbx.name) in fake.events[fake.events.index(("created", sbx.name)):])
        await until(lambda: fake.containers() == [])


async def test_a_failed_docker_run_still_removes_by_name(fake):
    fake.run_rc = 125  # the run created the container, then failed
    sbx = _sbx()
    with pytest.raises(RuntimeError, match="failed to start"):
        await sbx.start()
    assert fake.containers() == [] and ("rm", sbx.name) in fake.events


# ---------------------------------------------------------------- close


async def test_close_is_rc_checked_and_recorded_and_the_scope_retries(fake):
    scope = _attempt(renewable=False)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        await sbx.start()
        fake.fail_rm.add(sbx.name)
        ok, err = await sbx.close()  # never raises: a lost removal must not replace a grade that was already computed
        assert not ok and "unreachable" in err
        assert [f["name"] for f in scope.failed] == [sbx.name] and sbx.name in fake.containers()
        fake.fail_rm.clear()
    rep = scope.report()
    assert sbx.name in rep["terminated"] and rep["still_up"] == [] and rep["failed"] == [] and fake.containers() == []


async def test_close_is_shielded_and_idempotent(fake):
    sbx = _sbx()
    await sbx.start()
    fake.rm_delay = 0.2
    t = asyncio.create_task(sbx.close())
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert fake.containers() == []  # the removal ran to its end
    rms = len(fake.verbs("rm"))
    assert await sbx.close() == (True, "")  # a second close returns at once
    assert len(fake.verbs("rm")) == rms


async def test_concurrent_closes_share_one_removal(fake):
    sbx = _sbx()
    await sbx.start()
    fake.rm_delay = 0.05
    assert await asyncio.gather(sbx.close(), sbx.close()) == [(True, ""), (True, "")]
    assert len(fake.verbs("rm")) == 1


async def test_the_attempt_scope_removes_what_it_started_and_lists_what_survives(fake):
    scope = _attempt(renewable=False)
    async with lifecycle.attempt(scope):
        a, b = _sbx(), _sbx()
        await a.start()
        await b.start()
        await a.close()
        mine = {"crucible.attempt": "s-a1", "crucible.lease": "/x/run.lock"}
        fake.add("", "crucible-sbx-c", mine)  # the daemon race: a run finished after its client was gone
        fake.add("", "crucible-sbx-d", mine)
        fake.fail_rm.add("crucible-sbx-d")
        fake.add("", "crucible-sbx-other", {"crucible.attempt": "s-a2", "crucible.lease": "/x/run.lock"})  # another attempt's: never touched
    rep = scope.report()
    assert fake.containers() == ["crucible-sbx-d", "crucible-sbx-other"]
    assert {a.name, b.name, "crucible-sbx-c"} <= set(rep["terminated"])
    assert rep["still_up"] == ["crucible-sbx-d"] and [f.get("name") for f in rep["failed"]] == ["crucible-sbx-d"]


async def test_an_attempt_reclaim_never_touches_another_runs_container_that_shares_its_attempt_id(fake):
    """Review round 1: the attempt id is ``<sample key>-a<N>`` and a sample key does not hash the run, so two runs of one plan on one
    daemon (the lab host) share ids. The reclaim selects by attempt *and* lease: the other run's live sandbox survives and is not
    listed in this attempt's ``still_up``."""
    other = {"crucible.attempt": "s-a1", "crucible.lease": "/other/run.lock", "crucible.run": "other"}
    scope = _attempt(run="r", lock="/x/run.lock")
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        await sbx.start()
        fake.add("", "crucible-sbx-theirs", other)  # run B's, live, with run A's attempt id
        fake.add("", "crucible-hgrade-theirs", other)
        fake.add("", "crucible-sbx-late", {"crucible.attempt": "s-a1", "crucible.lease": "/x/run.lock"})  # a late docker run of A's own
        fake.fail_rm.add("crucible-sbx-late")  # still up: it is listed, theirs are not
    rep = scope.report()
    assert fake.containers() == ["crucible-hgrade-theirs", "crucible-sbx-late", "crucible-sbx-theirs"]
    assert sbx.name in rep["terminated"] and not {"crucible-sbx-theirs", "crucible-hgrade-theirs"} & set(rep["terminated"] + rep["still_up"])
    assert rep["still_up"] == ["crucible-sbx-late"]
    assert all(n not in a for _, a in fake.verbs("rm") for n in ("crucible-sbx-theirs", "crucible-hgrade-theirs"))
    # both listings (the reclaim's and the still_up one) carried both filters: docker ANDs repeated ``--filter label=``
    lists = [a for _, a in fake.verbs("ps")]
    assert lists and all(a.count("--filter") == 2 and "label=crucible.attempt=s-a1" in a and "label=crucible.lease=/x/run.lock" in a for a in lists)


async def test_a_scope_without_a_lease_selects_by_its_run(fake):
    scope = Scope(run="r", attempt="s-a1", ttl_s=600)
    assert scope.attempt_labels() == ["crucible.attempt=s-a1", "crucible.run=r"]
    assert Scope(run="r", attempt="s-a1", lock="/x/run.lock").attempt_labels() == ["crucible.attempt=s-a1", "crucible.lease=/x/run.lock"]
    assert Scope(attempt="s-a1").attempt_labels() == ["crucible.attempt=s-a1"]
    async with lifecycle.attempt(scope):
        await _sbx().start()  # the scope reclaims on the hosts it used
        fake.add("", "crucible-sbx-mine", {"crucible.attempt": "s-a1", "crucible.run": "r"})
        fake.add("", "crucible-sbx-theirs", {"crucible.attempt": "s-a1", "crucible.run": "other"})
    assert fake.containers() == ["crucible-sbx-theirs"]


async def test_started_journals_each_docker_host_once_per_run(fake):
    events: list[tuple] = []
    hosts: set[str] = set()
    scope = _attempt(event=lambda t, **kw: events.append((t, kw)), hosts=hosts)
    async with lifecycle.attempt(scope):
        for sb in (SandboxSpec(image="img"), SandboxSpec(image="img"), SandboxSpec(image="img", docker_host="ssh://lab")):
            await DockerSandbox(sb, _spec()).start()
    assert events == [("sandbox_host", {"host": ""}), ("sandbox_host", {"host": "ssh://lab"})] and hosts == {"", "ssh://lab"}
    assert fake.containers() == []  # the attempt's reclaim removed all three, on both daemons


async def test_the_module_default_scope_records_nothing(fake):
    sbx = _sbx()
    await sbx.start()
    await sbx.close()
    d = lifecycle._DEFAULT
    assert d.terminated == [] and d.failed == [] and d._live == {} and len(d.worlds) == 0
    d.track_world(SimpleNamespace(spec=SimpleNamespace(world="w")))  # no attempt: never holds a World
    assert len(d.worlds) == 0


# ---------------------------------------------------------------- renewal


async def test_a_renewable_scope_renews_its_containers_and_stops_with_the_attempt(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "RENEW_EVERY_S", 0.01)
    monkeypatch.setattr(lifecycle, "RENEW_BEFORE_S", 10**6)
    scope = _attempt(ttl_s=40, renewable=True)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        await sbx.start()
        sbx.deadline = scope._live[sbx.name].deadline = time.time() + 1  # nearly out of life: the renewer pushes it to now + ttl
        orig = sbx.deadline
        await until(lambda: fake.renewals)
        name, n = fake.renewals[0]
        assert name == sbx.name and n > orig and n >= int(time.time()) + 39 - 1
        _, _, argv = fake.execs[0]
        assert argv[:7] == ["docker", "exec", "--user", "0", sbx.name, "sh", "-c"] and argv[7] == f"echo {n} > /crucible-ctl/deadline"
        await until(lambda: sbx.deadline >= n)  # the sandbox moves its own deadline once the exec succeeded
    seen = len(fake.execs)
    await asyncio.sleep(0.05)
    assert len(fake.execs) == seen and scope._renewer is None  # no exec after the attempt


async def test_a_fixed_keeper_is_never_renewed(fake):
    sbx = DockerSandbox(SandboxSpec(image="img"), _spec(), scope=_attempt(renewable=False))
    assert await sbx.renew() is True and fake.execs == []


async def test_a_borrowed_pooled_container_is_renewed_by_the_attempt_and_never_removed(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "RENEW_EVERY_S", 0.01)
    monkeypatch.setattr(lifecycle, "RENEW_BEFORE_S", 10**6)
    asked: list[float] = []

    async def renew(lifetime_s: float) -> bool:
        asked.append(lifetime_s)
        return True

    scope = _attempt(ttl_s=40, renewable=True)
    async with lifecycle.attempt(scope):
        scope.borrowed("crucible-sbx-pooled", None, deadline=time.time() + 1, renew=renew)
        await until(lambda: asked)
        assert asked[0] == 40
    assert scope.report()["released"] == ["crucible-sbx-pooled"] and fake.verbs("rm") == []  # the pool, not the attempt, ends it


async def test_a_failed_renewal_is_journaled_once(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "RENEW_EVERY_S", 0.01)
    monkeypatch.setattr(lifecycle, "RENEW_BEFORE_S", 10**6)
    events: list[tuple] = []
    scope = _attempt(ttl_s=40, renewable=True, event=lambda t, **kw: events.append((t, kw)))
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        await sbx.start()
        fake.dead.add(sbx.name)
        scope._live[sbx.name].deadline = time.time() + 1
        await until(lambda: len(fake.execs) >= 3)  # it keeps trying, and says so once
    assert [e for e in events if e[0] == "renew_failed"] == [("renew_failed", {"attempt": "s-a1", "name": sbx.name})]


# ---------------------------------------------------------------- pools


async def test_a_pooled_container_carries_run_labels_and_no_attempt(fake):
    scope = _attempt(run="r", lock="/x/run.lock", renewable=True)
    async with lifecycle.attempt(scope):
        pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=2)
        await pool.warm()
        names = fake.containers()
        assert len(names) == 2
        for n in names:
            labels, cmd = fake.labels(n), fake.container(n)["cmd"]
            assert labels["crucible.run"] == "r" and labels["crucible.lease"] == "/x/run.lock" and "crucible.attempt" not in labels
            assert labels["crucible.owner"] == lifecycle.owner_id() and labels["crucible.renewable"] == "1"
            assert cmd[:4] == ["sh", "-c", lifecycle.LIFE_LOOP, "crucible-life"]
            assert int(labels["crucible.expires"]) >= time.time() + 2 * 600  # kept twice the ttl, plus the renewal slack
    assert fake.containers() == names and scope.report()["terminated"] == []  # the attempt's exit leaves the run's containers alone
    await pool.close()
    assert fake.containers() == []


async def test_an_acquire_renews_a_short_lived_container_and_replaces_a_dead_one(fake):
    pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=1)
    try:
        await pool.warm()
        s = pool._ready.get_nowait()
        s.deadline = time.time() + 1  # less than the acquiring scope's ttl: it needs a renewal
        pool._ready.put_nowait(s)
        fake.dead.add(s.name)  # ... and its keeper ended: the renewal exec finds no container
        new = await pool.acquire()
        assert new is not s and pool.stats["dead"] == 1 and pool.stats["misses"] == 1
        assert s not in pool._handed and new in pool._handed and s.name not in fake.containers()
        assert fake.execs and fake.execs[0][1] == s.name
    finally:
        await pool.close()


async def test_an_acquire_that_renews_successfully_hands_out_the_same_container(fake):
    pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=1)
    try:
        await pool.warm()
        s = pool._ready.get_nowait()
        s.deadline = time.time() + 1
        pool._ready.put_nowait(s)
        got = await pool.acquire()
        assert got is s and pool.stats["hits"] == 1 and pool.stats["dead"] == 0
        assert fake.renewals and fake.renewals[0][0] == s.name and s.deadline > time.time() + 3600
    finally:
        await pool.close()


async def test_a_cancelled_acquire_leaves_no_container(fake):
    pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=1)
    try:
        await pool.warm()
        s = pool._ready.get_nowait()
        s.deadline = time.time() + 1
        pool._ready.put_nowait(s)
        fake.exec_gate = asyncio.Event()  # the renewal exec blocks
        t = asyncio.create_task(pool.acquire())
        await until(lambda: fake.execs)
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert s.name not in fake.containers() and s not in pool._handed and pool._ready.empty()
    finally:
        await pool.close()


async def test_close_pools_closes_one_runs_pools_and_their_handed_out_containers(fake):
    from crucible.world import pool as pool_mod

    handed, pools = {}, {}
    for run in ("a", "b"):
        async with lifecycle.attempt(Scope(run=run, attempt=f"{run}-a1", lock=f"/{run}/run.lock", ttl_s=600, renewable=True)):
            pools[run] = pool_for(SandboxSpec(image="img", pool=1), _spec())
            await pools[run].warm()
            handed[run] = PooledSandbox(pools[run])
            await handed[run].start()
    assert {k[0] for k in pool_mod._POOLS} == {"/a/run.lock", "/b/run.lock"}  # keyed by lease (review round 1), not by run id
    runs = lambda: {fake.labels(n)["crucible.run"] for n in fake.containers()}  # noqa: E731
    assert runs() == {"a", "b"}
    rep = await close_pools("/a/run.lock")
    assert rep["failed"] == [] and len(rep["terminated"]) >= 1 and rep["unfinished"] == 0
    assert runs() == {"b"} and {k[0] for k in pool_mod._POOLS} == {"/b/run.lock"}
    await handed["a"].close()  # a release into a closed pool destroys the container
    assert pools["a"].stats["destroyed"] == 1 and handed["a"]._sbx is None
    await handed["b"].close()
    await close_pools()
    assert fake.containers() == [] and pool_mod._POOLS == {}


async def test_a_run_id_two_roots_share_does_not_share_a_pool_or_its_close(fake):
    """Review round 1: a run id is unique only within a root; the lease is unique. Two Runners with one run id and different roots keep
    two pools, and one's ``close_pools`` leaves the other's warm and handed-out containers alone. ``run=`` still selects by run id."""
    from crucible.world import pool as pool_mod

    handed, pools = {}, {}
    for root in ("one", "two"):
        async with lifecycle.attempt(Scope(run="same", attempt="s-a1", lock=f"/{root}/same/run.lock", ttl_s=600, renewable=True)):
            pools[root] = pool_for(SandboxSpec(image="img", pool=1), _spec())
            await pools[root].warm()
            handed[root] = PooledSandbox(pools[root])
            await handed[root].start()
    assert pools["one"] is not pools["two"] and {k[0] for k in pool_mod._POOLS} == {"/one/same/run.lock", "/two/same/run.lock"}
    await until(lambda: len(fake.containers()) == 4)  # each pool: the handed-out container and its refill
    before = set(fake.containers())
    rep = await close_pools(owner="/one/same/run.lock")
    assert rep["failed"] == [] and rep["terminated"]
    assert {k[0] for k in pool_mod._POOLS} == {"/two/same/run.lock"}
    survivors = set(fake.containers())
    assert survivors and survivors < before and handed["two"]._sbx.name in survivors  # run two's handed-out container is untouched
    assert {fake.labels(n)["crucible.lease"] for n in survivors} == {"/two/same/run.lock"}
    rep = await close_pools(run="same")  # the run-id alias: every pool of that run id
    assert pool_mod._POOLS == {} and fake.containers() == [] and rep["failed"] == []
    await handed["one"].close()
    await handed["two"].close()


async def test_a_pool_without_a_lease_is_keyed_by_its_run_id(fake):
    from crucible.world import pool as pool_mod

    async with lifecycle.attempt(Scope(run="r", attempt="s-a1", ttl_s=600, renewable=True)):
        pool_for(SandboxSpec(image="img", pool=1), _spec())
    assert {k[0] for k in pool_mod._POOLS} == {"r"}
    await close_pools("r")
    assert pool_mod._POOLS == {}


async def test_the_pool_belongs_to_the_run_not_to_the_attempt_that_created_it(fake):
    scope = _attempt(run="r", attempt="r-a1", renewable=True)
    async with lifecycle.attempt(scope):
        pool = pool_for(SandboxSpec(image="img", pool=1), _spec())
        ps = PooledSandbox(pool)
        await ps.start()
        name = ps._sbx.name
        assert scope._live[name].owned is False and pool.scope.attempt is None and pool.scope.run == "r"
        await ps.close()
    assert scope.report()["terminated"] == [name]  # destroyed on release (fresh mode): the attempt's record says so
    await until(lambda: len(fake.containers()) == 1)  # the pool's refilled warm container: the attempt's reclaim never touched it
    assert fake.labels(fake.containers()[0])["crucible.run"] == "r"
    await close_pools(owner="/x/run.lock")
    assert fake.containers() == []


async def test_a_pool_close_does_not_wait_for_a_refill_stuck_in_docker_run(fake, monkeypatch):
    """Review round 1: ``close`` waited for every refill with no bound, so a slow daemon held ``Runner.run``'s return (and the run's
    lease) for up to a ``docker run``'s 30 min. Now it waits ``POOL_CLOSE_WAIT_S``, reports ``unfinished``, and the refill removes its
    own container when its ``docker run`` returns (the pool is closed by then)."""
    monkeypatch.setattr(lifecycle, "POOL_CLOSE_WAIT_S", 0.05)
    pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=1)
    fake.run_gate = asyncio.Event()
    pool._refill()
    await until(lambda: fake.verbs("run"))
    t0 = time.monotonic()
    rep = await pool.close()
    assert time.monotonic() - t0 < 1.5 and rep["unfinished"] == 1 and rep["failed"] == []
    fake.run_gate.set()  # the daemon answers late
    await until(lambda: not pool._tasks and fake.containers() == [])
    assert any(e[0] == "rm" for e in fake.events)  # removed by the refill, not left to the keeper's ttl


async def test_close_pools_reports_the_unfinished_refills_of_every_pool(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "POOL_CLOSE_WAIT_S", 0.05)
    async with lifecycle.attempt(_attempt(renewable=True)):
        pool = pool_for(SandboxSpec(image="img", pool=1), _spec())
        fake.run_gate = asyncio.Event()
        pool._refill()
        await until(lambda: fake.verbs("run"))
        rep = await close_pools(owner="/x/run.lock")
        assert rep["unfinished"] == 1 and rep["terminated"] == [] and rep["failed"] == []
        fake.run_gate.set()
        await until(lambda: not pool._tasks and fake.containers() == [])


# ---------------------------------------------------------------- a start that is cancelled after docker run, and a sandbox outside a Runner


async def test_a_cancel_during_the_image_inspect_removes_the_container(fake):
    """Review round 1: the image digest is read after ``docker run`` returned; a cancel there left the container running (and, for a
    pooled start, in neither the ready queue nor the handed-out set)."""
    fake.inspect_gate = asyncio.Event()
    scope = _attempt(renewable=True)
    async with lifecycle.attempt(scope):
        sbx = _sbx()
        t = asyncio.create_task(sbx.start())
        await until(lambda: any(a[1] == "inspect" for _, a in fake.calls))
        assert fake.containers() == [sbx.name]
        t.cancel()
        with pytest.raises(asyncio.CancelledError):
            await t
        assert fake.containers() == []
    assert scope.report()["terminated"] == [sbx.name] and scope.report()["still_up"] == []


async def test_a_pooled_cold_start_cancelled_in_the_image_inspect_leaves_no_container(fake):
    fake.inspect_gate = asyncio.Event()
    pool = SandboxPool(SandboxSpec(image="img"), _spec(), size=1)
    t = asyncio.create_task(pool.acquire())
    await until(lambda: any(a[1] == "inspect" for _, a in fake.calls))
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert fake.containers() == [] and not pool._handed and pool._ready.empty()
    await pool.close()


async def test_outside_a_runner_a_sandbox_renews_itself_while_it_is_open(fake, monkeypatch):
    """Review round 1: ``sleep 21600`` ended a World held open for more than 6 h with no Runner (an idle OpenEnv session, a long
    episode). The sandbox renews its own life while open, and stops with ``close``."""
    monkeypatch.setattr(lifecycle, "RENEW_EVERY_S", 0.01)
    sbx = _sbx()
    await sbx.start()
    r = sbx._renewer
    assert r is not None and not r.done()
    assert fake.execs == []  # plenty of life left: no exec
    sbx.deadline = time.time() + 1  # nearly out of life: the next tick pushes it to now + the default ttl
    await until(lambda: fake.renewals)
    name, n = fake.renewals[0]
    assert name == sbx.name and n >= int(time.time()) + lifecycle.DEFAULT_LIFETIME_S - 5
    await until(lambda: sbx.deadline >= n)
    await sbx.close()
    assert sbx._renewer is None
    await until(r.done)
    assert fake.containers() == []


async def test_a_renewer_ends_when_the_container_is_gone(fake, monkeypatch):
    monkeypatch.setattr(lifecycle, "RENEW_EVERY_S", 0.01)
    sbx = _sbx()
    await sbx.start()
    r = sbx._renewer
    fake.dead.add(sbx.name)
    sbx.deadline = time.time() + 1
    await until(r.done)  # the renewal found no container: the keep-alive ends, quietly
    await sbx.close()


async def test_a_sandbox_a_scope_records_gets_no_renewer_of_its_own(fake):
    """Under a Runner attempt the scope renews (``Scope._renew_loop``); a second renewer per sandbox would double every exec."""
    async with lifecycle.attempt(_attempt(renewable=True)):
        sbx = _sbx()
        await sbx.start()
        assert sbx._renewer is None


# ---------------------------------------------------------------- exit reaper


def test_the_exit_reaper_removes_this_processes_containers(monkeypatch):
    calls: list[tuple[list[str], str | None]] = []

    def run(argv, **kw):
        calls.append((list(argv), (kw.get("env") or {}).get("DOCKER_HOST")))
        return SimpleNamespace(stdout="c1\nc2\n" if argv[1] == "ps" else "", returncode=0)

    monkeypatch.setattr(lifecycle, "subprocess", SimpleNamespace(run=run, SubprocessError=subprocess.SubprocessError))
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: "/usr/bin/docker")  # CI without a docker CLI
    monkeypatch.setattr(lifecycle, "_HOSTS", {"", "ssh://lab"})
    removed = lifecycle._reap_own()
    ps = [c for c in calls if c[0][1] == "ps"]
    assert [c[1] for c in ps] == [None, "ssh://lab"]
    assert all(c[0] == ["docker", "ps", "-a", "-q", "--filter", f"label=crucible.owner={lifecycle.owner_id()}"] for c in ps)
    assert [c[0] for c in calls if c[0][1] == "rm"] == [["docker", "rm", "-f", "c1", "c2"]] * 2
    assert removed == ["c1", "c2", "c1", "c2"]


def test_the_exit_reaper_does_nothing_without_a_docker_cli(monkeypatch):
    monkeypatch.setattr(lifecycle.shutil, "which", lambda name: None)
    monkeypatch.setattr(lifecycle, "_HOSTS", {""})
    assert lifecycle._reap_own() == []


# ---------------------------------------------------------------- SIGTERM (review round 1)

_REAL_INSTALL = lifecycle._install_reaper  # captured at import: the autouse fixtures replace the module attribute with a no-op


def _signals(handlers: dict[int, object], *, kills: list | None = None) -> SimpleNamespace:
    """A stand-in for the ``signal`` module: one handler table, ``signal.signal`` and ``os.kill`` recorded."""
    return SimpleNamespace(SIGTERM=15, SIG_DFL=0, SIG_IGN=1, getsignal=lambda n: handlers.get(n, 0),
                           signal=lambda n, h: (handlers.__setitem__(n, h), kills.append(("signal", n, h)) if kills is not None else None)[0])


def test_sigterm_reaps_this_processs_containers_then_takes_the_default_action(monkeypatch):
    """Python's default SIGTERM action ends the process without atexit or ``finally``: ``kill <pid>``, systemd, Slurm and Kubernetes
    left every World sandbox running until its ttl."""
    seen: list = []
    handlers: dict[int, object] = {15: lifecycle._on_sigterm}
    monkeypatch.setattr(lifecycle, "_reap_own", lambda: seen.append("reap") or [])
    monkeypatch.setattr(lifecycle, "signal", _signals(handlers, kills=seen))
    monkeypatch.setattr(lifecycle, "os", SimpleNamespace(kill=lambda pid, sig: seen.append(("kill", pid, sig)), getpid=lambda: 4242))
    monkeypatch.setitem(lifecycle._SIGTERM, "prev", None)
    lifecycle._on_sigterm(15, None)
    assert seen == ["reap", ("signal", 15, 0), ("kill", 4242, 15)]  # the default action restored, then the signal re-raised
    assert handlers[15] == 0


def test_sigterm_chains_to_the_handler_it_replaced(monkeypatch):
    seen: list = []
    monkeypatch.setattr(lifecycle, "_reap_own", lambda: seen.append("reap") or [])
    monkeypatch.setattr(lifecycle, "signal", _signals({}, kills=seen))
    monkeypatch.setattr(lifecycle, "os", SimpleNamespace(kill=lambda pid, sig: seen.append("kill"), getpid=lambda: 1))
    monkeypatch.setitem(lifecycle._SIGTERM, "prev", lambda signum, frame: seen.append(("prev", signum)))
    lifecycle._on_sigterm(15, None)
    assert seen == ["reap", ("prev", 15)]  # the application's (or the RLM kernel's) handler decides what happens next


def test_a_reap_that_raises_does_not_stop_the_signal(monkeypatch):
    seen: list = []

    def boom() -> list:
        raise RuntimeError("docker is gone")

    monkeypatch.setattr(lifecycle, "_reap_own", boom)
    monkeypatch.setitem(lifecycle._SIGTERM, "prev", lambda signum, frame: seen.append("prev"))
    lifecycle._on_sigterm(15, None)
    assert seen == ["prev"]


def test_the_reaper_installs_a_sigterm_handler_that_chains_to_one_already_there(monkeypatch):
    handlers: dict[int, object] = {}
    registered: list = []
    app = lambda signum, frame: None  # noqa: E731
    handlers[15] = app
    monkeypatch.setattr(lifecycle, "signal", _signals(handlers))
    monkeypatch.setattr(lifecycle, "atexit", SimpleNamespace(register=registered.append))
    monkeypatch.setitem(lifecycle._REAPER, "installed", False)
    monkeypatch.setitem(lifecycle._SIGTERM, "prev", None)
    _REAL_INSTALL()
    assert handlers[15] is lifecycle._on_sigterm and lifecycle._SIGTERM["prev"] is app and registered == [lifecycle._reap_own]
    _REAL_INSTALL()  # once per process
    assert registered == [lifecycle._reap_own]


def test_the_reaper_leaves_a_process_that_ignores_sigterm_alone(monkeypatch):
    handlers: dict[int, object] = {15: 1}  # SIG_IGN
    monkeypatch.setattr(lifecycle, "signal", _signals(handlers))
    monkeypatch.setattr(lifecycle, "atexit", SimpleNamespace(register=lambda f: None))
    monkeypatch.setitem(lifecycle._REAPER, "installed", False)
    _REAL_INSTALL()
    assert handlers[15] == 1


def test_the_reaper_never_touches_signals_off_the_main_thread(monkeypatch):
    import threading

    handlers: dict[int, object] = {}
    monkeypatch.setattr(lifecycle, "signal", _signals(handlers))
    monkeypatch.setattr(lifecycle, "atexit", SimpleNamespace(register=lambda f: None))
    monkeypatch.setitem(lifecycle._REAPER, "installed", False)
    t = threading.Thread(target=_REAL_INSTALL)
    t.start()
    t.join()
    assert handlers == {}  # ``signal.signal`` would raise ValueError there: atexit alone covers a worker thread's first start


def test_the_kernels_sigterm_handler_chains_to_the_lifecycles_and_the_other_way_round(monkeypatch):
    """Neither reaper pre-empts the other: the kernel's used to install only over ``SIG_DFL`` (so a World sandbox's handler would have
    hidden it) and ignored any handler already there."""
    import atexit

    from crucible.actors.rlm import kernel

    seen: list = []
    monkeypatch.setattr(kernel, "reap_own", lambda: seen.append("kernel") or [])
    monkeypatch.setitem(kernel._REAPER, "prev_sigterm", lambda signum, frame: seen.append("lifecycle"))
    kernel._on_sigterm(15, None)
    assert seen == ["kernel", "lifecycle"]

    handlers: dict[int, object] = {15: lifecycle._on_sigterm}  # a World sandbox started first
    monkeypatch.setattr(kernel, "signal", _signals(handlers))
    monkeypatch.setattr(atexit, "register", lambda f, *a, **k: None)
    monkeypatch.setitem(kernel._REAPER, "installed", False)
    monkeypatch.setitem(kernel._REAPER, "prev_sigterm", None)
    kernel._install_reaper()
    assert handlers[15] is kernel._on_sigterm and kernel._REAPER["prev_sigterm"] is lifecycle._on_sigterm
