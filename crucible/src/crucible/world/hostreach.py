"""Remote hosts that stop answering: fail fast, pause the episode, then resume it or mask it (2026-10-26).

A remote World runs on a remote host (e.g. the lab RTX 5090) over SSH: ``container_exec`` routes every App into a
container there (``DockerExec`` through the shared tunnel ``world.docker_tunnel``), and a remote grader runs there too.
When this machine lost its network (2026-10-26, on the since-removed GPU kernel lane), every in-flight episode's tool
calls hung until the 9,000 s attempt clock, and all 8 episodes of a measurement were masked as infrastructure (correct per V8) after
$29.5 of model sampling. Here every remote operation runs under its host's ``HostReach``:

- **Fail fast.** ``guard``: an operation still running after ``heartbeat_s`` has the host probed every ``heartbeat_s``, and
  ``misses`` failed probes in a row cut it with the typed ``exec.HostUnreachable`` (``ReachRule.detect_s``: 50 s with the
  defaults). A transport failure the operation reports itself (the Docker CLI cannot reach the daemon, the tunnel does not
  come up, ssh exits 255) is confirmed by one probe (``confirm``): a failed command is never read as an outage from its
  words alone. A probe is bounded (``probe_timeout_s``), one probe per host runs at a time for every caller, and its answer
  is reused for ``fresh_s``.
- **Pause, then retry.** ``outage`` holds every clock registered in the caller's context (``clock``: the Runner's attempt
  clock, the episode's agent clock, the World's tool deadline, a grade's scoring budget) and probes again with a doubling
  backoff, within the attempt's pause budget (``episode``; ``pause_s``, 30 min in all) and never past ``pause_s`` after the
  host went down, so a later attempt does not wait again for an outage that has already outlasted the budget.
- **Resume or mask.** ``call``: when the host is back, the operation runs again if that is safe: nothing reached the host,
  or the operation is ``replayable`` (every framework operation, and a read-only tool). A non-replayable tool call that was
  cut mid-way comes back to the World as ``HostUnreachable(resumed=True)``; the agent sees an interrupted call and the
  episode goes on. A model call already made is never sampled again. A host that does not come back raises
  ``HostUnreachable``: the World records ``transport_unreachable`` with detail ``host_unreachable`` and masks the episode.
- **Fleet stop.** ``down`` and ``wait_up``: a launcher opens no new episode while the host is known to be down.
- **Diagnostics.** Every outage of an attempt is a row of its ledger (``episode``; the Runner writes it to the record's
  ``meta.host_outages`` and to the run journal), and every down, outage and up event goes to the ``listen`` subscribers (a
  swarm journal's ``hosts.jsonl``).

Hosts are keyed by their transport (``for_docker_host``: a ``DOCKER_HOST``; ``for_ssh``: an SSH target) and shared by the
process. Imports ``crucible.world.exec`` (the error type and the subprocess path) and the stdlib only.
"""

from __future__ import annotations

import asyncio
import contextlib
import contextvars
import time
from collections.abc import Awaitable, Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeVar

from crucible.world.exec import HOST_UNREACHABLE, HostUnreachable

T = TypeVar("T")
# what the agent reads when a tool call it made was cut by an outage and the host came back (framework_health reads it too)
INTERRUPTED = "the connection to the environment's host dropped while it ran"


@dataclass(frozen=True)
class ReachRule:
    """How fast an outage is found and how long an episode waits for the host (operational: when a run pauses, never what
    it measures)."""

    probe_timeout_s: float = 10.0  # one probe's wall clock
    heartbeat_s: float = 15.0  # an operation running this long has the host probed, and again every heartbeat
    misses: int = 2  # failed probes in a row that cut a running operation
    fresh_s: float = 5.0  # a probe's answer is reused this long
    pause_s: float = 1800.0  # the longest an attempt pauses in all, and the longest anyone waits after the host went down
    backoff_s: float = 5.0  # the first wait between probes while paused; doubles
    backoff_cap_s: float = 120.0

    @property
    def detect_s(self) -> float:
        """The longest a hung operation runs before it is cut: the first heartbeat, then ``misses`` probes that each run
        out their timeout, with a heartbeat between them."""
        return self.heartbeat_s + self.misses * self.probe_timeout_s + (self.misses - 1) * self.heartbeat_s


# ---- the clocks an outage holds ------------------------------------------------------------------------------------------------
_CLOCKS: contextvars.ContextVar[tuple[Any, ...]] = contextvars.ContextVar("crucible_pausable_clocks", default=())
_HOLDS: dict[int, list[Any]] = {}  # id(clock) -> [clock, holders, seconds it had left]: concurrent holds of one clock


@contextlib.contextmanager
def clock(cm: Any) -> Iterator[None]:
    """While the block runs, an outage in this context holds ``cm`` (an entered ``asyncio.Timeout``). Tasks started inside
    the block see it too (they copy the context)."""
    if cm is None:
        yield
        return
    tok = _CLOCKS.set((*_CLOCKS.get(), cm))
    try:
        yield
    finally:
        _CLOCKS.reset(tok)


@contextlib.contextmanager
def held() -> Iterator[None]:
    """Every clock of this context stands still while the block runs: its deadline is lifted at entry and set at exit to
    what it had left then. Holds of one clock by several outages at once count: the last to end sets it again. A clock
    that has no deadline, has fired, or is not entered (or no longer) is left alone."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        yield
        return
    mine: list[list[Any]] = []
    for cm in _CLOCKS.get():
        h = _HOLDS.get(id(cm))
        if h is not None and h[0] is cm:
            h[1] += 1
            mine.append(h)
            continue
        try:
            when = cm.when()
            if when is None or cm.expired():
                continue
            cm.reschedule(None)
        except (RuntimeError, AttributeError):
            continue
        h = [cm, 1, max(0.0, when - loop.time())]
        _HOLDS[id(cm)] = h
        mine.append(h)
    try:
        yield
    finally:
        now = loop.time()
        for h in mine:
            h[1] -= 1
            if h[1] == 0:
                _HOLDS.pop(id(h[0]), None)
                with contextlib.suppress(RuntimeError):
                    h[0].reschedule(now + h[2])


# ---- what may run again --------------------------------------------------------------------------------------------------------
_REPLAY: contextvars.ContextVar[bool] = contextvars.ContextVar("crucible_remote_replayable", default=True)


@contextlib.contextmanager
def replayable(ok: bool) -> Iterator[None]:
    """Whether an operation cut mid-way may run again once the host is back. True outside (every framework operation:
    snapshots, state reads, grades, the bundle and template builds are the same when repeated); the World sets it per
    policy tool call, True only for a read-only tool."""
    tok = _REPLAY.set(bool(ok))
    try:
        yield
    finally:
        _REPLAY.reset(tok)


# ---- one attempt's outages --------------------------------------------------------------------------------------------------------
@dataclass
class Outages:
    """One attempt's outages (``episode``): what the record's ``meta.host_outages`` holds. ``budget_s``: the most this
    attempt pauses in all (None: the host's ``ReachRule.pause_s``)."""

    budget_s: float | None = None
    paused_s: float = 0.0
    rows: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {"reason": HOST_UNREACHABLE, "paused_s": round(self.paused_s, 3), "outages": len(self.rows), "resumed": sum(1 for r in self.rows if r["resumed"]),
                "gave_up": sum(1 for r in self.rows if not r["resumed"]), "hosts": sorted({str(r["host"]) for r in self.rows}), "rows": self.rows[:50]}


_LEDGER: contextvars.ContextVar[Outages | None] = contextvars.ContextVar("crucible_host_outages", default=None)


@contextlib.contextmanager
def episode(ledger: Outages | None = None) -> Iterator[Outages]:
    """The attempt's ledger for every outage inside the block (the Runner opens one per attempt)."""
    led = ledger if ledger is not None else Outages()
    tok = _LEDGER.set(led)
    try:
        yield led
    finally:
        _LEDGER.reset(tok)


def current() -> Outages | None:
    return _LEDGER.get()


# ---- subscribers ------------------------------------------------------------------------------------------------------------------
_LISTENERS: list[Callable[[dict[str, Any]], Any]] = []


def listen(fn: Callable[[dict[str, Any]], Any]) -> Callable[[], None]:
    """``fn(row)`` for every ``host_down``, ``outage`` and ``host_up`` event of any host of this process; returns the
    unsubscribe. A subscriber that raises is skipped."""
    _LISTENERS.append(fn)

    def stop() -> None:
        with contextlib.suppress(ValueError):
            _LISTENERS.remove(fn)

    return stop


def _emit(row: dict[str, Any]) -> None:
    for fn in _LISTENERS.copy():  # a subscriber may unsubscribe while it runs
        try:
            fn(dict(row))
        except Exception:  # noqa: BLE001, S112 - a journal that cannot write never ends an episode
            continue


# ---- one host ---------------------------------------------------------------------------------------------------------------------
class HostReach:
    """One remote host's reachability, shared by every operation of this process on it (module doc). ``probe()`` answers
    whether the host is reachable now; ``now`` and ``sleep`` are injectable for tests."""

    def __init__(self, name: str, probe: Callable[[], Awaitable[bool]], rule: ReachRule | None = None, *, now: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep):
        self.name, self._probe_fn, self.rule = name, probe, rule or ReachRule()
        self._now, self._sleep = now, sleep
        self.down_since: float | None = None
        self.why = ""
        self.probes = 0
        self._probing: asyncio.Future[bool] | None = None
        self._answer: tuple[float, bool] | None = None

    @property
    def down(self) -> bool:
        return self.down_since is not None

    # -- probing -----------------------------------------------------------------------------------------------------------------
    async def probe(self) -> bool:
        """Is the host reachable now? One bounded probe per host at a time (concurrent callers share it; a caller that is
        cancelled leaves it running for the others), its answer reused for ``fresh_s``. An answer marks the host up."""
        if self._answer is not None and self._now() - self._answer[0] < self.rule.fresh_s:
            return self._answer[1]
        t = self._probing
        if t is None or t.done() or t.get_loop() is not asyncio.get_running_loop():
            t = self._probing = asyncio.ensure_future(self._probe_once())
        return await asyncio.shield(t)

    async def _probe_once(self) -> bool:
        self.probes += 1
        try:
            ok = bool(await asyncio.wait_for(self._probe_fn(), self.rule.probe_timeout_s))
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 - a probe that raises or runs out its timeout is a host that does not answer
            ok = False
        self._answer = (self._now(), ok)
        if ok:
            self._mark_up()
        return ok

    def _mark_down(self, why: str) -> None:
        if self.down_since is None:
            self.down_since, self.why = self._now(), why[:300]
            _emit({"event": "host_down", "host": self.name, "why": self.why, "ts": round(time.time(), 3)})

    def _mark_up(self) -> None:
        if self.down_since is not None:
            down_s = self._now() - self.down_since
            self.down_since = None
            _emit({"event": "host_up", "host": self.name, "down_s": round(down_s, 3), "why": self.why, "ts": round(time.time(), 3)})

    async def confirm(self, why: str) -> bool:
        """An operation reported a transport failure: True, and the host marked down, when it does not answer a probe
        either; False when it does (the failure was the operation's own)."""
        if await self.probe():
            return False
        self._mark_down(why)
        return True

    # -- one operation ------------------------------------------------------------------------------------------------------------
    async def guard(self, op: Callable[[], Awaitable[T]], *, what: str = "") -> T:
        """``op()``, cut with ``HostUnreachable(dispatched=True)`` once it has run ``heartbeat_s`` and the host then fails
        ``misses`` probes in a row (``detect_s`` at most). The cut operation is cancelled (a subprocess is killed with it)."""
        task = asyncio.ensure_future(op())
        misses = 0
        try:
            while True:
                done, _ = await asyncio.wait({task}, timeout=self.rule.heartbeat_s)
                if done:
                    return task.result()
                if await self.probe():
                    misses = 0
                    continue
                misses += 1
                if misses >= self.rule.misses:
                    why = f"{what or 'an operation'} on {self.name}: the host stopped answering ({misses} probes failed in a row)"
                    self._mark_down(why)
                    raise HostUnreachable(self.name, why, dispatched=True)
        finally:
            if not task.done():
                task.cancel()
                await asyncio.wait({task})
            if not task.cancelled():
                task.exception()  # retrieved: a cut operation's own error is not reported as never retrieved

    def _left(self) -> float:
        """Seconds the caller may still pause: the attempt's budget less what it already paused, and never past
        ``pause_s`` after the host went down."""
        led = _LEDGER.get()
        budget = self.rule.pause_s if led is None or led.budget_s is None else led.budget_s
        left = budget - (led.paused_s if led is not None else 0.0)
        if self.down_since is not None:
            left = min(left, self.rule.pause_s - (self._now() - self.down_since))
        return max(0.0, left)

    async def outage(self, why: str, *, what: str = "") -> bool:
        """The host is down (``confirm`` or ``guard`` said so): hold this context's clocks and probe with a doubling backoff
        until the host answers (True) or the pause budget is spent (False). One row in the attempt's ledger either way."""
        self._mark_down(why)
        led = _LEDGER.get()
        left = self._left()
        t0 = self._now()
        wait = self.rule.backoff_s
        back = False
        with held():
            while True:
                if await self.probe():
                    back = True
                    break
                spent = self._now() - t0
                if spent >= left:
                    break
                await self._sleep(min(wait, left - spent))
                wait = min(wait * 2, self.rule.backoff_cap_s)
        paused = self._now() - t0
        row = {"host": self.name, "what": what, "why": why[:300], "paused_s": round(paused, 3), "resumed": back, "at": round(time.time(), 3)}
        if led is not None:
            led.paused_s += paused
            led.rows.append(row)
        _emit({"event": "outage", **row})
        return back

    def _gave_up(self, what: str, why: str, paused: float) -> str:
        return (f"{HOST_UNREACHABLE}: {what or 'an operation'} on {self.name}: the host did not answer again within the pause budget "
                f"({paused:.0f}s paused, at most {self.rule.pause_s:.0f}s): {why}")[:500]

    async def ready(self, *, what: str = "") -> float:
        """Before an operation starts: when the host is known to be down, pause for it (``outage``); ``HostUnreachable`` when
        it does not come back. Returns the seconds paused."""
        if not self.down:
            return 0.0
        led = _LEDGER.get()
        before = led.paused_s if led is not None else 0.0
        t0 = self._now()
        if not await self.outage(self.why, what=what):
            raise HostUnreachable(self.name, self._gave_up(what, self.why, self._now() - t0), dispatched=False, paused_s=self._now() - t0)
        return (led.paused_s - before) if led is not None else self._now() - t0

    async def call(self, op: Callable[[], Awaitable[T]], *, what: str = "", replay: bool | None = None) -> T:
        """The whole contract for one remote operation: wait while the host is known to be down, run ``op`` under
        ``guard``, and on a confirmed loss pause for the host. Back: ``op`` runs again when nothing reached the host or it
        is replayable (``replay``, default ``replayable()``), else ``HostUnreachable(resumed=True)``. Not back:
        ``HostUnreachable``. ``op`` raises ``HostUnreachable`` itself for a transport failure it saw and ``confirm``-ed."""
        replay = _REPLAY.get() if replay is None else replay
        paused = 0.0
        while True:
            paused += await self.ready(what=what)
            try:
                return await self.guard(op, what=what)
            except HostUnreachable as e:
                if e.resumed:  # an inner call already paused for this outage and came back
                    raise
                t0 = self._now()
                back = await self.outage(e.why, what=what)
                paused += self._now() - t0
                if not back:
                    raise HostUnreachable(self.name, self._gave_up(what, e.why, paused), dispatched=e.dispatched, paused_s=paused) from e
                if e.dispatched and not replay:
                    raise HostUnreachable(self.name, e.why, dispatched=True, resumed=True, paused_s=paused) from e
                # the host is back and the operation may run again: nothing reached the host, or it is the same when repeated

    async def wait_up(self, max_s: float | None = None) -> bool:
        """Fleet stop: True at once when the host is not known to be down; else probe with backoff until it answers (True)
        or ``max_s`` (default ``pause_s``) passes (False). Holds no clock: nothing has started."""
        if not self.down:
            return True
        limit = self.rule.pause_s if max_s is None else max_s
        t0 = self._now()
        wait = self.rule.backoff_s
        while not await self.probe():
            spent = self._now() - t0
            if spent >= limit:
                return False
            await self._sleep(min(wait, limit - spent))
            wait = min(wait * 2, self.rule.backoff_cap_s)
        return True

    def state(self) -> dict[str, Any]:
        return {"host": self.name, "down": self.down, "down_s": None if self.down_since is None else round(self._now() - self.down_since, 3),
                "why": self.why if self.down else "", "probes": self.probes}


# ---- probes and the process's hosts ------------------------------------------------------------------------------------------------
def docker_probe(docker_host: str, timeout_s: float = 10.0) -> Callable[[], Awaitable[bool]]:
    """The daemon at ``docker_host`` answers ``docker version`` (through the shared tunnel for ``ssh://``, which the call
    restarts when it died): the path every World call takes."""

    async def probe() -> bool:
        from crucible.world.exec import _run_subprocess

        out = await _run_subprocess(["docker", "version", "--format", "{{.Server.Version}}"], cwd=None, env={"DOCKER_HOST": docker_host}, timeout_s=timeout_s,
                                    stdin=None)
        return out.ok and bool(out.stdout.strip())

    return probe


def ssh_probe(argv: Sequence[str], timeout_s: float = 10.0) -> Callable[[], Awaitable[bool]]:
    """``argv`` (an ``ssh ... <target> true`` with BatchMode and ConnectTimeout) exits 0."""

    async def probe() -> bool:
        from crucible.world.exec import _run_subprocess

        out = await _run_subprocess(list(argv), cwd=None, env=None, timeout_s=timeout_s, stdin=None)
        return out.rc == 0

    return probe


_HOSTS: dict[str, HostReach] = {}
DEFAULT_RULE = ReachRule()  # the rule of a host made without one (the test suite shortens its pause: tests/conftest.py)


def reach_for(key: str, probe: Callable[[], Awaitable[bool]], *, name: str | None = None, rule: ReachRule | None = None) -> HostReach:
    """The process's ``HostReach`` for ``key``, made with ``probe`` on first use. ``name`` (the host's name in records and
    journals, e.g. ``rtx5090``) renames it when given."""
    r = _HOSTS.get(key)
    if r is None:
        r = _HOSTS[key] = HostReach(name or key, probe, rule or DEFAULT_RULE)
    elif name:
        r.name = name
    return r


def for_docker_host(docker_host: str, *, name: str | None = None) -> HostReach:
    return reach_for(f"docker:{docker_host}", docker_probe(docker_host), name=name)


def for_ssh(key: str, argv: Sequence[str], *, name: str | None = None) -> HostReach:
    return reach_for(f"ssh:{key}", ssh_probe(argv), name=name)


def register(key: str, reach: HostReach) -> HostReach:
    """Install ``reach`` for ``key`` (a test's fake host; a launcher with its own probe)."""
    _HOSTS[key] = reach
    return reach


def hosts() -> list[HostReach]:
    return list(_HOSTS.values())


def reset() -> None:
    """Forget every host and subscriber (tests)."""
    _HOSTS.clear()
    _LISTENERS.clear()
    _HOLDS.clear()
