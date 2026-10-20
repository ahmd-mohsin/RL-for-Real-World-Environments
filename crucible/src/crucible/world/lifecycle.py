"""Container lifecycle: every container Crucible starts names its owner and an expiry, dies on its own at that expiry, is removed
on every exit path of the code that started it, and is swept when its owner is gone (M10 lab audit R-6).

Before R-6 a World sandbox was ``docker run -d --rm --init ... sleep infinity`` with two labels (``crucible.world``,
``crucible.sandbox``): a runner that was killed or OOM-killed left its containers (GPU ones included) running for ever, an
``open()`` that failed after ``World.start`` leaked one more container per retry (``HealthcheckError`` is retryable infra), and
PerfGuard counted every orphan as contention, so one crashed run held gate R1 for 12 h. This module is the shared design:

- **Labels** (``Scope.labels``): ``crucible.owner=<host>:<pid>:<process start>``, ``crucible.expires=<epoch>``, and, inside a Runner
  attempt, ``crucible.run``, ``crucible.attempt`` and ``crucible.lease`` (the absolute path of the run's R-4 ``run.lock``: the sweep
  decides liveness by that lock, never by pid, which has no meaning on Windows). ``crucible.renewable=1`` marks a keeper whose life a
  holder extends: its expiry label is a floor, not a promise (PerfGuard never ignores a running renewable container).
- **A bounded life** (``Scope.keeper``): ``sleep <ttl>`` (``--rm`` makes the container remove itself when it ends), or, where a
  holder renews it (every Runner attempt, the warm pools, a sandbox outside a Runner), the RLM kernel's life loop. The TTL is the attempt backstop plus a
  margin (``lifetime_for``).
- **Every exit path**: ``shielded`` / ``finish`` (a cancelled awaiter lets the ``docker run`` or ``docker rm`` finish, bounded),
  ``remove`` (rc-checked), and the ``Scope`` an attempt runs in (``attempt``): what started under it is recorded, renewed while the
  attempt lives, and on exit closed, removed by name and by its ``crucible.attempt`` label *and its run's lease* (a ``docker run`` that
  a daemon finished after its client was killed; the attempt id is ``<sample key>-a<N>`` and a sample key does not hash the run, so
  two runs of one plan on one daemon share attempt ids: the lease, or ``crucible.run`` without one, is what makes the filter this
  attempt's), and reported as ``{terminated, failed, still_up, released}`` (``attempt/teardown.json``).
- **At interpreter exit and on SIGTERM** (``_reap_own``): the process removes the containers labelled with its own owner id (atexit,
  and a chained SIGTERM handler installed with the reaper on the main thread: it reaps, then calls the handler it found or re-raises
  under ``SIG_DFL``; the RLM kernel chains the same way, so neither pre-empts the other). A runner stopped by systemd, Slurm,
  Kubernetes or ``kill <pid>`` therefore reclaims; Ctrl-C reaches the attempt scopes' finally blocks. SIGKILL bypasses atexit, and so do
  OOM kills: the TTL and the sweep cover them.
- **The sweep** (``sweep``, called by ``Runner`` at every start, a resume included, once the run's lease is held): a runner that was
  killed leaves containers whose keeper may outlive it (a renewed life loop) or whose TTL has not run out. The sweep lists every
  ``crucible.owner`` container on the local daemon, on each docker host the run used (``RunJournal.sandbox_hosts``) and on each docker
  host the plan names (``Runner._plan_hosts``: a task's ``docker_hosts()``: the grading host, the agent's World host) and removes
  (``stale``): a container carrying *this run's* lease path from this host (``own_run``: no attempt of this run is live yet, and the
  lease guarantees no other writer); a same-host container whose run's ``run.lock`` is free or gone (``lease_free``: the run that
  owns it has ended or died); and any other container past its ``crucible.expires``. A *renewable* container's label is a floor
  (renewal pushes its real deadline past it), so label expiry alone never removes one: its real deadline (``/crucible-ctl/deadline``,
  read with ``docker exec``) decides, and one whose deadline cannot be read is removed only a day past its label
  (``UNREADABLE_GRACE_S``). It never decides by pid liveness (``os.kill(pid, 0)`` has no meaning on Windows), never touches a
  container without ``crucible.owner`` (RLM kernels, Harbor compose containers, the gVisor toolbox) and never creates a lock file. A
  lease it cannot resolve here (a path of the other OS flavour, a runs root that is not mounted) is "cannot tell", never "gone".
  Every sweep is journaled (``sweep``); a failed one never fails the run. A host the plan does not name and the run has not used is
  covered by the TTL and the expiry rule only (there is no per-host lazy sweep: a grade must not spend its scoring budget on one).

Grade containers (``crucible.interop.harbor.verify_dir``, ``crucible.verify.stage``) carry the labels and an in-container bound of their
own (``BOUND_SH``, ``BOUND_PY``: the grade's timeout plus ``GRADE_MARGIN_S``, so a grade a killed runner left ends by itself even where
no later Runner reaches its host; where the image has no ``timeout`` the bound is skipped and the label and the sweep are what is left).
In a live run the ``docker run`` timeout and its ``rm -f`` end a hung grade long before that.

Outside a Runner (the OpenEnv session, the gym wrapper, ``train.world_env``, a direct ``World``) a sandbox runs the renewable life loop
and renews itself while it is open (``DockerSandbox``: a per-sandbox renewer, cancelled by ``close``), so the 6 h default is the crash
backstop and never a cap on a live World.

PATTERN only, no code copied: mini-swe-agent ``environments/docker.py:30-35,77-89`` ``docker run -d --rm ... sleep 2h`` (MIT);
NeMo-Gym ``sandbox/providers/docker/provider.py:295-373`` a label plus ``sleep ttl_s`` with ``--rm`` (Apache-2.0); verifiers
``v1/runtimes/deployment.py:101-109,193-202`` and ``base.py:106-133`` register before start, a shielded idempotent close, an atexit
backstop, and the note that SIGKILL bypasses atexit (MIT); agentenv-framework ``agent_env/task/teardown.py:27-118`` a bounded
teardown that reports ``still_up`` (Apache-2.0); karotte ``run_helpers.py:681-761`` and ``firecracker/vm.py:435-534`` run-keyed
cleanup and a lock-probed liveness (MIT). The design reused is in-tree: the RLM kernel's container lifecycle
(``crucible.actors.rlm.kernel``, M14 campaign B4 and M16 B9), whose life loop, deadline file and shielded close moved here and are
imported back by the kernel: there is one lifecycle design in the repository.

Layering: stdlib, ``crucible.core`` (locks) and ``crucible.world.exec`` only; ``World`` is imported under ``TYPE_CHECKING``. Its
callers (``world.sandbox``, ``world.pool``, and above them runtime, verify, interop, bench, actors) sit at or above ``world``.
Every docker call made while a loop runs goes through ``world.exec._run_subprocess`` (``docker`` / ``remove`` / ``labelled``); the
only blocking calls are ``_reap_own``'s, which run after the loop has ended (or from the SIGTERM handler, with the process ending).
"""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import contextvars
import os
import re
import shutil
import signal
import socket
import subprocess
import threading
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable, Mapping, Sequence
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, TypeVar

from crucible.core.locks import FileLock, process_start_time
from crucible.world.exec import ExecOut, _run_subprocess

if TYPE_CHECKING:
    from crucible.world.world import World

T = TypeVar("T")

RUN_LABEL, OWNER_LABEL, EXPIRES_LABEL = "crucible.run", "crucible.owner", "crucible.expires"
ATTEMPT_LABEL, LEASE_LABEL, RENEWABLE_LABEL = "crucible.attempt", "crucible.lease", "crucible.renewable"
VOLATILE_LABELS = (RUN_LABEL, OWNER_LABEL, EXPIRES_LABEL, ATTEMPT_LABEL, LEASE_LABEL, RENEWABLE_LABEL)  # what a pool signature ignores

# Moved verbatim from ``actors/rlm/kernel.py`` (M14 B4, M16 B9), which imports them back under its old names.
DEFAULT_LIFETIME_S = 6 * 3600
RENEW_SLACK_S = 3600  # a renewable container's expiry label lies this far past its start deadline (a label cannot be renewed)
CTL_DIR = "/crucible-ctl"  # renewable containers: a root-owned tmpfs holding the deadline (the policy's user cannot write it)
# PID 1 of a renewable container: exits at the later of its start deadline and the one ``renew`` wrote
LIFE_LOOP = ('d=$1; while :; do n=$(cat ' + CTL_DIR + '/deadline 2>/dev/null); [ "${n:-0}" -gt "$d" ] 2>/dev/null && d=$n; '
             '[ "$(date +%s)" -lt "$d" ] || exit 0; sleep 2; done')

CLOSE_TIMEOUT_S = 120.0  # one ``docker rm -f``
START_GRACE_S = 120.0  # a cancelled ``docker run`` is given this long to finish (an image pull) before it is removed in the background
RECHECK_S = 2.0  # a container a daemon created after the first removal: look again this much later
RENEW_EVERY_S = 60.0
RENEW_BEFORE_S = 900.0  # a renewable container is renewed once its remaining life is shorter than this (or half the scope's ttl)
GRADE_MARGIN_S = 900.0  # a grade container's expiry: its timeout plus this
LIST_TIMEOUT_S = 60.0
WORLD_CLOSE_TIMEOUT_S = 600.0  # what ``World.close`` may take (``World.close_timeout_s`` defaults to it); the attempt's reclaim allows for it
PROBE_TIMEOUT_S = 30.0  # the sweep's ``docker exec <c> cat /crucible-ctl/deadline`` (one per renewable container past its label)
UNREADABLE_GRACE_S = 24 * 3600.0  # a renewable container whose real deadline cannot be read is removed this long past its expiry label
POOL_CLOSE_WAIT_S = 60.0  # ``SandboxPool.close`` waits this long for refills that are still in ``docker run`` (they clean up after themselves)
POOLS_CLOSE_S = 300.0  # what ``Runner`` allows ``close_pools`` on the normal path (and for a cancelled run to finish it)
# M10 lab audit R-6 review round 1: the in-container bound of a grade container (its keeper is the task's own command, so there is no
# ``sleep <ttl>``): ``--entrypoint sh|bash <image> -c BOUND_SH crucible-bound <seconds> <program> <args>`` runs the program under
# ``timeout -k 30`` where the image has one (coreutils, busybox) and as it was otherwise; ``--entrypoint python3 <image> -I -c BOUND_PY
# <seconds> -I <script> ...`` arms a SIGALRM that survives the exec (no binary needed). Both fire only after the grade's own client-side
# timeout would have: the backstop of a grade a killed runner left, on a host no later Runner sweeps.
BOUND_SH = 'n=$1; shift; if command -v timeout >/dev/null 2>&1; then exec timeout -k 30 "$n" "$@"; fi; exec "$@"'
BOUND_PY = "import os, signal, sys; signal.alarm(int(sys.argv[1])); os.execvp('python3', ['python3', *sys.argv[2:]])"


@dataclass(frozen=True)
class Hygiene:
    """``RunSpec``'s container lifecycle settings: the TTL margin and floor, the lifetime without an attempt backstop, and whether a
    Runner start sweeps orphans. Operational: ``identity.DENY`` drops ``hygiene``, so changing it never drifts a run (it says how
    long an orphan lives, not what is measured)."""

    margin_s: float = 900.0
    floor_s: float = 600.0
    default_s: int = DEFAULT_LIFETIME_S
    sweep: bool = True


def lifetime_for(wall_s: float | None, *, margin_s: float = 900.0, floor_s: float = 600.0, default_s: int = DEFAULT_LIFETIME_S) -> int:
    """A container's lifetime for a holder with wall budget ``wall_s`` (M16 B9: a fixed 6 h ended long episodes): the budget plus a
    margin for restores and grading; ``default_s`` (6 h) when there is no wall budget."""
    if not wall_s or wall_s <= 0:
        return int(default_s)
    return int(max(floor_s, float(wall_s) + margin_s))


_OWNER: dict[int, str] = {}


def owner_id() -> str:
    """``<host>:<pid>:<process start>`` (``crucible.owner``); the start time (``core.locks.process_start_time``, 0 where it cannot be
    told) tells a reused pid from the process that started a container. Cached per pid: a forked child recomputes."""
    pid = os.getpid()
    if pid not in _OWNER:
        _OWNER.clear()
        _OWNER[pid] = f"{socket.gethostname()}:{pid}:{int(process_start_time() or 0)}"
    return _OWNER[pid]


def owner_host(owner: str) -> str:
    """The host part of an owner label: ``host:pid:start`` (this module) or ``host:pid`` (the RLM kernel's) without its numeric tail."""
    head, _, tail = owner.rpartition(":")
    if head and tail.isdigit():
        h2, _, t2 = head.rpartition(":")
        return h2 if h2 and t2.isdigit() else head
    return owner


# ---- cancellation and bounded waits ----------------------------------------------------------------------------------------

_BACKGROUND: set[asyncio.Future] = set()  # a shielded task nobody awaits any more must not be garbage collected mid-flight


def hold(aw: Awaitable[T]) -> asyncio.Future[T]:
    """``ensure_future`` that keeps the task alive until it ends."""
    fut = asyncio.ensure_future(aw)
    _BACKGROUND.add(fut)
    fut.add_done_callback(_BACKGROUND.discard)
    return fut


async def finish(task: asyncio.Future, timeout: float) -> None:
    """Let a shielded task finish after its awaiter was cancelled (bounded; the kernel's ``_finish``)."""
    with contextlib.suppress(BaseException):
        await asyncio.wait_for(asyncio.shield(task), timeout)


async def shielded(aw: Awaitable[T], timeout: float) -> T:
    """Await ``aw`` so that a cancel of the caller does not cancel it: the work runs to its end (bounded by ``timeout`` for the
    caller that was cancelled; it keeps running in the background after that) and the ``CancelledError`` is re-raised then."""
    fut = hold(aw)
    try:
        return await asyncio.shield(fut)
    except asyncio.CancelledError:
        await finish(fut, timeout)
        raise


# ---- the docker seam (tests patch ``_run_subprocess`` here and in ``world.sandbox``) ------------------------------------------


async def docker(args: list[str], docker_host: str | None, *, timeout_s: float) -> ExecOut:
    """One docker CLI call off the event loop, bounded (``world.exec._run_subprocess``)."""
    return await _run_subprocess(["docker", *args], cwd=None, env={"DOCKER_HOST": docker_host} if docker_host else None, timeout_s=timeout_s, stdin=None)


async def remove(name: str, docker_host: str | None) -> tuple[bool, str]:
    """``docker rm -f``, rc-checked: removed on rc 0 or "No such container" (it never existed or ended by itself); any other result
    (a daemon that cannot be reached, a timeout) is ``(False, why)``. Never raises."""
    try:
        out = await docker(["rm", "-f", name], docker_host, timeout_s=CLOSE_TIMEOUT_S)
    except OSError as e:
        return False, str(e)
    if out.ok or "no such container" in (out.stderr or "").lower():
        return True, ""
    if out.timed_out:
        return False, f"docker rm -f ran past {CLOSE_TIMEOUT_S:g}s"
    return False, (out.stderr or out.stdout or f"rc {out.rc}").strip()[-300:]


async def labelled(docker_host: str | None, label: str, *also: str) -> list[str] | None:
    """Names of every container (running or not) carrying ``label`` (``key`` or ``key=value``) and every label of ``also``
    (``docker`` ANDs repeated ``--filter label=``); ``None`` when the listing failed."""
    filters = [x for lb in (label, *also) for x in ("--filter", f"label={lb}")]
    try:
        out = await docker(["ps", "-a", *filters, "--format", "{{.Names}}"], docker_host, timeout_s=LIST_TIMEOUT_S)
    except OSError:
        return None
    if not out.ok:
        return None
    return [n for n in out.stdout.split() if n]


# ---- the scope --------------------------------------------------------------------------------------------------------------


@dataclass
class _Live:
    docker_host: str | None
    renewable: bool
    deadline: float
    owned: bool = True  # False: borrowed from a pool (the pool, not this scope, ends it)
    renew: Callable[[float], Awaitable[bool]] | None = None
    remove: Callable[[], Awaitable[tuple[bool, str]]] | None = None


@dataclass(eq=False)
class Scope:
    """Who a container belongs to, and what became of the ones started under it. ``Runner`` enters one per attempt (``attempt``);
    outside a Runner every container is started under the module default (``current()``), which labels and installs the exit
    reaper but records nothing (``record=False``: a long-lived training process must not grow a list per rollout).

    ``run``, ``attempt`` and ``lock`` become labels; ``ttl_s`` is the lifetime of a container started here (``lifetime_for`` of the
    attempt backstop); ``renewable`` makes keepers the kernel's life loop, renewed while an *attempt* scope lives; ``event`` is the
    run journal's ``event(type, **data)`` (``None`` after the run's lease is released); ``hosts`` is the set of docker hosts the run
    has used (``''`` is the local daemon), shared by all of a run's scopes so a resume sweeps each of them."""

    run: str | None = None
    attempt: str | None = None
    lock: str | None = None
    ttl_s: int = DEFAULT_LIFETIME_S
    renewable: bool = False
    event: Callable[..., None] | None = None
    hosts: set[str] = field(default_factory=set)
    record: bool = True
    # what became of the containers started here
    terminated: list[str] = field(default_factory=list)
    released: list[str] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    still_up: list[str] = field(default_factory=list)
    raced: bool = False  # a docker run was cancelled: a daemon may still create its container, so reclaim looks twice
    worlds: Any = field(default_factory=weakref.WeakSet, repr=False)
    _live: dict[str, _Live] = field(default_factory=dict, repr=False, init=False)
    _used: set[str] = field(default_factory=set, repr=False, init=False)
    _renewer: asyncio.Future | None = field(default=None, repr=False, init=False)

    # ---- what a container is started with
    def labels(self, *, expires: float, attempt: bool = True, renewable: bool = False) -> list[str]:
        """The ``--label k=v`` arguments: owner and expiry always; run, lease and attempt where known."""
        pairs = [(OWNER_LABEL, owner_id()), (EXPIRES_LABEL, str(int(expires)))]
        if self.run:
            pairs.append((RUN_LABEL, self.run))
        if self.lock:
            pairs.append((LEASE_LABEL, self.lock))
        if attempt and self.attempt:
            pairs.append((ATTEMPT_LABEL, self.attempt))
        if renewable:
            pairs.append((RENEWABLE_LABEL, "1"))
        return [x for k, v in pairs for x in ("--label", f"{k}={v}")]

    def keeper(self, *, renewable: bool, lifetime_s: int | None = None) -> tuple[list[str], list[str], int, int]:
        """``(docker run args, container command, deadline, expires)``: the kernel's life loop (``--tmpfs`` for its deadline file) or
        ``sleep <ttl>``. A renewable container's expiry label lies ``RENEW_SLACK_S`` past its deadline."""
        lifetime = int(lifetime_s or self.ttl_s)
        deadline = int(time.time()) + lifetime
        if renewable:
            return ["--tmpfs", f"{CTL_DIR}:rw,size=64k,mode=0755"], ["sh", "-c", LIFE_LOOP, "crucible-life", str(deadline)], deadline, deadline + RENEW_SLACK_S
        return [], ["sleep", str(lifetime)], deadline, deadline

    def run_scope(self) -> Scope:
        """The same run without an attempt: what a pool owns. A pool's refill tasks copy the attempt's context, so a pool builds its
        containers with this scope and an attempt's reclaim never removes a warm container (they belong to the run)."""
        return Scope(run=self.run, lock=self.lock, ttl_s=self.ttl_s, renewable=True, event=self.event, hosts=self.hosts, record=self.record)

    # ---- recording
    def _event(self, type: str, **data: Any) -> None:
        if self.event is not None:
            with contextlib.suppress(Exception):
                self.event(type, **data)

    def started(self, name: str, docker_host: str | None, *, renewable: bool, deadline: float, renew: Callable[[float], Awaitable[bool]] | None = None,
                remove: Callable[[], Awaitable[tuple[bool, str]]] | None = None, owned: bool = True) -> None:
        """Recorded *before* the container exists (verifiers: register before start), so every exit path knows its name."""
        _install_reaper()
        key = docker_host or ""
        _HOSTS.add(key)
        self._used.add(key)
        if key not in self.hosts:
            self.hosts.add(key)
            self._event("sandbox_host", host=key)
        if not self.record:
            return
        self._live[name] = _Live(docker_host, renewable, float(deadline), owned, renew, remove)
        if renewable:
            self._ensure_renewer()

    def closed(self, name: str, *, ok: bool, error: str | None = None) -> None:
        """A removal's outcome. A failed one stays live (reclaim retries it) and is reported; a name this scope never started is ignored."""
        if not self.record or name not in self._live:
            return
        if ok:
            self._mark_terminated(name)
        else:
            self.failed.append({"name": name, "error": error or ""})

    def _mark_terminated(self, name: str) -> None:
        self._live.pop(name, None)
        self.failed = [f for f in self.failed if f.get("name") != name]  # ``failed`` is what is still failing
        if name not in self.terminated:
            self.terminated.append(name)

    def borrowed(self, name: str, docker_host: str | None, *, deadline: float, renew: Callable[[float], Awaitable[bool]] | None, renewable: bool = True) -> None:
        """A pooled container handed to this scope: renewed while the attempt lives (an in-place grade can outlive its hand-out ttl),
        never removed by it (the pool owns it)."""
        if not self.record:
            return
        self._live[name] = _Live(docker_host, renewable, float(deadline), False, renew, None)
        if renewable and renew is not None:
            self._ensure_renewer()

    def returned(self, name: str, *, destroyed: bool) -> None:
        if not self.record or name not in self._live:
            return
        self._live.pop(name, None)
        (self.terminated if destroyed else self.released).append(name)

    def track_world(self, w: World) -> None:
        if self.attempt is not None:  # the module default must never hold Worlds
            self.worlds.add(w)

    def world_closed(self, w: World, error: str | None) -> None:
        self.worlds.discard(w)
        if error and self.record:
            self.failed.append({"world": getattr(getattr(w, "spec", None), "world", "?"), "error": error})

    def report(self) -> dict[str, list]:
        return {"terminated": list(self.terminated), "failed": [dict(f) for f in self.failed], "still_up": list(self.still_up),
                "released": list(self.released)}

    # ---- renewal
    def _ensure_renewer(self) -> None:
        if not (self.renewable and self.attempt is not None):
            return
        if self._renewer is None or self._renewer.done():
            try:
                self._renewer = hold(self._renew_loop())
            except RuntimeError:  # no running loop: nothing to renew from
                self._renewer = None

    async def _renew_loop(self) -> None:
        """While the attempt lives, push each renewable container's deadline out to ``now + ttl_s`` once its remaining life is short.
        R-3's grading window can move the attempt clock past ``attempt_timeout_s``; a fixed life would end an in-place grade or its
        control. A failed renewal (the container is gone) is journaled once per name. Never raises."""
        warned: set[str] = set()
        try:
            while True:
                await asyncio.sleep(min(RENEW_EVERY_S, self.ttl_s / 4))
                window = min(RENEW_BEFORE_S, self.ttl_s / 2)
                for name, e in list(self._live.items()):
                    if not e.renewable or e.renew is None or e.deadline - time.time() >= window:
                        continue
                    try:
                        ok = await e.renew(self.ttl_s)
                    except Exception:  # noqa: BLE001 -- a failed renewal is a journaled fact, not a failed attempt
                        ok = False
                    if ok:
                        e.deadline = max(e.deadline, time.time() + self.ttl_s)
                    elif name not in warned:
                        warned.add(name)
                        self._event("renew_failed", attempt=self.attempt, name=name)
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001
            return

    # ---- reclaim
    def attempt_labels(self) -> list[str]:
        """What selects exactly this attempt's containers on a daemon other runs share: its ``crucible.attempt`` *and* its run's
        ``crucible.lease`` (``crucible.run`` when the scope has no lock). The attempt id (``<sample key>-a<N>``) is the same string in
        any two runs of one plan (``SampleKey`` hashes neither the run id nor the root), so on its own it would also select, and
        reclaim, another run's live sandbox or grade container (M10 lab audit R-6 review round 1)."""
        out = [f"{ATTEMPT_LABEL}={self.attempt}"]
        if self.lock:
            out.append(f"{LEASE_LABEL}={self.lock}")
        elif self.run:
            out.append(f"{RUN_LABEL}={self.run}")
        return out

    async def reclaim(self) -> None:
        """An attempt scope's exit (attempts only; never raises): stop renewing, close the Worlds still open, remove what this
        attempt started by name, then by its ``crucible.attempt`` and lease labels (``attempt_labels``) on every host it used (a
        ``docker run`` that a daemon finished after its client was killed), and list what survived (``still_up``)."""
        try:
            await self._reclaim()
        except Exception as e:  # noqa: BLE001 -- the attempt's exit never fails on its own cleanup
            self.failed.append({"error": f"reclaim: {type(e).__name__}: {e}"})

    async def _reclaim(self) -> None:
        if self.attempt is None:
            return
        r, self._renewer = self._renewer, None
        if r is not None:
            r.cancel()
            await asyncio.wait({r})
        for w in list(self.worlds):
            if getattr(w, "closed", False):
                continue
            try:
                await w.close()
            except Exception as e:  # noqa: BLE001 -- World.close records itself; this is for one that did not
                if w in self.worlds:
                    self.world_closed(w, f"{type(e).__name__}: {e}")
        for name, e in list(self._live.items()):
            if not e.owned:
                self._live.pop(name, None)
                self.released.append(name)  # a pooled container the attempt never gave back: close_pools ends it with the run
                continue
            try:
                ok, err = await (e.remove() if e.remove is not None else remove(name, e.docker_host))
            except Exception as ex:  # noqa: BLE001
                ok, err = False, f"{type(ex).__name__}: {ex}"
            self.closed(name, ok=ok, error=err)
        label, *rest = self.attempt_labels()
        clean: set[str] = set()
        for host in sorted(self._used):
            for round_ in range(2 if self.raced else 1):
                if round_:
                    await asyncio.sleep(RECHECK_S)
                names = await labelled(host or None, label, *rest)
                clean.discard(host)
                if names is None:
                    break
                for n in names:
                    ok, err = await remove(n, host or None)
                    if ok:
                        self._mark_terminated(n)
                    else:
                        self.failed.append({"name": n, "error": err})
                if not names:
                    clean.add(host)  # listed empty and nothing removed after it: that listing is the final one
        still: set[str] = set()
        for host in sorted(self._used - clean):
            names = await labelled(host or None, label, *rest)
            if names is None:
                self.failed.append({"host": host, "error": "listing the attempt's containers failed"})
                gone = set(self.terminated)
                still |= {n for n in [*self._live, *(f["name"] for f in self.failed if "name" in f)] if n not in gone}
            else:
                still |= set(names)
        self.still_up = sorted(still)


# ---- module state and context -----------------------------------------------------------------------------------------------

_SCOPE: contextvars.ContextVar[Scope | None] = contextvars.ContextVar("crucible_lifecycle_scope", default=None)
# renewable: a sandbox outside a Runner runs the life loop and renews itself while it is open (``DockerSandbox``; this scope records and
# renews nothing), so the default ttl is a crash backstop, not a cap on a live World (review round 1: a fixed ``sleep 21600`` ended
# sessions and episodes that held a World for more than 6 h)
_DEFAULT = Scope(record=False, renewable=True)
_HOSTS: set[str] = set()  # every docker host this process started a container on ('' is the local daemon)
_REAPER: dict[str, bool] = {"installed": False}
_SIGTERM: dict[str, Any] = {"prev": None}  # the handler this module's replaced (``None``: the default action)


def current() -> Scope:
    return _SCOPE.get() or _DEFAULT


@asynccontextmanager
async def attempt(scope: Scope) -> AsyncIterator[Scope]:
    """Run an attempt under ``scope``: containers started inside (this task and the tasks it creates) belong to it. On exit, however
    it ends, ``reclaim`` runs shielded from cancellation (a cancelled run still removes its containers), bounded by four removals
    and one World close; a ``CancelledError`` is re-raised after it."""
    token = _SCOPE.set(scope)
    try:
        yield scope
    finally:
        try:
            await shielded(scope.reclaim(), CLOSE_TIMEOUT_S * 4 + WORLD_CLOSE_TIMEOUT_S)
        finally:
            with contextlib.suppress(ValueError):
                _SCOPE.reset(token)


@asynccontextmanager
async def opening(world: World) -> AsyncIterator[World]:
    """The async context every ``RunTask.open`` uses (M10 lab audit R-6): ``async with lifecycle.opening(World(...)) as w:`` awaits
    ``w.start()`` and yields it; if anything in the block (a reset, a health check, the task's own setup) raises or is cancelled,
    the World is closed before the error propagates, so an ``open()`` that fails after ``start`` closes what ``start`` started.
    ``HealthcheckError`` and ``ResetHookError`` are retryable infra (``runtime.infra`` NORMALIZE), so each retry used to leak one
    more container. On success the World stays open: the caller (the harness, through ``RunTask.open``'s return value) owns it.

    Duck-typed on purpose (``start`` and ``close`` only): a test's fake World works. ``World.close`` is shielded, idempotent and
    never raises anything new, so closing here and again in the caller's cleanup is safe. An ``open`` that does not use this is
    still covered, under a Runner, by the attempt scope's reclaim (``Scope.worlds``)."""
    await world.start()
    try:
        yield world
    except BaseException:
        with contextlib.suppress(Exception):
            await world.close()
        raise


def _reap_own() -> list[str]:
    """Remove every container labelled with this process as its owner, on every docker host it used (atexit). Synchronous: it runs
    after the event loop has ended, as the RLM kernel's ``_docker_sync`` does. Covers what ``close_pools`` and an attempt's reclaim
    could not reach (an interpreter exiting mid-run); SIGKILL and an OOM kill bypass atexit (verifiers ``base.py:110-111``), and the
    keeper's TTL and the next Runner's sweep cover those."""
    removed: list[str] = []
    if not shutil.which("docker"):
        return removed
    me = owner_id()
    for host in sorted(_HOSTS):
        env = {**os.environ, "DOCKER_HOST": host} if host else None
        try:
            r = subprocess.run(["docker", "ps", "-a", "-q", "--filter", f"label={OWNER_LABEL}={me}"], capture_output=True, text=True, timeout=30,
                               check=False, env=env)
            ids = (r.stdout or "").split()
            if ids:
                subprocess.run(["docker", "rm", "-f", *ids], capture_output=True, text=True, timeout=120, check=False, env=env)
                removed += ids
        except (OSError, subprocess.SubprocessError):
            continue
    return removed


def _on_sigterm(signum: int, frame: Any) -> None:
    """SIGTERM (systemd, Slurm, Kubernetes, ``kill <pid>``): Python's default action ends the process without atexit or ``finally``, so
    nothing reclaimed (M10 lab audit R-6 review round 1). Reap this process's containers, then do what would have happened without
    this handler: call the handler it replaced (the RLM kernel's, which reaps its own and re-raises), or restore the default action
    and re-raise the signal. ``_reap_own`` is synchronous and bounded; the process is ending."""
    try:
        _reap_own()
    except Exception:  # noqa: BLE001 -- the signal must still take effect
        pass
    prev = _SIGTERM["prev"]
    if callable(prev):
        prev(signum, frame)
        return
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    os.kill(os.getpid(), signal.SIGTERM)


def _install_reaper() -> None:
    """Once per process: remove this process's own containers at exit and on SIGTERM. The SIGTERM handler is installed on the main
    thread only, chains to a handler already there (the RLM kernel's installs the same way, so whichever is first is the other's
    ``prev``) and is left out when the process ignores SIGTERM."""
    if _REAPER["installed"]:
        return
    _REAPER["installed"] = True
    atexit.register(_reap_own)
    if threading.current_thread() is threading.main_thread():
        with contextlib.suppress(ValueError, OSError):
            prev = signal.getsignal(signal.SIGTERM)
            if prev is not signal.SIG_IGN:
                _SIGTERM["prev"] = prev if callable(prev) else None  # SIG_DFL or a handler that was not set from Python
                signal.signal(signal.SIGTERM, _on_sigterm)


# ---- the sweep (M10 lab audit R-6: after a crash, kill -9 or OOM kill, the next start reclaims) -----------------------------------

RUN_LOCK = "run.lock"  # mirrors ``runtime.journal.LOCK`` (this layer cannot import runtime): the only lease file a sweep will probe
PS_FORMAT = ('{{.Names}}\t{{.Label "crucible.run"}}\t{{.Label "crucible.attempt"}}\t{{.Label "crucible.owner"}}\t'
             '{{.Label "crucible.expires"}}\t{{.Label "crucible.lease"}}\t{{.Label "crucible.renewable"}}')


@dataclass(frozen=True)
class Row:
    """One ``crucible.owner`` container as ``docker ps --format PS_FORMAT`` lists it (an unset label is the empty string).
    ``renewable`` is ``crucible.renewable`` (``"1"`` for a keeper whose life a holder extends)."""

    name: str
    run: str
    attempt: str
    owner: str
    expires: str
    lease: str
    renewable: str = ""


def rows(text: str) -> list[Row]:
    """Parse ``PS_FORMAT`` output: tab-split lines of seven fields; anything else (a warning line, a blank) is ignored."""
    out: list[Row] = []
    for line in text.splitlines():
        f = line.rstrip("\r\n").split("\t")
        if len(f) == 7 and f[0].strip():
            out.append(Row(f[0].strip(), *f[1:]))
    return out


def docker_available() -> bool:
    """Whether a docker CLI is on PATH: without one there is nothing to sweep (a seam: unit tests patch it, ``tests/unit/conftest.py``)."""
    return shutil.which("docker") is not None


def is_run_lock(path: str) -> bool:
    """An absolute path whose file name is ``run.lock``: the only kind of ``crucible.lease`` a sweep probes. A label is data any
    container on a shared daemon can carry; opening and ``flock``-ing whatever path it names would be a way to make a runner
    touch arbitrary files, so any other lease counts as "cannot tell" and the expiry rule decides."""
    return bool(path) and (PurePosixPath(path).is_absolute() or PureWindowsPath(path).is_absolute()) and re.split(r"[\\/]", path)[-1] == RUN_LOCK


def lease_state(path: str) -> bool | None:
    """Whether the run that owns ``path`` (its R-4 ``run.lock``) is gone: the lease is the owner's liveness (``os.kill(pid, 0)`` has
    no meaning on Windows). ``True``: free or gone (the file does not exist under a runs root that does, or its lock can be taken
    and is given back at once); ``False``: held; ``None``: cannot tell, and the caller keeps the container to the next rule. That is a
    path that is not a ``run.lock`` (``is_run_lock``), one that is not absolute in *this* OS's flavour (a WSL2 runner and a Windows
    runner sharing Docker Desktop's engine: a POSIX path never resolves on Windows and a Windows one never does on Linux, so
    "does not exist" would be read as "the run is gone"), a file that is missing because the runs root is (an unmounted share), and
    an ``OSError``. Never creates a lock file. A probe holds the lock for microseconds, so an opener of that run in that moment sees
    ``RunBusy`` (the mirror of the race ``runtime.journal.assert_free`` documents). M10 lab audit R-6 review round 1."""
    if not is_run_lock(path):
        return None
    p = Path(path)
    if not p.is_absolute():
        return None
    try:
        if not p.exists():
            runs_root = p.parent.parent  # <root>/<run id>/run.lock: a run directory is gone only while its root is there to say so
            return True if runs_root.is_dir() and os.access(runs_root, os.R_OK) else None
        lk = FileLock(p)
        try:
            lk.acquire()
        except BlockingIOError:
            return False
        lk.release()
        return True
    except OSError:
        return None


def lease_free(path: str) -> bool:
    """``lease_state(path) is True``: the run that owns ``path`` is known to be gone (held and cannot-tell are both ``False``)."""
    return lease_state(path) is True


_owner_host = owner_host  # ``stale`` has a parameter of that name


def _past(label: str, now: float) -> bool:
    try:
        return float(label) < now
    except ValueError:  # an unparseable expiry is not expired
        return False


def _verdict(r: Row, *, own_lock: str | None, reclaim_own: bool, owner_host: str, now: float, lease: Callable[[str], bool | None]) -> str | None:
    """``own_run`` | ``lease_free`` | ``expired`` | ``renewable`` (past its label, and only its real deadline can say more) | ``None``
    (kept) for one container: the rules of ``stale``."""
    mine = _owner_host(r.owner) == owner_host
    if own_lock and r.lease == own_lock and mine:
        return "own_run" if reclaim_own else None
    if mine and is_run_lock(r.lease):
        state = lease(r.lease)
        if state is not None:
            return "lease_free" if state else None
    if not _past(r.expires, now):
        return None
    return "renewable" if r.renewable else "expired"


def pending_deadlines(found: Sequence[Row], *, own_lock: str | None, reclaim_own: bool, owner_host: str, now: float,
                      lease_free: Callable[[str], bool | None]) -> list[Row]:
    """The renewable containers ``stale`` cannot judge from their labels: past ``crucible.expires``, owned by no run this host can see
    live. A sweep reads each one's real deadline (``_read_deadlines``) and hands the answers to ``stale``."""
    return [r for r in found if r.owner and _verdict(r, own_lock=own_lock, reclaim_own=reclaim_own, owner_host=owner_host, now=now, lease=lease_free) == "renewable"]


def stale(found: Sequence[Row], *, own_lock: str | None, reclaim_own: bool, owner_host: str, now: float,
          lease_free: Callable[[str], bool | None], deadlines: Mapping[str, float | None] | None = None) -> list[tuple[str, str]]:
    """``[(name, why)]`` of the containers a sweep removes; pure, so the rules are a table. Per container, first match wins
    (``lease_free(path)`` is ``lease_state``: ``True`` free or gone, ``False`` held, ``None`` cannot tell):

    1. its lease is ``own_lock`` and its owner is this host: ``own_run`` when ``reclaim_own`` (a Runner start: no attempt of this run
       is live yet), else kept (a live run's own container, whatever its label says). A container of another host that carries the
       same lease *string* (two machines with one checkout path and a fixed run id) is not this run's: it goes on to rule 4;
    2. it has a ``run.lock`` lease and its owner is this host: ``lease_free`` when that lock is free or gone, kept when it is held
       (the run is live, even past the label: a renewed container outlives its expiry label); a lease that cannot be told goes on;
    3. not past ``crucible.expires`` (an unparseable value is not expired): kept;
    4. past it, not renewable: ``expired`` (its label is exact);
    5. past it, renewable: its label is a floor (renewal pushes the real deadline past it), so the label alone never removes it. Its
       real deadline (``deadlines[name]``: the sweep read ``/crucible-ctl/deadline``) decides: ``expired`` once that is past, kept while
       it is not; one that could not be read (absent from ``deadlines`` or ``None``: a paused or unreachable container) is
       ``expired`` only ``UNREADABLE_GRACE_S`` past its label, so a live holder on another host (or outside a Runner) keeps its
       container.

    A container without ``crucible.owner`` is never touched. Never pid liveness (see ``lease_state``)."""
    out: list[tuple[str, str]] = []
    real = deadlines or {}
    for r in found:
        if not r.owner:
            continue
        v = _verdict(r, own_lock=own_lock, reclaim_own=reclaim_own, owner_host=owner_host, now=now, lease=lease_free)
        if v == "renewable":
            d = real.get(r.name)
            v = "expired" if (d < now if d is not None else float(r.expires) < now - UNREADABLE_GRACE_S) else None
        if v:
            out.append((r.name, v))
    return out


@dataclass
class SweepReport:
    """What one sweep did: the hosts it looked at, what it removed (``{name, host, why}``), what it could not remove (``{name, host,
    error}``), how many containers it kept and, per host (``''`` is the local daemon), why it could not list."""

    hosts: list[str] = field(default_factory=list)
    removed: list[dict[str, str]] = field(default_factory=list)
    failed: list[dict[str, str]] = field(default_factory=list)
    kept: int = 0
    errors: dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {"hosts": list(self.hosts), "removed": list(self.removed), "failed": list(self.failed), "kept": self.kept, "errors": dict(self.errors)}


async def _call(seam: Callable[..., Awaitable[ExecOut]] | None, args: list[str], host: str | None, timeout_s: float) -> ExecOut:
    fn = seam if seam is not None else docker  # resolved at call time: a test patches ``lifecycle.docker``
    return await fn(args, host, timeout_s=timeout_s)


def _why(out: ExecOut, what: str, timeout_s: float) -> str:
    if out.timed_out:
        return f"{what} ran past {timeout_s:g}s"
    return (out.stderr or out.stdout or f"rc {out.rc}").strip()[-300:]


async def _read_deadlines(pending: Sequence[Row], host: str | None, seam: Callable[..., Awaitable[ExecOut]] | None) -> dict[str, float | None]:
    """The real deadline of each renewable container (``docker exec <c> cat /crucible-ctl/deadline``, as root: the file is the life
    loop's, written by ``DockerSandbox.renew``), concurrently and bounded; ``None`` when it could not be read (the container is
    paused, gone or unreachable, or the file was never written). Never raises (a ``CancelledError`` excepted)."""
    gate = asyncio.Semaphore(8)

    async def one(r: Row) -> tuple[str, float | None]:
        async with gate:
            try:
                out = await _call(seam, ["exec", "--user", "0", r.name, "cat", f"{CTL_DIR}/deadline"], host, PROBE_TIMEOUT_S)
                return r.name, (float(out.stdout.strip()) if out.ok else None)
            except Exception:  # noqa: BLE001 -- an unreadable deadline is "cannot tell", never a failed sweep
                return r.name, None

    return dict(await asyncio.gather(*(one(r) for r in pending)))


async def _sweep_host(key: str, rep: SweepReport, *, own_lock: str | None, reclaim_own: bool, seam: Callable[..., Awaitable[ExecOut]] | None) -> None:
    host = key or None
    listing = ["ps", "-a", "--filter", f"label={OWNER_LABEL}", "--format"]
    out = await _call(seam, [*listing, PS_FORMAT], host, LIST_TIMEOUT_S)
    if not out.ok:
        rep.errors[key] = _why(out, "docker ps", LIST_TIMEOUT_S)
        return
    found = rows(out.stdout)
    states: dict[str, bool | None] = {}

    def lease(path: str) -> bool | None:  # one probe per lease, however many containers carry it (and the pending pass reuses it)
        if path not in states:
            states[path] = lease_state(path)
        return states[path]

    rules = {"own_lock": own_lock, "reclaim_own": reclaim_own, "owner_host": socket.gethostname(), "now": time.time(), "lease_free": lease}
    pending = pending_deadlines(found, **rules)  # renewable and past their label: their real deadline decides (review round 1)
    deadlines = await _read_deadlines(pending, host, seam) if pending else {}
    victims = stale(found, **rules, deadlines=deadlines)
    rep.kept += len(found) - len(victims)
    if not victims:
        return
    names, why = [n for n, _ in victims], dict(victims)
    rm = await _call(seam, ["rm", "-f", *names], host, CLOSE_TIMEOUT_S)
    left = await _call(seam, [*listing, "{{.Names}}"], host, LIST_TIMEOUT_S)  # rc-checked by looking again: a name still listed was not removed
    if left.ok:
        still = set(left.stdout.split())
        err = _why(rm, "docker rm -f", CLOSE_TIMEOUT_S) if not rm.ok else "still listed after docker rm -f"
    else:
        still = set() if rm.ok else set(names)
        err = _why(rm, "docker rm -f", CLOSE_TIMEOUT_S)
    for n in names:
        if n in still:
            rep.failed.append({"name": n, "host": key, "error": err})
        else:
            rep.removed.append({"name": n, "host": key, "why": why[n]})


async def sweep(hosts: Iterable[str], *, own_lock: str | None, reclaim_own: bool,
                docker: Callable[..., Awaitable[ExecOut]] | None = None) -> SweepReport:
    """Remove the orphans (``stale``) among the ``crucible.owner`` containers of exactly the docker hosts given (``''`` is the local
    daemon; the Runner adds it when a docker CLI exists). ``docker`` is the seam (``lifecycle.docker``'s signature). Every docker
    call goes through ``world.exec._run_subprocess`` and is bounded; a host that cannot be listed is ``errors[host]`` and the others
    are still swept. Never raises (a ``CancelledError`` excepted: the Runner's own start was cancelled)."""
    distinct = sorted({h or "" for h in hosts})
    rep = SweepReport(hosts=distinct)
    for key in distinct:
        try:
            await _sweep_host(key, rep, own_lock=own_lock, reclaim_own=reclaim_own, seam=docker)
        except Exception as e:  # noqa: BLE001 -- a sweep never fails the run it protects
            rep.errors[key] = f"{type(e).__name__}: {e}"[:300]
    return rep
