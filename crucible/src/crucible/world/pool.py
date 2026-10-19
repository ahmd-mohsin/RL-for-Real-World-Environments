"""A warm pool of pre-started sandbox containers (m13 audit 04, S10; SkyPilot / Cua / Smol warm pools as PATTERN).

``docker run`` dominates a sandboxed World's ``provision`` at rollout scale. A ``SandboxPool`` keeps ``size`` containers
per launch signature (image + every hardening flag: network, pids, GPUs, cpus, memory, capabilities) started ahead of
time, and refills in the background as they are taken.

- **Fresh by default** (``reuse=False``): a released container is destroyed and a new one started in its place, so no
  rollout can inherit another's processes, files or users (the guard creates a demoted policy user per World).
- **Reuse** (opt-in): a released container is reset by hygiene (kill every process but init and the keeper, empty
  ``/tmp``, ``/var/tmp``, ``/dev/shm``, home directories, the workdir, System V IPC, and remove users the guard added),
  then *verified*; a container that does not verify clean is destroyed, never pooled.
- ``PooledSandbox`` is a ``Sandbox`` (``start`` acquires, ``close`` releases) that ``open_sandbox`` returns when the
  SandboxSpec asks for ``pool > 0``; one pool per (run, signature) (``pool_for``).

**Lifecycle (M10 lab audit R-6, ``crucible.world.lifecycle``).** Pooled containers belong to the *run*, never to an attempt: a
pool builds its containers with the run's scope (``Scope.run_scope()``), not the attempt scope its refill tasks copy from the
contextvar, so an attempt's reclaim never removes a warm container. They carry the run labels (``crucible.run``,
``crucible.lease``, owner, expiry; no ``crucible.attempt``) and the kernel's renewable life loop, kept for twice the scope's ttl.
A container handed out is renewed if its remaining life is shorter than the acquiring scope's ttl (the attempt's scope then renews
it while the attempt lives: ``Scope.borrowed``); one whose renewal fails is dead (its keeper ended) and is replaced. Pools are keyed
by (lease, signature) (``_owner``: the run's ``run.lock`` path, the run id where there is no lease; a run id is unique only within a
root, the lease is unique) so ``close_pools(owner=lease)`` at the end of every ``Runner.run`` / ``regrade`` closes exactly that run's
warm and handed-out containers, in bounded time (``POOL_CLOSE_WAIT_S`` for refills still in ``docker run``: they remove themselves
when it returns); the exit reaper and the TTL cover a runner that never gets there.
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world import lifecycle
from crucible.world.exec import DockerExec, Exec
from crucible.world.sandbox import DockerSandbox

_PROCS = r"""
import os, signal
def procs():
    out = {}
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                st = open(f"/proc/{d}/stat").read()
                f = st[st.rindex(")") + 2:].split()
                ppid, start = int(f[1]), int(f[19])  # field 4 and field 22 (clock ticks after boot)
                cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
                out[int(d)] = (ppid, start, cmd)
            except (OSError, ValueError, IndexError):
                pass
    return out
def grow(keep, ps):
    grew = True
    while grew:
        grew = False
        for p, (pp, _, _) in ps.items():
            if pp in keep and p not in keep:
                keep.add(p); grew = True
    return keep
def mine(ps):
    # this exec session: the ancestors of this process up to the one docker exec started (ppid 0), and their descendants
    me, root = os.getpid(), os.getpid()
    while root in ps and ps[root][0] not in (0, 1):
        root = ps[root][0]
    return grow({root, me}, ps)
def keeper(ps):
    # M10 lab audit R-6: the keeper is no longer `sleep infinity` (it is `sleep <ttl>` or the renewable life loop `sh -c ...`), and
    # matching any `sleep N` would spare a planted orphan named so. It is the earliest-started child of PID 1 with a keeper form,
    # plus its descendants (the life loop's `sleep 2`); an orphan the policy planted starts later.
    kids = sorted((st, p) for p, (pp, st, c) in ps.items() if pp == 1 and (c.startswith("sleep ") or c.startswith("sh -c ")))
    return grow({kids[0][1]}, ps) if kids else set()
def strangers():
    ps = procs()
    keep = mine(ps) | {1} | keeper(ps)
    return [p for p in ps if p not in keep]
"""

HYGIENE = "python3 - <<'PY'\n" + _PROCS + r"""
import glob, os, shutil, signal, subprocess, time
for _ in range(3):
    for p in strangers():
        try: os.kill(p, signal.SIGKILL)
        except OSError: pass
    time.sleep(0.2)
for d in ("/tmp", "/var/tmp", "/dev/shm", "{workdir}"):
    for x in glob.glob(d + "/*") + glob.glob(d + "/.[!.]*"):
        shutil.rmtree(x, ignore_errors=True) if os.path.isdir(x) and not os.path.islink(x) else os.unlink(x)
for line in open("/etc/passwd").read().splitlines():
    f = line.split(":")
    if len(f) > 3 and 1000 <= int(f[2]) < 60000:
        subprocess.run(["userdel", "-f", "-r", f[0]], capture_output=True)
for h in glob.glob("/home/*"):
    shutil.rmtree(h, ignore_errors=True)
subprocess.run("command -v ipcrm >/dev/null && ipcrm -a", shell=True, capture_output=True)
os.makedirs("{workdir}", exist_ok=True)
PY
"""

VERIFY = "python3 - <<'PY'\n" + _PROCS + r"""
import glob
files = sum(len(glob.glob(d + "/*") + glob.glob(d + "/.[!.]*")) for d in ("/tmp", "/var/tmp", "/dev/shm", "{workdir}"))
users = sum(1 for l in open("/etc/passwd").read().splitlines() if len(l.split(":")) > 3 and 1000 <= int(l.split(":")[2]) < 60000)
print(f"procs={len(strangers())} files={files} users={users}")
PY
"""


def signature(sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None, memory: str | None) -> tuple:
    """Every launch flag that decides what a container is, minus what is per container or per owner: the name, the volatile labels
    (``crucible.world`` and the R-6 run, owner, expiry, attempt, lease and renewable labels), the control tmpfs and the keeper
    command (``--entrypoint '' <image>`` is where the hardening flags end)."""
    a = DockerSandbox(sb, spec, pids_limit=pids_limit, memory=memory, scope=lifecycle.Scope(), renewable=True).argv()
    a = a[: a.index("--entrypoint") + 3]
    drop: set[int] = set()
    for k, x in enumerate(a):
        if x == "--name" or (x == "--tmpfs" and str(a[k + 1]).startswith(lifecycle.CTL_DIR)):
            drop |= {k, k + 1}
        elif x == "--label" and str(a[k + 1]).startswith(("crucible.world=", *(f"{lb}=" for lb in lifecycle.VOLATILE_LABELS))):
            drop |= {k, k + 1}
    return (sb.docker_host or sb.gpu_host or "", tuple(x for k, x in enumerate(a) if k not in drop))


class SandboxPool:
    def __init__(self, sb: SandboxSpec, spec: WorldSpec, *, size: int = 4, reuse: bool = False, pids_limit: int | None = None, memory: str | None = None):
        self.sb, self.spec, self.size, self.reuse = sb, spec, size, reuse
        self.pids_limit, self.memory = pids_limit, memory
        self.scope = lifecycle.current().run_scope()  # the run's, never the attempt's (see the module docstring)
        self.lifetime_s = 2 * self.scope.ttl_s
        self._ready: asyncio.Queue[DockerSandbox] = asyncio.Queue()
        self._handed: set[DockerSandbox] = set()  # taken from ``_ready`` or started for a caller, not yet released
        self._starting = 0
        self._out = 0  # handed out and not yet released: reuse mode counts them, they come back
        self._tasks: set[asyncio.Task] = set()
        self.closed = False
        self.stats = {"hits": 0, "misses": 0, "destroyed": 0, "reused": 0, "dirty": 0, "dead": 0}

    def _new(self) -> DockerSandbox:
        return DockerSandbox(self.sb, self.spec, pids_limit=self.pids_limit, memory=self.memory, scope=self.scope, renewable=True, lifetime_s=self.lifetime_s)

    async def _start_one(self) -> None:
        sbx = self._new()
        try:
            await sbx.start()
            if self.closed:  # the pool was closed while this one started
                await sbx.close()
            else:
                await self._ready.put(sbx)
        finally:
            self._starting -= 1

    def _refill(self) -> None:
        # fresh mode: a handed-out container never returns, so refill as soon as one leaves; reuse mode: it returns cleaned,
        # so only a destroyed one is replaced (found live: refilling on every acquire filled the pool before the release,
        # and every clean container was destroyed — reuse mode never reused)
        if self.closed:
            return
        held = self._out if self.reuse else 0
        while self._ready.qsize() + self._starting + held < self.size:
            self._starting += 1
            t = asyncio.create_task(self._start_one())
            self._tasks.add(t)
            t.add_done_callback(self._tasks.discard)

    async def warm(self) -> None:
        """Fill the pool and wait until it is full."""
        self._refill()
        while self._ready.qsize() < self.size:
            await asyncio.sleep(0.05)

    async def acquire(self) -> DockerSandbox:
        """A warm container, or a cold one. M10 lab audit R-6, cancel-safe: a container taken from the queue is in ``_handed`` before
        any await, so a cancel during its renewal closes it (it is in nobody's hands); one whose renewal fails is dead (its keeper
        ended: the TTL or a daemon restart) and is discarded and replaced."""
        sbx: DockerSandbox | None = None
        while sbx is None:
            try:
                cand = self._ready.get_nowait()
            except asyncio.QueueEmpty:
                break
            self._handed.add(cand)
            need = lifecycle.current().ttl_s
            if (cand.deadline or 0.0) - time.time() < need:  # not enough life left for the acquiring scope: renew first
                try:
                    ok = await cand.renew(max(need, self.scope.ttl_s))
                except BaseException:
                    self._handed.discard(cand)
                    with contextlib.suppress(Exception):
                        await cand.close()
                    raise
                if not ok:
                    self.stats["dead"] += 1
                    self._handed.discard(cand)
                    with contextlib.suppress(Exception):
                        await cand.close()
                    continue
            self.stats["hits"] += 1
            sbx = cand
        if sbx is None:
            self.stats["misses"] += 1
            sbx = self._new()
            await sbx.start()  # shielded and self-cleaning on a cancel or a failure
            self._handed.add(sbx)
        self._out += 1
        self._refill()
        return sbx

    async def release(self, sbx: DockerSandbox) -> bool:
        """Give a container back. ``True`` when it was destroyed (the run scope recorded the removal), ``False`` when it went back to
        the pool (reuse mode, verified clean). It stays in ``_handed`` while it is cleaned, so ``close`` still reaches it."""
        self._out = max(0, self._out - 1)
        if not self.closed and self.reuse and await self._clean(sbx) and not self.closed and self._ready.qsize() + self._starting < self.size:
            self._handed.discard(sbx)
            self.stats["reused"] += 1
            await self._ready.put(sbx)
            return False
        self._handed.discard(sbx)
        self.stats["destroyed"] += 1
        with contextlib.suppress(Exception):
            await sbx.close()
        self._refill()
        return True

    async def _clean(self, sbx: DockerSandbox) -> bool:
        wd = self.sb.workdir
        await sbx.exec.run(HYGIENE.replace("{workdir}", wd), timeout_s=60, user="0")
        out = await sbx.exec.run(VERIFY.replace("{workdir}", wd), timeout_s=60, user="0")
        ok = out.ok and "procs=0 files=0 users=0" in out.stdout
        if not ok:
            self.stats["dirty"] += 1
        return ok

    async def close(self) -> dict:
        """End the pool: wait (at most ``POOL_CLOSE_WAIT_S``) for the containers still starting, remove the warm ones and the handed-out
        ones (their holders' attempts are over or ending), and report what the run scope recorded: ``{terminated, failed, unfinished}``."""
        self.closed = True
        unfinished = 0
        if self._tasks:
            tasks = list(self._tasks)
            done, pending = await asyncio.wait(tasks, timeout=lifecycle.POOL_CLOSE_WAIT_S)
            for t in done:
                if not t.cancelled():
                    t.exception()  # a failed start was already cleaned up by DockerSandbox.start
            # review round 1: a refill whose ``docker run`` is still going (a slow pull, a hung daemon) is not waited for past the bound:
            # ``_start_one`` removes its container when the run returns (the pool is closed), and the keeper's TTL and the next sweep
            # cover one that never does. The lease is released on time
            unfinished = len(pending)
        doomed: list[DockerSandbox] = []
        while not self._ready.empty():
            doomed.append(self._ready.get_nowait())
        doomed += list(self._handed)
        self._handed.clear()
        await asyncio.gather(*(s.close() for s in doomed), return_exceptions=True)
        rep = self.scope.report()
        return {"terminated": rep["terminated"], "failed": rep["failed"], "unfinished": unfinished}


_POOLS: dict[tuple, SandboxPool] = {}


def _owner(scope: lifecycle.Scope) -> str | None:
    """Who a pool belongs to: the run's lease (``Scope.lock``, an absolute ``run.lock`` path: unique across roots), else its run id.
    ``None`` outside a Runner: one pool per signature for the process."""
    return scope.lock or scope.run


def pool_for(sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None) -> SandboxPool:
    key = (_owner(lifecycle.current()), signature(sb, spec, pids_limit=pids_limit, memory=memory))
    if key not in _POOLS:
        _POOLS[key] = SandboxPool(sb, spec, size=sb.pool, reuse=sb.pool_reuse, pids_limit=pids_limit, memory=memory)
    return _POOLS[key]


async def close_pools(owner: str | None = None, *, run: str | None = None) -> dict:
    """Close and forget the pools of ``owner`` (a run's lease path, ``Runner`` passes its ``run.lock``) and, if given, of the run id
    ``run``; every pool when neither is given. Returns the merged ``{terminated, failed, unfinished}`` (``unfinished``: refills still in
    ``docker run`` after ``POOL_CLOSE_WAIT_S``, left to remove themselves). Called at the end of every ``Runner.run`` / ``regrade`` (M10
    lab audit R-6: it had no caller) and by tests. A run id is unique only within a root, so two Runners on different roots with one
    run id must not close each other's pools: ``Runner`` closes by lease. A closed pool's scope no longer journals (the run's lease is
    released after this)."""
    out: dict[str, Any] = {"terminated": [], "failed": [], "unfinished": 0}
    for key, pool in [(k, p) for k, p in _POOLS.items() if (owner is None or k[0] == owner) and (run is None or p.scope.run == run)]:
        if _POOLS.pop(key, None) is None:
            continue
        rep = await pool.close()
        pool.scope.event = None
        out["terminated"] += rep["terminated"]
        out["failed"] += rep["failed"]
        out["unfinished"] += rep["unfinished"]
    return out


class PooledSandbox:
    """A ``Sandbox`` backed by the warm pool: ``start`` acquires a pre-started container, ``close`` releases it."""

    def __init__(self, pool: SandboxPool):
        self.pool = pool
        self._sbx: DockerSandbox | None = None
        self._scope = lifecycle.current()
        self.exec: Exec = DockerExec("unstarted")
        self.gpus = pool._new().gpus
        self.description = f"pooled {pool._new().description}"
        self.acquire_s: float | None = None

    @property
    def image_id(self) -> Any:
        return self._sbx.image_id if self._sbx else None

    async def start(self) -> None:
        t0 = time.perf_counter()
        self._scope = lifecycle.current()
        self._sbx = await self.pool.acquire()
        self.acquire_s = time.perf_counter() - t0
        self.exec = self._sbx.exec
        # the acquiring attempt's scope renews it while the attempt lives, and never removes it: the pool owns it
        self._scope.borrowed(self._sbx.name, self._sbx.docker_host, deadline=self._sbx.deadline or 0.0, renew=self._sbx.renew)

    async def pause(self) -> None:
        await self._sbx.pause()

    async def resume(self) -> None:
        await self._sbx.resume()

    async def close(self) -> None:
        if self._sbx is not None:
            sbx = self._sbx
            destroyed = await self.pool.release(sbx)
            self._sbx = None
            self._scope.returned(sbx.name, destroyed=destroyed)
