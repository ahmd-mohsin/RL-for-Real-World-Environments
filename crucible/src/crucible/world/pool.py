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
  SandboxSpec asks for ``pool > 0``; one process-wide pool per signature (``pool_for``).
"""

from __future__ import annotations

import asyncio
import contextlib
import time
from typing import Any

from crucible.core.schema.world import SandboxSpec, WorldSpec
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
                ppid = int(st[st.rindex(")") + 2:].split()[1])
                cmd = open(f"/proc/{d}/cmdline", "rb").read().replace(b"\0", b" ").decode(errors="replace").strip()
                out[int(d)] = (ppid, cmd)
            except (OSError, ValueError):
                pass
    return out
def mine(ps):
    # this exec session: the ancestors of this process up to the one docker exec started (ppid 0), and their descendants
    me, root = os.getpid(), os.getpid()
    while root in ps and ps[root][0] not in (0, 1):
        root = ps[root][0]
    keep, grew = {root, me}, True
    while grew:
        grew = False
        for p, (pp, _) in ps.items():
            if pp in keep and p not in keep:
                keep.add(p); grew = True
    return keep
def strangers():
    ps = procs()
    keep = mine(ps) | {1} | {p for p, (pp, c) in ps.items() if pp == 1 and c == "sleep infinity"}
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
    a = DockerSandbox(sb, spec, pids_limit=pids_limit, memory=memory).argv()
    i = a.index("--name")
    return (sb.docker_host or sb.gpu_host or "", tuple(x for k, x in enumerate(a) if k not in (i, i + 1) and not str(x).startswith("crucible.world=")))


class SandboxPool:
    def __init__(self, sb: SandboxSpec, spec: WorldSpec, *, size: int = 4, reuse: bool = False, pids_limit: int | None = None, memory: str | None = None):
        self.sb, self.spec, self.size, self.reuse = sb, spec, size, reuse
        self.pids_limit, self.memory = pids_limit, memory
        self._ready: asyncio.Queue[DockerSandbox] = asyncio.Queue()
        self._starting = 0
        self._tasks: set[asyncio.Task] = set()
        self.stats = {"hits": 0, "misses": 0, "destroyed": 0, "reused": 0, "dirty": 0}

    def _new(self) -> DockerSandbox:
        return DockerSandbox(self.sb, self.spec, pids_limit=self.pids_limit, memory=self.memory)

    async def _start_one(self) -> None:
        sbx = self._new()
        try:
            await sbx.start()
            await self._ready.put(sbx)
        finally:
            self._starting -= 1

    def _refill(self) -> None:
        while self._ready.qsize() + self._starting < self.size:
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
        try:
            sbx = self._ready.get_nowait()
            self.stats["hits"] += 1
        except asyncio.QueueEmpty:
            self.stats["misses"] += 1
            sbx = self._new()
            await sbx.start()
        self._refill()
        return sbx

    async def release(self, sbx: DockerSandbox) -> None:
        if self.reuse and await self._clean(sbx) and self._ready.qsize() + self._starting < self.size:
            self.stats["reused"] += 1
            await self._ready.put(sbx)
            return
        self.stats["destroyed"] += 1
        await sbx.close()
        self._refill()

    async def _clean(self, sbx: DockerSandbox) -> bool:
        wd = self.sb.workdir
        await sbx.exec.run(HYGIENE.replace("{workdir}", wd), timeout_s=60, user="0")
        out = await sbx.exec.run(VERIFY.replace("{workdir}", wd), timeout_s=60, user="0")
        ok = out.ok and "procs=0 files=0 users=0" in out.stdout
        if not ok:
            self.stats["dirty"] += 1
        return ok

    async def close(self) -> None:
        for t in list(self._tasks):
            with contextlib.suppress(Exception):
                await t
        while not self._ready.empty():
            await self._ready.get_nowait().close()


_POOLS: dict[tuple, SandboxPool] = {}


def pool_for(sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None) -> SandboxPool:
    key = signature(sb, spec, pids_limit=pids_limit, memory=memory)
    if key not in _POOLS:
        _POOLS[key] = SandboxPool(sb, spec, size=sb.pool, reuse=sb.pool_reuse, pids_limit=pids_limit, memory=memory)
    return _POOLS[key]


async def close_pools() -> None:
    for p in list(_POOLS.values()):
        await p.close()
    _POOLS.clear()


class PooledSandbox:
    """A ``Sandbox`` backed by the warm pool: ``start`` acquires a pre-started container, ``close`` releases it."""

    def __init__(self, pool: SandboxPool):
        self.pool = pool
        self._sbx: DockerSandbox | None = None
        self.exec: Exec = DockerExec("unstarted")
        self.gpus = pool._new().gpus
        self.description = f"pooled {pool._new().description}"
        self.acquire_s: float | None = None

    @property
    def image_id(self) -> Any:
        return self._sbx.image_id if self._sbx else None

    async def start(self) -> None:
        t0 = time.perf_counter()
        self._sbx = await self.pool.acquire()
        self.acquire_s = time.perf_counter() - t0
        self.exec = self._sbx.exec

    async def pause(self) -> None:
        await self._sbx.pause()

    async def resume(self) -> None:
        await self._sbx.resume()

    async def close(self) -> None:
        if self._sbx is not None:
            await self.pool.release(self._sbx)
            self._sbx = None
