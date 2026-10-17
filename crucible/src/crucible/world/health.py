"""Health checks, reset hooks and the run-serially lock (M1 item 6; Harbor ``HealthcheckConfig`` / ``run_healthcheck``,
TheAgentCompany ``init.sh`` resets and ``-j 1``, OpenEnv ``SUPPORTS_CONCURRENT_SESSIONS``).

- ``run_probe``: one shell command in the App's exec, or one HTTP request made from inside that exec (a container's
  ``localhost`` is the container's), with the stdlib only.
- ``run_healthcheck``: Docker HEALTHCHECK semantics. A start period during which failures do not count, then
  ``retries`` consecutive failures fail it. A failed check is ``HealthcheckError``: the environment is not ready,
  an infrastructure fault (masked), never the policy's.
- ``SerialLock``: an exclusive ``flock`` per world name under ``$CRUCIBLE_LOCK_DIR`` (default ``~/.crucible/locks``),
  held from ``World.start`` to ``World.close``, so episodes of a world over a shared live service never overlap on
  this host, across processes.
"""

from __future__ import annotations

import asyncio
import fcntl
import hashlib
import json
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from crucible.core.schema.world import HealthcheckSpec, ProbeSpec
    from crucible.world.exec import Exec


class HealthcheckError(RuntimeError):
    """An App did not become healthy (masked as ``crucible:backend_error``)."""


class ResetHookError(RuntimeError):
    """An App's reset hook failed: the episode cannot start from a known state (masked as ``crucible:backend_error``)."""


_HTTP = """
import json, urllib.request, urllib.error
def main(a):
    data = None if a.get("body") is None else json.dumps(a["body"]).encode()
    req = urllib.request.Request(a["url"], data=data, method=a["method"], headers={"content-type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(req, timeout=a["timeout"]) as r:
            return {"status": r.status}
    except urllib.error.HTTPError as e:
        return {"status": e.code}
    except Exception as e:
        return {"status": None, "error": f"{type(e).__name__}: {e}"}
"""


async def run_probe(exec_: Exec, probe: ProbeSpec) -> tuple[bool, str]:
    """(ok, detail) for one probe."""
    if probe.command:
        out = await exec_.run(probe.command, timeout_s=probe.timeout_s)
        return out.ok, (out.stderr or out.stdout)[-300:] if not out.ok else ""
    try:
        r = await exec_.run_python(_HTTP, {"url": probe.http, "method": probe.method, "body": probe.body, "timeout": probe.timeout_s})
    except Exception as e:  # noqa: BLE001 -- the probe itself could not run: not healthy
        return False, f"{type(e).__name__}: {e}"[:300]
    status = r.get("status")
    ok = status is not None and (status in probe.expect_status if probe.expect_status else 200 <= status < 400)
    return ok, "" if ok else f"status {status} {r.get('error', '')}".strip()


async def run_healthcheck(exec_: Exec, hc: HealthcheckSpec, *, what: str) -> dict[str, Any]:
    t0 = time.monotonic()
    start_end = t0 + hc.start_period_s
    fails, attempts, last = 0, 0, ""
    while True:
        attempts += 1
        ok, last = await run_probe(exec_, hc)
        if ok:
            return {"healthy": True, "attempts": attempts, "seconds": round(time.monotonic() - t0, 3)}
        if time.monotonic() < start_end:
            await asyncio.sleep(hc.start_interval_s)
            continue
        fails += 1
        if fails >= hc.retries:
            raise HealthcheckError(f"{what}: unhealthy after {hc.retries} consecutive failures ({attempts} attempts): {last}")
        await asyncio.sleep(hc.interval_s)


def lock_dir() -> Path:
    from crucible.core import env as knobs

    return knobs.get("CRUCIBLE_LOCK_DIR") or Path.home() / ".crucible" / "locks"


class SerialLock:
    """Exclusive per-world host lock (``WorldSpec.serial``)."""

    def __init__(self, world: str, *, wait_s: float = 3600.0, poll_s: float = 0.2):
        self.world, self.wait_s, self.poll_s = world, wait_s, poll_s
        self.path = lock_dir() / (hashlib.sha256(world.encode()).hexdigest()[:24] + ".lock")
        self._fd: int | None = None

    async def acquire(self) -> float:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600)
        t0 = time.monotonic()
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() - t0 > self.wait_s:
                    os.close(fd)
                    raise TimeoutError(f"world {self.world!r} is serial and stayed busy for {self.wait_s:.0f}s ({self.path})") from None
                await asyncio.sleep(self.poll_s)
        os.ftruncate(fd, 0)
        os.write(fd, json.dumps({"world": self.world, "pid": os.getpid(), "since": time.time()}).encode())
        self._fd = fd
        return time.monotonic() - t0

    def release(self) -> None:
        if self._fd is not None:
            try:
                fcntl.flock(self._fd, fcntl.LOCK_UN)
            finally:
                os.close(self._fd)
                self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None
