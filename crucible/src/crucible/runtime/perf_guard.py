"""Dispatch-side guard for performance grades (Appendix B.5 row 7; research m9-m10/02 §3.3 row 7).

The grader's own ``LoadGuard`` and round-spread check (``verify/runtime/patterns/perf.py``) catch contamination
*inside* a grade. This module keeps contaminated grades from starting at all:

- **one perf grade per host at a time** (a per-host lock: two of our own grades on one GPU contaminate each other);
- **a host check before the grade:** foreign GPU compute processes holding memory, GPU utilization, CPU load per
  core. A busy host is waited on; if it never quiets within ``wait_s`` the grade does not run and the attempt is
  ``crucible:node_contention`` (infra, masked; re-graded later from the stored artifacts, never re-run by the agent);
- samples are returned for the attempt's evidence.

``probe`` runs a shell command on the host (locally, or over ssh/docker for a lab node) and returns ``(rc, stdout)``.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import time
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field

from crucible.core.schema.run import Failure
from crucible.runtime.infra import StageFailure

Probe = Callable[[str], tuple[int, str]]


def local_probe(cmd: str) -> tuple[int, str]:
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=30, check=False)
    return r.returncode, r.stdout


def sample(probe: Probe) -> dict:
    s: dict = {"gpu_util": None, "foreign": [], "foreign_mib": 0, "load_per_core": None}
    rc, out = probe("nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null")
    vals = [int(x) for x in out.split() if x.strip().isdigit()] if rc == 0 else []
    s["gpu_util"] = max(vals) if vals else None
    rc, out = probe("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null")
    if rc == 0:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) == 2 and parts[0].isdigit() and parts[1].isdigit():
                s["foreign"].append({"pid": int(parts[0]), "mib": int(parts[1])})
        s["foreign_mib"] = sum(p["mib"] for p in s["foreign"])
    rc, out = probe("cat /proc/loadavg 2>/dev/null || sysctl -n vm.loadavg; nproc 2>/dev/null || sysctl -n hw.ncpu")
    nums = [x for x in out.replace("{", " ").replace("}", " ").split()]
    try:
        cores = int(nums[-1])
        s["load_per_core"] = round(float(nums[0]) / max(cores, 1), 3)
    except (IndexError, ValueError):
        pass
    return s


@dataclass
class PerfGuard:
    probe: Probe = local_probe
    max_gpu_util: int = 20
    max_foreign_mib: int = 2048
    max_load_per_core: float = 0.5
    wait_s: float = 900.0
    poll_s: float = 15.0
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep
    _locks: dict[str, asyncio.Lock] = field(default_factory=dict)

    def busy(self, s: dict) -> list[str]:
        why = []
        if s["gpu_util"] is not None and s["gpu_util"] > self.max_gpu_util:
            why.append(f"gpu {s['gpu_util']}%")
        if s["foreign_mib"] > self.max_foreign_mib:
            why.append(f"foreign GPU memory {s['foreign_mib']} MiB in {len(s['foreign'])} processes")
        if s["load_per_core"] is not None and s["load_per_core"] > self.max_load_per_core:
            why.append(f"load/core {s['load_per_core']}")
        return why

    @asynccontextmanager
    async def slot(self, host: str = "local"):
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            t0 = time.monotonic()
            samples = []
            while True:
                s = await asyncio.to_thread(sample, self.probe)
                samples.append(s)
                why = self.busy(s)
                if not why:
                    break
                if time.monotonic() - t0 >= self.wait_s:
                    raise StageFailure(Failure(kind="crucible:node_contention", stage="verification",
                                               reason=f"host {host} never quiet in {self.wait_s:.0f}s: {'; '.join(why)}",
                                               detail="perf_guard"))
                await self.sleep(self.poll_s)
            yield {"host": host, "waited_s": round(time.monotonic() - t0, 1), "samples": samples[-3:], "pid": os.getpid()}
