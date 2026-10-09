"""Process / toolchain State Adapter (``process``): the place where builds, tests, benchmarks and GPU
probes run. It holds no persistent state of its own — its "state" is what deterministic probes
report — so snapshots carry only a fingerprint.

Params: ``cwd`` (default working dir), ``env`` (extra env vars), ``probes`` (list of
deterministic commands whose outputs define the fingerprint; default none -> constant).
"""

from __future__ import annotations

import json
from typing import Any

from crucible.core.ids import sha256_text
from crucible.core.schema.state import QueryResult, StateDiff, StateRef
from crucible.world.state.base import StateAdapter


class ProcessAdapter(StateAdapter):
    kind = "process"
    OPS = {
        "run": "run a command: {cmd, cwd, timeout_s, env} -> exit code, stdout/stderr tails",
        "gpu": "nvidia-smi summary (name, memory, utilization, driver)",
        "which": "tool availability: {tools: [...]}",
    }

    async def fingerprint(self) -> str:
        probes = self.params.get("probes") or []
        outs = []
        for cmd in probes:
            r = await self.exec.run(cmd, cwd=self.params.get("cwd"), timeout_s=120)
            outs.append([cmd, r.rc, r.stdout])
        return sha256_text(json.dumps(outs, sort_keys=True))

    async def snapshot(self) -> StateRef:
        return StateRef(app=self.app, adapter=self.kind, fingerprint=await self.fingerprint(), blob=None, meta={"stateless": True})

    async def restore(self, ref: StateRef) -> None:
        return None  # stateless

    async def diff(self, a: StateRef, b: StateRef) -> StateDiff:
        return StateDiff(app=self.app, adapter=self.kind, before=a.fingerprint, after=b.fingerprint)

    async def q_run(self, cmd: str, cwd: str | None = None, timeout_s: float = 3600, env: dict[str, str] | None = None, tail: int = 20000) -> Any:
        r = await self.exec.run(cmd, cwd=cwd or self.params.get("cwd"), timeout_s=timeout_s, env={**self.params.get("env", {}), **(env or {})})
        return QueryResult(ok=r.ok, exit_code=r.rc, stdout=r.stdout[-tail:], stderr=r.stderr[-tail:], duration_s=r.duration_s, value={"timed_out": r.timed_out})

    async def q_gpu(self) -> Any:
        r = await self.exec.run(
            "nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version,compute_cap --format=csv,noheader,nounits",
            timeout_s=60,
        )
        if not r.ok:
            return QueryResult(ok=False, exit_code=r.rc, error="nvidia-smi failed", stdout=r.stdout[-2000:], stderr=r.stderr[-2000:])
        gpus = []
        for line in r.stdout.strip().splitlines():
            name, total, used, util, driver, cc = [x.strip() for x in line.split(",")]
            gpus.append({"name": name, "memory_total_mb": int(total), "memory_used_mb": int(used), "utilization_pct": int(util), "driver": driver, "compute_cap": cc})
        return {"gpus": gpus}

    async def q_which(self, tools: list[str]) -> Any:
        out = {}
        for t in tools:
            r = await self.exec.run(f"command -v {t}", timeout_s=30)
            out[t] = r.stdout.strip() if r.ok else None
        return out
