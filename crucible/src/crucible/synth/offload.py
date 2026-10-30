"""Process-parallel deterministic falsifiers and concurrent agent falsifiers (``synth.batch``).

**Why.** A finance_ap World runs in the calling process (``CrucibleWorldAdapter``): one round of the deterministic
classes on a dense task is about 12 s of CPU, almost all of it MUT's mutants, and while it runs the event loop serves no
model call. With eight candidates refining at once the CPU work serialises and every model call waits behind it.

**``FalsifyPool``** runs ``run_classes(task, cheap())`` for one task version in a worker process (``spawn``; the worker
builds its own World from the task's ``WorldRef`` and its own ``FalsifyContext``) and returns the class results as JSON.
``falsifiers()`` gives one ``Offloaded`` stand-in per class for ``cegis.refine``: the first class asked for a task
version starts the job, every class of that version reads its own result from it. The stand-in hands ``falsify`` the
worker's findings as they are (probes, outcomes, labels, already finalised), so the certificate counts exactly what an
in-process run counts; ``replay`` (the archive's regressions) runs in-process on the real class. MUT stays wrapped in
``Adjudicated`` by the pipeline, so its suspects are still judged by the panel, in the parent.

**``ConcurrentAgents``** wraps ALT or ATK: its ``propose`` runs every agent's attempt at once, each on its own
``FalsifyContext`` (its own World: one World per context is reset before every run, so two agents may not share one),
and ``execute`` returns the stored outcome. Labels, replay and the archive identity are the wrapped class's.
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from crucible.synth.falsify import cheap
from crucible.synth.falsify.base import BaseFalsifier, ClassResult, Counterexample, FalsifyContext, Finding, Outcome, Probe, run_classes
from crucible.synth.task import Task


def _job(task_json: str, max_mutants: int, names: Sequence[str]) -> str:
    """Worker side: every named class of ``cheap()`` on one task version, in the scheduled order (REP first)."""

    async def go() -> str:
        task = Task.model_validate_json(task_json)
        ctx = FalsifyContext(max_mutants=max_mutants)
        t0 = time.monotonic()
        try:
            res = await run_classes(task, [f for f in cheap() if f.name in set(names)], ctx)
        finally:
            await ctx.close()
        return json.dumps({"results": [r.model_dump(mode="json") for r in res], "cpu_s": round(time.monotonic() - t0, 3)})

    return asyncio.run(go())


class FalsifyPool:
    """See the module docstring. ``busy_s`` sums the workers' measured job times (the CPU the parent did not spend)."""

    def __init__(self, workers: int, *, max_mutants: int = 240, names: Sequence[str] | None = None):
        self.workers = int(workers)
        self.max_mutants = max_mutants
        self.names = tuple(names or [f.name for f in cheap()])
        self.ex = ProcessPoolExecutor(max_workers=self.workers, mp_context=mp.get_context("spawn"))
        self._jobs: dict[tuple[str, str], asyncio.Future[str]] = {}
        self.jobs = 0
        self.busy_s = 0.0

    async def results(self, task: Task) -> dict[str, ClassResult]:
        k = (task.id, task.digest())
        if k not in self._jobs:
            loop = asyncio.get_running_loop()
            fut = loop.run_in_executor(self.ex, _job, task.model_dump_json(), self.max_mutants, list(self.names))
            fut.add_done_callback(self._count)
            self._jobs[k] = fut  # type: ignore[assignment]
            self.jobs += 1
        raw = json.loads(await self._jobs[k])
        return {r["cls"]: ClassResult.model_validate(r) for r in raw["results"]}

    def _count(self, fut: Any) -> None:
        if not fut.cancelled() and fut.exception() is None:
            self.busy_s += float(json.loads(fut.result()).get("cpu_s") or 0.0)

    def forget(self, task_id: str) -> None:
        """Drop the finished jobs of a task (its refine is done)."""
        for k in [k for k in self._jobs if k[0] == task_id]:
            self._jobs.pop(k, None)

    def falsifiers(self) -> list[BaseFalsifier]:
        return [Offloaded(f, self) for f in cheap() if f.name in self.names]

    def close(self) -> None:
        self.ex.shutdown(wait=False, cancel_futures=True)


class Offloaded(BaseFalsifier):
    """One class whose propose/execute/label/finalize are the worker's (module docstring)."""

    def __init__(self, inner: BaseFalsifier, pool: FalsifyPool):
        self.inner, self.pool = inner, pool
        self.name = inner.name  # type: ignore[misc]
        self._f: dict[int, Finding] = {}

    async def _res(self, task: Task) -> ClassResult:
        r = (await self.pool.results(task)).get(self.name)
        return r if r is not None else ClassResult(cls=self.name, applicable=False, reason="not run by the worker")

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        r = await self._res(task)
        return None if r.applicable else (r.reason or "not applicable")

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        r = await self._res(task)
        for f in r.findings:
            self._f[id(f.probe)] = f
        return [f.probe for f in r.findings]

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        return self._f[id(probe)].outcome

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Any, str | None]:
        f = self._f.get(id(probe))
        return (f.label, f.defect) if f is not None else self.inner.label(probe, outcome)

    def finalize(self, findings: list[Finding]) -> list[Finding]:
        return findings  # the worker ran the class's own finalize

    async def replay(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool:
        return await self.inner.replay(cx, task, ctx)


class ConcurrentAgents(BaseFalsifier):
    """ALT or ATK with every agent's attempt run at once (module docstring)."""

    def __init__(self, inner: BaseFalsifier):
        self.inner = inner
        self.name = inner.name  # type: ignore[misc]
        self._out: dict[int, Outcome] = {}

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        f = getattr(self.inner, "applies", None)
        return await f(task, ctx) if f is not None else None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        probes = await self.inner.propose(task, ctx)

        async def one(p: Probe) -> Outcome:
            sub = FalsifyContext(ctx.make_world, exec_registry=ctx.exec_registry, deadline_s=ctx.deadline_s, max_mutants=ctx.max_mutants, seed=ctx.seed)
            try:
                return await self.inner.execute(p, task, sub)
            finally:
                await sub.close()

        outs = await asyncio.gather(*(one(p) for p in probes))
        for p, o in zip(probes, outs, strict=True):
            self._out[id(p)] = o
        return probes

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        return self._out.pop(id(probe))

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Any, str | None]:
        return self.inner.label(probe, outcome)

    async def replay(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool:
        return await self.inner.replay(cx, task, ctx)


__all__ = ["ConcurrentAgents", "FalsifyPool", "Offloaded"]
