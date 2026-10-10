"""Environment workers: environment code off the control loop (M12 checklist #8; research m11-m12/06 §3 "environments
always off the GPU pool", MiniMax Forge / slime rollout servers, AReaL's decoupled env workers).

The runtime's main event loop runs the scheduler, the engine I/O and the trainer hand-off. Environment code is
arbitrary: a sandbox RPC, a pure-Python simulator (PortSim's tools are synchronous), a grader that runs tests. Run on
the main loop, one blocking step would stall every rollout's model calls and the sample barrier.

``EnvWorkerPool(workers=N)``: N threads, each with its own event loop. Work is placed on the least-loaded worker
(in-flight count; ties → lowest index):

- ``run(fn, *args)``: one environment coroutine (a token-env ``reset`` / ``step``) on a worker;
- ``run_episode(fn, *args, bridge=...)``: a whole World episode on a worker; its model calls go back to the main loop
  through ``bridge(async_fn)`` (``asyncio.run_coroutine_threadsafe``), so the engine pool and its abort / pause /
  version semantics stay on the main loop.

A blocked environment holds only its own worker's loop (and episodes co-resident on it), never the engines.
Accounting: ``wait_s`` = submission → start latency summed over submissions (an environment waiting for a worker:
the ``idle/env_wait_s`` bubble), ``busy_s`` = start → end (environment time), ``max_inflight`` per worker.
"""

from __future__ import annotations

import asyncio
import threading
import time
from collections.abc import Awaitable, Callable
from typing import Any


class _Worker:
    def __init__(self, i: int, name: str):
        self.i, self.inflight, self.max_inflight = i, 0, 0
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, name=f"{name}-{i}", daemon=True)
        self.thread.start()


class EnvWorkerPool:
    def __init__(self, workers: int = 4, *, name: str = "crucible-env"):
        if workers < 1:
            raise ValueError("workers must be ≥ 1 (0 = run environments inline on the event loop: no pool)")
        self.workers = [_Worker(i, name) for i in range(workers)]
        self.wait_s = 0.0
        self.busy_s = 0.0
        self.submitted = 0
        self._lock = threading.Lock()
        self._main: asyncio.AbstractEventLoop | None = None

    def _pick(self) -> _Worker:
        with self._lock:
            w = min(self.workers, key=lambda x: (x.inflight, x.i))
            w.inflight += 1
            w.max_inflight = max(w.max_inflight, w.inflight)
            return w

    async def _timed(self, fn: Callable[..., Awaitable[Any]], submitted: float, args: tuple, kw: dict) -> Any:
        start = time.monotonic()
        with self._lock:
            self.wait_s += start - submitted
        try:
            return await fn(*args, **kw)
        finally:
            with self._lock:
                self.busy_s += time.monotonic() - start

    async def run(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kw: Any) -> Any:
        """Await ``fn(*args, **kw)`` executed on a worker loop (called from the main loop)."""
        self._main = self._main or asyncio.get_running_loop()
        w = self._pick()
        self.submitted += 1
        fut = asyncio.run_coroutine_threadsafe(self._timed(fn, time.monotonic(), args, kw), w.loop)
        try:
            return await asyncio.wrap_future(fut)
        finally:
            with self._lock:
                w.inflight -= 1

    def bridge(self, async_fn: Callable[..., Awaitable[Any]]) -> Callable[..., Awaitable[Any]]:
        """Wrap a main-loop coroutine function so code on a worker loop can await it (model calls from an episode)."""
        main = self._main or asyncio.get_running_loop()
        self._main = main

        async def call(*a: Any, **k: Any) -> Any:
            if asyncio.get_running_loop() is main:
                return await async_fn(*a, **k)
            return await asyncio.wrap_future(asyncio.run_coroutine_threadsafe(async_fn(*a, **k), main))

        return call

    async def run_episode(self, fn: Callable[..., Awaitable[Any]], *args: Any, **kw: Any) -> Any:
        return await self.run(fn, *args, **kw)

    def close(self) -> None:
        for w in self.workers:
            w.loop.call_soon_threadsafe(w.loop.stop)
        for w in self.workers:
            w.thread.join(timeout=5)
