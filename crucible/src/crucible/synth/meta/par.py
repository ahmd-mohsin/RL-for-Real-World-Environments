"""Parallel evaluation of falsifier populations on planted units (CPU only; no model is called).

The loop's evaluations are CPU-bound Python (a full base-population pass over RSI-1's 125 dev plants took 435 s in one
process). ``ShardPool`` keeps ``workers`` persistent child processes (``spawn``); every unit (a plant or an anchor task)
lives on one shard, chosen by a stable hash of its id, so the per-seed class cache of that shard keeps every result it
computed: re-evaluating a unit under a population that only adds an operator runs only the new operator.

Protocol (one duplex pipe per worker, JSON strings): ``add`` (units as ``Plant`` or ``Task`` JSON), ``eval`` (a population
as the F component's data, a seed, MUT's neighbourhood size, unit ids) → per unit: the classes that confirmed a
counterexample, the classes with suspects only, and the probe count. The parent awaits the shards concurrently, so the
swarm calls of a round can run while the shards compute.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import multiprocessing as mp
import zlib
from collections.abc import Callable, Sequence
from typing import Any

from crucible.synth.meta.plant import Plant
from crucible.synth.task import Task


def shard_of(unit_id: str, workers: int) -> int:
    return zlib.crc32(unit_id.encode()) % workers


def _worker(conn: Any, base_name: str) -> None:  # pragma: no cover - runs in the child
    from crucible.synth.falsify import FalsifyContext, cheap
    from crucible.synth.falsify.base import run_classes
    from crucible.synth.meta.library import Library

    base: Callable[[], list[Any]] = cheap
    if base_name != "cheap":
        mod, _, fn = base_name.rpartition(".")
        base = getattr(__import__(mod, fromlist=[fn]), fn)
    units: dict[str, Task] = {}
    ctxs: dict[tuple[int, int], FalsifyContext] = {}
    worlds: dict[str, Any] = {}

    def ctx_for(seed: int, mm: int) -> FalsifyContext:
        k = (seed, mm)
        if k not in ctxs:
            c = FalsifyContext(seed=seed, max_mutants=mm)
            c._worlds = worlds  # one World per WorldRef per shard (runs are sequential within a shard)
            ctxs[k] = c
        return ctxs[k]

    async def evaluate(req: dict[str, Any]) -> dict[str, Any]:
        lib = Library.model_validate(req.get("theta_f") or {})
        fs = [*(base() if req.get("base", True) else []), *lib.falsifiers()]
        ctx = ctx_for(int(req["seed"]), int(req["max_mutants"]))
        out = {}
        for uid in req["ids"]:
            res = await run_classes(units[uid], fs, ctx, cache=True)
            out[uid] = {"hit": [r.cls for r in res if r.applicable and r.c], "suspect": [r.cls for r in res if r.applicable and not r.c and r.suspects],
                        "probes": sum(len(r.findings) for r in res)}
        return out

    loop = asyncio.new_event_loop()
    while True:
        msg = json.loads(conn.recv())
        try:
            if msg["op"] == "stop":
                break
            if msg["op"] == "add":
                for d in msg["plants"]:
                    p = Plant.model_validate(d)
                    units[p.id] = p.task
                for d in msg["tasks"]:
                    t = Task.model_validate(d)
                    units[t.id] = t
                conn.send(json.dumps({"ok": True}))
            elif msg["op"] == "eval":
                conn.send(json.dumps({"ok": True, "units": loop.run_until_complete(evaluate(msg))}))
        except Exception as e:  # noqa: BLE001 - reported to the parent, which raises
            conn.send(json.dumps({"ok": False, "error": f"{type(e).__name__}: {str(e)[:500]}"}))
    for c in ctxs.values():
        with contextlib.suppress(Exception):
            loop.run_until_complete(c.close())
    conn.close()


class ShardPool:
    """Persistent worker processes over a fixed set of units (module docstring)."""

    def __init__(self, workers: int = 4, *, base: str = "cheap"):
        if workers < 1:
            raise ValueError("at least one worker")
        self.workers = workers
        ctx = mp.get_context("spawn")
        self._conns = []
        self._procs = []
        self._locks = [asyncio.Lock() for _ in range(workers)]
        for _ in range(workers):
            a, b = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(b, base), daemon=True)
            p.start()
            self._conns.append(a)
            self._procs.append(p)
        self.known: set[str] = set()

    async def _ask(self, w: int, msg: dict[str, Any]) -> dict[str, Any]:
        async with self._locks[w]:
            self._conns[w].send(json.dumps(msg))
            rep = json.loads(await asyncio.to_thread(self._conns[w].recv))
        if not rep.get("ok"):
            raise RuntimeError(f"shard {w}: {rep.get('error')}")
        return rep

    async def add(self, plants: Sequence[Plant] = (), tasks: Sequence[Task] = ()) -> None:
        by: dict[int, tuple[list[Any], list[Any]]] = {}
        for p in plants:
            if p.id not in self.known:
                by.setdefault(shard_of(p.id, self.workers), ([], []))[0].append(p.model_dump(mode="json"))
                self.known.add(p.id)
        for t in tasks:
            if t.id not in self.known:
                by.setdefault(shard_of(t.id, self.workers), ([], []))[1].append(t.model_dump(mode="json"))
                self.known.add(t.id)
        await asyncio.gather(*(self._ask(w, {"op": "add", "plants": ps, "tasks": ts}) for w, (ps, ts) in by.items()))

    async def evaluate(self, theta_f: Any, ids: Sequence[str], *, seed: int, max_mutants: int, base: bool = True) -> dict[str, dict[str, Any]]:
        missing = [i for i in ids if i not in self.known]
        if missing:
            raise KeyError(f"units not on any shard: {missing[:3]}")
        by: dict[int, list[str]] = {}
        for i in dict.fromkeys(ids):
            by.setdefault(shard_of(i, self.workers), []).append(i)
        reps = await asyncio.gather(*(self._ask(w, {"op": "eval", "theta_f": theta_f, "ids": xs, "seed": seed, "max_mutants": max_mutants, "base": base})
                                      for w, xs in by.items()))
        out: dict[str, dict[str, Any]] = {}
        for r in reps:
            out.update(r["units"])
        return out

    def close(self) -> None:
        for c in self._conns:
            try:
                c.send(json.dumps({"op": "stop"}))
            except (BrokenPipeError, OSError):
                pass
        for p in self._procs:
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
