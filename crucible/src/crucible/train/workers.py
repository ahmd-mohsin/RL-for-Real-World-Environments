"""Remote rollout workers (m13 D-2; audit 05 §2.2 D): episodes run in worker processes on any host; the controller keeps the
lease table and the model.

- **Controller** (``LeaseController``, an HTTP app in the runtime's own event loop): ``Runtime(rollout_fn=
  controller.rollout_fn())`` turns each rollout into a *job*. A worker takes jobs with a **lease** (``/lease``), renews it
  with heartbeats (``/heartbeat``, default every 1 s, expiry ``lease_s``) and returns the record (``/result``). Model
  calls come back to the controller (``/gen``) and go through ``Runtime._call``: version stamps, aborts at the sample
  barrier, pause, staleness and partial-rollout accounting are exactly the in-process ones.
- **A lost worker** (killed, partitioned, hung): its leases expire; each job is re-leased as a new *attempt* (its pool tag
  is reset: the dead attempt's versions no longer count toward staleness), up to ``max_attempts``, then masked (never
  scored 0). Every request carries ``(rid, attempt)``: calls and results from a superseded attempt are refused (409) or
  ignored, so **nothing is trained twice** and a zombie worker cannot overwrite the live attempt.
- **Worker** (``RolloutWorker``; ``python -m crucible.train.workers --controller URL --rollout copy``): leases up to
  ``concurrency`` jobs, runs each as an asyncio task with a remote ``gen``, heartbeats its active leases (a revoked
  lease cancels its task), posts results.

``env_rollout(env_factory)`` is the token-environment loop of ``Runtime._rollout`` as a rollout function (reset / step
contract), so token envs run remotely too; World tasks use ``world_env.world_rollout``.

Pattern: slime ``queue_data_source.py`` (take / heartbeat leases), verifiers ``EnvServer`` (failures as data), AReaL HTTP
callbacks; prime-rl's Ray-free process model.
"""

from __future__ import annotations

import asyncio
import contextlib
import dataclasses
import threading
import time
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.train.engine import GenResult


def env_rollout(env_factory: Callable[[], Any], *, top_p: float = 1.0) -> Callable:
    """A rollout function over a token environment (``reset(prompt, seed) -> ids``, ``step(completion) -> Step``)."""

    async def rollout(rid: str, prompt: str, seed: int, gen: Callable, *, temperature: float = 1.0, harness: str = "native:token@1"):
        env = env_factory()
        obs = await env.reset(prompt, seed)
        context, prev_full, turns, k = list(obs), None, [], 0
        while True:
            k += 1
            res = await gen(context, None, temperature)
            delta = context[len(prev_full):] if prev_full is not None and context[: len(prev_full)] == prev_full else context
            turns.append(Turn(call_id=f"{rid}:c{k}", actor="agent", prompt_is_delta=prev_full is not None, prompt_token_ids=list(delta),
                              completion_token_ids=list(res.completion_ids), logprobs=list(res.logprobs), policy_version=res.version,
                              version_spans=res.version_spans, finish=res.finish, candidates=res.candidates, routing=res.routing,
                              sampling={"temperature": temperature, "top_p": top_p}))
            prev_full = context + list(res.completion_ids)
            step = await env.step(list(res.completion_ids))
            context = prev_full + list(step.obs_ids)
            if step.done:
                return RolloutRecord(rollout_id=rid, sample=rid, task=prompt, harness=harness, policy="native", turns=turns, reward=step.reward, stop="done")

    return rollout


@dataclass
class _Job:
    rid: str
    prompt: str
    seed: int
    temperature: float
    harness: str
    gen: Callable
    future: asyncio.Future
    attempt: int = 0
    worker: str | None = None
    deadline: float = 0.0
    history: list[str] = field(default_factory=list)


class LeaseController:
    def __init__(self, *, lease_s: float = 10.0, max_attempts: int = 3, reap_every_s: float = 0.25, on_reset: Callable[[str], None] | None = None):
        self.lease_s, self.max_attempts, self.reap_every_s = lease_s, max_attempts, reap_every_s
        self.jobs: dict[str, _Job] = {}
        self.queue: deque[str] = deque()
        self.on_reset = on_reset  # called with the rid when a job is re-leased (the runtime resets its pool tag)
        self.stats = {"leased": 0, "expired": 0, "released": 0, "stale_results": 0, "stale_gen": 0, "masked": 0, "results": 0}
        self.workers: dict[str, float] = {}  # worker id -> last heartbeat
        self.id = uuid.uuid4().hex[:12]  # a restarted controller is a new id: workers drop the dead one's leases
        self._server = None
        self._tasks: list[asyncio.Task] = []
        self._gens: dict[str, set[asyncio.Task]] = {}  # rid -> in-flight /gen handlers (cancelled with their rollout)
        self.url: str | None = None

    # ---- runtime side ---------------------------------------------------------------------------------------------------
    def rollout_fn(self) -> Callable:
        async def remote(rid: str, prompt: str, seed: int, gen: Callable, *, temperature: float = 1.0, harness: str = "remote@1"):
            fut = asyncio.get_running_loop().create_future()
            self.jobs[rid] = _Job(rid, prompt, seed, temperature, harness, gen, fut)
            self.queue.append(rid)
            try:
                return await fut
            finally:
                self.jobs.pop(rid, None)
                for t in self._gens.pop(rid, set()):  # the rollout ended (or the run did): its model calls must not keep re-issuing
                    t.cancel()

        return remote

    def attach(self, runtime: Any) -> None:
        """Reset a re-leased rollout's pool tag (versions / length of the dead attempt must not count)."""

        def reset(rid: str) -> None:
            t = runtime.pool.tags.get(rid)
            if t is not None:
                t.versions, t.length = [], 0

        self.on_reset = reset

    # ---- lease table ------------------------------------------------------------------------------------------------------
    def lease(self, worker: str, n: int) -> list[dict[str, Any]]:
        out, now = [], time.monotonic()
        self.workers[worker] = now
        while self.queue and len(out) < n:
            rid = self.queue.popleft()
            j = self.jobs.get(rid)
            if j is None or j.future.done() or j.worker is not None:
                continue
            j.worker, j.deadline = worker, now + self.lease_s
            j.history.append(worker)
            self.stats["leased"] += 1
            out.append({"rid": rid, "attempt": j.attempt, "prompt": j.prompt, "seed": j.seed, "temperature": j.temperature, "harness": j.harness})
        return out

    def heartbeat(self, worker: str, leases: list[list]) -> list[str]:
        """Renew ``[(rid, attempt)]``; returns the rids this worker no longer owns (cancel them)."""
        now = time.monotonic()
        self.workers[worker] = now
        revoked = []
        for rid, attempt in leases:
            j = self.jobs.get(rid)
            if j is None or j.worker != worker or j.attempt != attempt or j.future.done():
                revoked.append(rid)
            else:
                j.deadline = now + self.lease_s
        return revoked

    def _owned(self, worker: str, rid: str, attempt: int) -> _Job | None:
        j = self.jobs.get(rid)
        return j if j is not None and j.worker == worker and j.attempt == attempt and not j.future.done() else None

    def result(self, worker: str, rid: str, attempt: int, record: dict | None, error: str | None) -> bool:
        j = self._owned(worker, rid, attempt)
        if j is None:  # superseded attempt (its lease expired and was re-leased) or already resolved: never applied twice
            self.stats["stale_results"] += 1
            return False
        self.stats["results"] += 1
        rec = RolloutRecord.model_validate(record) if record is not None else None
        if rec is not None:
            rec.timings["worker_attempts"] = float(j.attempt + 1)
        j.future.set_result(rec)
        return True

    def reap(self) -> int:
        now, n = time.monotonic(), 0
        for j in list(self.jobs.values()):
            if j.worker is not None and now > j.deadline and not j.future.done():
                n += 1
                self.stats["expired"] += 1
                j.worker, j.attempt = None, j.attempt + 1
                if j.attempt >= self.max_attempts:
                    self.stats["masked"] += 1
                    j.future.set_result(None)  # masked: never scored 0
                    continue
                if self.on_reset is not None:
                    self.on_reset(j.rid)
                self.queue.appendleft(j.rid)  # first in line: it has waited the longest
        return n

    # ---- HTTP ---------------------------------------------------------------------------------------------------------
    def app(self):
        from fastapi import FastAPI, HTTPException

        app = FastAPI()

        @app.post("/lease")
        async def lease(body: dict) -> dict:
            return {"jobs": self.lease(str(body["worker"]), int(body.get("max", 1))), "controller": self.id}

        @app.post("/heartbeat")
        async def heartbeat(body: dict) -> dict:
            return {"revoked": self.heartbeat(str(body["worker"]), list(body.get("leases", []))), "controller": self.id}

        @app.post("/gen")
        async def gen(body: dict) -> dict:
            j = self._owned(str(body["worker"]), str(body["rid"]), int(body["attempt"]))
            if j is None:
                self.stats["stale_gen"] += 1
                raise HTTPException(status_code=409, detail="lease not held")
            me = asyncio.current_task()
            self._gens.setdefault(j.rid, set()).add(me)
            try:
                res = await j.gen(list(body["ids"]), body.get("max_tokens"), float(body.get("temperature", j.temperature)))
            except asyncio.CancelledError:
                raise HTTPException(status_code=409, detail="rollout ended") from None
            finally:
                self._gens.get(j.rid, set()).discard(me)
            return dataclasses.asdict(res)

        @app.post("/result")
        async def result(body: dict) -> dict:
            return {"accepted": self.result(str(body["worker"]), str(body["rid"]), int(body["attempt"]), body.get("record"), body.get("error"))}

        @app.get("/health")
        async def health() -> dict:
            return {"ok": True, "queued": len(self.queue), "jobs": len(self.jobs), "workers": len(self.workers), "controller": self.id}

        return app

    async def start(self, host: str = "127.0.0.1", port: int = 0) -> str:
        import socket

        import uvicorn

        if port == 0:
            with socket.socket() as s:
                s.bind((host, 0))
                port = s.getsockname()[1]
        self._server = uvicorn.Server(uvicorn.Config(self.app(), host=host, port=port, log_level="warning", loop="asyncio", lifespan="off",
                                                     timeout_graceful_shutdown=2))
        self._tasks.append(asyncio.create_task(self._server.serve()))

        async def reaper() -> None:
            while True:
                await asyncio.sleep(self.reap_every_s)
                self.reap()

        self._tasks.append(asyncio.create_task(reaper()))
        while not self._server.started:
            await asyncio.sleep(0.02)
        self.url = f"http://{host}:{port}"
        return self.url

    async def stop(self) -> None:
        for ts in self._gens.values():  # in-flight model calls of a finished run: cancel, do not wait on them
            for t in ts:
                t.cancel()
        self._gens.clear()
        if self._server is not None:
            self._server.should_exit = True
        for t in self._tasks[1:]:
            t.cancel()
        await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks = []


class _Revoked(Exception):
    pass


class RolloutWorker:
    def __init__(self, controller: str, rollout_fn: Callable, *, worker_id: str | None = None, concurrency: int = 64, heartbeat_s: float = 1.0,
                 poll_s: float = 0.05, prefetch: int = 4, orphan_after_s: float = 60.0):
        """``prefetch``: jobs taken per lease request. Small batches pulled as capacity frees spread work across workers (a
        CPU-saturated worker polls less often); a greedy first worker leasing ``concurrency`` jobs starved the others.

        ``orphan_after_s``: with no successful contact with the controller (lease, heartbeat, gen, result) for this long,
        the worker cancels its jobs and exits (``exit_reason = "controller_lost"``; the CLI exits 75 so a supervisor
        restarts or reaps it). Before this, a worker whose controller died polled forever. A controller that comes
        back as a new process (a new controller id) does not orphan the worker: its leases died with the old one, so
        their jobs are cancelled and the worker serves the new controller (``controller_restarts``)."""
        self.prefetch, self.orphan_after_s = prefetch, orphan_after_s
        self._last_contact = time.monotonic()
        self.controller_id: str | None = None
        self.controller_restarts = 0
        self.exit_reason: str | None = None
        import httpx

        self.url, self.rollout_fn = controller.rstrip("/"), rollout_fn
        self.id = worker_id or f"w-{uuid.uuid4().hex[:8]}"
        self.concurrency, self.heartbeat_s, self.poll_s = concurrency, heartbeat_s, poll_s
        self.client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=120.0), limits=httpx.Limits(max_connections=concurrency + 8))
        self.active: dict[str, tuple[int, asyncio.Task]] = {}
        self.done = 0

    async def _post(self, path: str, body: dict, *, retries: int = 5) -> Any:
        import httpx

        for k in range(retries + 1):
            try:
                r = await self.client.post(self.url + path, json=body)
                break
            except (httpx.ConnectError, httpx.ConnectTimeout):  # never sent: safe to retry (a sent /gen is never re-sent)
                if k == retries:
                    raise
                await asyncio.sleep(0.1 * 2**k)
        if r.status_code == 409:
            self._last_contact = time.monotonic()
            raise _Revoked(path)
        r.raise_for_status()
        self._last_contact = time.monotonic()
        d = r.json()
        self._seen_controller(d.get("controller") if isinstance(d, dict) else None)
        return d

    def _seen_controller(self, cid: str | None) -> None:
        if not cid:
            return
        if self.controller_id is not None and cid != self.controller_id:  # a new controller process: the old leases are gone
            self.controller_restarts += 1
            for rid in list(self.active):
                self._revoke(rid)
        self.controller_id = cid

    async def _run_job(self, job: dict) -> None:
        rid, attempt = job["rid"], int(job["attempt"])

        async def gen(ids: list[int], max_tokens: int | None, temperature: float) -> GenResult:
            d = await self._post("/gen", {"worker": self.id, "rid": rid, "attempt": attempt, "ids": list(ids), "max_tokens": max_tokens, "temperature": temperature})
            return GenResult(**{**d, "version_spans": [tuple(x) for x in d["version_spans"]] if d.get("version_spans") else None})

        rec, err = None, None
        try:
            rec = await self.rollout_fn(rid, job["prompt"], int(job["seed"]), gen, temperature=float(job["temperature"]), harness=job["harness"])
        except (_Revoked, asyncio.CancelledError):
            return  # the lease moved on: nothing to report
        except Exception as e:  # noqa: BLE001 -- an environment failure: reported, masked by the controller (record None)
            err = f"{type(e).__name__}: {e}"[:500]
        with contextlib.suppress(Exception):
            await self._post("/result", {"worker": self.id, "rid": rid, "attempt": attempt, "error": err,
                                         "record": rec.model_dump(mode="json", by_alias=True) if rec is not None else None})
        self.done += 1

    def _heartbeat_thread(self, loop: asyncio.AbstractEventLoop, stop: threading.Event) -> None:
        """Heartbeats from their own thread: a CPU-bound environment saturating the event loop must not starve them (its
        leases would expire while it works)."""
        import httpx

        with httpx.Client(timeout=10.0) as c:
            while not stop.wait(self.heartbeat_s):
                leases = [[rid, a] for rid, (a, _t) in list(self.active.items())]
                if not leases:
                    continue
                try:
                    r = c.post(self.url + "/heartbeat", json={"worker": self.id, "leases": leases})
                    revoked = r.json()["revoked"] if r.status_code == 200 else []
                    if r.status_code == 200:
                        self._last_contact = time.monotonic()
                except Exception:  # noqa: BLE001, S112 -- controller briefly unreachable: keep working; the lease may expire
                    continue
                for rid in revoked:
                    loop.call_soon_threadsafe(self._revoke, rid)

    def _revoke(self, rid: str) -> None:
        ent = self.active.pop(rid, None)
        if ent is not None:
            ent[1].cancel()

    async def run(self, *, max_jobs: int | None = None, stop: asyncio.Event | None = None) -> None:
        hb_stop = threading.Event()
        hb = threading.Thread(target=self._heartbeat_thread, args=(asyncio.get_running_loop(), hb_stop), daemon=True)
        hb.start()
        try:
            while (stop is None or not stop.is_set()) and (max_jobs is None or self.done < max_jobs):
                if time.monotonic() - self._last_contact > self.orphan_after_s:
                    self.exit_reason = "controller_lost"
                    break
                room = min(self.concurrency - len(self.active), self.prefetch)
                jobs = []
                if room > 0:
                    try:
                        jobs = (await self._post("/lease", {"worker": self.id, "max": room}))["jobs"]
                    except Exception:  # noqa: BLE001 -- controller not up yet / gone
                        jobs = []
                for job in jobs:
                    t = asyncio.create_task(self._run_job(job))
                    self.active[job["rid"]] = (int(job["attempt"]), t)
                    t.add_done_callback(lambda _t, rid=job["rid"]: self.active.pop(rid, None) if self.active.get(rid, (None, None))[1] is _t else None)
                await asyncio.sleep(self.poll_s if not jobs else 0.001)  # yield to running episodes between pulls
        finally:
            hb_stop.set()
            for _a, t in list(self.active.values()):
                t.cancel()
            await asyncio.gather(*(t for _a, t in self.active.values()), return_exceptions=True)
            await asyncio.to_thread(hb.join, 5.0)
            await self.client.aclose()


def _rollout_from_spec(spec: str, cpu_s: float) -> Callable:
    if spec == "copy":
        from crucible.train.envs import CopyTask

        return env_rollout(lambda: CopyTask(turns=3, symbols=8, cpu_s=cpu_s))
    import importlib

    mod, _, attr = spec.partition(":")
    obj = getattr(importlib.import_module(mod), attr)
    return obj() if callable(obj) and not asyncio.iscoroutinefunction(obj) else obj


def main() -> None:  # pragma: no cover - a worker process
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", required=True)
    ap.add_argument("--rollout", default="copy", help="'copy' or 'module:factory' returning a rollout function")
    ap.add_argument("--concurrency", type=int, default=64)
    ap.add_argument("--cpu-s", dest="cpu_s", type=float, default=0.0)
    ap.add_argument("--id", default=None)
    ap.add_argument("--orphan-after-s", dest="orphan_after_s", type=float, default=60.0)
    a = ap.parse_args()
    w = RolloutWorker(a.controller, _rollout_from_spec(a.rollout, a.cpu_s), worker_id=a.id, concurrency=a.concurrency, orphan_after_s=a.orphan_after_s)
    asyncio.run(w.run())
    if w.exit_reason == "controller_lost":
        raise SystemExit(75)  # EX_TEMPFAIL: the controller is gone; a supervisor restarts or reaps this worker


if __name__ == "__main__":  # pragma: no cover
    main()
