"""Model gateway (m13 D-1; audit 05 §2.2 C): many engines behind one ``EnginePool`` contract, with health, eviction and
session affinity that survives load and failures.

``GatewayPool`` is a drop-in ``EnginePool`` (token in, token out, version stamp, abort, pause) over any engines, in
process or remote (``HTTPEngine``):

- **Sticky routing that queues instead of spilling.** A rollout (``GenRequest.rid``) is pinned to one engine, so its
  KV prefix stays warm. A new session goes to the least-loaded live engine. A pinned session stays on its engine
  while that engine has fewer than ``spill_factor x capacity`` calls in flight (it queues), and only then spills and
  re-pins. ``EnginePool.route`` spilled at ``capacity`` and broke affinity under load (audit 05 H4).
- **Failures become aborts, not masked rollouts.** A call that raises on an engine returns ``finish="abort"``, and the
  runtime re-issues it (``Runtime._call``). The engine is evicted after ``failure_threshold`` failures, or at once on a
  connection error. Eviction drops only *its* sessions; the others keep their affinity. rLLM's router clears every
  session when one worker changes.
- **Health loop** (``start_health``): ``engine.health()`` every ``health_interval_s``. A failing engine is evicted. A
  recovered one is revived only after it has the current weights: the gateway replays the last ``load_weights`` to it
  first, so a stale engine never serves a newer run.
- **Version**: the minimum over *live* engines. A dead engine no longer pins the pool's version.

Remote engines: ``engine_app(engine)`` serves any engine over HTTP (``/generate``, ``/abort_all``, ``/pause``,
``/resume``, ``/load_weights`` (a checkpoint directory with ``model.safetensors``), ``/health``), and
``HTTPEngine(url)`` is its client. ``python -m crucible.train.gateway engine --port P ...`` starts a TinyPolicy engine
server (tests, local multi-process runs).

Pattern: rLLM ``rllm-model-gateway/session_router.py`` (StickyLeastLoaded + health loop), verl ``router.py``
(per-server sticky-cache clearing), prime-rl ``vllm-router`` ``sticky_least_loaded``.
"""

from __future__ import annotations

import asyncio
import dataclasses
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

from crucible.train.engine import Engine, EnginePool, GenRequest, GenResult


class EngineDown(RuntimeError):
    """The engine cannot be reached (connection refused / reset): evicted at once."""


class GatewayPool(EnginePool):
    def __init__(self, engines: Sequence[Engine], *, spill_factor: float = 2.0, max_sessions: int = 100_000, failure_threshold: int = 3,
                 health_interval_s: float = 5.0):
        super().__init__(engines)
        self.alive = [True] * len(self.engines)
        self.fails = [0] * len(self.engines)
        self.spill_factor, self.max_sessions, self.failure_threshold, self.health_interval_s = spill_factor, max_sessions, failure_threshold, health_interval_s
        self.sessions: OrderedDict[str, int] = OrderedDict()
        self.stats = {"routed": 0, "sticky": 0, "spilled": 0, "new_sessions": 0, "evictions": 0, "revived": 0, "failed_calls": 0}
        self.served: list[tuple[str, int]] = []  # (rid, engine) per call: affinity audits
        self._last_weights: tuple[Any, int] | None = None
        self._health_task: asyncio.Task | None = None

    # ---- routing -------------------------------------------------------------------------------------------------------
    def _live(self) -> list[int]:
        live = [i for i, a in enumerate(self.alive) if a]
        if not live:
            raise EngineDown("no live engines in the gateway")
        return live

    def _load(self, i: int) -> float:
        return self.inflight[i] / max(self.engines[i].capacity, 1)

    def route(self, rid: str) -> int:
        self.stats["routed"] += 1
        i = self.sessions.get(rid)
        if i is not None and self.alive[i]:
            if self.inflight[i] < self.spill_factor * self.engines[i].capacity:
                self.sessions.move_to_end(rid)
                self.stats["sticky"] += 1
                return i
            self.stats["spilled"] += 1
        else:
            self.stats["new_sessions"] += 1
        j = min(self._live(), key=lambda k: (self._load(k), k))
        self.sessions[rid] = j
        self.sessions.move_to_end(rid)
        while len(self.sessions) > self.max_sessions:
            self.sessions.popitem(last=False)
        return j

    def release_session(self, rid: str) -> None:
        self.sessions.pop(rid, None)

    # ---- failures ------------------------------------------------------------------------------------------------------
    def _fail(self, i: int, e: BaseException) -> None:
        self.fails[i] += 1
        down = isinstance(e, EngineDown | ConnectionError | OSError)
        try:
            import httpx

            down = down or isinstance(e, httpx.ConnectError | httpx.RemoteProtocolError | httpx.ReadError)
        except ImportError:  # pragma: no cover
            pass
        if self.alive[i] and (down or self.fails[i] >= self.failure_threshold):
            self.evict(i)

    def evict(self, i: int) -> None:
        if not self.alive[i]:
            return
        self.alive[i] = False
        self.stats["evictions"] += 1
        for rid in [r for r, k in self.sessions.items() if k == i]:  # only its sessions lose affinity
            del self.sessions[rid]

    async def revive(self, i: int) -> bool:
        """Bring engine ``i`` back: catch it up to the last published weights first."""
        if self.alive[i]:
            return True
        try:
            if self._last_weights is not None and self.engines[i].version != self._last_weights[1]:
                await self.engines[i].load_weights(*self._last_weights)
        except Exception:  # noqa: BLE001 -- still not healthy
            return False
        self.alive[i], self.fails[i] = True, 0
        self.stats["revived"] += 1
        return True

    # ---- the Engine contract ---------------------------------------------------------------------------------------------
    async def generate(self, req: GenRequest) -> GenResult:
        try:
            i = self.route(req.rid)
        except EngineDown:
            await asyncio.sleep(0.05)  # nothing alive right now: an abort, re-issued by the runtime
            return GenResult([], [], self.version_or_zero(), "abort")
        self.inflight[i] += 1
        self._enter()
        try:
            res = await self.engines[i].generate(req)
            self.fails[i] = 0
            self.served.append((req.rid, i))
            return res
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001 -- the engine failed mid-call: abort (re-issued elsewhere), never masked
            self.stats["failed_calls"] += 1
            self._fail(i, e)
            return GenResult(list(req.resume_completion), list(req.resume_logprobs), self.version_or_zero(), "abort",
                             list(req.resume_spans) or None)
        finally:
            self.inflight[i] -= 1
            self._exit()

    async def _each(self, method: str, *args: Any) -> list[Any]:
        out = []
        for i in self._live():
            try:
                out.append(await getattr(self.engines[i], method)(*args))
            except Exception as e:  # noqa: BLE001
                self._fail(i, e)
                self.evict(i)  # an engine that cannot pause / resume / load is out of sync: never serve from it
        return out

    async def abort_all(self) -> int:
        return sum(x or 0 for x in await self._each("abort_all"))

    async def pause(self) -> None:
        self._set(paused=True)
        await self._each("pause")

    async def resume(self) -> None:
        await self._each("resume")
        self._set(paused=False)

    async def load_weights(self, state: Any, version: int) -> None:
        self._last_weights = (state, version)
        await self._each("load_weights", state, version)

    @property
    def version(self) -> int:
        return min(self.engines[i].version for i in self._live())

    def version_or_zero(self) -> int:
        try:
            return self.version
        except EngineDown:
            return self._last_weights[1] if self._last_weights else 0

    # ---- health ----------------------------------------------------------------------------------------------------------
    async def check_health(self) -> None:
        for i, e in enumerate(self.engines):
            probe = getattr(e, "health", None)
            if probe is None:
                continue
            try:
                ok = await probe()
            except Exception:  # noqa: BLE001
                ok = False
            if ok:
                self.fails[i] = 0
                if not self.alive[i]:
                    await self.revive(i)
            else:
                self.fails[i] += 1
                if self.alive[i] and self.fails[i] >= self.failure_threshold:
                    self.evict(i)

    def start_health(self) -> None:
        async def loop() -> None:
            while True:
                await asyncio.sleep(self.health_interval_s)
                await self.check_health()

        if self._health_task is None:
            self._health_task = asyncio.create_task(loop())

    async def stop_health(self) -> None:
        if self._health_task is not None:
            self._health_task.cancel()
            await asyncio.gather(self._health_task, return_exceptions=True)
            self._health_task = None

    def affinity(self, dead: set[int] | None = None) -> float:
        """Share of calls served by their session's first engine, over sessions whose first engine stayed alive."""
        first: dict[str, int] = {}
        hit = n = 0
        dead = dead if dead is not None else {i for i, a in enumerate(self.alive) if not a}
        for rid, i in self.served:
            f = first.setdefault(rid, i)
            if f in dead:
                continue
            n += 1
            hit += i == f
        return hit / n if n else 1.0


# ---- remote engines -----------------------------------------------------------------------------------------------------
def _result_json(res: GenResult) -> dict[str, Any]:
    return dataclasses.asdict(res)


def engine_app(engine: Engine):
    """A FastAPI app serving one engine (the remote side of ``HTTPEngine``)."""
    from fastapi import FastAPI

    app = FastAPI()

    @app.post("/generate")
    async def generate(body: dict) -> dict:
        req = GenRequest(**{k: v for k, v in body.items() if k in {f.name for f in dataclasses.fields(GenRequest)}})
        return _result_json(await engine.generate(req))

    @app.post("/abort_all")
    async def abort_all() -> dict:
        return {"aborted": await engine.abort_all()}

    @app.post("/pause")
    async def pause() -> dict:
        await engine.pause()
        return {"ok": True}

    @app.post("/resume")
    async def resume() -> dict:
        await engine.resume()
        return {"ok": True}

    @app.post("/load_weights")
    async def load_weights(body: dict) -> dict:
        state: Any = body["path"]
        if hasattr(engine, "_serve"):  # an in-process torch engine loads a state dict from the checkpoint directory
            from pathlib import Path

            from safetensors.torch import load_file

            state = load_file(str(Path(body["path"]) / "model.safetensors"))
        await engine.load_weights(state, int(body["version"]))
        return {"ok": True, "version": engine.version}

    @app.get("/health")
    async def health() -> dict:
        return {"ok": True, "version": engine.version, "capacity": engine.capacity}

    return app


def generate_app(pool: Any, *, model: str, max_model_len: int = 32768, session_header: str = "x-session-id", max_aborts: int = 64):
    """The vLLM ``/inference/v1/generate`` protocol (token ids in, token ids + logprobs out) served from any Crucible engine
    pool (m13 E: verifiers ``TrainClient`` / prime-rl ``renderers.client.generate``). The client renders and parses
    itself, so its traces carry exact token records; Crucible supplies the engines (sticky on the client's session
    header), re-issues aborts, and stamps every completion with the engine's policy version."""
    import uuid

    from fastapi import FastAPI, Header

    app = FastAPI()

    @app.get("/v1/models")
    async def models() -> dict:
        return {"object": "list", "data": [{"id": model, "object": "model", "max_model_len": max_model_len}]}

    # (annotations are postponed in this module: only builtins in signatures, FastAPI resolves them from globals)
    @app.post("/inference/v1/generate")
    async def generate(body: dict, session: str | None = Header(default=None, alias=session_header)) -> dict:
        sp = body.get("sampling_params") or {}
        rid = session or f"gen-{uuid.uuid4().hex[:12]}"
        g = GenRequest(rid=rid, call_id=f"{rid}:{uuid.uuid4().hex[:6]}", prompt_ids=list(body["token_ids"]), max_tokens=int(sp.get("max_tokens") or 512),
                       temperature=float(sp.get("temperature", 1.0)), stop_ids=tuple(sp.get("stop_token_ids") or ()), seed=sp.get("seed"),
                       top_p=float(sp.get("top_p", 1.0)))
        for _ in range(max_aborts):
            res = await pool.generate(g)
            if res.finish != "abort":
                break
            g.resume_completion, g.resume_logprobs, g.resume_spans = [], [], []  # re-issued whole under the new version
        ids, lps = list(res.completion_ids), list(res.logprobs)
        finish = "stop" if (res.finish == "stop" or (ids and ids[-1] in g.stop_ids)) else "length"
        return {"request_id": g.call_id, "prompt_token_ids": list(g.prompt_ids),
                "choices": [{"index": 0, "token_ids": ids, "finish_reason": finish, "policy_version": res.version,
                             "logprobs": {"content": [{"token": f"token_id:{t}", "logprob": float(lp)} for t, lp in zip(ids, lps, strict=True)]}}],
                "usage": {"prompt_tokens": len(g.prompt_ids), "completion_tokens": len(ids), "total_tokens": len(g.prompt_ids) + len(ids)}}

    return app


class HTTPEngine:
    def __init__(self, url: str, *, capacity: int = 64, timeout_s: float = 600.0):
        import httpx

        self.url, self.capacity = url.rstrip("/"), capacity
        self.version = 0
        self.client = httpx.AsyncClient(timeout=timeout_s)

    async def _post(self, path: str, json: Any = None) -> dict:
        import httpx

        try:
            r = await self.client.post(self.url + path, json=json)
        except (httpx.ConnectError, httpx.RemoteProtocolError, httpx.ReadError) as e:
            raise EngineDown(f"{self.url}: {type(e).__name__}: {e}") from e
        r.raise_for_status()
        return r.json()

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "remote", "abort": "all", "pause": ["keep"], "sleep": False, "top_p_candidates": True, "routing": False,
                "weight_load": "checkpoint_dir"}

    async def generate(self, req: GenRequest) -> GenResult:
        d = await self._post("/generate", dataclasses.asdict(req))
        spans = [tuple(x) for x in d["version_spans"]] if d.get("version_spans") else None
        return GenResult(**{**d, "version_spans": spans})

    async def abort_all(self) -> int:
        return int((await self._post("/abort_all"))["aborted"])

    async def pause(self) -> None:
        await self._post("/pause")

    async def resume(self) -> None:
        await self._post("/resume")

    async def load_weights(self, state: Any, version: int) -> None:
        if not isinstance(state, str):
            raise TypeError("HTTPEngine.load_weights takes a checkpoint directory (CheckpointPublisher), not a state dict")
        d = await self._post("/load_weights", {"path": state, "version": version})
        self.version = int(d["version"])

    async def health(self) -> bool:
        import httpx

        try:
            r = await self.client.get(self.url + "/health", timeout=5.0)
            if r.status_code == 200:
                self.version = int(r.json().get("version", self.version))
                return True
        except httpx.HTTPError:
            pass
        return False


def main() -> None:  # pragma: no cover - a server process
    import argparse

    import uvicorn

    from crucible.train.engine import TinyPolicy, TorchEngine

    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["engine"])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--vocab", type=int, default=32)
    ap.add_argument("--dim", type=int, default=48)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--capacity", type=int, default=64)
    ap.add_argument("--delay", type=float, default=0.002)
    ap.add_argument("--numerics", default="bf16")
    a = ap.parse_args()
    eng = TorchEngine(TinyPolicy(a.vocab, a.dim, seed=a.seed), capacity=a.capacity, numerics=a.numerics, decode_delay_s=a.delay)
    uvicorn.run(engine_app(eng), host="127.0.0.1", port=a.port, log_level="warning")


if __name__ == "__main__":  # pragma: no cover
    main()
