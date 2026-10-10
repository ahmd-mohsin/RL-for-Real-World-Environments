"""A vLLM OpenAI-compatible server as an M12 engine (the real-model inference side).

Token-in-token-out: ``/v1/completions`` with a token-id ``prompt``, ``return_token_ids`` and ``logprobs=0`` (the sampled
token's logprob; vLLM ``--logprobs-mode processed_logprobs`` makes it the post-temperature value the trainer recomputes).
Versions are ours: the engine's ``version`` changes only through ``load_weights``, and every response is stamped with
the version that was serving when the call started (call-level switching: a call cut by ``abort_all``/``pause`` returns
``finish="abort"`` and is re-issued whole by the runtime).

Lifecycle on one GPU (research m11-m12/03 finding 1): the server keeps a fixed memory fraction. In ``colocated``
placement ``pause()`` also puts it to ``/sleep?level=1`` (weights to CPU, KV discarded) so the trainer gets the memory;
``load_weights`` wakes the weights, loads the published checkpoint through ``/collective_rpc`` →
``CrucibleWorkerExtension.crucible_load_weights``, resets the prefix cache (a cache from old weights is stale) and wakes
the KV cache; ``resume()`` reopens generation. In ``split`` placement it never sleeps (memory split: disaggregated
semantics on one device).

Requires the server started with ``VLLM_SERVER_DEV_MODE=1 --enable-sleep-mode --worker-extension-cls
crucible_vllm_ext.CrucibleWorkerExtension`` (and ``--logprobs-mode processed_logprobs``).
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx

from crucible.train.engine import GenRequest, GenResult


class VLLMServerEngine:
    def __init__(self, base_url: str, model: str, *, capacity: int = 64, placement: str = "colocated", timeout_s: float = 1800.0, sleep_level: int = 1):
        self.base, self.model, self.capacity, self.placement, self.sleep_level = base_url.rstrip("/"), model, capacity, placement, sleep_level
        self.version = 0
        self.client = httpx.AsyncClient(timeout=timeout_s)
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._tasks: dict[str, asyncio.Task] = {}
        self.sleeping = False

    async def _post(self, path: str, json: Any = None, params: Any = None) -> httpx.Response:
        r = await self.client.post(self.base + path, json=json, params=params)
        r.raise_for_status()
        return r

    async def generate(self, req: GenRequest) -> GenResult:
        if req.top_p < 1.0:  # checklist #37: without the per-token nucleus the trainer would compare against the wrong distribution
            raise NotImplementedError("VLLMServerEngine does not report top-p candidate sets yet; sample with top_p=1.0")
        await self._unpaused.wait()
        version = self.version
        rid = f"{req.call_id}-{uuid.uuid4().hex[:6]}"
        body = {"model": self.model, "prompt": list(req.prompt_ids), "max_tokens": req.max_tokens, "temperature": req.temperature, "logprobs": 0,
                "return_token_ids": True, "skip_special_tokens": False, "stop_token_ids": list(req.stop_ids) or None, "seed": req.seed,
                "request_id": rid}
        task = asyncio.create_task(self.client.post(self.base + "/v1/completions", json={k: v for k, v in body.items() if v is not None},
                                                    headers={"X-Request-Id": rid}))
        self._tasks[rid] = task
        try:
            r = await task
        except asyncio.CancelledError:
            if task.cancelled():
                return GenResult([], [], version, "abort")
            raise
        finally:
            self._tasks.pop(rid, None)
        r.raise_for_status()
        ch = r.json()["choices"][0]
        ids = list(ch.get("token_ids") or [])
        lps = list(((ch.get("logprobs") or {}).get("token_logprobs")) or [])
        if ch.get("finish_reason") == "abort" or (not ids and self.version != version):
            return GenResult(ids, lps, version, "abort")
        return GenResult(ids, [float(x) for x in lps], version, "stop" if ch.get("finish_reason") == "stop" else "length", None, len(req.prompt_ids))

    async def abort_all(self) -> int:
        n = len(self._tasks)
        for t in list(self._tasks.values()):
            t.cancel()
        await self._post("/abort_requests", json={})
        return n

    async def pause(self) -> None:
        self._unpaused.clear()
        await self._post("/pause", params={"mode": "abort"})
        if self.placement == "colocated":
            await self._post("/sleep", params={"level": str(self.sleep_level)})
            self.sleeping = True

    async def resume(self) -> None:
        if self.sleeping:
            await self.client.post(self.base + "/wake_up")
            self.sleeping = False
        await self._post("/resume")
        self._unpaused.set()

    async def load_weights(self, path: str, version: int) -> None:
        if self.sleeping:
            await self.client.post(self.base + "/wake_up", params=[("tags", "weights")])
        await self._post("/collective_rpc", json={"method": "crucible_load_weights", "args": [str(path)]})
        await self._post("/reset_prefix_cache")
        if self.sleeping:
            await self.client.post(self.base + "/wake_up", params=[("tags", "kv_cache")])
            self.sleeping = False
        self.version = version
