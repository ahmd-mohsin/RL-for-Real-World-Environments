"""An SGLang server as an M12 engine (m13 F-3). The protocol is SGLang's native HTTP API, as slime drives it
(``slime/rollout/sglang_rollout.py``):

- ``POST /generate`` ``{input_ids, sampling_params: {max_new_tokens, temperature, top_p, stop_token_ids,
  skip_special_tokens: false}, return_logprob: true, rid, return_routed_experts?}`` →
  ``meta_info.output_token_logprobs = [[logprob, token_id, _], ...]`` (token ids *and* behaviour logprobs, token in /
  token out), ``meta_info.finish_reason.type`` (``stop`` | ``length`` | ``abort``), ``meta_info.routed_experts`` (per
  token, per layer, the top-k experts) when the server runs with ``--enable-return-routed-experts`` (rollout routing
  replay, R3).
- ``POST /abort_request`` ``{rid}`` aborts one call (per-call abort), ``{abort_all: true}`` all of them.
- ``POST /pause_generation`` / ``/continue_generation``: the colocated sample barrier.
- ``POST /release_memory_occupation`` / ``/resume_memory_occupation`` ``{tags}`` (sleep / wake: weights, KV cache), with
  the vLLM engine's ordering: a publish wakes only the weights; the KV cache wakes at ``resume`` after the trainer offloaded.
- ``POST /update_weights_from_disk`` ``{model_path}`` (``success`` checked), then ``/flush_cache``.

Every completion is stamped with the engine's version at call start.
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import httpx

from crucible.train.engine import GenRequest, GenResult


class SGLangServerEngine:
    kv_modes = ("discard",)

    def __init__(self, base_url: str, *, capacity: int = 64, placement: str = "colocated", timeout_s: float = 1800.0, routed_experts: bool = False):
        self.base, self.capacity, self.placement, self.routed_experts = base_url.rstrip("/"), capacity, placement, routed_experts
        self.version = 0
        self.client = httpx.AsyncClient(timeout=timeout_s)
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._rids: dict[str, str] = {}  # call id -> SGLang rid (per-call abort)
        self.asleep: set[str] = set()

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "server", "abort": "per_call", "pause": ["keep"], "sleep": self.placement == "colocated",
                "top_p_candidates": False, "routing": self.routed_experts, "weight_load": "checkpoint_dir"}

    async def _post(self, path: str, json: Any = None) -> httpx.Response:
        r = await self.client.post(self.base + path, json=json)
        r.raise_for_status()
        return r

    async def generate(self, req: GenRequest) -> GenResult:
        if req.top_p < 1.0:
            raise NotImplementedError("SGLangServerEngine does not report top-p candidate sets; sample with top_p=1.0")
        await self._unpaused.wait()
        version = self.version
        rid = f"{req.call_id}-{uuid.uuid4().hex[:6]}"
        self._rids[req.call_id] = rid
        sp = {"max_new_tokens": req.max_tokens - len(req.resume_completion), "temperature": req.temperature, "top_p": req.top_p,
              "skip_special_tokens": False, **({"stop_token_ids": list(req.stop_ids)} if req.stop_ids else {}), **({"seed": req.seed} if req.seed is not None else {})}
        body: dict[str, Any] = {"input_ids": list(req.prompt_ids) + list(req.resume_completion), "sampling_params": sp, "return_logprob": True, "rid": rid}
        if self.routed_experts:
            body["return_routed_experts"] = True
        try:
            r = await self._post("/generate", body)
        finally:
            self._rids.pop(req.call_id, None)
        meta = r.json().get("meta_info") or {}
        pairs = meta.get("output_token_logprobs") or []
        ids = list(req.resume_completion) + [int(p[1]) for p in pairs]
        lps = list(req.resume_logprobs) + [float(p[0]) for p in pairs]
        fin = (meta.get("finish_reason") or {}).get("type") if isinstance(meta.get("finish_reason"), dict) else meta.get("finish_reason")
        routing = None
        if self.routed_experts and meta.get("routed_experts") is not None:
            routing = list(req.resume_routing) + [[int(e) for layer in tok for e in (layer if isinstance(layer, list) else [layer])]
                                                  for tok in meta["routed_experts"]]
        spans = list(req.resume_spans) or None
        if req.resume_completion and spans and spans[-1][1] != version:
            spans.append((len(req.resume_completion), version))
        finish = "abort" if fin == "abort" else ("stop" if fin == "stop" else "length")
        return GenResult(ids, lps, version, finish, spans if spans and len(spans) > 1 else None, len(req.prompt_ids), None, routing)

    async def abort(self, call_id: str) -> None:
        rid = self._rids.get(call_id)
        if rid is not None:
            await self._post("/abort_request", {"rid": rid})

    async def abort_all(self) -> int:
        n = len(self._rids)
        await self._post("/abort_request", {"abort_all": True, "rid": ""})
        return n

    async def pause(self) -> None:
        self._unpaused.clear()
        await self._post("/pause_generation", {})
        if self.placement == "colocated":
            await self._post("/release_memory_occupation", {"tags": ["weights", "kv_cache"]})
            self.asleep = {"weights", "kv_cache"}

    async def resume(self) -> None:
        if "kv_cache" in self.asleep:  # after the trainer offloaded (the publish woke only the weights)
            await self._post("/resume_memory_occupation", {"tags": ["kv_cache"]})
            self.asleep.discard("kv_cache")
        if "weights" in self.asleep:
            await self._post("/resume_memory_occupation", {"tags": ["weights"]})
            self.asleep.discard("weights")
        await self._post("/continue_generation", {})
        self._unpaused.set()

    async def load_weights(self, path: str, version: int) -> None:
        if "weights" in self.asleep:
            await self._post("/resume_memory_occupation", {"tags": ["weights"]})
            self.asleep.discard("weights")
        d = (await self._post("/update_weights_from_disk", {"model_path": str(path)})).json()
        if not d.get("success", True):
            raise RuntimeError(f"SGLang refused the weights at {path}: {d.get('message')}")
        await self._post("/flush_cache", {})
        self.version = version
