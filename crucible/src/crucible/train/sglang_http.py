"""An SGLang server as an M12 engine (m13 F-3). The protocol is SGLang's native HTTP API, as slime drives it
(``slime/rollout/sglang_rollout.py``):

- ``POST /generate`` ``{input_ids, sampling_params: {max_new_tokens, temperature, top_p, stop_token_ids,
  skip_special_tokens: false}, return_logprob: true, rid, return_routed_experts?}`` →
  ``meta_info.output_token_logprobs = [[logprob, token_id, _], ...]`` (token ids *and* behaviour logprobs, token in /
  token out), ``meta_info.finish_reason.type`` (``stop`` | ``length`` | ``abort``), ``meta_info.weight_version``, and
  ``meta_info.routed_experts`` when the server runs with ``--enable-return-routed-experts`` (rollout routing replay, R3):
  base64 int32, ``len(tokens) - 1 - routed_experts_start_len`` rows of ``num_layers × topk`` (slime's contract).
  ``sampling_params.sampling_seed`` seeds a call.
- ``POST /abort_request`` ``{rid}`` aborts one call (per-call abort), ``{abort_all: true}`` all of them.
- ``POST /pause_generation`` / ``/continue_generation``: the colocated sample barrier.
- ``POST /release_memory_occupation`` / ``/resume_memory_occupation`` ``{tags}`` (sleep / wake: weights, KV cache), with
  the vLLM engine's ordering: a publish wakes only the weights; the KV cache wakes at ``resume`` after the trainer offloaded.
- ``POST /update_weights_from_disk`` ``{model_path}`` (``success`` checked), then ``/flush_cache``.

Every completion is stamped with the server's ``weight_version`` when it reports one, else the engine's version at call
start. A resumed call sends the kept tokens as the prompt's tail (token-level resume).
"""

from __future__ import annotations

import asyncio
import base64
import uuid
from typing import Any

import httpx
import numpy as np

from crucible.train.engine import GenRequest, GenResult, resume_spans


def _weight_version(v: Any) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


class SGLangServerEngine:
    kv_modes = ("discard",)

    def __init__(self, base_url: str, *, capacity: int = 64, placement: str = "colocated", timeout_s: float = 1800.0, routed_experts: bool = False,
                 model_path: str | None = None, routed_width: int | None = None):
        """``model_path``: the weights the server was launched with. SGLang's ``release_memory_occupation`` frees the weights
        without keeping their contents (unless the server runs with ``--enable-weights-cpu-backup``), so a resume with no
        publish in between must reload them. Found by the live check (2026-10-11): greedy output changed after a sleep/wake
        cycle with no new weights. The engine reloads the last loaded weights; with none known it refuses to resume.

        ``routed_width``: ``num_layers × topk`` of the MoE (each routing row's length), checked when given."""
        self.base, self.capacity, self.placement, self.routed_experts = base_url.rstrip("/"), capacity, placement, routed_experts
        self.routed_width = routed_width
        self.weights_path = model_path
        self._weights_released = False
        self.version = 0
        self.client = httpx.AsyncClient(timeout=timeout_s)
        self._unpaused = asyncio.Event()
        self._unpaused.set()
        self._loading = asyncio.Event()  # cleared while a weight update runs: no call starts inside it (G3 D11)
        self._loading.set()
        self._rids: dict[str, str] = {}  # call id -> SGLang rid (per-call abort)
        self.asleep: set[str] = set()

    def capabilities(self) -> dict[str, Any]:
        return {"token_ids": True, "logprobs": "server", "abort": "per_call", "pause": ["abort"], "sleep": self.placement == "colocated",
                "top_p_candidates": False, "routing": self.routed_experts, "weight_load": "checkpoint_dir", "token_resume": True}

    async def _post(self, path: str, json: Any = None) -> httpx.Response:
        r = await self.client.post(self.base + path, json=json)
        if r.is_error:  # the server's reason, not just the status (a 400 from SGLang names the offending field)
            raise httpx.HTTPStatusError(f"SGLang {path} -> {r.status_code}: {r.text[:800]}", request=r.request, response=r)
        return r

    def _routes(self, meta: dict[str, Any], n_prompt: int, n_kept: int, n_new: int, start: int) -> list[list[int]]:
        """slime's R3 row contract (``slime/utils/types.py``, ``rollout/sglang_rollout.py``): ``routed_experts`` is base64
        int32 (or nested lists), one row per sequence position but the last, from ``routed_experts_start_len`` on; a row
        is ``num_layers × topk`` expert ids. Row ``i`` holds the experts at position ``i``, which produced token ``i + 1``,
        so completion token ``t`` is row ``n_prompt - 1 + t``. Returned: the new completion tokens' rows, each flattened
        layer-major (``Turn.routing``). M12 audit V4 / G2 H5: the value was read as one nested row per completion token."""
        raw = meta.get("routed_experts")
        total = n_prompt + n_kept + n_new
        rows = max(0, total - 1 - start)
        if isinstance(raw, str):
            flat = np.frombuffer(base64.b64decode(raw), dtype="<i4")
        else:
            flat = np.asarray(raw, dtype=np.int64).reshape(-1)
        if rows == 0:
            if flat.size:
                raise ValueError(f"SGLang returned {flat.size} routed expert ids for 0 rows")
            return []
        if flat.size % rows:
            raise ValueError(f"SGLang routed_experts: {flat.size} ids do not split into {rows} rows (tokens={total}, start={start})")
        width = flat.size // rows
        if self.routed_width is not None and width != self.routed_width:
            raise ValueError(f"SGLang routed_experts rows hold {width} ids, expected num_layers x topk = {self.routed_width}")
        arr = flat.reshape(rows, width)
        first = n_prompt - 1 + n_kept - start  # the row that produced the first new completion token
        if first < 0:
            raise ValueError(f"routed_experts_start_len {start} is past the first new token's position {n_prompt - 1 + n_kept}")
        return [[int(x) for x in r] for r in arr[first:first + n_new]]

    async def generate(self, req: GenRequest) -> GenResult:
        if req.top_p < 1.0:
            raise NotImplementedError("SGLangServerEngine does not report top-p candidate sets; sample with top_p=1.0")
        await self._unpaused.wait()
        await self._loading.wait()
        version = self.version
        kept = list(req.resume_completion)
        rid = f"{req.call_id}-{uuid.uuid4().hex[:6]}"
        self._rids[req.call_id] = rid
        sp = {"max_new_tokens": req.max_tokens - len(kept), "temperature": req.temperature, "top_p": req.top_p,
              "skip_special_tokens": False, **({"stop_token_ids": list(req.stop_ids)} if req.stop_ids else {}),
              **({"sampling_seed": req.seed} if req.seed is not None else {})}  # SGLang's key (slime sglang_rollout.py; V13)
        body: dict[str, Any] = {"input_ids": list(req.prompt_ids) + kept, "sampling_params": sp, "return_logprob": True, "rid": rid}
        start = 0
        if self.routed_experts:
            body["return_routed_experts"] = True
            if kept:  # a continuation asks only for the new rows, from the kept output's end (slime's routed_experts_start_len)
                if len(req.resume_routing) != len(kept):
                    raise ValueError(f"R3 continuation has {len(req.resume_routing)} routes for {len(kept)} kept tokens: refusing to mix policies")
                start = len(req.prompt_ids) + len(kept) - 1
                body["routed_experts_start_len"] = start
        try:
            r = await self._post("/generate", body)
        finally:
            self._rids.pop(req.call_id, None)
        meta = r.json().get("meta_info") or {}
        pairs = meta.get("output_token_logprobs") or []
        new = [int(p[1]) for p in pairs]
        served = _weight_version(meta.get("weight_version"))
        if served is not None and new:
            version = served  # the server's own stamp for these tokens (slime records it per segment; G3 D11)
        ids = kept + new
        lps = list(req.resume_logprobs) + [float(p[0]) for p in pairs]
        fin = (meta.get("finish_reason") or {}).get("type") if isinstance(meta.get("finish_reason"), dict) else meta.get("finish_reason")
        routing = None
        if self.routed_experts and meta.get("routed_experts") is not None:
            got = int(meta.get("routed_experts_start_len", start) or 0)
            if got != start:
                raise ValueError(f"SGLang routed_experts_start_len {got} differs from the request's {start}")
            routing = list(req.resume_routing) + self._routes(meta, len(req.prompt_ids), len(kept), len(new), start)
        finish = "abort" if fin == "abort" else ("stop" if fin == "stop" else "length")
        spans = resume_spans(req.resume_spans, len(kept), len(new), version)
        keep = spans if (finish == "abort" and kept) or len(spans) > 1 else None
        return GenResult(ids, lps, spans[-1][1], finish, keep, len(req.prompt_ids), None, routing)

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
            self._weights_released = True

    async def resume(self) -> None:
        if "kv_cache" in self.asleep:  # after the trainer offloaded (the publish woke only the weights)
            await self._post("/resume_memory_occupation", {"tags": ["kv_cache"]})
            self.asleep.discard("kv_cache")
        if "weights" in self.asleep:
            await self._post("/resume_memory_occupation", {"tags": ["weights"]})
            self.asleep.discard("weights")
        if self._weights_released:  # woken memory holds no weights: reload the last ones (a publish would have cleared this)
            if not self.weights_path:
                raise RuntimeError("SGLang weights were released and no weights path is known to reload: pass model_path= or publish before resume")
            d = (await self._post("/update_weights_from_disk", {"model_path": str(self.weights_path), "weight_version": str(self.version)})).json()
            if not d.get("success", True):
                raise RuntimeError(f"SGLang refused to reload {self.weights_path}: {d.get('message')}")
            await self._post("/flush_cache", {})
            self._weights_released = False
        await self._post("/continue_generation", {})
        self._unpaused.set()

    async def load_weights(self, path: str, version: int) -> None:
        """``/update_weights_from_disk`` with ``weight_version`` (echoed per response as ``meta_info.weight_version``).
        SGLang drains running requests under its model-update lock; no new call starts until the version is set (G3
        D11: a call starting between the update and the version bump was served by v+1 and labelled v)."""
        self._loading.clear()
        try:
            if "weights" in self.asleep:
                await self._post("/resume_memory_occupation", {"tags": ["weights"]})
                self.asleep.discard("weights")
            d = (await self._post("/update_weights_from_disk", {"model_path": str(path), "weight_version": str(version)})).json()
            if not d.get("success", True):
                raise RuntimeError(f"SGLang refused the weights at {path}: {d.get('message')}")
            await self._post("/flush_cache", {})
            self.version = version
            self.weights_path, self._weights_released = str(path), False
        finally:
            self._loading.set()
