"""MoE routing replay: the experts vLLM chose at sampling time, replayed in prime-rl's training forward (swarm-pipeline/14 §2).

The student is a 256-expert top-8 MoE on every one of its 40 layers. When the trainer recomputes the router, a token
whose router scores are close can take a different expert set than the one that produced the sampled token. Its learner
logprob then comes from another sub-network, which inflates the importance ratio and the sampler–learner mismatch (the
MoE train/inference mismatch; prime-rl's "router replay"). Both halves exist upstream and are only switched on here:

- **Sampling** (prime-rl's vLLM server, ``vllm.enable_return_routed_experts = true``): ``/inference/v1/generate`` (token
  ids in and out) returns, per completion, ``routed_experts = {data: base64, shape: [rows, layers, top_k], dtype, start}``.
  These are the experts of every token the call forwarded: rows end at the last forwarded token (the last sampled token
  is only forwarded by the next call) and begin at ``start``, the first token not served from the prefix cache
  (prime-rl ``inference/vllm/serving_tokens.py:88-99``).
- **Training** (prime-rl's trainer, ``enable_router_replay = true``): ``TrainingSample.routed_experts`` (``[len(token_ids),
  layers, top_k]``) is gathered instead of the router's own top-k (``trainer/models/layers/moe.py:226-228``), sharded with
  the sequence under context parallel (``trainer/rl/train.py:417``).

What this module adds is the join. Crucible's loop keeps its own records, so the routing is held beside them, out of band
(``RoutingStore``, keyed by the sha of the call's prompt + completion ids). Train time stitches a datum's calls back into
one ``[T, layers, top_k]`` array (``stitch``): each position comes from the first call that forwarded it. The prefix cache
is salted per episode (``cache_salt``), so no position is served from another episode's cache without routing. The
episode's very last token is never forwarded, and its row is zero; it is no input to a trained logit.

``coverage`` (positions with a recorded route over all positions) is logged per step: replay is exact when it is 1.
``topk_overlap`` is the mismatch metric for comparing two routings when both are known (|S_sample ∩ S_train| / k per
token and layer). With replay off, the logprob-level mismatch the loss already logs (``kl/mismatch`` = E[r − 1 − log r])
is the signal that routing drift feeds.
"""

from __future__ import annotations

import base64
import hashlib
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np


def key_of(token_ids: Sequence[int]) -> str:
    """The call's identity: the sha256 of its prompt + completion ids (int32 bytes)."""
    return hashlib.sha256(np.asarray(token_ids, dtype=np.int32).tobytes()).hexdigest()


@dataclass(frozen=True)
class Routes:
    """One call's routing rows: ``rows[i]`` are the experts of token position ``start + i`` (``[rows, layers, top_k]``)."""

    rows: np.ndarray
    start: int

    @property
    def end(self) -> int:
        return self.start + int(self.rows.shape[0])


def decode(field_: dict[str, Any] | None) -> Routes | None:
    """prime-rl's packed ``routed_experts`` field -> ``Routes`` (None when the server returned none)."""
    if not field_:
        return None
    rows = np.frombuffer(base64.b64decode(field_["data"]), dtype=np.dtype(field_["dtype"])).reshape(field_["shape"])
    return Routes(rows=rows, start=int(field_.get("start", 0)))


@dataclass
class RoutingStore:
    """Calls' routes by ``key_of(prompt + completion)``, in memory, oldest dropped past ``max_calls`` (a resumed run
    re-samples the groups it had not trained, so nothing is lost by not persisting them)."""

    max_calls: int = 200_000
    by_key: OrderedDict[str, Routes] = field(default_factory=OrderedDict)
    shape: tuple[int, int] | None = None  # (layers, top_k), from the first call

    def put(self, token_ids: Sequence[int], routes: Routes) -> None:
        lk = (int(routes.rows.shape[1]), int(routes.rows.shape[2]))
        if self.shape is None:
            self.shape = lk
        elif lk != self.shape:
            raise ValueError(f"routing shape {lk} differs from the run's {self.shape}")
        k = key_of(token_ids)
        self.by_key[k] = routes
        self.by_key.move_to_end(k)
        while len(self.by_key) > self.max_calls:
            self.by_key.popitem(last=False)

    def get(self, token_ids: Sequence[int]) -> Routes | None:
        return self.by_key.get(key_of(token_ids))

    def drop(self, token_ids: Sequence[int]) -> None:
        self.by_key.pop(key_of(token_ids), None)


def stitch(full: Sequence[int], call_ends: Sequence[int], store: RoutingStore) -> tuple[np.ndarray | None, float]:
    """``[len(full), layers, top_k]`` routing for one training sequence whose sampled calls end at ``call_ends``
    (``full[:end]`` is each call's prompt + completion), and the share of positions covered. The first call that forwarded
    a position wins (later calls of the episode served it from the prefix cache, or recomputed the same input). None when
    no call has routes."""
    if store.shape is None:
        return None, 0.0
    layers, k = store.shape
    out: np.ndarray | None = None
    have = np.zeros(len(full), dtype=bool)
    for end in sorted(call_ends):
        r = store.get(full[:end])
        if r is None:
            continue
        if out is None:
            out = np.zeros((len(full), layers, k), dtype=r.rows.dtype)
        lo, hi = max(0, r.start), min(len(full), r.end)
        if hi <= lo:
            continue
        new = ~have[lo:hi]
        out[lo:hi][new] = r.rows[lo - r.start:hi - r.start][new]
        have[lo:hi] = True
    if out is None:
        return None, 0.0
    return out, float(have[:-1].mean()) if len(full) > 1 else 1.0  # the last token is never forwarded


def topk_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Mean |S_a ∩ S_b| / k over tokens and layers, for two routings ``[T, layers, k]`` of the same tokens."""
    if a.shape != b.shape:
        raise ValueError(f"shapes differ: {a.shape} vs {b.shape}")
    k = a.shape[-1]
    hits = np.zeros(a.shape[:-1], dtype=np.int64)
    for j in range(k):
        hits += (a == b[..., j:j + 1]).any(axis=-1)
    return float(hits.mean() / k)


def encode_for_wire(arr: np.ndarray) -> dict[str, Any]:
    """The bytes prime-rl's ``RoutedExperts`` carries (``transports/batch/types.py:16-19``)."""
    arr = np.ascontiguousarray(arr)
    return {"data": arr.tobytes(), "shape": list(arr.shape), "dtype": str(arr.dtype)}
