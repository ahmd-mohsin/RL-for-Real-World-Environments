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

**One version per replayed sequence** (async-rl audit R5, 2026-10-28). A call after a hot-load re-forwards the whole
prefix under the new adapter (the prefix cache is keyed by adapter), so a sequence whose calls span two policy versions
has two routings for its prefix, and the trainer takes one route per position: first-forward-wins trained the later
version's tokens over the earlier version's prefix routes. ``wire._routes`` refuses such a sequence (no routes, coverage 0,
``routing_version_mixed``), and the step recomputes the router (``wire.replay_all_or_none``). A synchronous run has one
version per episode and always replays. Splitting the sequence at the version change (slime forks a segment per call
under R3) would keep replay exact at the cost of forwarding the prefix again.

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
    """One call's routing rows: ``rows[i]`` are the experts of token position ``start + i`` (``[rows, layers, top_k]``).
    ``sampled`` (``[n]`` float32, optional): the logprobs of the call's n sampled tokens, at positions ``sampled_start``
    .. ``sampled_start + n``; they tell apart two calls with the same tokens (``RoutingStore``)."""

    rows: np.ndarray
    start: int
    sampled: np.ndarray | None = None
    sampled_start: int | None = None

    @property
    def end(self) -> int:
        return self.start + int(self.rows.shape[0])


def masked_logprobs(logprobs: Sequence[float], mask: Sequence[bool]) -> np.ndarray:
    """A training sequence's sampled logprobs (aligned with its token ids, as ``wire.to_sample`` sends them) with NaN at
    untrained positions, whose logprob the wire zeroes: what the stores compare calls on (``closest_call``)."""
    return np.where(np.asarray(mask, dtype=bool), np.asarray(logprobs, dtype=np.float32), np.float32(np.nan))


def closest_call(calls: Sequence[tuple[np.ndarray | None, int | None]], full_logprobs: np.ndarray | None) -> int | None:
    """Which of several calls with the same tokens a training sequence holds: ``calls`` are (sampled logprobs, position of
    the first sampled token) per call, ``full_logprobs`` the sequence's ``masked_logprobs``. The call whose sampled
    logprobs are closest (max abs gap over the trained positions) wins; with one call, or nothing to compare, the newest."""
    if not calls:
        return None
    if full_logprobs is None or len(calls) == 1:
        return len(calls) - 1

    def gap(c: tuple[np.ndarray | None, int | None]) -> float:
        smp, s0 = c
        if smp is None or s0 is None:
            return float("inf")
        want = np.asarray(full_logprobs[s0:s0 + len(smp)], dtype=np.float32)
        ok = np.isfinite(want)
        if want.shape != smp.shape or not ok.any():
            return float("inf")
        return float(np.abs(smp[ok] - want[ok]).max())

    return int(np.argmin([gap(c) for c in calls]))


def decode(field_: dict[str, Any] | None) -> Routes | None:
    """prime-rl's packed ``routed_experts`` field -> ``Routes`` (None when the server returned none)."""
    if not field_:
        return None
    rows = np.frombuffer(base64.b64decode(field_["data"]), dtype=np.dtype(field_["dtype"])).reshape(field_["shape"])
    return Routes(rows=rows, start=int(field_.get("start", 0)))


@dataclass
class RoutingStore:
    """Calls' routes by ``key_of(prompt + completion)``, in memory, oldest dropped past ``max_calls`` calls (a resumed run
    re-samples the groups it had not trained, so nothing is lost by not persisting them).

    **Several calls per key** (as ``topk.TopKStore``): two calls can return the same tokens, e.g. group members that open
    with the same short tool call, or one task in two groups sampled from different adapter versions under async
    training. With one entry per key the second call overwrote the first's routes (another version's experts replayed
    for the first group), and training one group dropped the key the other still needed, so its step lost coverage and
    replayed nothing. Each call keeps its own entry, and ``get`` / ``drop`` pick it by the training sequence's sampled
    logprobs (``closest_call``)."""

    max_calls: int = 200_000
    by_key: OrderedDict[str, list[Routes]] = field(default_factory=OrderedDict)
    shape: tuple[int, int] | None = None  # (layers, top_k), from the first call
    calls: int = 0

    def put(self, token_ids: Sequence[int], routes: Routes) -> None:
        lk = (int(routes.rows.shape[1]), int(routes.rows.shape[2]))
        if self.shape is None:
            self.shape = lk
        elif lk != self.shape:
            raise ValueError(f"routing shape {lk} differs from the run's {self.shape}")
        k = key_of(token_ids)
        self.by_key.setdefault(k, []).append(routes)
        self.by_key.move_to_end(k)
        self.calls += 1
        while self.calls > self.max_calls:
            _, old = self.by_key.popitem(last=False)
            self.calls -= len(old)

    def _index(self, key: str, full_logprobs: np.ndarray | None) -> int | None:
        rs = self.by_key.get(key)
        return None if not rs else closest_call([(r.sampled, r.sampled_start) for r in rs], full_logprobs)

    def get(self, token_ids: Sequence[int], *, full_logprobs: np.ndarray | None = None) -> Routes | None:
        """The routes of the call that returned ``token_ids``; ``full_logprobs`` (``masked_logprobs`` of the training
        sequence) picks among calls with the same tokens."""
        key = key_of(token_ids)
        i = self._index(key, full_logprobs)
        return None if i is None else self.by_key[key][i]

    def drop(self, token_ids: Sequence[int], *, full_logprobs: np.ndarray | None = None) -> None:
        """Forget the call ``get`` would return; other calls with the same tokens stay."""
        key = key_of(token_ids)
        i = self._index(key, full_logprobs)
        if i is None:
            return
        rs = self.by_key[key]
        rs.pop(i)
        self.calls -= 1
        if not rs:
            del self.by_key[key]


def stitch(full: Sequence[int], call_ends: Sequence[int], store: RoutingStore, *, logprobs: Sequence[float] | None = None,
           mask: Sequence[bool] | None = None) -> tuple[np.ndarray | None, float]:
    """``[len(full), layers, top_k]`` routing for one training sequence whose sampled calls end at ``call_ends``
    (``full[:end]`` is each call's prompt + completion), and the share of positions covered. The first call that forwarded
    a position wins (later calls of the episode served it from the prefix cache, or recomputed the same input). None when
    no call has routes. ``logprobs`` / ``mask`` (aligned with ``full``): the sequence's sampled logprobs, which pick the
    sequence's own call when several returned the same tokens (``RoutingStore``)."""
    if store.shape is None:
        return None, 0.0
    layers, k = store.shape
    picked = masked_logprobs(logprobs, mask) if logprobs is not None and mask is not None else None
    out: np.ndarray | None = None
    have = np.zeros(len(full), dtype=bool)
    for end in sorted(call_ends):
        r = store.get(full[:end], full_logprobs=picked)
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
