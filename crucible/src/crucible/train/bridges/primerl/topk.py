"""The sampler's top-k logprobs per generated token, kept for score centering (swarm-pipeline/14 §7).

Score centering (``loss``; arXiv 2609.20807) needs, at every trained position, the sampler's head H: the k most likely
token ids and their logprobs under the distribution that sampled the token. They come back with the completion, on the
same path as the sampled-token logprobs and the routed experts:

- **Sampling** (``backend._Sampler._generate``): ``/inference/v1/generate`` with ``sampling_params.logprobs = k``. vLLM
  0.31's sampler returns k + 1 entries per token (the sampled token first, then the top k, ``Sampler.gather_logprobs``)
  from the temperature-processed distribution (prime-rl sets ``logprobs_mode = "processed_logprobs"``). prime-rl's
  generate handler packs only the sampled entry; ``topk_server`` (loaded into vLLM's API server as a ``--middleware``
  import path) also packs the head as ``topk_ids`` (int32) and ``topk_logprobs`` (float32) arrays ``[n, k]``.
- **Out of band** (``TopKStore``, as ``routing.RoutingStore`` holds routes): one ``Heads`` per call, keyed by the sha of
  its prompt + completion ids; row i is the head of completion token i, at full-sequence position ``start + i``. Calls
  that returned the same tokens keep one entry each, told apart by their sampled-token logprobs (``TopKStore`` doc).
  Only calls sampled at the training temperature are kept (``backend._Sampler``): q must be the distribution the trainer
  scores at that temperature.
- **Stitched per training sequence** (``stitch``): every trained position takes the head of the call that sampled it.
  ``coverage`` is the share of trained positions that have one.
- **On prime-rl's wire** (``encode_payload``): the heads ride in ``TrainingSample.sampling_mask`` (int32 rows that
  prime-rl already packs, shifts by one, shards under context parallel and hands to the LM head). A row is
  ``[SC_MAGIC, ids_1..k, bits(logprob_1)..bits(logprob_k)]`` (the float32 logprobs bit-cast to int32), only at trained
  positions; others have count 0. ``SC_MAGIC`` marks the row as Crucible's: the trainer-side hook (``sc_head``) reads
  it as a head, never as a top-p/top-k sampling mask (prime-rl would renormalize the sampled token's logprob over a
  mask; Crucible never samples with one).
- **All or none per step** (``all_or_none``): the step centers every sequence or none. A step where any sequence lacks a
  head at a trained position sends no heads, and the loss falls back to MIS for the whole step (``loss`` doc), as routing
  replay refuses partial coverage.

Memory (§7): 8 bytes per head entry (int32 id + float32 logprob), so 256 bytes per sampled token at k = 32: 18.7 MB for
an episode's median 73K sampled tokens. Nothing is stored for prompt or observation tokens.
"""

from __future__ import annotations

import base64
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from crucible.train.bridges.primerl.routing import closest_call, key_of, masked_logprobs

DEFAULT_K = 32  # matches full centering in the paper's top-k sweep (App. A.4: k = 32 and 128 on par with full)
SC_MAGIC = -1_234_567_890  # int32 in column 0 of a head row: never a token id, never prime-rl's -1 padding


@dataclass(frozen=True)
class Heads:
    """One call's heads: ``ids[i]`` / ``logprobs[i]`` (``[n, k]``) are the head of the token at position ``start + i``.
    ``sampled`` (``[n]`` float32, optional): the logprobs of the call's sampled tokens, which tell apart two calls with
    the same tokens (``TopKStore``)."""

    ids: np.ndarray
    logprobs: np.ndarray
    start: int
    sampled: np.ndarray | None = None

    @property
    def end(self) -> int:
        return self.start + int(self.ids.shape[0])

    @property
    def k(self) -> int:
        return int(self.ids.shape[1])


def _array(f: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(base64.b64decode(f["data"]), dtype=np.dtype(f["dtype"])).reshape(f["shape"])


def decode(ids_field: dict[str, Any] | None, logprobs_field: dict[str, Any] | None, *, start: int, sampled: Sequence[float] | None = None) -> Heads | None:
    """``topk_server``'s ``topk_ids`` / ``topk_logprobs`` fields -> ``Heads`` (None when the server returned none);
    ``sampled``: the call's sampled-token logprobs."""
    if not ids_field or not logprobs_field:
        return None
    ids, lps = _array(ids_field).astype(np.int32), _array(logprobs_field).astype(np.float32)
    if ids.shape != lps.shape or ids.ndim != 2:
        raise ValueError(f"top-k ids {ids.shape} and logprobs {lps.shape} differ")
    smp = None if sampled is None else np.asarray(sampled, dtype=np.float32)
    if smp is not None and smp.shape != (ids.shape[0],):
        raise ValueError(f"{smp.shape[0]} sampled logprobs for {ids.shape[0]} top-k rows")
    return Heads(ids=ids, logprobs=lps, start=int(start), sampled=smp)


@dataclass
class TopKStore:
    """Calls' heads by ``routing.key_of(prompt + completion)``, oldest dropped past ``max_calls`` (``RoutingStore``'s rule).

    **Several calls per key.** Two calls can return the same tokens: group members that open with the same short tool
    call, or the same task in two groups sampled from different adapter versions under async training. Each keeps its
    own heads (the sampler q that produced it), and ``get`` / ``drop`` pick the call whose sampled-token logprobs
    (``Heads.sampled``) match the training sequence's. One entry per key would hand one call another call's heads (from
    another version, if the groups differ in version), and training the first group would drop the key the second still
    needs, so a later step would lose a head and fall back to MIS for all its sequences."""

    k: int = DEFAULT_K
    max_calls: int = 200_000
    by_key: OrderedDict[str, list[Heads]] = field(default_factory=OrderedDict)
    calls: int = 0

    def put(self, token_ids: Sequence[int], heads: Heads) -> None:
        if heads.k != self.k:
            raise ValueError(f"the server returned {heads.k} top-k logprobs per token, the run asked for {self.k}")
        key = key_of(token_ids)
        self.by_key.setdefault(key, []).append(heads)
        self.by_key.move_to_end(key)
        self.calls += 1
        while self.calls > self.max_calls:
            _, old = self.by_key.popitem(last=False)
            self.calls -= len(old)

    def _pick(self, key: str, full_logprobs: Sequence[float] | None) -> int | None:
        """The index of the call under ``key`` whose sampled logprobs are closest to ``full_logprobs`` (the training
        sequence's sampled logprobs aligned with its token ids, NaN where untrained: ``trained_logprobs``) at that call's
        completion positions; the newest call when there is one call or nothing to compare."""
        hs = self.by_key.get(key)  # the rule is ``routing.closest_call``, shared with ``RoutingStore``
        return None if not hs else closest_call([(h.sampled, h.start) for h in hs], None if full_logprobs is None else np.asarray(full_logprobs))

    def get(self, token_ids: Sequence[int], *, full_logprobs: Sequence[float] | None = None) -> Heads | None:
        """The heads of the call that returned ``token_ids`` (``full_logprobs``: ``_pick``)."""
        key = key_of(token_ids)
        i = self._pick(key, full_logprobs)
        return None if i is None else self.by_key[key][i]

    def drop(self, token_ids: Sequence[int], *, full_logprobs: Sequence[float] | None = None) -> None:
        """Forget the call ``get`` would return; other calls with the same tokens stay."""
        key = key_of(token_ids)
        i = self._pick(key, full_logprobs)
        if i is None:
            return
        hs = self.by_key[key]
        hs.pop(i)
        self.calls -= 1
        if not hs:
            del self.by_key[key]


trained_logprobs = masked_logprobs  # a training sequence's sampled logprobs, NaN where untrained (``routing.masked_logprobs``)


def stitch(full: Sequence[int], call_ends: Sequence[int], mask: Sequence[bool], store: TopKStore, *,
           logprobs: Sequence[float] | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``ids`` / ``logprobs`` ``[len(full), k]`` (-1 / -inf where absent), ``present`` (a head at a trained position) and
    the coverage of the trained positions for one training sequence whose calls end at ``call_ends``. Each position takes
    the head of the first call whose completion holds it (the call that sampled it). ``logprobs``: the sequence's sampled
    logprobs aligned with ``full``, which pick the right call when several returned the same tokens (``TopKStore``)."""
    n, k = len(full), store.k
    picked = trained_logprobs(logprobs, mask) if logprobs is not None else None
    ids = np.full((n, k), -1, dtype=np.int32)
    lps = np.full((n, k), -np.inf, dtype=np.float32)
    have = np.zeros(n, dtype=bool)
    for end in sorted(call_ends):
        h = store.get(full[:end], full_logprobs=picked)
        if h is None:
            continue
        lo, hi = max(0, h.start), min(n, h.end)
        if hi <= lo:
            continue
        new = ~have[lo:hi]
        ids[lo:hi][new] = h.ids[lo - h.start:hi - h.start][new]
        lps[lo:hi][new] = h.logprobs[lo - h.start:hi - h.start][new]
        have[lo:hi] = True
    m = np.asarray(mask, dtype=bool)
    present = have & m
    cov = float(present.sum() / m.sum()) if m.any() else 1.0
    return ids, lps, present, cov


def encode_payload(ids: np.ndarray, logprobs: np.ndarray, present: np.ndarray) -> dict[str, bytes]:
    """prime-rl's ``SamplingMask`` bytes (``ids``: flat int32, ``counts``: int32 per position) carrying the heads of the
    ``present`` positions as ``[SC_MAGIC, ids, bits(logprobs)]`` rows (module doc)."""
    k = int(ids.shape[1])
    rows = np.empty((int(present.sum()), 1 + 2 * k), dtype=np.int32)
    rows[:, 0] = SC_MAGIC
    rows[:, 1:1 + k] = ids[present]
    rows[:, 1 + k:] = np.ascontiguousarray(logprobs[present], dtype=np.float32).view(np.int32)
    counts = np.where(present, 1 + 2 * k, 0).astype(np.int32)
    return {"ids": rows.tobytes(), "counts": counts.tobytes()}


def decode_payload(ids: bytes, counts: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``encode_payload``'s inverse (tests and the trainer-side check): ``(ids [T, k], logprobs [T, k], present [T])``."""
    c = np.frombuffer(counts, dtype=np.int32)
    flat = np.frombuffer(ids, dtype=np.int32)
    present = c > 0
    w = int(c.max()) if c.size and c.max() > 0 else 1
    k = (w - 1) // 2
    rows = flat.reshape(-1, w) if flat.size else np.zeros((0, w), dtype=np.int32)
    if rows.size and not (rows[:, 0] == SC_MAGIC).all():
        raise ValueError("not a score-centering payload: a row without SC_MAGIC")
    out_ids = np.full((len(c), k), -1, dtype=np.int32)
    out_lps = np.full((len(c), k), -np.inf, dtype=np.float32)
    out_ids[present] = rows[:, 1:1 + k]
    out_lps[present] = np.ascontiguousarray(rows[:, 1 + k:]).view(np.float32)
    return out_ids, out_lps, present


def all_or_none(rows: Sequence[dict[str, Any]]) -> list[Any]:
    """The heads the step sends: every sequence's, or none (module doc). A sequence without trained tokens has coverage 1
    and an empty payload."""
    hs = [r.get("sc_heads") for r in rows]
    ok = bool(rows) and all(h is not None and float(r.get("sc_coverage", 0.0)) >= 1.0 for h, r in zip(hs, rows, strict=True))
    return hs if ok else [None] * len(rows)
