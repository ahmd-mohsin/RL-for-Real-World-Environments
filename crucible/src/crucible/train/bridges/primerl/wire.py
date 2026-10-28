"""Crucible's training datums on prime-rl's batch wire (``prime_rl/transports/batch/types.py``, v0.10.0).

A ``TokenDatum`` (the Tinker bridge's unit: ``input_ids`` = full[:-1], logprobs / advantages / mask aligned with
``target_ids`` = full[1:]) becomes one prime-rl ``TrainingSample``: one flat sequence ``token_ids`` = full, with per-token
``mask`` (the policy's own sampled tokens), ``logprobs``, ``temperatures`` and ``advantages``, all of ``token_ids``'
length (prime-rl shifts internally). Token 0 is never a target, so it carries mask False, logprob 0 and advantage 0.

``undo_token_mean``: prime-rl's rl loss is divided by the step's count of trainable tokens (``TrainingSample`` doc:
"each normalized by its own global token count"). The advantages Crucible built are already final (``datums.scale``:
``sum`` or ``token_mean``), so they are multiplied by that count before they leave: the trained objective is the one the
run configured, on either backend.

``sc_heads`` (score centering, ``topk``): with a ``TopKStore`` every sample also carries the sampler's top-k head at its
trained positions as prime-rl ``SamplingMask`` bytes (``topk.encode_payload``) and its ``sc_coverage``; a step sends all
of them or none (``topk.all_or_none``).

``pack``: prime-rl's own packer (``orchestrator/batch.py`` ``prepare_batch`` + ``build_bin_cost``: FLOP-balanced bins of
``seq_len`` tokens, one micro-batch list per trainer data rank) when prime-rl is installed; ``simple_pack`` (one sample
per micro batch, round robin over ranks, as plain dicts) only for tests and dry runs, refused for a real trainer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.train.bridges.tinker.backend import TokenDatum


def full_ids(d: TokenDatum) -> list[int]:
    return list(d.input_ids) + [d.target_ids[-1]]


def call_ends(d: TokenDatum) -> list[int]:
    """Where each sampled call of the datum ends in ``full_ids`` (``full[:end]`` is the call's prompt + completion): from
    the TITO call spans when the datum has them, else the whole sequence (one call)."""
    spans = d.meta.get("call_spans")
    if spans and d.meta.get("prompt_len") is not None:
        p = int(d.meta["prompt_len"])
        return [p + int(b) for _, _, b in spans]
    return [len(d.input_ids) + 1]


def to_sample(d: TokenDatum, *, temperature: float, env_name: str = "crucible", scale: float = 1.0, routing: Any = None, topk: Any = None) -> dict[str, Any]:
    """One ``TrainingSample`` as a field dict (module doc); with a ``routing.RoutingStore``, the sequence's routed
    experts stitched from its calls (``routed_experts``, None when no call has any) and their ``routing_coverage``; with a
    ``topk.TopKStore``, its sampler heads (``sc_heads``) and their ``sc_coverage``."""
    if not d.input_ids:
        raise ValueError("an empty datum")
    token_ids = list(d.input_ids) + [d.target_ids[-1]]
    if token_ids[1:] != list(d.target_ids):
        raise ValueError("target_ids must be input_ids shifted by one")
    mask = [False] + [bool(m) for m in d.mask]
    return {"token_ids": token_ids, "mask": mask, "logprobs": [0.0] + [float(x) if m else 0.0 for x, m in zip(d.logprobs, d.mask, strict=True)],
            "temperatures": [float(temperature)] * len(token_ids),
            "advantages": [0.0] + [float(a) * scale if m else 0.0 for a, m in zip(d.advantages, d.mask, strict=True)],
            "env_name": env_name, "trace_id": str(d.meta.get("rollout")) if d.meta.get("rollout") is not None else None,
            "branch_index": int(d.meta["segment"]) if isinstance(d.meta.get("segment"), int) else None, **_routes(d, token_ids, routing),
            **_heads(d, token_ids, mask, topk)}


def _heads(d: TokenDatum, token_ids: list[int], mask: list[bool], store: Any) -> dict[str, Any]:
    if store is None:
        return {}
    from crucible.train.bridges.primerl.topk import encode_payload, stitch

    ids, lps, present, cov = stitch(token_ids, call_ends(d), mask, store)
    return {"sc_heads": encode_payload(ids, lps, present), "sc_coverage": cov}


def _routes(d: TokenDatum, token_ids: list[int], routing: Any) -> dict[str, Any]:
    if routing is None:
        return {}
    from crucible.train.bridges.primerl.routing import encode_for_wire, stitch

    arr, cov = stitch(token_ids, call_ends(d), routing)
    return {"routed_experts": encode_for_wire(arr) if arr is not None else None, "routing_coverage": cov}


def undo_token_mean(datums: Sequence[TokenDatum]) -> float:
    """The factor that cancels prime-rl's division by the step's trainable-token count."""
    return float(sum(d.n_train for d in datums) or 1)


def samples(datums: Sequence[TokenDatum], *, temperature: float, env_name: str = "crucible", routing: Any = None, topk: Any = None) -> list[dict[str, Any]]:
    k = undo_token_mean(datums)
    return [to_sample(d, temperature=temperature, env_name=env_name, scale=k, routing=routing, topk=topk) for d in datums]


def simple_pack(rows: Sequence[dict[str, Any]], *, ranks: int) -> list[list[dict[str, Any]]]:
    """Round robin, one sample per micro batch; every rank gets the same number of micro batches (an empty
    ``{"dummy": True}`` pads), as FSDP requires. Tests and dry runs only."""
    import base64

    grid: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
    for i, r in enumerate(rows):
        row = dict(r)
        if isinstance(row.get("routed_experts"), dict):  # JSON: the bytes base64 (the real wire carries them raw)
            row["routed_experts"] = {**row["routed_experts"], "data": base64.b64encode(row["routed_experts"]["data"]).decode()}
        if isinstance(row.get("sc_heads"), dict):
            row["sc_heads"] = {k: base64.b64encode(v).decode() for k, v in row["sc_heads"].items()}
        grid[i % ranks].append(row)
    n = max(len(g) for g in grid) if grid else 0
    for g in grid:
        g.extend({"dummy": True} for _ in range(n - len(g)))
    return grid


def replay_all_or_none(rows: Sequence[dict[str, Any]]) -> list[Any]:
    """The routes the step sends: all its sequences', or none. prime-rl packs one routing stream per micro batch, and the
    trainer refuses replay with a missing stream. As the torch trainer's R3 replay does (M12 #36), incomplete coverage is
    refused rather than zero-filled into a wrong expert: a step where any sequence lacks a route for a forwarded position
    recomputes the router for all."""
    rx = [r.get("routed_experts") for r in rows]
    ok = all(x is not None and float(r.get("routing_coverage", 1.0)) >= 1.0 for x, r in zip(rx, rows, strict=True))
    return rx if ok else [None] * len(rows)


def pack(rows: Sequence[dict[str, Any]], *, seq_len: int, ranks: int, model: str, pad_to_multiple_of: int = 1) -> list[list[Any]]:
    """prime-rl's ``prepare_batch`` over ``TrainingSample`` structs (module doc). Raises ImportError without prime-rl."""
    from prime_rl.orchestrator.batch import build_bin_cost, prepare_batch  # type: ignore[import-not-found]
    from prime_rl.transports.batch.types import RoutedExperts, SamplingMask, TrainingSample  # type: ignore[import-not-found]
    from transformers import AutoConfig

    from crucible.train.bridges.primerl.topk import all_or_none

    too_long = [r["trace_id"] for r in rows if len(r["token_ids"]) > seq_len]
    if too_long:
        raise ValueError(f"{len(too_long)} sequences exceed seq_len {seq_len} (raise --seq-len or lower the episode budget): {too_long[:3]}")
    rx = replay_all_or_none(rows)
    hx = all_or_none(rows) if any("sc_heads" in r for r in rows) else [None] * len(rows)
    structs = [TrainingSample(token_ids=r["token_ids"], mask=r["mask"], logprobs=r["logprobs"], temperatures=r["temperatures"],
                              env_name=r["env_name"], advantages=r["advantages"], trace_id=r["trace_id"], branch_index=r["branch_index"],
                              routed_experts=RoutedExperts(data=x["data"], shape=x["shape"], dtype=x["dtype"]) if x is not None else None,
                              sampling_mask=SamplingMask(ids=h["ids"], counts=h["counts"]) if h is not None else None)
               for r, x, h in zip(rows, rx, hx, strict=True)]
    return prepare_batch(rollouts=structs, seq_len=seq_len, num_train_workers=ranks, bin_cost=build_bin_cost(AutoConfig.from_pretrained(model)),
                         pad_to_multiple_of=pad_to_multiple_of)
