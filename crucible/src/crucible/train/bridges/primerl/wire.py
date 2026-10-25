"""Crucible's training datums on prime-rl's batch wire (``prime_rl/transports/batch/types.py``, v0.10.0).

A ``TokenDatum`` (the Tinker bridge's unit: ``input_ids`` = full[:-1], logprobs / advantages / mask aligned with
``target_ids`` = full[1:]) becomes one prime-rl ``TrainingSample``: one flat sequence ``token_ids`` = full, with per-token
``mask`` (the policy's own sampled tokens), ``logprobs``, ``temperatures`` and ``advantages``, all of ``token_ids``'
length (prime-rl shifts internally). Token 0 is never a target, so it carries mask False, logprob 0 and advantage 0.

``undo_token_mean``: prime-rl's rl loss is divided by the step's count of trainable tokens (``TrainingSample`` doc:
"each normalized by its own global token count"). The advantages Crucible built are already final (``datums.scale``:
``sum`` or ``token_mean``), so they are multiplied by that count before they leave: the trained objective is the one the
run configured, on either backend.

``pack``: prime-rl's own packer (``orchestrator/batch.py`` ``prepare_batch`` + ``build_bin_cost``: FLOP-balanced bins of
``seq_len`` tokens, one micro-batch list per trainer data rank) when prime-rl is installed; ``simple_pack`` (one sample
per micro batch, round robin over ranks, as plain dicts) only for tests and dry runs, refused for a real trainer.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.train.bridges.tinker.backend import TokenDatum


def to_sample(d: TokenDatum, *, temperature: float, env_name: str = "crucible", scale: float = 1.0) -> dict[str, Any]:
    """One ``TrainingSample`` as a field dict (module doc)."""
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
            "branch_index": int(d.meta["segment"]) if isinstance(d.meta.get("segment"), int) else None}


def undo_token_mean(datums: Sequence[TokenDatum]) -> float:
    """The factor that cancels prime-rl's division by the step's trainable-token count."""
    return float(sum(d.n_train for d in datums) or 1)


def samples(datums: Sequence[TokenDatum], *, temperature: float, env_name: str = "crucible") -> list[dict[str, Any]]:
    k = undo_token_mean(datums)
    return [to_sample(d, temperature=temperature, env_name=env_name, scale=k) for d in datums]


def simple_pack(rows: Sequence[dict[str, Any]], *, ranks: int) -> list[list[dict[str, Any]]]:
    """Round robin, one sample per micro batch; every rank gets the same number of micro batches (an empty
    ``{"dummy": True}`` pads), as FSDP requires. Tests and dry runs only."""
    grid: list[list[dict[str, Any]]] = [[] for _ in range(ranks)]
    for i, r in enumerate(rows):
        grid[i % ranks].append(dict(r))
    n = max(len(g) for g in grid) if grid else 0
    for g in grid:
        g.extend({"dummy": True} for _ in range(n - len(g)))
    return grid


def pack(rows: Sequence[dict[str, Any]], *, seq_len: int, ranks: int, model: str, pad_to_multiple_of: int = 1) -> list[list[Any]]:
    """prime-rl's ``prepare_batch`` over ``TrainingSample`` structs (module doc). Raises ImportError without prime-rl."""
    from prime_rl.orchestrator.batch import build_bin_cost, prepare_batch  # type: ignore[import-not-found]
    from prime_rl.transports.batch.types import TrainingSample  # type: ignore[import-not-found]
    from transformers import AutoConfig

    too_long = [r["trace_id"] for r in rows if len(r["token_ids"]) > seq_len]
    if too_long:
        raise ValueError(f"{len(too_long)} sequences exceed seq_len {seq_len} (raise --seq-len or lower the episode budget): {too_long[:3]}")
    structs = [TrainingSample(token_ids=r["token_ids"], mask=r["mask"], logprobs=r["logprobs"], temperatures=r["temperatures"],
                              env_name=r["env_name"], advantages=r["advantages"], trace_id=r["trace_id"], branch_index=r["branch_index"])
               for r in rows]
    return prepare_batch(rollouts=structs, seq_len=seq_len, num_train_workers=ranks, bin_cost=build_bin_cost(AutoConfig.from_pretrained(model)),
                         pad_to_multiple_of=pad_to_multiple_of)
