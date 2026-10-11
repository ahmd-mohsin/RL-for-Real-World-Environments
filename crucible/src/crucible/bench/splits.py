"""Benchmark splits (DESIGN §11.2–11.4): split by group, never by row; salted, stable assignment; held-out time blocks;
OOD axes disjoint by construction; a sealed manifest; and per-model temporal eligibility.

- **Group key.** A task's key is its lineage root (``forge.curator``'s sealing unit) plus any axis group the caller adds:
  siblings of one seed never straddle train and bench. Row-level ``train_test_split(seed=42)`` (NeMo-Gym ``xlam_fc``,
  FineEnvs SFT scripts) leaks siblings and is the anti-pattern.
- **Assignment.** ``u = HMAC-SHA256(salt, key) / 2^256``; bench iff ``u < bench_frac``. Stable when items are added, and
  unpredictable without the salt (FineEnvs orders by ``sha1(task_id)``; prime-agent shards by ``crc32``).
- **Time axis.** Contiguous held-out blocks after ``time_cutoff``, with a guard gap so neighbours of a held-out block do
  not train (FineEnvs portsim ``dock.py:31-62``).
- **OOD axes.** ``ood={"repo": {"pcl"}}``: tasks whose axis value is held out go to ``ood:<axis>``, and the check proves
  no train task shares that value.
- **Showcase** tasks (tasks 01–04) are never in a published split (ARCHITECTURE §6).
- **Seal.** ``sha256(sorted bench ids + sha256(salt))`` is written before any builder sees test traces; ``verify_seal``
  re-derives it.
- **Temporal eligibility.** An item derived from a public trace counts for a model only if ``source_time >= cutoff +
  margin`` (half-open, NeMo-Gym LiveCodeBench ``prepare_utils.py:127``); an undated public-derived item is excluded.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any


@dataclass
class SplitItem:
    id: str
    lineage: str = ""  # the lineage root; defaults to the id
    axes: dict[str, str] = field(default_factory=dict)  # repo, api_family, horizon, harness, ...
    source_time: str | None = None  # RFC 3339, the public trace it derives from
    source_kind: str = "synthetic"  # pr | issue | cve | log | synthetic
    showcase: bool = False


def _u(salt: bytes, key: str) -> float:
    return int.from_bytes(hmac.new(salt, key.encode(), hashlib.sha256).digest(), "big") / 2**256


def _t(x: str) -> datetime:
    return datetime.fromisoformat(x)


def assign(items: Sequence[SplitItem], *, salt: bytes, bench_frac: float = 0.2, ood: Mapping[str, Iterable[str]] | None = None,
           time_cutoff: str | None = None, guard_days: float = 0.0) -> dict[str, Any]:
    """{"splits": {id: split}, "seal": hash, "counts": ...}. Splits: train | bench | ood:<axis> | time | showcase | guard."""
    ood = {k: set(v) for k, v in (ood or {}).items()}
    cut = _t(time_cutoff) if time_cutoff else None
    out: dict[str, str] = {}
    for it in sorted(items, key=lambda x: x.id):
        group = it.lineage or it.id
        if it.showcase:
            out[it.id] = "showcase"
            continue
        axis = next((a for a, held in sorted(ood.items()) if it.axes.get(a) in held), None)
        if axis:
            out[it.id] = f"ood:{axis}"
            continue
        if cut is not None and it.source_time:
            t = _t(it.source_time)
            if t >= cut:
                out[it.id] = "time"
                continue
            if t >= cut - timedelta(days=guard_days):
                out[it.id] = "guard"  # too close to the held-out block to train on, too early to test on
                continue
        out[it.id] = "bench" if _u(salt, group) < bench_frac else "train"
    # a lineage straddling splits would leak: every item of a lineage takes the split of its first (sorted) item
    first: dict[str, str] = {}
    for it in sorted(items, key=lambda x: x.id):
        g = it.lineage or it.id
        if out[it.id] in ("train", "bench"):
            out[it.id] = first.setdefault(g, out[it.id])
    held = sorted(i for i, s in out.items() if s != "train" and s != "guard")
    counts: dict[str, int] = {}
    for s in out.values():
        counts[s] = counts.get(s, 0) + 1
    return {"splits": out, "seal": seal(held, salt), "counts": dict(sorted(counts.items()))}


def seal(held_ids: Sequence[str], salt: bytes) -> str:
    return hashlib.sha256(("\n".join(sorted(held_ids)) + "\n" + hashlib.sha256(salt).hexdigest()).encode()).hexdigest()


def verify_seal(splits: Mapping[str, str], salt: bytes, expected: str) -> bool:
    return seal([i for i, s in splits.items() if s not in ("train", "guard")], salt) == expected


def check(items: Sequence[SplitItem], splits: Mapping[str, str], *, ood: Mapping[str, Iterable[str]] | None = None) -> list[str]:
    """Invariant violations: a lineage in two of {train, held-out}; a train task sharing a held-out OOD axis value; a
    showcase task in a published split."""
    problems = []
    by_lineage: dict[str, set[str]] = {}
    for it in items:
        by_lineage.setdefault(it.lineage or it.id, set()).add("train" if splits[it.id] == "train" else ("guard" if splits[it.id] == "guard" else "held"))
    problems += [f"lineage {g} straddles train and held-out" for g, s in sorted(by_lineage.items()) if {"train", "held"} <= s]
    for axis, held in (ood or {}).items():
        problems += [f"train task {it.id} shares held-out {axis}={it.axes.get(axis)}" for it in items
                     if splits[it.id] == "train" and it.axes.get(axis) in set(held)]
    problems += [f"showcase task {it.id} is in split {splits[it.id]}" for it in items if it.showcase and splits[it.id] != "showcase"]
    return problems


def eligible(item: SplitItem, model_cutoff: str | None, *, margin_days: float = 0.0) -> bool:
    """Temporal decontamination: a public-derived item counts for a model only if it post-dates the model's cutoff."""
    if item.source_kind == "synthetic" and item.source_time is None:
        return True  # nothing public to have trained on
    if item.source_time is None or model_cutoff is None:
        return item.source_time is not None and model_cutoff is None
    return _t(item.source_time) >= _t(model_cutoff) + timedelta(days=margin_days)


def eligible_subsets(items: Sequence[SplitItem], cutoffs: Mapping[str, str | None], *, margin_days: float = 0.0) -> dict[str, list[str]]:
    """Per model, the ids it may be scored on (report the size; compare models on IRT ability, not raw pass rates)."""
    return {m: sorted(it.id for it in items if eligible(it, c, margin_days=margin_days)) for m, c in sorted(cutoffs.items())}
