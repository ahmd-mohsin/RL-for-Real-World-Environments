"""Layered deduplication of forged tasks (research m7-m8/02 §12.7). Any layer hit marks a duplicate; the earliest
candidate is kept and the earliest layer that matched is reported.

| layer | key |
|---|---|
| L0 content | hash of the task's canonical content (Harbor packager hash + crucible documents) |
| L1 fingerprint | instruction word-3-shingle Jaccard >= ``j_min`` + sorted oracle footprint objects (pattern: Repo2RLEnv ``_oss_instruct.py``; M8 audit V21: an 80-character prefix made every toy task, which share a 170-character intro, compare on the footprint alone) |
| L2 state identity | (start-state fingerprint, oracle end-state fingerprint) |
| L3 check overlap | Jaccard of check signatures >= 0.8 within a family |

Semantic (embedding) and behavioural (per-check pass vectors across calibration rollouts) layers are hooks for the
Curator once calibration data exists. Dedup runs at seed time too: GitSwarm wasted about 1 build in 6 on duplicates.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field


@dataclass
class Candidate:
    id: str
    instruction: str = ""
    footprint: list[str] = field(default_factory=list)
    checks: list[str] = field(default_factory=list)  # canonical check signatures (app, op, args)
    content: str | None = None  # content hash
    start_fp: str | None = None
    end_fp: str | None = None
    family: str = ""


def _norm(text: str, n: int = 80) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())[:n]


def shingles(text: str, k: int = 3) -> set[str]:
    w = _norm(text, 10**9).split()
    return {" ".join(w[i:i + k]) for i in range(len(w) - k + 1)} if len(w) >= k else ({" ".join(w)} if w else set())


def jaccard(a: Sequence[str], b: Sequence[str]) -> float:
    x, y = set(a), set(b)
    return len(x & y) / len(x | y) if x | y else 0.0


def duplicate_of(c: Candidate, kept: Sequence[Candidate], *, j_min: float = 0.8) -> tuple[str, str] | None:
    for k in kept:
        if c.content and c.content == k.content:
            return k.id, "L0_content"
    for k in kept:
        if c.instruction.strip() and sorted(c.footprint) == sorted(k.footprint) and jaccard(list(shingles(c.instruction)), list(shingles(k.instruction))) >= j_min:
            return k.id, "L1_fingerprint"
    for k in kept:
        if c.start_fp and c.end_fp and (c.start_fp, c.end_fp) == (k.start_fp, k.end_fp):
            return k.id, "L2_state"
    for k in kept:
        if c.family == k.family and c.checks and jaccard(c.checks, k.checks) >= j_min:
            return k.id, "L3_checks"
    return None


def dedup(cands: Sequence[Candidate], *, j_min: float = 0.8) -> dict:
    kept: list[Candidate] = []
    dropped: dict[str, dict] = {}
    for c in cands:
        hit = duplicate_of(c, kept, j_min=j_min)
        if hit:
            dropped[c.id] = {"dup_of": hit[0], "layer": hit[1]}
        else:
            kept.append(c)
    return {"kept": [k.id for k in kept], "dropped": dropped}
