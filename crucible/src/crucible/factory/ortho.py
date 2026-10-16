"""Orthogonality (05 §8.3, R10) in four layers, a duplicate rule, and surface skew.

| layer | measure | built on |
|---|---|---|
| 1 prompt | word-3-shingle Jaccard (lexical); cosine of an injectable ``embed`` (semantic) | ``forge.dedup.shingles`` / ``jaccard`` |
| 2 state | identical (start, oracle end) fingerprints; Jaccard of the oracle footprints | ``forge.dedup`` L2 |
| 3 checks | Jaccard of check signatures | ``forge.dedup`` L3 |
| 4 behaviour | Pearson correlation of pass vectors over the shared panel (policy × attempt); cosine of failure-cause vectors | new |

**Duplicate rule** (FrontierSmith: a duplicate is the same decision structure, not the same topic): same intent *and* the
same dominant failure cause. The forge's layered dedup (content, fingerprint, state identity, check overlap) also drops a
near-copy whatever its intent. The earlier task is kept.

**Skew**: on every surface axis the top value covers ≤ 40% of the kept tasks (*v1*; MiMo shipped 59% one domain).
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from crucible.forge.dedup import Candidate, duplicate_of, jaccard, shingles


@dataclass
class TaskProfile:
    id: str
    intent: str
    prompt: str
    surface: dict[str, str] = field(default_factory=dict)
    start_fp: str | None = None
    end_fp: str | None = None
    footprint: list[str] = field(default_factory=list)  # oracle footprint objects ("tickets.issue_refund:41")
    checks: list[str] = field(default_factory=list)  # canonical check signatures
    passes: dict[str, bool | None] = field(default_factory=dict)  # panel key ("policy#attempt") -> pass
    causes: dict[str, int] = field(default_factory=dict)  # FailureCause -> count over the panel
    family: str = ""

    @property
    def dominant_cause(self) -> str | None:
        if not self.causes or not any(self.causes.values()):
            return None
        top = max(self.causes.values())
        best = sorted(c for c, n in self.causes.items() if n == top)
        return best[0] if len(best) == 1 else "|".join(best)


@dataclass
class OrthoConfig:
    lexical_max: float = 0.8  # forge.dedup j_min
    semantic_max: float = 0.92
    behaviour_corr_max: float = 0.9  # reported as behaviourally redundant (a warning; the duplicate rule decides)
    top_surface_max: float = 0.40  # v1
    skew_axes: tuple[str, ...] | None = None  # None: every axis present
    min_panel: int = 4  # shared panel entries needed for a correlation


def pearson(a: Sequence[float], b: Sequence[float]) -> float | None:
    n = len(a)
    if n < 2:
        return None
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va == 0 or vb == 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def cosine(a: Sequence[float], b: Sequence[float]) -> float | None:
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(y * y for y in b))
    return None if not na or not nb else sum(x * y for x, y in zip(a, b)) / (na * nb)


def pair_scores(a: TaskProfile, b: TaskProfile, *, emb: Mapping[str, Sequence[float]] | None = None, config: OrthoConfig | None = None) -> dict:
    cfg = config or OrthoConfig()
    keys = sorted(k for k in set(a.passes) & set(b.passes) if a.passes[k] is not None and b.passes[k] is not None)
    corr = pearson([float(bool(a.passes[k])) for k in keys], [float(bool(b.passes[k])) for k in keys]) if len(keys) >= cfg.min_panel else None
    cs = sorted(set(a.causes) | set(b.causes))
    return {
        "lexical": round(jaccard(list(shingles(a.prompt)), list(shingles(b.prompt))), 4),
        "semantic": None if emb is None else _round(cosine(emb[a.id], emb[b.id])),
        "same_state": bool(a.start_fp and a.end_fp and (a.start_fp, a.end_fp) == (b.start_fp, b.end_fp)),
        "footprint": round(jaccard(a.footprint, b.footprint), 4),
        "checks": round(jaccard(a.checks, b.checks), 4),
        "pass_corr": _round(corr),
        "panel": len(keys),
        "cause_cos": _round(cosine([a.causes.get(c, 0) for c in cs], [b.causes.get(c, 0) for c in cs])) if cs else None,
    }


def _round(x: float | None) -> float | None:
    return None if x is None else round(x, 4)


def _candidate(p: TaskProfile) -> Candidate:
    return Candidate(id=p.id, instruction=p.prompt, footprint=list(p.footprint), checks=list(p.checks), start_fp=p.start_fp, end_fp=p.end_fp,
                     family=p.family)


def duplicate(a: TaskProfile, b: TaskProfile) -> str | None:
    """The rule: same intent and same dominant failure cause (None if either has no failures to compare)."""
    if a.intent == b.intent and a.dominant_cause is not None and a.dominant_cause == b.dominant_cause:
        return f"same intent {a.intent!r} and failure cause {a.dominant_cause!r}"
    return None


def skew(profiles: Sequence[TaskProfile], *, config: OrthoConfig | None = None) -> dict:
    cfg = config or OrthoConfig()
    axes = cfg.skew_axes or tuple(sorted({k for p in profiles for k in p.surface}))
    out, bad = {}, []
    for ax in axes:
        vals = Counter(p.surface.get(ax, "?") for p in profiles)
        if not vals:
            continue
        top, n = vals.most_common(1)[0]
        share = n / len(profiles)
        out[ax] = {"top": top, "share": round(share, 4), "n": len(profiles)}
        if share > cfg.top_surface_max:
            bad.append(f"surface {ax}={top!r} is {n}/{len(profiles)} ({share:.0%}) > {cfg.top_surface_max:.0%}")
    return {"axes": out, "ok": not bad, "reasons": bad}


def orthogonality(profiles: Sequence[TaskProfile], *, embed: Callable[[list[str]], Sequence[Sequence[float]]] | None = None,
                  config: OrthoConfig | None = None) -> dict:
    """Keep the earliest of each duplicate set; score every pair; report skew on the kept tasks."""
    cfg = config or OrthoConfig()
    emb = None
    if embed is not None:
        vecs = embed([p.prompt for p in profiles])
        emb = {p.id: list(v) for p, v in zip(profiles, vecs)}
    kept: list[TaskProfile] = []
    dropped: dict[str, dict] = {}
    for p in profiles:
        hit = duplicate_of(_candidate(p), [_candidate(k) for k in kept], j_min=cfg.lexical_max)
        if hit:
            dropped[p.id] = {"dup_of": hit[0], "rule": f"forge dedup {hit[1]}"}
            continue
        rule = next(((k.id, r) for k in kept if (r := duplicate(p, k))), None)
        if rule:
            dropped[p.id] = {"dup_of": rule[0], "rule": rule[1]}
            continue
        if emb is not None:
            sem = next((k.id for k in kept if (cosine(emb[p.id], emb[k.id]) or 0.0) >= cfg.semantic_max and k.intent == p.intent), None)
            if sem:
                dropped[p.id] = {"dup_of": sem, "rule": f"semantic near-copy (cosine >= {cfg.semantic_max}) with the same intent"}
                continue
        kept.append(p)
    pairs, redundant = {}, []
    for i, a in enumerate(kept):
        for b in kept[i + 1:]:
            s = pair_scores(a, b, emb=emb, config=cfg)
            pairs[f"{a.id}|{b.id}"] = s
            if s["pass_corr"] is not None and s["pass_corr"] >= cfg.behaviour_corr_max:
                redundant.append([a.id, b.id])
    sk = skew(kept, config=cfg)
    return {"kept": [p.id for p in kept], "dropped": dropped, "pairs": pairs, "behaviourally_redundant": redundant, "skew": sk,
            "ok": sk["ok"], "reasons": sk["reasons"]}
