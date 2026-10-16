"""Transfer proxy on CPU (05 §8.4, R9): does the family teach its intent on held-out probes?

The base policy is measured on the family's held-out probe set (≥ 10 tasks on a surface the family never uses, sharing no
World, template or verifier code). Then the same policy is measured again with the family's experience: K solved family
traces in context (``exemplar_block``), or best-of-n with the family verifier selecting (``best_of_n``). The proxy holds
when:

1. **paired gain**: per probe, treated rate minus base rate; the t interval over probes (``bench.metrics.task_mean_ci``,
   probes are the independent units) lies above 0. An exact sign test is reported too;
2. **not surface**: each probe's overlap with the family (prompt shingle Jaccard against the exemplars, and the share of its
   surface values the family uses) is measured; the gain must survive on the low-overlap half (mean gain > 0);
3. **not generic context** (optional placebo arm: K solved traces of an unrelated family): the treated arm beats the
   placebo by the same paired test.

A proxy only: a small training run replaces it on GPU later.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.factory.floors import call, graded_from


@dataclass
class ProbeTask:
    id: str
    prompt: str
    surface: dict[str, str] = field(default_factory=dict)
    world: str = ""
    template: str = ""
    verifier: str = ""


@dataclass
class TransferConfig:
    attempts: int = 4  # per probe per arm
    min_probes: int = 10  # v1
    alpha: float = 0.05
    base_max: float | None = None  # when set, the base policy's mean probe rate must be at most this (it "fails the probe set")


def disjoint(probes: Sequence[ProbeTask], family: Sequence[ProbeTask], *, held_out_axis: str | None = None) -> list[str]:
    """Problems with the probe set's separation from the family (shared World, template or verifier; a held-out surface
    value the family uses)."""
    out = []
    for attr in ("world", "template", "verifier"):
        shared = sorted({getattr(p, attr) for p in probes if getattr(p, attr)} & {getattr(f, attr) for f in family if getattr(f, attr)})
        if shared:
            out.append(f"probes share {attr} with the family: {shared}")
    if held_out_axis:
        used = {f.surface.get(held_out_axis) for f in family}
        clash = sorted({p.id for p in probes if p.surface.get(held_out_axis) in used})
        if clash:
            out.append(f"probes on a {held_out_axis} the family uses: {clash}")
    return out


def exemplar_block(traces: Sequence[str], k: int, *, header: str = "Solved examples of related tasks:") -> str:
    """K solved family traces as an in-context block."""
    picked = list(traces)[:k]
    return header + "\n\n" + "\n\n".join(f"Example {i + 1}:\n{t}" for i, t in enumerate(picked)) if picked else ""


async def best_of_n(sample: Callable[[int], Any], score: Callable[[Any], Any], grade: Callable[[Any], Any], n: int) -> bool | None:
    """Draw ``n`` candidates, keep the one the *family* verifier scores highest (``score``), and grade it with the
    probe's own verifier (``grade``). None when nothing could be graded."""
    cands = [await call(sample, i) for i in range(n)]
    cands = [c for c in cands if c is not None]
    if not cands:
        return None
    scores = [graded_from(await call(score, c)) for c in cands]
    best = max(range(len(cands)), key=lambda i: (scores[i].reward if scores[i].finite else -math.inf, -i))
    g = graded_from(await call(grade, cands[best]))
    return g.passed() if not g.masked else None


def overlap(probe: ProbeTask, family: Sequence[ProbeTask]) -> dict:
    from crucible.forge.dedup import jaccard, shingles

    lex = max((jaccard(list(shingles(probe.prompt)), list(shingles(f.prompt))) for f in family), default=0.0)
    axes = list(probe.surface)
    used = {ax: {f.surface.get(ax) for f in family} for ax in axes}
    surf = sum(1 for ax in axes if probe.surface[ax] in used[ax]) / len(axes) if axes else 0.0
    return {"lexical": round(lex, 4), "surface": round(surf, 4), "score": round((lex + surf) / 2, 4)}


def _sign_p(pos: int, neg: int) -> float:
    """One-sided exact sign test: P(X >= pos) for X ~ Bin(pos + neg, 1/2)."""
    n = pos + neg
    return 1.0 if n == 0 else sum(math.comb(n, j) for j in range(pos, n + 1)) / 2**n


def paired(deltas: Sequence[float], *, alpha: float = 0.05) -> dict:
    from crucible.bench.metrics import task_mean_ci

    ci = task_mean_ci(list(deltas), alpha=alpha)
    pos, neg = sum(1 for d in deltas if d > 0), sum(1 for d in deltas if d < 0)
    mean = statistics.fmean(deltas) if deltas else None
    return {"mean": None if mean is None else round(mean, 6), "ci": ci, "positive": pos, "negative": neg, "sign_p": round(_sign_p(pos, neg), 6),
            "improved": bool(ci and ci["lo"] > 0)}


async def _rates(probes: Sequence[ProbeTask], run: Callable[[ProbeTask, int], Any], attempts: int) -> dict[str, float | None]:
    out = {}
    for p in probes:
        res = [await call(run, p, a) for a in range(attempts)]
        meas = [bool(r) for r in res if r is not None]
        out[p.id] = sum(meas) / len(meas) if meas else None
    return out


async def transfer_proxy(probes: Sequence[ProbeTask], run_base: Callable[[ProbeTask, int], Any], run_treated: Callable[[ProbeTask, int], Any], *,
                         family: Sequence[ProbeTask] = (), run_placebo: Callable[[ProbeTask, int], Any] | None = None,
                         config: TransferConfig | None = None) -> dict:
    """``run_*(probe, attempt) -> bool | None`` (None: masked). Returns the evidence and the verdict."""
    cfg = config or TransferConfig()
    base = await _rates(probes, run_base, cfg.attempts)
    treated = await _rates(probes, run_treated, cfg.attempts)
    placebo = await _rates(probes, run_placebo, cfg.attempts) if run_placebo is not None else None
    ids = [p.id for p in probes if base[p.id] is not None and treated[p.id] is not None and (placebo is None or placebo[p.id] is not None)]
    deltas = {i: treated[i] - base[i] for i in ids}  # type: ignore[operator]
    main = paired(list(deltas.values()), alpha=cfg.alpha)
    reasons = []
    if len(ids) < cfg.min_probes:
        reasons.append(f"{len(ids)} measured probes, need {cfg.min_probes}")
    if not main["improved"]:
        reasons.append(f"no paired gain on the probes (mean {main['mean']}, interval {main['ci'] and [main['ci']['lo'], main['ci']['hi']]})")
    base_mean = statistics.fmean([base[i] for i in ids]) if ids else None  # type: ignore[misc]
    if cfg.base_max is not None and base_mean is not None and base_mean > cfg.base_max:
        reasons.append(f"the base policy already solves the probes (mean {base_mean:.2f} > {cfg.base_max})")
    surf: dict[str, Any] = {}
    if family:
        ov = {p.id: overlap(p, family) for p in probes if p.id in deltas}
        cut = statistics.median([o["score"] for o in ov.values()]) if ov else 0.0
        low = [deltas[i] for i, o in ov.items() if o["score"] <= cut]
        high = [deltas[i] for i, o in ov.items() if o["score"] > cut]
        low_mean = statistics.fmean(low) if low else None
        surf = {"overlap": ov, "median": cut, "low_overlap_gain": low_mean, "high_overlap_gain": statistics.fmean(high) if high else None}
        if low_mean is None or low_mean <= 0:
            reasons.append(f"the gain is explained by surface overlap: low-overlap probes gain {low_mean}")
    else:
        reasons.append("no family given: the surface-overlap check could not run")
    vs_placebo = None
    if placebo is not None:
        vs_placebo = paired([treated[i] - placebo[i] for i in ids], alpha=cfg.alpha)  # type: ignore[operator]
        if not vs_placebo["improved"]:
            reasons.append(f"no gain over the placebo arm (unrelated traces in context): mean {vs_placebo['mean']}")
    return {"ok": not reasons, "reasons": reasons, "probes": len(ids), "base": base, "treated": treated, "placebo": placebo,
            "base_mean": base_mean, "paired": main, "surface": surf, "vs_placebo": vs_placebo, "attempts": cfg.attempts}
