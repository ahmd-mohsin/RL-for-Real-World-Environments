"""E10: do the regularisers matter? (part2-paper-core §9.6, §13). One regulariser removed at a time, at equal budget,
over several seeds.

| ablation | what is switched off (``rsi.ABLATIONS``) |
|---|---|
| none | nothing (the reference arm) |
| R1 | fresh views: every decision reuses one view (and its units) |
| R3 | per-class FR, P3 and P4 non-regression (the archive and FA stay: they are validity, not regularisation) |
| R4 | the cell-KL trust region and the entropy floor |
| R5 | the edit budget, the cost bar and pruning |
| R7 | the transfer requirement |
| R8 | the judge-only round cap |
| NL | the planter's natural-likeness bar (MMD ≤ μ) |

**Equal budget.** Every arm runs the same rounds, the same number of operator and planter candidates per round and
the same seeds; the spend of each arm is recorded and compared.

**Metrics per (ablation, seed)**, from the loop arm's last round:
- J: the head's recall on the whole dev pool (the F loop's J proxy);
- locked FR: the population's recall on the locked set (report only);
- transfer: recall on the held-out World family when one is given, else on the locked plants of domains outside the
  evidence;
- Goodhart drift: KL of the dev pool's class mix against round 0 (the F/P analogue of cell KL), the plant size (contract
  diff, witness length: the analogue of task size), the pool's MMD to the real-defect reference, the library size;
- accepted updates and the α spent.

``e10_table`` reports per ablation the mean, sd and a 95% t interval over seeds, and the difference to ``none`` paired by
seed.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.meta.rsi import ABLATIONS, EvolveConfig, Loop, Proposers, RoundRow

METRICS = ("j", "locked_fr", "transfer", "pool_kl", "diff_size", "witness_len", "pool_mmd", "library", "accepted", "spend")


class E10Row(CrucibleModel):
    ablation: str
    seed: int
    rounds: int
    metrics: dict[str, float] = Field(default_factory=dict)
    stop: str | None = None


def _transfer(row: RoundRow, evidence: Sequence[str] | None) -> float:
    if row.family:
        return float(row.family.get("fr_all", 0.0))
    pd = row.locked.get("per_domain", {})
    hit = n = 0
    for dom, by in pd.items():
        if evidence and dom in evidence:
            continue
        for c in by.values():
            hit += c["hit"]
            n += c["n"]
    return hit / n if n else float(row.locked.get("fr_all", 0.0))


def row_metrics(rows: Sequence[RoundRow], evidence: Sequence[str] | None = None, spend: float = 0.0) -> dict[str, float]:
    loop = [r for r in rows if r.arm == "loop"]
    last = loop[-1]
    size = last.game.get("size", {})
    return {"j": float(last.dev.get("recall", 0.0)), "locked_fr": float(last.locked.get("fr_all", 0.0)), "transfer": _transfer(last, evidence),
            "pool_kl": float(last.game.get("pool_kl", 0.0)), "diff_size": float(size.get("diff_size", 0.0)), "witness_len": float(size.get("witness_len", 0.0)),
            "pool_mmd": float(last.game.get("pool_mmd", 0.0)), "library": float(len(last.library)),
            "accepted": float(sum(1 for r in loop for c in r.candidates if c.accepted)), "spend": spend}


MakeLoop = Callable[[EvolveConfig, Path], Loop]
MakeProposers = Callable[[int], Proposers]


async def run_e10(make_loop: MakeLoop, proposers: MakeProposers, base: EvolveConfig, root: Path, *, ablations: Sequence[str] = tuple(ABLATIONS),
                  seeds: Sequence[int] = (0, 1, 2), spend: Callable[[], float] | None = None,
                  on_row: Callable[[E10Row], Awaitable[None] | None] | None = None) -> list[E10Row]:
    """Every (ablation, seed): a fresh loop arm (no control arm: the comparison is across ablations at equal budget)."""
    if len(seeds) < 3:
        raise ValueError("E10 needs at least 3 seeds")
    out = []
    for name in ablations:
        for s in seeds:
            cfg = base.model_copy(update={"seed": s}).ablated(name)
            loop = make_loop(cfg, Path(root) / f"{name}-s{s}")
            before = spend() if spend else 0.0
            rows = await loop.run(proposers(s), control=False)
            r = E10Row(ablation=name, seed=s, rounds=max(x.round for x in rows), metrics=row_metrics(rows, cfg.evidence_domains, (spend() - before) if spend else 0.0),
                       stop=next((x.stop for x in reversed(rows) if x.stop), None))
            out.append(r)
            if on_row is not None:
                res = on_row(r)
                if res is not None:
                    await res
    return out


_T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def _ci(xs: Sequence[float]) -> tuple[float, float, float, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    h = _T95.get(n - 1, 1.96) * sd / math.sqrt(n) if n > 1 else math.inf
    return m, sd, m - h, m + h


def e10_summary(rows: Sequence[E10Row]) -> dict[str, dict[str, Any]]:
    by: dict[str, dict[int, E10Row]] = {}
    for r in rows:
        by.setdefault(r.ablation, {})[r.seed] = r
    ref = by.get("none", {})
    out: dict[str, dict[str, Any]] = {}
    for name, rs in by.items():
        d: dict[str, Any] = {"seeds": sorted(rs)}
        for k in METRICS:
            xs = [rs[s].metrics.get(k, 0.0) for s in sorted(rs)]
            m, sd, lo, hi = _ci(xs)
            d[k] = {"mean": round(m, 4), "sd": round(sd, 4), "ci95": [round(lo, 4), round(hi, 4)]}
            common = [s for s in sorted(rs) if s in ref]
            if name != "none" and len(common) >= 2:
                diffs = [rs[s].metrics.get(k, 0.0) - ref[s].metrics.get(k, 0.0) for s in common]
                dm, _dsd, dlo, dhi = _ci(diffs)
                d[k]["vs_none"] = {"mean": round(dm, 4), "ci95": [round(dlo, 4), round(dhi, 4)]}
        out[name] = d
    return out


def e10_table(rows: Sequence[E10Row]) -> str:
    s = e10_summary(rows)
    cols = ("j", "locked_fr", "transfer", "pool_kl", "diff_size", "library", "accepted")
    lines = ["| ablation | seeds | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for name, d in s.items():
        cells = []
        for k in cols:
            x = d[k]
            c = f"{x['mean']:.3f} ± {x['sd']:.3f}"
            if "vs_none" in x:
                c += f" (Δ {x['vs_none']['mean']:+.3f} [{x['vs_none']['ci95'][0]:+.3f}, {x['vs_none']['ci95'][1]:+.3f}])"
            cells.append(c)
        lines.append(f"| {name} | {len(d['seeds'])} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"
