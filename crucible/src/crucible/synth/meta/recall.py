"""Falsifier recall on planted sets, and the residual-defect estimates it calibrates (part2-paper-core §6, §8; E1, E8).

**The matrix.** ``measure(plants, anchors, falsifiers, ctx)`` runs the falsifier population on every planted task, in
the scheduled order (``falsify.base.run_classes``: REP first, gating MUT/TWIN/PERT), and on every certified anchor:
- ``FR_k,d`` = the fraction of planted tasks of class d on which class k finds a **confirmed** counterexample (any
  label: the source task was certified, so whatever k confirms is the plant), with a Wilson 95% interval. Two more
  counts ride along: ``matched`` (a confirmed counterexample *labelled* d) and ``suspects`` (only suspects; these need
  adjudication and never count as recall).
- ``FR_d`` (the population row) = some class confirmed.
- ``FA_k`` = the fraction of anchors on which class k confirms anything, with its Wilson interval. An alarm is either
  a real defect in the anchor (adjudicate, keep, repair) or a falsifier bug; either way it is reported, never dropped.

**Residual defects (error seeding, Mills).** With n_d natural defects of class d confirmed among candidates and recall
FR_d on the planted set, the residual among admitted tasks is ``n_d (1 − FR_d) / FR_d``. Its interval is the delta
method (``SE = n_d / FR² · sqrt(FR (1 − FR) / m)``, clipped at 0) or a seeded parametric bootstrap over FR's binomial
(``method="bootstrap"``; an FR* of 0 draws an infinite residual, which the percentile keeps). FR = 0 gives an unbounded
estimate (``None``; models hold finite floats only): the class is unmeasured, not clean.

**Capture–recapture.** Two classes with no shared mechanism that both target d find natural-defect sets A and B with
overlap m: Lincoln–Petersen ``N̂ = n_A n_B / m`` and the Chapman correction ``(n_A + 1)(n_B + 1)/(m + 1) − 1`` with its
variance ``(n_A+1)(n_B+1)(n_A−m)(n_B−m) / ((m+1)²(m+2))`` and a normal interval floored at the number found.

**Dev and locked sets.** A ``PlantSet`` is a named, digested list of plants with ``kind`` ``dev`` or ``locked``.
``SetStore`` keeps dev sets under ``dev_root`` and locked sets under ``locked_root`` (the caller's path; in the repo,
``crucible/data/sealed/synth-locked/`` behind the seal guard). Any read of a locked set, through the store or by path
(``load_set``), raises ``LockedSetError`` unless ``purpose="report"``; ``measure`` refuses locked plants unless
``purpose="report"``; and ``update.py`` refuses a view that carries one. The loop cannot read the yardstick it is
scored on (R1).
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, run_classes
from crucible.synth.meta.plant import Plant
from crucible.synth.task import Task

Z95 = 1.959964
SetKind = Literal["dev", "locked"]
Purpose = Literal["loop", "report"]


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    """Wilson score interval for k successes in n (``(0, 1)`` at n = 0: nothing measured)."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


class Cell(CrucibleModel):
    hits: int = 0
    n: int = 0
    matched: int = 0
    suspects: int = 0
    lo: float = 0.0
    hi: float = 1.0

    @property
    def rate(self) -> float:
        return self.hits / self.n if self.n else 0.0

    def close(self) -> Cell:
        lo, hi = wilson(self.hits, self.n)
        return self.model_copy(update={"lo": lo, "hi": hi})


class PlantOutcome(CrucibleModel):
    plant_id: str
    defect: str
    op: str
    domain: str
    hit: list[str] = Field(default_factory=list)  # classes with a confirmed counterexample
    matched: list[str] = Field(default_factory=list)  # ... labelled with the planted class
    suspect_only: list[str] = Field(default_factory=list)
    not_applicable: list[str] = Field(default_factory=list)


class RecallMatrix(CrucibleModel):
    schema_version: str = "crucible.synth.recall/v1"
    set_name: str = ""
    set_kind: SetKind = "dev"
    set_digest: str = ""
    classes: list[str] = Field(default_factory=list)
    defects: list[str] = Field(default_factory=list)
    fr: dict[str, dict[str, Cell]] = Field(default_factory=dict)  # class -> defect -> cell
    population: dict[str, Cell] = Field(default_factory=dict)  # defect -> cell (any class)
    fa: dict[str, Cell] = Field(default_factory=dict)  # class -> anchors with a confirmed alarm
    fa_population: Cell = Field(default_factory=Cell)
    plants: list[PlantOutcome] = Field(default_factory=list)
    anchors: dict[str, list[str]] = Field(default_factory=dict)  # anchor id -> classes that alarmed

    def rate(self, cls: str, defect: str) -> float:
        return self.fr[cls][defect].rate


class LockedSetError(PermissionError):
    """A loop component tried to read the locked planted set (R1: it is read for reporting only)."""


def _defect_key(d: str) -> int:
    return int(d[1:]) if d[1:].isdigit() else 99


async def measure(plants: Sequence[Plant], anchors: Sequence[Task], falsifiers: Sequence[BaseFalsifier], ctx: FalsifyContext, *,
                  set_name: str = "", set_kind: SetKind = "dev", purpose: Purpose = "loop") -> RecallMatrix:
    """The FR/FA matrix (see the module docstring)."""
    if set_kind == "locked" and purpose != "report":
        raise LockedSetError("the locked set is measured for reporting only (purpose='report')")
    classes = [f.name for f in sorted(falsifiers, key=lambda f: f.name != "REP")]
    defects = sorted({p.defect for p in plants}, key=_defect_key)
    fr = {k: {d: Cell() for d in defects} for k in classes}
    pop = {d: Cell() for d in defects}
    outcomes = []
    for p in plants:
        res = {r.cls: r for r in await run_classes(p.task, falsifiers, ctx, cache=True)}
        po = PlantOutcome(plant_id=p.id, defect=p.defect, op=p.op.key, domain=p.domain)
        for k in classes:
            r = res[k]
            cell = fr[k][p.defect]
            cell.n += 1
            if not r.applicable:
                po.not_applicable.append(k)
                continue
            if r.c:
                cell.hits += 1
                po.hit.append(k)
                if any(f.defect == p.defect for f in r.confirmed):
                    cell.matched += 1
                    po.matched.append(k)
            elif r.suspects:
                cell.suspects += 1
                po.suspect_only.append(k)
        c = pop[p.defect]
        c.n += 1
        c.hits += bool(po.hit)
        c.matched += bool(po.matched)
        c.suspects += bool(po.suspect_only) and not po.hit
        outcomes.append(po)
    fa = {k: Cell() for k in classes}
    fa_pop = Cell()
    alarms: dict[str, list[str]] = {}
    for a in anchors:
        res = {r.cls: r for r in await run_classes(a, falsifiers, ctx, cache=True)}
        hit = [k for k in classes if res[k].applicable and res[k].c]
        for k in classes:
            fa[k].n += 1
            fa[k].hits += k in hit
            fa[k].suspects += bool(res[k].applicable and res[k].suspects)
        fa_pop.n += 1
        fa_pop.hits += bool(hit)
        alarms[a.id] = hit
    return RecallMatrix(set_name=set_name, set_kind=set_kind, set_digest=set_digest(plants), classes=classes, defects=defects,
                        fr={k: {d: c.close() for d, c in row.items()} for k, row in fr.items()}, population={d: c.close() for d, c in pop.items()},
                        fa={k: c.close() for k, c in fa.items()}, fa_population=fa_pop.close(), plants=outcomes, anchors=alarms)


# ---------------------------------------------------------------------------------------------------------- estimates

class Residual(CrucibleModel):
    """``None`` is +∞ (models hold finite floats only): FR = 0, or a bootstrap tail that drew FR* = 0."""

    defect: str
    n_found: int
    fr: float
    fr_n: int
    estimate: float | None
    lo: float | None
    hi: float | None
    method: str

    @property
    def unbounded(self) -> bool:
        return self.estimate is None


def _fin(x: float) -> float | None:
    return x if math.isfinite(x) else None


def residual(defect: str, n_found: int, hits: int, n: int, *, method: Literal["delta", "bootstrap"] = "delta", z: float = Z95, draws: int = 4000,
             seed: int = 0) -> Residual:
    """Error-seeding residual ``n_d (1 − FR) / FR`` with its interval (see the module docstring)."""
    fr = hits / n if n else 0.0
    if fr <= 0:
        return Residual(defect=defect, n_found=n_found, fr=fr, fr_n=n, estimate=None if n_found else 0.0, lo=None if n_found else 0.0, hi=None if n_found else 0.0,
                        method=method)
    est = n_found * (1 - fr) / fr
    if method == "delta":
        se = n_found / (fr * fr) * math.sqrt(fr * (1 - fr) / n)
        return Residual(defect=defect, n_found=n_found, fr=fr, fr_n=n, estimate=est, lo=max(0.0, est - z * se), hi=est + z * se, method=method)
    rng = random.Random(f"residual/{defect}/{seed}")
    vals = []
    for _ in range(draws):
        f = rng.binomialvariate(n, fr) / n
        vals.append(math.inf if f <= 0 else n_found * (1 - f) / f)
    vals.sort()
    cover = math.erf(z / math.sqrt(2))  # the two-sided coverage of +-z
    lo_i, hi_i = int((1 - cover) / 2 * draws), min(draws - 1, int((1 + cover) / 2 * draws))
    return Residual(defect=defect, n_found=n_found, fr=fr, fr_n=n, estimate=est, lo=_fin(vals[lo_i]), hi=_fin(vals[hi_i]), method=method)


class CaptureRecapture(CrucibleModel):
    defect: str
    n_a: int
    n_b: int
    overlap: int
    lincoln_petersen: float | None  # None: no overlap (unbounded)
    chapman: float
    chapman_var: float
    lo: float
    hi: float


def capture_recapture(defect: str, found_a: Iterable[str], found_b: Iterable[str], z: float = Z95) -> CaptureRecapture:
    """Lincoln–Petersen and Chapman estimates of the total natural defects of a class from two classes' finds."""
    a, b = set(found_a), set(found_b)
    na, nb, m = len(a), len(b), len(a & b)
    lp = None if m == 0 else na * nb / m
    ch = (na + 1) * (nb + 1) / (m + 1) - 1
    var = (na + 1) * (nb + 1) * (na - m) * (nb - m) / ((m + 1) ** 2 * (m + 2))
    seen = len(a | b)
    return CaptureRecapture(defect=defect, n_a=na, n_b=nb, overlap=m, lincoln_petersen=lp, chapman=ch, chapman_var=var, lo=max(float(seen), ch - z * math.sqrt(var)),
                            hi=ch + z * math.sqrt(var))


# ---------------------------------------------------------------------------------------------------------- reports

def report(matrix: RecallMatrix, residuals: Sequence[Residual] = (), recaptures: Sequence[CaptureRecapture] = ()) -> dict:
    """The matrix and the estimates as one JSON document (an unbounded value is ``null``)."""
    out = matrix.model_dump(mode="json")
    out["residuals"] = [r.model_dump(mode="json") for r in residuals]
    out["capture_recapture"] = [c.model_dump(mode="json") for c in recaptures]
    return out


def report_json(matrix: RecallMatrix, residuals: Sequence[Residual] = (), recaptures: Sequence[CaptureRecapture] = ()) -> str:
    return json.dumps(report(matrix, residuals, recaptures), indent=1, sort_keys=True, allow_nan=False)


def _f(x: float | None) -> str:
    return "∞" if x is None else f"{x:.3f}"


def _cell(c: Cell) -> str:
    return "·" if c.n == 0 else f"{c.hits}/{c.n} ({c.lo:.2f}–{c.hi:.2f})"


def report_markdown(matrix: RecallMatrix, residuals: Sequence[Residual] = (), recaptures: Sequence[CaptureRecapture] = ()) -> str:
    """The FR/FA table (rows: classes and the population; columns: defect classes; then FA), then the estimates."""
    d = matrix.defects
    lines = [f"### Recall on `{matrix.set_name or 'planted set'}` ({matrix.set_kind}, {len(matrix.plants)} plants, digest {matrix.set_digest[:12]})", "",
             "| class | " + " | ".join(d) + " | FA |", "|---|" + "---|" * (len(d) + 1)]
    for k in matrix.classes:
        lines.append(f"| {k} | " + " | ".join(_cell(matrix.fr[k][x]) for x in d) + f" | {_cell(matrix.fa[k])} |")
    lines.append("| **any** | " + " | ".join(_cell(matrix.population[x]) for x in d) + f" | {_cell(matrix.fa_population)} |")
    if residuals:
        lines += ["", "| defect | natural found | FR | residual | interval | method |", "|---|---|---|---|---|---|"]
        for r in residuals:
            lines.append(f"| {r.defect} | {r.n_found} | {r.fr:.3f} (n={r.fr_n}) | {_f(r.estimate)} | {_f(r.lo)}–{_f(r.hi)} | {r.method} |")
    if recaptures:
        lines += ["", "| defect | n_A | n_B | overlap | Lincoln–Petersen | Chapman | interval |", "|---|---|---|---|---|---|---|"]
        for c in recaptures:
            lines.append(f"| {c.defect} | {c.n_a} | {c.n_b} | {c.overlap} | {_f(c.lincoln_petersen)} | {c.chapman:.2f} | {c.lo:.2f}–{c.hi:.2f} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------- sets

class PlantSet(CrucibleModel):
    schema_version: str = "crucible.synth.plantset/v1"
    name: str
    kind: SetKind = "dev"
    plants: list[Plant] = Field(default_factory=list)
    digest: str = ""

    def sealed(self) -> PlantSet:
        return self.model_copy(update={"digest": set_digest(self.plants)})


def set_digest(plants: Sequence[Plant]) -> str:
    return content_hash(sorted(p.id for p in plants), 32)


_LOCKED_ROOTS: set[Path] = set()


def register_locked_root(root: str | os.PathLike[str]) -> Path:
    """Declare a directory as holding locked sets: every ``load_set`` under it needs ``purpose="report"``."""
    p = Path(root).resolve()
    _LOCKED_ROOTS.add(p)
    return p


def _under_locked(path: Path) -> bool:
    rp = path.resolve()
    return any(rp == r or r in rp.parents for r in _LOCKED_ROOTS)


def load_set(path: str | os.PathLike[str], *, purpose: Purpose = "loop") -> PlantSet:
    p = Path(path)
    if _under_locked(p) and purpose != "report":
        raise LockedSetError(f"{p} is a locked planted set: read it for reporting only (purpose='report')")
    s = PlantSet.model_validate_json(p.read_text())
    if s.kind == "locked" and purpose != "report":
        raise LockedSetError(f"{p} holds a locked set: read it for reporting only (purpose='report')")
    if s.digest and s.digest != set_digest(s.plants):
        raise ValueError(f"{p}: the set digest does not match its plants")
    return s


class SetStore:
    """Dev sets under ``dev_root``; locked sets under ``locked_root`` (registered as locked on construction)."""

    def __init__(self, dev_root: str | os.PathLike[str], locked_root: str | os.PathLike[str] | None = None):
        self.dev_root = Path(dev_root)
        self.locked_root = register_locked_root(locked_root) if locked_root is not None else None
        if self.locked_root is not None and (self.dev_root.resolve() == self.locked_root or self.locked_root in self.dev_root.resolve().parents):
            raise ValueError("the dev root may not lie inside the locked root")

    def path(self, name: str, kind: SetKind) -> Path:
        if not name or "/" in name or name.startswith("."):
            raise ValueError(f"bad set name {name!r}")
        root = self.dev_root if kind == "dev" else self.locked_root
        if root is None:
            raise ValueError("this store has no locked root")
        return root / f"{name}.json"

    def write(self, s: PlantSet) -> Path:
        p = self.path(s.name, s.kind)
        if p.exists():
            raise FileExistsError(f"{p}: planted sets are write-once")
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(".json.tmp")
        tmp.write_text(s.sealed().model_dump_json(indent=1))
        os.replace(tmp, p)
        return p

    def read(self, name: str, kind: SetKind = "dev", *, purpose: Purpose = "loop") -> PlantSet:
        if kind == "locked" and purpose != "report":
            raise LockedSetError("the locked set is read for reporting only (purpose='report')")
        return load_set(self.path(name, kind), purpose=purpose)


def split_sets(plants: Sequence[Plant], *, locked_domains: Iterable[str], locked_ops: Iterable[str], name: str) -> tuple[PlantSet, PlantSet]:
    """Dev and locked sets from one pool (§8): the locked set holds the plants of held-out *operator variants* on
    held-out *domains*; plants of a held-out variant or domain never enter the dev set, so the two share neither."""
    dom, ops = set(locked_domains), set(locked_ops)
    locked = [p for p in plants if p.domain in dom and p.op.key in ops]
    dev = [p for p in plants if p.domain not in dom and p.op.key not in ops]
    return PlantSet(name=f"{name}-dev", kind="dev", plants=dev).sealed(), PlantSet(name=f"{name}-locked", kind="locked", plants=locked).sealed()
