"""The regularised recursive self-improvement loop, run end to end (part2-paper-core §9; experiments E1, E2, E3).

    sources   admitted tasks (certified or provisional) of every domain, re-checked clean by the deterministic classes,
              split per domain into dev sources and locked sources (no task is a source of both)
    sets      dev set: ``base_planter`` on the dev sources; locked set: ``held_out_planter`` (disjoint variants) on the
              locked sources, stored under the locked root and read only with purpose="report"
    E1        the FR/FA matrix on the locked set (Wilson intervals), error-seeding residuals from the archive's natural
              defects, capture–recapture between classes with no shared mechanism
    E2        R rounds per arm. The LOOP arm: the swarm abstractor turns natural and miss clusters into DSL operators;
              each passes the birth gate, then the update rule R1–R10 on a fresh dev view; the planter loop (swarm
              planters: witness validity, MMD to the archive's natural defects, fictitious play through R1's views);
              the meta loop (A: the abstractor's guidance, judged by the birth-gate pass rate it produces). The CONTROL
              arm: the same rounds with operator proposals drawn at random from the DSL, the same number per round.
              After every round each arm's falsifier population is measured on the locked set (reporting only).
    E3        the loop grown on one domain's evidence only, measured on the other domain's locked plants

**The J proxy of an F update.** Falsifier changes do not change tasks, so K and B are fixed and J = K(1 − ρ̂)/B moves
only through ρ̂. A view's units are its planted tasks; a unit scores 1 when some class confirms a counterexample on it
(the population's recall on the view: 1 − the miss rate that ρ̂ is estimated from). False alarms on the view's anchors
are R3's hard rule (FA ≤ 0), never traded against recall. Cost is probes × ``PROBE_USD`` (a nominal CPU price, so R5's
"recall per probe-dollar" is defined). The update rule (``update.UpdateEngine``) is used unchanged.

**What a view is (R1).** A seeded, stratified draw of ``view_frac`` of every planter version's dev plants per domain,
plus every dev anchor. M3's "new proposals" are not drawn per view (that would cost a synthesis run per decision); the
views differ by their planted units only, and the ledger keeps every view hash.

Everything model-driven is injected (``Proposers``); tests pass scripted ones, ``scripts/synth.py`` binds the swarm.
"""

from __future__ import annotations

import asyncio
import json
import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar, Literal, Protocol

from pydantic import ConfigDict, Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, content_hash
from crucible.synth.cegis import CegisReport
from crucible.synth.falsify import cheap
from crucible.synth.falsify.base import BaseFalsifier, Counterexample, FalsifyContext, run_classes
from crucible.synth.meta.abstract import Cluster, PlantVerdict, miss_clusters, natural_clusters, random_operator
from crucible.synth.meta.library import Library, OperatorSpec, accept_planter, birth_gate
from crucible.synth.meta.plant import FEATURES, Plant, Planter, PlantOp, base_planter, feature_matrix, held_out_planter, mmd, plant_all
from crucible.synth.meta.recall import PlantSet, RecallMatrix, capture_recapture, measure, residual, set_digest
from crucible.synth.meta.update import (
    Decision,
    Evidence,
    Ledger,
    Measurement,
    RuleFailure,
    ThetaStore,
    Update,
    UpdateConfig,
    UpdateEngine,
    View,
    entropy,
    kl,
    leakage_critic,
)
from crucible.synth.program import Fn, Lit, Program
from crucible.synth.task import Task

PROBE_USD = 1e-6  # nominal price of one executed probe (CPU), for R5's probe-dollars


# ---------------------------------------------------------------------------------------------------------- sources

class SourceInfo(CrucibleModel):
    task_id: str
    domain: str
    run: str
    status: str  # the run's outcome: certified | provisional | rejected
    alarms: list[str] = Field(default_factory=list)  # classes confirming a counterexample on it now (it is then no anchor)
    mut_n: int | None = None


def load_tasks(domain_dir: str | Path, statuses: Iterable[str] = ("certified", "provisional")) -> list[tuple[Task, SourceInfo]]:
    """Every task of a domain's runs whose final outcome is in ``statuses``, as its final (certified) version."""
    want = set(statuses)
    out = []
    root = Path(domain_dir)
    for run in sorted(p for p in root.iterdir() if (p / "report.json").exists()) if root.exists() else []:
        rep = json.loads((run / "report.json").read_text())
        for t in rep.get("tasks", []):
            if t.get("status") not in want:
                continue
            p = run / "tasks" / f"t{int(t['n']):02d}" / "cegis.json"
            if not p.exists():
                continue
            c = CegisReport.model_validate_json(p.read_text())
            if c.task is None:
                continue
            out.append((c.task, SourceInfo(task_id=c.task.id, domain=c.task.domain, run=run.name, status=t["status"])))
    return out


async def recheck(tasks: Sequence[tuple[Task, SourceInfo]], ctx: FalsifyContext, falsifiers: Callable[[], list[BaseFalsifier]] = cheap
                  ) -> tuple[list[Task], list[SourceInfo]]:
    """Re-run the deterministic classes on every candidate source: a task with a confirmed counterexample today (a
    falsifier fixed or added since it was admitted) is no anchor; it is reported, not used."""
    clean, infos = [], []
    for t, info in tasks:
        res = await run_classes(t, falsifiers(), ctx, cache=True)
        info = info.model_copy(update={"alarms": [r.cls for r in res if r.applicable and r.c],
                                       "mut_n": next((r.n for r in res if r.cls == "MUT" and r.applicable), None)})
        infos.append(info)
        if not info.alarms:
            clean.append(t)
    return clean, infos


def split_sources(tasks: Sequence[Task], *, seed: int = 0) -> tuple[list[Task], list[Task]]:
    """Per domain, a seeded alternation into dev and locked sources (a domain with one task gives it to dev)."""
    dev, locked = [], []
    by: dict[str, list[Task]] = {}
    for t in tasks:
        by.setdefault(t.domain, []).append(t)
    for _d, ts in sorted(by.items()):
        ts = sorted(ts, key=lambda t: content_hash({"id": t.id, "seed": seed}, 16))
        for i, t in enumerate(ts):
            (locked if i % 2 == 1 else dev).append(t)
    return dev, locked


async def build_sets(dev_src: Sequence[Task], locked_src: Sequence[Task], ctx: FalsifyContext, *, name: str, dev_planter: Planter | None = None,
                     locked_planter: Planter | None = None) -> tuple[PlantSet, PlantSet, dict[str, Any]]:
    dp, lp = dev_planter or base_planter(), locked_planter or held_out_planter()
    if {o.key for o in dp.ops} & {o.key for o in lp.ops}:
        raise ValueError("the dev and locked planters share operator variants")
    dev = await plant_all(dev_src, dp, ctx)
    locked = await plant_all(locked_src, lp, ctx)
    info = {"dev": {"kept": len(dev.kept), "discarded": len(dev.discarded), "by_op": dict(Counter(p.op.key for p in dev.kept)),
                    "discard_reasons": dict(Counter(d.reason[:60] for d in dev.discarded))},
            "locked": {"kept": len(locked.kept), "discarded": len(locked.discarded), "by_op": dict(Counter(p.op.key for p in locked.kept)),
                       "discard_reasons": dict(Counter(d.reason[:60] for d in locked.discarded))}}
    return (PlantSet(name=f"{name}-dev", kind="dev", plants=dev.kept).sealed(), PlantSet(name=f"{name}-locked", kind="locked", plants=locked.kept).sealed(),
            info)


# ---------------------------------------------------------------------------------------------------------- E1

def defect_instances(cxs: Sequence[Counterexample]) -> dict[str, dict[str, set[str]]]:
    """domain -> defect -> the tasks with a confirmed natural defect of that class (one instance per task and class)."""
    out: dict[str, dict[str, set[str]]] = {}
    for c in cxs:
        if c.label != "confirmed" or not c.defect:
            continue
        out.setdefault(c.domain, {}).setdefault(c.defect, set()).add(c.task_id)
    return out


RECAPTURE_PAIRS: tuple[tuple[str, str, str], ...] = (("D9", "MUT", "TWIN"), ("D4", "RED", "NULL"), ("D1", "MUT", "NULL"), ("D2", "MUT", "NULL"),
                                                     ("D8", "NULL", "ATK"), ("D6", "MUT", "ALT"))


def e1_estimates(m: RecallMatrix, natural: Sequence[Counterexample]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Per domain and defect class: the Mills residual from the locked set's population recall of that class in that
    domain (``m.plants`` outcomes) and the archive's natural defect instances; and the Chapman capture–recapture
    estimate for each class pair of ``RECAPTURE_PAIRS`` (natural finds of two classes with no shared mechanism)."""
    inst = defect_instances(natural)
    res_rows = []
    for dom, by_d in sorted(inst.items()):
        for d, tasks in sorted(by_d.items(), key=lambda kv: int(kv[0][1:])):
            outs = [o for o in m.plants if o.defect == d and o.domain == dom]
            pooled = [o for o in m.plants if o.defect == d]
            use, scope = (outs, dom) if outs else (pooled, "all domains")
            hits = sum(1 for o in use if o.hit)
            r = residual(d, len(tasks), hits, len(use), method="delta")
            res_rows.append({"domain": dom, "fr_scope": scope, **r.model_dump(mode="json")})
    cap_rows = []
    by_task: dict[tuple[str, str, str], set[str]] = {}
    for c in natural:
        if c.label == "confirmed" and c.defect:
            by_task.setdefault((c.domain, c.defect, c.cls), set()).add(c.task_id)
    for dom in sorted(inst):
        for d, a, b in RECAPTURE_PAIRS:
            fa, fb = by_task.get((dom, d, a), set()), by_task.get((dom, d, b), set())
            if fa or fb:
                cap_rows.append({"domain": dom, "classes": [a, b], **capture_recapture(d, fa, fb).model_dump(mode="json")})
    return res_rows, cap_rows


def per_domain(m: RecallMatrix) -> dict[str, dict[str, dict[str, int]]]:
    """domain -> defect -> {n, hit} from the matrix's plant outcomes (E3's restriction)."""
    out: dict[str, dict[str, dict[str, int]]] = {}
    for o in m.plants:
        c = out.setdefault(o.domain, {}).setdefault(o.defect, {"n": 0, "hit": 0})
        c["n"] += 1
        c["hit"] += bool(o.hit)
    return out


def summary(m: RecallMatrix) -> dict[str, Any]:
    pop = {d: {"hit": c.hits, "n": c.n, "lo": round(c.lo, 3), "hi": round(c.hi, 3)} for d, c in m.population.items()}
    tot_n = sum(c.n for c in m.population.values())
    tot_h = sum(c.hits for c in m.population.values())
    return {"fr": pop, "fr_all": round(tot_h / tot_n, 4) if tot_n else 0.0, "hits": tot_h, "n": tot_n, "fa": m.fa_population.hits, "fa_n": m.fa_population.n,
            "fa_by_class": {k: c.hits for k, c in m.fa.items() if c.hits}, "by_class": {k: sum(c.hits for c in row.values()) for k, row in m.fr.items()},
            "per_domain": per_domain(m)}


# ---------------------------------------------------------------------------------------------------------- natural-likeness

_CLASS_TARGET = {"TAINT": "instruction", "RED": "required", "TWIN": "required", "PERT": "required", "REP": "world", "ALT": "trace", "ATK": "world"}
_OP_TARGET = {"ExtraWrite": "allowed", "write_all": "allowed", "shortcut": "world", "random_k": "allowed", "WrongAnswer": "answer"}


def natural_features(cx: Counterexample) -> dict[str, float]:
    """The ``plant.features`` description of a natural (archived) counterexample, as far as the archive determines it:
    the class, which part of the task the defect sits in (from the class and operator that found it), the literal type
    the probe moved, and the length of its program. The contract-diff size and the instruction edit are unknown for a
    natural defect and set to the typical planted value (one edit). An approximation, reported as such."""
    f = dict.fromkeys(FEATURES, 0.0)
    if cx.defect:
        f[f"d:{cx.defect}"] = 1.0
    tgt = _OP_TARGET.get(cx.op) or _CLASS_TARGET.get(cx.cls) or "required"
    f[f"t:{tgt}"] = 1.0
    f["diff_size"] = math.log1p(2)
    if cx.defect == "D7":
        f["ins_edit"] = 0.1
    lit = cx.probe.data.get("value", cx.probe.data.get("literal"))
    if isinstance(lit, bool) or lit is None:
        pass
    elif isinstance(lit, int | float):
        f["lit:number"] = 1.0
    else:
        f["lit:string"] = 1.0
    f["coll_written"] = 1.0 if tgt in ("required", "allowed") else 0.0
    if cx.probe.program is not None:
        f["witness_len"] = math.log1p(len(cx.probe.program.steps))
    return f


_MECH_TARGET = {"unstated_rule": "instruction", "undescribed_fault": "instruction", "over_hidden_feedback": "instruction", "changelog_hint": "instruction",
                "leak_other": "world", "fake_link_load": "world", "grading_bug_load": "world", "env_bug": "world", "stale_visible_test": "required",
                "grading_bug": "required", "fake_link_coarse_test": "required", "path_rule_false_positive": "trace"}


def real_defect_features(records: Iterable[Mapping[str, Any]]) -> list[dict[str, float]]:
    """The natural-likeness reference from the M5 real-defect set (``synth.realdefects``; ``runs/m5/realdefects.json``):
    only INDEPENDENTLY found validity defects (``independent``, quality P1, a D class), since falsifier-found ones are
    detected by construction (lead, 2026-10-29). Each record maps onto the plant features by its class and by where its
    mechanism sits (instruction, World, the checker's requirements or its trace rules); the size and literal features are
    unknown for a person-found defect and set to the typical planted value (one edit). An approximation, reported as such."""
    out = []
    for r in records:
        d = r.get("d_class")
        if not (r.get("independent") and r.get("quality") == "P1" and d):
            continue
        f = dict.fromkeys(FEATURES, 0.0)
        f[f"d:{d}"] = 1.0
        tgt = _MECH_TARGET.get(str(r.get("mechanism")), "required")
        f[f"t:{tgt}"] = 1.0
        f["diff_size"] = math.log1p(2)
        f["coll_written"] = 1.0 if tgt == "required" else 0.0
        if d == "D7":
            f["ins_edit"] = 0.1
        out.append(f)
    return out


def load_real_defects(path: str | Path) -> list[dict[str, Any]]:
    doc = json.loads(Path(path).read_text())
    if isinstance(doc, list):
        return doc
    return [r for v in doc.values() if isinstance(v, list) for r in v if isinstance(r, dict)]


# ---------------------------------------------------------------------------------------------------------- views and evaluation

def _strip_why(v: Any) -> Any:
    if isinstance(v, dict):
        return {k: _strip_why(x) for k, x in v.items() if k not in ("why", "id")}
    if isinstance(v, list):
        return [_strip_why(x) for x in v]
    return v


def plant_key(p: Plant) -> str:
    """A plant's content as the grader sees it (source, World, instruction, contract without why texts and pattern
    ids): two plants with one key are one planted defect, however their planters were parameterised."""
    t = p.task
    return content_hash({"src": p.source_id, "world": t.world.model_dump(mode="json"), "ins": t.instruction,
                         "c": _strip_why(t.contract.model_dump(mode="json"))}, 32)


class Pool:
    """The dev plants by the planter version that produced them (``P@<v>``), and the dev anchors.

    **The decision split** (``split``; paired mode, R1-strict): a share of the dev *source tasks* per domain is reserved
    for decisions. Their plants (of every planter, past and future) are the only units a decision view draws, and they
    are never shown to a proposer, a birth gate or the planter loop (``proposal_ids``). Units of one source stay on one
    side, so a candidate never meets a sibling plant of a unit it was built from."""

    def __init__(self, anchors: Sequence[Task]):
        self.plants: dict[str, Plant] = {}
        self.by_planter: dict[str, list[str]] = {}
        self.anchors = list(anchors)
        self.sources = list(anchors)  # every dev source task (anchors may drop one that alarms at some seed; its plants stay source-relative)
        self.dropped: dict[str, list[str]] = {}  # anchor id -> why it stopped being an anchor
        self.decide_sources: set[str] = set()

    def add(self, key: str, plants: Iterable[Plant]) -> None:
        ids = self.by_planter.setdefault(key, [])
        for p in plants:
            if p.id not in self.plants:
                self.plants[p.id] = p
                ids.append(p.id)

    def copy(self) -> Pool:
        q = Pool(self.anchors)
        q.plants = dict(self.plants)
        q.by_planter = {k: list(v) for k, v in self.by_planter.items()}
        q.decide_sources = set(self.decide_sources)
        q.sources, q.dropped = list(self.sources), dict(self.dropped)
        return q

    def domains(self) -> list[str]:
        return sorted({p.domain for p in self.plants.values()})

    def split(self, frac: float, *, seed: int = 0) -> set[str]:
        """Reserve ``frac`` of each domain's source tasks (at least one when the domain has two) for decisions."""
        by: dict[str, set[str]] = {}
        for p in self.plants.values():
            by.setdefault(p.domain, set()).add(p.source_id)
        out: set[str] = set()
        for _d, srcs in sorted(by.items()):
            ss = sorted(srcs, key=lambda s: content_hash({"src": s, "seed": seed, "split": "decide"}, 16))
            k = round(frac * len(ss))
            if len(ss) >= 2:
                k = min(max(k, 1), len(ss) - 1)
            else:
                k = 0
            out |= set(ss[:k])
        self.decide_sources = out
        return out

    def is_decide(self, pid: str) -> bool:
        return self.plants[pid].source_id in self.decide_sources

    def decide_ids(self) -> list[str]:
        return [i for i in self.plants if self.is_decide(i)]

    def proposal_ids(self) -> list[str]:
        return [i for i in self.plants if not self.is_decide(i)]

    def proposal_plants(self) -> list[Plant]:
        return [self.plants[i] for i in self.proposal_ids()]


class Sampler:
    """R1's fresh views (module docstring): seeded and stratified by planter version and domain.

    ``strict`` (paired mode): a ``decide`` or ``prune`` view is up to ``n_max`` units drawn without replacement from the
    decision split's units no earlier decision consumed (``consume``/``release`` keep the bank; a reserved unit is
    never handed out twice); every other purpose (calibration, A/A, audits) draws from the proposal split only.
    ``fixed`` (the E10 ablation of R1): every view is the first one, reused."""

    def __init__(self, pool: Pool, *, frac: float = 0.6, seed: int = 0, strict: bool = False, n_max: int = 60, fixed: bool = False):
        self.pool, self.frac, self.seed, self.n = pool, frac, seed, 0
        self.strict, self.n_max, self.fixed = strict, n_max, fixed
        self.consumed: set[str] = set()
        self.reserved: set[str] = set()
        self._first: View | None = None

    def draw(self, tag: str, plants: Sequence[str] | None = None, frac: float | None = None) -> list[str]:
        rng = random.Random(f"view/{self.seed}/{tag}")
        out: list[str] = []
        keep = None if plants is None else set(plants)
        for _k, ids in sorted(self.pool.by_planter.items()):
            # strata: domain only. RSI-1 stratified by (domain, defect) too: with deterministic falsifiers every such view
            # had the same J, so the measured noise floor was δ = 0 and R2 degenerated to "any positive LCB"
            by: dict[str, list[str]] = {}
            for i in ids if keep is None else [x for x in ids if x in keep]:
                by.setdefault(self.pool.plants[i].domain, []).append(i)
            for _d, xs in sorted(by.items()):
                k = max(1, round((frac or self.frac) * len(xs)))
                out += rng.sample(sorted(xs), min(k, len(xs)))
        return out

    def remaining(self) -> list[str]:
        return sorted(i for i in self.pool.decide_ids() if i not in self.consumed and i not in self.reserved)

    async def __call__(self, round: int, purpose: str, used: frozenset[str]) -> View:
        if self.fixed and self._first is not None:
            return self._first
        anchors = [a.id for a in self.pool.anchors]
        if self.strict and purpose in ("decide", "prune"):
            self.n += 1
            rem = self.remaining()
            ids = random.Random(f"decide/{self.seed}/{round}/{purpose}/{self.n}").sample(rem, min(self.n_max, len(rem)))
            self.reserved |= set(ids)
            h = content_hash({"plants": sorted(ids), "anchors": sorted(anchors), "n": self.n, "strict": True}, 32)
            v = View(hash=h, round=round, domains=self.pool.domains(), planters=sorted(self.pool.by_planter), seed=self.n, strict=True,
                     payload={"plants": ids, "anchors": anchors})
        elif purpose == "aa":  # the A/A view: the decision size (n_max units) when the proposal split has them
            self.n += 1
            prop = sorted(self.pool.proposal_ids() if self.strict else self.pool.plants)
            ids = random.Random(f"aa/{self.seed}/{round}/{self.n}").sample(prop, min(self.n_max, len(prop)))
            h = content_hash({"plants": sorted(ids), "anchors": sorted(anchors), "aa": True}, 32)
            v = View(hash=h, round=round, domains=self.pool.domains(), planters=sorted(self.pool.by_planter), seed=self.n,
                     payload={"plants": ids, "anchors": anchors})
        else:
            restrict = self.pool.proposal_ids() if self.strict else None
            for _ in range(64):
                self.n += 1
                ids = self.draw(f"{round}/{purpose}/{self.n}", plants=restrict)
                h = content_hash({"plants": sorted(ids), "anchors": sorted(anchors)}, 32)
                if h not in used:
                    break
            v = View(hash=h, round=round, domains=self.pool.domains(), planters=sorted(self.pool.by_planter), seed=self.n,
                     payload={"plants": ids, "anchors": anchors})
        if self.fixed:
            self._first = v
        return v

    def consume(self, view: View, units: Sequence[str]) -> None:
        self.consumed |= set(units)
        self.reserved -= set(units)

    def release(self, view: View, units: Sequence[str]) -> None:
        self.reserved -= set(units)

    def state(self) -> dict[str, Any]:
        return {"n": self.n, "consumed": sorted(self.consumed)}

    def load(self, d: Mapping[str, Any]) -> None:
        self.n = int(d.get("n", 0))
        self.consumed = set(d.get("consumed", []))
        self.reserved = set()


def population(theta_f: Any, base: Callable[[], list[BaseFalsifier]] = cheap) -> list[BaseFalsifier]:
    lib = Library.model_validate(theta_f or {})
    return [*base(), *lib.falsifiers()]


async def verdicts(plants: Sequence[Plant], falsifiers: Sequence[BaseFalsifier], ctx: FalsifyContext) -> tuple[dict[str, PlantVerdict], int]:
    out, probes = {}, 0
    for p in plants:
        res = await run_classes(p.task, falsifiers, ctx, cache=True)
        probes += sum(len(r.findings) for r in res)
        out[p.id] = PlantVerdict(plant_id=p.id, hit=[r.cls for r in res if r.applicable and r.c],
                                 suspect_only=[r.cls for r in res if r.applicable and not r.c and r.suspects])
    return out, probes


class FEvaluator:
    """``update.Evaluator`` for F updates (module docstring). ``budget`` (R9's control): the incumbent re-run with MUT's
    neighbourhood enlarged until its probe spend reaches the candidate's. ``view.eval_seed`` selects the falsifiers'
    seed (MUT's sibling sample, NULL's random calls, TWIN, PERT): paired mode's A/A replicates and decisions. ``shards``
    (a ``par.ShardPool`` holding the pool's plants and anchors) evaluates in worker processes; without it, evaluations
    run one at a time in this process (a lock: the Worlds are shared)."""

    def __init__(self, pool: Pool, ctx: FalsifyContext, *, base: Callable[[], list[BaseFalsifier]] = cheap, shards: Any = None,
                 seeded: dict[tuple[int, int], FalsifyContext] | None = None):
        self.pool, self.ctx, self.base, self.shards = pool, ctx, base, shards
        self._big: dict[tuple[int, int], FalsifyContext] = {} if seeded is None else seeded  # shared: results are a function of (seed, task, class)
        self._lock = asyncio.Lock()

    def ctx_for(self, max_mutants: int, seed: int) -> FalsifyContext:
        if max_mutants == self.ctx.max_mutants and seed == self.ctx.seed:
            return self.ctx
        k = (max_mutants, seed)
        if k not in self._big:
            c = FalsifyContext(max_mutants=max_mutants, seed=seed)
            c._worlds = self.ctx._worlds  # share the Worlds (one per WorldRef), not the class cache
            self._big[k] = c
        return self._big[k]

    async def _units(self, theta_f: Any, plants: Sequence[Plant], anchors: Sequence[Task], max_mutants: int, seed: int) -> tuple[dict[str, PlantVerdict], dict[str, bool], int]:
        """Verdicts on plants (source-relative) and alarms on anchors. **Source-relative hits** (RSI-2 pilot): a class
        that confirms a counterexample on a plant's clean source task at the same seed is not credited on that plant (the
        A/A replicates found TWIN alarming on one it_helpdesk source at two of eleven seeds; without this rule its plants
        were "caught" by the source's own false alarm). Sources are evaluated with the plants (cached)."""
        by_id = {a.id: a for a in [*self.pool.sources, *self.pool.anchors]}
        srcs = [by_id[s] for s in dict.fromkeys(p.source_id for p in plants) if s in by_id]
        tasks = list({t.id: t for t in [*anchors, *srcs]}.values())
        if self.shards is not None:
            await self.shards.add(plants=[p for p in plants if p.id not in self.shards.known], tasks=[a for a in tasks if a.id not in self.shards.known])
            res = await self.shards.evaluate(theta_f, [*(p.id for p in plants), *(a.id for a in tasks)], seed=seed, max_mutants=max_mutants)
            hits = {k: list(v["hit"]) for k, v in res.items()}
            susp = {k: list(v["suspect"]) for k, v in res.items()}
            probes = sum(r["probes"] for r in res.values())
        else:
            async with self._lock:
                ctx = self.ctx_for(max_mutants, seed)
                fs = population(theta_f, self.base)
                hits, susp, probes = {}, {}, 0
                for u_id, t in [*((p.id, p.task) for p in plants), *((a.id, a) for a in tasks)]:
                    rs = await run_classes(t, fs, ctx, cache=True)
                    probes += sum(len(r.findings) for r in rs)
                    hits[u_id] = [r.cls for r in rs if r.applicable and r.c]
                    susp[u_id] = [r.cls for r in rs if r.applicable and not r.c and r.suspects]
        vs = {}
        for p in plants:
            own = set(hits.get(p.source_id, [])) if p.source_id in by_id else set()
            vs[p.id] = PlantVerdict(plant_id=p.id, hit=[c for c in hits[p.id] if c not in own], suspect_only=susp[p.id])
        alarms = {a.id: bool(hits[a.id]) for a in anchors}
        return vs, alarms, probes

    async def measure(self, theta_f: Any, plant_ids: Sequence[str], anchor_ids: Sequence[str], *, max_mutants: int | None = None, seed: int | None = None) -> Measurement:
        plants = [self.pool.plants[i] for i in plant_ids]
        anchors = [a for a in self.pool.anchors if a.id in set(anchor_ids)]
        mm = self.ctx.max_mutants if max_mutants is None else max_mutants
        vs, alarms, probes = await self._units(theta_f, plants, anchors, mm, self.ctx.seed if seed is None else seed)
        units = [1.0 if vs[p.id].hit else 0.0 for p in plants]
        by_d: dict[str, list[float]] = {}
        for p, u in zip(plants, units, strict=True):
            by_d.setdefault(p.defect, []).append(u)
        ops = list(Library.model_validate(theta_f or {}).ops)  # keys are the compiled class names (OP:<name>@<v>)
        gain = {}
        for op in ops:
            only = sum(1 for p in plants if vs[p.id].hit == [op])
            gain[op] = only / len(plants) if plants else 0.0
        n_units = len(plants) + len(anchors)
        return Measurement(j_units=units, unit_domains=[p.domain for p in plants], fr={d: sum(x) / len(x) for d, x in by_d.items()},
                           fa=sum(alarms.values()) / len(anchors) if anchors else 0.0, cost=PROBE_USD * probes / max(1, n_units), item_gain=gain,
                           unit_ids=[p.id for p in plants], unit_classes=[p.defect for p in plants], unit_clusters=[p.source_id for p in plants],
                           n_anchors=len(anchors))

    async def __call__(self, theta: Any, view: View, budget: float | None) -> Measurement:
        f = theta.data.get("F")
        seed = view.eval_seed
        m = await self.measure(f, view.payload["plants"], view.payload["anchors"], seed=seed)
        if budget is None or budget <= m.cost:
            return m
        mm = min(4000, math.ceil(self.ctx.max_mutants * budget / max(m.cost, 1e-12)))
        return await self.measure(f, view.payload["plants"], view.payload["anchors"], max_mutants=mm, seed=seed)


# ---------------------------------------------------------------------------------------------------------- proposers

ProposeOps = Callable[..., Awaitable[tuple[list[tuple[OperatorSpec, list[str]]], dict[str, Any]]]]
ProposePlanters = Callable[..., Awaitable[tuple[list[PlantOp], dict[str, Any]]]]
ProposeGuidance = Callable[..., Awaitable[tuple[list[str], dict[str, Any]]]]


@dataclass
class Proposers:
    """What the loop asks models for. ``ops(clusters=, misses=, guidance=, library=, k=, round=)`` -> ([(spec, cites)],
    info); ``planters(miss_rates=, natural_mix=, k=, round=)`` -> ([PlantOp], info); ``guidance(guidance=,
    gate_outcomes=)`` -> (new guidance list, info). ``family`` is the author family (R6)."""

    ops: ProposeOps | None = None
    planters: ProposePlanters | None = None
    guidance: ProposeGuidance | None = None
    family: str = ""


def swarm_proposers(client: Any, *, planter_client: Any = None, salt: str = "") -> Proposers:
    """The loop's proposers on one ``roles.RoleClient`` (the abstractor's family; R6: never a builder's). ``salt`` enters every call's tag:
    with a client that seeds by tag (``WireClient(seed_by_tag=True)``), E10's seeds and every round get independent samples."""
    sfx = f":{salt}" if salt else ""
    calls = [0]
    from crucible.synth.meta.abstract import DSL_DOC, GUIDANCE_EDITOR, OPERATOR_ABSTRACTOR, PLANTER_PROPOSER, parse_operators
    from crucible.synth.roles import DEFECTS_DOC, ask

    async def ops(*, clusters: list[dict[str, Any]], misses: list[dict[str, Any]], guidance: list[str], library: list[dict[str, Any]], k: int, round: int,
                  tag: str = "A") -> tuple[list[tuple[OperatorSpec, list[str]]], dict[str, Any]]:
        errs: list[str] = []

        def check(o: Any) -> None:
            got, bad = parse_operators(o, prefix=f"r{round}{tag.replace(chr(39), 'b')}", limit=k)
            if not got:
                raise ValueError("no valid operator: " + "; ".join(bad[:4]))
            errs.extend(bad)

        out = await ask(client, OPERATOR_ABSTRACTOR, tag=f"abstract:r{round}:{tag}{sfx}", check=check, retries=1, clusters=clusters, misses=misses,
                        defects=DEFECTS_DOC, dsl=DSL_DOC, guidance="\n".join(f"{i}. {g}" for i, g in enumerate(guidance)),
                        library=[{"name": d.get("name"), "body": d.get("body", {}).get("kind"), "defect": d.get("label", {}).get("defect")} for d in library] or None,
                        count=k)
        got, bad = parse_operators(out, prefix=f"r{round}{tag.replace(chr(39), 'b')}", limit=k)
        return got, {"lemmas": [x.model_dump(mode="json") for x in out.lemmas], "strategies": list(out.strategies), "errors": bad or errs,
                     "notes": out.notes[:500]}

    async def planters(*, miss_rates: dict[str, float], natural_mix: dict[str, int], k: int, round: int,
                       history: list[dict[str, Any]] | None = None) -> tuple[list[PlantOp], dict[str, Any]]:
        from crucible.synth.meta.abstract import dev_variants

        dv = dev_variants()

        def check(o: Any) -> None:
            for i, d in enumerate(o.planters):
                if d.get("name") not in dv or d.get("variant", "base") not in dv[d["name"]]:
                    raise ValueError(f"planter {i}: unknown operator or variant {d.get('name')}:{d.get('variant')} (have {dv})")

        out = await ask(planter_client or client, PLANTER_PROPOSER, tag=f"plant:r{round}{sfx}", check=check, retries=1,
                        operators=sorted(miss_rates), miss_rates=miss_rates, natural_mix=natural_mix, count=k, history=history or None)
        allowed = {"why", "factor", "phrase", "template", "suffix", "field"}
        res = [PlantOp(name=d["name"], variant=d.get("variant", "base"), version=100 + round,
                       params={kk: v for kk, v in (d.get("params") or {}).items() if kk in allowed and isinstance(v, str | int | float)}) for d in out.planters[:k]]
        return res, {"notes": out.notes[:500]}

    async def guidance(*, guidance: list[str], gate_outcomes: dict[str, int]) -> tuple[list[str], dict[str, Any]]:
        calls[0] += 1
        out = await ask(client, GUIDANCE_EDITOR, tag=f"abstract:guidance:{calls[0]}{sfx}", retries=1, guidance="\n".join(f"{i}. {g}" for i, g in enumerate(guidance)),
                        gate_outcomes=gate_outcomes, dsl=DSL_DOC)
        keep = [g for i, g in enumerate(guidance) if i not in set(out.remove)]
        return [*keep, *[a.strip()[:300] for a in out.add[:2] if a.strip()]], {"notes": out.notes[:300], "removed": out.remove}

    return Proposers(ops=ops, planters=planters, guidance=guidance, family=getattr(client, "family", ""))


def random_proposer(seed: int) -> ProposeOps:
    async def ops(*, k: int, round: int, **_: Any) -> tuple[list[tuple[OperatorSpec, list[str]]], dict[str, Any]]:
        rng = random.Random(f"control/{seed}/{round}")
        return [(random_operator(rng, f"rnd{round}_{i}"), []) for i in range(k)], {"source": "random DSL samples"}

    return ops


# ---------------------------------------------------------------------------------------------------------- the loop

DEFAULT_GUIDANCE = [("An operator must make the checker ACCEPT a wrong run (expect fail) or REJECT a valid run (expect pass); probes "
                    "that change nothing the checker reads find nothing."),
                    "Keep operators generic: globs like * or *.get_* and selectors, never names or values of one application."]

# E10: one regulariser removed at a time (part2-paper-core §9.6). Each maps onto the rules the update engine switches
# off (``UpdateConfig.disabled``) and, for R1 and the planter's natural-likeness, onto the loop itself.
ABLATIONS: dict[str, dict[str, Any]] = {
    "none": {},
    "R1": {"disabled": ["R1"], "fixed_view": True},  # every decision on one reused view
    "R3": {"disabled": ["R3"]},  # no per-class FR / P3 / P4 non-regression (archive and FA stay: they are hard validity rules)
    "R4": {"disabled": ["R4"]},  # no trust region on the task distribution
    "R5": {"disabled": ["R5"]},  # no edit budget, no cost bar, no pruning
    "R7": {"disabled": ["R7"]},  # no transfer requirement
    "R8": {"disabled": ["R8"]},  # no judge-only round cap
    "NL": {"natural_likeness": False},  # the planter's MMD bar off
}


class StopRules(CrucibleModel):
    """When a continuous run stops (the preregistration fixes these before the run; ``Loop.stop_reason``)."""

    max_rounds: int | None = None  # None: EvolveConfig.rounds
    max_spend_usd: float | None = None
    futility_rounds: int | None = None  # this many consecutive rounds with no accepted F update and no planter accepted
    min_headroom: float = 0.0  # ... and dev headroom at or below this (when measured)
    max_rollbacks: int | None = None
    max_pool_kl: float | None = None  # Goodhart alarm: KL(class mix of the dev pool now ‖ at round 0)


class EvolveConfig(CrucibleModel):
    rounds: int = 4
    candidates: int = 4
    view_frac: float = 0.6
    gate_fresh: int = 24  # other-domain plants a birth gate searches
    gate_sources: int = 4
    cost_bar: float = 48.0  # mean probes per task (birth gate)
    seed: int = 0
    evidence_domains: list[str] | None = None  # None: every dev domain
    planter_loop: bool = True
    meta_loop: bool = True
    planter_candidates: int = 3
    mu_margin: float = 1.1  # μ = margin × MMD(base dev plants, natural defects): a candidate planter may not be less natural
    audit: bool = True
    update: UpdateConfig = Field(default_factory=lambda: UpdateConfig(rounds=4, min_band=0.02))
    # -- RSI-2 statistics (runs/rsi-stats.md); all off by default (RSI-1's behaviour) ------------------------------------
    decide_frac: float = 0.5  # paired mode: the share of each domain's dev sources reserved for decisions
    planter_reward: Literal["miss", "headroom"] = "miss"  # headroom: a planter must plant misses some expressible operator catches
    headroom: bool = False  # measure dev headroom every round (the P/F game's trajectory)
    natural_likeness: bool = True  # E10 "NL" switches the planter's MMD bar off
    fixed_view: bool = False  # E10 "R1": decisions reuse one view
    ablation: str = "none"
    locked_gate: bool = False  # a decision reads the locked set through the reusable holdout (generalisation gate "G")
    holdout_threshold: float = 0.04
    holdout_sigma: float = 0.01
    holdout_budget: int = 10
    locked_tolerance: float = 0.0  # G: the holdout's answer for the locked gain must be ≥ −tolerance
    meta_min: int = 4  # the meta loop tests A' against A only with at least this many candidates per side
    stop: StopRules = Field(default_factory=StopRules)

    def ablated(self, name: str) -> EvolveConfig:
        """This configuration with one E10 ablation applied (``ABLATIONS``)."""
        a = ABLATIONS[name]
        upd = self.update.model_copy(update={"disabled": sorted({*self.update.disabled, *a.get("disabled", [])})})
        return self.model_copy(update={"update": upd, "fixed_view": a.get("fixed_view", self.fixed_view),
                                       "natural_likeness": a.get("natural_likeness", self.natural_likeness), "ablation": name})


class Candidate(CrucibleModel):
    op: str
    origin: str  # swarm | random | swarm:A' | swarm:A
    cites: list[str] = Field(default_factory=list)
    spec: dict[str, Any] = Field(default_factory=dict)  # the operator itself (refused ones too: negative evidence, and E3's post-hoc transfer)
    gate: dict[str, Any] = Field(default_factory=dict)
    decision: dict[str, Any] | None = None
    refused_by: list[str] = Field(default_factory=list)  # "gate:<check>" or "R<k>"
    accepted: bool = False


class RoundRow(CrucibleModel):
    arm: str
    round: int
    library: list[str] = Field(default_factory=list)
    planters: int = 1
    candidates: list[Candidate] = Field(default_factory=list)
    planter_decisions: list[dict[str, Any]] = Field(default_factory=list)
    guidance_decision: dict[str, Any] | None = None
    pruned: list[str] = Field(default_factory=list)
    rollback: dict[str, Any] | None = None
    dev: dict[str, Any] = Field(default_factory=dict)  # the head's recall on the whole dev pool (J proxy, report only)
    locked: dict[str, Any] = Field(default_factory=dict)  # summary() of the locked-set matrix (report only)
    family: dict[str, Any] = Field(default_factory=dict)  # summary() on the held-out World family (report only)
    game: dict[str, Any] = Field(default_factory=dict)  # the P/F game: recall, misses, headroom, planter rewards, drift
    lemmas: list[dict[str, Any]] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)
    stop: str | None = None


@dataclass
class Arm:
    name: str
    store: ThetaStore
    engine: UpdateEngine
    pool: Pool
    sampler: Sampler
    evaluator: FEvaluator
    propose_ops: ProposeOps
    family: str = ""
    rows: list[RoundRow] = field(default_factory=list)
    guidance: list[str] = field(default_factory=lambda: list(DEFAULT_GUIDANCE))
    dir: Path | None = None
    ctx: FalsifyContext | None = None
    planter_history: list[dict[str, Any]] = field(default_factory=list)
    stopped: str | None = None
    holdout: Any = None  # the arm's Thresholdout over the locked set (generalisation gate), when enabled


def _lits(p: Program) -> list[Any]:
    out: list[Any] = []

    def walk(a: Any) -> None:
        if isinstance(a, Lit) and not isinstance(a.v, dict | list):
            out.append(a.v)
        elif isinstance(a, Fn):
            for x in a.args:
                walk(x)

    for s in p.steps:
        for a in s.args.values():
            walk(a)
    return out


def evidence_of(plants: Sequence[Plant], builders: Sequence[str], domains: Sequence[str]) -> Evidence:
    lits: list[Any] = []
    for p in plants:
        lits += _lits(p.task.oracle)
        if p.witness.program is not None:
            lits += _lits(p.witness.program)
        if p.change.get("literal") is not None:
            lits.append(p.change["literal"])
    keys = sorted({str(x) for x in lits if isinstance(x, str | int) and not isinstance(x, bool)})
    return Evidence(domains=sorted(set(domains)), task_ids=sorted({p.source_id for p in plants}), plant_ids=sorted({p.id for p in plants}),
                    entity_keys=keys, literals=list({cjson(x): x for x in lits}.values()), builder_families=list(builders), kind="executed",
                    new_executed=len(plants))


def _cluster_plants(cites: Sequence[str], clusters: Sequence[Cluster], pool: Pool) -> list[Plant]:
    ids = [m for c in clusters if c.id in set(cites) and c.kind == "miss" for m in c.members]
    return [pool.plants[i] for i in ids if i in pool.plants]


def class_mix(plants: Iterable[Plant]) -> dict[str, float]:
    return dict(Counter(p.defect for p in plants))


def plant_size(plants: Sequence[Plant]) -> dict[str, float]:
    """Goodhart's "task size" for planted tasks: the mean contract-diff size and witness length (log1p features)."""
    if not plants:
        return {"diff_size": 0.0, "witness_len": 0.0}
    return {k: round(sum(p.features.get(k, 0.0) for p in plants) / len(plants), 4) for k in ("diff_size", "witness_len")}


class Loop:
    """E2/E3's driver: one or more arms over a shared locked set (module docstring).

    **Continuous and resumable.** Every arm keeps its versioned Θ (``theta/``), its hash-chained ledger (with write-ahead
    ``test`` records of every α spend and decision-unit reservation), its rows, and a ``state.json`` written after every
    round (the engine's floor, spender, used views and units, pruning bookkeeping, the sampler's consumed units, the
    planter history and the plants accepted planters added, in ``plants.jsonl``). ``run(resume=True)`` continues an
    arm from its last completed round; a crashed round is redone, and what its tests spent stays spent.

    **Stopping** (``StopRules``, fixed by the preregistration): the round cap, the spend cap, the α budget, the decision
    split running dry, futility (rounds without an accepted update or planter while the dev headroom is gone), too many
    rollbacks, a Goodhart alarm on the dev pool's class mix.

    **Student signals** enter only through the existing channels: ``natural_refresh`` re-reads the archive every round,
    so online-monitor counterexamples the panel confirmed (class ONL) become natural clusters for the abstractor (F); the
    S and K channels are in ``meta/signals.py``. Student reward never enters J.
    """

    def __init__(self, root: Path, pool: Pool, locked: PlantSet, locked_anchors: Sequence[Task], natural: Sequence[Counterexample], *,
                 ctx: FalsifyContext, report_ctx: FalsifyContext, config: EvolveConfig, dev_sources: Sequence[Task] = (), builders: Sequence[str] = (),
                 natural_sources: Mapping[str, Task] | None = None, spend: Callable[[], float] | None = None,
                 reference: Sequence[dict[str, float]] | None = None, family: PlantSet | None = None, family_anchors: Sequence[Task] = (),
                 natural_refresh: Callable[[], Sequence[Counterexample]] | None = None, shards: Callable[[], Any] | None = None,
                 make_ctx: Callable[[], FalsifyContext] | None = None, seeded: dict[tuple[int, int], FalsifyContext] | None = None):
        self.root, self.base_pool, self.locked, self.locked_anchors = Path(root), pool, locked, list(locked_anchors)
        self.dev_sources = list(dev_sources)
        self.natural = list(natural)
        self.ctx, self.report_ctx, self.cfg = ctx, report_ctx, config
        self.builders = list(builders)
        self.natural_sources = dict(natural_sources or {})
        self.spend = spend or (lambda: 0.0)
        self.arms: list[Arm] = []
        self.family, self.family_anchors = family, list(family_anchors)
        self.natural_refresh = natural_refresh
        self.make_shards = shards
        self.make_ctx = make_ctx
        self.seeded = seeded  # seeded contexts shared by every arm of this loop (only without make_ctx: they share the Worlds)
        self.headroom_cache: Any = None
        self._report_lock = asyncio.Lock()
        self._round_misses: dict[str, list[Plant]] = {}  # arm -> the proposal-split plants its incumbent misses this round
        self._round_tested: dict[str, set[str]] = {}  # arm -> proposal-split catches of the candidates sent to a test this round
        # the natural-likeness reference: given (the independently found real defects), else the archive's natural
        # counterexamples (falsifier-found: a weaker reference, used only when no independent set exists)
        ref = list(reference) if reference else [natural_features(c) for c in self.natural]
        self.reference_kind = "independent real defects" if reference else "archive (falsifier-found)"
        self.reference = ref
        base = [p.features for p in pool.plants.values()]
        self.mu = (self.cfg.mu_margin * mmd(feature_matrix(base), feature_matrix(ref))) if base and ref else 0.0
        if not self.cfg.natural_likeness:
            self.mu = math.inf
        self.mix0 = class_mix(pool.plants.values())

    @property
    def planter_history(self) -> list[dict[str, Any]]:  # RSI-1's attribute: the loop arm's history
        return self.arms[0].planter_history if self.arms else []

    def arm(self, name: str, propose_ops: ProposeOps, *, family: str = "") -> Arm:
        d = self.root / name
        store = ThetaStore(d / "theta")
        if store.head("F") is None:
            store.init("F", Library().model_dump(mode="json"))
            store.init("P", {"planters": [base_planter().model_dump(mode="json")]})
            store.init("S", {"version": 1, "lemmas": [], "retired": []})
            store.init("A", {"guidance": list(DEFAULT_GUIDANCE)})
        pool = self.base_pool.copy()
        pool.by_planter = {"P@1": [i for ids in pool.by_planter.values() for i in ids]}
        st = self.cfg.update.stats
        if st.paired and not pool.decide_sources:
            pool.split(self.cfg.decide_frac, seed=self.cfg.seed)
        sampler = Sampler(pool, frac=self.cfg.view_frac, seed=int(content_hash({"arm": name, "seed": self.cfg.seed}, 8), 16) % 10_000,
                          strict=st.paired, n_max=st.n_max, fixed=self.cfg.fixed_view)
        ctx = self.make_ctx() if self.make_ctx is not None else self.ctx
        ev = FEvaluator(pool, ctx, shards=self.make_shards() if self.make_shards is not None else None, seeded=self.seeded if self.make_ctx is None else None)
        eng = UpdateEngine(store, ev, sampler, config=self.cfg.update, ledger=Ledger(d / "ledger.jsonl"))
        a = Arm(name=name, store=store, engine=eng, pool=pool, sampler=sampler, evaluator=ev, propose_ops=propose_ops, family=family, dir=d, ctx=ctx)
        if self.cfg.locked_gate:
            eng.holdout_gate = self._holdout_gate(a)
        self.arms.append(a)
        return a

    # -- state ------------------------------------------------------------------------------------------------------
    def save_state(self, arm: Arm) -> None:
        assert arm.dir is not None
        arm.dir.mkdir(parents=True, exist_ok=True)
        doc = {"round": arm.rows[-1].round if arm.rows else -1, "engine": arm.engine.state(), "sampler": arm.sampler.state(),
               "planter_history": arm.planter_history, "by_planter": arm.pool.by_planter, "decide_sources": sorted(arm.pool.decide_sources), "dropped": arm.pool.dropped,
               "rows": [r.model_dump(mode="json") for r in arm.rows], "stopped": arm.stopped}
        tmp = arm.dir / "state.json.tmp"
        tmp.write_text(json.dumps(doc, default=str))
        tmp.replace(arm.dir / "state.json")

    def load_state(self, arm: Arm) -> int:
        """Restore an arm from its ``state.json``; returns the last completed round (−1: a fresh arm)."""
        assert arm.dir is not None
        p = arm.dir / "state.json"
        if not p.exists():
            return -1
        doc = json.loads(p.read_text())
        extra = arm.dir / "plants.jsonl"
        if extra.exists():
            for line in extra.read_text().splitlines():
                if line.strip():
                    pl = Plant.model_validate_json(line)
                    arm.pool.plants.setdefault(pl.id, pl)
        arm.pool.by_planter = {k: list(v) for k, v in doc["by_planter"].items()}
        arm.pool.decide_sources = set(doc.get("decide_sources", []))
        arm.pool.dropped = dict(doc.get("dropped", {}))
        arm.pool.anchors = [t for t in arm.pool.anchors if t.id not in arm.pool.dropped]
        arm.engine.load_state(doc["engine"])
        arm.sampler.load(doc["sampler"])
        arm.sampler.consumed |= arm.engine.used_units
        arm.planter_history = list(doc.get("planter_history", []))
        arm.rows = [RoundRow.model_validate(r) for r in doc.get("rows", [])]
        arm.guidance = list(arm.store.get("A")["guidance"])
        arm.stopped = None  # re-judged by stop_reason under the (possibly extended) configuration of the resumed run
        return int(doc.get("round", -1))

    # -- per round ---------------------------------------------------------------------------------------------------
    def evidence_plants(self, arm: Arm) -> list[Plant]:
        """What proposers and gates may see: the proposal split (paired mode keeps the decision split out of it)."""
        doms = self.cfg.evidence_domains
        src = arm.pool.proposal_plants() if self.cfg.update.stats.paired else list(arm.pool.plants.values())
        return [p for p in src if doms is None or p.domain in doms]

    async def head_verdicts(self, arm: Arm, plants: Sequence[Plant]) -> dict[str, PlantVerdict]:
        if arm.evaluator.shards is not None:  # the worker processes (their caches hold every unit already measured)
            vs, _a, _p = await arm.evaluator._units(arm.store.get("F"), plants, [], arm.evaluator.ctx.max_mutants, arm.evaluator.ctx.seed)
            return vs
        vs, _ = await verdicts(plants, population(arm.store.get("F")), arm.ctx or self.ctx)
        return vs

    def _natural(self) -> list[Counterexample]:
        doms = self.cfg.evidence_domains
        return [c for c in self.natural if doms is None or c.domain in doms]

    def refresh_natural(self) -> int:
        """Channel (b): re-read the archive (natural counterexamples, the panel-confirmed ONL ones from student rollouts
        among them). Returns how many are new."""
        if self.natural_refresh is None:
            return 0
        have = {c.id for c in self.natural}
        new = [c for c in self.natural_refresh() if c.id not in have]
        self.natural += new
        return len(new)

    def _sources(self, cites: Sequence[str], clusters: Sequence[Cluster], arm: Arm, rng: random.Random, misses: Sequence[Plant]) -> tuple[list[Plant | Task], str]:
        """A candidate's birth-gate sources: plants of the miss clusters it cites (else natural-defect tasks of the natural
        clusters it cites, else random misses), all from ONE evidence domain, so its transfer is tested elsewhere."""
        plants = _cluster_plants(cites, clusters, arm.pool)
        if not plants:
            nat_ids = {m for c in clusters if c.id in set(cites) and c.kind == "natural" for m in c.members}
            tasks = []
            for c in self.natural:
                t = self.natural_sources.get(c.task_id)
                if c.id in nat_ids and t is not None and t not in tasks:
                    tasks.append(t)
            if tasks:
                dom = Counter(t.domain for t in tasks).most_common(1)[0][0]
                return [t for t in tasks if t.domain == dom][: self.cfg.gate_sources], dom
            plants = rng.sample(list(misses), min(len(misses), self.cfg.gate_sources)) if misses else []
        if not plants:
            return [], ""
        dom = Counter(p.domain for p in plants).most_common(1)[0][0]
        same = [p for p in plants if p.domain == dom]
        return rng.sample(same, min(len(same), self.cfg.gate_sources)), dom

    async def _gate_par(self, arm: Arm, spec: OperatorSpec, sources: Sequence[Plant | Task], fresh: Sequence[Plant]) -> Any:
        """``library.birth_gate`` on the shards (the same four checks; the operator alone, every unit in parallel), so a
        round's candidates are gated concurrently."""
        from crucible.synth.meta.library import GateReport

        sh = arm.evaluator.shards
        lib = Library().add(spec, 0).model_dump(mode="json")
        src_plants = [s for s in sources if isinstance(s, Plant)]
        src_tasks = [s for s in sources if not isinstance(s, Plant)]
        await sh.add(plants=[p for p in [*src_plants, *fresh] if p.id not in sh.known], tasks=[t for t in [*arm.pool.anchors, *src_tasks] if t.id not in sh.known])
        ids = [*(a.id for a in arm.pool.anchors), *(s.id for s in sources), *(p.id for p in fresh)]
        res = await sh.evaluate(lib, ids, seed=arm.evaluator.ctx.seed, max_mutants=arm.evaluator.ctx.max_mutants, base=False)
        alarms = [a.id for a in arm.pool.anchors if res[a.id]["hit"]]
        repro = [bool(res[s.id]["hit"]) for s in sources]
        src_ids = {s.source_id if isinstance(s, Plant) else s.id for s in sources}
        src_doms = {s.domain for s in sources}
        others = [p for p in fresh if p.source_id not in src_ids and p.domain not in src_doms]
        hits = [p.id for p in others if res[p.id]["hit"]]
        n = len(set(ids))
        ppt = sum(r["probes"] for r in res.values()) / n if n else 0.0
        checks = {"reproduces_sources": bool(repro) and all(repro), "finds_on_other_domains": len(hits) >= 1, "zero_false_alarms": not alarms,
                  "under_cost_bar": ppt <= self.cfg.cost_bar}
        return GateReport(op=spec.cls_name, passed=all(checks.values()), checks=checks,
                          detail={"false_alarms": alarms, "reproduced": repro, "transfer_hits": hits, "transfer_candidates": len(others), "probes_per_task": ppt})

    def _prepare(self, arm: Arm, spec: OperatorSpec, cites: list[str], origin: str, clusters: Sequence[Cluster], misses: Sequence[Plant], r: int,
                 rng: random.Random) -> tuple[Candidate, list[Plant | Task], str, list[Plant]] | Candidate:
        """A candidate's gate inputs, drawn in proposal order from the round's rng (so a concurrent gate phase stays
        reproducible)."""
        cand = Candidate(op=spec.cls_name, origin=origin, cites=list(cites), spec=spec.model_dump(mode="json"))
        lib = Library.model_validate(arm.store.get("F"))
        if spec.cls_name in lib.ops:
            cand.refused_by = ["duplicate"]
            return cand
        sources, dom = self._sources(cites, clusters, arm, rng, misses)
        visible = set(arm.pool.proposal_ids()) if self.cfg.update.stats.paired else None
        others = [i for i in arm.sampler.draw(f"gate/{r}/{spec.cls_name}", plants=None if visible is None else sorted(visible), frac=1.0)
                  if arm.pool.plants[i].domain != dom]
        fresh = [arm.pool.plants[i] for i in rng.sample(others, min(len(others), self.cfg.gate_fresh))]
        return cand, sources, dom, fresh

    async def _gate(self, arm: Arm, spec: OperatorSpec, prep: tuple[Candidate, list[Plant | Task], str, list[Plant]]) -> Any:
        cand, sources, dom, fresh = prep
        try:
            if arm.evaluator.shards is not None:
                g = await self._gate_par(arm, spec, sources, fresh)
            else:
                g = await birth_gate(spec, sources=sources, fresh=fresh, anchors=arm.pool.anchors, ctx=arm.ctx or self.ctx, cost_bar=self.cfg.cost_bar,
                                     early_stop=True)
        except Exception as e:  # noqa: BLE001 - an operator that crashes on a task is refused, never fatal
            cand.refused_by = [f"gate:error:{type(e).__name__}"]
            cand.gate = {"error": str(e)[:300]}
            return None
        cand.gate = {"checks": g.checks, "transfer_hits": len(g.detail.get("transfer_hits", [])), "false_alarms": len(g.detail.get("false_alarms", [])),
                     "reproduced": g.detail.get("reproduced"), "probes_per_task": round(g.detail.get("probes_per_task", 0.0), 2), "evidence_domain": dom}
        if not g.passed:
            cand.refused_by = [f"gate:{k}" for k, v in g.checks.items() if not v]
            return None
        if self.cfg.update.stats.paired:
            # pilot p4: three candidates passed the gate on plants the population already caught, then spent α and 60 of
            # the 64 decision units showing a gain of 0. A candidate must first catch a proposal-split plant the
            # incumbent misses (proposal data only: the decision split stays unseen)
            misses = self._round_misses.get(arm.name, [])
            adds = await self._adds(arm, spec, misses)
            cand.gate["adds_on_proposal"] = len(adds)
            cand.gate["adds_ids"] = sorted(adds)
            if not adds:
                cand.refused_by = ["gate:adds_on_proposal"]
                return None
        return g

    def _no_new(self, arm: Arm, cand: Candidate) -> bool:
        """Paired mode: a candidate whose proposal-split catches are all caught by a candidate this round already sent to
        a test is refused before it spends α or decision units (pilot p5: three operators catching the same 4 plants
        took three tests and 60 units)."""
        if not self.cfg.update.stats.paired:
            return False
        seen = self._round_tested.setdefault(arm.name, set())
        mine = set(cand.gate.get("adds_ids", []))
        if mine and mine <= seen:
            cand.refused_by = ["gate:no_new_catches"]
            return True
        seen |= mine
        return False

    async def _adds(self, arm: Arm, spec: OperatorSpec, misses: Sequence[Plant]) -> list[str]:
        """The proposal-split misses of the incumbent that the operator alone confirms a counterexample on."""
        if not misses:
            return []
        if arm.evaluator.shards is not None:
            sh = arm.evaluator.shards
            await sh.add(plants=[p for p in misses if p.id not in sh.known])
            res = await sh.evaluate(Library().add(spec, 0).model_dump(mode="json"), [p.id for p in misses], seed=arm.evaluator.ctx.seed,
                                    max_mutants=arm.evaluator.ctx.max_mutants, base=False)
            return [p.id for p in misses if res[p.id]["hit"]]
        from crucible.synth.meta.library import compile_operator

        op = compile_operator(spec)
        out = []
        for p in misses:
            (r,) = await run_classes(p.task, [op], arm.ctx or self.ctx, cache=True)
            if r.applicable and r.c:
                out.append(p.id)
        return out

    async def _decide_candidate(self, arm: Arm, spec: OperatorSpec, prep: tuple[Candidate, list[Plant | Task], str, list[Plant]], r: int) -> Candidate:
        cand, sources, dom, _fresh = prep
        lib = Library.model_validate(arm.store.get("F"))
        if spec.cls_name in lib.ops:
            cand.refused_by = ["duplicate"]
            return cand
        src_plants = [s for s in sources if isinstance(s, Plant)]
        ev = evidence_of(src_plants, self.builders, [dom])
        new = lib.add(spec, r)
        u = Update(id=f"{arm.name}/r{r}/{spec.cls_name}", component="F", data=new.model_dump(mode="json"), edits=[spec.cls_name], diff=cjson(spec.model_dump(mode="json")),
                   items=[spec.cls_name], author_family=arm.family, evidence=ev)
        dec = await arm.engine.decide(u, r)
        cand.decision = {"accepted": dec.accepted, "rules": dec.rules, "failed": [f.model_dump(mode="json") for f in dec.failed], "metrics": _round_metrics(dec.metrics)}
        cand.accepted = dec.accepted
        cand.refused_by = dec.rules
        return cand

    async def try_candidate(self, arm: Arm, spec: OperatorSpec, cites: list[str], origin: str, clusters: Sequence[Cluster], misses: Sequence[Plant],
                            r: int, rng: random.Random) -> Candidate:
        prep = self._prepare(arm, spec, cites, origin, clusters, misses, r, rng)
        if isinstance(prep, Candidate):
            return prep
        if await self._gate(arm, spec, prep) is None or self._no_new(arm, prep[0]):
            return prep[0]
        return await self._decide_candidate(arm, spec, prep, r)

    async def try_candidates(self, arm: Arm, props: Sequence[tuple[OperatorSpec, list[str], str]], clusters: Sequence[Cluster], misses: Sequence[Plant],
                             r: int, rng: random.Random) -> list[Candidate]:
        """Every candidate of a round: inputs drawn in order; on shards the birth gates run concurrently (independent
        evaluations); the update rule then decides the gate-passing ones one at a time, in proposal order (each decision
        may change the incumbent the next is compared with)."""
        if arm.evaluator.shards is None:
            return [await self.try_candidate(arm, s, c, o, clusters, misses, r, rng) for s, c, o in props]
        preps = [self._prepare(arm, s, c, o, clusters, misses, r, rng) for s, c, o in props]
        todo = [(i, p) for i, p in enumerate(preps) if not isinstance(p, Candidate)]
        gates = await asyncio.gather(*(self._gate(arm, props[i][0], p) for i, p in todo))
        out: list[Candidate] = [p if isinstance(p, Candidate) else p[0] for p in preps]
        for (i, p), g in zip(todo, gates, strict=True):
            if g is not None and not self._no_new(arm, p[0]):
                out[i] = await self._decide_candidate(arm, props[i][0], p, r)
        return out

    def _holdout_gate(self, arm: Arm) -> Any:
        """The generalisation gate "G" on the locked set, read only through Thresholdout (every access logged in the arm's
        ``holdout_access.jsonl``, never the held-out value itself): the locked recall gain of Θ′ over Θ, answered by the
        holdout only when it disagrees with the dev gain by more than the noisy threshold, must be ≥ −tolerance."""
        from crucible.synth.meta.stats import Thresholdout

        async def locked_gain(theta: Any, cand: Any) -> float:
            fs0, fs1 = population(theta.data.get("F")), population(cand.data.get("F"))
            m0 = await self._rmeasure(self.locked.plants, [], fs0, self.report_ctx, set_name=self.locked.name, set_kind="locked", purpose="report")
            m1 = await self._rmeasure(self.locked.plants, [], fs1, self.report_ctx, set_name=self.locked.name, set_kind="locked", purpose="report")
            h0 = {o.plant_id: bool(o.hit) for o in m0.plants}
            h1 = {o.plant_id: bool(o.hit) for o in m1.plants}
            return sum(h1[k] - h0[k] for k in h0) / max(1, len(h0))

        values: dict[str, float] = {}
        assert arm.dir is not None
        ho = Thresholdout(lambda q: values[q], threshold=self.cfg.holdout_threshold, sigma=self.cfg.holdout_sigma, budget=self.cfg.holdout_budget,
                          seed=self.cfg.seed, log=arm.dir / "holdout_access.jsonl")
        arm.holdout = ho

        async def gate(theta: Any, cand: Any, u: Update, train_gain: float) -> list[RuleFailure]:
            if ho.budget > 0:
                values[u.id] = await locked_gain(theta, cand)
            ans = ho.ask(u.id, train_gain, purpose="decide:G")
            values.pop(u.id, None)
            if ans.refused:
                return [RuleFailure(rule="G", detail="the reusable holdout's budget is spent (fail closed)")]
            if ans.answer < -self.cfg.locked_tolerance:
                return [RuleFailure(rule="G", detail=f"locked-set gain (holdout answer) {ans.answer:.4f} < -{self.cfg.locked_tolerance}")]
            return []

        return gate

    async def _propose_planters(self, arm: Arm, r: int, proposers: Proposers) -> tuple[list[PlantOp], dict[str, Any]] | Exception:
        vs = await self.head_verdicts(arm, self.evidence_plants(arm))
        rates: dict[str, list[int]] = {}
        for p in self.evidence_plants(arm):
            rates.setdefault(f"{p.op.name}:{p.op.variant}", []).append(0 if vs[p.id].hit else 1)
        miss_rates = {k: round(sum(v) / len(v), 3) for k, v in sorted(rates.items())}
        mix = dict(Counter(c.defect or "?" for c in self._natural()).most_common())
        try:
            assert proposers.planters is not None
            return await proposers.planters(miss_rates=miss_rates, natural_mix=mix, k=self.cfg.planter_candidates, round=r, history=arm.planter_history[-12:])
        except Exception as e:  # noqa: BLE001 - a failed proposal is recorded
            return e

    async def _headroom(self, arm: Arm, plants: Sequence[Plant], vs: Mapping[str, PlantVerdict]) -> Any:
        from crucible.synth.meta.headroom import HeadroomCache, catchable

        if self.headroom_cache is None:
            self.headroom_cache = HeadroomCache()
        missed = [p.id for p in plants if not vs[p.id].hit]
        anchors = [a for a in arm.pool.anchors if a.id not in arm.pool.decide_sources]
        return await catchable(plants, missed, anchors, arm.ctx or self.ctx, cache=self.headroom_cache)

    async def planter_round(self, arm: Arm, r: int, proposers: Proposers, row: RoundRow, proposed: Any = None) -> None:
        if proposers.planters is None:
            return
        got = proposed if proposed is not None else await self._propose_planters(arm, r, proposers)
        if isinstance(got, Exception):
            row.planter_decisions.append({"error": f"{type(got).__name__}: {str(got)[:300]}"})
            return
        ops, info = got
        held = {(o.name, o.variant) for o in held_out_planter().ops} - {(o.name, o.variant) for o in base_planter().ops}
        tasks = list(self.dev_sources)
        fs = population(arm.store.get("F"))
        ctx = arm.ctx or self.ctx
        for i, op in enumerate(ops[: self.cfg.planter_candidates]):
            if (op.name, op.variant) in held:  # a planter may never produce the locked set's variants on dev
                row.planter_decisions.append({"op": op.key, "accepted": False, "refused_by": ["locked-variant"]})
                continue
            cand = Planter(id=f"{arm.name}-p{r}-{i}", ops=[op], max_per_op=2)

            def critic(pl: Planter) -> list[str]:
                ev = evidence_of(list(arm.pool.plants.values())[:200], self.builders, [])
                return leakage_critic(Update(id="planter", component="P", data=pl.model_dump(mode="json"), diff=cjson([o.params for o in pl.ops]), evidence=ev))

            try:
                dec, plants = await accept_planter(cand, tasks, ctx, reference=self.reference, mu=self.mu, falsifiers=fs, critic=critic,
                                                   population=[p.features for p in arm.pool.plants.values()])
            except Exception as e:  # noqa: BLE001
                row.planter_decisions.append({"op": op.key, "accepted": False, "refused_by": [f"error:{type(e).__name__}"], "detail": str(e)[:300]})
                continue
            # RSI-1: parameterisations that do not change a plant (a factor on eq_to_any, another why text) re-planted copies
            # of existing plants, and planters the population already catches entirely were accepted every round. A
            # planter now joins only with NOVEL plants (content without why texts) and a positive reward (§9.3: P is
            # rewarded for planting defects the current F misses), measured on the novel plants. Paired mode: the reward
            # is measured on the novel plants of PROPOSAL sources only (the decision split's verdicts never reach a
            # proposer); its novel plants on decision sources join the decision split.
            have = {plant_key(p) for p in arm.pool.plants.values()}
            novel = [p for p in plants if plant_key(p) not in have]
            visible = [p for p in novel if p.source_id not in arm.pool.decide_sources]
            nvs, _ = await verdicts(visible, fs, ctx) if visible else ({}, 0)
            missed = [p for p in visible if not nvs[p.id].hit]
            reward = len(missed) / len(visible) if visible else 0.0
            checks = {**dec.checks, "novel": bool(novel), "rewarded": reward > 0}
            if not self.cfg.natural_likeness:
                checks["natural_like"] = True
            head = None
            if self.cfg.planter_reward == "headroom":  # the P/F game: a miss no expressible operator could catch gives F nothing to learn
                h = await self._headroom(arm, visible, nvs) if missed else None
                head = (len(h.catchable) / len(visible)) if h is not None and visible else 0.0
                checks["headroom"] = head > 0
            ok = all(checks.values())
            d = {"op": op.key, "params": op.params, "accepted": ok, "checks": checks, "mmd": dec.mmd, "mu": self.mu, "miss_rate": reward,
                 "headroom_reward": head, "kept": len(novel), "planted": dec.kept, "refused_by": [k for k, v in checks.items() if not v]}
            plants = novel
            arm.planter_history.append({"round": r, "op": f"{op.name}:{op.variant}", "params": op.params, "accepted": ok, "refused_by": d["refused_by"],
                                        "reward": round(reward, 3), "headroom": None if head is None else round(head, 3), "novel_plants": len(novel)})
            if ok:
                data = arm.store.get("P")
                data["planters"].append(cand.model_dump(mode="json"))
                v = arm.store.put("P", data)
                arm.store.set_head("P", v)
                arm.engine.head_digest["P"] = arm.store.digest("P", v)
                arm.pool.add(f"P@{v}", plants)
                if arm.dir is not None:
                    arm.dir.mkdir(parents=True, exist_ok=True)
                    with (arm.dir / "plants.jsonl").open("a") as fh:
                        for p in plants:
                            fh.write(p.model_dump_json() + "\n")
                d["version"] = v
            arm.engine.ledger.append(Decision(kind="update", round=r, update_id=f"{arm.name}/r{r}/planter/{op.key}", component="P", accepted=ok,
                                              evaluated=True, failed=[RuleFailure(rule="P", detail=k) for k, v in checks.items() if not v],
                                              metrics={k: d[k] for k in ("mmd", "mu", "miss_rate", "headroom_reward", "kept")}))
            row.planter_decisions.append(d)
            row.info.setdefault("planter_info", info)

    def _meta_decision(self, arm: Arm, r: int, row: RoundRow, alt_gl: list[str], ev_plants: Sequence[Plant]) -> None:
        """The meta loop (A): A' replaces A when its candidates pass the birth gate more often. Paired mode: a one-sided
        Fisher exact test at an α_t of the run's spender (with fewer than ``meta_min`` candidates a side, no test is spent
        and A' is refused: two against two can never reach significance)."""
        def rate(tag: str) -> tuple[int, int]:
            cs = [c for c in row.candidates if c.origin.endswith(":" + tag)]
            return sum(1 for c in cs if not any(x.startswith("gate:") for x in c.refused_by)), len(cs)

        pa, na = rate("A")
        pb, nb = rate("A'")
        leaks = leakage_critic(Update(id="A", component="A", data={"guidance": alt_gl}, diff=cjson(alt_gl), evidence=evidence_of(ev_plants[:200], self.builders, [])))
        metrics: dict[str, Any] = {"pass_A": [pa, na], "pass_A2": [pb, nb]}
        fail: list[RuleFailure] = []
        if leaks:
            fail.append(RuleFailure(rule="R6", detail=", ".join(leaks[:6])))
        if self.cfg.update.stats.paired:
            from crucible.synth.meta.stats import fisher_one_sided

            if min(na, nb) < self.cfg.meta_min:
                fail.append(RuleFailure(rule="R2", detail=f"meta test needs >= {self.cfg.meta_min} candidates per side (A {na}, A' {nb}): not tested"))
            elif not fail and not arm.engine.spender.exhausted:
                sp = arm.engine._spend(f"{arm.name}/r{r}/guidance", r, None)
                p = fisher_one_sided(pb, nb, pa, na)
                arm.engine.spender.record(sp.t, p <= sp.alpha_t)
                metrics.update({"t": sp.t, "alpha_t": sp.alpha_t, "fisher_p": p})
                if not p <= sp.alpha_t:
                    fail.append(RuleFailure(rule="R2", detail=f"gate pass rate A' {pb}/{nb} vs A {pa}/{na}: Fisher p {p:.4f} > alpha_t {sp.alpha_t:.5f}"))
            elif arm.engine.spender.exhausted:
                fail.append(RuleFailure(rule="R2", detail="the run's α budget is spent"))
        elif not (nb > 0 and na > 0 and pb / nb > pa / na):
            fail.append(RuleFailure(rule="R2", detail=f"gate pass rate A' {pb}/{nb} not above A {pa}/{na}"))
        ok = not fail
        if ok:
            v = arm.store.put("A", {"guidance": alt_gl})
            arm.store.set_head("A", v)
        arm.engine.ledger.append(Decision(kind="update", round=r, update_id=f"{arm.name}/r{r}/guidance", component="A", accepted=ok, evaluated=True,
                                          failed=fail, metrics=metrics))
        row.guidance_decision = {"accepted": ok, **metrics, "rules": [f.rule for f in fail], "new": alt_gl}

    async def run_round(self, arm: Arm, r: int, proposers: Proposers) -> RoundRow:
        row = RoundRow(arm=arm.name, round=r)
        rng = random.Random(f"{self.cfg.seed}/{arm.name}/{r}")
        new_nat = self.refresh_natural()
        if new_nat:
            row.info["natural_new"] = new_nat
        ev_plants = self.evidence_plants(arm)
        vs = await self.head_verdicts(arm, ev_plants)
        misses_c = miss_clusters(ev_plants, vs)
        nat_c = natural_clusters(self._natural())
        clusters = [*nat_c, *misses_c]
        missed = [p for p in ev_plants if not vs[p.id].hit]
        lib = Library.model_validate(arm.store.get("F"))
        k = self.cfg.candidates
        props: list[tuple[OperatorSpec, list[str], str]] = []
        # the planter proposal (a swarm call) runs while this round's operator proposals and decisions run
        pl_task = asyncio.create_task(self._propose_planters(arm, r, proposers)) if (self.cfg.planter_loop and arm.name == "loop"
                                                                                       and proposers.planters is not None) else None
        gl = list(arm.store.get("A")["guidance"])
        alt_gl: list[str] | None = None
        if arm.name == "loop" and self.cfg.meta_loop and proposers.guidance is not None and arm.rows and arm.rows[-1].candidates:
            outcomes = Counter(x for c in arm.rows[-1].candidates for x in c.refused_by)
            try:
                alt_gl, ginfo = await proposers.guidance(guidance=gl, gate_outcomes=dict(outcomes))
                row.info["guidance_info"] = ginfo
            except Exception as e:  # noqa: BLE001
                row.info["guidance_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        try:
            arms_g = [(gl, "A", k)] if alt_gl is None else [(gl, "A", k - k // 2), (alt_gl, "A'", k // 2)]
            outs = await asyncio.gather(*(arm.propose_ops(clusters=[c.shown() for c in nat_c], misses=[c.shown() for c in misses_c], guidance=g,
                                                          library=[s.model_dump(mode="json") for s in lib.ops.values()], k=kk, round=r, tag=tag)
                                          for g, tag, kk in arms_g), return_exceptions=True)
            for (_g, tag, kk), out in zip(arms_g, outs, strict=True):
                if isinstance(out, BaseException):
                    row.info.setdefault("propose_error", f"{type(out).__name__}: {str(out)[:300]}")
                    continue
                ops, info = out
                props += [(s, c, f"{'random' if arm.name == 'control' else 'swarm'}:{tag}") for s, c in ops[:kk]]
                row.lemmas += info.get("lemmas", [])
                row.strategies += info.get("strategies", [])
                if info.get("errors"):
                    row.info.setdefault("parse_errors", []).extend(info["errors"])
        except Exception as e:  # noqa: BLE001 - a failed proposal round is recorded; the round still measures
            row.info["propose_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        self._round_misses[arm.name] = missed
        self._round_tested[arm.name] = set()
        row.candidates += await self.try_candidates(arm, props, clusters, missed, r, rng)
        if alt_gl is not None:
            self._meta_decision(arm, r, row, alt_gl, ev_plants)
        if pl_task is not None:
            await self.planter_round(arm, r, proposers, row, proposed=await pl_task)

        def remove(data: Any, items: list[str]) -> Any:
            lib2 = Library.model_validate(data)
            for i in items:
                lib2.ops.pop(i, None)
                lib2.retired.append(i)
            return lib2.model_dump(mode="json")

        pr = await arm.engine.prune("F", r, remove)
        if pr is not None:
            row.pruned = list(pr.metrics.get("items", [])) if pr.accepted else []
            row.info["prune"] = {"accepted": pr.accepted, "rules": pr.rules}
        if self.cfg.audit:
            au = await arm.engine.audit(r)
            row.rollback = {"applied": au.accepted, **_round_metrics(au.metrics)}
        await self._report(arm, row)
        row.info["spend_usd"] = round(self.spend(), 4)
        arm.rows.append(row)
        return row

    async def _report(self, arm: Arm, row: RoundRow) -> None:
        row.library = sorted(Library.model_validate(arm.store.get("F")).ops)
        row.planters = len(arm.store.get("P")["planters"])
        allp = list(arm.pool.plants.values())
        dev_vs = await self.head_verdicts(arm, allp)
        row.dev = {"recall": round(sum(1 for v in dev_vs.values() if v.hit) / max(1, len(dev_vs)), 4), "n": len(dev_vs)}
        prop = self.evidence_plants(arm) if self.cfg.update.stats.paired else allp
        hits = sum(1 for p in prop if dev_vs[p.id].hit)
        game: dict[str, Any] = {"f_recall": round(hits / max(1, len(prop)), 4), "missed": len(prop) - hits, "n": len(prop),
                                "planters": row.planters, "pool_kl": round(kl(class_mix(allp), self.mix0), 4), "size": plant_size(allp),
                                "library": len(row.library)}
        if self.reference and allp:
            game["pool_mmd"] = round(mmd(feature_matrix([p.features for p in allp]), feature_matrix(self.reference)), 4)
        if self.cfg.update.stats.paired:
            game["decide_left"] = len(arm.sampler.remaining())
            game["alpha_spent"] = round(arm.engine.spender.total, 6)
            game["tests"] = len(arm.engine.spender.spent)
        rewards = [d.get("miss_rate") for d in row.planter_decisions if d.get("accepted")]
        game["planter_rewards"] = rewards
        if self.cfg.headroom:
            h = await self._headroom(arm, prop, dev_vs)
            game["headroom_abs"] = round(h.headroom_abs, 4)
            game["headroom_rel"] = round(h.headroom_rel, 4)
            game["headroom_by_class"] = h.by_class
            game["headroom_witnesses"] = dict(list(h.witnesses.items())[:6])
        row.game = game
        row.locked = await self.locked_summary(arm)
        if self.family is not None:
            fm = await self._rmeasure(self.family.plants, self.family_anchors, population(arm.store.get("F")), self.report_ctx, set_name=self.family.name,
                               set_kind="locked", purpose="report")
            row.family = summary(fm)

    async def _rmeasure(self, *args: Any, **kw: Any) -> RecallMatrix:
        """``recall.measure`` on the shared report context, one caller at a time (pilot p3: two arms measuring the locked
        set concurrently through one context reset a file-backed World under each other: SnapshotMismatch)."""
        async with self._report_lock:
            return await measure(*args, **kw)

    async def locked_summary(self, arm: Arm) -> dict[str, Any]:
        m = await self._rmeasure(self.locked.plants, self.locked_anchors, population(arm.store.get("F")), self.report_ctx, set_name=self.locked.name,
                          set_kind="locked", purpose="report")
        return summary(m)

    def stop_reason(self, arm: Arm, r: int) -> str | None:
        st = self.cfg.stop
        if r >= (st.max_rounds or self.cfg.rounds):
            return "max_rounds"
        if st.max_spend_usd is not None and self.spend() >= st.max_spend_usd:
            return "spend"
        if self.cfg.update.stats.paired:
            if arm.engine.spender.exhausted:
                return "alpha_spent"
            if len(arm.sampler.remaining()) < self.cfg.update.stats.n_min:
                return "decision_split_dry"
        rb = sum(1 for x in arm.rows if x.rollback and x.rollback.get("applied"))
        if st.max_rollbacks is not None and rb >= st.max_rollbacks:
            return "rollbacks"
        if st.max_pool_kl is not None and arm.rows and arm.rows[-1].game.get("pool_kl", 0.0) > st.max_pool_kl:
            return "goodhart_pool_kl"
        if st.futility_rounds:
            tail = [x for x in arm.rows if x.round > 0][-st.futility_rounds:]
            if len(tail) == st.futility_rounds and not any(c.accepted for x in tail for c in x.candidates) \
                    and not any(d.get("accepted") for x in tail for d in x.planter_decisions) \
                    and all(x.game.get("headroom_abs", 0.0) <= st.min_headroom for x in tail):
                return "futility"
        return None

    async def clean_anchors(self, a: Arm) -> dict[str, list[int]]:
        """Paired mode: an anchor must be clean at EVERY evaluator seed (the RSI-2 pilot's A/A found TWIN alarming on one
        it_helpdesk source at seeds 3 and 6: with it as an anchor, R3's FA ≤ 0 would refuse every candidate drawn at
        those seeds). An anchor that alarms at some seed is dropped and reported; its plants stay source-relative."""
        seeds = sorted({a.evaluator.ctx.seed, *self.cfg.update.stats.seeds})
        res = await asyncio.gather(*(a.evaluator._units({}, [], a.pool.anchors, a.evaluator.ctx.max_mutants, s) for s in seeds))
        bad: dict[str, list[int]] = {}
        for s, (_vs, alarms, _p) in zip(seeds, res, strict=True):
            for aid, on in alarms.items():
                if on:
                    bad.setdefault(aid, []).append(s)
        if bad:
            a.pool.anchors = [t for t in a.pool.anchors if t.id not in bad]
            a.pool.dropped.update({k: [f"alarms at seeds {v}"] for k, v in bad.items()})
        return bad

    async def _start(self, a: Arm) -> None:
        dropped = await self.clean_anchors(a) if self.cfg.update.stats.paired else {}
        await a.engine.calibrate(0)
        r0 = RoundRow(arm=a.name, round=0, library=[], planters=1)
        await self._report(a, r0)
        r0.info = {"noise_floor": a.engine.floor.model_dump(mode="json") if a.engine.floor else None, "mu": self.mu,
                   "decide_sources": sorted(a.pool.decide_sources), "decide_units": len(a.pool.decide_ids()), "anchors_dropped": dropped,
                   "anchors": len(a.pool.anchors)}
        a.rows.append(r0)
        self.save_state(a)

    async def run(self, proposers: Proposers, *, control: bool = True, save: Callable[[], None] | None = None, resume: bool = False,
                  max_new_rounds: int | None = None) -> list[RoundRow]:
        """Run (or, with ``resume``, continue) every arm round after round until a stop rule fires. The arms run
        concurrently (each with its own Worlds when ``make_ctx`` is given): the control arm's CPU work overlaps the loop
        arm's swarm calls."""
        if not self.arms:
            self.arm("loop", proposers.ops or random_proposer(self.cfg.seed), family=proposers.family)
            if control:
                self.arm("control", random_proposer(self.cfg.seed + 1))
        arms = list(self.arms)
        last = {a.name: (self.load_state(a) if resume else -1) for a in arms}
        await asyncio.gather(*(self._start(a) for a in arms if last[a.name] < 0))
        for a in arms:
            last[a.name] = max(last[a.name], 0)
        if save:
            save()
        done = 0

        async def step(a: Arm) -> None:
            r = last[a.name] + 1
            row = await self.run_round(a, r, proposers)
            last[a.name] = r
            a.stopped = self.stop_reason(a, r)
            row.stop = a.stopped
            self.save_state(a)

        while True:
            for a in arms:
                why = None if a.stopped else self.stop_reason(a, last[a.name])
                if why is not None:
                    a.stopped = why
                    if a.rows:
                        a.rows[-1].stop = why
                    self.save_state(a)
            live = [a for a in arms if a.stopped is None]
            if not live or (max_new_rounds is not None and done >= max_new_rounds):
                break
            await asyncio.gather(*(step(a) for a in live))
            done += 1
            if save:
                save()
        return [row for a in arms for row in a.rows]


def _round_metrics(m: Mapping[str, Any]) -> dict[str, Any]:
    out = {}
    for k, v in m.items():
        out[k] = round(v, 5) if isinstance(v, float) and math.isfinite(v) else v
    return out


# ---------------------------------------------------------------------------------------------------------- reports

def rows_markdown(rows: Sequence[RoundRow]) -> str:
    lines = ["| arm | round | library | planters | candidates | gate passed | accepted | dev recall | locked FR (hits/n) | locked FA | spend $ |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        gp = sum(1 for c in r.candidates if c.gate and not any(x.startswith("gate:") for x in c.refused_by))
        acc = sum(1 for c in r.candidates if c.accepted)
        lk = r.locked
        lines.append(f"| {r.arm} | {r.round} | {len(r.library)} | {r.planters} | {len(r.candidates)} | {gp} | {acc} | {r.dev.get('recall', '')} | "
                     f"{lk.get('fr_all', '')} ({lk.get('hits', '')}/{lk.get('n', '')}) | {lk.get('fa', '')}/{lk.get('fa_n', '')} | {r.info.get('spend_usd', '')} |")
    return "\n".join(lines) + "\n"


def refusals(rows: Sequence[RoundRow]) -> dict[str, dict[str, int]]:
    """arm -> refusing rule or gate check -> count (non-compensatory: a candidate counts once per failing rule)."""
    out: dict[str, Counter[str]] = {}
    for r in rows:
        c = out.setdefault(r.arm, Counter())
        for x in r.candidates:
            for k in (x.refused_by or (["accepted"] if x.accepted else [])):
                c[k] += 1
    return {a: dict(c.most_common()) for a, c in out.items()}


def plants_digest(plants: Sequence[Plant]) -> str:
    return set_digest(plants)


# ---------------------------------------------------------------------------------------------------------- the per-task probe loop
#
# RSI at task-creation time (lead, 2026-10-29; runs/rsi-stats.md §P). Nothing here trains anything:
#   1. a task is created (admitted by its certificate);
#   2. the student runs on it as PROBE rollouts (``StudentProbe``: the learnability agent's runner);
#   3. each outcome is classified (``OutcomeClassifier``: the learnability agent's classifier) as an intended pass (the
#      skill was used), an intended fail (a genuine skill gap), an unintended pass (shortcut, leak, vacuous check, frame
#      hole, hack path) or an unintended fail (ambiguous instruction, hidden information, over-strict check, World or
#      tool bug, contradictory docs, unsolvable variant);
#   4. unintended outcomes WITH executed or calibrated evidence become counterexamples (executed: a replayable student
#      program with the verdict the task should have given) or panel-adjudicated suspects, and drive a repair of the
#      task, its World instance or its contract through the existing CEGIS repairer under the monotone archive;
#   5. the lesson is lifted into the system: the counterexample enters the archive (natural clusters → falsifier
#      operators, through ``Loop``), the repair becomes a candidate design lemma for the writers, a World or tool bug
#      becomes a World-builder fix; system-level updates are accepted only under the update rule with generalisation
#      (R7 on held-out domains, the held-out World family, the locked set through the reusable holdout);
#   6. repeat, round after round.
#
# The regularisers of a task repair (``ProbeRules``; every one is checked, non-compensatory):
#   T1 evidence      a repair needs ≥ 1 unintended outcome with executed or calibrated evidence. Judged-only (uncalibrated)
#                    or evidence-free "unintended" labels are demoted: a failure becomes an unattributed fail, treated as
#                    a skill gap; a pass becomes a suspect for the panel queue. No repair is ever triggered by a pass rate.
#   T2 no easing     every intended (and unattributed) fail's program is replayed on the repaired task: one that now
#                    passes means the repair eased a genuine skill gap → refused.
#   T3 no tightening every intended pass's program is replayed: one that now fails means the repair over-constrained
#                    the task (a new D6) → refused.
#   T4 closes        every unintended pass's program must now fail and every executed unintended fail's (valid)
#                    program must now pass: the repair fixes what it was justified by.
#   T5 archive       the monotone archive: no archived counterexample re-opens; the deterministic classes stay clean.
#   T6 caps          ≤ ``max_repairs`` per task; calibrated-only (judge) evidence stops after ``judge_cap`` repairs of a
#                    task without new executed evidence (R8).
#   T7 attribution   a failure counts as task-caused only with an independent check: an unintended fail claiming the
#                    task cannot be solved as stated (ambiguous, hidden information, unsolvable, contradictory docs) is
#                    demoted when an ALT solver of a non-student, non-builder family solved the same task.
# J and every acceptance gate never read the student pass rate (``ProbeLoop.decide`` never receives it). The
# student-tuning drift monitor (``TuningMonitor``) reads it on re-probes of repaired tasks, for an alarm only.

OutcomeKind = Literal["intended_pass", "intended_fail", "unintended_pass", "unintended_fail", "masked"]
Evid = Literal["executed", "calibrated", "judged", "none"]
UNSOLVABLE_CLAIMS = ("ambiguous", "hidden_info", "unsolvable", "contradictory_docs")
WORLD_MECHANISMS = ("world_bug", "tool_bug", "contradictory_docs")
PASS_MECHANISMS = ("shortcut", "leak", "vacuous_check", "frame_hole", "hack")
FAIL_MECHANISMS = ("ambiguous", "hidden_info", "over_strict", "world_bug", "tool_bug", "contradictory_docs", "unsolvable")


class ProbeOutcome(CrucibleModel):
    """One classified probe rollout, as this loop consumes the outcome classifier (extra fields are ignored)."""

    model_config = ConfigDict(extra="ignore")
    i: int
    passed: bool | None
    kind: OutcomeKind
    mechanism: str = ""
    defect: str | None = None
    evidence: Evid = "none"
    alt_solved: bool | None = None  # an independent ALT solver (another family) solved the same task
    alt_family: str = ""
    program: dict[str, Any] | None = None  # the rollout as a literal program (replayable)
    note: str = ""


class OutcomeClassifier(Protocol):
    async def classify(self, task: Task, rollouts: Sequence[Any]) -> list[ProbeOutcome]: ...


_CAUSE_MECHANISM = (("grader_error", "grader_error"), ("unsolvable", "unsolvable"), ("nondeterministic", "world_bug"), ("vacuous", "vacuous_check"),
                    ("leak", "leak"), ("affordance", "hack"), ("frame hole", "frame_hole"), ("lax requirement", "shortcut"), ("lax value", "shortcut"),
                    ("over-strict", "over_strict"), ("hidden information", "hidden_info"), ("the World crashed", "world_bug"),
                    ("skill_gap", "skill_gap"))


def mechanism_of(cause: str) -> str:
    c = cause.strip()
    for prefix, mech in _CAUSE_MECHANISM:
        if c.startswith(prefix):
            return mech
    return c.split(":", 1)[0].split(";", 1)[0][:40]


def from_outcome_record(rec: Any, rollout: Any, *, alt: Sequence[int] | None = None) -> ProbeOutcome:
    """The learnability agent's ``synth.outcome.OutcomeRecord`` as this loop's ``ProbeOutcome``: its ``uncertain`` label
    (never a task defect) maps to ``masked`` for a pass-less or masked rollout and otherwise to an unevidenced label that
    T1 demotes (an uncertain fail is an unattributed fail, an uncertain pass a suspect); decided by execution → executed
    evidence, by the calibrated panel → calibrated; the certification's ALT solvers (another family, never a builder)
    are the independent solver of T7."""
    label = rec.label
    ev: Evid = {"execution": "executed", "panel": "calibrated"}.get(rec.decided_by, "none")  # type: ignore[assignment]
    if label == "uncertain":
        if rollout.passed is None:
            kind: OutcomeKind = "masked"
        else:
            kind, ev = ("unintended_pass" if rollout.passed else "unintended_fail"), "none"
    else:
        kind = label
    solved = bool(alt and alt[0] > 0)
    return ProbeOutcome(i=rollout.i, passed=rollout.passed, kind=kind, mechanism=mechanism_of(rec.cause), defect=None, evidence=ev,
                        alt_solved=solved if alt else None, alt_family="certification-alt" if alt else "", program=getattr(rollout, "program", None),
                        note=rec.cause[:300])


class LearnClassifier:
    """``OutcomeClassifier`` over ``synth.outcome`` (the learnability agent's classifier): task evidence once per task
    version (``task_evidence``), then ``classify`` per rollout, each mapped by ``from_outcome_record``."""

    def __init__(self, ctx: FalsifyContext, *, panel: Any = None, alt: Mapping[str, Sequence[int]] | None = None, mut: bool = False):
        self.ctx, self.panel, self.alt, self.mut = ctx, panel, dict(alt or {}), mut
        self.records: list[Any] = []

    async def classify(self, task: Task, rollouts: Sequence[Any]) -> list[ProbeOutcome]:
        from crucible.synth.outcome import classify, task_evidence

        alt = self.alt.get(task.id)
        tev = await task_evidence(task, self.ctx, alt=alt, mut=self.mut)
        out = []
        for r in rollouts:
            rec = await classify(task, r, self.ctx, tev, panel=self.panel)
            self.records.append(rec)
            out.append(from_outcome_record(rec, r, alt=alt))
        return out


class ProbeRepairer(Protocol):
    """The existing CEGIS repairer (task, World instance or contract) under the monotone archive."""

    async def __call__(self, task: Task, counterexamples: list[Counterexample]) -> Task | None: ...


class ProbeRules(CrucibleModel):
    rollouts: int = 8
    max_repairs: int = 3
    judge_cap: int = 2
    student_family: str = ""
    builder_families: list[str] = Field(default_factory=list)
    reprobe: int = 0  # re-probe rollouts after a repair (the drift monitor only; 0: no re-probe)


class TaskProbe(CrucibleModel):
    """One task's probe round: the counted outcomes, the counterexamples, the repair decision and the lifted lessons."""

    task_id: str
    version: int
    round: int
    outcomes: list[ProbeOutcome] = Field(default_factory=list)
    counted: dict[str, int] = Field(default_factory=dict)  # kind -> count, after demotions
    demoted: list[str] = Field(default_factory=list)
    counterexamples: list[str] = Field(default_factory=list)
    suspects: list[str] = Field(default_factory=list)  # unintended passes without executed evidence: the panel queue
    repair: Literal["none", "accepted", "refused", "declined", "capped"] = "none"
    refused_by: list[str] = Field(default_factory=list)
    new_version: int | None = None
    lessons: dict[str, list[str]] = Field(default_factory=dict)  # archive | lemma | world_fix -> ids
    drift: dict[str, float] | None = None


class ProbeReplay(BaseFalsifier):
    """Replays archived student-probe counterexamples (class PROBE) in the monotone archive's regressions; it proposes
    nothing itself."""

    name: ClassVar[str] = "PROBE"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Any]:
        return []


def probe_counterexample(task: Task, o: ProbeOutcome) -> Counterexample | None:
    """An executed unintended outcome as an archive counterexample: the student's program with the verdict the task
    should have given (an unintended pass must fail; an over-strict or World-caused fail of a valid program must pass)."""
    if o.evidence != "executed" or o.program is None or o.kind not in ("unintended_pass", "unintended_fail"):
        return None
    from crucible.synth.falsify.base import Probe as _Probe

    prog = Program.model_validate(o.program)
    expect = "fail" if o.kind == "unintended_pass" else "pass"
    probe = _Probe(cls="PROBE", op=o.mechanism or o.kind, site=f"rollout{o.i}", program=prog, expect=expect)
    cid = content_hash({"task": task.id, "v": task.version, "probe": probe.digest(), "kind": o.kind}, 24)
    return Counterexample(id=f"cx-{cid}", domain=task.domain, task_id=task.id, task_version=task.version, cls="PROBE", op=probe.op, site=probe.site,
                          defect=o.defect, label="confirmed", probe=probe)


def count_outcome(o: ProbeOutcome, rules: ProbeRules) -> tuple[OutcomeKind | Literal["unattributed_fail", "suspect"], str | None]:
    """T1 and T7: the kind this loop acts on, and the demotion reason when it differs from the classifier's."""
    if o.kind == "masked" or o.passed is None:
        return "masked", None
    if o.kind.endswith("_pass") != bool(o.passed):  # the trusted grader's verdict decides pass or fail; a label that contradicts it is void
        return "masked", f"rollout {o.i}: labelled {o.kind} but the grader says {'pass' if o.passed else 'fail'} (ignored)"
    if o.kind in ("intended_pass", "intended_fail"):
        return o.kind, None
    if o.evidence not in ("executed", "calibrated"):
        return ("suspect", f"rollout {o.i}: {o.kind} with {o.evidence} evidence (T1)") if o.kind == "unintended_pass" else \
               ("unattributed_fail", f"rollout {o.i}: {o.kind} with {o.evidence} evidence (T1)")
    if o.kind == "unintended_fail" and o.mechanism in UNSOLVABLE_CLAIMS and o.alt_solved and o.alt_family and o.alt_family != rules.student_family \
            and o.alt_family not in rules.builder_families:
        return "unattributed_fail", f"rollout {o.i}: claims {o.mechanism}, but the independent {o.alt_family} solver solved the task (T7)"
    return o.kind, None


class TuningMonitor(CrucibleModel):
    """Student-tuning drift: on re-probed repaired tasks, the change of the student's pass rate that the fixed defects do
    not explain. Per task: d = p_after − (passes_before − unintended_passes)/n_before; a fixed unintended fail may
    explain a rise of up to unintended_fails/n_before; u = d − clip(d, 0, that bound). The alarm fires when at least
    ``min_tasks`` tasks give a mean u whose 95% bootstrap interval excludes 0 and whose size exceeds ``threshold``.
    It reads pass rates; nothing that accepts anything reads it."""

    threshold: float = 0.1
    min_tasks: int = 5
    rows: list[dict[str, float]] = Field(default_factory=list)

    def add(self, *, n_before: int, passes_before: int, unintended_passes: int, unintended_fails: int, n_after: int, passes_after: int) -> dict[str, float]:
        base = (passes_before - unintended_passes) / n_before
        d = passes_after / n_after - base
        hi = unintended_fails / n_before
        u = d - min(max(d, 0.0), hi)
        row = {"d": round(d, 6), "explained_max": round(hi, 6), "unexplained": round(u, 6)}
        self.rows.append(row)
        return row

    def alarm(self) -> dict[str, Any]:
        from crucible.synth.meta.stats import bootstrap_ci

        us = [r["unexplained"] for r in self.rows]
        if len(us) < self.min_tasks:
            return {"alarm": False, "n": len(us), "why": f"fewer than {self.min_tasks} re-probed repaired tasks"}
        m = sum(us) / len(us)
        lo, hi = bootstrap_ci(us, seed=len(us))
        on = abs(m) > self.threshold and (lo > 0 or hi < 0)
        return {"alarm": on, "n": len(us), "mean_unexplained": round(m, 4), "ci95": [round(lo, 4), round(hi, 4)]}


Replayer = Callable[[Task, dict[str, Any]], Awaitable[bool | None]]  # program -> passes the task's contract (None: no verdict)
Recheck = Callable[[Task], Awaitable[list[str]]]  # the monotone archive and the deterministic classes: failures, [] = clean
LemmaLift = Callable[[Task, Task, list[Counterexample]], Awaitable[list[Any]]]


async def replay_program(task: Task, program: dict[str, Any], ctx: FalsifyContext) -> bool | None:
    """Run a literal program on the task's World and grade it with the task's contract (the trusted grader)."""
    r = await ctx.run(task, Program.model_validate(program))
    if r.error is not None and r.error.startswith("agent:"):
        return None
    return bool(ctx.grade(r, task.contract).passed)


def archive_recheck(archive: Any, ctx: FalsifyContext, falsifiers: Callable[[], list[BaseFalsifier]] = cheap, *, heldout: float | None = None) -> Recheck:
    """T5 on the real archive: no archived counterexample (student-probe ones included) re-opens, and no deterministic
    class confirms a counterexample on the repaired version. With held-out mutants (Part 1 V5) pass
    ``falsifiers=lambda: population(h)``: a repair is then accepted on the shown mutants only, and ``MUTHeldOut(holdout=h)``
    stays for the certified version. ``heldout`` > 0 does that for the default population (DESIGN §7.7 item 22: the
    re-certified pool's manifest carries the h its certificates used)."""
    from crucible.synth.archive import regress
    from crucible.synth.falsify import population as held_population

    if heldout is None:  # R-holdout (DESIGN item 24): the policy's h; 0 stays available for a measurement
        from crucible.synth import policy

        heldout = policy.holdout()
    if heldout > 0 and falsifiers is cheap:
        falsifiers = lambda: held_population(heldout)

    async def check(task: Task) -> list[str]:
        fs = [*falsifiers(), ProbeReplay()]
        rep = await regress(archive, task, ctx, fs)
        out = [f"archive: {x} re-opened" for x in rep.reopened]
        res = await run_classes(task, falsifiers(), ctx)
        out += [f"{r.cls} confirms a counterexample" for r in res if r.applicable and r.c]
        return out

    return check


class ProbeLoop:
    """The per-task student-probe → classify → repair → lift loop (section comment above). Continuous and resumable:
    every task's rounds are appended to a hash-chained ledger (``probe_ledger.jsonl``: kinds ``probe`` and ``repair``)
    and ``state.json`` keeps each task's current version, repairs and judge streak."""

    def __init__(self, root: Path, *, probe: Any, classify: OutcomeClassifier, repair: ProbeRepairer, replay: Replayer, recheck: Recheck,
                 archive: Any = None, lift_lemmas: LemmaLift | None = None, rules: ProbeRules | None = None, monitor: TuningMonitor | None = None):
        self.root = Path(root)
        self.probe, self.classify, self.repair, self.replay, self.recheck = probe, classify, repair, replay, recheck
        self.archive, self.lift_lemmas = archive, lift_lemmas
        self.rules = rules or ProbeRules()
        self.monitor = monitor or TuningMonitor()
        self.ledger = Ledger(self.root / "probe_ledger.jsonl")
        self.tasks: dict[str, Task] = {}
        self.repairs: dict[str, int] = {}
        self.judge_streak: dict[str, int] = {}
        self.history: list[TaskProbe] = []
        self.lemma_candidates: list[Any] = []
        self.world_fixes: list[dict[str, Any]] = []
        self._load()

    def _load(self) -> None:
        p = self.root / "state.json"
        if not p.exists():
            return
        d = json.loads(p.read_text())
        self.tasks = {k: Task.model_validate(v) for k, v in d.get("tasks", {}).items()}
        self.repairs = {k: int(v) for k, v in d.get("repairs", {}).items()}
        self.judge_streak = {k: int(v) for k, v in d.get("judge_streak", {}).items()}
        self.history = [TaskProbe.model_validate(x) for x in d.get("history", [])]
        self.monitor = TuningMonitor.model_validate(d["monitor"]) if d.get("monitor") else self.monitor
        self.world_fixes = list(d.get("world_fixes", []))

    def save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        doc = {"tasks": {k: t.model_dump(mode="json") for k, t in self.tasks.items()}, "repairs": self.repairs, "judge_streak": self.judge_streak,
               "history": [h.model_dump(mode="json") for h in self.history], "monitor": self.monitor.model_dump(mode="json"), "world_fixes": self.world_fixes}
        tmp = self.root / "state.json.tmp"
        tmp.write_text(json.dumps(doc, default=str))
        tmp.replace(self.root / "state.json")

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        for t in tasks:
            self.tasks.setdefault(t.id, t)

    async def decide(self, task: Task, new: Task, outcomes: Sequence[tuple[ProbeOutcome, str]], cxs: Sequence[Counterexample]) -> list[str]:
        """T2–T5 on a candidate repair. It is given the counted outcomes and their programs, never a pass rate."""
        why: list[str] = []
        for o, kind in outcomes:
            if o.program is None:
                continue
            ok = await self.replay(new, o.program)
            if ok is None:
                continue
            if kind in ("intended_fail", "unattributed_fail") and ok:
                why.append(f"T2: rollout {o.i} ({kind}) now passes: the repair eases a skill gap")
            elif kind == "intended_pass" and not ok:
                why.append(f"T3: rollout {o.i} (intended pass) now fails: the repair over-constrains the task")
            elif kind == "unintended_pass" and ok:
                why.append(f"T4: rollout {o.i} (unintended pass, {o.mechanism}) still passes")
            elif kind == "unintended_fail" and o.evidence == "executed" and o.mechanism not in WORLD_MECHANISMS and not ok:
                why.append(f"T4: rollout {o.i} (unintended fail, {o.mechanism}) still fails")
        why += [f"T5: {x}" for x in await self.recheck(new)]
        return why

    async def probe_task(self, task: Task, round: int) -> TaskProbe:
        rules = self.rules
        rec = TaskProbe(task_id=task.id, version=task.version, round=round)
        rollouts = await self.probe.rollouts(task, rules.rollouts)
        for r in rollouts:  # the loop may see each rollout's outcome, never an aggregate it could tune towards
            if isinstance(r, Mapping) and any(r.get(k) is not None for k in ("pass_rate", "p_hat")):
                raise ValueError("a probe rollout carries an aggregate pass rate")
        outs = await self.classify.classify(task, rollouts)
        rec.outcomes = list(outs)
        counted: list[tuple[ProbeOutcome, str]] = []
        for o in outs:
            kind, why = count_outcome(o, rules)
            counted.append((o, kind))
            if why:
                rec.demoted.append(why)
            if kind == "suspect":
                rec.suspects.append(f"rollout{o.i}")
        rec.counted = dict(Counter(k for _, k in counted))
        unintended = [(o, k) for o, k in counted if k in ("unintended_pass", "unintended_fail")]
        cxs = [c for o, _ in unintended if (c := probe_counterexample(task, o)) is not None]
        rec.counterexamples = [c.id for c in cxs]
        if self.archive is not None and cxs:
            rec.lessons["archive"] = self.archive.add_all(cxs)
        for o, _ in unintended:
            if o.mechanism in WORLD_MECHANISMS:
                fix = {"task_id": task.id, "domain": task.domain, "world": task.world.model_dump(mode="json"), "mechanism": o.mechanism, "rollout": o.i,
                       "evidence": o.evidence, "round": round}
                self.world_fixes.append(fix)
                rec.lessons.setdefault("world_fix", []).append(f"{task.id}/rollout{o.i}")
        self.ledger.append(Decision(kind="probe", round=round, update_id=f"{task.id}@{task.version}", component="T", evaluated=True, accepted=True,
                                    metrics={"counted": rec.counted, "demoted": rec.demoted, "counterexamples": rec.counterexamples}))
        if not unintended:  # T1: nothing unintended with evidence, nothing to repair (an intended fail is a skill gap: never eased)
            self.history.append(rec)
            return rec
        executed = any(o.evidence == "executed" for o, _ in unintended)
        n = self.repairs.get(task.id, 0)
        if n >= rules.max_repairs or (not executed and self.judge_streak.get(task.id, 0) >= rules.judge_cap):
            rec.repair, rec.refused_by = "capped", [f"T6: {n} repairs, judge streak {self.judge_streak.get(task.id, 0)}"]
            self.history.append(rec)
            return rec
        new = await self.repair(task, list(cxs))
        if new is None:
            rec.repair = "declined"
        else:
            why = await self.decide(task, new, counted, cxs)
            rec.refused_by = why
            if why:
                rec.repair = "refused"
            else:
                rec.repair, rec.new_version = "accepted", new.version
                self.tasks[task.id] = new
                self.repairs[task.id] = n + 1
                self.judge_streak[task.id] = 0 if executed else self.judge_streak.get(task.id, 0) + 1
                if self.lift_lemmas is not None:
                    lem = await self.lift_lemmas(task, new, list(cxs))
                    self.lemma_candidates += lem
                    rec.lessons["lemma"] = [getattr(x, "id", str(x)) for x in lem]
                if rules.reprobe:
                    after = await self.probe.rollouts(new, rules.reprobe, start=rules.rollouts)
                    graded = [r for r in after if getattr(r, "passed", None) is not None]
                    nb = sum(1 for o, k in counted if k != "masked")
                    if graded and nb:
                        rec.drift = self.monitor.add(n_before=nb, passes_before=sum(1 for o, k in counted if k != "masked" and o.passed),
                                                     unintended_passes=sum(1 for _, k in counted if k == "unintended_pass"),
                                                     unintended_fails=sum(1 for _, k in counted if k == "unintended_fail"),
                                                     n_after=len(graded), passes_after=sum(1 for r in graded if r.passed))
        self.ledger.append(Decision(kind="repair", round=round, update_id=f"{task.id}@{task.version}", component="T", evaluated=True,
                                    accepted=rec.repair == "accepted", to_version=rec.new_version,
                                    failed=[RuleFailure(rule=x.split(":", 1)[0], detail=x) for x in rec.refused_by], metrics={"repair": rec.repair}))
        self.history.append(rec)
        return rec

    async def run_round(self, round: int, *, ids: Sequence[str] | None = None) -> list[TaskProbe]:
        out = [await self.probe_task(self.tasks[i], round) for i in (ids or sorted(self.tasks))]
        self.save()
        return out

    def summary(self) -> dict[str, Any]:
        tot: Counter[str] = Counter()
        for h in self.history:
            tot.update(h.counted)
        return {"probed": len(self.history), "counted": dict(tot), "repairs": dict(Counter(h.repair for h in self.history)),
                "counterexamples": sum(len(h.counterexamples) for h in self.history), "suspects": sum(len(h.suspects) for h in self.history),
                "world_fixes": len(self.world_fixes), "lemma_candidates": len(self.lemma_candidates), "drift": self.monitor.alarm()}


# ---------------------------------------------------------------------------------------------------------- the typed per-task loop
#
# The lead's spec (2026-10-30): "works for all tasks" means every task gets a correct, typed decision, not that every
# task is admitted. Per task and round:
#
#   precondition  ORACLE: the reference decision is checked against the World's executed rules (falsify.oracle.ORACLE)
#                 and against independent runs (decision_disagreement). Wrong → REPAIR the oracle; disputed → HOLD for
#                 the panel. No difficulty move is made on a task whose reference decision is wrong or disputed.
#   REPAIR        an unintended fail or pass with executed or calibrated evidence (T1–T7 of ProbeLoop)
#   DESCEND       intended fails, band too hard: one declared knob step easier → a NEW variant; the original is kept
#   ASCEND        intended passes, band too easy: one declared knob step harder → a new variant
#   KEEP          in band, valid
#   RESERVE       (was RETIRE; lead, 2026-10-30: kept for stronger checkpoints, never deleted) no lawful move (no declared knob in the direction, a move against the lineage's direction), or
#                 repeated refusals / the round cap without new executed information
#   HOLD          the evidence does not decide (a disputed oracle, or most failures uncertain): nothing is edited
#
# Regularisers (all checked; ``TaskRules``):
#   G1 evidence-gated edits: a REPAIR cites a counted unintended outcome; a move cites a declared knob and its direction
#   G2 frozen intent: the variant keeps the skill and the cell, passes the P3 probes (``p3_check``); else refused
#   G3 full re-certification under the monotone archive (``recheck``) after every edit
#   G4 sample splitting: the rollouts that diagnosed a task never accept its edit; acceptance draws FRESH rollouts (new
#      rollout indices, so new seeds), sequentially up to ``k_max`` (the band error at k ≤ 8 is 0.29–0.37)
#   G5 bounded movement: one knob step per round, in its declared direction; ≤ ``max_rounds`` rounds per lineage, more
#      only with new executed information; a lineage is never moved both up and down
#   G6 not tuned to the student: the difficulty of a variant is also confirmed by a non-student reference (``alt_check``:
#      an independent ALT family solves it); the drift monitor watches repaired tasks
#   G7 pool regularisation: KL(cell histogram with the variant ‖ the pool's start histogram) ≤ κ and entropy ≥ H_min;
#      originals are never retired by a variant
#   G8 system-level lifts go to the update rule (R1–R10, generalisation gates); nothing here changes Θ
#   G9 every decision is ledgered; a variant or repair is a new version, the previous one restorable

TaskMove = Literal["REPAIR", "RESKILL", "DESCEND", "ASCEND", "KEEP", "RESERVE", "HOLD"]
RANK = {"too_easy": 0, "in_band": 1, "too_hard": 2}


def _knob_name(knob: str) -> str:
    """A ``Knobs`` move's name in R-too-hard's vocabulary (``ComposeKnobs`` calls the undo move ``undo``)."""
    return {"undo": "undo_harder_knob"}.get(knob, knob)


class KnobMove(CrucibleModel):
    knob: str
    direction: Literal["easier", "harder"]
    params: dict[str, Any] = Field(default_factory=dict)


class Knobs(Protocol):
    async def moves(self, task: Task, direction: Literal["easier", "harder"]) -> list[KnobMove]: ...

    async def apply(self, task: Task, move: KnobMove) -> Task | None: ...


def _pol(rule: str, name: str) -> Callable[[], Any]:
    def get() -> Any:
        from crucible.synth import policy

        v = policy.param(rule, name)
        return tuple(v) if isinstance(v, list) else v

    return get


class TaskRules(ProbeRules):
    """The per-task loop's parameters; the decided ones default to the policy ledger (DESIGN item 24: R-band, R-too-hard,
    R-uncertain), so a run records the version they came from."""

    k_diag: int = 8
    looks: tuple[int, ...] = Field(default_factory=_pol("R-band", "looks"))  # acceptance (fresh) rollouts, sequential; the last is k_max
    # R-band v2 (taskloop-2): an undecided measurement escalates through these looks up to k_cap (fresh rollouts on the same
    # measurement), within task_rollout_cap rollout indices per lineage; then HOLD
    escalation_looks: tuple[int, ...] = Field(default_factory=_pol("R-band", "escalation_looks"))
    k_cap: int = Field(default_factory=_pol("R-band", "k_cap"))
    task_rollout_cap: int = Field(default_factory=_pol("R-band", "task_rollout_cap"))
    band_lo: float = Field(default_factory=_pol("R-band", "band_lo"))
    band_hi: float = Field(default_factory=_pol("R-band", "band_hi"))
    band_alpha: float = Field(default_factory=_pol("R-band", "alpha"))
    max_rounds: int = Field(default_factory=_pol("R-too-hard", "max_rounds"))
    max_refusals: int = 2
    uncertain_hold: float = Field(default_factory=_pol("R-uncertain", "uncertain_hold"))  # HOLD above this share of uncertain graded rollouts
    min_agree: int = 2  # decision_disagreement
    kappa: float = Field(default_factory=_pol("R-too-hard", "kappa"))
    h_min: float = 0.0
    require_alt: bool = True
    reskill_cap: float = Field(default_factory=_pol("R-too-hard", "reskill_cap"))  # RS5: accepted RESKILLs per round ≤ max(1, cap × the round's tasks)


class TaskDecision(CrucibleModel):
    task_id: str
    version: int
    round: int
    move: TaskMove
    why: list[str] = Field(default_factory=list)
    counted: dict[str, int] = Field(default_factory=dict)
    band: dict[str, Any] | None = None
    oracle: dict[str, Any] = Field(default_factory=dict)
    knob: KnobMove | None = None
    variant_id: str | None = None
    new_version: int | None = None
    accepted: bool = False
    refused_by: list[str] = Field(default_factory=list)
    acceptance_band: dict[str, Any] | None = None
    rollouts: dict[str, list[int]] = Field(default_factory=dict)  # diagnosis | acceptance -> rollout indices
    reskill: dict[str, Any] | None = None  # RESKILL: the old and new card, the proposal's reasons and its checks
    policy_version: str = ""  # DESIGN item 24: the policy the decision was resolved under, and the rule
    rule: str = ""
    remedies: list[dict[str, Any]] = Field(default_factory=list)  # every condition raised, its facts, the remedy and what was passed over
    lessons: dict[str, list[str]] = Field(default_factory=dict)  # lemma -> the candidates an accepted repair touched (R-lemma-lift)


class Lineage(CrucibleModel):
    root: str
    direction: Literal["up", "down"] | None = None
    members: list[str] = Field(default_factory=list)
    rounds: int = 0
    refusals: int = 0
    status: Literal["active", "kept", "reserved", "held", "rejected"] = "active"
    reskilled: bool = False  # RS6: one RESKILL attempt per lineage (accepted or refused): no ping-pong
    refused_knobs: list[str] = Field(default_factory=list)  # a knob refused once is not re-tried on fresh rollouts (no re-measuring to a pass)
    original: dict[str, Any] | None = None  # RS6: the label and version before an accepted RESKILL (rollback)


class TaskLoop(ProbeLoop):
    """The typed per-task loop (section comment above). ``probe`` is ``learn.StudentProbe`` (rollout index → seed);
    ``classify`` an ``OutcomeClassifier`` (``LearnClassifier``); ``repair`` the CEGIS repairer; ``knobs`` the declared
    difficulty knobs (``ComposeKnobs``); ``oracle`` an ``ORACLE`` with the Worlds' policy truths; ``p3_check(task)`` →
    (on_intent, reasons) (``intent.p3``); ``alt_check(task)`` → solvable by an independent ALT family (None: unknown);
    ``cell_of(task)`` → the task's cell."""

    def __init__(self, root: Path, *, probe: Any, classify: OutcomeClassifier, repair: ProbeRepairer, replay: Replayer, recheck: Recheck, knobs: Knobs,
                 ctx: FalsifyContext, oracle: Any = None, p3_check: Callable[[Task], Awaitable[tuple[bool | None, list[str]]]] | None = None,
                 alt_check: Callable[[Task], Awaitable[bool | None]] | None = None, cell_of: Callable[[Task], str] | None = None, archive: Any = None,
                 lift_lemmas: LemmaLift | None = None, rules: TaskRules | None = None, monitor: TuningMonitor | None = None,
                 derive_intent: Callable[[Task, FalsifyContext], Awaitable[Any]] | None = None, skills_of: Callable[[Task], Sequence[str]] | None = None,
                 leak_check: Callable[[Task], Awaitable[list[str]]] | None = None,
                 novel_check: Callable[[Task, str, Sequence[Task]], Awaitable[tuple[bool, list[str]]]] | None = None,
                 validity: Callable[[Task], Awaitable[tuple[str, list[Counterexample]]]] | None = None,
                 stored: Callable[[Task], Sequence[Any] | None] | None = None,
                 panel_calibrated: Callable[[Task], bool] | None = None,
                 readjudicate: Callable[[Task], Awaitable[None]] | None = None):
        self.derive_intent, self.skills_of, self.leak_check, self.novel_check = derive_intent, skills_of, leak_check, novel_check
        self.validity = validity  # P1 (item 23): (certificate status, the shown counterexamples of a rejected version)
        # R-panel on an on-hold certificate (pilot-10x10: the fact was hard-coded False, so a calibrated panel never decided
        # the suspects): ``panel_calibrated(task)`` is the measured fact; ``readjudicate(task)`` drops the stored status so
        # ``validity`` re-certifies the version with the calibrated panels
        self.panel_calibrated, self.readjudicate = panel_calibrated, readjudicate
        self.stored = stored  # prereg §8.1: rollouts measured before on this exact version (the learnability store) diagnose round 1
        self._reserved: dict[str, float] = {}  # cells of in-flight moves (concurrent steps: G7 and RS5 count them)
        self._gate = asyncio.Lock()
        self.originals: dict[str, Task] = {}  # RS6: task id -> the version before its RESKILL
        self._round_reskills: dict[int, int] = {}
        self._round_size: dict[int, int] = {}
        self.lineages: dict[str, Lineage] = {}
        self.of: dict[str, str] = {}  # task id -> lineage root
        self.decisions: list[TaskDecision] = []
        self.next_index: dict[str, int] = {}  # lineage root -> the next unused rollout index (fresh seeds)
        self.cells0: dict[str, float] = {}
        super().__init__(root, probe=probe, classify=classify, repair=repair, replay=replay, recheck=recheck, archive=archive, lift_lemmas=lift_lemmas,
                         rules=rules or TaskRules(), monitor=monitor)
        self.rules: TaskRules = self.rules if isinstance(self.rules, TaskRules) else TaskRules(**self.rules.model_dump())
        self.knobs, self.ctx, self.oracle = knobs, ctx, oracle
        self.p3_check, self.alt_check = p3_check, alt_check
        self.cell_of = cell_of or (lambda t: str(t.intent.skill))
        self.knob_calibration: list[dict[str, Any]] = []

    # -- state ------------------------------------------------------------------------------------------------------
    def _load(self) -> None:
        super()._load()
        p = self.root / "task_state.json"
        if p.exists():
            d = json.loads(p.read_text())
            self.lineages = {k: Lineage.model_validate(v) for k, v in d.get("lineages", {}).items()}
            self.of = dict(d.get("of", {}))
            self.decisions = [TaskDecision.model_validate(x) for x in d.get("decisions", [])]
            self.next_index = {k: int(v) for k, v in d.get("next_index", {}).items()}
            self.cells0 = dict(d.get("cells0", {}))
            self.originals = {k: Task.model_validate(v) for k, v in d.get("originals", {}).items()}

    def save(self) -> None:
        super().save()
        doc = {"lineages": {k: v.model_dump(mode="json") for k, v in self.lineages.items()}, "of": self.of,
               "decisions": [x.model_dump(mode="json") for x in self.decisions], "next_index": self.next_index, "cells0": self.cells0,
               "originals": {k: t.model_dump(mode="json") for k, t in self.originals.items()}}
        tmp = self.root / "task_state.json.tmp"
        tmp.write_text(json.dumps(doc, default=str))
        tmp.replace(self.root / "task_state.json")

    def add_tasks(self, tasks: Iterable[Task]) -> None:
        for t in tasks:
            if t.id not in self.tasks:
                self.tasks[t.id] = t
                self.lineages.setdefault(t.id, Lineage(root=t.id, members=[t.id]))
                self.of.setdefault(t.id, t.id)
                c = self.cell_of(t)
                self.cells0[c] = self.cells0.get(c, 0.0) + 1.0

    def pool_cells(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for tid, t in self.tasks.items():
            lin = self.lineages.get(self.of.get(tid, tid))
            if lin is not None and lin.status == "reserved":
                continue
            c = self.cell_of(t)
            out[c] = out.get(c, 0.0) + 1.0
        for c, v in self._reserved.items():
            out[c] = max(0.0, out.get(c, 0.0) + v)
        return out

    def _reserve(self, moves: Mapping[str, float], sign: float = 1.0) -> None:
        for c, v in moves.items():
            self._reserved[c] = self._reserved.get(c, 0.0) + sign * v
            if abs(self._reserved[c]) < 1e-9:
                del self._reserved[c]

    # -- rollouts and the band ----------------------------------------------------------------------------------------
    async def _fresh(self, task: Task, n: int) -> tuple[list[Any], list[int]]:
        """n rollouts at rollout indices this lineage never used (new seeds: the sample-splitting rule)."""
        root = self.of.get(task.id, task.id)
        i = self.next_index.get(root, 0)
        rs = await self.probe.rollouts(task, n, start=i)
        self.next_index[root] = i + n
        return list(rs), list(range(i, i + n))

    def _spec(self) -> Any:
        from crucible.synth.learn import BandSpec

        return BandSpec(lo=self.rules.band_lo, hi=self.rules.band_hi, alpha=self.rules.band_alpha, looks=tuple(self.rules.looks))

    def _band(self, passes: int, n: int, *, final: bool) -> Any:
        from crucible.synth.learn import decide

        return decide(passes, n, self._spec(), final=final)

    def _settled(self, res: Any) -> bool:
        from crucible.synth.learn import settled

        return settled(res, self._spec())

    async def _sequential_band(self, task: Task, start: Sequence[Any] = (), *, looks: Sequence[int] | None = None) -> tuple[Any, list[Any], list[int]]:
        """The band on rollouts drawn in looks up to k_max (``start`` counts toward the first look). It stops at the first
        settled look (R-band v2: confident and the interval inside one class); at the final look it returns the forced
        result, which the caller tests with ``_settled``."""
        rolls, idx = list(start), []
        looks = list(looks or self.rules.looks)
        res = None
        for look in looks:
            if len(rolls) < look:
                more, ix = await self._fresh(task, look - len(rolls))
                rolls += more
                idx += ix
            graded = [r for r in rolls if getattr(r, "passed", None) is not None]
            res = self._band(sum(1 for r in graded if r.passed), len(graded), final=look == looks[-1])
            if res is not None and self._settled(res):
                return res, rolls, idx
        return res, rolls, idx

    async def _placed_band(self, dec: TaskDecision, task: Task, start: Sequence[Any] = (), *, label: str, first: Any = None
                           ) -> tuple[Any, bool, list[int], str]:
        """One band measurement that may place a task (R-band v2; taskloop-2). The sequential test runs to k_max; a
        result that is not settled, or that disagrees with ``first`` (an earlier measurement of the same task: a
        RESKILL's fresh band against the diagnosis band), raises ``band_undecided`` and takes the policy's remedy:
        more fresh rollouts on THIS measurement (escalation looks up to k_cap; ``start`` and ``first``'s rollouts are never
        pooled in), else HOLD. Returns (the last result, placed?, the fresh indices drawn, the reason when not placed)."""
        rolls, idx = list(start), []
        res, more, ix = await self._sequential_band(task, rolls)
        rolls, idx = more, list(ix)
        esc = [k for k in self.rules.escalation_looks if k <= self.rules.k_cap]
        root = self.of.get(task.id, task.id)
        while True:
            ok = self._settled(res)
            disagree = bool(ok and first is not None and res.decision != first.decision)
            if ok and not disagree:
                return res, True, idx, ""
            n = len([r for r in rolls if getattr(r, "passed", None) is not None])
            nxt = next((k for k in esc if k > len(rolls)), None)
            used = self.next_index.get(root, 0)
            budget = nxt is not None and used + (nxt - len(rolls)) <= self.rules.task_rollout_cap and self._probe_budget_ok()
            what = (f"disagrees with the {first.decision} diagnosis band" if disagree else
                    f"is {'unconfident' if res is not None and not res.confident else 'straddling a band edge'}")
            rem = self._resolve(dec, task, "band_undecided", {"rollouts": n, "probe_budget_ok": bool(budget), "measurements": 1 if first is None else 2,
                                                             "disagree": disagree, "straddles": not ok, "lineage_rollouts": used, "measurement": label})
            if rem.action != "more_rollouts" or nxt is None:
                d = res.decision if res is not None else "none"
                return res, False, idx, (f"band_undecided: the {label} band ({d}, {res.passes if res is not None else 0}/{n}) {what}; "
                                         f"{rem.rule}: {rem.action}")
            res, more, ix = await self._sequential_band(task, rolls, looks=[nxt])
            rolls, idx = more, idx + list(ix)

    def _probe_budget_ok(self) -> bool:
        """Whether the student probe may still be called (a probe that reports its own cap; True otherwise: the wire's
        cap still stops a call past it)."""
        f = getattr(self.probe, "budget_ok", None)
        try:
            return bool(f()) if callable(f) else True
        except Exception:  # noqa: BLE001 - an unreadable budget is no budget
            return False

    # -- the precondition ------------------------------------------------------------------------------------------
    async def _oracle(self, task: Task, fails: Sequence[ProbeOutcome]) -> dict[str, Any]:
        from crucible.synth.falsify.base import falsify
        from crucible.synth.falsify.oracle import decision_disagreement

        out: dict[str, Any] = {"status": "unknown"}
        if self.oracle is not None:
            r = await falsify(task, self.oracle, self.ctx)
            if not r.applicable:
                out["policy"] = f"n/a: {r.reason}"
            elif r.c:
                out.update({"status": "wrong", "policy": "the World's rules require another decision",
                            "counterexamples": [Counterexample.of(task, f) for f in r.confirmed]})
                return out
            else:
                out.update({"status": "right", "policy": "matches the World's rules"})
        progs = [o.program for o in fails if o.program is not None]
        d = await decision_disagreement(task, progs, self.ctx, min_agree=self.rules.min_agree)
        if d.disputed and out["status"] == "right":  # R-oracle: the World's executed rules settle a dispute (taskloop-1 §2):
            out["overruled_dispute"] = d.fields      # runs agreeing on another value are a shared mistake, not evidence
        elif d.disputed:
            out.update({"status": "disputed", "dispute": d.fields})
        return out

    # -- one task ----------------------------------------------------------------------------------------------------
    def _record(self, dec: TaskDecision) -> TaskDecision:
        self.decisions.append(dec)
        self.ledger.append(Decision(kind="update", round=dec.round, update_id=f"{dec.task_id}@{dec.version}:{dec.move}", component="T", evaluated=True,
                                    accepted=dec.accepted, to_version=dec.new_version, failed=[RuleFailure(rule=x.split(":", 1)[0], detail=x) for x in dec.refused_by],
                                    metrics={"move": dec.move, "why": dec.why[:6], "variant": dec.variant_id, "counted": dec.counted, "reskill": dec.reskill,
                                             "policy_version": dec.policy_version, "rule": dec.rule, "remedies": dec.remedies, "lessons": dec.lessons}))
        return dec

    async def _variant_checks(self, task: Task, var: Task, dec: TaskDecision | None = None) -> list[str]:
        why = []
        if var.intent.skill != task.intent.skill or self.cell_of(var) != self.cell_of(task):
            why.append("G2: the variant changes the skill or the cell (what is taught)")
        why += await self._p3(dec, var, "variant", "G2")
        why += [f"G3: {x}" for x in await self.recheck(var)]
        if self.rules.require_alt:
            ok = await self.alt_check(var) if self.alt_check is not None else None
            if ok is not True:
                why.append("G6: no independent (non-student, non-builder) solver solves the variant" if ok is False else "G6: the variant's solvability by an independent solver is unmeasured")
        async with self._gate:  # concurrent steps: the trust region counts the moves in flight (reserved until decided)
            cells = self.pool_cells()
            c = self.cell_of(var)
            cells[c] = cells.get(c, 0.0) + 1.0
            k = kl(cells, self.cells0)
            if k > self.rules.kappa:
                why.append(f"G7: pool cell KL {k:.3f} > kappa {self.rules.kappa}")
            if entropy(cells) < self.rules.h_min:
                why.append(f"G7: pool entropy {entropy(cells):.3f} < H_min {self.rules.h_min}")
            if not why:
                self._reserve({c: 1.0})
        return why

    async def step(self, task: Task, round: int, *, diagnosis: tuple[Sequence[Any], Sequence[ProbeOutcome]] | None = None) -> TaskDecision:
        """One typed decision. ``diagnosis``: rollouts and their classified outcomes measured before (the learnability
        records of the pool), used instead of drawing new ones; their indices are marked used, so acceptance stays fresh."""
        rules = self.rules
        root = self.of.get(task.id, task.id)
        lin = self.lineages.setdefault(root, Lineage(root=root, members=[task.id]))
        dec = TaskDecision(task_id=task.id, version=task.version, round=round, move="HOLD")
        if lin.status in ("reserved", "kept", "rejected"):
            dec.move, dec.why = {"reserved": "RESERVE", "kept": "KEEP", "rejected": "HOLD"}[lin.status], [f"lineage already {lin.status}"]  # type: ignore[assignment]
            return self._record(dec)
        if lin.rounds >= rules.max_rounds:
            lin.status = "reserved"
            dec.move, dec.why = "RESERVE", [f"G5: {lin.rounds} rounds (the cap) without a lawful decision"]
            return self._record(dec)
        lin.rounds += 1
        if diagnosis is None and self.stored is not None and lin.rounds == 1 and task.id == root:
            rs = list(self.stored(task) or [])
            if rs:  # no diagnosis rollouts are bought; their indices are marked used, so acceptance stays fresh
                diagnosis = (rs, await self.classify.classify(task, rs))
                dec.why.append(f"diagnosis: {len(rs)} stored rollouts of this version")
        if diagnosis is not None:
            rolls, outs = list(diagnosis[0]), list(diagnosis[1])
            idx = [int(getattr(r, "i", j)) for j, r in enumerate(rolls)]
            self.next_index[root] = max(self.next_index.get(root, 0), max(idx, default=-1) + 1)
        else:
            rolls, idx = await self._fresh(task, rules.k_diag)
            outs = await self.classify.classify(task, rolls)
        dec.rollouts["diagnosis"] = idx
        counted = [(o, *count_outcome(o, rules)) for o in outs]
        dec.counted = dict(Counter(k for _, k, _ in counted))
        dec.why += [w for _, _, w in counted if w]
        graded = [o for o, k, _ in counted if k != "masked"]
        unc = sum(1 for o, k, _ in counted if k in ("suspect", "unattributed_fail") and o.evidence == "none")
        fails = [o for o, k, _ in counted if k in ("intended_fail", "unattributed_fail")]
        orc = await self._oracle(task, fails)
        dec.oracle = {k: v for k, v in orc.items() if k != "counterexamples"}
        unintended = [(o, k) for o, k, _ in counted if k in ("unintended_pass", "unintended_fail")]
        truth = bool(orc.get("policy")) and not str(orc.get("policy", "")).startswith("n/a")
        if orc["status"] == "wrong":  # R-oracle: repair with the ORACLE counterexample, else reject
            rem = self._resolve(dec, task, "wrong_oracle", {"repair_attempts": self.repairs.get(task.id, 0)})
            if rem.action == "repair_oracle":
                cxs = list(orc["counterexamples"])
                if self.archive is not None:
                    self.archive.add_all(cxs)
                # taskloop-1 §2: every label was graded against the wrong reference decision (a student who applied the
                # policy "failed"), so the labels are void: T2-T4 have no evidence here; acceptance is the re-certification
                # (ORACLE must now agree with the World's rules) and fresh rollouts
                void = [(o, "masked", f"rollout {o.i}: label void (graded against a wrong reference decision)") for o, _k, _w in counted]
                if void:
                    dec.why.append(f"{len(void)} diagnosis label(s) void: graded against a wrong reference decision")
                return await self._repair(task, dec, void, cxs, lin, why="ORACLE: the reference decision contradicts the World's rules",
                                          invalid=self.validity is not None)  # a wrong oracle is an invalid task (item 23): held, never reserved
            return self._reject(dec, lin, f"ORACLE: wrong reference decision; {rem.rule}: {rem.action}")
        if orc["status"] == "disputed":  # R-oracle: executed rules (already run), a calibrated panel, else a person
            rem = self._resolve(dec, task, "disputed_oracle", {"truth_available": truth, "oracle_checked": True, "panel_calibrated": False})
            d0 = orc["dispute"][0]
            dec.move, dec.why = "HOLD", [*dec.why, f"ORACLE: disputed by {d0['runs']} independent runs ({d0['address']}): {rem.rule} {rem.action}"]
            lin.status = "held"
            self.world_fixes.append({"task_id": task.id, "kind": "oracle_review", "dispute": orc["dispute"], "round": round})
            return self._record(dec)
        status, vcx = (await self.validity(task)) if self.validity is not None else ("certified", [])
        if status == "on_hold" and self.panel_calibrated is not None and self.readjudicate is not None and self.panel_calibrated(task):
            rem = self._resolve(dec, task, "open_suspects", {"executed_check_available": False, "panel_calibrated": True, "more_anchors_available": False})
            if rem.action == "panel_decide":  # R-panel: the calibrated panel decides the open suspects (a full re-certification)
                await self.readjudicate(task)
                status, vcx = await self.validity(task)
                dec.why.append(f"R-panel panel_decide: the calibrated panel re-certified the on-hold version: {status}")
        dec.oracle["certificate"] = status
        if status == "rejected":  # P1 (item 23), R-admission-fixed: repaired from its own counterexamples, else rejected; never moved or kept
            rem = self._resolve(dec, task, "confirmed_counterexample", {"cls": sorted({c.cls for c in vcx}), "repairer_available": bool(vcx),
                                                                         "repair_rounds": self.repairs.get(task.id, 0)})
            if rem.action != "repair_with_counterexamples":
                return self._reject(dec, lin, f"P1: the certificate rejects this version and no counterexample a repairer may see "
                                              f"remains ({rem.rule}: {rem.action})")
            cxs = list(vcx) + [c for o, _ in unintended if (c := probe_counterexample(task, o)) is not None]
            if self.archive is not None and cxs:
                self.archive.add_all(cxs)
            return await self._repair(task, dec, counted, cxs, lin, why=f"P1: the certificate rejects this version ({len(vcx)} shown counterexample(s))",
                                      invalid=True)
        if unintended:  # T1-T7 (R-too-hard's REPAIR rung: unintended outcomes with evidence, never a pass rate)
            cxs = [c for o, _ in unintended if (c := probe_counterexample(task, o)) is not None]
            if self.archive is not None and cxs:
                self.archive.add_all(cxs)
            return await self._repair(task, dec, counted, cxs, lin, why=f"{len(unintended)} unintended outcome(s) with evidence")
        if status not in ("certified", "provisional"):  # R-panel: open suspects wait for a calibrated panel or a person
            cal = bool(self.panel_calibrated(task)) if self.panel_calibrated is not None else False
            rem = self._resolve(dec, task, "open_suspects", {"executed_check_available": False, "panel_calibrated": cal, "more_anchors_available": False})
            lin.status = "held"
            dec.move, dec.why = "HOLD", [*dec.why, f"P1: the certificate is {status} (undecided suspects): {rem.rule} {rem.action}"]
            return self._record(dec)
        if graded and unc / len(graded) > rules.uncertain_hold:  # R-uncertain: the task's executed evidence ran (the classifier), so HOLD
            rem = self._resolve(dec, task, "mostly_uncertain_rollouts", {"task_evidence_run": True})
            lin.status = "held"  # waits for the panel or a person: re-probing it would only buy more of the same evidence
            dec.move, dec.why = "HOLD", [*dec.why, f"{unc} of {len(graded)} graded rollouts uncertain: the evidence does not decide ({rem.rule} {rem.action})"]
            return self._record(dec)
        band, placed, ix2, undecided = await self._placed_band(dec, task, rolls, label="diagnosis")
        dec.rollouts["diagnosis"] += ix2
        dec.band = band.model_dump(mode="json") if band is not None else None
        if not placed:  # R-band v2: more fresh rollouts on this measurement to k_cap within the caps, else HOLD
            lin.status = "held"  # prereg §8.1: a task the band cannot place is held, never moved (and not re-probed)
            dec.move, dec.why = "HOLD", [*dec.why, undecided]
            return self._record(dec)
        if band.decision == "in_band":  # Amendment 1: an in-band task whose evidence shows another label is re-labelled, then kept
            if await self._try_reskill(task, dec, lin, round, first=band):
                return dec
            lin.status = "kept"
            dec.move, dec.accepted = "KEEP", True
            return self._record(dec)
        direction: Literal["easier", "harder"] = "easier" if band.decision == "too_hard" else "harder"
        move: TaskMove = "DESCEND" if direction == "easier" else "ASCEND"
        want = "down" if direction == "easier" else "up"
        if lin.direction is not None and lin.direction != want:
            lin.status = "reserved"
            dec.move, dec.why = "RESERVE", [*dec.why, f"G5: the lineage moved {lin.direction}; moving {want} would oscillate"]
            return self._record(dec)
        if direction == "easier" and self.alt_check is not None and (await self.alt_check(task)) is False:
            dec.why.append("too hard, and no independent solver solves the original either")
        knobs = [k for k in await self.knobs.moves(task, direction) if k.knob not in lin.refused_knobs]
        fam = self._family_knobs(task.domain, direction)
        if fam is not None:  # R-too-hard family_knobs (taskloop-2): a knob the policy records inapplicable for this family is never tried
            off = [k.knob for k in knobs if _knob_name(k.knob) not in fam]
            knobs = [k for k in knobs if _knob_name(k.knob) in fam]
            if off:
                dec.why.append(f"R-too-hard family_knobs: {', '.join(off)} inapplicable for {task.domain}")
        kname = _knob_name(knobs[0].knob) if knobs else None
        kind = "band_too_hard" if direction == "easier" else "band_too_easy"
        facts = {"truth_available": truth, "oracle_checked": True, "unintended_with_evidence": 0, "reskill_attempted": lin.reskilled,
                 "deriver_available": self.derive_intent is not None, ("easier_knob" if direction == "easier" else "harder_knob"): kname}
        rem = self._resolve(dec, task, kind, facts)  # R-too-hard: RESKILL, then DESCEND / ASCEND, then RESERVE
        if rem.action == "reskill":
            if await self._try_reskill(task, dec, lin, round, first=band):
                return dec
            rem = self._resolve(dec, task, kind, {**facts, "reskill_attempted": True})
        if rem.action not in ("descend", "ascend"):
            lin.status = "reserved"
            dec.move, dec.why = "RESERVE", [*dec.why, f"no lawful move makes this task {direction} ({rem.rule}: {rem.action}; knob {kname})"]
            return self._record(dec)
        dec.move = move
        var = None
        for k in knobs:  # the first declared knob that applies to this task's World (taskloop-1 §2: finance_ap's SQL World takes none)
            if _knob_name(k.knob) not in (self._lawful_knobs(direction) or {k.knob}):
                continue
            dec.knob = k
            var = await self.knobs.apply(task, k)
            if var is not None:
                break
        if var is None:
            lin.status = "reserved"
            dec.move, dec.why = "RESERVE", [*dec.why, f"no declared {direction} knob applies to this task's World ({[k.knob for k in knobs]})"]
            return self._record(dec)
        knobs = [dec.knob] if dec.knob is not None else knobs
        why = await self._variant_checks(task, var, dec)
        if why:
            lin.refused_knobs.append(dec.knob.knob if dec.knob is not None else "?")
            return self._refuse(dec, lin, why)
        self.of.setdefault(var.id, root)  # its acceptance rollouts come from the lineage's unused indices
        try:  # R-band v2: the variant's fresh band must be settled (escalated to k_cap if not), never pooled with the diagnosis
            vband, vplaced, ix3, vund = await self._placed_band(dec, var, label="variant")
        finally:
            self._reserve({self.cell_of(var): 1.0}, -1.0)  # decided below: accepted variants count as pool tasks
        dec.rollouts["acceptance"] = ix3
        dec.acceptance_band = vband.model_dump(mode="json") if vband is not None else None
        moved = vplaced and ((RANK[vband.decision] < RANK[band.decision]) if direction == "easier" else (RANK[vband.decision] > RANK[band.decision]))
        self.knob_calibration.append({"task": task.id, "knob": knobs[0].knob, "predicted": direction, "before": band.decision,
                                      "after": vband.decision if vplaced else None, "moved": moved})
        if not moved:
            lin.refused_knobs.append(knobs[0].knob)
            seen = vband.decision if vplaced else f"undecided ({vund})"
            return self._refuse(dec, lin, [f"G4: on fresh rollouts the variant is {seen}, not {direction} than {band.decision}"])
        dec.accepted, dec.variant_id = True, var.id
        lin.direction = want
        lin.members.append(var.id)
        self.tasks[var.id] = var  # the original stays: a harder (or easier) rung of the same lineage
        self.of[var.id] = root
        return self._record(dec)

    # -- RESKILL (lead, 2026-10-30; meta/reskill.py) ---------------------------------------------------------------
    async def _try_reskill(self, task: Task, dec: TaskDecision, lin: Lineage, round: int, *, first: Any = None) -> bool:
        """Re-label a valid task with the skill its evidence shows. RS1 an existing skill of the domain; RS2 the
        derivation reads only the task's evidence (never the student) and its derive values are values π* writes or
        answers that the instruction does not state; RS3 P3 under the new card, LEAK with the new derive set, full
        re-certification under the monotone archive; RS4 novelty in the new cell, and the band on FRESH rollouts places
        the task (in band: KEEP; out of band: RESERVE under its honest label); RS5 the pool's cell KL and entropy with the
        task moved, and a per-round cap; RS6 one attempt per lineage, the old card and version kept for rollback.
        R-band v2 (taskloop-2): only the card changes, so the student's pass rate cannot; the fresh band places the task
        only when it is settled AND agrees with ``first`` (the diagnosis band). Otherwise ``band_undecided``: more fresh
        rollouts on the fresh measurement (never pooled with the diagnosis), then HOLD: the label is kept (it rests on
        the task's evidence, not the band) and the lineage is held.
        Returns True when the RESKILL was accepted (the decision is recorded)."""
        if self.derive_intent is None or lin.reskilled:
            return False
        from crucible.synth.meta.reskill import check_proposal, derivation_evidence

        try:
            prop = await self.derive_intent(task, self.ctx)
        except Exception as e:  # noqa: BLE001 - taskloop-1 §2: a deriver whose every answer fails RS1/RS2 is no proposal, not a crash
            lin.reskilled = True  # RS6: the one attempt is spent
            dec.why.append(f"RESKILL refused: the deriver gave no lawful proposal ({type(e).__name__}: {str(e)[:160]})")
            dec.reskill = {"refused_by": [f"RS1/RS2: {str(e)[:300]}"]}
            return False
        if prop is None:
            return False
        old = task.intent
        same = (prop.skill == old.skill and sorted(cjson(x) for x in prop.derive) == sorted(cjson(x) for x in old.derive)
                and sorted(cjson(x) for x in prop.given) == sorted(cjson(x) for x in old.given))
        from crucible.synth.program import text_spans

        stated = [v for v in old.derive if text_spans(v, task.instruction)]  # the old card calls a stated value derived: wrong by RS2
        if same or (self.cell_of(task.next_version(intent=prop.card())) == self.cell_of(task) and not stated):
            dec.why.append("RESKILL: the derivation confirms the label (the same skill cell)")
            return False
        lin.reskilled = True
        why: list[str] = []
        ev = await derivation_evidence(task, self.ctx)
        why += check_proposal(prop, list(self.skills_of(task)) if self.skills_of is not None else [], ev)
        size = self._round_size.get(round, 1)
        cap = max(1, math.floor(self.rules.reskill_cap * size))
        slot = False
        async with self._gate:  # concurrent steps reserve their slot of the cap before the band rollouts
            if self._round_reskills.get(round, 0) >= cap:
                why.append(f"RS5: this round's RESKILL cap ({cap} of {size} tasks) is used")
            elif not why:
                self._round_reskills[round] = self._round_reskills.get(round, 0) + 1
                slot = True
        new = task.next_version(intent=prop.card(notes=f"RESKILL from {old.skill!r} v{task.version}: {prop.why}"[:300]))
        oc, nc = self.cell_of(task), self.cell_of(new)
        if not why:
            why += await self._p3(dec, new, "reskill", "RS3")
            if self.leak_check is not None:
                why += [f"RS3: LEAK under the new derive set: {x}" for x in await self.leak_check(new)]
            why += [f"RS3: {x}" for x in await self.recheck(new)]
        moved = {oc: -1.0, nc: 1.0} if oc != nc else {}
        held = False
        if not why:
            async with self._gate:
                cells = self.pool_cells()
                cells[oc] = max(0.0, cells.get(oc, 0.0) - 1.0)
                cells[nc] = cells.get(nc, 0.0) + 1.0
                k = kl(cells, self.cells0)
                if k > self.rules.kappa:
                    why.append(f"RS5: pool cell KL {k:.3f} > kappa {self.rules.kappa} with the task moved to {nc!r}")
                if entropy(cells) < self.rules.h_min:
                    why.append(f"RS5: pool entropy {entropy(cells):.3f} < H_min {self.rules.h_min}")
                if not why and moved:
                    self._reserve(moved)
                    held = True
        if not why and self.novel_check is not None:
            pool = [t for tid, t in self.tasks.items() if tid != task.id and self.cell_of(t) == nc]
            ok, rs = await self.novel_check(new, nc, pool)
            if not ok:
                why.append(f"RS4: not novel in {nc!r}: {'; '.join(rs)[:200]}")
        vband, placed, undecided = None, False, ""
        if not why:
            self.of.setdefault(new.id, self.of.get(task.id, task.id))
            vband, placed, ix, undecided = await self._placed_band(dec, new, label="reskill", first=first)
            dec.rollouts["acceptance"] = ix
            dec.acceptance_band = vband.model_dump(mode="json") if vband is not None else None
        dec.reskill = {"from": old.model_dump(mode="json"), "to": prop.model_dump(mode="json"), "cell": [oc, nc], "refused_by": why,
                       "placed": placed, **({"band_undecided": undecided} if undecided else {})}
        if held:
            self._reserve(moved, -1.0)  # decided: the accepted move is in ``self.tasks`` below
        if why:
            if slot:
                self._round_reskills[round] -= 1  # the reserved slot of the cap is released
            dec.why.append(f"RESKILL refused: {why[0]}")
            return False
        assert vband is not None
        self.originals[task.id] = task
        self.tasks[task.id] = new
        lin.original = {"skill": old.skill, "version": task.version, "card": old.model_dump(mode="json")}
        dec.move, dec.accepted, dec.new_version = "RESKILL", True, new.version
        if not placed:  # R-band v2: never placed by the second measurement alone; the label stands, the lineage waits
            lin.status = "held"
            dec.why.append(f"RESKILL {old.skill!r} → {prop.skill!r}; {undecided}: held under its honest label")
        else:
            lin.status = "kept" if vband.decision == "in_band" else "reserved"
            dec.why.append(f"RESKILL {old.skill!r} → {prop.skill!r}; fresh band {vband.decision} agrees with the diagnosis band: "
                           f"{'kept' if lin.status == 'kept' else 'reserved under its honest label'}")
        self._record(dec)
        return True

    def rollback_reskill(self, task_id: str, round: int = 0) -> Task:
        """RS6: restore the version before an accepted RESKILL (the lineage keeps ``reskilled``: no second attempt)."""
        old = self.originals.pop(task_id)
        lin = self.lineages[self.of.get(task_id, task_id)]
        self.tasks[task_id] = old
        lin.original, lin.status = None, "active"
        self._record(TaskDecision(task_id=task_id, version=old.version, round=round, move="HOLD", accepted=True, why=["RESKILL rolled back"]))
        return old

    @staticmethod
    def _lawful_knobs(direction: str) -> set[str]:
        from crucible.synth import policy

        return set(policy.param("R-too-hard", "easier_knobs" if direction == "easier" else "harder_knobs"))

    @staticmethod
    def _family_knobs(domain: str, direction: str) -> set[str] | None:
        """R-too-hard ``family_knobs`` (taskloop-2): the knobs lawful on this World family in this direction, or None when the
        policy records nothing for the family (every declared knob may be tried)."""
        from crucible.synth import policy

        fam = (policy.current().rule("R-too-hard").params.get("family_knobs") or {}).get(domain)
        return None if fam is None else set(fam.get(direction) or [])

    def _resolve(self, dec: TaskDecision, task: Task, kind: str, facts: dict[str, Any]) -> Any:
        """Raise a blocking condition with its measured facts and take the policy's lawful remedy (DESIGN item 24,
        R-oracle, R-admission-fixed, R-panel, R-uncertain, R-band, R-too-hard); the version, the rule and what was passed
        over are recorded on the decision."""
        from crucible.synth.policy import Condition, resolve

        rem = resolve(Condition(kind=kind, subject=f"{task.id}@v{task.version}", facts=facts, raised_by="meta.rsi.TaskLoop.step"))
        dec.policy_version, dec.rule = rem.policy_version, rem.rule
        dec.remedies.append({"condition": kind, "facts": facts, "action": rem.action, "rule": rem.rule, "policy_version": rem.policy_version,
                             "considered": rem.considered[:8]})
        return rem

    async def _p3(self, dec: TaskDecision | None, cand: Task, edit: str, tag: str) -> list[str]:
        """P3 on a candidate edit (G2 for repairs and variants, RS3 for a RESKILL). Off intent refuses the edit. An
        UNMEASURED P3 (no valid EVID, SWAP or TWIN trial) is never on intent and never a silent refusal: it raises
        ``p3_unmeasured`` (R-p3) and takes the policy's remedy. ``p3_check`` has already run every executed P3 probe the
        World supports (EVID through the SQL evidence channel included, when the caller wired it), so the remedy is HOLD:
        the edit is not accepted and the condition is recorded on the decision."""
        if self.p3_check is None:
            return []
        on, rs = await self.p3_check(cand)
        if on is True:
            return []
        if on is False:
            return [f"{tag}: P3 probes: {'; '.join(rs)[:200] or 'not on intent'}"]
        if dec is None:
            return [f"{tag}: p3_unmeasured ({'; '.join(rs)[:160]})"]
        rem = self._resolve(dec, cand, "p3_unmeasured", {"edit": edit, "executed_check_available": False})
        return [f"{tag}: p3_unmeasured ({'; '.join(rs)[:160] or 'no valid EVID, SWAP or TWIN trial'}): {rem.rule} {rem.action}"]

    def _reject(self, dec: TaskDecision, lin: Lineage, why: str) -> TaskDecision:
        lin.status = "rejected"  # invalid and not repairable under the policy: kept with its reasons, never moved, kept or deleted
        dec.move, dec.why = "HOLD", [*dec.why, why]
        return self._record(dec)

    def _refuse(self, dec: TaskDecision, lin: Lineage, why: list[str], *, invalid: bool = False) -> TaskDecision:
        dec.refused_by = why
        lin.refusals += 1
        if invalid:  # one repair attempt (the repairer retries with feedback inside it); refused or declined, the task waits
            lin.status = "held"  # an invalid task is never reserved as a good task: it waits, unrepaired, for a person
            dec.why.append("P1: the task stays invalid: held (not repaired, never moved or kept)")
        elif lin.refusals >= self.rules.max_refusals:
            lin.status = "reserved"
            dec.why.append(f"{lin.refusals} refused edits: reserved (kept for stronger checkpoints)")
        return self._record(dec)

    async def _repair(self, task: Task, dec: TaskDecision, counted: Sequence[tuple[ProbeOutcome, str, str | None]], cxs: list[Counterexample], lin: Lineage,
                      *, why: str, invalid: bool = False) -> TaskDecision:
        dec.move = "REPAIR"
        dec.why.append(why)
        n = self.repairs.get(task.id, 0)
        executed = any(o.evidence == "executed" for o, k, _ in counted if k.startswith("unintended")) or why.startswith(("ORACLE", "P1"))
        if n >= self.rules.max_repairs or (not executed and self.judge_streak.get(task.id, 0) >= self.rules.judge_cap):
            lin.status = "held" if invalid else "reserved"
            return self._refuse(dec, lin, [f"T6: {n} repairs, judge streak {self.judge_streak.get(task.id, 0)}"], invalid=invalid)
        if not cxs:
            return self._refuse(dec, lin, ["G1: no counterexample a repairer may see (e.g. only held-out mutants, or ALT found no pass)"], invalid=invalid)
        new = await self.repair(task, list(cxs))
        if new is None:
            return self._refuse(dec, lin, ["the repairer declined"], invalid=invalid)
        pairs = [(o, k) for o, k, _ in counted]
        why2 = await self.decide(task, new, pairs, cxs)
        if self.cell_of(new) != self.cell_of(task) or new.intent.skill != task.intent.skill:
            why2.append("G2: the repair changes the skill or the cell")
        why2 += await self._p3(dec, new, "repair", "G2")
        if why2:
            return self._refuse(dec, lin, why2, invalid=invalid)
        # G4: acceptance on FRESH rollouts: no unintended outcome of the same mechanism recurs
        fresh, ix = await self._fresh(new, self.rules.looks[0])
        dec.rollouts["acceptance"] = ix
        fouts = await self.classify.classify(new, fresh)
        mechs = {o.mechanism for o, k, _ in counted if k.startswith("unintended")}
        again = [o for o in fouts if count_outcome(o, self.rules)[0].startswith("unintended") and o.mechanism in mechs]
        if again:
            return self._refuse(dec, lin, [f"G4: on fresh rollouts {len(again)} unintended outcome(s) of the same mechanism recur"], invalid=invalid)
        dec.accepted, dec.new_version = True, new.version
        self.tasks[task.id] = new
        self.repairs[task.id] = n + 1
        self.judge_streak[task.id] = 0 if executed else self.judge_streak.get(task.id, 0) + 1
        nb = sum(1 for o, k, _ in counted if k != "masked")
        graded = [r for r in fresh if getattr(r, "passed", None) is not None]
        if nb and graded:
            self.monitor.add(n_before=nb, passes_before=sum(1 for o, k, _ in counted if k != "masked" and o.passed),
                             unintended_passes=sum(1 for _, k, _ in counted if k == "unintended_pass"),
                             unintended_fails=sum(1 for _, k, _ in counted if k == "unintended_fail"), n_after=len(graded),
                             passes_after=sum(1 for r in graded if r.passed))
        if self.lift_lemmas is not None:  # G8 / R-lemma-lift: the pair is recorded; a recurring class becomes a gated candidate
            lem = await self.lift_lemmas(task, new, list(cxs))
            self.lemma_candidates += lem
            dec.lessons["lemma"] = [f"{getattr(x, 'id', x)}:{getattr(x, 'key', '')}:{getattr(x, 'status', '')}" for x in lem]
        return self._record(dec)

    async def run_typed_round(self, round: int, *, ids: Sequence[str] | None = None, concurrency: int = 1,
                              on_error: Callable[[str, BaseException], None] | None = None) -> list[TaskDecision]:
        """One decision per active lineage member. ``concurrency`` > 1 runs steps concurrently (the pool gates G7 and
        RS5 count the moves in flight); held lineages wait for a person and are skipped like kept and reserved ones.
        ``on_error`` records a step that raised (that task gets no decision this round) instead of stopping the round."""
        todo = [i for i in (ids or sorted(self.tasks))
                if ids is not None or (lin := self.lineages.get(self.of.get(i, i))) is None or lin.status not in ("reserved", "kept", "held", "rejected")]
        self._round_size[round] = len(todo)
        sem = asyncio.Semaphore(max(1, concurrency))
        res: dict[str, TaskDecision] = {}

        async def one(i: str) -> None:
            async with sem:
                try:
                    res[i] = await self.step(self.tasks[i], round)
                except Exception as e:
                    if on_error is None:
                        raise
                    on_error(i, e)
                self.save()

        await asyncio.gather(*(one(i) for i in todo))
        return [res[i] for i in todo if i in res]

    def typed_summary(self) -> dict[str, Any]:
        return {"decisions": dict(Counter(d.move for d in self.decisions)), "accepted": dict(Counter(d.move for d in self.decisions if d.accepted)),
                "refused_by": dict(Counter(r.split(":", 1)[0] for d in self.decisions for r in d.refused_by)),
                "lineages": dict(Counter(lin.status for lin in self.lineages.values())), "drift": self.monitor.alarm(),
                "knob_calibration": {"n": len(self.knob_calibration), "moved": sum(1 for k in self.knob_calibration if k["moved"])}}


class ComposeKnobs:
    """``Knobs`` over ``synth.compose``: ASCEND applies the first declared harder knob that moves the behaviour
    signature (distractors, paginate; actor and derive need a writer); DESCEND steps a lineage back down: a variant made
    by a recorded harder knob has its parent rung as the easier move (the parent must be known to ``rungs``). A plain
    task has no declared easier knob: every compose knob is "harder" (``compose.KNOB_PREDICTED``)."""

    def __init__(self, ctx: FalsifyContext, *, writer: Any = None, rungs: Mapping[str, Task] | None = None, kinds: Sequence[str] = ("distractors", "paginate")):
        self.ctx, self.writer, self.rungs, self.kinds = ctx, writer, dict(rungs or {}), list(kinds)
        self.parent: dict[str, str] = {}

    async def moves(self, task: Task, direction: Literal["easier", "harder"]) -> list[KnobMove]:
        """Easier moves keep the skill (lead, 2026-10-30): undo a recorded harder knob, then the structural easier knobs
        a variant World makes cheap: ``fewer_distractors`` (drop one distractor clone) and ``unpaginate`` (one read tool
        returns its whole list). derive → given is never an easier move: it changes the skill (a RESKILL to
        lookup-and-apply)."""
        from crucible.synth.compose import split_world

        if direction == "easier":
            out = []
            p = self.parent.get(task.id)
            if p in self.rungs:
                out.append(KnobMove(knob="undo", direction="easier", params={"to": p}))
            spec = split_world(task.world)
            if any(x.op == "clone" for x in spec.perturb):
                out.append(KnobMove(knob="fewer_distractors", direction="easier"))
            if spec.paginate:
                out.append(KnobMove(knob="unpaginate", direction="easier", params={"tool": min(spec.paginate)}))
            return out
        return [KnobMove(knob=k, direction="harder") for k in self.kinds]

    async def _easier_world(self, task: Task, move: KnobMove) -> Task | None:
        """A variant World one structural step easier; kept only if π* still runs and passes the task's contract there."""
        from crucible.synth.compose import split_world

        spec = split_world(task.world)
        if move.knob == "fewer_distractors":
            clones = [i for i, x in enumerate(spec.perturb) if x.op == "clone"]
            if not clones:
                return None
            spec = spec.model_copy(update={"perturb": [x for i, x in enumerate(spec.perturb) if i != clones[-1]]}, deep=True)
        else:
            tool = str(move.params.get("tool"))
            if tool not in spec.paginate:
                return None
            spec = spec.model_copy(update={"paginate": {k: v for k, v in spec.paginate.items() if k != tool}}, deep=True)
        t = task.model_copy(update={"id": f"{task.id}~{move.knob}", "version": 1, "world": spec.ref()}, deep=True)
        try:
            r = await self.ctx.run(t, t.oracle)
        except Exception:  # noqa: BLE001 - a World that refuses the variant makes the knob inapplicable
            return None
        if r.error is not None or not self.ctx.grade(r, t.contract).passed:
            return None
        self.parent[t.id] = task.id
        self.rungs[task.id] = task
        return t

    async def apply(self, task: Task, move: KnobMove) -> Task | None:
        from crucible.synth.compose import KnobNotApplicable, KnobSpec, apply_knob, single

        if move.knob == "undo":
            return self.rungs.get(str(move.params.get("to")))
        if move.knob in ("fewer_distractors", "unpaginate"):
            return await self._easier_world(task, move)
        try:
            v = await apply_knob(KnobSpec(kind=move.knob, params=move.params), await single(task, self.ctx), ctx=self.ctx, writer=self.writer)
        except KnobNotApplicable:
            return None
        if not v.record.kept:
            return None
        t = v.composed.task
        self.parent[t.id] = task.id
        self.rungs[task.id] = task
        return t
