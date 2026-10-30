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

import json
import math
import random
from collections import Counter
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, content_hash
from crucible.synth.cegis import CegisReport
from crucible.synth.falsify import cheap
from crucible.synth.falsify.base import BaseFalsifier, Counterexample, FalsifyContext, run_classes
from crucible.synth.meta.abstract import Cluster, PlantVerdict, miss_clusters, natural_clusters, random_operator
from crucible.synth.meta.library import Library, OperatorSpec, accept_planter, birth_gate
from crucible.synth.meta.plant import FEATURES, Plant, Planter, PlantOp, base_planter, feature_matrix, held_out_planter, mmd, plant_all
from crucible.synth.meta.recall import PlantSet, RecallMatrix, capture_recapture, measure, residual, set_digest
from crucible.synth.meta.update import Decision, Evidence, Ledger, Measurement, RuleFailure, ThetaStore, Update, UpdateConfig, UpdateEngine, View, leakage_critic
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
    """The dev plants by the planter version that produced them (``P@<v>``), and the dev anchors."""

    def __init__(self, anchors: Sequence[Task]):
        self.plants: dict[str, Plant] = {}
        self.by_planter: dict[str, list[str]] = {}
        self.anchors = list(anchors)

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
        return q

    def domains(self) -> list[str]:
        return sorted({p.domain for p in self.plants.values()})


class Sampler:
    """R1's fresh views (module docstring): seeded and stratified by planter version and domain."""

    def __init__(self, pool: Pool, *, frac: float = 0.6, seed: int = 0):
        self.pool, self.frac, self.seed, self.n = pool, frac, seed, 0

    def draw(self, tag: str, plants: Sequence[str] | None = None, frac: float | None = None) -> list[str]:
        rng = random.Random(f"view/{self.seed}/{tag}")
        out: list[str] = []
        for _k, ids in sorted(self.pool.by_planter.items()):
            # strata: domain only. RSI-1 stratified by (domain, defect) too: with deterministic falsifiers every such view
            # had the same J, so the measured noise floor was δ = 0 and R2 degenerated to "any positive LCB"
            by: dict[str, list[str]] = {}
            for i in ids if plants is None else [x for x in ids if x in set(plants)]:
                by.setdefault(self.pool.plants[i].domain, []).append(i)
            for _d, xs in sorted(by.items()):
                k = max(1, round((frac or self.frac) * len(xs)))
                out += rng.sample(sorted(xs), min(k, len(xs)))
        return out

    async def __call__(self, round: int, purpose: str, used: frozenset[str]) -> View:
        for _ in range(64):
            self.n += 1
            ids = self.draw(f"{round}/{purpose}/{self.n}")
            h = content_hash({"plants": sorted(ids), "anchors": sorted(a.id for a in self.pool.anchors)}, 32)
            if h not in used:
                break
        return View(hash=h, round=round, domains=self.pool.domains(), planters=sorted(self.pool.by_planter), seed=self.n,
                    payload={"plants": ids, "anchors": [a.id for a in self.pool.anchors]})


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
    neighbourhood enlarged until its probe spend reaches the candidate's."""

    def __init__(self, pool: Pool, ctx: FalsifyContext, *, base: Callable[[], list[BaseFalsifier]] = cheap):
        self.pool, self.ctx, self.base = pool, ctx, base
        self._big: dict[int, FalsifyContext] = {}

    def _ctx_for(self, max_mutants: int) -> FalsifyContext:
        if max_mutants not in self._big:
            c = FalsifyContext(max_mutants=max_mutants)
            c._worlds = self.ctx._worlds  # share the Worlds (one per WorldRef), not the class cache
            self._big[max_mutants] = c
        return self._big[max_mutants]

    async def measure(self, theta_f: Any, plant_ids: Sequence[str], anchor_ids: Sequence[str], ctx: FalsifyContext) -> Measurement:
        fs = population(theta_f, self.base)
        plants = [self.pool.plants[i] for i in plant_ids]
        vs, probes = await verdicts(plants, fs, ctx)
        units = [1.0 if vs[p.id].hit else 0.0 for p in plants]
        by_d: dict[str, list[float]] = {}
        for p, u in zip(plants, units, strict=True):
            by_d.setdefault(p.defect, []).append(u)
        anchors = [a for a in self.pool.anchors if a.id in set(anchor_ids)]
        alarms = 0
        for a in anchors:
            res = await run_classes(a, fs, ctx, cache=True)
            probes += sum(len(r.findings) for r in res)
            alarms += any(r.applicable and r.c for r in res)
        ops = [f.name for f in fs if f.name.startswith("OP:")]
        gain = {}
        for op in ops:
            only = sum(1 for p in plants if vs[p.id].hit == [op])
            gain[op] = only / len(plants) if plants else 0.0
        n_units = len(plants) + len(anchors)
        return Measurement(j_units=units, unit_domains=[p.domain for p in plants], fr={d: sum(x) / len(x) for d, x in by_d.items()},
                           fa=alarms / len(anchors) if anchors else 0.0, cost=PROBE_USD * probes / max(1, n_units), item_gain=gain)

    async def __call__(self, theta: Any, view: View, budget: float | None) -> Measurement:
        f = theta.data.get("F")
        m = await self.measure(f, view.payload["plants"], view.payload["anchors"], self.ctx)
        if budget is None or budget <= m.cost:
            return m
        mm = min(4000, math.ceil(self.ctx.max_mutants * budget / max(m.cost, 1e-12)))
        return await self.measure(f, view.payload["plants"], view.payload["anchors"], self._ctx_for(mm))


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


def swarm_proposers(client: Any, *, planter_client: Any = None) -> Proposers:
    """The loop's proposers on one ``roles.RoleClient`` (the abstractor's family; R6: never a builder's)."""
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

        out = await ask(client, OPERATOR_ABSTRACTOR, tag=f"abstract:r{round}:{tag}", check=check, retries=1, clusters=clusters, misses=misses,
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

        out = await ask(planter_client or client, PLANTER_PROPOSER, tag=f"plant:r{round}", check=check, retries=1,
                        operators=sorted(miss_rates), miss_rates=miss_rates, natural_mix=natural_mix, count=k, history=history or None)
        allowed = {"why", "factor", "phrase", "template", "suffix", "field"}
        res = [PlantOp(name=d["name"], variant=d.get("variant", "base"), version=100 + round,
                       params={kk: v for kk, v in (d.get("params") or {}).items() if kk in allowed and isinstance(v, str | int | float)}) for d in out.planters[:k]]
        return res, {"notes": out.notes[:500]}

    async def guidance(*, guidance: list[str], gate_outcomes: dict[str, int]) -> tuple[list[str], dict[str, Any]]:
        out = await ask(client, GUIDANCE_EDITOR, tag="abstract:guidance", retries=1, guidance="\n".join(f"{i}. {g}" for i, g in enumerate(guidance)),
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
    lemmas: list[dict[str, Any]] = Field(default_factory=list)
    strategies: list[str] = Field(default_factory=list)
    info: dict[str, Any] = Field(default_factory=dict)


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


class Loop:
    """E2/E3's driver: one or more arms over a shared locked set (module docstring)."""

    def __init__(self, root: Path, pool: Pool, locked: PlantSet, locked_anchors: Sequence[Task], natural: Sequence[Counterexample], *,
                 ctx: FalsifyContext, report_ctx: FalsifyContext, config: EvolveConfig, dev_sources: Sequence[Task] = (), builders: Sequence[str] = (),
                 natural_sources: Mapping[str, Task] | None = None, spend: Callable[[], float] | None = None,
                 reference: Sequence[dict[str, float]] | None = None):
        self.root, self.base_pool, self.locked, self.locked_anchors = Path(root), pool, locked, list(locked_anchors)
        self.dev_sources = list(dev_sources)
        self.planter_history: list[dict[str, Any]] = []
        self.natural = list(natural)
        self.ctx, self.report_ctx, self.cfg = ctx, report_ctx, config
        self.builders = list(builders)
        self.natural_sources = dict(natural_sources or {})
        self.spend = spend or (lambda: 0.0)
        self.arms: list[Arm] = []
        # the natural-likeness reference: given (the independently found real defects), else the archive's natural
        # counterexamples (falsifier-found: a weaker reference, used only when no independent set exists)
        ref = list(reference) if reference else [natural_features(c) for c in self.natural]
        self.reference_kind = "independent real defects" if reference else "archive (falsifier-found)"
        self.reference = ref
        base = [p.features for p in pool.plants.values()]
        self.mu = (self.cfg.mu_margin * mmd(feature_matrix(base), feature_matrix(ref))) if base and ref else 0.0

    def arm(self, name: str, propose_ops: ProposeOps, *, family: str = "") -> Arm:
        d = self.root / name
        store = ThetaStore(d / "theta")
        if store.head("F") is None:
            store.init("F", Library().model_dump(mode="json"))
            store.init("P", {"planters": [base_planter().model_dump(mode="json")]})
            store.init("S", {"version": 1, "lemmas": [], "retired": []})
            store.init("A", {"guidance": list(DEFAULT_GUIDANCE)})
        pool = self.base_pool.copy()
        pool.by_planter = {f"P@{store.head('P')}": [i for ids in pool.by_planter.values() for i in ids]}
        sampler = Sampler(pool, frac=self.cfg.view_frac, seed=int(content_hash({"arm": name, "seed": self.cfg.seed}, 8), 16) % 10_000)
        ev = FEvaluator(pool, self.ctx)
        eng = UpdateEngine(store, ev, sampler, config=self.cfg.update, ledger=Ledger(d / "ledger.jsonl"))
        a = Arm(name=name, store=store, engine=eng, pool=pool, sampler=sampler, evaluator=ev, propose_ops=propose_ops, family=family)
        self.arms.append(a)
        return a

    # -- per round ---------------------------------------------------------------------------------------------------
    def evidence_plants(self, arm: Arm) -> list[Plant]:
        doms = self.cfg.evidence_domains
        return [p for p in arm.pool.plants.values() if doms is None or p.domain in doms]

    async def head_verdicts(self, arm: Arm, plants: Sequence[Plant]) -> dict[str, PlantVerdict]:
        vs, _ = await verdicts(plants, population(arm.store.get("F")), self.ctx)
        return vs

    def _natural(self) -> list[Counterexample]:
        doms = self.cfg.evidence_domains
        return [c for c in self.natural if doms is None or c.domain in doms]

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

    async def try_candidate(self, arm: Arm, spec: OperatorSpec, cites: list[str], origin: str, clusters: Sequence[Cluster], misses: Sequence[Plant],
                            r: int, rng: random.Random) -> Candidate:
        cand = Candidate(op=spec.cls_name, origin=origin, cites=list(cites), spec=spec.model_dump(mode="json"))
        lib = Library.model_validate(arm.store.get("F"))
        if spec.cls_name in lib.ops:
            cand.refused_by = ["duplicate"]
            return cand
        sources, dom = self._sources(cites, clusters, arm, rng, misses)
        others = [i for i in arm.sampler.draw(f"gate/{r}/{spec.cls_name}", frac=1.0) if arm.pool.plants[i].domain != dom]
        fresh = [arm.pool.plants[i] for i in rng.sample(others, min(len(others), self.cfg.gate_fresh))]
        try:
            g = await birth_gate(spec, sources=sources, fresh=fresh, anchors=arm.pool.anchors, ctx=self.ctx, cost_bar=self.cfg.cost_bar, early_stop=True)
        except Exception as e:  # noqa: BLE001 - an operator that crashes on a task is refused, never fatal
            cand.refused_by = [f"gate:error:{type(e).__name__}"]
            cand.gate = {"error": str(e)[:300]}
            return cand
        cand.gate = {"checks": g.checks, "transfer_hits": len(g.detail.get("transfer_hits", [])), "false_alarms": len(g.detail.get("false_alarms", [])),
                     "reproduced": g.detail.get("reproduced"), "probes_per_task": round(g.detail.get("probes_per_task", 0.0), 2), "evidence_domain": dom}
        if not g.passed:
            cand.refused_by = [f"gate:{k}" for k, v in g.checks.items() if not v]
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

    async def planter_round(self, arm: Arm, r: int, proposers: Proposers, row: RoundRow) -> None:
        if proposers.planters is None:
            return
        vs = await self.head_verdicts(arm, list(arm.pool.plants.values()))
        rates: dict[str, list[int]] = {}
        for p in arm.pool.plants.values():
            rates.setdefault(f"{p.op.name}:{p.op.variant}", []).append(0 if vs[p.id].hit else 1)
        miss_rates = {k: round(sum(v) / len(v), 3) for k, v in sorted(rates.items())}
        mix = dict(Counter(c.defect or "?" for c in self._natural()).most_common())
        try:
            ops, info = await proposers.planters(miss_rates=miss_rates, natural_mix=mix, k=self.cfg.planter_candidates, round=r,
                                                 history=self.planter_history[-12:])
        except Exception as e:  # noqa: BLE001 - a failed proposal is recorded
            row.planter_decisions.append({"error": f"{type(e).__name__}: {str(e)[:300]}"})
            return
        held = {(o.name, o.variant) for o in held_out_planter().ops} - {(o.name, o.variant) for o in base_planter().ops}
        tasks = list(self.dev_sources)
        fs = population(arm.store.get("F"))
        for i, op in enumerate(ops[: self.cfg.planter_candidates]):
            if (op.name, op.variant) in held:  # a planter may never produce the locked set's variants on dev
                row.planter_decisions.append({"op": op.key, "accepted": False, "refused_by": ["locked-variant"]})
                continue
            cand = Planter(id=f"{arm.name}-p{r}-{i}", ops=[op], max_per_op=2)

            def critic(pl: Planter) -> list[str]:
                ev = evidence_of(list(arm.pool.plants.values())[:200], self.builders, [])
                return leakage_critic(Update(id="planter", component="P", data=pl.model_dump(mode="json"), diff=cjson([o.params for o in pl.ops]), evidence=ev))

            try:
                dec, plants = await accept_planter(cand, tasks, self.ctx, reference=self.reference, mu=self.mu, falsifiers=fs, critic=critic,
                                                   population=[p.features for p in arm.pool.plants.values()])
            except Exception as e:  # noqa: BLE001
                row.planter_decisions.append({"op": op.key, "accepted": False, "refused_by": [f"error:{type(e).__name__}"], "detail": str(e)[:300]})
                continue
            # RSI-1: parameterisations that do not change a plant (a factor on eq_to_any, another why text) re-planted copies
            # of existing plants, and planters the population already catches entirely were accepted every round. A
            # planter now joins only with NOVEL plants (content without why texts) and a positive reward (§9.3: P is
            # rewarded for planting defects the current F misses), measured on the novel plants.
            have = {plant_key(p) for p in arm.pool.plants.values()}
            novel = [p for p in plants if plant_key(p) not in have]
            nvs, _ = await verdicts(novel, fs, self.ctx) if novel else ({}, 0)
            reward = sum(1 for p in novel if not nvs[p.id].hit) / len(novel) if novel else 0.0
            checks = {**dec.checks, "novel": bool(novel), "rewarded": reward > 0}
            ok = all(checks.values())
            d = {"op": op.key, "params": op.params, "accepted": ok, "checks": checks, "mmd": dec.mmd, "mu": self.mu, "miss_rate": reward,
                 "kept": len(novel), "planted": dec.kept, "refused_by": [k for k, v in checks.items() if not v]}
            plants = novel
            self.planter_history.append({"round": r, "op": f"{op.name}:{op.variant}", "params": op.params, "accepted": ok, "refused_by": d["refused_by"],
                                         "reward": round(reward, 3), "novel_plants": len(novel)})
            if ok:
                data = arm.store.get("P")
                data["planters"].append(cand.model_dump(mode="json"))
                v = arm.store.put("P", data)
                arm.store.set_head("P", v)
                arm.engine.head_digest["P"] = arm.store.digest("P", v)
                arm.pool.add(f"P@{v}", plants)
                d["version"] = v
            arm.engine.ledger.append(Decision(kind="update", round=r, update_id=f"{arm.name}/r{r}/planter/{op.key}", component="P", accepted=ok,
                                              evaluated=True, failed=[RuleFailure(rule="P", detail=k) for k, v in checks.items() if not v],
                                              metrics={k: d[k] for k in ("mmd", "mu", "miss_rate", "kept")}))
            row.planter_decisions.append(d)
            row.info.setdefault("planter_info", info)

    async def run_round(self, arm: Arm, r: int, proposers: Proposers) -> RoundRow:
        row = RoundRow(arm=arm.name, round=r)
        rng = random.Random(f"{self.cfg.seed}/{arm.name}/{r}")
        ev_plants = self.evidence_plants(arm)
        vs = await self.head_verdicts(arm, ev_plants)
        misses_c = miss_clusters(ev_plants, vs)
        nat_c = natural_clusters(self._natural())
        clusters = [*nat_c, *misses_c]
        missed = [p for p in ev_plants if not vs[p.id].hit]
        lib = Library.model_validate(arm.store.get("F"))
        k = self.cfg.candidates
        props: list[tuple[OperatorSpec, list[str], str]] = []
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
            for g, tag, kk in arms_g:
                ops, info = await arm.propose_ops(clusters=[c.shown() for c in nat_c], misses=[c.shown() for c in misses_c], guidance=g,
                                                  library=[s.model_dump(mode="json") for s in lib.ops.values()], k=kk, round=r, tag=tag)
                props += [(s, c, f"{'random' if arm.name == 'control' else 'swarm'}:{tag}") for s, c in ops[:kk]]
                row.lemmas += info.get("lemmas", [])
                row.strategies += info.get("strategies", [])
                if info.get("errors"):
                    row.info.setdefault("parse_errors", []).extend(info["errors"])
        except Exception as e:  # noqa: BLE001 - a failed proposal round is recorded; the round still measures
            row.info["propose_error"] = f"{type(e).__name__}: {str(e)[:300]}"
        for spec, cites, origin in props:
            cand = await self.try_candidate(arm, spec, cites, origin, clusters, missed, r, rng)
            row.candidates.append(cand)
        if alt_gl is not None:  # the meta loop: A' is kept when its candidates pass the gate more often than A's (paired round)
            def rate(tag: str) -> tuple[int, int]:
                cs = [c for c in row.candidates if c.origin.endswith(":" + tag)]
                return sum(1 for c in cs if not any(x.startswith("gate:") for x in c.refused_by)), len(cs)

            pa, na = rate("A")
            pb, nb = rate("A'")
            leaks = leakage_critic(Update(id="A", component="A", data={"guidance": alt_gl}, diff=cjson(alt_gl),
                                          evidence=evidence_of(ev_plants[:200], self.builders, [])))
            ok = nb > 0 and na > 0 and pb / nb > pa / na and not leaks
            fail = [] if ok else [RuleFailure(rule="R2", detail=f"gate pass rate A' {pb}/{nb} not above A {pa}/{na}")] if not leaks else [
                RuleFailure(rule="R6", detail=", ".join(leaks[:6]))]
            if ok:
                v = arm.store.put("A", {"guidance": alt_gl})
                arm.store.set_head("A", v)
            arm.engine.ledger.append(Decision(kind="update", round=r, update_id=f"{arm.name}/r{r}/guidance", component="A", accepted=ok, evaluated=True,
                                              failed=fail, metrics={"pass_A": [pa, na], "pass_A2": [pb, nb]}))
            row.guidance_decision = {"accepted": ok, "pass_A": [pa, na], "pass_A2": [pb, nb], "rules": [f.rule for f in fail], "new": alt_gl}
        if self.cfg.planter_loop and arm.name == "loop":
            await self.planter_round(arm, r, proposers, row)

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
        row.library = sorted(Library.model_validate(arm.store.get("F")).ops)
        row.planters = len(arm.store.get("P")["planters"])
        dev_vs, _ = await verdicts(list(arm.pool.plants.values()), population(arm.store.get("F")), self.ctx)
        row.dev = {"recall": round(sum(1 for v in dev_vs.values() if v.hit) / max(1, len(dev_vs)), 4), "n": len(dev_vs)}
        row.locked = await self.locked_summary(arm)
        row.info["spend_usd"] = round(self.spend(), 4)
        arm.rows.append(row)
        return row

    async def locked_summary(self, arm: Arm) -> dict[str, Any]:
        m = await measure(self.locked.plants, self.locked_anchors, population(arm.store.get("F")), self.report_ctx, set_name=self.locked.name,
                          set_kind="locked", purpose="report")
        return summary(m)

    async def run(self, proposers: Proposers, *, control: bool = True, save: Callable[[], None] | None = None) -> list[RoundRow]:
        loop = self.arm("loop", proposers.ops or random_proposer(self.cfg.seed), family=proposers.family)
        arms = [loop] + ([self.arm("control", random_proposer(self.cfg.seed + 1))] if control else [])
        for a in arms:
            await a.engine.calibrate(0)
            r0 = RoundRow(arm=a.name, round=0, library=[], planters=1)
            dev_vs, _ = await verdicts(list(a.pool.plants.values()), population(a.store.get("F")), self.ctx)
            r0.dev = {"recall": round(sum(1 for v in dev_vs.values() if v.hit) / max(1, len(dev_vs)), 4), "n": len(dev_vs)}
            r0.locked = await self.locked_summary(a)
            r0.info = {"noise_floor": a.engine.floor.model_dump(mode="json") if a.engine.floor else None, "mu": self.mu}
            a.rows.append(r0)
        if save:
            save()
        for r in range(1, self.cfg.rounds + 1):
            for a in arms:
                await self.run_round(a, r, proposers)
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
