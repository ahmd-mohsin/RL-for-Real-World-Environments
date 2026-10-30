"""P4 novelty and coverage (part2-paper-core §0 P4; evidence doc §6): a task is novel when it adds behaviour the pool lacks.

**Behaviour signature** (``Signature``, ``signature``), from the task and its executed oracle run, with no model:
- ``tools``: the abstracted tool-call pattern, ``R:``/``W:`` + tool per call, consecutive repeats collapsed to ``+``;
- ``read`` / ``written``: the collections the oracle's observations show (observed objects matched to start-state
  rows) and the collections its writes changed;
- ``shape``: the contract pattern shape, one token per required pattern (kind, collection, field, predicate op; never a
  key or value), per allowed pattern, answer check, trace constraint and exec check;
- ``choices``: the decision vocabulary π* writes (``tool.arg=value`` for enumerated values: a reason code, a status);
  entity-independent, so a twin keeps it, while another decision changes it (M-P4 finding: without it, "decide invoice
  A" and "decide invoice B" with different policy outcomes had one signature);
- ``deps`` and ``depth``: the ref graph between steps as tool pairs (``src>dst``) and its longest chain;
- ``cell``: the skill cell (the ``CellGrid``'s assignment, else the intent card's skill).
``key()`` hashes the exact pattern (tools, written, shape, choices, depth, cell): a task is **novel** when its key is
not in the pool and no pool task is a near duplicate of it.

**Near duplicates** (``similarity``, ``instruction_similarity``, ``score``, ``calibrate``): the signature similarity is
a weighted mean of Jaccard similarities (tool unigrams and bigrams as multisets, read and written collections, shape
multiset, choices, dependency pairs), a depth term and a cell term; the instruction similarity is the mean of a
token-set Jaccard and a character-trigram cosine after masking numbers and id-like tokens. The pair score is
``w·sig + (1 − w)·ins``. ``calibrate`` picks (w, θ) on labelled pairs (known duplicates and known distinct tasks) by
the largest Youden index (TPR − FPR), ties to the larger margin, and reports TPR, FPR and their Wilson intervals.
``known_pairs`` builds the labels by construction: positives are TWIN twins (another entity, the instruction rewritten)
and, when a paraphraser is injected, a twin's instruction paraphrased (another entity *and* other words: the hard
positive); negatives are enum-swapped variants (the same text with another stated decision: the hard negative), task
pairs the proposer put in different cells of one domain, and pairs from different domains.

**Coverage** (``CellGrid``, ``coverage``, ``diversity``): the grid is any object with ``cells`` (a mapping of cell id to
a description or ``{"match": [keywords]}``) and optional ``cell_of(task)``; ``synth.domain``'s ``DomainSpec`` will
satisfy it, and a plain dict works meanwhile. Coverage is the share of grid cells with at least one task, the per-cell
histogram, its entropy and Gini–Simpson index; diversity adds signature richness (distinct keys per task), the mean
pairwise signature distance and the near-duplicate clusters.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from itertools import pairwise
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.credit.operators.branch_value import wilson
from crucible.synth.canon import CanonState, content_hash
from crucible.synth.contract import Contract
from crucible.synth.program import Program, Ref, Run, walk_args
from crucible.synth.task import Task

# ---------------------------------------------------------------------------------------------------------- signature


class Signature(CrucibleModel):
    task_id: str = ""
    domain: str = ""
    tools: list[str] = Field(default_factory=list)
    read: list[str] = Field(default_factory=list)
    written: list[str] = Field(default_factory=list)
    shape: list[str] = Field(default_factory=list)
    choices: list[str] = Field(default_factory=list)
    deps: list[str] = Field(default_factory=list)
    depth: int = 0
    cell: str = ""
    instruction: str = ""

    def key(self) -> str:
        return content_hash({"tools": self.tools, "written": self.written, "shape": self.shape, "choices": self.choices, "depth": self.depth,
                             "cell": self.cell}, 16)


def tool_pattern(calls: Sequence[tuple[str, bool]]) -> list[str]:
    out: list[str] = []
    for tool, ro in calls:
        tok = f"{'R' if ro else 'W'}:{tool}"
        if out and out[-1].rstrip("+") == tok:
            out[-1] = tok + "+"
        else:
            out.append(tok)
    return out


def contract_shape(c: Contract) -> list[str]:
    def pred(p: Any) -> str:
        return getattr(p, "op", "none") if p is not None else "none"

    out = [f"R:{p.kind}:{p.app}.{p.coll}.{p.field}:{pred(p.value)}" + (f":w{len(p.where)}" if p.where else "") for p in c.required]
    out += [f"A:{p.kind}:{p.app}.{p.coll}" for p in c.allowed]
    out += [f"Q:{q.op}" for q in c.answer]
    out += [f"T:{t.op}" for t in c.trace]
    out += [f"X:{getattr(x, 'kind', 'exec')}" for x in getattr(c, "exec", [])]
    return sorted(out)


def ref_edges(p: Program) -> tuple[list[str], int]:
    tool = {s.id: s.tool for s in p.steps}
    edges: list[str] = []
    depth: dict[str, int] = {}
    for s in p.steps:
        srcs: set[str] = set()
        for name, a in s.args.items():
            for _, x in walk_args(a, f"{s.id}.{name}"):
                if isinstance(x, Ref):
                    srcs.add(x.path.split(".", 1)[0])
        edges += sorted(f"{tool.get(x, '?')}>{s.tool}" for x in srcs)
        depth[s.id] = 1 + max((depth.get(x, 0) for x in srcs), default=0)
    return sorted(edges), max(depth.values(), default=0)


def observed_collections(value: Any, s0: CanonState | None) -> set[str]:
    """Collections whose rows an observation shows: an object with an id-like field equal to a row key that agrees
    with the row on at least one more field."""
    if s0 is None:
        return set()
    index: dict[str, list[tuple[str, str, dict[str, Any]]]] = {}
    for app, coll, key, row in s0.entities():
        index.setdefault(str(key), []).append((app, coll, row))
    out: set[str] = set()

    def walk(v: Any) -> None:
        if isinstance(v, list):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for f, x in v.items():
                if isinstance(x, int | str) and not isinstance(x, bool) and str(x) in index:
                    for app, coll, row in index[str(x)]:
                        # the object is the row itself (its key field under the same name), not a foreign key to it
                        if f in row and str(row[f]) == str(x) and sum(1 for g, y in v.items() if g != f and g in row and row[g] == y) >= 1:
                            out.add(f"{app}.{coll}")
                walk(x)

    walk(value)
    return out


def choices(task: Task, oracle: Run, tools: Sequence[Any] = ()) -> list[str]:
    """The decisions the oracle's author resolved: ``tool.arg=value`` for every write argument of π* that is a plain
    **literal** (never a ref: a value copied from the data is data flow, and a twin's differs) whose value the tool
    catalogue enumerates in its schema or names in a description (a reason code, a status). Entity-independent, so a
    twin keeps it; another stated decision changes it."""
    from crucible.synth.program import Lit

    enums: dict[tuple[str, str], list[Any]] = {}
    for t in tools:
        for name, p in ((t.input_schema or {}).get("properties") or {}).items():
            if isinstance(p, dict) and p.get("enum"):
                enums[(t.name, name)] = list(p["enum"])
    spec = {e.loc for e in oracle.taint if e.has("tool_spec") and not e.has("structural")}
    ro = {c.step: c.read_only for c in oracle.calls}
    out: set[str] = set()
    for s in task.oracle.steps:
        if ro.get(s.id, True):
            continue
        for name, a in s.args.items():
            v = a.v if isinstance(a, Lit) else None
            if not isinstance(v, str | int | float) or isinstance(v, bool):
                continue
            if v in enums.get((s.tool, name), []) or f"{s.id}.{name}" in spec:
                out.add(f"{s.tool}.{name}={v}")
    return sorted(out)


def signature(task: Task, oracle: Run | None = None, *, cell: str | None = None, tools: Sequence[Any] = ()) -> Signature:
    calls = [(c.tool, c.read_only) for c in oracle.calls] if oracle is not None else [(s.tool, False) for s in task.oracle.steps]
    written: set[str] = set()
    read: set[str] = set()
    if oracle is not None:
        for c in oracle.calls:
            for e in c.effects or []:
                written.add(f"{e[0]}.{e[1]}")
            if c.read_only and c.obs.value is not None:
                read |= observed_collections(c.obs.value, oracle.start)
    deps, depth = ref_edges(task.oracle)
    return Signature(task_id=task.id, domain=task.domain, tools=tool_pattern(calls), read=sorted(read), written=sorted(written),
                     shape=contract_shape(task.contract), choices=choices(task, oracle, tools) if oracle is not None else [], deps=deps, depth=depth,
                     cell=cell if cell is not None else task.intent.skill.strip().lower(), instruction=task.instruction)


# ---------------------------------------------------------------------------------------------------------- similarity

def jaccard(a: Iterable[str], b: Iterable[str], *, multiset: bool = False) -> float:
    if multiset:
        ca, cb = Counter(a), Counter(b)
        inter, union = sum((ca & cb).values()), sum((ca | cb).values())
    else:
        sa, sb = set(a), set(b)
        inter, union = len(sa & sb), len(sa | sb)
    return 1.0 if union == 0 else inter / union


SIG_WEIGHTS = {"tools": 0.15, "bigrams": 0.1, "read": 0.1, "written": 0.1, "shape": 0.15, "choices": 0.2, "deps": 0.1, "depth": 0.05,
               "cell": 0.05}


def similarity(a: Signature, b: Signature, weights: Mapping[str, float] | None = None) -> float:
    w = weights or SIG_WEIGHTS
    big = lambda t: [f"{x}|{y}" for x, y in pairwise(t)]
    parts = {"tools": jaccard(a.tools, b.tools, multiset=True), "bigrams": jaccard(big(a.tools), big(b.tools), multiset=True),
             "read": jaccard(a.read, b.read), "written": jaccard(a.written, b.written), "shape": jaccard(a.shape, b.shape, multiset=True),
             "deps": jaccard(a.deps, b.deps, multiset=True), "depth": 1.0 - abs(a.depth - b.depth) / max(a.depth, b.depth, 1),
             "choices": jaccard(a.choices, b.choices), "cell": 1.0 if a.cell == b.cell else 0.0}
    tot = sum(w.values()) or 1.0
    return sum(w[k] * parts[k] for k in w) / tot


_ID = re.compile(r"\b(?=[\w-]*\d)[\w-]+\b")
_WORD = re.compile(r"[a-z]+")


def mask(text: str) -> str:
    return _ID.sub(" # ", text.lower())


def _trigrams(s: str) -> Counter[str]:
    s = re.sub(r"\s+", " ", s)
    return Counter(s[i:i + 3] for i in range(max(0, len(s) - 2)))


def instruction_similarity(a: str, b: str) -> float:
    ma, mb = mask(a), mask(b)
    tj = jaccard(_WORD.findall(ma), _WORD.findall(mb))
    ca, cb = _trigrams(ma), _trigrams(mb)
    dot = sum(ca[k] * cb.get(k, 0) for k in ca)
    na, nb = math.sqrt(sum(v * v for v in ca.values())), math.sqrt(sum(v * v for v in cb.values()))
    cos = dot / (na * nb) if na and nb else 0.0
    return 0.5 * tj + 0.5 * cos


DECISION_GATE = 0.5  # two tasks whose stated decisions differ (both write enumerated choices, and the sets differ) score × this


def score(a: Signature, b: Signature, w: float = 0.6, gate: float = DECISION_GATE) -> float:
    """The pair score. **Decision gate**: when both tasks write enumerated choices and the sets differ, the score is
    multiplied by ``gate`` (1.0 = off): another stated decision is another behaviour (SWAP's rule), even in the same words."""
    s = w * similarity(a, b) + (1.0 - w) * instruction_similarity(a.instruction, b.instruction)
    if a.choices and b.choices and set(a.choices) != set(b.choices):
        s *= gate
    return s


class Calibration(CrucibleModel):
    w: float
    theta: float
    gate: float = DECISION_GATE
    tpr: float
    fpr: float
    tpr_ci: list[float]
    fpr_ci: list[float]
    positives: int
    negatives: int
    youden: float
    margin: float
    by_kind: dict[str, dict[str, float]] = Field(default_factory=dict)
    grid: list[dict[str, float]] = Field(default_factory=list)


def calibrate(pairs: Sequence[tuple[Signature, Signature, bool, str]], *, ws: Sequence[float] = (0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0),
              gates: Sequence[float] = (1.0, DECISION_GATE)) -> Calibration:
    """(w, gate, θ) with the largest Youden index on labelled pairs (sig a, sig b, is_duplicate, kind); θ is the
    midpoint between the two scores it separates, so the margin is reported. ``gates`` includes 1.0 (no decision gate)
    so the report shows what the gate buys."""
    pos = [p for p in pairs if p[2]]
    neg = [p for p in pairs if not p[2]]
    if not pos or not neg:
        raise ValueError("calibration needs both known duplicates and known distinct pairs")
    best: tuple[float, float, float, float, float, float, float] | None = None  # youden, margin, w, gate, theta, tpr, fpr
    grid = []
    for g in gates:
        for w in ws:
            sp = sorted(score(a, b, w, g) for a, b, _, _ in pos)
            sn = sorted(score(a, b, w, g) for a, b, _, _ in neg)
            cands = sorted(set(sp + sn))
            best_here = -1.0
            for i, t in enumerate(cands):
                lo = cands[i - 1] if i else t - 1e-3
                theta = (lo + t) / 2
                tpr = sum(1 for s in sp if s >= theta) / len(sp)
                fpr = sum(1 for s in sn if s >= theta) / len(sn)
                j = tpr - fpr
                best_here = max(best_here, j)
                margin = t - lo
                if best is None or (j, margin) > (best[0], best[1]):
                    best = (j, margin, w, g, theta, tpr, fpr)
            grid.append({"w": w, "gate": g, "best_youden": round(best_here, 4)})
    assert best is not None
    j, margin, w, g, theta, tpr, fpr = best
    by_kind: dict[str, dict[str, float]] = {}
    for kind in sorted({p[3] for p in pairs}):
        ks = [p for p in pairs if p[3] == kind]
        hit = sum(1 for a, b, lab, _ in ks if (score(a, b, w, g) >= theta) == lab)
        by_kind[kind] = {"n": len(ks), "correct": hit, "rate": round(hit / len(ks), 4)}
    tp, fp = round(tpr * len(pos)), round(fpr * len(neg))
    return Calibration(w=w, theta=round(theta, 4), gate=g, tpr=round(tpr, 4), fpr=round(fpr, 4), tpr_ci=[round(x, 4) for x in wilson(tp, len(pos))],
                       fpr_ci=[round(x, 4) for x in wilson(fp, len(neg))], positives=len(pos), negatives=len(neg), youden=round(j, 4),
                       margin=round(margin, 4), by_kind=by_kind, grid=grid)


DEFAULT_W, DEFAULT_THETA = 0.6, 0.85  # used until a calibration is given


def near_duplicates(sigs: Sequence[Signature], *, w: float = DEFAULT_W, theta: float = DEFAULT_THETA, gate: float = DECISION_GATE
                    ) -> list[tuple[str, str, float]]:
    out = []
    for i in range(len(sigs)):
        for j in range(i + 1, len(sigs)):
            s = score(sigs[i], sigs[j], w, gate)
            if s >= theta:
                out.append((sigs[i].task_id, sigs[j].task_id, round(s, 4)))
    return out


class Novelty(CrucibleModel):
    novel: bool
    key: str
    new_key: bool
    nearest: str | None = None
    nearest_score: float | None = None
    reasons: list[str] = Field(default_factory=list)


def novel(sig: Signature, pool: Sequence[Signature], *, w: float = DEFAULT_W, theta: float = DEFAULT_THETA, gate: float = DECISION_GATE) -> Novelty:
    """Novel = its signature key is new to the pool and no pool task scores ≥ θ against it."""
    others = [s for s in pool if s.task_id != sig.task_id]
    keys = {s.key() for s in others}
    k = sig.key()
    best, bs = None, -1.0
    for s in others:
        x = score(sig, s, w, gate)
        if x > bs:
            best, bs = s.task_id, x
    reasons = []
    if k in keys:
        reasons.append(f"signature {k} already in the pool")
    if best is not None and bs >= theta:
        reasons.append(f"near duplicate of {best} (score {bs:.3f} >= {theta})")
    return Novelty(novel=not reasons, key=k, new_key=k not in keys, nearest=best, nearest_score=round(bs, 4) if best is not None else None,
                   reasons=reasons)


# ---------------------------------------------------------------------------------------------------------- known pairs

async def twin_variant(task: Task, ctx: Any, probe: Any) -> Task | None:
    """A task from a TWIN probe (another entity; ι, π* and C rewritten): a known duplicate of ``task``."""
    if probe.program is None or probe.instruction is None:
        return None
    t2 = task.model_copy(update={"id": f"{task.id}~{probe.site}", "instruction": probe.instruction, "oracle": probe.program,
                                 "contract": probe.contract or task.contract}, deep=True)
    r = await ctx.run(t2, t2.oracle)
    if r.error is not None:
        return None
    return t2


async def known_pairs(tasks: Sequence[Task], ctx: Any, *, cells: Mapping[str, str] | None = None, per_task: int = 2,
                      paraphrase: Callable[[str], Any] | None = None, cell_of: Callable[[Task], str] | None = None
                      ) -> list[tuple[Signature, Signature, bool, str]]:
    """Labelled pairs by construction (module docstring). ``cells`` maps task id -> the proposer's cell; ``paraphrase``
    (async or sync, a non-student model) rewrites an instruction with the same meaning."""
    import inspect

    from crucible.synth.falsify.twin import TWIN
    from crucible.synth.intent import SWAP

    cof = cell_of or (lambda t: t.intent.skill.strip().lower())
    base: dict[str, Signature] = {}
    for t in tasks:
        o = await ctx.oracle(t)
        base[t.id] = signature(t, o, cell=cof(t), tools=await (await ctx.world(t)).tools())
    out: list[tuple[Signature, Signature, bool, str]] = []
    for t in tasks:
        o = await ctx.oracle(t)
        if not ctx.grade(o, t.contract).passed:
            continue
        tools = await (await ctx.world(t)).tools()
        twins: list[Signature] = []
        try:
            probes = await TWIN(twins=per_task).propose(t, ctx) if o.start is not None else []
        except Exception:  # noqa: BLE001 - a World that cannot be twinned gives no pair
            probes = []
        for p in probes[:per_task]:
            t2 = await twin_variant(t, ctx, p)
            if t2 is not None:
                s2 = signature(t2, await ctx.run(t2, t2.oracle, taint=True), cell=cof(t), tools=tools)
                twins.append(s2)
                out.append((base[t.id], s2, True, "twin"))
        try:
            sw = [p for p in await SWAP().propose(t, ctx) if p.op == "enum"][:1]
        except Exception:  # noqa: BLE001
            sw = []
        for p in sw:  # another decision the instruction states: distinct behaviour (a hard negative)
            t2 = t.model_copy(update={"id": f"{t.id}~{p.site}", "instruction": p.instruction, "oracle": p.program}, deep=True)
            r = await ctx.run(t2, t2.oracle, taint=True)
            s2 = signature(t2, r, cell=cof(t), tools=tools) if r.error is None else None
            if s2 is not None and set(s2.choices) != set(base[t.id].choices):  # a swapped search word only is no other decision
                out.append((base[t.id], s2, False, "enum_swap"))
        if paraphrase is not None:
            # the hard positive: another entity (a twin) and other words; else the same task reworded
            src = twins[0] if twins else base[t.id]
            x = paraphrase(src.instruction)
            text = await x if inspect.isawaitable(x) else x
            if isinstance(text, str) and text.strip() and text.strip() != src.instruction:
                out.append((base[t.id], src.model_copy(update={"instruction": text.strip(), "task_id": f"{src.task_id}~para"}), True,
                            "twin+paraphrase" if twins else "paraphrase"))
    ids = [t.id for t in tasks]
    dom = {t.id: t.domain for t in tasks}
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            a, b = ids[i], ids[j]
            if dom[a] != dom[b]:
                out.append((base[a], base[b], False, "cross_domain"))
            elif cells and cells.get(a) and cells.get(b) and cells[a] != cells[b]:
                out.append((base[a], base[b], False, "cross_cell"))
    return out


# ---------------------------------------------------------------------------------------------------------- coverage

@runtime_checkable
class CellGrid(Protocol):
    cells: Mapping[str, Any]


def _match_words(desc: Any) -> list[str]:
    if isinstance(desc, Mapping):
        return [str(x).lower() for x in desc.get("match", [])] or _WORD.findall(str(desc.get("description", "")).lower())
    return [w for w in _WORD.findall(str(desc).lower()) if len(w) > 3]


def assign_cell(task: Task, grid: CellGrid | Mapping[str, Any], *, recorded: str | None = None) -> str | None:
    """The task's cell: ``grid.cell_of(task)`` when the grid has one; a recorded cell (the proposer's) that is a grid
    cell id or description; else the cell whose match words overlap the intent skill and instruction most."""
    cells = grid.cells if hasattr(grid, "cells") else grid
    f = getattr(grid, "cell_of", None)
    if callable(f):
        c = f(task)
        if c is not None:
            return str(c)
    if recorded is not None:
        for cid, d in cells.items():
            if recorded == cid or (isinstance(d, str) and recorded == d) or (isinstance(d, Mapping) and recorded == d.get("description")):
                return cid
    words = set(_WORD.findall((task.intent.skill + " " + task.instruction).lower()))
    best, bs = None, 0
    for cid, d in cells.items():
        s = len(words & set(_match_words(d)))
        if s > bs:
            best, bs = cid, s
    return best


def entropy(counts: Mapping[str, int]) -> float:
    tot = sum(counts.values())
    return -sum((v / tot) * math.log(v / tot) for v in counts.values() if v > 0) if tot else 0.0


def coverage(assigned: Mapping[str, str | None], grid: CellGrid | Mapping[str, Any]) -> dict[str, Any]:
    cells = list((grid.cells if hasattr(grid, "cells") else grid).keys())
    hist = {c: 0 for c in cells}
    unassigned = 0
    for c in assigned.values():
        if c in hist:
            hist[c] += 1
        else:
            unassigned += 1
    tot = sum(hist.values())
    covered = sum(1 for v in hist.values() if v)
    return {"cells": len(cells), "covered": covered, "coverage": round(covered / len(cells), 4) if cells else None, "histogram": hist,
            "unassigned": unassigned, "entropy": round(entropy(hist), 4), "entropy_max": round(math.log(len(cells)), 4) if cells else 0.0,
            "gini_simpson": round(1 - sum((v / tot) ** 2 for v in hist.values()), 4) if tot else None,
            "empty_cells": [c for c, v in hist.items() if not v]}


def diversity(sigs: Sequence[Signature], *, w: float = DEFAULT_W, theta: float = DEFAULT_THETA, gate: float = DECISION_GATE) -> dict[str, Any]:
    keys = Counter(s.key() for s in sigs)
    dists = [1.0 - similarity(sigs[i], sigs[j]) for i in range(len(sigs)) for j in range(i + 1, len(sigs))]
    dups = near_duplicates(sigs, w=w, theta=theta, gate=gate)
    parent = {s.task_id: s.task_id for s in sigs}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b, _ in dups:
        parent[find(a)] = find(b)
    clusters = Counter(find(s.task_id) for s in sigs)
    return {"tasks": len(sigs), "distinct_signatures": len(keys), "richness": round(len(keys) / len(sigs), 4) if sigs else None,
            "repeated_signatures": {k: v for k, v in keys.items() if v > 1},
            "mean_pairwise_distance": round(sum(dists) / len(dists), 4) if dists else None,
            "near_duplicate_pairs": len(dups), "near_duplicate_clusters": sum(1 for v in clusters.values() if v > 1),
            "effective_tasks": len(clusters), "tools_patterns": len({tuple(s.tools) for s in sigs}),
            "written_collections": sorted({c for s in sigs for c in s.written}), "max_depth": max((s.depth for s in sigs), default=0)}
