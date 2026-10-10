"""Stage induction (DESIGN §5.3.4): milestones -> ordered stages, from when they are first achieved across the family's
traces.

1. **Precedence.** For each milestone pair (A, B) reached in the same trace, record whether A was achieved strictly
   before B. A -> B is an edge when A precedes B in at least ``min_support`` (default 0.9) of the traces that reach
   both, over at least ``min_traces`` traces.
2. **Layers.** Stage index = the longest precedence path into the milestone (a milestone with no edges is placed by its
   median normalised achievement time against the layers' medians), so a stage only contains milestones that are
   not ordered among themselves.
3. **Names** are proposed separately (``name_stages``: a deterministic default from the milestones' kinds and files, or
   an LLM labeller passed in); membership is decided by the data, never by the namer.

``stage_of`` locates a state: the current stage is the first stage whose milestones (by weight) are not yet at least
``done_share`` satisfied, given that every earlier stage is.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

from crucible.core.schema.milestone import Milestone


@dataclass
class Precedence:
    a: str
    b: str
    support: float
    n: int


@dataclass
class Stage:
    index: int
    milestones: list[str]
    name: str = ""
    median_time: float = 0.0


@dataclass
class StageModel:
    stages: list[Stage]
    edges: list[Precedence] = field(default_factory=list)
    unplaced: list[str] = field(default_factory=list)  # never achieved in any trace

    def stage_index(self) -> dict[str, int]:
        return {m: s.index for s in self.stages for m in s.milestones}


def precedence(achievements: Sequence[Mapping[str, float]], *, min_support: float = 0.9, min_traces: int = 3) -> list[Precedence]:
    """``achievements[k]``: milestone -> normalised time (0..1) it was achieved in trace k (absent = not reached)."""
    pair_n: Counter[tuple[str, str]] = Counter()
    pair_before: Counter[tuple[str, str]] = Counter()
    for ach in achievements:
        ids = sorted(ach)
        for i, a in enumerate(ids):
            for b in ids[i + 1:]:
                pair_n[(a, b)] += 1
                if ach[a] < ach[b]:
                    pair_before[(a, b)] += 1
                elif ach[b] < ach[a]:
                    pair_before[(b, a)] += 1
    out = []
    for (a, b), n in pair_n.items():
        if n < min_traces:
            continue
        for x, y in ((a, b), (b, a)):
            sup = pair_before[(x, y)] / n
            if sup >= min_support:
                out.append(Precedence(x, y, sup, n))
    return out


def induce(milestones: Sequence[Milestone], achievements: Sequence[Mapping[str, float]], *, min_support: float = 0.9,
           min_traces: int = 3) -> StageModel:
    ids = [m.id for m in milestones]
    reached = {m for ach in achievements for m in ach}
    edges = [e for e in precedence(achievements, min_support=min_support, min_traces=min_traces) if e.a in ids and e.b in ids]
    # prerequisites declared on the milestones are precedence too
    for m in milestones:
        for p in m.requires:
            if not any(e.a == p and e.b == m.id for e in edges):
                edges.append(Precedence(p, m.id, 1.0, 0))
    preds: dict[str, set[str]] = {m: set() for m in ids}
    for e in edges:
        preds[e.b].add(e.a)
    level: dict[str, int] = {}

    def depth(m: str, seen: frozenset[str] = frozenset()) -> int:
        if m in level:
            return level[m]
        if m in seen:  # contradictory cycle in the data: break it
            return 0
        d = 1 + max((depth(p, seen | {m}) for p in preds[m]), default=-1)
        level[m] = d
        return d

    med = {m: statistics.median([ach[m] for ach in achievements if m in ach]) for m in ids if m in reached}
    linked = {e.a for e in edges} | {e.b for e in edges}
    for m in ids:
        if m in linked:
            depth(m)
    layers: dict[int, list[str]] = {}
    for m, lv in level.items():
        layers.setdefault(lv, []).append(m)
    layer_med = {lv: statistics.median([med[m] for m in ms if m in med] or [0.0]) for lv, ms in layers.items()}
    for m in ids:
        if m in level or m not in med:
            continue
        if layer_med:
            lv = min(layer_med, key=lambda x: abs(layer_med[x] - med[m]))
        else:
            lv = 0
        layers.setdefault(lv, []).append(m)
    stages = [Stage(index=i, milestones=sorted(layers[lv]), median_time=statistics.median([med[m] for m in layers[lv] if m in med] or [0.0]))
              for i, lv in enumerate(sorted(layers))]
    return StageModel(stages=stages, edges=edges, unplaced=sorted(m for m in ids if m not in reached and m not in level))


def default_name(milestone_ids: Sequence[str]) -> str:
    """A readable default: the dominant milestone kind and file (code), or the fields reached (event logs)."""
    if milestone_ids and all(m.count("/") == 1 for m in milestone_ids):
        return " + ".join(sorted({m.split("/", 1)[1] for m in milestone_ids}))[:60]
    kinds = Counter(m.split("/")[1] if m.count("/") >= 2 else m.split("/")[0] for m in milestone_ids)
    files = Counter(m.split("/")[-1].split(":")[0] for m in milestone_ids if ":" in m)
    k = kinds.most_common(1)[0][0] if kinds else "stage"
    f = files.most_common(1)[0][0] if files else ""
    return f"{k} {f}".strip()


def name_stages(model: StageModel, namer: Callable[[list[str]], str] | None = None) -> StageModel:
    for s in model.stages:
        s.name = (namer or default_name)(s.milestones)
    return model


def stage_of(model: StageModel, milestones: Sequence[Milestone], phis: Mapping[str, float | None], *, done_share: float = 0.8) -> int:
    w = {m.id: m.weight for m in milestones}
    for s in model.stages:
        tot = sum(w.get(m, 1.0) for m in s.milestones)
        got = sum(w.get(m, 1.0) for m in s.milestones if (phis.get(m) or 0) >= 1.0)
        if tot and got / tot < done_share:
            return s.index
    return len(model.stages)  # every stage done
