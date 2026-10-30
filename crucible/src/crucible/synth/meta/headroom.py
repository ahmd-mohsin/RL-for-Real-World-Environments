"""Dev headroom: how much of what the falsifier population misses an operator of the DSL could catch at all.

**Definition.** For a set of dev plants, the current population's misses M, and the dev anchors A:

    catchable(p)  ⇔  some expressible operator o confirms a counterexample on p and confirms none on any anchor of A
    headroom_abs   = |{p ∈ M : catchable(p)}| / |plants|     (the most a single round of DSL operators could add to J)
    headroom_rel   = |{p ∈ M : catchable(p)}| / |M|

**Expressible operators** (``expressible_bodies`` × ``LABELS``) are the DSL grammar instantiated on the same finite
parameter sets the random control samples from (``abstract._GLOBS``, ``_ARGS``, its selectors and transforms):
``mutate_arg`` over every tool glob, write flag, argument glob, key flag and value selector or transform; ``insert_call``
over every tool selector and position; ``drop_calls``; one-call ``agent`` bodies; each with every label rule that can
confirm (``expect`` fail or pass × ``confirm`` always, outside_delta, missing_delta). An operator the grammar allows with
other constants (a glob the random control never draws) is not counted, so headroom is a lower bound on what the full
DSL could express and an upper bound on what one operator of this grammar can add.

**Cost.** A body's probes are programs. Per task, every body's programs are generated (no World call), deduplicated, and
each distinct program runs once (``CompiledOperator.execute``: the grade, MUT's equivalence and the delta). Every label
rule is then applied to the stored outcomes. Results are cached per (task digest, program digest).
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson
from crucible.synth.falsify.base import FalsifyContext, Outcome, Probe, dedupe_programs
from crucible.synth.meta.abstract import _ARGS, _GLOBS
from crucible.synth.meta.library import (
    AgentCalls,
    CompiledOperator,
    DropCalls,
    InsertCall,
    LabelRule,
    MutateArg,
    OperatorSpec,
    SAnyWriteTool,
    SConst,
    SFieldValues,
    SObservationOf,
    SReadKey,
    SReadTool,
    SSiblingOf,
    STool,
    SWrittenKey,
    TOtherSeen,
    TScale,
    TShift,
    TSuffix,
    _Env,
)
from crucible.synth.meta.plant import Plant
from crucible.synth.task import Task

LABELS: tuple[tuple[str, str], ...] = tuple((e, c) for e in ("fail", "pass") for c in ("always", "outside_delta", "missing_delta"))


def _selectors() -> list[Any]:
    out: list[Any] = [SWrittenKey(), SReadKey()]
    out += [SSiblingOf(of=o, limit=lim) for o in (SWrittenKey(), SReadKey()) for lim in (4, 8, 16)]
    out += [SObservationOf(tool=g, path=p) for g in _GLOBS for p in (None, "id", "status")]
    out += [SFieldValues(field=f) for f in ("status", "id", "priority", "reason")]
    out += [SConst(values=v) for v in ([0], [1], ["x"], [""], [None])]
    return out


def _transforms() -> list[Any]:
    return [*(TShift(by=b) for b in (-1, 1, 10, 0.01)), *(TScale(by=b) for b in (0, 2, 10, 0.5)), *(TSuffix(s=s) for s in ("-x", " ", "_1")), TOtherSeen()]


def _tool_sels() -> list[Any]:
    return [SAnyWriteTool(unused=False), SAnyWriteTool(unused=True), SReadTool(unused=False), SReadTool(unused=True), *(STool(glob=g) for g in _GLOBS)]


def expressible_bodies() -> list[Any]:
    """The bounded DSL grammar (module docstring), as operator bodies."""
    out: list[Any] = []
    for tool, wo, arg, ko in itertools.product(_GLOBS, (True, False), _ARGS, (False, True)):
        out += [MutateArg(tool=tool, write_only=wo, arg=arg, key_only=ko, values=s) for s in _selectors()]
        out += [MutateArg(tool=tool, write_only=wo, arg=arg, key_only=ko, transform=t) for t in _transforms()]
    inserts = [InsertCall(tool=t, args={}, borrow_rest=True, position=pos) for t in _tool_sels() for pos in ("start", "end", "after_writes")]
    out += inserts
    out += [DropCalls(tool=g, write_only=w) for g in _GLOBS for w in (False, True)]
    out += [AgentCalls(calls=[i]) for i in inserts if i.position == "end"]
    return out


def _confirms(o: Outcome, agent: bool, expect: str, confirm: str) -> bool:
    """``CompiledOperator.label`` for one label rule, reduced to "confirmed or not"."""
    if not o.valid:
        return False
    if not agent and o.equivalent:
        return not o.passed  # an equivalent run that fails is a confirmed D6 whatever the rule says
    violated = bool(o.passed) if expect == "fail" else not o.passed
    if not violated:
        return False
    return {"always": True, "outside_delta": bool(o.data.get("outside")), "missing_delta": bool(o.data.get("missing"))}[confirm]


class HeadroomReport(CrucibleModel):
    plants: int
    missed: int
    catchable: list[str] = Field(default_factory=list)  # missed plant ids some admissible operator catches
    headroom_abs: float = 0.0
    headroom_rel: float = 0.0
    bodies: int = 0
    operators: int = 0
    admissible: int = 0  # operators with no anchor alarm
    by_class: dict[str, list[int]] = Field(default_factory=dict)  # defect -> [missed, catchable]
    witnesses: dict[str, str] = Field(default_factory=dict)  # plant id -> one operator that catches it (body kind + label)
    runs: int = 0


class HeadroomCache:
    """Outcomes per (task digest, program digest), kept across rounds (plants and anchors are immutable)."""

    def __init__(self) -> None:
        self.outcomes: dict[tuple[str, str], Outcome] = {}
        self.programs: dict[str, list[list[tuple[str, bool]]]] = {}  # task digest -> per body: [(program key, agent)]


async def _task_table(task: Task, bodies: Sequence[Any], ctx: FalsifyContext, cache: HeadroomCache) -> tuple[list[list[tuple[str, bool]]], int]:
    """Per body, the (program key, agent) list of its probes on the task; every distinct program executed once."""
    td = task.digest()
    runs = 0
    if td not in cache.programs:
        oracle = await ctx.oracle(task)
        tools = await (await ctx.world(task)).tools()
        env = _Env(task, oracle, tools)
        per: list[list[tuple[str, bool]]] = []
        progs: dict[str, Probe] = {}
        for b in bodies:
            op = CompiledOperator(OperatorSpec(name="h", body=b, label=LabelRule(defect="D1")))
            probes = [Probe(cls=op.name, op="h", site=s, program=p, data={"agent": a}) for s, p, a in op._programs(env)]
            probes = dedupe_programs(probes, {task.oracle.digest()})[: op.spec.max_probes]
            row = []
            for p in probes:
                k = cjson(p.program.model_dump(mode="json")) if p.program is not None else p.site
                progs.setdefault(k, p)
                row.append((k, bool(p.data.get("agent"))))
            per.append(row)
        cache.programs[td] = per
        runner = CompiledOperator(OperatorSpec(name="h", body=DropCalls(), label=LabelRule(defect="D1")))
        for k, p in progs.items():
            if (td, k) not in cache.outcomes:
                cache.outcomes[(td, k)] = await runner.execute(p, task, ctx)
                runs += 1
    return cache.programs[td], runs


async def catchable(plants: Sequence[Plant], missed: Iterable[str], anchors: Sequence[Task], ctx: FalsifyContext, *, bodies: Sequence[Any] | None = None,
                    cache: HeadroomCache | None = None) -> HeadroomReport:
    """The headroom of a dev set (module docstring): ``missed`` are the plant ids the current population does not catch."""
    bodies = list(bodies) if bodies is not None else expressible_bodies()
    cache = cache or HeadroomCache()
    miss = [p for p in plants if p.id in set(missed)]
    runs = 0
    n_ops = len(bodies) * len(LABELS)
    alarm = [[False] * len(LABELS) for _ in bodies]  # per (body, label): alarms on some anchor
    for a in anchors:
        table, r = await _task_table(a, bodies, ctx, cache)
        runs += r
        td = a.digest()
        for bi, row in enumerate(table):
            for li, (e, c) in enumerate(LABELS):
                if not alarm[bi][li] and any(_confirms(cache.outcomes[(td, k)], ag, e, c) for k, ag in row):
                    alarm[bi][li] = True
    admissible = sum(1 for row in alarm for x in row if not x)
    caught: list[str] = []
    wit: dict[str, str] = {}
    for p in miss:
        table, r = await _task_table(p.task, bodies, ctx, cache)
        runs += r
        td = p.task.digest()
        done = False
        for bi, row in enumerate(table):
            for li, (e, c) in enumerate(LABELS):
                if alarm[bi][li]:
                    continue
                if any(_confirms(cache.outcomes[(td, k)], ag, e, c) for k, ag in row):
                    caught.append(p.id)
                    wit[p.id] = f"{bodies[bi].kind}:{cjson(bodies[bi].model_dump(mode='json'))[:160]} expect={e} confirm={c}"
                    done = True
                    break
            if done:
                break
    by: dict[str, list[int]] = {}
    for p in miss:
        row2 = by.setdefault(p.defect, [0, 0])
        row2[0] += 1
        row2[1] += p.id in set(caught)
    n = len(plants)
    return HeadroomReport(plants=n, missed=len(miss), catchable=caught, headroom_abs=len(caught) / n if n else 0.0,
                          headroom_rel=len(caught) / len(miss) if miss else 0.0, bodies=len(bodies), operators=n_ops, admissible=admissible,
                          by_class=dict(sorted(by.items())), witnesses=wit, runs=runs)
