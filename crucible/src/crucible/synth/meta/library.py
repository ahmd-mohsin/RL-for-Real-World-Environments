"""The falsifier operator library: a typed DSL, its compiled probes, the birth gate, retirement, and the planter loop's
acceptance (part2-paper-core §9.3, §9.5).

**The DSL** (JSON, pydantic discriminated unions, so an abstractor can emit it and a ledger can store it):
- *Selectors* over the action/state algebra of a task (π*'s run, s₀, Δ*, the tool catalogue): ``written_key`` (entities
  Δ* changes), ``read_key`` (entity keys π*'s reads name), ``sibling_of(sel)`` (other keys of the same collection that π*
  does not write), ``any_write_tool`` / ``read_tool`` (optionally only tools π* never calls), ``tool(glob)``,
  ``observation_of(tool, path)`` (scalar leaves π* observed from a read), ``field_values(field)`` (s₀ values of a field),
  ``const(values)``.
- *Transforms* on a value: ``shift(by)``, ``scale(by)``, ``suffix(s)``, ``other_seen`` (another value seen for the same
  argument name: the schema's enum, s₀ cells, π*'s other arguments).
- *Operator bodies*: ``mutate_arg`` (steps by tool glob and write-only flag, arguments by glob, optionally only entity-key
  arguments; each value of a selector, or a transform, makes one mutant), ``insert_call`` (a call of each selected tool,
  arguments from selectors or ``borrow``ed by name from π*, at the start, the end or after each write),
  ``drop_calls``, and ``agent`` (NULL-style: calls from scratch, no π*).
- *A label rule*: ``expect`` (``fail``: the probe must fail; ``pass``: it must pass), the ``defect`` it confirms, and
  ``confirm`` (``always``; ``outside_delta``: the run changed something outside Δ*; ``missing_delta``: it lacks part of
  Δ*; ``never``: survivors are suspects). An equivalent mutant (MUT's equivalence) must pass whatever the rule says, and
  failing is a confirmed D6, exactly as in MUT.

``compile_operator(spec)`` gives a ``CompiledOperator`` (a ``BaseFalsifier`` named ``OP:<name>@<version>``), so a
library operator runs, labels, certifies and archives like any built-in class.

**The birth gate** (§9.5, R1–R7 on one operator): it must reproduce its source counterexamples, find planted defects on
a fresh dev view of *other* tasks in *other* domains, raise zero false alarms on certified anchors, and stay under its
cost bar (mean probes per task). ``retire`` removes operators with no unique hit on the union of all past planters'
sets for W rounds, only if the library's recall on that union does not drop (``fictitious_play``: F must keep recall on
every past planter's set, not just the newest).

**The planter loop** (``accept_planter``): a candidate planter is accepted when it produces witnessed plants (each one
re-validated by execution), its features are within MMD ≤ μ of the reference (real-defect) features, and the injected
critic (R6) finds no leakage in its spec. Its reward, reported, is the current population's miss rate on its plants.
"""

from __future__ import annotations

import fnmatch
import inspect
import itertools
from collections.abc import Awaitable, Callable, Sequence
from typing import Annotated, Any, ClassVar, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson, is_number
from crucible.synth.falsify.base import (
    BaseFalsifier,
    FalsifyContext,
    Label,
    Outcome,
    Probe,
    dedupe_programs,
    delta_of,
    effect_signature,
    entity_index,
    equivalent,
    falsify,
    key_text,
    run_classes,
    typed_like,
    written_entities,
)
from crucible.synth.falsify.mut import MUT, enum_alternatives, key_colls
from crucible.synth.falsify.null import borrowed_args
from crucible.synth.iface import ToolInfo
from crucible.synth.meta.plant import Plant, Planter, feature_matrix, mmd, plant_all, validate
from crucible.synth.program import Lit, Program, Run, Step, leaves
from crucible.synth.task import Task

# ---------------------------------------------------------------------------------------------------------- selectors


class SWrittenKey(CrucibleModel):
    sel: Literal["written_key"] = "written_key"


class SReadKey(CrucibleModel):
    sel: Literal["read_key"] = "read_key"


class SSiblingOf(CrucibleModel):
    sel: Literal["sibling_of"] = "sibling_of"
    of: Selector = Field(default_factory=SWrittenKey)
    limit: int = 16


class SAnyWriteTool(CrucibleModel):
    sel: Literal["any_write_tool"] = "any_write_tool"
    unused: bool = False


class SReadTool(CrucibleModel):
    sel: Literal["read_tool"] = "read_tool"
    unused: bool = False


class STool(CrucibleModel):
    sel: Literal["tool"] = "tool"
    glob: str = "*"


class SObservationOf(CrucibleModel):
    sel: Literal["observation_of"] = "observation_of"
    tool: str = "*"
    path: str | None = None


class SFieldValues(CrucibleModel):
    sel: Literal["field_values"] = "field_values"
    field: str


class SConst(CrucibleModel):
    sel: Literal["const"] = "const"
    values: list[Any] = Field(default_factory=list)


Selector = Annotated[SWrittenKey | SReadKey | SSiblingOf | SAnyWriteTool | SReadTool | STool | SObservationOf | SFieldValues | SConst, Field(discriminator="sel")]
SSiblingOf.model_rebuild()


class TShift(CrucibleModel):
    tf: Literal["shift"] = "shift"
    by: float = 1


class TScale(CrucibleModel):
    tf: Literal["scale"] = "scale"
    by: float = 2


class TSuffix(CrucibleModel):
    tf: Literal["suffix"] = "suffix"
    s: str = "-x"


class TOtherSeen(CrucibleModel):
    tf: Literal["other_seen"] = "other_seen"


Transform = Annotated[TShift | TScale | TSuffix | TOtherSeen, Field(discriminator="tf")]


# ---------------------------------------------------------------------------------------------------------- bodies

class MutateArg(CrucibleModel):
    kind: Literal["mutate_arg"] = "mutate_arg"
    tool: str = "*"
    write_only: bool = True
    arg: str = "*"
    key_only: bool = False
    values: Selector | None = None
    transform: Transform | None = None


class InsertCall(CrucibleModel):
    kind: Literal["insert_call"] = "insert_call"
    tool: Selector = Field(default_factory=SAnyWriteTool)
    args: dict[str, Selector | Literal["borrow"]] = Field(default_factory=dict)
    borrow_rest: bool = True  # fill the tool's other arguments from π*'s same-named arguments
    position: Literal["start", "end", "after_writes"] = "end"


class DropCalls(CrucibleModel):
    kind: Literal["drop_calls"] = "drop_calls"
    tool: str = "*"
    write_only: bool = False


class AgentCalls(CrucibleModel):
    kind: Literal["agent"] = "agent"
    calls: list[InsertCall] = Field(default_factory=list)


Body = Annotated[MutateArg | InsertCall | DropCalls | AgentCalls, Field(discriminator="kind")]


class LabelRule(CrucibleModel):
    expect: Literal["fail", "pass"] = "fail"
    defect: str
    confirm: Literal["always", "outside_delta", "missing_delta", "never"] = "always"


class OperatorSpec(CrucibleModel):
    name: str
    version: int = 1
    body: Body
    label: LabelRule
    max_probes: int = 64
    why: str = ""  # the abstractor's description of the counterexample cluster it generalises

    @property
    def cls_name(self) -> str:
        return f"OP:{self.name}@{self.version}"


# ---------------------------------------------------------------------------------------------------------- evaluation

class _Env:
    def __init__(self, task: Task, oracle: Run, tools: list[ToolInfo]):
        self.task, self.oracle, self.tools = task, oracle, tools
        self.delta = delta_of(oracle, task.contract)
        self.written = written_entities(self.delta)
        self.idx = entity_index(oracle.start)
        self.used = {s.tool for s in task.oracle.steps}

    def entities(self, s: Any) -> list[tuple[str, str, str]]:
        if isinstance(s, SWrittenKey):
            return sorted(self.written)
        if isinstance(s, SReadKey):
            out = set()
            for c in self.oracle.calls:
                if c.read_only:
                    for v in c.args.values():
                        for app, coll in key_colls(v, c.tool, self.idx)[:1]:
                            out.add((app, coll, key_text(v) or ""))
            return sorted(out)
        if isinstance(s, SSiblingOf):
            base = self.entities(s.of)
            out: list[tuple[str, str, str]] = []
            for app, coll, key in base:
                for k in sorted(self.oracle.start.collection(app, coll) if self.oracle.start else {}):
                    e = (app, coll, k)
                    if k != key and e not in self.written and e not in out:
                        out.append(e)
            return out[: s.limit]
        return []

    def tools_of(self, s: Any) -> list[ToolInfo]:
        if isinstance(s, SAnyWriteTool):
            return [t for t in self.tools if not t.read_only and not (s.unused and t.name in self.used)]
        if isinstance(s, SReadTool):
            return [t for t in self.tools if t.read_only and not (s.unused and t.name in self.used)]
        if isinstance(s, STool):
            return [t for t in self.tools if fnmatch.fnmatchcase(t.name, s.glob)]
        return []

    def values(self, s: Any) -> list[Any]:
        if isinstance(s, SWrittenKey | SReadKey | SSiblingOf):
            return [k for _, _, k in self.entities(s)]
        if isinstance(s, SObservationOf):
            out = []
            for c in self.oracle.calls:
                if fnmatch.fnmatchcase(c.tool, s.tool) and c.obs.error is None:
                    for p, v in leaves(c.obs.value):
                        if (s.path is None or p == s.path or p.endswith("." + s.path)) and v is not None and not isinstance(v, bool | dict | list):
                            out.append(v)
            return _uniq(out)
        if isinstance(s, SFieldValues):
            return _uniq([row[s.field] for _, _, _, row in (self.oracle.start.entities() if self.oracle.start else []) if s.field in row])
        if isinstance(s, SConst):
            return list(s.values)
        return []


def _uniq(xs: Sequence[Any]) -> list[Any]:
    seen, out = set(), []
    for x in xs:
        k = cjson(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def _transform(t: Any, v: Any, name: str, env: _Env, tool: ToolInfo | None) -> list[Any]:
    if isinstance(t, TShift) and is_number(v):
        return [v + (int(t.by) if isinstance(v, int) and float(t.by).is_integer() else t.by)]
    if isinstance(t, TScale) and is_number(v):
        return [v * t.by]
    if isinstance(t, TSuffix) and isinstance(v, str):
        return [v + t.s]
    if isinstance(t, TOtherSeen):
        return enum_alternatives(name, v, tool, env.oracle, env.task.oracle)
    return []


# ---------------------------------------------------------------------------------------------------------- the compiled operator

class CompiledOperator(BaseFalsifier):
    """A DSL operator as a falsifier class (see the module docstring)."""

    name: ClassVar[str] = "OP"

    def __init__(self, spec: OperatorSpec):
        self.spec = spec
        self.name = spec.cls_name  # type: ignore[misc]

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        oracle = await ctx.oracle(task)
        tools = await (await ctx.world(task)).tools()
        env = _Env(task, oracle, tools)
        progs = self._programs(env)
        probes = [Probe(cls=self.name, op=self.spec.name, site=site, program=prog, data={"agent": agent}) for site, prog, agent in progs]
        return dedupe_programs(probes, {task.oracle.digest()})[: self.spec.max_probes]

    def _programs(self, env: _Env) -> list[tuple[str, Program, bool]]:
        b = self.spec.body
        p = env.task.oracle
        tinfo = {t.name: t for t in env.tools}
        ro = {t.name: t.read_only for t in env.tools}
        out: list[tuple[str, Program, bool]] = []
        if isinstance(b, MutateArg):
            for i, s in enumerate(p.steps):
                if not fnmatch.fnmatchcase(s.tool, b.tool) or (b.write_only and ro.get(s.tool, False)):
                    continue
                rec = env.oracle.call(s.id)
                if rec is None:
                    continue
                for name in sorted(s.args):
                    if not fnmatch.fnmatchcase(name, b.arg) or name not in rec.args:
                        continue
                    v = rec.args[name]
                    if b.key_only and not key_colls(v, s.tool, env.idx):
                        continue
                    cands = [typed_like(str(x), v) if isinstance(x, str) and key_text(v) is not None else x for x in env.values(b.values)] if b.values else []
                    if b.transform is not None:
                        cands += _transform(b.transform, v, name, env, tinfo.get(s.tool))
                    for x in _uniq(cands):
                        if cjson(x) == cjson(v):
                            continue
                        steps = [st.model_copy(deep=True) for st in p.steps]
                        steps[i].args[name] = Lit(v=x)
                        out.append((f"{s.id}.{name}={cjson(x)}", p.model_copy(update={"steps": steps}, deep=True), False))
        elif isinstance(b, DropCalls):
            for i, s in enumerate(p.steps):
                if fnmatch.fnmatchcase(s.tool, b.tool) and not (b.write_only and ro.get(s.tool, False)):
                    out.append((f"drop:{s.id}", p.model_copy(update={"steps": p.steps[:i] + p.steps[i + 1:]}, deep=True), False))
        elif isinstance(b, InsertCall):
            for site, step in self._calls(b, env, "x"):
                if b.position == "start":
                    steps = [step, *p.steps]
                    out.append((f"start:{site}", p.model_copy(update={"steps": steps}, deep=True), False))
                elif b.position == "end":
                    out.append((f"end:{site}", p.model_copy(update={"steps": [*p.steps, step]}, deep=True), False))
                else:
                    for i, s in enumerate(p.steps):
                        if not ro.get(s.tool, False):
                            st = step.model_copy(update={"id": f"{step.id}_{i}"})
                            out.append((f"after:{s.id}:{site}", p.model_copy(update={"steps": [*p.steps[:i + 1], st, *p.steps[i + 1:]]}, deep=True), False))
        elif isinstance(b, AgentCalls):
            steps: list[Step] = []
            for j, c in enumerate(b.calls):
                calls = self._calls(c, env, f"a{j}")
                if calls:
                    steps.append(calls[0][1])
            if steps:
                out.append(("agent", Program(steps=steps), True))
        return out

    def _calls(self, b: InsertCall, env: _Env, prefix: str) -> list[tuple[str, Step]]:
        base = borrowed_args(env.oracle)
        out = []
        for t in env.tools_of(b.tool):
            props = t.input_schema.get("properties") or {}
            req = t.input_schema.get("required") or []
            choices: dict[str, list[Any]] = {}
            for name in props:
                sel = b.args.get(name)
                if sel == "borrow" or (sel is None and b.borrow_rest):
                    if name in base:
                        choices[name] = [base[name]]
                elif sel is not None:
                    vs = env.values(sel)
                    if vs:
                        like = base.get(name)
                        choices[name] = [typed_like(str(x), like) if like is not None and isinstance(x, str) else x for x in vs]
            if any(r not in choices for r in req):
                continue
            names = sorted(choices)
            for n, combo in enumerate(itertools.islice(itertools.product(*(choices[k] for k in names)), self.spec.max_probes)):
                args = dict(zip(names, combo, strict=True))
                out.append((f"{t.name}{cjson(args)}", Step(id=f"{prefix}{n}", tool=t.name, args={k: Lit(v=v) for k, v in args.items()})))
        return out

    finalize = MUT.finalize  # one trial per normalised effect, as in MUT

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        assert probe.program is not None
        oracle = await ctx.oracle(task)
        c = task.contract
        r = await ctx.run(task, probe.program)
        v = ctx.grade(r, c)
        d, d0 = delta_of(r, c), await ctx.oracle_delta(task, c)
        got, want = {str(x.addr) for x in d}, {str(x.addr) for x in d0}
        return Outcome(passed=v.passed, reasons=v.reasons, equivalent=equivalent(r, oracle, c, task.instruction, d_run=d, d_oracle=d0), error=r.error,
                       data={"outside": sorted(got - want), "missing": sorted(want - got), "effect": effect_signature(d, r.answer, v)})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        rule = self.spec.label
        if not probe.data.get("agent") and outcome.equivalent:
            return ("equivalent", None) if outcome.passed else ("confirmed", "D6")
        violated = bool(outcome.passed) if rule.expect == "fail" else not outcome.passed
        if not violated:
            return "clean", None
        ok = {"always": True, "outside_delta": bool(outcome.data.get("outside")), "missing_delta": bool(outcome.data.get("missing")), "never": False}[rule.confirm]
        return ("confirmed" if ok else "suspect"), rule.defect


def compile_operator(spec: OperatorSpec) -> CompiledOperator:
    return CompiledOperator(OperatorSpec.model_validate(spec.model_dump(mode="json")))  # the JSON round trip is the DSL's contract


# ---------------------------------------------------------------------------------------------------------- birth gate

class GateReport(CrucibleModel):
    op: str
    passed: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    detail: dict[str, Any] = Field(default_factory=dict)


def _src(x: Plant | Task) -> tuple[Task, str, str]:
    return (x.task, x.source_id, x.domain) if isinstance(x, Plant) else (x, x.id, x.domain)


async def birth_gate(spec: OperatorSpec, *, sources: Sequence[Plant | Task], fresh: Sequence[Plant], anchors: Sequence[Task], ctx: FalsifyContext,
                     cost_bar: float, min_hits: int = 1) -> GateReport:
    """§9.5's birth gate for one operator (see the module docstring). ``fresh`` is a newly planted dev view; only its
    plants from tasks *and* domains other than the sources' count towards transfer."""
    op = compile_operator(spec)
    rep: dict[str, Any] = {}
    probes = 0
    trials = 0
    repro = []
    src_ids, src_domains = set(), set()
    for s in sources:
        task, sid, dom = _src(s)
        src_ids.add(sid)
        src_domains.add(dom)
        r = await falsify(task, op, ctx)
        probes, trials = probes + len(r.findings), trials + 1
        repro.append(r.c > 0)
    rep["reproduced"] = repro
    others = [p for p in fresh if p.source_id not in src_ids and p.domain not in src_domains]
    hits = []
    for p in others:
        r = await falsify(p.task, op, ctx)
        probes, trials = probes + len(r.findings), trials + 1
        if r.c:
            hits.append(p.id)
    rep["transfer_hits"] = hits
    rep["transfer_domains"] = sorted({p.domain for p in others if p.id in hits})
    alarms = []
    for a in anchors:
        r = await falsify(a, op, ctx)
        probes, trials = probes + len(r.findings), trials + 1
        if r.c:
            alarms.append(a.id)
    rep["false_alarms"] = alarms
    rep["probes_per_task"] = probes / trials if trials else 0.0
    checks = {"reproduces_sources": bool(repro) and all(repro), "finds_on_other_domains": len(hits) >= min_hits, "zero_false_alarms": not alarms,
              "under_cost_bar": rep["probes_per_task"] <= cost_bar}
    return GateReport(op=spec.cls_name, passed=all(checks.values()), checks=checks, detail=rep)


# ---------------------------------------------------------------------------------------------------------- library, fictitious play, retirement

class Library(CrucibleModel):
    """The F component's operator library as data (versioned by ``update.ThetaStore``)."""

    ops: dict[str, OperatorSpec] = Field(default_factory=dict)
    born: dict[str, int] = Field(default_factory=dict)
    last_unique: dict[str, int] = Field(default_factory=dict)
    retired: list[str] = Field(default_factory=list)

    def add(self, spec: OperatorSpec, round: int) -> Library:
        out = self.model_copy(deep=True)
        out.ops[spec.cls_name] = spec
        out.born[spec.cls_name] = round
        out.last_unique[spec.cls_name] = round
        return out

    def falsifiers(self) -> list[CompiledOperator]:
        return [compile_operator(s) for _, s in sorted(self.ops.items())]


async def _hits(plants: Sequence[Plant], falsifiers: Sequence[BaseFalsifier], ctx: FalsifyContext) -> dict[str, set[str]]:
    """Plant id sets caught (confirmed) by each class, in the scheduled order."""
    out: dict[str, set[str]] = {f.name: set() for f in falsifiers}
    for p in plants:
        for r in await run_classes(p.task, falsifiers, ctx, cache=True):
            if r.applicable and r.c:
                out[r.cls].add(p.id)
    return out


async def recall_on(plants: Sequence[Plant], falsifiers: Sequence[BaseFalsifier], ctx: FalsifyContext) -> float:
    hits = await _hits(plants, falsifiers, ctx)
    caught = set().union(*hits.values()) if hits else set()
    return len(caught) / len(plants) if plants else 0.0


class FictitiousPlay(CrucibleModel):
    ok: bool
    recall_new: float
    recall_old: float
    per_planter: dict[str, tuple[float, float]] = Field(default_factory=dict)  # planter -> (old, new)


async def fictitious_play(new: Sequence[BaseFalsifier], old: Sequence[BaseFalsifier], union: Sequence[Plant], ctx: FalsifyContext, band: float = 0.0) -> FictitiousPlay:
    """F must keep recall on the union of all past planters' sets, and on each planter's own set, within ``band``."""
    by: dict[str, list[Plant]] = {}
    for p in union:
        by.setdefault(p.planter or "?", []).append(p)
    per = {}
    ok = True
    for k, ps in sorted(by.items()):
        o, n = await recall_on(ps, old, ctx), await recall_on(ps, new, ctx)
        per[k] = (o, n)
        ok = ok and n >= o - band
    ro, rn = await recall_on(union, old, ctx), await recall_on(union, new, ctx)
    return FictitiousPlay(ok=ok and rn >= ro - band, recall_new=rn, recall_old=ro, per_planter=per)


async def retire(lib: Library, *, base: Sequence[BaseFalsifier], union: Sequence[Plant], ctx: FalsifyContext, round: int, window: int) -> tuple[Library, list[str]]:
    """Operators with no unique hit on the union (nothing the base population and the other operators miss) for
    ``window`` rounds are retired, if fictitious play holds without them. Returns the new library and the retired names."""
    ops = lib.falsifiers()
    hits = await _hits(union, [*base, *ops], ctx)
    out = lib.model_copy(deep=True)
    for op in ops:
        others = set().union(*(h for k, h in hits.items() if k != op.name))
        if hits[op.name] - others:
            out.last_unique[op.name] = round
    stale = [k for k in sorted(out.ops) if round - out.last_unique.get(k, out.born.get(k, round)) >= window]
    if not stale:
        return out, []
    keep = [o for o in ops if o.name not in stale]
    fp = await fictitious_play([*base, *keep], [*base, *ops], union, ctx)
    if not fp.ok:
        return out, []
    for k in stale:
        out.ops.pop(k)
        out.retired.append(k)
    return out, stale


# ---------------------------------------------------------------------------------------------------------- the planter loop

class PlanterDecision(CrucibleModel):
    planter: str
    accepted: bool
    checks: dict[str, bool] = Field(default_factory=dict)
    mmd: float | None = None
    miss_rate: float | None = None  # the planter's reward: the population's misses on its plants
    kept: int = 0
    discarded: int = 0
    leaks: list[str] = Field(default_factory=list)


PlanterCritic = Callable[[Planter], Sequence[str] | Awaitable[Sequence[str]]]


async def accept_planter(candidate: Planter, tasks: Sequence[Task], ctx: FalsifyContext, *, reference: Sequence[dict[str, float]], mu: float,
                         falsifiers: Sequence[BaseFalsifier], critic: PlanterCritic | None = None, gamma: float | None = None) -> tuple[PlanterDecision, list[Plant]]:
    """Plant with the candidate, then: every plant's witness re-validates by execution; MMD(features, reference) ≤ μ;
    the critic finds no leakage. Returns the decision and the plants (which join the union when accepted)."""
    batch = await plant_all(tasks, candidate, ctx)
    srcs = {t.id: t for t in tasks}
    valid = [(await validate(p, srcs[p.source_id], ctx))[0] for p in batch.kept]
    leaks: list[str] = []
    if critic is not None:
        res = critic(candidate)
        leaks = list(await res) if inspect.isawaitable(res) else list(res)  # type: ignore[arg-type]
    m = mmd(feature_matrix(p.features for p in batch.kept), feature_matrix(reference), gamma) if batch.kept and reference else None
    hits = await _hits(batch.kept, falsifiers, ctx) if batch.kept else {}
    caught = set().union(*hits.values()) if hits else set()
    miss = 1 - len(caught) / len(batch.kept) if batch.kept else None
    checks = {"witnessed": bool(batch.kept) and all(valid), "natural_like": m is not None and m <= mu, "no_leakage": not leaks}
    dec = PlanterDecision(planter=candidate.key, accepted=all(checks.values()), checks=checks, mmd=m, miss_rate=miss, kept=len(batch.kept),
                          discarded=len(batch.discarded), leaks=leaks)
    return dec, batch.kept
