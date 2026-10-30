"""P3 on-intent probes beyond TAINT and TWIN (part2-paper-core §0 P3, §10): success must require the skill.

Two executed probe classes, both ``BaseFalsifier`` s so ``run_classes`` schedules them like any other. A confirmed
finding means the task is **off intent**; neither class feeds certificates or recall (P3 is not one of the eleven
defects), and their findings are never archived as counterexamples.

**EVID: evidence removal.** The derived values of π* are the scalars it writes or answers (and the intent card's
``derive`` values) that neither ι nor ``given`` states. Each is traced to the observation π* read it from, that
observation's object is matched to its start-state row (the row agreeing with it on the most fields), and the cell is
the **evidence site**: a record field, or a span inside a long text field (a policy or KB document). The probe removes
it (a ``set`` perturbation: a string becomes ``""``, other scalars ``None``; a document span becomes ``[removed]``):
- **oracle**: π* with the literals that carry the value replaced by the same blank runs on the removed World; it must
  fail C. A pass is **confirmed off intent**: C does not depend on the evidence (the value is not checked, or C passes
  without it).
- **ALT** (injected agent runners, never the student): each runs on the removed World; a pass is a **suspect** (the
  information reached it another way, or a guess was enough), for a judge to decide.
- **Control**: the same blank applied to a decoy (the same field of an untouched sibling row, else another field of the
  row that no candidate value, pattern or key uses); π* must still pass on the decoy World, otherwise the World does
  not tolerate blanks and the probe is invalid (not a trial).
Needs a ``Perturbable`` World (``iface.Perturbable``); without one the class is not applicable and says so.

**SWAP: instruction swap.** Success must follow the instruction:
- **enum**: a literal of π* that ι states and the tool catalogue enumerates (a decision vocabulary: hold vs reject) is
  swapped for another value of the same enum through ι and π*; π*′ must run (a call that newly errors makes the twin
  dead, not a trial) and change the effect; C must reject π*′. A pass is **confirmed D9**: the contract ignores the
  choice the instruction states.
- **pool**: every other task of the pool on the same World and start state: C must reject that task's oracle run under
  its own instruction. The same normalised effect is no trial (the two tasks are duplicates; P4 reports it); a pass is
  **confirmed D9**.

``p3(task, ctx)`` runs TAINT (its ``derive`` probes: D7), TWIN (D9), EVID and SWAP, and says ``on_intent`` = True when
at least one valid EVID, SWAP or TWIN trial ran and nothing is confirmed, False on any confirmed finding, None
(unmeasured) when no probe applies. ``pool_report`` gives the on-intent share per pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, ClassVar

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import CanonState, cjson
from crucible.synth.falsify.agents import AgentSpec, AgentView, RecordingWorld
from crucible.synth.falsify.base import BaseFalsifier, ClassResult, FalsifyContext, Label, Outcome, Probe, delta_of, run_classes
from crucible.synth.falsify.taint import TAINT
from crucible.synth.falsify.twin import TWIN, map_program, rewrite_instruction, substitute
from crucible.synth.iface import Perturbation
from crucible.synth.program import (
    Fn,
    Lit,
    Program,
    Ref,
    Run,
    _loose_eq,
    is_structural,
    program_locs,
    scalar_leaves,
    text_spans,
    value_paths,
    walk_args,
)
from crucible.synth.task import Task

REMOVED = "[removed]"
LONG_TEXT = 60  # a text leaf longer than this is a document: remove the span, not the field


def blank(v: Any) -> Any:
    if isinstance(v, str):
        return ""
    return None


def _given(v: Any, task: Task) -> bool:
    if is_structural(v):
        return True
    if text_spans(v, task.instruction):
        return True
    return any(_loose_eq(v, g) or (isinstance(g, str) and text_spans(v, g, min_len=3)) for g in task.intent.given)


def _at(obj: Any, path: str) -> tuple[Any, Any, str | None]:
    """(parent object, leaf, last segment) of a dot path inside an observation value."""
    cur, parent, last = obj, None, None
    for seg in [s for s in path.split(".") if s != ""]:
        parent, last = cur, seg
        if isinstance(cur, dict):
            cur = cur.get(seg)
        elif isinstance(cur, list):
            try:
                cur = cur[int(seg)]
            except (ValueError, IndexError):
                return None, None, None
        else:
            return None, None, None
    return parent, cur, last


def match_row(obj: dict[str, Any], s0: CanonState, field: str) -> tuple[str, str, str] | None:
    """The start-state row an observation object shows: the row agreeing with it on the most fields (at least two,
    the leaf's field among them when the row has it)."""
    best, score = None, 1
    for app, coll, key, row in s0.entities():
        if field not in row:
            continue
        sc = sum(1 for f, v in obj.items() if f in row and not isinstance(v, dict | list) and _loose_eq(row[f], v))
        if sc > score:
            best, score = (app, coll, key), sc
    return best


class Site(CrucibleModel):
    value: Any
    app: str
    coll: str
    key: str
    field: str
    kind: str = "field"  # field | doc
    via: str = ""  # where π* read it: "<step>.<path>"
    used: list[str] = Field(default_factory=list)  # where π* uses it ("<step>.<arg>", "answer", "derive")


class _Use(CrucibleModel):
    value: Any
    where: str
    upto: int  # calls before this index may hold the evidence
    exact: list[str] = Field(default_factory=list)  # "<step>.<path>": the observation the value was read from (a direct ref or a literal's source)
    steps: list[str] = Field(default_factory=list)  # the steps a fn's refs read: search only their observations
    direct: bool = False  # a direct ref: its parent object is the only evidence (an aggregate has none)


def _arg_use(a: Any, v: Any, where: str, upto: int, oracle: Run) -> _Use | None:
    if isinstance(a, Ref):
        return _Use(value=v, where=where, upto=upto, exact=[a.path], direct=True)
    if isinstance(a, Fn):
        steps = sorted({x.path.split(".", 1)[0] for _, x in walk_args(a, where) if isinstance(x, Ref)})
        return _Use(value=v, where=where, upto=upto, steps=steps) if steps else None
    if isinstance(a, Lit):
        ent = next((e for e in oracle.taint if e.loc == where), None)
        if ent is None:
            return _Use(value=v, where=where, upto=upto)
        obs = [f"{s.step}.{s.path}" for s in ent.sources if s.kind == "observation" and s.step]
        return _Use(value=v, where=where, upto=upto, exact=obs) if obs else None
    return None


def _uses(task: Task, oracle: Run) -> list[_Use]:
    """Every derived value π* writes or answers, with where it was read (module docstring)."""
    out: list[_Use] = []
    by_id = {s.id: s for s in task.oracle.steps}
    for i, c in enumerate(oracle.calls):
        if c.read_only or c.step not in by_id:
            continue
        for name, a in by_id[c.step].args.items():
            v = c.args.get(name)
            if isinstance(v, dict | list):
                for loc, x in scalar_leaves(v, f"{c.step}.{name}"):
                    out.append(_Use(value=x, where=loc, upto=i))
                continue
            u = _arg_use(a, v, f"{c.step}.{name}", i, oracle)
            if u is not None:
                out.append(u)
    n = len(oracle.calls)
    if oracle.answer is not None and task.oracle.answer is not None:
        if isinstance(oracle.answer, dict | list):
            out += [_Use(value=x, where="answer", upto=n) for _, x in scalar_leaves(oracle.answer, "answer")]
        else:
            u = _arg_use(task.oracle.answer, oracle.answer, "answer", n, oracle)
            if u is not None:
                out.append(u)
    out += [_Use(value=v, where="derive", upto=n) for v in task.intent.derive if not isinstance(v, dict | list)]
    return out


def _site_at(c_obs: Any, p: str, v: Any, s0: CanonState, via: str) -> Site | None:
    parent, leaf, last = _at(c_obs, p)
    if not isinstance(parent, dict) or last is None:
        return None
    if isinstance(leaf, str) and len(leaf) > LONG_TEXT and not _loose_eq(leaf, v):
        row = match_row(parent, s0, last)  # a document: the value is a span of a long text
        return Site(value=v, app=row[0], coll=row[1], key=row[2], field=last, kind="doc", via=via) if row else None
    row = match_row(parent, s0, last)
    if row is None or str(row[2]) == str(v):  # no row (an aggregate) or the entity's own key (a lookup: TWIN's territory)
        return None
    return Site(value=v, app=row[0], coll=row[1], key=row[2], field=last, via=via)


def evidence_sites(task: Task, oracle: Run, *, limit: int = 8) -> list[Site]:
    """The derived values' evidence sites (module docstring)."""
    s0 = oracle.start
    if s0 is None:
        return []
    sites: dict[tuple[str, str, str, str], Site] = {}
    obs = {c.step: c for c in oracle.calls}
    order = [c.step for c in oracle.calls]
    for u in _uses(task, oracle):
        v, where = u.value, u.where
        if _given(v, task) or (isinstance(v, str) and len(v) < 2):
            continue
        found = None
        if u.exact:
            for ep in u.exact:
                sid, _, path = ep.partition(".")
                c = obs.get(sid)
                if c is not None and c.read_only and c.obs.value is not None:
                    found = _site_at(c.obs.value, path, v, s0, ep)
                    if found is not None:
                        break
        elif not u.direct:
            steps = u.steps or [s for s in order[: u.upto]]
            for sid in steps:
                c = obs.get(sid)
                if c is None or not c.read_only or c.obs.value is None:
                    continue
                for p in value_paths(v, c.obs.value)[:3]:
                    found = _site_at(c.obs.value, p, v, s0, f"{sid}.{p}")
                    if found is not None:
                        break
                if found is not None:
                    break
        if found is None:
            continue
        k = (found.app, found.coll, found.key, found.field)
        if k in sites:
            if where not in sites[k].used:
                sites[k].used.append(where)
            continue
        found.used = [where]
        sites[k] = found
        if len(sites) >= limit:
            break
    return list(sites.values())


def decision_reads(task: Task, oracle: Run, *, limit: int = 4) -> list[Site]:
    """Records π* reads but never refers to: the evidence of a decision the oracle's author resolved offline (a
    straight-line π* does not depend on them, so only an agent probe can test them). One ``record`` site per row."""
    s0 = oracle.start
    if s0 is None:
        return []
    used = {x.path.split(".", 1)[0] for s in task.oracle.steps for name, a in s.args.items() for _, x in walk_args(a, f"{s.id}.{name}")
            if isinstance(x, Ref)}
    if task.oracle.answer is not None:
        used |= {x.path.split(".", 1)[0] for _, x in walk_args(task.oracle.answer, "answer") if isinstance(x, Ref)}
    out: list[Site] = []
    for c in oracle.calls:
        if not c.read_only or c.step in used or not isinstance(c.obs.value, dict):
            continue
        objs = [c.obs.value] + [x for v in c.obs.value.values() if isinstance(v, list) for x in v if isinstance(x, dict)][:3]
        for o in objs:
            f = next((k for k in o if not isinstance(o[k], dict | list)), None)
            row = match_row(o, s0, f) if f else None
            if row is not None and not any((s.app, s.coll, s.key) == row for s in out):
                out.append(Site(value=None, app=row[0], coll=row[1], key=row[2], field="*", kind="record", via=c.step, used=["decision"]))
        if len(out) >= limit:
            break
    return out[:limit]


def _removal(site: Site, s0: CanonState) -> Perturbation:
    row = s0.collection(site.app, site.coll).get(site.key) or {}
    if site.kind == "record":
        return Perturbation(op="set", app=site.app, coll=site.coll, key=site.key,
                            fields={f: blank(v) for f, v in row.items() if not isinstance(v, dict | list) and str(v) != str(site.key) and f not in ("id", "key")})
    cur = row.get(site.field)
    if site.kind == "doc" and isinstance(cur, str):
        spans = text_spans(site.value, cur, min_len=1)
        out, last = [], 0
        for a, b in sorted(spans):
            out += [cur[last:a], REMOVED]
            last = b
        new: Any = "".join([*out, cur[last:]])
    else:
        new = blank(cur)
    return Perturbation(op="set", app=site.app, coll=site.coll, key=site.key, fields={site.field: new})


def _decoy(site: Site, s0: CanonState, avoid: set[str], touched: set[tuple[str, str, str]], pattern_fields: set[str]) -> Perturbation | None:
    rows = s0.collection(site.app, site.coll)
    if site.kind == "record":
        for k2, _row in sorted(rows.items()):
            if k2 != site.key and (site.app, site.coll, k2) not in touched and k2 not in avoid:
                return _removal(site.model_copy(update={"key": k2}), s0)
        return None
    cur = (rows.get(site.key) or {}).get(site.field)
    for k2, row in sorted(rows.items()):
        if k2 == site.key or (site.app, site.coll, k2) in touched or site.field not in row:
            continue
        v2 = row[site.field]
        if is_structural(v2) or cjson(v2) in avoid or type(v2) is not type(cur):
            continue
        if site.kind == "doc":
            return Perturbation(op="set", app=site.app, coll=site.coll, key=k2, fields={site.field: (v2 + " " + REMOVED) if isinstance(v2, str) else v2})
        return Perturbation(op="set", app=site.app, coll=site.coll, key=k2, fields={site.field: blank(v2)})
    row = rows.get(site.key) or {}
    for f2, v2 in sorted(row.items()):
        if f2 == site.field or f2 in pattern_fields or is_structural(v2) or cjson(v2) in avoid or str(v2) == str(site.key) or f2 in ("id", "key"):
            continue
        if isinstance(v2, dict | list):
            continue
        return Perturbation(op="set", app=site.app, coll=site.coll, key=site.key, fields={f2: blank(v2)})
    return None


def blind_program(p: Program, value: Any, task: Task, oracle: Run) -> Program:
    """π* with every literal equal to ``value`` that ι and ``given`` do not state replaced by the blank of its type:
    the oracle without the evidence (it cannot know the value)."""
    q = p.model_copy(deep=True)
    for _loc, a in program_locs(q):
        if isinstance(a, Lit) and not isinstance(a.v, dict | list) and _loose_eq(a.v, value) and not _given(a.v, task):
            a.v = blank(a.v)
    return q


async def run_agent_on(agent: AgentSpec, task: Task, ctx: FalsifyContext, perturb: Sequence[Perturbation]) -> Run:
    """``falsify.agents.run_agent`` on an edited start state."""
    w = await ctx.world(task)
    await w.reset(task.seed)
    for p in perturb:
        await w.perturb(p)  # type: ignore[attr-defined]
    tools = await w.tools()
    start = await w.canon()
    rec = RecordingWorld(w, tools)
    err, answer = None, None
    try:
        answer = await agent.runner(AgentView(instruction=task.instruction, tools=tools), rec)  # type: ignore[arg-type]
    except Exception as e:  # noqa: BLE001 - an agent failure is no attempt
        err = f"agent: {type(e).__name__}"
    return Run(instruction=task.instruction, seed=task.seed, calls=rec.calls, answer=answer, start=start, end=await w.canon(),
               read_only={t.name: t.read_only for t in tools}, error=err)


class EVID(BaseFalsifier):
    name: ClassVar[str] = "EVID"

    def __init__(self, agents: Sequence[AgentSpec] = (), *, limit: int = 8):
        self.agents, self.limit = list(agents), limit

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        w = await ctx.world(task)
        if not callable(getattr(w, "perturb", None)):
            return "the World cannot edit its start state (no perturb)"
        o = await ctx.oracle(task)
        if o.start is None:
            return "no canonical state"
        if not ctx.grade(o, task.contract).passed:
            return "the oracle fails its own contract"
        if not evidence_sites(task, o, limit=1) and not (self.agents and decision_reads(task, o, limit=1)):
            return "no derived value with an evidence site (every value π* uses is stated, a key lookup or an aggregate)"
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        o = await ctx.oracle(task)
        s0 = o.start
        assert s0 is not None
        sites = evidence_sites(task, o, limit=self.limit)
        records = decision_reads(task, o) if self.agents else []
        avoid = {cjson(s.value) for s in sites}
        touched = {(a, c, k) for a, c, k in {ch.addr.entity for ch in delta_of(o, task.contract)}}
        pfields = {p.field for p in task.contract.required} | {f for p in task.contract.required for f in p.where}
        out = []
        for s in [*sites, *records]:
            rm = _removal(s, s0)
            dec = _decoy(s, s0, avoid | {s.key for s in records}, touched, pfields)
            base = f"{s.app}.{s.coll}[{s.key}].{s.field}"
            data = {"site": s.model_dump(mode="json"), "decoy": dec.model_dump(mode="json") if dec else None}
            if s.kind != "record":
                prog = blind_program(task.oracle, s.value, task, o)
                out.append(Probe(cls="EVID", op="oracle", site=f"evidence:{base}", program=prog, perturb=[rm], expect="property", data=data))
            for a in self.agents:
                out.append(Probe(cls="EVID", op="agent", site=f"evidence:{base}:{a.name}", perturb=[rm], expect="property",
                                 data={**data, "agent": a.name, "family": a.family}))
        return out

    async def _control(self, task: Task, ctx: FalsifyContext, decoy: dict[str, Any] | None) -> tuple[bool, str]:
        if decoy is None:
            return False, "no decoy field to check that the World tolerates a blank"
        try:
            r = await ctx.run(task, task.oracle, perturb=[Perturbation.model_validate(decoy)])
        except (KeyError, ValueError, TypeError) as e:
            return False, f"decoy perturbation refused: {type(e).__name__}"
        if not ctx.grade(r, task.contract).passed:
            return False, "π* fails on the decoy-blanked World: the World does not tolerate blanks"
        return True, ""

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        ok, why = await self._control(task, ctx, probe.data.get("decoy"))
        if not ok:
            return Outcome(valid=False, error=why, data={"control": why})
        try:
            if probe.op == "oracle":
                assert probe.program is not None
                r = await ctx.run(task, probe.program, perturb=probe.perturb)
                v = ctx.grade(r, task.contract)
                return Outcome(passed=v.passed, reasons=v.reasons, error=r.error, data={"site": probe.data["site"]})
            agent = next(a for a in self.agents if a.name == probe.data["agent"])
            r = await run_agent_on(agent, task, ctx, probe.perturb)
        except (KeyError, ValueError, TypeError) as e:
            return Outcome(valid=False, error=f"perturb: {type(e).__name__}")
        if r.error is not None and r.error.startswith("agent:"):
            return Outcome(valid=False, error=r.error)
        v = ctx.grade(r, task.contract)
        return Outcome(passed=v.passed, reasons=v.reasons, error=r.error, data={"site": probe.data["site"], "family": probe.data.get("family"),
                                                                                 "calls": len(r.calls)})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if not outcome.passed:
            return "clean", None
        return ("confirmed", None) if probe.op == "oracle" else ("suspect", None)


def _enum_alternatives(v: Any, tools: Sequence[Any]) -> list[Any]:
    for t in tools:
        for p in (t.input_schema.get("properties") or {}).values():
            if isinstance(p, dict) and any(_loose_eq(v, e) for e in p.get("enum", [])):
                return [e for e in p["enum"] if not _loose_eq(v, e) and not is_structural(e)]
    return []


class SWAP(BaseFalsifier):
    name: ClassVar[str] = "SWAP"

    def __init__(self, pool: Sequence[Task] = (), *, per_literal: int = 2):
        self.pool, self.per_literal = list(pool), per_literal

    def _peers(self, task: Task) -> list[Task]:
        wk = task.world.model_dump_json()
        return [t for t in self.pool if t.id != task.id and t.world.model_dump_json() == wk and t.seed == task.seed and t.instruction != task.instruction]

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        o = await ctx.oracle(task)
        if not ctx.grade(o, task.contract).passed:
            return "the oracle fails its own contract"
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        o = await ctx.oracle(task)
        tools = await (await ctx.world(task)).tools()
        out: list[Probe] = []
        seen: set[str] = set()
        for e in o.taint:
            v = e.value
            if not (e.has("instruction") and e.has("tool_spec")) or e.has("structural") or cjson(v) in seen:
                continue
            seen.add(cjson(v))
            for alt in _enum_alternatives(v, tools)[: self.per_literal]:
                sigma = [(v, alt)]
                ins2 = rewrite_instruction(task.instruction, sigma)
                if ins2 == task.instruction:
                    continue
                prog2 = map_program(task.oracle, lambda x, sg=sigma: substitute(x, sg))
                out.append(Probe(cls="SWAP", op="enum", site=f"enum:{cjson(v)}->{cjson(alt)}", program=prog2, instruction=ins2, expect="property",
                                 data={"literal": v, "swap": alt}))
        for t in self._peers(task):
            out.append(Probe(cls="SWAP", op="pool", site=f"pool:{t.id}@{t.version}", program=t.oracle, instruction=t.instruction, expect="property",
                             data={"peer": t.id}))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        assert probe.program is not None
        o = await ctx.oracle(task)
        r = await ctx.run(task, probe.program, instruction=probe.instruction)
        if probe.op == "enum":
            before = {c.step for c in o.calls if c.obs.error is not None}
            new_err = [c.step for c in r.calls if c.obs.error is not None and c.step not in before]
            if r.error is not None or new_err or len(r.calls) < len(o.calls):
                return Outcome(valid=False, error=r.error, data={"dead": True, "errored": new_err})
        d, d0 = delta_of(r, task.contract), await ctx.oracle_delta(task)
        from crucible.synth.canon import same_diff
        from crucible.synth.contract import answer_equal

        same_answer = (r.answer is None and o.answer is None) or (r.answer is not None and o.answer is not None and answer_equal(r.answer, o.answer))
        if same_diff(d, d0) and same_answer:
            return Outcome(valid=False, data={"same_effect": True, **({"duplicate_of": probe.data["peer"]} if probe.op == "pool" else {})})
        v = ctx.grade(r, task.contract)
        return Outcome(passed=v.passed, reasons=v.reasons, error=r.error, data=dict(probe.data))

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", "D9") if outcome.passed else ("clean", None)


# ---------------------------------------------------------------------------------------------------------- the P3 verdict

class P3Result(CrucibleModel):
    task_id: str
    version: int
    on_intent: bool | None
    reasons: list[str] = Field(default_factory=list)
    trials: dict[str, int] = Field(default_factory=dict)
    confirmed: dict[str, int] = Field(default_factory=dict)
    suspects: dict[str, int] = Field(default_factory=dict)
    applicable: dict[str, bool] = Field(default_factory=dict)
    not_applicable: dict[str, str] = Field(default_factory=dict)
    duplicates: list[str] = Field(default_factory=list)  # pool peers with the same effect (SWAP pool: for P4)
    findings: list[dict[str, Any]] = Field(default_factory=list)


def _count(r: ClassResult, *, defect: str | None = None) -> tuple[int, int, int]:
    if r.cls == "TAINT":
        fs = [f for f in r.findings if f.op == "derive" and f.outcome.valid]
        return len(fs), sum(1 for f in fs if f.label == "confirmed"), 0
    return r.n, sum(1 for f in r.confirmed if defect is None or f.defect == defect), len(r.suspects)


async def p3(task: Task, ctx: FalsifyContext, *, pool: Sequence[Task] = (), agents: Sequence[AgentSpec] = (),
             classes: Sequence[BaseFalsifier] | None = None) -> P3Result:
    """The on-intent verdict of one task (module docstring)."""
    fs = list(classes) if classes is not None else [TAINT(), TWIN(), EVID(agents), SWAP(pool)]
    res = await run_classes(task, fs, ctx)
    out = P3Result(task_id=task.id, version=task.version, on_intent=None)
    for r in res:
        out.applicable[r.cls] = r.applicable
        if not r.applicable:
            out.not_applicable[r.cls] = r.reason or ""
            continue
        n, c, s = _count(r, defect="D9" if r.cls == "TWIN" else None)
        out.trials[r.cls], out.confirmed[r.cls], out.suspects[r.cls] = n, c, s
        for f in r.findings:
            if f.outcome.data.get("duplicate_of"):
                out.duplicates.append(str(f.outcome.data["duplicate_of"]))
            if f.outcome.valid and f.label in ("confirmed", "suspect") and (r.cls != "TAINT" or f.op == "derive"):
                out.findings.append({"cls": f.cls, "op": f.op, "site": f.site, "label": f.label, "defect": f.defect, "reasons": f.outcome.reasons[:6]})
                if f.label == "confirmed":
                    out.reasons.append(f"{f.cls}:{f.op}:{f.site}")
    probes = sum(out.trials.get(k, 0) for k in ("EVID", "SWAP", "TWIN"))
    if any(out.confirmed.values()):
        out.on_intent = False
    elif probes > 0:
        out.on_intent = True
    else:
        out.reasons.append("unmeasured: no valid EVID, SWAP or TWIN trial")
    return out


def pool_report(results: Sequence[P3Result]) -> dict[str, Any]:
    """The on-intent share of a pool, with the per-class trial and confirmed counts."""
    measured = [r for r in results if r.on_intent is not None]
    on = sum(1 for r in measured if r.on_intent)
    per: dict[str, dict[str, int]] = {}
    for r in results:
        for k in set(r.trials) | set(r.not_applicable):
            d = per.setdefault(k, {"tasks_applicable": 0, "tasks_not_applicable": 0, "trials": 0, "confirmed": 0, "suspects": 0, "tasks_confirmed": 0})
            if k in r.not_applicable:
                d["tasks_not_applicable"] += 1
                continue
            d["tasks_applicable"] += 1
            d["trials"] += r.trials.get(k, 0)
            d["confirmed"] += r.confirmed.get(k, 0)
            d["suspects"] += r.suspects.get(k, 0)
            d["tasks_confirmed"] += 1 if r.confirmed.get(k, 0) else 0
    return {"tasks": len(results), "measured": len(measured), "on_intent": on, "share": round(on / len(measured), 4) if measured else None,
            "share_of_all": round(on / len(results), 4) if results else None, "unmeasured": len(results) - len(measured), "by_class": per,
            "duplicates": sorted({f"{r.task_id}~{d}" for r in results for d in r.duplicates})}
