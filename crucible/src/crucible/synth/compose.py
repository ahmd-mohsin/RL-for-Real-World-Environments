"""Composition, dependency depth, difficulty knobs and ladders (part2-paper-core §10).

A long-horizon task is a composition of tasks in which later work needs the results of earlier work.

**seq(τ₁, …, τ_k)** (``compose``). Part j starts from the state parts 1..j-1 leave, s_T(π₁*;…;π_{j-1}*):
- **Program.** π₁*;…;π_k*, every step renamed ``t<j>_<id>`` and its refs re-pointed (refs never cross a part).
- **Each part must still hold.** Each part's program runs on the state the earlier parts leave, and must still pass its
  own contract there; otherwise the parts interfere and composition is refused (``CompositionError``).
- **R = R₁ ⊕ … ⊕ R_k.** Every part's patterns, ids prefixed ``t<j>.``, under the merged normalizers. A pattern the
  composite oracle no longer satisfies is examined change by change: the changes it covered in its own part's run,
  under that part's own normalizers, mapped into the merged key space (``_in_merged``: a merged ``id_canon`` re-keys
  an insert by content).
  - **Superseded.** A later part wrote that same address, and one of the later part's patterns covers the composite
    change: s₀ → s_T shows only the last value. An insert is never superseded, because a later insert of equal
    content is a second row. A later glob that merely *matches* the change does not own it.
  - **Rebased.** Otherwise the requirement stays, as the exact skeleton pattern of the composite change (``x.r<k>``).
    This happens, for example, when the merged normalizers re-key the part's insert, when its ``rel(old)`` predicate
    was relative to the part's own start state, or when a later update lands on a row it inserted. A change no
    pattern covers any more gets one too.
  - **Restored.** A later part set the address back to its start value, so there is no net change and only the frame
    rules govern it (``restored``). Known gap: a merged *allowed* pattern can then permit the earlier value, and MUT
    finds the lost requirement as DropStep D1. Certification rejects the task, and the repairer may fix it.

  The composite oracle passes the merged contract by construction.
- **A = A₁ ∪ … ∪ A_k.**
- **Q.** The answer checks are merged. When several parts report, the answer is ``list(a₁, a₃, …)`` (the library's
  ``list``) and each check's path gains the part's position.
- **M.** Trace constraints are merged and keep their ``stated`` quotes.
  - ``at_most`` adds the matching calls the other parts make.
  - A ``never`` that another part's oracle breaks refuses the composition.
- **Instruction.** The ``instruction_writer`` role (``roles.INSTRUCTION_WRITER``, via a ``RoleClient``, never Claude)
  writes one instruction from the parts' instructions. Its text is checked by execution:
  - every stated quote is kept;
  - no value that flows across a part boundary is stated (``must_not_state``);
  - TAINT confirms nothing that the plain concatenation does not already show;
  - the dependency depth does not fall.

  Without a writer, or when every attempt fails, the instruction is the parts' own texts joined in order
  (``instruction_source = "joined"``).

**par(τ₁, …, τ_k).** The parts' written address sets must be pairwise disjoint, and each part alone from s₀ must do
exactly what it does in sequence (non-interference). Its depth is 0 by construction.

**Dependency depth.** dep(τ) is the longest chain of **confirmed** cross-part edges i → j (i < j). An edge is a use in
part j: a literal argument, a ref, or the part's answer. Two conditions make it count.
- **Taint.** The value is not given (not in ι, the intent card's given, or the tool catalogue), and an earlier
  observation shows it inside a record that part i wrote. That means either a field part i set (to that value) or the
  key of an entity part i changed, beside the changed field. A ref to a list or record that contains such a record
  counts as well.
- **Counterfactual.** Re-run with part i's writes removed. The edge is confirmed when part j's calls or answer change,
  when part j can no longer run, or when the observation that showed the value no longer shows that record holding
  it. The last case matters for literals: π₂*'s literal does not change when π₁* is dropped, but the agent could no
  longer have read it there. A record found independently of part i still shows the value, so it does not count.

dep = 0 is just tasks in a row. ``dep_unconfirmed`` is the same chain over taint candidates alone. A **carry** edge is
a literal of part j, not given, that only an earlier part's observations show (not of state it wrote: the agent must
remember what it read). It is a context dependency and never counts in dep; ``carry`` is the longest chain over
confirmed state edges and carry edges.

**Knobs** (``KnobSpec``, ``apply_knob``) are typed transformations. Each one predicts a direction ("harder") and
names the behaviour-signature features it must move. It is kept only if every one of those features increases.

| knob | does | moves |
|---|---|---|
| ``depth`` / ``horizon`` | another part (``compose``, ``ladder``) | dep / horizon |
| ``distractors`` | k distractor clones of entities π* touches, at reset (PERT's distractor rule; Perturbable Worlds only); list refs rebased by content | distractors, observed_entities |
| ``paginate`` | read tools whose lists exceed ``size`` return pages (``page``, ``next_page``); π* reads every page | pages |
| ``actor`` | an instruction-only literal moves to a person the agent must ask (``actors.ask``); the writer drops it from ι | asks |
| ``derive`` | a literal ι states and an observation also shows is dropped from ι and added to the intent card's derive | derived_literals |

A knob's effect on the student is not measured here. ``LearnabilityProbe`` and ``student_check`` are the hook a
learnability module calls ("p̂ moves in the declared direction"); nothing in this module calls the student.

**Variant Worlds.** ``VariantSpec`` describes a World derived from a base ``WorldRef``:
- a prefix program, run at reset: the start state of a follow-up link, s_T of the earlier links;
- perturbations;
- paginated tools;
- actors.

``variant_world`` is its ``WorldRef`` factory, so a variant task is an ordinary ``Task``, and the falsifiers, the archive
and certification apply unchanged.

**Ladders** (``ladder``) emit, for one chain of links, candidates of increasing horizon, depth and knob count. Each rung
is a new task (a new id) and must pass the full falsifier population and certification again (the CLI does that).
**Chains** (``build_chain``) are built by the swarm through ``pipeline.Runner``. Each link is proposed, programmed and
contracted on the prefix World of the links before it. The candidate kept has, in order of preference:
1. a confirmed state edge into it;
2. a write of its own (only a write can be depended on later);
3. the greatest depth;
4. carried information;
5. length.

A candidate that takes the composition past ``max_horizon`` calls is refused.

**Certification** (``certify_composed``) is ``pipeline.Runner.refine_one``, the same code as any proposed task. MUT's
proposal cap is ``MUT_PER_CALL`` × horizon, and the ALT and ATK budgets also scale with the horizon.
"""

from __future__ import annotations

import asyncio
import inspect
import json
import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import Normalizers, cjson, content_hash, diff, is_number, norm_text, same_diff
from crucible.synth.contract import AnswerCheck, Contract, EffectPattern, PFromAnswer, TraceConstraint, grade
from crucible.synth.falsify.base import FalsifyContext, key_text, run_classes
from crucible.synth.iface import Obs, Perturbable, Perturbation, SynthWorld, ToolInfo
from crucible.synth.infer import _lit_glob as _lit
from crucible.synth.infer import skeleton
from crucible.synth.program import Arg, Fn, Lit, Program, Ref, RefError, Run, Step, _step_into, is_structural, leaves, program_locs, run, text_spans
from crucible.synth.roles import INSTRUCTION_WRITER, RoleClient, RoleFailed, ask, tool_catalogue
from crucible.synth.task import IntentCard, Task, WorldRef

VARIANT_FACTORY = "crucible.synth.compose:variant_world"
ACTOR_TOOL = "actors.ask"
PART = re.compile(r"^(?:t\d+_)+")


class CompositionError(ValueError):
    """The tasks cannot be composed (interference, a conflicting constraint, another World, a dead part)."""


class KnobNotApplicable(ValueError):
    """A knob has nothing to act on in this task (no Perturbable World, no list longer than a page, no literal)."""


# ---------------------------------------------------------------------------------------------------------- variant Worlds

class PageSpec(CrucibleModel):
    field: str = ""  # the list field of a dict observation; "" = the observation is the list itself
    size: int = 5


class ActorSpec(CrucibleModel):
    name: str
    facts: dict[str, Any] = Field(default_factory=dict)  # topic -> the value the person holds


class VariantSpec(CrucibleModel):
    base: WorldRef
    prefix: Program | None = None
    perturb: list[Perturbation] = Field(default_factory=list)
    paginate: dict[str, PageSpec] = Field(default_factory=dict)
    actors: list[ActorSpec] = Field(default_factory=list)

    def features(self) -> dict[str, Any]:
        """Everything but the base and the prefix (two parts of one composition must agree on it)."""
        return self.model_dump(mode="json", exclude={"base", "prefix"})

    def ref(self) -> WorldRef:
        if not (self.prefix and self.prefix.steps) and not self.perturb and not self.paginate and not self.actors:
            return self.base
        return WorldRef(factory=VARIANT_FACTORY, params={"spec": self.model_dump(mode="json")})


def split_world(ref: WorldRef) -> VariantSpec:
    """A task's World as (base, variant features); a plain World is its own base."""
    if ref.factory == VARIANT_FACTORY:
        return VariantSpec.model_validate(ref.params["spec"])
    return VariantSpec(base=ref)


class PrefixError(RuntimeError):
    """The prefix program (the earlier links) no longer replays on the base World: a World defect, not an observation."""


class VariantWorld:
    """A ``SynthWorld`` over ``inner`` with a ``VariantSpec``: reset replays the prefix, then applies the perturbations;
    paginated tools return pages; ``actors.ask`` answers from the actors' facts. ``canon`` is the inner World's (actors
    and pages hold no state)."""

    def __init__(self, inner: SynthWorld, spec: VariantSpec):
        self.inner, self.spec = inner, spec

    async def reset(self, seed: int) -> Any:
        tok = await self.inner.reset(seed)
        if self.spec.prefix is not None and self.spec.prefix.steps:
            r = await run(self.inner, self.spec.prefix, seed=seed, reset=False, taint=False, track_effects=False)
            bad = r.error or next((c.obs.error for c in r.calls if c.obs.error), None)
            if bad:
                raise PrefixError(f"the earlier work does not replay: {bad[:300]}")
        for p in self.spec.perturb:
            await self.inner.perturb(p)  # type: ignore[attr-defined]
        return content_hash([tok, self.spec.features(), self.spec.prefix.digest() if self.spec.prefix else None], 32)

    async def tools(self) -> list[ToolInfo]:
        out = []
        for t in await self.inner.tools():
            ps = self.spec.paginate.get(t.name)
            out.append(paged_info(t, ps) if ps is not None else t)
        if self.spec.actors:
            out.append(actor_tool(self.spec.actors))
        return sorted(out, key=lambda t: t.name)

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        if tool == ACTOR_TOOL and self.spec.actors:
            return self._ask(args)
        ps = self.spec.paginate.get(tool)
        if ps is None or not isinstance(args, dict):
            return await self.inner.call(tool, args)
        a = dict(args)
        page = a.pop("page", 0)
        if page is None:
            page = 0
        if isinstance(page, bool) or not isinstance(page, int) or page < 0:
            return Obs(error="page must be a whole number >= 0")
        o = await self.inner.call(tool, a)
        return o if not o.ok else page_of(o.value, ps, page)

    def _ask(self, args: Mapping[str, Any]) -> Obs:
        who, topic = args.get("actor"), args.get("topic")
        a = next((x for x in self.spec.actors if x.name == who), None)
        if a is None:
            return Obs(error=f"there is no one called {who!r} to ask")
        if topic not in a.facts:
            return Obs(error=f"{who} does not know about {topic!r}")
        return Obs(value={"actor": who, "topic": topic, "answer": a.facts[topic]})

    async def snapshot(self) -> Any:
        return await self.inner.snapshot()

    async def restore(self, ref: Any) -> None:
        await self.inner.restore(ref)

    async def canon(self) -> Any:
        return await self.inner.canon()

    async def close(self) -> None:
        c = getattr(self.inner, "close", None)
        if c is not None:
            res = c()
            if inspect.isawaitable(res):
                await res


class PerturbableVariantWorld(VariantWorld):
    async def perturb(self, p: Perturbation) -> None:
        await self.inner.perturb(p)  # type: ignore[attr-defined]


async def variant_world(spec: Mapping[str, Any]) -> SynthWorld:
    """The ``WorldRef`` factory of variant Worlds."""
    s = VariantSpec.model_validate(spec)
    inner = await s.base.build()
    if s.perturb and not isinstance(inner, Perturbable):
        raise ValueError("this World cannot edit its start state (no perturb)")
    return (PerturbableVariantWorld if isinstance(inner, Perturbable) else VariantWorld)(inner, s)  # type: ignore[return-value]


def paged_info(t: ToolInfo, ps: PageSpec) -> ToolInfo:
    schema = dict(t.input_schema or {"type": "object"})
    props = dict(schema.get("properties") or {})
    props["page"] = {"type": "integer", "default": 0, "description": "which page of results to return (0 is the first)"}
    schema["properties"] = props
    what = f"the {ps.field} list" if ps.field else "results"
    desc = (t.description + " " if t.description else "") + (f"Returns {what} a page at a time, {ps.size} per page: pass page=1, 2, ... "
                                                             "for the rest; next_page is the next page number, or null after the last.")
    return t.model_copy(update={"input_schema": schema, "description": desc})


def page_of(v: Any, ps: PageSpec, page: int) -> Obs:
    items = v if ps.field == "" else (v.get(ps.field) if isinstance(v, dict) else None)
    if not isinstance(items, list):
        return Obs(value=v)
    n = max(1, math.ceil(len(items) / ps.size))
    if page >= n:
        return Obs(error=f"page {page} is past the last page ({n - 1})")
    sl = items[page * ps.size:(page + 1) * ps.size]
    meta = {"page": page, "pages": n, "next_page": page + 1 if page + 1 < n else None}
    return Obs(value={"items": sl, **meta} if ps.field == "" else {**v, ps.field: sl, **meta})


def actor_tool(actors: Sequence[ActorSpec]) -> ToolInfo:
    names = [a.name for a in actors]
    topics = sorted({t for a in actors for t in a.facts})
    return ToolInfo(name=ACTOR_TOOL, read_only=True, writes=[],
                    description=f"Ask a person involved in this work for information only they hold. People you can ask: {', '.join(names)}.",
                    input_schema={"type": "object", "properties": {"actor": {"type": "string", "enum": names}, "topic": {"type": "string", "enum": topics}},
                                  "required": ["actor", "topic"]})


# ---------------------------------------------------------------------------------------------------------- programs

def map_arg(a: Arg, f: Callable[[Ref], Arg]) -> Arg:
    if isinstance(a, Ref):
        return f(a)
    if isinstance(a, Fn):
        return Fn(name=a.name, args=[map_arg(x, f) for x in a.args])
    return Lit(v=a.v)


def rename(p: Program, pre: str) -> Program:
    """``p`` with every step id prefixed and its refs re-pointed."""
    ids = {s.id: f"{pre}{s.id}" for s in p.steps}

    def rr(r: Ref) -> Arg:
        head, dot, rest = r.path.partition(".")
        return Ref(path=ids.get(head, head) + dot + rest)

    steps = [Step(id=ids[s.id], tool=s.tool, args={k: map_arg(v, rr) for k, v in s.args.items()}) for s in p.steps]
    return Program(steps=steps, answer=map_arg(p.answer, rr) if p.answer is not None else None)


def alpha_digest(p: Program | None) -> str:
    """A program's steps up to step names (the prefix of a link compared with the earlier links)."""
    if p is None or not p.steps:
        return ""
    q = rename(Program(steps=list(p.steps)), "\x00")
    ids = {s.id: f"p{i}" for i, s in enumerate(q.steps)}

    def rr(r: Ref) -> Arg:
        head, dot, rest = r.path.partition(".")
        return Ref(path=ids.get(head, head) + dot + rest)

    return Program(steps=[Step(id=ids[s.id], tool=s.tool, args={k: map_arg(v, rr) for k, v in s.args.items()}) for s in q.steps]).digest()


def drop_writes(p: Program, read_only: Mapping[str, bool]) -> Program:
    """``p`` without its write steps (later steps that need them stop the part there, as a mutant does)."""
    return Program(steps=[s for s in p.steps if read_only.get(s.tool, False)], answer=p.answer)


def concat(progs: Sequence[Program]) -> Program:
    return Program(steps=[s for p in progs for s in p.steps])


async def run_parts(world: SynthWorld, parts: Sequence[Program], *, seed: int, instruction: str = "", given: Sequence[Any] = (), taint: bool = False) -> list[Run]:
    """Reset once, then run each part from the state the earlier parts left (one ``Run`` per part: its own start and
    end state). A part that stops does not stop the later parts."""
    await world.reset(seed)
    return [await run(world, p, instruction, seed=seed, given=given, reset=False, taint=taint) for p in parts]


def run_errors(r: Run) -> list[str]:
    out = [r.error] if r.error else []
    return out + [f"step {c.step} ({c.tool}): {(c.obs.error or '')[:200]}" for c in r.calls if c.obs.error]


# ---------------------------------------------------------------------------------------------------------- records

class Segment(CrucibleModel):
    index: int
    source: str
    version: int = 1
    digest: str = ""
    prefix: str = ""
    steps: list[str] = Field(default_factory=list)
    calls: int = 0
    writes: int = 0
    answers: bool = False
    instruction: str = ""
    skill: str = ""


class DepEdge(CrucibleModel):
    """A use in part ``dst`` (``loc``) of a value an earlier observation (``obs_step``) shows inside the record of an
    entity part ``src`` wrote (``entity``, its key ``key``, the field it set ``field``). ``value`` is the scalar used, or
    None for a ref to a whole list or record. ``confirmed``: without part src's writes, part dst's calls or answer change,
    or ``obs_step`` no longer shows that record holding the value."""

    src: int
    dst: int
    kind: Literal["literal", "ref", "answer", "carry"]
    loc: str
    entity: str
    key: str = ""
    field: str = ""
    value: Any = None
    obs_step: str = ""
    confirmed: bool = False


class KnobSpec(CrucibleModel):
    kind: Literal["depth", "horizon", "distractors", "paginate", "actor", "derive"]
    params: dict[str, Any] = Field(default_factory=dict)


KNOB_MOVES: dict[str, tuple[str, ...]] = {"depth": ("dep",), "horizon": ("horizon",), "distractors": ("distractors", "observed_entities"),
                                          "paginate": ("pages",), "actor": ("asks",), "derive": ("derived_literals",)}
KNOB_PREDICTED: dict[str, Literal["harder", "easier"]] = {k: "harder" for k in KNOB_MOVES}


class KnobRecord(CrucibleModel):
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    predicted: Literal["harder", "easier"] = "harder"
    moves: list[str] = Field(default_factory=list)
    before: dict[str, Any] = Field(default_factory=dict)
    after: dict[str, Any] = Field(default_factory=dict)
    moved: bool = False
    kept: bool = False
    reason: str = ""
    student: Literal["pending", "moved", "not_moved", "unmeasured"] = "pending"
    p_before: float | None = None
    p_after: float | None = None


class Composition(CrucibleModel):
    id: str
    op: Literal["seq", "par", "single"] = "seq"
    segments: list[Segment] = Field(default_factory=list)
    edges: list[DepEdge] = Field(default_factory=list)
    dep: int = 0
    dep_unconfirmed: int = 0
    carry: int = 0  # ``carry_depth``: state edges plus information kept from earlier parts' observations
    horizon: int = 0
    writes: int = 0
    answer_parts: list[int] = Field(default_factory=list)
    superseded: list[str] = Field(default_factory=list)
    rebased: list[str] = Field(default_factory=list)
    restored: list[str] = Field(default_factory=list)  # addresses a later part set back to their start value
    instruction_source: Literal["writer", "joined", "kept"] = "joined"
    writer: dict[str, Any] = Field(default_factory=dict)
    derive_added: list[Any] = Field(default_factory=list)
    knobs: list[KnobRecord] = Field(default_factory=list)
    base: str | None = None  # the task a knob variant was derived from
    signature: dict[str, Any] = Field(default_factory=dict)


@dataclass
class Composed:
    task: Task
    comp: Composition
    run: Run  # the composite oracle run under the final instruction (with taint)

    def dump(self) -> dict[str, Any]:
        return {"task": self.task.model_dump(mode="json"), "composition": self.comp.model_dump(mode="json")}


def load_composed(doc: Mapping[str, Any]) -> tuple[Task, Composition]:
    return Task.model_validate(doc["task"]), Composition.model_validate(doc["composition"])


# ---------------------------------------------------------------------------------------------------------- contracts

def merge_normalizers(ns: Sequence[Normalizers]) -> Normalizers:
    out = Normalizers()
    for n in ns:
        out = out.merged(n)
    data = {}
    for f in type(out).model_fields:
        seen, keep = set(), []
        for x in getattr(out, f):
            k = cjson(x.model_dump(mode="json"))
            if k not in seen:
                seen.add(k)
                keep.append(x)
        data[f] = keep
    return Normalizers(**data)


def _calls_matching(r: Run, glob: str) -> int:
    import fnmatch

    return sum(1 for c in r.calls if fnmatch.fnmatchcase(c.tool, glob))


def _in_merged(ch: Any, merged: Sequence[Any], n: Normalizers, own: Normalizers) -> Any:
    """The change of ``merged`` (a diff under the merged normalizers ``n``) that is ``ch`` (a change under the part's own
    normalizers ``own``): the same address, or, for an insert the merge re-keys by content, the insert of that collection
    whose content ``ch`` holds."""
    a = ch.addr
    if ch.kind == "insert" and n.id_canon_for(a.app, a.coll) is not None and own.id_canon_for(a.app, a.coll) is None:
        from crucible.synth.canon import same_value

        for m in merged:
            if m.kind == "insert" and (m.addr.app, m.addr.coll) == (a.app, a.coll) and all(same_value((ch.new or {}).get(f), v) for f, v in (m.new or {}).items()):
                return m
        return None
    return next((m for m in merged if str(m.addr) == str(a) and m.kind == ch.kind), None)


def merge_contracts(tasks: Sequence[Task], segs: Sequence[Run], full: Run, answer_parts: Sequence[int]) -> tuple[Contract, list[str], list[str], list[str]]:
    """R₁ ⊕ … ⊕ R_k with supersession and rebasing (module docstring). Returns (contract, superseded ids, rebased ids,
    restored addresses: an earlier part's write a later part undid, which only the frame rules govern in the composite)."""
    if any(t.contract.observation_only for t in tasks) and not all(t.contract.observation_only for t in tasks):
        raise CompositionError("an observation-only part cannot be composed with a stateful one")
    n = merge_normalizers([t.contract.normalizers for t in tasks])
    req: list[EffectPattern] = []
    allowed: list[EffectPattern] = []
    answer: list[AnswerCheck] = []
    trace: list[TraceConstraint] = []
    execs = []
    multi = len(answer_parts) > 1
    for i, t in enumerate(tasks):
        pre = f"t{i + 1}."
        pos = answer_parts.index(i) if i in answer_parts else None

        def apath(p: str | None, pos: int | None = pos) -> str | None:
            if not multi or pos is None:
                return p
            return str(pos) if p in (None, "") else f"{pos}.{p}"

        for p in t.contract.required:
            q = p.model_copy(update={"id": pre + p.id}, deep=True)
            if isinstance(q.value, PFromAnswer):
                q.value = q.value.model_copy(update={"path": apath(q.value.path)})
            q.where = {f: (v.model_copy(update={"path": apath(v.path)}) if isinstance(v, PFromAnswer) else v) for f, v in q.where.items()}
            req.append(q)
        allowed += [a.model_copy(update={"id": pre + a.id}, deep=True) for a in t.contract.allowed]
        answer += [q.model_copy(update={"id": pre + q.id, "path": apath(q.path)}, deep=True) for q in t.contract.answer]
        for m in t.contract.trace:
            m2 = m.model_copy(update={"id": pre + m.id}, deep=True)
            others = [s for j, s in enumerate(segs) if j != i]
            if m.op == "never" and any(_calls_matching(s, m.tool or "") for s in others):
                raise CompositionError(f"part {i + 1} forbids {m.tool!r}, which another part's work calls")
            if m.op == "at_most":
                m2.n = (m.n or 0) + sum(_calls_matching(s, m.tool or "") for s in others)
            trace.append(m2)
        execs += [x.model_copy(update={"id": pre + x.id}, deep=True) for x in t.contract.exec]
    c = Contract(required=req, allowed=allowed, answer=answer, trace=trace, normalizers=n, exec=execs,
                 observation_only=all(t.contract.observation_only for t in tasks))
    v = grade(full, c)
    trace_bad = [x for x in v.reasons if x.startswith("TRACE:")]
    if trace_bad:
        raise CompositionError(f"the parts conflict on a trace constraint: {trace_bad[:4]}")
    if any(x.startswith(("ANSWER:", "GRADER_ERROR")) for x in v.reasons):
        raise CompositionError(f"the merged contract fails the composite oracle: {v.reasons[:6]}")
    missing = [x.split(":", 1)[1] for x in v.reasons if x.startswith("MISSING_REQUIRED:")]
    superseded, rebased, restored = [], [], []
    add: list[EffectPattern] = []
    if missing:
        # each failing pattern, per change it covered in its own part's run: superseded when a later part's pattern covers
        # the composite change at that address (or no net change is left there); otherwise the requirement stays, as the
        # exact skeleton pattern of the composite change (compose-1: dropping it outright let an allowed pattern absorb
        # the change, and MUT found the lost requirement as DropStep D1)
        from crucible.synth.contract import _Ctx, matches

        assert full.start is not None and full.end is not None
        comp_delta = diff(full.start, full.end, n)
        by_addr = {str(ch.addr): k for k, ch in enumerate(comp_delta)}
        sk = skeleton(full, n)
        fctx = _Ctx(n.apply(full.start), n.apply(full.end), full.answer, math.inf)
        kept = [p for p in c.required if p.id not in set(missing)]
        added: set[int] = set()
        for pid in missing:
            part = int(pid.split(".", 1)[0][1:]) - 1
            p = next(x for x in c.required if x.id == pid)
            later = [q for q in kept if q.id.startswith("t") and int(q.id.split(".", 1)[0][1:]) - 1 > part]
            seg = segs[part]
            if p.kind == "state" or seg.start is None or seg.end is None:
                rebased.append(pid)  # a count of the end state: nothing to carry over change by change
                continue
            # the changes the pattern covered under its own part's normalizers, in the merged key space (a merged id_canon
            # re-keys an insert by content: compose-1's note pattern keyed by its raw id matched nothing under the merge)
            pn = tasks[part].contract.normalizers
            sctx = _Ctx(pn.apply(seg.start), pn.apply(seg.end), seg.answer, math.inf)
            seg_n = diff(seg.start, seg.end, n)
            mine = [m for ch in diff(seg.start, seg.end, pn) if matches(p, ch, sctx) and (m := _in_merged(ch, seg_n, n, pn)) is not None]
            if not mine:
                rebased.append(pid)
                continue
            later_addrs = {str(ch.addr) for s2 in segs[part + 1:] if s2.start is not None and s2.end is not None for ch in diff(s2.start, s2.end, n)}
            lost = False
            for ch in mine:
                k = by_addr.get(str(ch.addr))
                if k is None:  # no net change left: a later part restored the start value (the frame rules govern it now)
                    restored.append(str(ch.addr))
                    continue
                # superseded only when a later part itself wrote this address and one of its patterns covers the result:
                # a later glob that merely matches the change does not own it (compose-1: a note insert re-keyed by a
                # merged id_canon was dropped this way, and MUT found the lost requirement as DropStep D1)
                # an insert is never superseded: a later insert with equal content is a second row (under id_canon both
                # parts' segments key it by the same content hash)
                if ch.kind != "insert" and str(ch.addr) in later_addrs and any(matches(q, comp_delta[k], fctx) for q in later):
                    continue
                lost = True
                if k not in added:
                    added.add(k)
                    # keep the part's own pattern (its relaxations) re-keyed to the composite change when it still matches
                    # that change; else the exact skeleton pattern (compose-1: exact rebases turned relaxed free text into
                    # equality checks)
                    cc = comp_delta[k]
                    key = "*" if (cc.kind == "insert" and n.id_canon_for(cc.addr.app, cc.addr.coll) is not None) else _lit(cc.addr.key)
                    own_p = p.model_copy(update={"id": f"x.{pid}", "key": key, "count": 1, "count_max": None}, deep=True)
                    if len(mine) == 1 and p.count == 1 and matches(own_p, cc, fctx):
                        add.append(own_p)
                    else:
                        add.append(sk.required[k].model_copy(update={"id": f"x.{sk.required[k].id}"}, deep=True))
            (rebased if lost else superseded).append(pid)
        c = c.model_copy(update={"required": [*kept, *add]}, deep=True)
        rebased += [x.id for x in add]
    v = grade(full, c)
    if not v.passed:
        uncovered = {x.split(":", 1)[1] for x in v.reasons if x.startswith("FORBIDDEN_EFFECT:")}
        others = [x for x in v.reasons if not x.startswith("FORBIDDEN_EFFECT:")]
        if others:
            raise CompositionError(f"the merged contract fails the composite oracle: {others[:6]}")
        sk = skeleton(full, n)
        assert full.start is not None and full.end is not None
        changes = {str(ch.addr): k for k, ch in enumerate(diff(full.start, full.end, n))}
        add = []
        for addr in sorted(uncovered):
            k = changes.get(addr)
            if k is None:
                raise CompositionError(f"an uncovered change has no skeleton pattern: {addr}")
            add.append(sk.required[k].model_copy(update={"id": f"x.{sk.required[k].id}"}, deep=True))
        rebased += [p.id for p in add]
        c = c.model_copy(update={"required": [*c.required, *add]}, deep=True)
        v = grade(full, c)
        if not v.passed:
            raise CompositionError(f"rebasing did not make the composite oracle pass: {v.reasons[:6]}")
    return c, superseded, rebased, restored


# ---------------------------------------------------------------------------------------------------------- dependency

def _written(r: Run) -> dict[str, list[tuple[str, dict[str, Any]]]]:
    """key text -> [(entity, {field: new value})] of every entity a part changed (updates: the fields it set; inserts:
    the whole new row). Deletes leave nothing to observe."""
    out: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    if r.start is None or r.end is None:
        return out
    acc: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ch in diff(r.start, r.end):
        if ch.kind == "delete":
            continue
        e = acc.setdefault(ch.addr.entity, {})
        if ch.kind == "insert":
            e.update(ch.new or {})
        else:
            e[ch.addr.field] = ch.new
    for (app, coll, key), fields in acc.items():
        out.setdefault(key, []).append((f"{app}.{coll}[{key}]", fields))
    return out


def _loose(a: Any, b: Any) -> bool:
    from crucible.synth.program import _loose_eq

    if isinstance(a, bool) or isinstance(b, bool) or a is None or b is None:
        return False
    return _loose_eq(a, b)


def _record_hit(d: Mapping[str, Any], written: Mapping[str, list[tuple[str, dict[str, Any]]]]) -> tuple[str, str, Any, str] | None:
    """Is dict ``d`` (one level) a view of a written entity: does it hold the entity's key and a field the part set, with
    the value it set? Returns (entity, field, value, key) of the first hit."""
    for v in d.values():
        kt = key_text(v)
        if kt is None or kt not in written:
            continue
        for ent, fields in written[kt]:
            for f, nv in fields.items():
                if f in d and _loose(d[f], nv) and not (is_structural(nv)):
                    return ent, f, nv, kt
    return None


def _dicts(v: Any, path: str = "") -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if isinstance(v, dict):
        out.append((path, v))
        for k in v:
            out += _dicts(v[k], f"{path}.{k}" if path else str(k))
    elif isinstance(v, list):
        for i, x in enumerate(v):
            out += _dicts(x, f"{path}.{i}" if path else str(i))
    return out


def _resolve(obs: Any, segs: Sequence[str]) -> Any:
    cur = obs
    for s in segs:
        cur = _step_into(cur, s)
    return cur


def _given_text(v: Any, instruction: str, given: Sequence[Any], tools: Sequence[ToolInfo], *, ignore_given: bool = False) -> bool:
    """Can the agent know ``v`` without earlier work: the tool catalogue always counts; the instruction and the intent
    card's given count unless ``ignore_given``."""
    from crucible.synth.falsify.taint import sourced

    src = set(sourced(v, instruction, given, tools, []))
    return bool(src & ({"tool_spec", "structural"} if ignore_given else {"instruction", "given", "tool_spec", "structural"}))


def part_of_step(sid: str, comp: Composition) -> int:
    for s in comp.segments:
        if sid in s.steps:
            return s.index
    m = PART.match(sid)
    if m:
        first = m.group(0).split("_", 1)[0]
        return int(first[1:]) - 1
    return 0


def part_programs(task: Task, comp: Composition) -> list[Program]:
    """The composite program split back into its parts (each with its own answer)."""
    steps: list[list[Step]] = [[] for _ in comp.segments] or [[]]
    for s in task.oracle.steps:
        steps[min(part_of_step(s.id, comp), len(steps) - 1)].append(s)
    answers: list[Arg | None] = [None] * len(steps)
    a = task.oracle.answer
    if a is not None:
        if len(comp.answer_parts) > 1 and isinstance(a, Fn) and a.name == "list" and len(a.args) == len(comp.answer_parts):
            for k, i in enumerate(comp.answer_parts):
                answers[i] = a.args[k]
        else:
            answers[comp.answer_parts[0] if comp.answer_parts else len(steps) - 1] = a
    return [Program(steps=st, answer=answers[i]) for i, st in enumerate(steps)]


async def dependency(world: SynthWorld, parts: Sequence[Program], *, seed: int, instruction: str, given: Sequence[Any] = (),
                     counterfactual: bool = True, ignore_given: bool = False) -> tuple[list[DepEdge], int, int]:
    """Cross-part dependency edges and the depth (module docstring): (edges, dep over confirmed edges, dep over taint
    candidates). ``ignore_given`` drops the "not given" condition (the values that WOULD flow across a boundary if the
    instruction did not state them: the writer's ``must_not_state``)."""
    tools = await world.tools()
    segs = await run_parts(world, parts, seed=seed, instruction=instruction, given=given, taint=True)
    written = [_written(r) for r in segs]
    calls = [c for r in segs for c in r.calls]
    cands: dict[tuple[int, int, str], DepEdge] = {}

    def add(i: int, j: int, kind: str, loc: str, hit: tuple[str, str, Any, str], value: Any, step: str) -> None:
        k = (i, j, loc)
        if k not in cands:
            cands[k] = DepEdge(src=i, dst=j, kind=kind, loc=loc, entity=hit[0], key=hit[3], field=hit[1], value=value, obs_step=step)  # type: ignore[arg-type]

    def views(d: Mapping[str, Any], j: int) -> list[tuple[int, tuple[str, str, Any, str]]]:
        return [(i, h) for i in range(j) if (h := _record_hit(d, written[i])) is not None]

    for j, r in enumerate(segs):
        if j == 0:
            continue
        base = sum(len(x.calls) for x in segs[:j])
        obs = {c.step: c.obs.value for c in r.calls if c.obs.error is None}
        idx = {c.step: base + k for k, c in enumerate(r.calls)}
        # literals: shown, before their use, inside the record of an entity an earlier part wrote
        for e in r.taint:
            if e.has("structural") or e.value is None or isinstance(e.value, dict | list):
                continue
            if e.has("tool_spec") or (not ignore_given and (e.has("instruction") or e.has("given"))):
                continue
            sid = e.loc.split(".", 1)[0].split("/", 1)[0]
            upto = idx.get(sid, base + len(r.calls)) if sid != "answer" else base + len(r.calls)
            hit = None
            for c in calls[:upto]:
                if c.obs.error is not None:
                    continue
                for _, d in _dicts(c.obs.value):
                    if any(_loose(e.value, x) for x in d.values()):
                        vs = views(d, j)
                        if vs:
                            hit = (vs[0], c.step)
                            break
                if hit:
                    break
            if hit:
                (i, h), step = hit
                add(i, j, "literal" if sid != "answer" else "answer", e.loc, h, e.value, step)
            elif _statable(e.value):  # carried: only an earlier part's observations show it (the agent must remember them)
                own = [c for c in calls[base:upto] if c.obs.error is None and _shows_value(c.obs.value, e.value)]
                prev = [(i, c) for i in range(j) for c in segs[i].calls if c.obs.error is None and _shows_value(c.obs.value, e.value)]
                if prev and not own:
                    i, c = prev[-1]
                    k = (i, j, e.loc)
                    cands.setdefault(k, DepEdge(src=i, dst=j, kind="carry", loc=e.loc, entity="", value=e.value, obs_step=c.step))
        # refs: the path walks through such a record, or the value is a list or record that holds one
        for loc, a in program_locs(parts[j]):
            if not isinstance(a, Ref):
                continue
            head, *path = a.path.split(".")
            if head not in obs:
                continue
            try:
                val = _resolve(obs[head], path)
            except RefError:
                continue
            kind = "answer" if loc.startswith("answer") else "ref"
            if isinstance(val, dict | list):
                for _, d in _dicts(val):
                    for i, h in views(d, j):
                        add(i, j, kind, loc, h, None, head)
                continue
            if is_structural(val) or _given_text(val, instruction, given, tools, ignore_given=ignore_given):
                continue
            for depth in range(len(path) - 1, -1, -1):
                try:
                    d = _resolve(obs[head], path[:depth])
                except RefError:
                    continue
                if isinstance(d, dict):
                    vs = views(d, j)
                    if vs:
                        add(vs[0][0], j, kind, loc, vs[0][1], val, head)
                        break
    edges = sorted(cands.values(), key=lambda e: (e.src, e.dst, e.loc))
    state = [e for e in edges if e.kind != "carry"]
    if counterfactual and state:
        ro = {t.name: t.read_only for t in tools}
        for i in sorted({e.src for e in state}):
            alt = [drop_writes(p, ro) if k == i else p for k, p in enumerate(parts)]
            cf = await run_parts(world, alt, seed=seed, instruction=instruction, given=given)
            cf_obs = {c.step: c.obs for x in cf for c in x.calls}
            for e in state:
                if e.src != i:
                    continue
                a, b = segs[e.dst], cf[e.dst]
                acted = not (b.error is None and not any(c.obs.error for c in b.calls) and len(a.calls) == len(b.calls)
                             and all(x.tool == y.tool and cjson(x.args) == cjson(y.args) for x, y in zip(a.calls, b.calls, strict=True))
                             and cjson(a.answer) == cjson(b.answer))
                o = cf_obs.get(e.obs_step)
                shown = e.value is None or (o is not None and o.error is None and _shows(o.value, e.key, e.value))
                e.confirmed = acted or not shown
    return edges, longest_chain([e for e in edges if e.confirmed]), longest_chain(state)


def carry_depth(edges: Sequence[DepEdge]) -> int:
    """The longest chain over confirmed state edges and carry edges (information the agent must keep from earlier
    parts' observations, which is a context dependency, not a state one)."""
    return longest_chain([e for e in edges if e.confirmed or e.kind == "carry"])


def _shows_value(v: Any, value: Any) -> bool:
    return any(_loose(value, x) for _, x in leaves(v))


def _shows(v: Any, key: str, value: Any) -> bool:
    """Does an observation hold a record with ``key`` among its fields and ``value`` among its leaves?"""
    for _, d in _dicts(v):
        if any(key_text(x) == key for x in d.values()) and any(_loose(value, x) or key_text(value) == key_text(x) is not None for _, x in leaves(d)):
            return True
    return False


def longest_chain(edges: Sequence[DepEdge]) -> int:
    best: dict[int, int] = {}
    for e in sorted(edges, key=lambda e: (e.dst, e.src)):
        best[e.dst] = max(best.get(e.dst, 0), best.get(e.src, 0) + 1)
    return max(best.values(), default=0)


# ---------------------------------------------------------------------------------------------------------- signature

def behaviour_signature(task: Task, r: Run, comp: Composition | None = None) -> dict[str, Any]:
    """What the oracle *does* (P4, §10's "the behaviour signature moves"): call and write counts, the tool sequence, the
    fields written, depth and edges, distractors in the World, entity records observed, pages read, people asked, and
    how many literals the agent must derive (no instruction or given source)."""
    spec = split_world(task.world)
    keys = {k for _, _, k, _ in (r.start.entities() if r.start is not None else [])}
    observed = set()
    for c in r.calls:
        if c.obs.error is None:
            for _, d in _dicts(c.obs.value):
                for v in d.values():
                    kt = key_text(v)
                    if kt is not None and kt in keys:
                        observed.add(cjson(d))
                        break
    written = sorted({f"{ch.addr.app}.{ch.addr.coll}.{ch.addr.field}" for ch in diff(r.start, r.end)}) if r.start is not None and r.end is not None else []
    lits = [e for e in r.taint if not e.has("structural")]
    sig = {"horizon": len(r.calls), "writes": sum(1 for c in r.calls if not c.read_only), "parts": len(comp.segments) if comp else 1,
           "dep": comp.dep if comp else 0, "carry": comp.carry if comp else 0, "edges": sorted({(e.src, e.dst) for e in (comp.edges if comp else []) if e.confirmed}),
           "tools": dict(sorted(Counter(c.tool for c in r.calls).items())), "written_fields": written, "distractors": len(spec.perturb),
           "observed_entities": len(observed), "pages": sum(1 for s in task.oracle.steps if "__p" in s.id), "asks": sum(1 for c in r.calls if c.tool == ACTOR_TOOL),
           "derived_literals": sum(1 for e in lits if not (e.has("instruction") or e.has("given"))),
           "instruction_literals": sum(1 for e in lits if e.has("instruction")), "derive": len(task.intent.derive)}
    sig["digest"] = content_hash([[c.tool for c in r.calls], written, sig["edges"], sig["distractors"], sig["pages"], sig["asks"], sig["derived_literals"]], 16)
    return sig


def moved(before: Mapping[str, Any], after: Mapping[str, Any], moves: Sequence[str], predicted: str = "harder") -> bool:
    if predicted == "harder":
        return all((after.get(m) or 0) > (before.get(m) or 0) for m in moves)
    return all((after.get(m) or 0) < (before.get(m) or 0) for m in moves)


@runtime_checkable
class LearnabilityProbe(Protocol):
    """The learnability module's estimate of the student's pass rate p̂ on a task (None: not measured). This module
    never implements or calls it; ``student_check`` is the hook."""

    async def p_hat(self, task: Task) -> float | None: ...


def direction_holds(p_before: float | None, p_after: float | None, predicted: str, margin: float = 0.0) -> bool | None:
    if p_before is None or p_after is None:
        return None
    return p_after < p_before - margin if predicted == "harder" else p_after > p_before + margin


async def student_check(record: KnobRecord, base: Task, variant: Task, probe: LearnabilityProbe, *, margin: float = 0.0) -> KnobRecord:
    """§10 "the student's p̂ moves in the declared direction", for a learnability module to call (never called here)."""
    pb, pa = await probe.p_hat(base), await probe.p_hat(variant)
    ok = direction_holds(pb, pa, record.predicted, margin)
    return record.model_copy(update={"p_before": pb, "p_after": pa, "student": "unmeasured" if ok is None else ("moved" if ok else "not_moved")})


# ---------------------------------------------------------------------------------------------------------- the writer

COMPOSE_JOB = ("These parts are one piece of work done in one session, in this order: each part continues from the state the earlier "
               "parts leave. Write the one instruction for all of it.")
ACTOR_JOB = ("Rewrite the instruction (one part) so that it no longer states the values in must_not_state. The agent gets each of "
             "them by asking the person named in must_mention (through the tool actors.ask). Say who to ask about what, as a user "
             "would (\"ask the requester for the invoice number\"). Keep everything else.")
DERIVE_JOB = ("Rewrite the instruction (one part) so that it no longer states the values in must_not_state: the agent must find each "
              "of them in the system itself. Refer to what the value is about without giving it. Keep everything else.")


def _quotes(c: Contract, instruction: str) -> list[str]:
    ins = norm_text(instruction)
    return [m.stated for m in c.trace if m.stated and norm_text(m.stated) in ins]


def _statable(v: Any) -> bool:
    """A value worth keeping out of an instruction: an id, a name or an amount (not a tiny number or one character)."""
    if isinstance(v, bool) or v is None:
        return False
    if is_number(v):
        return abs(v) >= 10
    return isinstance(v, str) and len(v.strip()) >= 2


_LEAKS: Any = None


def _stated(v: Any, text: str) -> bool:
    """Does ``text`` state ``v``: its exact tokens, or a paraphrase TAINT's layered leak matcher counts (DESIGN §7.7 item
    20: "missing receipt" for missing_receipt). Every compose check uses this one test, so the writer's must-not-state,
    the derive merge and TAINT agree (after the rebase, an exact-only derive filter left a paraphrased value TAINT
    flagged as D7)."""
    global _LEAKS
    if text_spans(v, text):
        return True
    if _LEAKS is None:
        from crucible.synth.falsify.leak import LeakMatcher

        _LEAKS = LeakMatcher()
    return bool(_LEAKS.counted(_LEAKS.find(v, text)))


def joined(instructions: Sequence[str], answer_parts: Sequence[int]) -> str:
    """The parts' own instructions in order (the fallback when there is no writer)."""
    body = "\n".join(f"{k + 1}. {x.strip()}" for k, x in enumerate(instructions))
    tail = ""
    if len(answer_parts) > 1:
        tail = ("\nReply with one list of the values asked for, in this order: " + ", ".join(f"the value part {i + 1} asks for" for i in answer_parts) + ".")
    return "Do the following, in this order; each step continues from the state the earlier steps leave.\n" + body + tail


async def taint_sites(task: Task, ctx: FalsifyContext) -> set[str]:
    """The sites TAINT confirms on ``task`` (D5 unsourced literals and constants, D7 leaks)."""
    from crucible.synth.falsify.taint import TAINT

    res = await run_classes(task, [TAINT()], ctx)
    return {f.site for r in res for f in r.confirmed}


@dataclass
class WriterResult:
    instruction: str
    source: Literal["writer", "joined", "kept"]
    attempts: int = 0
    notes: str = ""
    error: str = ""


async def write_instruction(writer: RoleClient | None, *, job: str, parts: list[dict[str, Any]], fallback: str, check: Callable[[str], Awaitable[None]],
                            must_keep: Sequence[str] = (), must_not_state: Sequence[Any] = (), must_mention: Sequence[str] = (),
                            answer_format: str | None = None, tools: Sequence[ToolInfo] = (), tag: str = "compose", retries: int = 2,
                            fallback_source: Literal["joined", "kept"] = "joined") -> WriterResult:
    """One writer call checked by execution (``check`` raises ``ValueError`` with feedback). Falls back to ``fallback``."""
    if writer is None:
        return WriterResult(fallback, fallback_source)
    n = 0

    async def valid(o: Any) -> str:
        nonlocal n
        n += 1
        ins = o.instruction.strip()
        miss = [q for q in must_keep if norm_text(q) not in norm_text(ins)]
        if miss:
            raise ValueError(f"the instruction must contain these texts word for word: {miss}")
        # the layered leak matcher (DESIGN §7.7 item 20): "on hold" for on_hold, "the second invoice" for a number
        said = [v for v in must_not_state if _stated(v, ins)]
        if said:
            raise ValueError(f"the instruction states values it must not state (the agent finds them itself): {said}")
        gone = [m for m in must_mention if m.casefold() not in ins.casefold()]
        if gone:
            raise ValueError(f"the instruction must mention: {gone}")
        await check(ins)
        return ins

    try:
        out = await ask(writer, INSTRUCTION_WRITER, tag=tag, check=valid, retries=retries, job=job, parts=parts, must_keep=list(must_keep) or None,
                        must_not_state=list(must_not_state) or None, must_mention=list(must_mention) or None, answer_format=answer_format,
                        tools=[{"name": t["name"], "description": t.get("description", "")[:300]} for t in tool_catalogue(tools)] or None)
    except RoleFailed as e:
        return WriterResult(fallback, fallback_source, attempts=n, error=str(e)[:600])
    return WriterResult(str(out), "writer", attempts=n)


# ---------------------------------------------------------------------------------------------------------- compose

def _stub(world: WorldRef, seed: int, domain: str) -> Task:
    return Task(id="compose-stub", domain=domain, world=world, seed=seed, instruction="", contract=Contract(), oracle=Program())


def _union(xs: Sequence[Sequence[Any]]) -> list[Any]:
    seen, out = set(), []
    for x in xs:
        for v in x:
            k = cjson(v)
            if k not in seen:
                seen.add(k)
                out.append(v)
    return out


async def compose(tasks: Sequence[Task], *, op: Literal["seq", "par"] = "seq", ctx: FalsifyContext | None = None, writer: RoleClient | None = None,
                  task_id: str | None = None, tag: str = "compose", writer_retries: int = 2, counterfactual: bool = True) -> Composed:
    """seq or par composition of ``tasks`` (module docstring)."""
    if len(tasks) < 2:
        raise CompositionError("composition needs two or more tasks")
    own = ctx is None
    ctx = ctx or FalsifyContext()
    try:
        return await _compose(list(tasks), op=op, ctx=ctx, writer=writer, task_id=task_id, tag=tag, writer_retries=writer_retries, counterfactual=counterfactual)
    finally:
        if own:
            await ctx.close()


async def _compose(tasks: list[Task], *, op: str, ctx: FalsifyContext, writer: RoleClient | None, task_id: str | None, tag: str, writer_retries: int,
                   counterfactual: bool) -> Composed:
    specs = [split_world(t.world) for t in tasks]
    base, feats, seed = specs[0].base, specs[0].features(), tasks[0].seed
    for k, (t, s) in enumerate(zip(tasks, specs, strict=True)):
        if s.base.model_dump_json() != base.model_dump_json() or s.features() != feats:
            raise CompositionError(f"part {k + 1} runs in another World")
        if t.seed != seed:
            raise CompositionError(f"part {k + 1} starts from another seed")
    progs = [rename(t.oracle, f"t{k + 1}_") for k, t in enumerate(tasks)]
    p0 = specs[0].prefix
    for k in range(1, len(tasks)):
        pk = specs[k].prefix
        if not (pk and pk.steps) or alpha_digest(pk) == alpha_digest(p0):
            if op == "par" and pk is not None and alpha_digest(pk) != alpha_digest(p0):
                raise CompositionError("par parts start from the same state")
            continue
        if op == "par":
            raise CompositionError("par parts start from the same state")
        want = concat([*([p0] if p0 is not None else []), *progs[:k]])
        if alpha_digest(pk) != alpha_digest(want):
            raise CompositionError(f"part {k + 1} was built after other earlier work than the parts before it")
    wref = VariantSpec(base=base, prefix=p0, perturb=specs[0].perturb, paginate=specs[0].paginate, actors=specs[0].actors).ref()
    domain = tasks[0].domain
    w = await ctx.world(_stub(wref, seed, domain))
    given = _union([t.intent.given for t in tasks])
    segs = await run_parts(w, progs, seed=seed, given=given)
    for k, (t, r) in enumerate(zip(tasks, segs, strict=True)):
        errs = run_errors(r)
        if errs:
            raise CompositionError(f"part {k + 1} does not run after the earlier parts: {errs[:3]}")
        v = grade(r, t.contract)
        if not v.passed:
            raise CompositionError(f"part {k + 1} fails its own contract after the earlier parts: {v.reasons[:6]}")
    if op == "par":  # under the merged normalizers: a generated id or a clock stamp is no interference
        nz = merge_normalizers([t.contract.normalizers for t in tasks])
        addrs = [{str(c.addr) for c in diff(r.start, r.end, nz)} for r in segs if r.start is not None and r.end is not None]
        for a in range(len(addrs)):
            for b in range(a + 1, len(addrs)):
                both = addrs[a] & addrs[b]
                if both:
                    raise CompositionError(f"par parts {a + 1} and {b + 1} write the same addresses: {sorted(both)[:4]}")
        for k in range(1, len(progs)):
            alone = (await run_parts(w, [progs[k]], seed=seed, given=given))[0]
            if alone.start is None or not same_diff(diff(alone.start, alone.end, nz), diff(segs[k].start, segs[k].end, nz)):  # type: ignore[arg-type]
                raise CompositionError(f"par part {k + 1} does something else after the parts before it (they interfere)")
    answer_parts = [k for k, p in enumerate(progs) if p.answer is not None]
    if not answer_parts:
        answer = None
    elif len(answer_parts) == 1:
        answer = progs[answer_parts[0]].answer
    else:
        answer = Fn(name="list", args=[progs[k].answer for k in answer_parts])  # type: ignore[misc]
    program = Program(steps=[s for p in progs for s in p.steps], answer=answer)
    base_ins = joined([t.instruction for t in tasks], answer_parts)
    full = await _run_full(w, program, base_ins, seed, given)
    contract, superseded, rebased, restored = merge_contracts(tasks, segs, full, answer_parts)
    segments = [Segment(index=k, source=t.id, version=t.version, digest=t.digest(), prefix=f"t{k + 1}_", steps=[s.id for s in progs[k].steps],
                        calls=len(segs[k].calls), writes=sum(1 for c in segs[k].calls if not c.read_only), answers=progs[k].answer is not None,
                        instruction=t.instruction, skill=t.intent.skill) for k, t in enumerate(tasks)]
    tid = task_id or f"{domain}/compose/{content_hash([op, [t.id for t in tasks], [t.version for t in tasks]], 10)}"
    # a part's derive value that another part's instruction states is no leak of that part's decision: the composite
    # instruction must state it for the other part (compose-1: "critical" derived by part 5, stated by part 1: a false D7)
    derive = [d for d in _union([t.intent.derive for t in tasks]) if not any(_stated(d, t.instruction) for t in tasks)]
    intent = IntentCard(skill=" -> ".join(t.intent.skill or f"part {k + 1}" for k, t in enumerate(tasks))[:300], given=given,
                        derive=derive, notes=f"{op} composition of {len(tasks)} tasks")
    task = Task(id=tid, domain=domain, world=wref, seed=seed, instruction=base_ins, contract=contract, oracle=program, intent=intent)
    comp = Composition(id=tid, op=op, segments=segments, horizon=len(full.calls), writes=sum(1 for c in full.calls if not c.read_only),  # type: ignore[arg-type]
                       answer_parts=answer_parts, superseded=superseded, rebased=rebased, restored=sorted(set(restored)))
    parts = [Program(steps=p.steps, answer=p.answer) for p in progs]
    edges, dep, dep_u = await dependency(w, parts, seed=seed, instruction=base_ins, given=given, counterfactual=counterfactual)
    flow, _, _ = await dependency(w, parts, seed=seed, instruction=base_ins, given=given, counterfactual=False, ignore_given=True)
    base_sites = await taint_sites(task, ctx)
    quotes = [q for t in tasks for q in _quotes(t.contract, t.instruction)]
    # values that flow across a boundary and that no earlier part's instruction states: the writer must not state them
    no_state = _union([[e.value] for e in flow if e.value is not None and _statable(e.value)
                       and not any(_stated(e.value, tasks[i].instruction) for i in range(e.dst))])
    fmt = None
    if len(answer_parts) > 1:
        fmt = ("The agent reports one list with " + str(len(answer_parts)) + " values, in this order: " + ", then ".join(f"the value part {k + 1} asks for" for k in answer_parts) + ".")
    elif answer_parts:
        fmt = f"The agent reports the value part {answer_parts[0] + 1} asks for, as that part says."
    tools = await w.tools()

    async def check(ins: str) -> None:
        cand = task.model_copy(update={"instruction": ins}, deep=True)
        new = await taint_sites(cand, ctx) - base_sites
        if new:
            raise ValueError("the instruction loses information the agent needs, or gives away a decision: "
                             + "; ".join(sorted(new)[:6]) + ". State every fact the parts state that the agent cannot look up; never state a decision.")
        _, d2, _ = await dependency(w, parts, seed=seed, instruction=ins, given=given, counterfactual=counterfactual)
        if d2 < dep:
            raise ValueError("the instruction states a result of earlier work that the agent must take from that work instead; refer to it by what "
                             "was done, not by its value")

    wr = await write_instruction(writer, job=COMPOSE_JOB, parts=[{"part": k + 1, "instruction": t.instruction} for k, t in enumerate(tasks)],
                                 fallback=base_ins, check=check, must_keep=quotes, must_not_state=no_state, answer_format=fmt, tools=tools, tag=tag,
                                 retries=writer_retries)
    ins = wr.instruction
    if ins != base_ins:
        edges, dep, dep_u = await dependency(w, parts, seed=seed, instruction=ins, given=given, counterfactual=counterfactual)
    added = [e.value for e in edges if e.confirmed and e.value is not None and _statable(e.value) and not _stated(e.value, ins)
             and not any(cjson(e.value) == cjson(d) for d in intent.derive)]
    added = _union([added])
    intent = intent.model_copy(update={"derive": [*intent.derive, *added]})
    task = task.model_copy(update={"instruction": ins, "intent": intent}, deep=True)
    full = await _run_full(w, program, ins, seed, given)
    v = grade(full, task.contract)
    if not v.passed:
        raise CompositionError(f"the composite oracle fails the composite contract: {v.reasons[:6]}")
    comp = comp.model_copy(update={"edges": edges, "dep": dep, "dep_unconfirmed": dep_u, "carry": carry_depth(edges), "instruction_source": wr.source, "derive_added": added,
                                   "writer": {"attempts": wr.attempts, "error": wr.error, **({"family": writer.family, "model": writer.model} if writer else {})}})
    comp.signature = behaviour_signature(task, full, comp)
    return Composed(task=task, comp=comp, run=full)


async def _run_full(w: SynthWorld, program: Program, instruction: str, seed: int, given: Sequence[Any]) -> Run:
    r = await run(w, program, instruction, seed=seed, given=given, taint=True)
    errs = run_errors(r)
    if errs:
        raise CompositionError(f"the composite program does not run: {errs[:3]}")
    return r


async def single(task: Task, ctx: FalsifyContext) -> Composed:
    """A plain task as a one-part ``Composed`` (so knobs apply to tasks and compositions alike)."""
    r = await ctx.oracle(task)
    comp = Composition(id=task.id, op="single", horizon=len(r.calls), writes=sum(1 for c in r.calls if not c.read_only),
                       answer_parts=[0] if task.oracle.answer is not None else [], instruction_source="kept",
                       segments=[Segment(index=0, source=task.id, version=task.version, digest=task.digest(), steps=[s.id for s in task.oracle.steps],
                                         calls=len(r.calls), writes=sum(1 for c in r.calls if not c.read_only), answers=task.oracle.answer is not None,
                                         instruction=task.instruction, skill=task.intent.skill)])
    comp.signature = behaviour_signature(task, r, comp)
    return Composed(task=task, comp=comp, run=r)


# ---------------------------------------------------------------------------------------------------------- knobs

def rebase_refs(p: Program, anchors: Mapping[str, list[list[Any]]], r: Run) -> Program:
    """``p`` with every anchored list index re-pointed at the element with the same content in ``r``'s observations."""
    obs = {c.step: c.obs.value for c in r.calls}
    q = p.model_copy(deep=True)
    for loc, a in program_locs(q):
        if not isinstance(a, Ref) or loc not in anchors:
            continue
        want = {int(d): h for d, h in anchors[loc]}
        head, *segs = a.path.split(".")
        cur, out = obs.get(head), []
        for depth, seg in enumerate(segs):
            if isinstance(cur, list) and depth in want:
                j = next((i for i, x in enumerate(cur) if content_hash(x) == want[depth]), None)
                if j is None:
                    raise KnobNotApplicable(f"{loc}: the anchored element is gone")
                seg = str(j)
            out.append(str(seg))
            cur = _step_into(cur, seg)
        a.path = ".".join([head, *out])
    return q


def _list_of(v: Any, ps: PageSpec) -> list[Any] | None:
    items = v if ps.field == "" else (v.get(ps.field) if isinstance(v, dict) else None)
    return items if isinstance(items, list) else None


def page_specs(r: Run, tools: Sequence[ToolInfo], size: int) -> dict[str, PageSpec]:
    """Read tools whose oracle observations hold a list longer than one page (the bare list, or a dict's only list field)."""
    out: dict[str, PageSpec] = {}
    bad: set[str] = set()
    ro = {t.name: t.read_only for t in tools}
    for c in r.calls:
        if not ro.get(c.tool) or c.obs.error is not None or c.tool == ACTOR_TOOL:
            continue
        v = c.obs.value
        if isinstance(v, list):
            f = ""
        elif isinstance(v, dict):
            lists = [k for k, x in v.items() if isinstance(x, list)]
            if len(lists) != 1 or any(k in v for k in ("page", "pages", "next_page")):
                continue
            f = lists[0]
        else:
            continue
        if c.tool in out and out[c.tool].field != f:
            bad.add(c.tool)
        n = len(v if f == "" else v[f])
        if n > size:
            out[c.tool] = PageSpec(field=f, size=size)
    return {k: v for k, v in out.items() if k not in bad}


def paginate_program(p: Program, r: Run, paged: Mapping[str, PageSpec]) -> Program:
    """π* reading every page (module docstring): ``s`` is page 0, ``s__p<k>`` page k (``page`` from the previous page's
    ``next_page``); refs into a paged list are re-pointed to the page that holds the element, a ref to the whole list
    becomes the concat of every page's list."""
    obs = {c.step: c for c in r.calls}
    pages: dict[str, tuple[PageSpec, int]] = {}
    for s in p.steps:
        ps = paged.get(s.tool)
        c = obs.get(s.id)
        if ps is None or c is None or c.obs.error is not None:
            continue
        items = _list_of(c.obs.value, ps)
        if items is not None:
            pages[s.id] = (ps, len(items))

    def pid(head: str, k: int) -> str:
        return head if k == 0 else f"{head}__p{k}"

    def fix(rf: Ref) -> Arg:
        head, *segs = rf.path.split(".")
        if head not in pages:
            return rf
        ps, n = pages[head]
        box = "items" if ps.field == "" else ps.field
        if ps.field:
            if not segs or segs[0] != ps.field:
                return rf
            rest = segs[1:]
        else:
            rest = segs
        npages = max(1, math.ceil(n / ps.size))
        if not rest:
            return Fn(name="concat", args=[Fn(name="get", args=[Ref(path=pid(head, k)), Lit(v=box)]) for k in range(npages)])
        try:
            i = int(rest[0])
        except ValueError:
            raise KnobNotApplicable(f"ref {rf.path!r} does not index the paged list") from None
        if i < 0:
            i += n
        k, j = divmod(i, ps.size)
        return Ref(path=".".join([pid(head, k), box, str(j), *rest[1:]]))

    steps: list[Step] = []
    for s in p.steps:
        s2 = Step(id=s.id, tool=s.tool, args={k: map_arg(v, fix) for k, v in s.args.items()})
        steps.append(s2)
        if s.id in pages:
            ps, n = pages[s.id]
            for k in range(1, max(1, math.ceil(n / ps.size))):
                steps.append(Step(id=pid(s.id, k), tool=s.tool, args={**s2.args, "page": Ref(path=f"{pid(s.id, k - 1)}.next_page")}))
    return Program(steps=steps, answer=map_arg(p.answer, fix) if p.answer is not None else None)


@dataclass
class Variant:
    composed: Composed
    record: KnobRecord


async def _verify_same(base: Composed, task: Task, ctx: FalsifyContext, *, what: str) -> Run:
    """The variant's oracle runs, does what the base oracle does (same normalised diff and answer) and passes."""
    r = await ctx.oracle(task)
    errs = run_errors(r)
    if errs:
        raise KnobNotApplicable(f"{what}: the rewritten oracle does not run: {errs[:3]}")
    n = task.contract.normalizers
    if base.run.start is None or r.start is None or not same_diff(diff(r.start, r.end, n), diff(base.run.start, base.run.end, n)):  # type: ignore[arg-type]
        raise KnobNotApplicable(f"{what}: the rewritten oracle changes something else")
    if cjson(r.answer) != cjson(base.run.answer):
        raise KnobNotApplicable(f"{what}: the rewritten oracle reports another answer")
    v = ctx.grade(r, task.contract)
    if not v.passed:
        raise KnobNotApplicable(f"{what}: the rewritten oracle fails the contract: {v.reasons[:4]}")
    return r


def _derived(base: Composed, task: Task, label: str, comp_update: dict[str, Any]) -> tuple[Task, Composition]:
    t = task.model_copy(update={"id": f"{base.task.id}~{label}", "version": 1}, deep=True)
    c = base.comp.model_copy(update={"id": t.id, "base": base.task.id, **comp_update}, deep=True)
    return t, c


def distractor_clones(r: Run, tools: Sequence[ToolInfo], k: int) -> list[Perturbation]:
    """Up to ``k`` look-alike clones of entities π* touches (PERT's distractor machinery, a stricter field rule). A clone
    differs from its original on every field holding a value π* selects or acts on: a literal or a resolved argument,
    except the catalogue's enum values (a status stays a status, so the clone still shows up where the original does).
    An entity with no such field is not cloned: a clone the task cannot tell apart would change the intended outcome."""
    from crucible.synth.falsify.base import delta_of, entity_index, written_entities
    from crucible.synth.falsify.mut import key_colls
    from crucible.synth.falsify.pert import _new_key, _perturbed
    from crucible.synth.program import scalar_leaves

    s0 = r.start
    if s0 is None:
        return []
    enums = {cjson(e) for t in tools for p in (t.input_schema.get("properties") or {}).values() if isinstance(p, dict) for e in p.get("enum", [])}
    marks = {cjson(e.value) for e in r.taint if not e.has("structural") and not e.has("tool_spec") and not is_structural(e.value)}
    marks |= {cjson(v) for c in r.calls for _, v in scalar_leaves(c.args, "") if not is_structural(v)}
    marks -= enums
    marks = {m for m in marks if not _given_text(json.loads(m), "", [], tools, ignore_given=True)}  # catalogue values (defaults, described codes)
    idx = entity_index(s0)
    touched: set[tuple[str, str, str]] = set(written_entities(delta_of(r, Contract())))
    for c in r.calls:
        for _, v in scalar_leaves(c.args, ""):
            for app, coll in key_colls(v, c.tool, idx)[:1]:
                touched.add((app, coll, key_text(v) or ""))
    out: list[Perturbation] = []
    used: dict[tuple[str, str], list[str]] = {}
    for app, coll, key in sorted(touched):
        row = s0.entity(app, coll, key)
        if row is None:
            continue
        over = {f: _perturbed(v) for f, v in row.items() if key_text(v) != key and (cjson(v) in marks or (key_text(v) is not None and cjson(key_text(v)) in marks))
                and not isinstance(v, bool)}
        if not over:
            continue
        keys = sorted(s0.collection(app, coll)) + used.setdefault((app, coll), [])
        nk = _new_key(key, keys)
        used[(app, coll)].append(nk)
        pos = list(s0.collection(app, coll)).index(key)
        out.append(Perturbation(op="clone", app=app, coll=coll, key=key, new_key=nk, fields=over, position=pos))
        if len(out) >= k:
            break
    return out


async def knob_distractors(base: Composed, ctx: FalsifyContext, k: int = 3) -> tuple[Task, Composition]:
    task = base.task
    w = await ctx.world(task)
    if not isinstance(w, Perturbable):
        raise KnobNotApplicable("the World cannot edit its start state (no perturb)")
    clones = distractor_clones(base.run, await w.tools(), k)
    if not clones:
        raise KnobNotApplicable("no entity π* touches has a field holding a value π* selects or acts on")
    spec = split_world(task.world)
    spec = spec.model_copy(update={"perturb": [*spec.perturb, *clones]}, deep=True)
    stub = task.model_copy(update={"world": spec.ref()}, deep=True)
    try:
        anchored = await ctx.run(stub, task.oracle, anchors=base.run.anchors)
    except (KeyError, ValueError) as e:
        raise KnobNotApplicable(f"the World refused the clones: {type(e).__name__}: {str(e)[:200]}") from e
    prog = rebase_refs(task.oracle, base.run.anchors, anchored)
    return _derived(base, stub.model_copy(update={"oracle": prog}, deep=True), f"d{len(clones)}", {})


async def knob_paginate(base: Composed, ctx: FalsifyContext, size: int = 5) -> tuple[Task, Composition]:
    task = base.task
    tools = await (await ctx.world(task)).tools()
    paged = page_specs(base.run, tools, size)
    if not paged:
        raise KnobNotApplicable(f"no observation of a read tool holds a list longer than {size}")
    spec = split_world(task.world)
    spec = spec.model_copy(update={"paginate": {**spec.paginate, **paged}}, deep=True)
    prog = paginate_program(task.oracle, base.run, paged)
    segs = [s.model_copy(update={"steps": [x.id for x in prog.steps if x.id.split("__p", 1)[0] in set(s.steps)]}) for s in base.comp.segments]
    t, c = _derived(base, task.model_copy(update={"world": spec.ref(), "oracle": prog}, deep=True), f"p{size}", {"segments": segs})
    return t, c


def _same(a: Any, b: Any) -> bool:
    return cjson(a) == cjson(b) or _loose(a, b)


def _first_use(p: Program, loc: str) -> int:
    sid = loc.split(".", 1)[0].split("/", 1)[0]
    return next((i for i, s in enumerate(p.steps) if s.id == sid), len(p.steps))


async def knob_actor(base: Composed, ctx: FalsifyContext, writer: RoleClient | None, *, actor: str = "requester", tag: str = "knob:actor") -> tuple[Task, Composition]:
    task = base.task
    if writer is None:
        raise KnobNotApplicable("the actor knob rewrites the instruction: it needs the instruction writer")
    lits = [e for e in base.run.taint if not e.has("structural")]
    # the value's FIRST use must come from the instruction alone (later uses may also see it in observations: the ask
    # comes before the first use; compose-1: requiring every use to be instruction-only left no candidate on real tasks)
    first: dict[str, Any] = {}
    for x in lits:
        k = cjson(x.value)
        if k not in first or _first_use(task.oracle, x.loc) < _first_use(task.oracle, first[k].loc):
            first[k] = x
    cands = [e for e in first.values() if e.has("instruction") and _statable(e.value) and not e.loc.startswith("answer")
             and not (e.has("observation") or e.has("given") or e.has("tool_spec"))]
    if not cands:
        raise KnobNotApplicable("no literal comes from the instruction alone")
    e = cands[0]
    topic = re.sub(r"[^a-z0-9_]+", "_", e.loc.split(".", 1)[1].split("/", 1)[0].lower()).strip("_") or "detail"
    spec = split_world(task.world)
    actors = [a.model_copy(deep=True) for a in spec.actors]
    a = next((x for x in actors if x.name == actor), None)
    if a is None:
        a = ActorSpec(name=actor)
        actors.append(a)
    if topic in a.facts and cjson(a.facts[topic]) != cjson(e.value):
        topic = f"{topic}_{len(a.facts)}"
    a.facts[topic] = e.value
    spec = spec.model_copy(update={"actors": actors}, deep=True)
    i = min(_first_use(task.oracle, x.loc) for x in lits if _same(x.value, e.value))
    pre = PART.match(task.oracle.steps[i].id).group(0) if i < len(task.oracle.steps) and PART.match(task.oracle.steps[i].id) else ""  # type: ignore[union-attr]
    sid = f"{pre}ask{sum(1 for s in task.oracle.steps if s.tool == ACTOR_TOOL) + 1}"
    steps = list(task.oracle.steps)
    steps.insert(i, Step(id=sid, tool=ACTOR_TOOL, args={"actor": Lit(v=actor), "topic": Lit(v=topic)}))
    prog = Program(steps=steps, answer=task.oracle.answer)
    stub = task.model_copy(update={"world": spec.ref(), "oracle": prog, "id": f"{task.id}~actor-probe"}, deep=True)
    base_sites = await taint_sites(stub, ctx)
    tools = await (await ctx.world(stub)).tools()

    async def check(ins: str) -> None:
        new = await taint_sites(stub.model_copy(update={"instruction": ins}, deep=True), ctx) - base_sites
        if new:
            raise ValueError("the instruction loses information the agent needs, or gives away a decision: " + "; ".join(sorted(new)[:6]))

    wr = await write_instruction(writer, job=ACTOR_JOB, parts=[{"part": 1, "instruction": task.instruction}], fallback=task.instruction, check=check,
                                 must_keep=_quotes(task.contract, task.instruction), must_not_state=[e.value], must_mention=[actor], tools=tools, tag=tag,
                                 fallback_source="kept")
    if wr.source != "writer":
        raise KnobNotApplicable(f"the writer could not move the value to the actor: {wr.error[:300]}")
    segs = [s.model_copy(update={"steps": [x.id for x in prog.steps if part_of_step(x.id, base.comp) == s.index]}) for s in base.comp.segments]
    t, c = _derived(base, stub.model_copy(update={"instruction": wr.instruction}, deep=True), "actor", {"segments": segs})
    return t, c


async def knob_derive(base: Composed, ctx: FalsifyContext, writer: RoleClient | None, *, tag: str = "knob:derive") -> tuple[Task, Composition]:
    task = base.task
    if writer is None:
        raise KnobNotApplicable("the derive knob rewrites the instruction: it needs the instruction writer")
    lits = [e for e in base.run.taint if not e.has("structural")]
    cands = [e for e in lits if e.has("instruction") and not e.has("tool_spec") and _statable(e.value)
             and all(x.has("observation") for x in lits if _same(x.value, e.value))]
    cands.sort(key=lambda e: (not isinstance(e.value, str), -len(str(e.value))))  # a name or an id before a number
    if not cands:
        raise KnobNotApplicable("no literal is both stated in the instruction and shown by an earlier observation")
    vals = _union([[cands[0].value]])
    base_sites = await taint_sites(task, ctx)
    tools = await (await ctx.world(task)).tools()
    intent = task.intent.model_copy(update={"derive": _union([task.intent.derive, vals])})

    async def check(ins: str) -> None:
        new = await taint_sites(task.model_copy(update={"instruction": ins, "intent": intent}, deep=True), ctx) - base_sites
        if new:
            raise ValueError("the instruction loses information the agent needs, or gives away a decision: " + "; ".join(sorted(new)[:6]))

    wr = await write_instruction(writer, job=DERIVE_JOB, parts=[{"part": 1, "instruction": task.instruction}], fallback=task.instruction, check=check,
                                 must_keep=_quotes(task.contract, task.instruction), must_not_state=vals, tools=tools, tag=tag, fallback_source="kept")
    if wr.source != "writer":
        raise KnobNotApplicable(f"the writer could not drop the value: {wr.error[:300]}")
    return _derived(base, task.model_copy(update={"instruction": wr.instruction, "intent": intent}, deep=True), "derive", {})


async def apply_knob(spec: KnobSpec, base: Composed, *, ctx: FalsifyContext, writer: RoleClient | None = None) -> Variant:
    """One knob on ``base`` (a composition or a ``single`` task). The variant is kept only if every feature the knob
    names moves in the predicted direction; a knob that cannot act raises ``KnobNotApplicable``. Depth and horizon are
    compositions (``compose``, ``ladder``), not rewrites of one task."""
    if spec.kind in ("depth", "horizon"):
        raise KnobNotApplicable(f"{spec.kind} adds a part: use compose() or ladder()")
    if spec.kind == "distractors":
        t, c = await knob_distractors(base, ctx, int(spec.params.get("k", 3)))
    elif spec.kind == "paginate":
        t, c = await knob_paginate(base, ctx, int(spec.params.get("size", 5)))
    elif spec.kind == "actor":
        t, c = await knob_actor(base, ctx, writer, actor=str(spec.params.get("actor", "requester")))
    else:
        t, c = await knob_derive(base, ctx, writer)
    r = await _verify_same(base, t, ctx, what=spec.kind)
    if c.segments and len(c.segments) > 1:
        edges, dep, dep_u = await dependency(await ctx.world(t), part_programs(t, c), seed=t.seed, instruction=t.instruction, given=t.intent.given)
        c = c.model_copy(update={"edges": edges, "dep": dep, "dep_unconfirmed": dep_u, "carry": carry_depth(edges)})
    c = c.model_copy(update={"horizon": len(r.calls), "writes": sum(1 for x in r.calls if not x.read_only)})
    sig = behaviour_signature(t, r, c)
    rec = KnobRecord(kind=spec.kind, params=spec.params, predicted=KNOB_PREDICTED[spec.kind], moves=list(KNOB_MOVES[spec.kind]),
                     before={m: base.comp.signature.get(m) for m in KNOB_MOVES[spec.kind]}, after={m: sig.get(m) for m in KNOB_MOVES[spec.kind]})
    rec.moved = moved(base.comp.signature, sig, rec.moves, rec.predicted)
    rec.kept = rec.moved
    rec.reason = "" if rec.moved else "the behaviour signature did not move"
    c = c.model_copy(update={"signature": sig, "knobs": [*base.comp.knobs, rec]})
    return Variant(Composed(task=t, comp=c, run=r), rec)


# ---------------------------------------------------------------------------------------------------------- ladders

class Rung(CrucibleModel):
    n: int
    label: str
    knobs: list[KnobSpec] = Field(default_factory=list)
    task_id: str = ""
    parts: int = 1
    horizon: int = 0
    dep: int = 0
    signature: dict[str, Any] = Field(default_factory=dict)
    moved: bool = True
    kept: bool = True
    reason: str = ""
    status: str = "candidate"  # candidate | certified | provisional | rejected | dropped
    certificate: dict[str, Any] | None = None


DEFAULT_KNOBS: tuple[KnobSpec, ...] = (KnobSpec(kind="distractors", params={"k": 3}), KnobSpec(kind="paginate", params={"size": 5}),
                                       KnobSpec(kind="actor"), KnobSpec(kind="derive"))


@dataclass
class Ladder:
    rungs: list[Rung] = field(default_factory=list)
    candidates: list[Composed] = field(default_factory=list)  # one per kept rung, in rung order


async def ladder(links: Sequence[Task], *, ctx: FalsifyContext, writer: RoleClient | None = None, start: int = 1,
                 knobs: Sequence[KnobSpec] = DEFAULT_KNOBS, id_prefix: str | None = None, parts: Sequence[int] | None = None) -> Ladder:
    """For one chain of links: rungs of increasing parts (depth when dep rises, else horizon), then the knobs applied in
    turn to the longest composition. Each kept rung is a new candidate task the caller certifies again. ``parts`` picks
    the part counts of the rungs (default: every count from ``start``)."""
    out = Ladder()
    prev: Composed | None = None
    pre = id_prefix or f"{links[0].domain}/ladder/{content_hash([t.id for t in links], 8)}"
    for m in (sorted(p for p in parts if 1 <= p <= len(links)) if parts else range(start, len(links) + 1)):
        try:
            c = await single(links[0], ctx) if m == 1 else await compose(links[:m], ctx=ctx, writer=writer, task_id=f"{pre}/k{m}")
        except CompositionError as e:
            out.rungs.append(Rung(n=len(out.rungs), label=f"parts={m}", parts=m, kept=False, moved=False, reason=str(e)[:300], status="dropped"))
            break
        label = "base" if prev is None else ("depth" if c.comp.dep > prev.comp.dep else "horizon")
        mv = prev is None or moved(prev.comp.signature, c.comp.signature, KNOB_MOVES[label] if label != "base" else ())
        out.rungs.append(Rung(n=len(out.rungs), label=label, knobs=[KnobSpec(kind=label)] if label != "base" else [], task_id=c.task.id, parts=m,  # type: ignore[arg-type]
                              horizon=c.comp.horizon, dep=c.comp.dep, signature=c.comp.signature, moved=mv, kept=mv,
                              reason="" if mv else "the behaviour signature did not move"))
        if mv:
            out.candidates.append(c)
            prev = c
    cur = prev
    for k in knobs:
        if cur is None:
            break
        try:
            v = await apply_knob(k, cur, ctx=ctx, writer=writer)
        except KnobNotApplicable as e:
            out.rungs.append(Rung(n=len(out.rungs), label=k.kind, knobs=[k], kept=False, moved=False, reason=str(e)[:300], status="dropped"))
            continue
        out.rungs.append(Rung(n=len(out.rungs), label=k.kind, knobs=[k], task_id=v.composed.task.id, parts=len(v.composed.comp.segments),
                              horizon=v.composed.comp.horizon, dep=v.composed.comp.dep, signature=v.composed.comp.signature, moved=v.record.moved,
                              kept=v.record.kept, reason=v.record.reason))
        if v.record.kept:
            out.candidates.append(v.composed)
            cur = v.composed
    return out


# ---------------------------------------------------------------------------------------------------------- selection

class Target(CrucibleModel):
    """A target distribution over (horizon bin, depth) for selecting compositions; dep >= 1 is preferred."""

    horizon_bins: list[tuple[int, int]] = Field(default_factory=lambda: [(1, 14), (15, 40), (41, 10_000)])
    horizon_share: list[float] = Field(default_factory=lambda: [0.2, 0.6, 0.2])
    dep_share: dict[int, float] = Field(default_factory=lambda: {0: 0.1, 1: 0.3, 2: 0.6})
    dep_cap: int = 2  # depths at or above it share one cell

    def cell(self, c: Composition) -> tuple[int, int]:
        hb = next((i for i, (lo, hi) in enumerate(self.horizon_bins) if lo <= c.horizon <= hi), len(self.horizon_bins) - 1)
        return hb, min(c.dep, self.dep_cap)

    def share(self, cell: tuple[int, int]) -> float:
        return self.horizon_share[cell[0]] * self.dep_share.get(cell[1], 0.0)


def select(pool: Sequence[Composition], k: int, target: Target | None = None) -> list[Composition]:
    """Greedy selection of ``k`` compositions toward ``target``: each pick fills the cell furthest below its share
    (ties: deeper, then longer)."""
    t = target or Target()
    left = sorted(pool, key=lambda c: (-c.dep, -c.horizon, c.id))
    out: list[Composition] = []
    got: Counter[tuple[int, int]] = Counter()
    def deficit(c: Composition, n: int) -> tuple[float, int, int]:
        cell = t.cell(c)
        return (t.share(cell) * n - got[cell], c.dep, c.horizon)

    while left and len(out) < k:
        n = len(out) + 1
        best = max(left, key=lambda c: deficit(c, n))
        left.remove(best)
        out.append(best)
        got[t.cell(best)] += 1
    return out


# ---------------------------------------------------------------------------------------------------------- chains (swarm)

FOLLOWUP_CELLS = (
    ("a follow-up whose work needs the results of the earlier work in this session: it acts on records the earlier work changed or created, "
     "found through what the system now shows"),
    "a follow-up that reports a figure or list computed from the records as the earlier work left them, plus a write that depends on it",
    "a follow-up decision on a related entity that is only correct given what the earlier work did",
)
FOLLOWUP_RULE = ("Earlier in this session the agent already did the work below; its changes are in the start state shown. Propose follow-up "
                 "tasks that continue it. Refer to the earlier work's results by what was done (\"the invoice you just put on hold\", \"the "
                 "agent you assigned\"), never by restating ids, numbers or names that only the earlier work produced: the agent must find "
                 "them in the system. Prefer follow-ups that also change records (a decision, an update, a note), so that later work can "
                 "build on them in turn.")


def effects_summary(r: Run, limit: int = 24) -> list[str]:
    if r.start is None or r.end is None:
        return []
    out = []
    for ch in diff(r.start, r.end)[:limit]:
        if ch.kind == "update":
            out.append(f"{ch.addr}: {cjson(ch.old)[:60]} -> {cjson(ch.new)[:60]}")
        else:
            out.append(f"{ch.kind} {ch.addr}")
    return out


def followup_env(env: Any, done: Sequence[Task], prefix: Program | None, effects: Sequence[str], *, seeds: Sequence[int] | None = None) -> Any:
    """A ``pipeline.DomainEnv`` whose World starts after ``prefix`` (the earlier links) and whose proposer is asked for
    follow-ups (``FOLLOWUP_CELLS``)."""
    import dataclasses

    from crucible.synth.pipeline import DomainEnv

    if not done:
        return dataclasses.replace(env, seeds=tuple(seeds or env.seeds))
    ctxt = "\n".join(f"{k + 1}. {t.instruction}" for k, t in enumerate(done))
    desc = f"{env.description}\n\n{FOLLOWUP_RULE}\nEarlier work:\n{ctxt}\nIts changes:\n" + "\n".join(effects)
    return DomainEnv(name=env.name, description=desc, seeds=tuple(seeds or env.seeds), normalizers=env.normalizers, slice=env.slice, max_rows=env.max_rows,
                     cells=FOLLOWUP_CELLS, world=lambda s: VariantSpec(base=env.world(s), prefix=prefix).ref())


class LinkChoice(CrucibleModel):
    link: int
    candidates: list[dict[str, Any]] = Field(default_factory=list)
    chosen: str | None = None
    dep: int = 0
    horizon: int = 0
    note: str = ""


async def build_chain(env: Any, roles: Any, *, length: int, run_dir: Path, archive_root: Path, seed: int, candidates: int = 2,
                      cfg: Any = None, cost: Callable[[], dict[str, Any]] | None = None, families: Mapping[str, Any] | None = None,
                      ctx: FalsifyContext | None = None, on_link: Callable[[int, list[Task], LinkChoice], None] | None = None,
                      retries: int = 1, allow_horizon: bool = True, max_horizon: int = 48) -> tuple[list[Task], list[LinkChoice]]:
    """Grow a chain of ``length`` links with the swarm (``pipeline.Runner``: propose, oracle, contract; no falsifiers,
    no repair): link j is proposed on the World after links 1..j-1. The candidate kept is the one with a confirmed
    state edge into it, then the deepest, then one that carries information from earlier parts, then the longest; a
    round without a state edge is retried (``retries``) before settling. ``allow_horizon=False`` refuses a link nothing
    depends on. Stops early when no candidate composes."""
    from crucible.synth.pipeline import RunConfig, Runner

    own = ctx is None
    ctx = ctx or FalsifyContext()
    links: list[Task] = []
    log: list[LinkChoice] = []
    prefix: Program | None = None
    effects: list[str] = []
    try:
        for j in range(1, length + 1):
            choice = LinkChoice(link=j)
            best: tuple[tuple[int, int, int], Task, Composed | None] | None = None
            for attempt in range(retries + 1):
                base_env = followup_env(env, links, prefix, effects, seeds=(seed,))
                c = cfg or RunConfig()
                rcfg = RunConfig(n_tasks=candidates, batch=candidates, calibrate=False, paid=False, repair=False, concurrency=candidates,
                                 oracle_retries=c.oracle_retries, max_mutants=c.max_mutants)
                rid = f"link{j:02d}" + (f"-r{attempt}" if attempt else "")
                # the task ids carry the chain's directory name: two chains of one run never share a link id
                runner = Runner(base_env, roles, rcfg, Path(run_dir) / rid, archive_root, run_id=f"{Path(run_dir).name}-{rid}", cost=cost, families=families)
                await runner.survey()
                recs = await runner.propose()
                await asyncio.gather(*(runner.build(r) for r in recs))
                runner.report.tasks = recs
                runner.save()
                if runner.stop:
                    choice.note = runner.stop
                    break
                ready = {rec.n for rec, _t, _r in runner.ready}
                for rec in recs:
                    if rec.n not in ready:
                        choice.candidates.append({"id": rec.id, "skill": rec.skill, "status": rec.status, "reasons": rec.reasons[:3], "note": rec.note[:200]})
                for rec, task, _r in sorted(runner.ready, key=lambda x: x[0].n):
                    row: dict[str, Any] = {"id": task.id, "skill": task.intent.skill, "instruction": task.instruction}
                    try:
                        if links:
                            cc = await compose([*links, task], ctx=ctx, task_id=f"{task.id}/probe")
                            into = sum(1 for e2 in cc.comp.edges if e2.confirmed and e2.dst == len(links))
                            carried = sum(1 for e2 in cc.comp.edges if e2.kind == "carry" and e2.dst == len(links))
                            wr = cc.comp.segments[-1].writes
                            # a state edge into the new link first; then a link that writes (only a write can be depended on
                            # later: compose-1's read-only re-check links kept dep at 1); then depth, carried information, length
                            key = (min(into, 1), min(wr, 1), cc.comp.dep, min(carried, 1), cc.comp.carry, cc.comp.horizon)
                            row.update(dep=cc.comp.dep, carry=cc.comp.carry, horizon=cc.comp.horizon, edges_into=into, carry_into=carried, writes=wr)
                        else:
                            sc = await single(task, ctx)
                            cc = None
                            key = (0, min(sc.comp.writes, 1), 0, 0, 0, sc.comp.horizon)
                            row.update(dep=0, horizon=sc.comp.horizon, writes=sc.comp.writes)
                    except (CompositionError, KnobNotApplicable, PrefixError) as ex:
                        row["refused"] = str(ex)[:300]
                        choice.candidates.append(row)
                        continue
                    choice.candidates.append(row)
                    if key[5] > max_horizon:  # compose-1: one swarm link alone made 135 calls (a vendor sweep)
                        row["refused"] = f"over the horizon cap ({key[5]} > {max_horizon} calls)"
                        continue
                    if links and not (key[0] or key[3] or allow_horizon):
                        continue  # a link nothing depends on adds horizon only
                    if best is None or key > best[0]:
                        best = (key, task, cc)
                if best is not None and (best[0][0] or not links or attempt == retries):
                    break  # a state dependency into the new link; else try again for one before settling
            if best is None:
                choice.note = choice.note or "no candidate composes after the earlier links"
                log.append(choice)
                if on_link:
                    on_link(j, links, choice)
                break
            key, task, cc = best
            links.append(task)
            choice.chosen, choice.dep, choice.horizon = task.id, key[2], key[5]
            choice.note = choice.note or ("" if not links[:-1] or key[0] else ("carried information only" if key[3] else "horizon only"))
            log.append(choice)
            prefix = concat([rename(t.oracle, f"t{k + 1}_") for k, t in enumerate(links)])
            w = await ctx.world(_stub(env.world(seed), seed, env.name))
            r = (await run_parts(w, [prefix], seed=seed))[0]
            effects = effects_summary(r)
            if on_link:
                on_link(j, links, choice)
    finally:
        if own:
            await ctx.close()
    return links, log


# ---------------------------------------------------------------------------------------------------------- certification

MUT_PER_CALL = 12  # MUT proposals per oracle call when certifying a composition


def agent_budget_for(horizon: int, base: Any = None, *, per_call: float = 1.5, slack: int = 6, obs_chars: int = 1500) -> Any:
    """An ALT/ATK budget that scales with the oracle's horizon (a 30-call task needs more than the default 24 calls),
    with shorter observations so the transcript stays affordable."""
    from crucible.synth.falsify.agents import AgentBudget

    b = base or AgentBudget()
    calls = max(b.max_calls, math.ceil(per_call * horizon) + slack)
    return AgentBudget(max_calls=calls, max_turns=calls + 6, obs_chars=min(b.obs_chars, obs_chars))


async def certify_composed(cands: Sequence[Composed], env: Any, roles: Any, *, run_dir: Path, archive_root: Path, run_id: str, cfg: Any = None,
                           cost: Callable[[], dict[str, Any]] | None = None, families: Mapping[str, Any] | None = None) -> Any:
    """Each candidate through the full falsifier population, adjudication, repair and certification of the pipeline
    (``pipeline.Runner.refine_one``: the same code as any proposed task), the panel calibrated on the candidates. ALT
    and ATK budgets scale with the longest horizon. Returns the run report (``report.json`` / ``report.md`` under
    ``run_dir``)."""
    import dataclasses
    import time

    from crucible.synth.pipeline import RunConfig, Runner, TaskRecord

    if not cands:
        raise ValueError("nothing to certify")
    c0 = cfg or RunConfig()
    hz = max(c.comp.horizon for c in cands)
    # MUT's proposal cap scales with the horizon: at 240 a 37-49 call composite kept only 42-52 distinct effects (n < 59),
    # at 600 it kept 173-189 (compose-1)
    c1 = dataclasses.replace(c0, alt_budget=agent_budget_for(hz, c0.alt_budget), atk_budget=agent_budget_for(hz, c0.atk_budget, per_call=1.0),
                             max_mutants=max(c0.max_mutants, MUT_PER_CALL * hz))
    seeds = tuple(sorted({c.task.seed for c in cands}))
    e1 = dataclasses.replace(env, seeds=seeds)
    runner = Runner(e1, roles, c1, Path(run_dir), archive_root, run_id=run_id, cost=cost, families=families)
    await runner.survey()
    recs = [TaskRecord(n=k + 1, id=c.task.id, seed=c.task.seed, skill=c.task.intent.skill[:120], stage="contract", oracle_writes=c.comp.writes)
            for k, c in enumerate(cands)]
    runner.report.tasks = recs
    runner.ready = [(rec, c.task, c.run) for rec, c in zip(recs, cands, strict=True)]
    for rec, c in zip(recs, cands, strict=True):
        d = Path(run_dir) / "tasks" / f"t{rec.n:02d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "composed.json").write_text(cjson(c.dump()))
    await runner.calibrate_panel()
    sem = asyncio.Semaphore(c1.concurrency)

    async def one(rec: Any, task: Task) -> None:
        async with sem:
            if runner.stop:
                rec.status, rec.reasons = "budget", [runner.stop[:300]]
                return
            await runner.refine_one(rec, task)
            runner.save()

    await asyncio.gather(*(one(r, c.task) for r, c in zip(recs, cands, strict=True)))
    runner.report.stopped = runner.stop
    runner.report.finished = time.time()
    runner.save()
    return runner.report

