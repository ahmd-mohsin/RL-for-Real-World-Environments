"""The planter: contract and World mutation that plants one defect of known class into a certified task
(part2-paper-core §8; the yardstick of §9's loop, level 2).

**Operators** (one per row of §8's table; each is parameterised and versioned, ``PlantOp(name, version, variant,
params)``, so a planter *population* exists for fictitious play and the dev and locked sets can use disjoint variants):

| operator | plants | mechanism (variants) |
|---|---|---|
| ``drop_required`` | D1 | one R pattern moves to A (the effect is still allowed, no longer required) |
| ``widen_frame`` | D2 | A gains ``(app, coll, *, *)`` with a plausible why (``coll``: one written collection; ``app``: the whole App) |
| ``relax_value`` | D3 | ``eq`` → ``any`` on an R value or where predicate (``eq_to_any``); a tolerance × ``factor`` on an R or answer check, an exact numeric answer → ``approx`` (``tolerance``) |
| ``vacuous_required`` | D4 | an R pattern is replaced by a ``state`` pattern s₀ already satisfies (``in [old, new]`` for an update, count ≥ 0 for an insert) and its effect is allowed |
| ``hide_info`` | D5 | the instruction sentence carrying a literal π* needs is deleted (``sentence``), or the literal is replaced by a vague phrase (``redact``) |
| ``leak_decision`` | D7 | a ``derive`` value is appended to ι (``reason``, ``because``; ``humanised`` spells ``price_variance`` as "price variance") |
| ``plant_affordance`` | D8 | the World gains a write tool: a one-call ``quick_*`` tool that does π*'s writes for a key (``solve_button``), or an ``admin_set_*`` tool that writes the target fields directly (``field_setter``) |
| ``over_constrain`` | D6 | an unstated trace constraint π* happens to satisfy: ``before`` (a read before a write), ``at_most`` (π*'s read count), ``never`` (a read π* does not make) |
| ``instruction_drift`` | D9 | every R pattern on an entity π* writes gets key ``*``, and where predicates naming that key become ``any`` |
| ``break_solvability`` | D10 | an R ``eq`` value becomes one no tool can produce (``unknown_code``: a suffixed code; numbers × 1000 + 7) |
| ``inject_nondet`` | D11 | a write tool of π* stamps a run-noise token on the entity it changes, and C reads it (``pinned``: R requires the token of one episode; ``allowed``: A allows it with no normalizer) |

**Witnesses (executed, §9.3).** A planted "defect" without a witness is not a defect, so every candidate plant is
discarded unless its witness is found and then re-validated by execution (``validate``):
- **lenient** (D1, D2, D3, D4, D9; D8 over the planted World): a run that *passes* the planted task and *fails* the
  original. Candidates are the source task's own MUT neighbourhood and NULL agents (and, for D8, the planted tool
  called directly), searched in a per-class family order and stopped at the first witness.
- **strict** (D6, the mirror case §8 leaves implicit): a run equivalent to π* under the *stated* constraints that passes
  C and fails C′: a valid solution the planted contract rejects.
- **hidden** (D5): π* re-executed with taint under ι′ has a literal that is unsourced, which ι sourced.
- **leak** (D7): π* still passes under ι′, and ι′ quotes the derive value (exactly, or humanised) at a recorded span
  that ι does not contain. This evidence is independent of TAINT's matcher, so a humanised leak is a valid plant that
  TAINT may miss: exactly what recall must measure.
- **unsolvable** (D10): π* fails C′ on the planted pattern, the planted value occurs nowhere the agent can read it (s₀,
  the tool catalogue, ι, π*'s observations), and no candidate run passes C′.
- **nondet** (D11): two consecutive executions of π* from ``reset(seed)`` in the planted World differ, in verdict
  (``pinned``) or in Δ (``allowed``).
Every lenient, strict, hidden and leak plant must also keep π* passing (else it changed more than one thing).

**Planted Worlds** are data: ``WorldRef(factory="crucible.synth.meta.plant:planted_world", params={"base": <the source
WorldRef>, "patches": [...]})`` builds the source World and wraps it (``PlantedWorld``). A patch is an affordance (the
extra tool, replaying π*'s recorded write calls for the given key, or writing fields through ``perturb``) or a nondet
stamp (an overlay in ``canon()``, a token per episode; the episode counter survives ``reset``, which is the run noise).

**Natural-likeness (§9.3).** ``features(plant)`` describes a plant by class, contract-diff size, instruction-edit size,
the part of the task it moved (R, A, Q, M, ι, World), whether a wildcard appeared, the type of literal moved, whether the
collection is one π* writes, and the witness length. ``mmd`` is the RBF-kernel maximum mean discrepancy between two such
samples, with the bandwidth set by the median heuristic on the *reference* sample (so a candidate cannot move it).
"""

from __future__ import annotations

import copy
import difflib
import math
import re
from collections.abc import Iterable, Sequence
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import MISSING, Address, CanonState, cjson, content_hash, diff, is_number, same_diff
from crucible.synth.contract import Contract, EffectPattern, PAny, PApprox, PEq, PIn, TraceConstraint, is_glob, unglob
from crucible.synth.falsify.base import FalsifyContext, delta_of, entity_index, equivalent, key_text, typed_like, written_entities
from crucible.synth.falsify.mut import MUT, key_colls
from crucible.synth.falsify.null import NULL, borrowed_args
from crucible.synth.iface import Obs, Perturbable, Perturbation, SynthWorld, ToolInfo
from crucible.synth.infer import _lit_glob
from crucible.synth.program import Lit, Program, Run, Step, _loose_eq, leaves, text_spans
from crucible.synth.task import Task, WorldRef

PLANTED_FACTORY = "crucible.synth.meta.plant:planted_world"
WitnessKind = Literal["lenient", "strict", "hidden", "leak", "unsolvable", "nondet"]


# ---------------------------------------------------------------------------------------------------------- operators

class OpInfo(CrucibleModel):
    defect: str
    witness: WitnessKind
    variants: list[str]
    families: list[str] = Field(default_factory=list)  # candidate families searched first for a lenient/strict witness


OPS: dict[str, OpInfo] = {
    "drop_required": OpInfo(defect="D1", witness="lenient", variants=["base"], families=["DropStep", "Truncate", "minus_last_write"]),
    "widen_frame": OpInfo(defect="D2", witness="lenient", variants=["coll", "app"], families=["ExtraWrite", "write_all", "shortcut"]),
    "relax_value": OpInfo(defect="D3", witness="lenient", variants=["eq_to_any", "tolerance"], families=["PerturbValue", "WrongAnswer", "SwapEntity"]),
    "vacuous_required": OpInfo(defect="D4", witness="lenient", variants=["base"], families=["DropStep", "Truncate", "minus_last_write", "noop"]),
    "hide_info": OpInfo(defect="D5", witness="hidden", variants=["sentence", "redact"]),
    "over_constrain": OpInfo(defect="D6", witness="strict", variants=["before", "at_most", "never"], families=["DropStep", "DupStep", "Reorder"]),
    "leak_decision": OpInfo(defect="D7", witness="leak", variants=["reason", "because", "humanised"]),
    "plant_affordance": OpInfo(defect="D8", witness="lenient", variants=["solve_button", "field_setter"], families=["shortcut", "planted"]),
    "instruction_drift": OpInfo(defect="D9", witness="lenient", variants=["entity"], families=["SwapEntity", "PerturbValue", "ExtraWrite"]),
    "break_solvability": OpInfo(defect="D10", witness="unsolvable", variants=["unknown_code"]),
    "inject_nondet": OpInfo(defect="D11", witness="nondet", variants=["pinned", "allowed"]),
}


class PlantOp(CrucibleModel):
    """One parameterised, versioned operator: a member of a planter population."""

    name: str
    version: int = 1
    variant: str = "base"
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def info(self) -> OpInfo:
        return OPS[self.name]

    @property
    def defect(self) -> str:
        return OPS[self.name].defect

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}:{self.variant}"


class Planter(CrucibleModel):
    """A planter population member: a versioned list of operators. ``max_per_op`` caps plants per operator per task."""

    id: str = "planter"
    version: int = 1
    ops: list[PlantOp] = Field(default_factory=list)
    max_per_op: int = 2

    @property
    def key(self) -> str:
        return f"{self.id}@{self.version}"

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 24)


def base_planter() -> Planter:
    """The dev planter: every operator, in the variants the dev set uses."""
    ops = [PlantOp(name="drop_required"), PlantOp(name="widen_frame", variant="coll"), PlantOp(name="widen_frame", variant="app"),
           PlantOp(name="relax_value", variant="eq_to_any"), PlantOp(name="relax_value", variant="tolerance", params={"factor": 100}),
           PlantOp(name="vacuous_required"), PlantOp(name="hide_info", variant="sentence"), PlantOp(name="leak_decision", variant="reason"),
           PlantOp(name="plant_affordance", variant="solve_button"), PlantOp(name="over_constrain", variant="before"),
           PlantOp(name="over_constrain", variant="at_most"), PlantOp(name="instruction_drift", variant="entity"),
           PlantOp(name="break_solvability", variant="unknown_code"), PlantOp(name="inject_nondet", variant="pinned")]
    return Planter(id="dev", ops=ops)


def held_out_planter() -> Planter:
    """Held-out variants (different phrasing, different mechanisms) for a locked set; disjoint from ``base_planter``."""
    ops = [PlantOp(name="hide_info", variant="redact", params={"phrase": "the item"}), PlantOp(name="leak_decision", variant="humanised"),
           PlantOp(name="leak_decision", variant="because"), PlantOp(name="plant_affordance", variant="field_setter"),
           PlantOp(name="over_constrain", variant="never"), PlantOp(name="inject_nondet", variant="allowed"),
           PlantOp(name="relax_value", variant="tolerance", version=2, params={"factor": 10})]
    return Planter(id="held-out", ops=ops)


# ---------------------------------------------------------------------------------------------------------- plants

class Witness(CrucibleModel):
    kind: WitnessKind
    program: Program | None = None  # lenient / strict: the run that tells the planted task from the source
    source_op: str | None = None  # which candidate family produced it
    detail: dict[str, Any] = Field(default_factory=dict)


class Plant(CrucibleModel):
    schema_version: str = "crucible.synth.plant/v1"
    id: str
    defect: str
    op: PlantOp
    planter: str = ""
    domain: str
    source_id: str
    source_version: int
    source_digest: str
    task: Task
    change: dict[str, Any] = Field(default_factory=dict)  # what moved (target, pattern, field, coll, literal, ...)
    witness: Witness
    features: dict[str, float] = Field(default_factory=dict)


class Discard(CrucibleModel):
    op: str
    source_id: str
    site: str
    reason: str


class PlantBatch(CrucibleModel):
    kept: list[Plant] = Field(default_factory=list)
    discarded: list[Discard] = Field(default_factory=list)

    def extend(self, other: PlantBatch) -> None:
        self.kept += other.kept
        self.discarded += other.discarded


class _Draft(CrucibleModel):
    task: Task
    site: str
    change: dict[str, Any] = Field(default_factory=dict)
    program: Program | None = None  # a directly constructed witness candidate (affordance, never-read)


# ---------------------------------------------------------------------------------------------------------- planted Worlds

class PlantedWorld:
    """A ``SynthWorld`` wrapper that applies World patches (see the module docstring). Snapshots carry the overlay."""

    def __init__(self, inner: SynthWorld, patches: Sequence[dict[str, Any]]):
        self.inner = inner
        self.patches = [dict(p) for p in patches]
        self.overlay: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.episodes = 0  # survives reset: the run noise a nondet patch reads
        self._afford = {p["tool"]: p for p in self.patches if p.get("kind") == "affordance"}
        self._nondet = {p["tool"]: p for p in self.patches if p.get("kind") == "nondet"}

    async def reset(self, seed: int) -> Any:
        tok = await self.inner.reset(seed)
        self.overlay = {}
        self.episodes += 1
        return tok

    async def tools(self) -> list[ToolInfo]:
        out = list(await self.inner.tools())
        for p in self._afford.values():
            props = {p["key_arg"]: {"type": p.get("key_type", "string")}, **{f: {} for f in p.get("fields", [])}}
            out.append(ToolInfo(name=p["tool"], description=p.get("description", ""), input_schema={"type": "object", "properties": props,
                                "required": [p["key_arg"]]}, read_only=False, writes=list(p.get("writes") or []) or None))
        return sorted(out, key=lambda t: t.name)

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        if tool in self._afford:
            return await self._affordance(self._afford[tool], args)
        nd = self._nondet.get(tool)
        before = await self.inner.canon() if nd is not None else None
        obs = await self.inner.call(tool, args)
        if nd is not None and obs.ok and before is not None:
            after = await self.inner.canon()
            token = nd["tokens"][(self.episodes - 1) % len(nd["tokens"])]
            for ch in diff(before, after or CanonState()):
                if ch.kind == "update":
                    self.overlay.setdefault(ch.addr.entity, {})[nd["field"]] = token
            if isinstance(obs.value, dict):
                obs = Obs(value={**obs.value, nd["field"]: token})
        return obs

    async def _affordance(self, p: dict[str, Any], args: dict[str, Any]) -> Obs:
        key = args.get(p["key_arg"])
        if key is None:
            return Obs(error=f"missing {p['key_arg']}")
        if p["mode"] == "set":
            if not isinstance(self.inner, Perturbable):
                return Obs(error="not supported")
            fields = {f: args[f] for f in p.get("fields", []) if f in args}
            try:
                await self.inner.perturb(Perturbation(op="set", app=p["app"], coll=p["coll"], key=str(key), fields=fields))
            except (KeyError, ValueError) as e:
                return Obs(error=f"{type(e).__name__}")
            return Obs(value={p["key_arg"]: key, "updated": sorted(fields)})
        for tool, cargs in p["calls"]:
            sub = {k: (typed_like(str(key), v) if key_text(v) == key_text(p["key"]) else v) for k, v in cargs.items()}
            o = await self.inner.call(tool, sub)
            if not o.ok:
                return o
        return Obs(value={p["key_arg"]: key, "done": True})

    async def snapshot(self) -> Any:
        return (await self.inner.snapshot(), copy.deepcopy(self.overlay))

    async def restore(self, ref: Any) -> None:
        inner_ref, overlay = ref
        await self.inner.restore(inner_ref)
        self.overlay = copy.deepcopy(overlay)

    async def canon(self) -> CanonState | None:
        s = await self.inner.canon()
        if s is None or not self.overlay:
            return s
        apps = copy.deepcopy(s.apps)
        for (app, coll, key), fields in self.overlay.items():
            row = apps.get(app, {}).get(coll, {}).get(key)
            if row is not None:
                row.update(fields)
        return CanonState.build(apps, s.unobserved)

    async def close(self) -> None:
        c = getattr(self.inner, "close", None)
        if c is not None:
            res = c()
            if hasattr(res, "__await__"):
                await res


class PerturbablePlantedWorld(PlantedWorld):
    async def perturb(self, p: Perturbation) -> None:
        await self.inner.perturb(p)  # type: ignore[attr-defined]


async def planted_world(base: dict[str, Any], patches: list[dict[str, Any]]) -> SynthWorld:
    """The ``WorldRef`` factory of a planted World (see the module docstring)."""
    inner = await WorldRef.model_validate(base).build()
    cls = PerturbablePlantedWorld if isinstance(inner, Perturbable) else PlantedWorld
    return cls(inner, patches)


def planted_ref(base: WorldRef, patch: dict[str, Any]) -> WorldRef:
    return WorldRef(factory=PLANTED_FACTORY, params={"base": base.model_dump(mode="json"), "patches": [patch]})


# ---------------------------------------------------------------------------------------------------------- helpers

def _fresh_id(c: Contract, base: str) -> str:
    ids = {x.id for x in [*c.required, *c.allowed, *c.answer, *c.trace, *c.exec]}
    i, out = 0, base
    while out in ids:
        i += 1
        out = f"{base}_{i}"
    return out


def _with_contract(src: Task, c: Contract, **upd: Any) -> Task:
    return src.model_copy(update={"contract": c, **upd}, deep=True)


def _planted_id(src: Task, op: PlantOp, site: str) -> str:
    return f"plant/{op.name}/{content_hash({'src': src.id, 'v': src.version, 'op': op.key, 'site': site, 'p': op.params}, 12)}"


def _sentences(text: str) -> list[str]:
    return [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]


def _values_visible(src: Task, oracle: Run, tools: Sequence[ToolInfo]) -> list[Any]:
    """Every value an agent could read: s₀ cells, the tool catalogue (enums, defaults), π*'s arguments and observations."""
    out: list[Any] = list(oracle.values())
    if oracle.start is not None:
        out += [v for _, v in oracle.start.cells()]
    for t in tools:
        for p in (t.input_schema.get("properties") or {}).values():
            if isinstance(p, dict):
                out += list(p.get("enum", []))
                if "default" in p:
                    out.append(p["default"])
    return out


def _unknown_value(v: Any, suffix: str) -> Any:
    if isinstance(v, bool) or v is None:
        return None
    if is_number(v):
        return v * 1000 + 7
    if isinstance(v, str) and v:
        return f"{v}{suffix}"
    return None


def _pattern_entity(p: EffectPattern) -> tuple[str, str, str] | None:
    if any(is_glob(x) for x in (p.app, p.coll, p.key)):
        return None
    return (unglob(p.app), unglob(p.coll), unglob(p.key))


# ---------------------------------------------------------------------------------------------------------- drafts per operator

def _drafts(src: Task, op: PlantOp, oracle: Run, tools: list[ToolInfo]) -> list[_Draft]:
    c = src.contract
    v = op.variant
    out: list[_Draft] = []
    delta = delta_of(oracle, c)
    written = {(a, co) for a, co, _ in written_entities(delta)}
    if op.name == "drop_required":
        for p in c.required:
            if p.kind == "state":
                continue
            allow = p.model_copy(update={"id": _fresh_id(c, f"a_{p.id}"), "why": op.params.get("why", "this effect is routine and need not be checked")})
            c2 = c.model_copy(update={"required": [q for q in c.required if q.id != p.id], "allowed": [*c.allowed, allow]}, deep=True)
            out.append(_Draft(task=_with_contract(src, c2), site=p.id, change={"target": "required", "pattern": p.id, "coll": f"{p.app}.{p.coll}"}))
    elif op.name == "widen_frame":
        colls = sorted(written) if v == "coll" else sorted({(a, "*") for a, _ in written})
        for app, coll in colls:
            a = EffectPattern(id=_fresh_id(c, f"a_wide_{coll.strip('*') or 'all'}"), app=app, coll=coll, key="*", field="*", kind="any",
                              why=op.params.get("why", "routine edits in this area are expected"))
            c2 = c.model_copy(update={"allowed": [*c.allowed, a]}, deep=True)
            out.append(_Draft(task=_with_contract(src, c2), site=f"{app}.{coll}", change={"target": "allowed", "coll": f"{app}.{coll}", "wildcard": True}))
    elif op.name == "relax_value":
        factor = float(op.params.get("factor", 100))
        for i, p in enumerate(c.required):
            if v == "eq_to_any" and isinstance(p.value, PEq):
                req = list(c.required)
                req[i] = p.model_copy(update={"value": PAny()})
                out.append(_Draft(task=_with_contract(src, c.model_copy(update={"required": req}, deep=True)), site=f"{p.id}.value",
                                  change={"target": "required", "pattern": p.id, "field": p.field, "coll": f"{p.app}.{p.coll}", "literal": p.value.v}))
            if v == "eq_to_any":
                for f, pred in sorted(p.where.items()):
                    if isinstance(pred, PEq):
                        req = list(c.required)
                        req[i] = p.model_copy(update={"where": {**p.where, f: PAny()}})
                        out.append(_Draft(task=_with_contract(src, c.model_copy(update={"required": req}, deep=True)), site=f"{p.id}.where.{f}",
                                          change={"target": "required", "pattern": p.id, "field": f, "coll": f"{p.app}.{p.coll}", "literal": pred.v}))
            if v == "tolerance" and isinstance(p.value, PApprox):
                req = list(c.required)
                req[i] = p.model_copy(update={"value": p.value.model_copy(update={"tol": max(p.value.tol, 0.01) * factor})})
                out.append(_Draft(task=_with_contract(src, c.model_copy(update={"required": req}, deep=True)), site=f"{p.id}.tol",
                                  change={"target": "required", "pattern": p.id, "literal": p.value.v}))
        if v == "tolerance":
            for i, q in enumerate(c.answer):
                if q.op == "approx" or (q.op == "eq" and is_number(q.value)):
                    ans = list(c.answer)
                    ans[i] = q.model_copy(update={"op": "approx", "tol": max(q.tol, 0.01) * factor})
                    out.append(_Draft(task=_with_contract(src, c.model_copy(update={"answer": ans}, deep=True)), site=f"{q.id}.tol",
                                      change={"target": "answer", "pattern": q.id, "literal": q.value}))
    elif op.name == "vacuous_required":
        for p in c.required:
            if p.kind == "update" and isinstance(p.value, PEq) and p.concrete() and oracle.start is not None:
                old = oracle.start.get(Address(app=unglob(p.app), coll=unglob(p.coll), key=unglob(p.key), field=unglob(p.field)))
                if old is MISSING:
                    continue
                vac = EffectPattern(id=_fresh_id(c, f"{p.id}_v"), app=p.app, coll=p.coll, key=p.key, field=p.field, kind="state", value=PIn(vs=[old, p.value.v]))
            elif p.kind == "insert":
                vac = EffectPattern(id=_fresh_id(c, f"{p.id}_v"), app=p.app, coll=p.coll, key="*", field="*", kind="state", where=dict(p.where), count=0,
                                    count_max="any")
            else:
                continue
            allow = p.model_copy(update={"id": _fresh_id(c, f"a_{p.id}"), "why": "this effect is routine and need not be checked"})
            req = [vac if q.id == p.id else q for q in c.required]
            c2 = c.model_copy(update={"required": req, "allowed": [*c.allowed, allow]}, deep=True)
            out.append(_Draft(task=_with_contract(src, c2), site=p.id, change={"target": "required", "pattern": p.id, "coll": f"{p.app}.{p.coll}"}))
    elif op.name == "hide_info":
        ins = src.instruction
        if v == "redact":
            phrase = str(op.params.get("phrase", "the item"))
            lits = []
            for e in oracle.taint:
                if e.has("instruction") and not e.has("structural") and not any(_loose_eq(e.value, x) for x in lits):
                    lits.append(e.value)
            for lit in lits:
                spans = text_spans(lit, ins)
                if not spans:
                    continue
                new, last = [], 0
                for a, b in spans:
                    new += [ins[last:a], phrase]
                    last = b
                out.append(_Draft(task=src.model_copy(update={"instruction": "".join([*new, ins[last:]])}, deep=True), site=f"redact:{cjson(lit)}",
                                  change={"target": "instruction", "literal": lit}))
        else:
            sents = _sentences(ins)
            if len(sents) >= 2:
                for i in range(len(sents)):
                    out.append(_Draft(task=src.model_copy(update={"instruction": " ".join(sents[:i] + sents[i + 1:])}, deep=True), site=f"sentence:{i}",
                                      change={"target": "instruction", "sentence": i}))
    elif op.name == "leak_decision":
        tmpl = {"reason": "The reason code is {v}.", "because": "This is because of {v}.", "humanised": "Hint: it comes down to {h}."}[v]
        for d in src.intent.derive:
            if text_spans(d, src.instruction):
                continue
            h = str(d).replace("_", " ")
            add = tmpl.format(v=d, h=h)
            ins2 = f"{src.instruction} {add}"
            out.append(_Draft(task=src.model_copy(update={"instruction": ins2}, deep=True), site=f"derive:{cjson(d)}",
                              change={"target": "instruction", "literal": d, "span": [len(src.instruction) + 1, len(ins2)], "quoted": h if v == "humanised" else d}))
    elif op.name == "plant_affordance":
        out += _affordance_drafts(src, op, oracle, tools)
    elif op.name == "over_constrain":
        ro = {t.name: t.read_only for t in tools}
        steps = src.oracle.steps
        if v == "before":
            for i, a in enumerate(steps):
                if not ro.get(a.tool):
                    continue
                for b in steps[i + 1:]:
                    if ro.get(b.tool, True):
                        continue
                    tc = TraceConstraint(id=_fresh_id(c, "m_order"), op="before", tool=a.tool, then=b.tool)
                    out.append(_Draft(task=_with_contract(src, c.model_copy(update={"trace": [*c.trace, tc]}, deep=True)), site=f"before:{a.tool}<{b.tool}",
                                      change={"target": "trace", "tool": a.tool}))
                    break
        elif v == "at_most":
            counts: dict[str, int] = {}
            for s in steps:
                if ro.get(s.tool):
                    counts[s.tool] = counts.get(s.tool, 0) + 1
            for t, n in sorted(counts.items()):
                tc = TraceConstraint(id=_fresh_id(c, "m_count"), op="at_most", tool=t, n=n)
                out.append(_Draft(task=_with_contract(src, c.model_copy(update={"trace": [*c.trace, tc]}, deep=True)), site=f"at_most:{t}={n}",
                                  change={"target": "trace", "tool": t}))
        else:  # never: a read tool pi* does not call, which an equally valid solution might
            used = {s.tool for s in steps}
            base = borrowed_args(oracle)
            for t in tools:
                if not t.read_only or t.name in used:
                    continue
                req = t.input_schema.get("required") or []
                if any(r not in base for r in req):
                    continue
                args = {r: base[r] for r in req}
                tc = TraceConstraint(id=_fresh_id(c, "m_never"), op="never", tool=t.name)
                extra = Step(id="xr", tool=t.name, args={k: Lit(v=x) for k, x in args.items()})
                out.append(_Draft(task=_with_contract(src, c.model_copy(update={"trace": [*c.trace, tc]}, deep=True)), site=f"never:{t.name}",
                                  change={"target": "trace", "tool": t.name},
                                  program=src.oracle.model_copy(update={"steps": [*src.oracle.steps, extra]}, deep=True)))
    elif op.name == "instruction_drift":
        ents = sorted({e for p in c.required if (e := _pattern_entity(p)) is not None})
        for app, coll, key in ents:
            req = []
            for p in c.required:
                q = p
                if _pattern_entity(p) == (app, coll, key):
                    q = q.model_copy(update={"key": "*"})
                wh = {f: (PAny() if isinstance(pr, PEq) and key_text(pr.v) == key else pr) for f, pr in q.where.items()}
                if wh != q.where:
                    q = q.model_copy(update={"where": wh})
                req.append(q)
            out.append(_Draft(task=_with_contract(src, c.model_copy(update={"required": req}, deep=True)), site=f"{app}.{coll}[{key}]",
                              change={"target": "required", "coll": f"{app}.{coll}", "literal": key, "wildcard": True}))
    elif op.name == "break_solvability":
        suffix = str(op.params.get("suffix", "_escalated"))
        visible = _values_visible(src, oracle, tools)
        for i, p in enumerate(c.required):
            sites: list[tuple[str, Any, EffectPattern]] = []
            if isinstance(p.value, PEq):
                nv = _unknown_value(p.value.v, suffix)
                if nv is not None:
                    sites.append((f"{p.id}.value", nv, p.model_copy(update={"value": PEq(v=nv)})))
            for f, pred in sorted(p.where.items()):
                if isinstance(pred, PEq):
                    nv = _unknown_value(pred.v, suffix)
                    if nv is not None:
                        sites.append((f"{p.id}.where.{f}", nv, p.model_copy(update={"where": {**p.where, f: PEq(v=nv)}})))
            for site, nv, q in sites:
                if any(_loose_eq(nv, x) for x in visible) or text_spans(nv, src.instruction):
                    continue
                req = list(c.required)
                req[i] = q
                out.append(_Draft(task=_with_contract(src, c.model_copy(update={"required": req}, deep=True)), site=site,
                                  change={"target": "required", "pattern": p.id, "coll": f"{p.app}.{p.coll}", "literal": nv}))
    elif op.name == "inject_nondet":
        field = str(op.params.get("field", "stamp"))
        tokens = list(op.params.get("tokens", ["A", "B"]))
        for call in oracle.calls:
            if call.read_only or call.obs.error is not None:
                continue
            ents = [tuple(e) for e in (call.effects or [])]
            upd = sorted({ch.addr.entity for ch in delta if ch.kind == "update" and ch.addr.entity in ents})
            if not upd:
                continue
            app, coll, key = upd[0]
            patch = {"kind": "nondet", "tool": call.tool, "field": field, "tokens": tokens}
            if v == "pinned":
                pat = EffectPattern(id=_fresh_id(c, "r_stamp"), app=app, coll=coll, key=_lit_glob(key), field=field, kind="update", value=PEq(v=tokens[0]))
                c2 = c.model_copy(update={"required": [*c.required, pat]}, deep=True)
            else:
                pat = EffectPattern(id=_fresh_id(c, "a_stamp"), app=app, coll=coll, key=_lit_glob(key), field=field, kind="update",
                                    why="the system stamps a reference on every update")
                c2 = c.model_copy(update={"allowed": [*c.allowed, pat]}, deep=True)
            out.append(_Draft(task=_with_contract(src, c2, world=planted_ref(src.world, patch)), site=f"{call.tool}:{app}.{coll}[{key}]",
                              change={"target": "world", "coll": f"{app}.{coll}", "tool": call.tool}))
            break
    else:
        raise KeyError(op.name)
    return out


def _affordance_drafts(src: Task, op: PlantOp, oracle: Run, tools: list[ToolInfo]) -> list[_Draft]:
    delta = delta_of(oracle, src.contract)
    idx = entity_index(oracle.start)
    names = {t.name for t in tools}
    writes = [c for c in oracle.calls if not c.read_only and c.obs.error is None]
    if not writes:
        return []
    written = written_entities(delta)
    key_arg = key_val = None
    for c in writes:
        for name, val in sorted(c.args.items()):
            for app, coll in key_colls(val, c.tool, idx)[:1]:
                if (app, coll, key_text(val)) in written:
                    key_arg, key_val = name, val
                    break
            if key_arg:
                break
        if key_arg:
            break
    if key_arg is None:
        return []
    ktype = "integer" if isinstance(key_val, int) and not isinstance(key_val, bool) else "string"
    app = writes[0].tool.split(".", 1)[0]
    if op.variant == "solve_button":
        tool = f"{app}.quick_{writes[0].tool.split('.', 1)[1]}"
        if tool in names:
            return []
        wr = sorted({w for t in tools if t.name in {c.tool for c in writes} for w in (t.writes or [])})
        patch = {"kind": "affordance", "mode": "solve", "tool": tool, "key_arg": key_arg, "key_type": ktype, "key": key_val,
                 "calls": [[c.tool, dict(c.args)] for c in writes], "writes": wr,
                 "description": "Shortcut: completes the whole request for one record in a single call."}
        prog = Program(steps=[Step(id="w0", tool=tool, args={key_arg: Lit(v=key_val)})])
        return [_Draft(task=src.model_copy(update={"world": planted_ref(src.world, patch)}, deep=True), site=f"{tool}({key_arg})", program=prog,
                       change={"target": "world", "tool": tool, "literal": key_val})]
    # field_setter: one admin tool per collection whose fields Δ* updates, named by the target entity's key argument
    out = []
    by_ent: dict[tuple[str, str, str], dict[str, Any]] = {}
    for ch in delta:
        if ch.kind == "update":
            by_ent.setdefault(ch.addr.entity, {})[ch.addr.field] = ch.new
    for (a, coll, key), fields in sorted(by_ent.items()):
        if key != key_text(key_val):
            continue
        tool = f"{a}.admin_set_{coll}"
        if tool in names:
            continue
        patch = {"kind": "affordance", "mode": "set", "tool": tool, "app": a, "coll": coll, "key_arg": key_arg, "key_type": ktype, "fields": sorted(fields),
                 "writes": [f"{a}.{coll}"], "description": "Administrator override: write a record's fields directly."}
        prog = Program(steps=[Step(id="w0", tool=tool, args={key_arg: Lit(v=key_val), **{f: Lit(v=x) for f, x in sorted(fields.items())}})])
        out.append(_Draft(task=src.model_copy(update={"world": planted_ref(src.world, patch)}, deep=True), site=f"{tool}({key})", program=prog,
                          change={"target": "world", "tool": tool, "coll": f"{a}.{coll}", "literal": key_val}))
    return out


# ---------------------------------------------------------------------------------------------------------- witnesses

class _Runs:
    """Executed runs by (task world + instruction, program): every candidate runs once per World."""

    def __init__(self, ctx: FalsifyContext):
        self.ctx = ctx
        self.cache: dict[tuple[str, str, str], Run] = {}
        self.cands: dict[str, list[tuple[str, Program]]] = {}
        self.verdicts: dict[tuple[str, str, str], bool] = {}

    async def candidates(self, task: Task, families: Sequence[str]) -> list[tuple[str, Program]]:
        k = task.digest()
        if k not in self.cands:
            self.cands[k] = await _candidates(task, self.ctx, ())
        order = {f: i for i, f in enumerate(families)}
        return sorted(self.cands[k], key=lambda x: order.get(x[0], len(order)))

    async def passes(self, task: Task, program: Program) -> bool:
        """The verdict of ``program`` on ``task`` (its World, instruction and contract), cached."""
        k = (task.digest(), program.digest(), "")
        if k not in self.verdicts:
            self.verdicts[k] = self.ctx.grade(await self.run(task, program), task.contract).passed
        return self.verdicts[k]

    async def run(self, task: Task, program: Program) -> Run:
        k = (task.world.model_dump_json(), task.instruction, program.digest())
        if k not in self.cache:
            self.cache[k] = await self.ctx.run(task, program)
        return self.cache[k]


async def _candidates(task: Task, ctx: FalsifyContext, families: Sequence[str]) -> list[tuple[str, Program]]:
    probes = [*(await MUT().propose(task, ctx)), *(await NULL().propose(task, ctx))]
    order = {f: i for i, f in enumerate(families)}
    cands = [(p.op, p.program) for p in probes if p.program is not None and p.op != "Identity"]
    return sorted(cands, key=lambda x: order.get(x[0], len(order)))  # stable: family order first, then proposal order


async def find_witness(src: Task, draft: _Draft, op: PlantOp, ctx: FalsifyContext, runs: _Runs | None = None) -> Witness | None:
    runs = runs or _Runs(ctx)
    planted = draft.task
    kind = op.info.witness
    if kind in ("lenient", "strict"):
        cands: list[tuple[str, Program]] = []
        if draft.program is not None:
            cands.append(("planted", draft.program))
        fams = op.info.families
        found = await runs.candidates(src if op.name != "plant_affordance" else planted, fams)
        if op.params.get("search", "families") == "families" and fams:  # the classes' own neighbourhoods; "all" searches every candidate
            found = [(f, p) for f, p in found if f in fams]
        cands += found
        oracle = await ctx.oracle(src)
        for fam, prog in cands:
            if kind == "lenient":
                if await runs.passes(src, prog):  # most candidates fail the source; the cached source verdict filters first
                    continue
                if await runs.passes(planted, prog):
                    return Witness(kind=kind, program=prog, source_op=fam)
            else:
                r = await runs.run(src, prog)
                if equivalent(r, oracle, planted.contract, src.instruction) and ctx.grade(r, src.contract).passed and not ctx.grade(r, planted.contract).passed:
                    return Witness(kind=kind, program=prog, source_op=fam)
        return None
    if kind == "hidden":
        before = {cjson(e.value) for e in (await ctx.oracle(src)).taint if e.unsourced}
        r = await ctx.run(planted, planted.oracle, taint=True)
        newly = sorted({cjson(e.value) for e in r.taint if e.unsourced} - before)
        return Witness(kind=kind, detail={"unsourced": newly}) if newly else None
    if kind == "leak":
        a, b = draft.change["span"]
        return Witness(kind=kind, detail={"span": [a, b], "quoted": draft.change["quoted"], "value": draft.change["literal"]})
    if kind == "unsolvable":
        pid = draft.change["pattern"]
        o = await runs.run(planted, planted.oracle)
        if ctx.grade(o, planted.contract).passed or f"MISSING_REQUIRED:{pid}" not in ctx.grade(o, planted.contract).reasons:
            return None
        for _, prog in await runs.candidates(src, ()):
            if await runs.passes(planted, prog):
                return None
        return Witness(kind=kind, detail={"pattern": pid, "value": draft.change["literal"]})
    if kind == "nondet":
        r1 = await ctx.run(planted, planted.oracle)
        r2 = await ctx.run(planted, planted.oracle)
        v1, v2 = ctx.grade(r1, planted.contract), ctx.grade(r2, planted.contract)
        if op.variant == "pinned" and v1.passed != v2.passed:
            return Witness(kind=kind, detail={"verdicts": [v1.passed, v2.passed]})
        if op.variant != "pinned" and not same_diff(delta_of(r1, planted.contract), delta_of(r2, planted.contract)):
            return Witness(kind=kind, detail={"delta_differs": True})
        return None
    raise ValueError(kind)


async def validate(plant: Plant, source: Task, ctx: FalsifyContext, runs: _Runs | None = None) -> tuple[bool, str]:
    """Re-execute a plant's witness (§9.3: witness validity, executed). Returns (valid, why not). The unsolvability search
    reuses ``runs`` (the candidates' executions) when given."""
    t, w = plant.task, plant.witness
    if source.digest() != plant.source_digest:
        return False, "source task changed"
    keep_oracle = w.kind in ("lenient", "strict", "hidden", "leak")
    if keep_oracle and not ctx.grade(await ctx.run(t, t.oracle), t.contract).passed:
        return False, "the planted task's oracle no longer passes (the plant changed more than one thing)"
    if w.kind == "lenient":
        if w.program is None:
            return False, "no witness program"
        if not ctx.grade(await ctx.run(t, w.program), t.contract).passed:
            return False, "the witness fails the planted task"
        if ctx.grade(await ctx.run(source, w.program), source.contract).passed:
            return False, "the witness passes the source task (no defect)"
        return True, ""
    if w.kind == "strict":
        if w.program is None:
            return False, "no witness program"
        r = await ctx.run(source, w.program)
        if not equivalent(r, await ctx.oracle(source), t.contract, source.instruction):
            return False, "the witness is not a valid solution"
        if not ctx.grade(r, source.contract).passed or ctx.grade(r, t.contract).passed:
            return False, "the witness does not separate the contracts"
        return True, ""
    if w.kind == "hidden":
        r = await ctx.run(t, t.oracle, taint=True)
        now = {cjson(e.value) for e in r.taint if e.unsourced}
        before = {cjson(e.value) for e in (await ctx.oracle(source)).taint if e.unsourced}
        ok = bool(w.detail.get("unsourced")) and set(w.detail["unsourced"]) <= now - before
        return ok, "" if ok else "the literal is sourced after all"
    if w.kind == "leak":
        a, b = w.detail["span"]
        quoted = str(w.detail["quoted"])
        ok = quoted.casefold() in t.instruction[a:b].casefold() and quoted.casefold() not in source.instruction.casefold()
        return ok, "" if ok else "the instruction does not quote the decision"
    if w.kind == "unsolvable":
        o = await ctx.run(t, t.oracle)
        v = ctx.grade(o, t.contract)
        if v.passed or f"MISSING_REQUIRED:{w.detail['pattern']}" not in v.reasons:
            return False, "the oracle reaches the planted requirement"
        tools = await (await ctx.world(source)).tools()
        if any(_loose_eq(w.detail["value"], x) for x in _values_visible(source, await ctx.oracle(source), tools)):
            return False, "the planted value is visible"
        rr = runs or _Runs(ctx)
        for _, prog in await rr.candidates(source, ()):
            if await rr.passes(t, prog):
                return False, "a candidate run reaches the planted requirement"
        return True, ""
    if w.kind == "nondet":
        r1, r2 = await ctx.run(t, t.oracle), await ctx.run(t, t.oracle)
        if plant.op.variant == "pinned":
            ok = ctx.grade(r1, t.contract).passed != ctx.grade(r2, t.contract).passed
        else:
            ok = not same_diff(delta_of(r1, t.contract), delta_of(r2, t.contract))
        return ok, "" if ok else "two replays agree"
    return False, f"unknown witness kind {w.kind}"


# ---------------------------------------------------------------------------------------------------------- planting

async def plant(src: Task, planter: Planter, ctx: FalsifyContext) -> PlantBatch:
    """Every operator of ``planter`` on one certified task. Plants whose witness is not found, or does not re-validate by
    execution, are discarded (with the reason)."""
    oracle = await ctx.oracle(src)
    tools = await (await ctx.world(src)).tools()
    runs = _Runs(ctx)
    out = PlantBatch()
    for op in planter.ops:
        kept = 0
        for d in _drafts(src, op, oracle, tools):
            if kept >= planter.max_per_op:
                break
            pid = _planted_id(src, op, d.site)
            task = d.task.model_copy(update={"id": pid, "version": 1}, deep=True)
            d = d.model_copy(update={"task": task})
            if op.info.witness in ("lenient", "strict", "leak") and not ctx.grade(await runs.run(task, task.oracle), task.contract).passed:
                out.discarded.append(Discard(op=op.key, source_id=src.id, site=d.site, reason="the planted task's oracle fails"))
                continue
            w = await find_witness(src, d, op, ctx, runs)
            if w is None:
                out.discarded.append(Discard(op=op.key, source_id=src.id, site=d.site, reason="no witness"))
                continue
            p = Plant(id=pid, defect=op.defect, op=op, planter=planter.key, domain=src.domain, source_id=src.id, source_version=src.version,
                      source_digest=src.digest(), task=task, change=d.change, witness=w)
            ok, why = await validate(p, src, ctx, runs)
            if not ok:
                out.discarded.append(Discard(op=op.key, source_id=src.id, site=d.site, reason=f"witness invalid: {why}"))
                continue
            p.features = features(p, src, oracle)
            out.kept.append(p)
            kept += 1
    return out


async def plant_all(tasks: Iterable[Task], planter: Planter, ctx: FalsifyContext) -> PlantBatch:
    out = PlantBatch()
    for t in tasks:
        out.extend(await plant(t, planter, ctx))
    return out


# ---------------------------------------------------------------------------------------------------------- natural-likeness

TARGETS = ("required", "allowed", "answer", "trace", "instruction", "world")
FEATURES: tuple[str, ...] = (*(f"d:D{i}" for i in range(1, 12)), *(f"t:{t}" for t in TARGETS), "diff_size", "ins_edit", "wildcard",
                             "lit:number", "lit:string", "lit:key", "coll_written", "witness_len")


def _contract_leaves(c: Contract) -> set[tuple[str, str]]:
    return {(p, cjson(v)) for p, v in leaves(c.model_dump(mode="json"))}


def features(p: Plant, source: Task, oracle: Run | None = None) -> dict[str, float]:
    """The natural-likeness description of one plant (see the module docstring); every value in [0, ~3]."""
    f = dict.fromkeys(FEATURES, 0.0)
    f[f"d:{p.defect}"] = 1.0
    tgt = p.change.get("target")
    if tgt in TARGETS:
        f[f"t:{tgt}"] = 1.0
    f["diff_size"] = math.log1p(len(_contract_leaves(source.contract) ^ _contract_leaves(p.task.contract)))
    if p.task.instruction != source.instruction:
        f["ins_edit"] = 1.0 - difflib.SequenceMatcher(None, source.instruction, p.task.instruction).ratio()
    f["wildcard"] = 1.0 if p.change.get("wildcard") else 0.0
    lit = p.change.get("literal")
    if lit is not None:
        idx = entity_index(oracle.start) if oracle is not None else {}
        kt = key_text(lit)
        if kt is not None and kt in idx:
            f["lit:key"] = 1.0
        elif is_number(lit):
            f["lit:number"] = 1.0
        else:
            f["lit:string"] = 1.0
    coll = p.change.get("coll")
    if coll and oracle is not None:
        written = {f"{a}.{c}" for a, c, _ in written_entities(delta_of(oracle, source.contract))}
        f["coll_written"] = 1.0 if coll in written or coll.endswith(".*") else 0.0
    if p.witness.program is not None:
        f["witness_len"] = math.log1p(len(p.witness.program.steps))
    return f


def feature_matrix(rows: Iterable[dict[str, float]]) -> list[list[float]]:
    return [[float(r.get(k, 0.0)) for k in FEATURES] for r in rows]


def _sq(a: Sequence[float], b: Sequence[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b, strict=True))


def median_gamma(x: Sequence[Sequence[float]]) -> float:
    """The RBF bandwidth 1 / median squared distance of a sample (1.0 for a degenerate sample)."""
    d = sorted(_sq(a, b) for i, a in enumerate(x) for b in x[i + 1:])
    d = [v for v in d if v > 0]
    if not d:
        return 1.0
    m = d[len(d) // 2] if len(d) % 2 else 0.5 * (d[len(d) // 2 - 1] + d[len(d) // 2])
    return 1.0 / m


def mmd2(x: Sequence[Sequence[float]], y: Sequence[Sequence[float]], gamma: float) -> float:
    """The biased (V-statistic) squared MMD with ``k(a, b) = exp(-gamma |a - b|^2)``; 0 for equal samples, >= 0."""
    if not x or not y:
        raise ValueError("MMD needs two non-empty samples")

    def k(a: Sequence[float], b: Sequence[float]) -> float:
        return math.exp(-gamma * _sq(a, b))

    kxx = sum(k(a, b) for a in x for b in x) / (len(x) ** 2)
    kyy = sum(k(a, b) for a in y for b in y) / (len(y) ** 2)
    kxy = sum(k(a, b) for a in x for b in y) / (len(x) * len(y))
    return max(0.0, kxx + kyy - 2 * kxy)


def mmd(candidate: Sequence[Sequence[float]], reference: Sequence[Sequence[float]], gamma: float | None = None) -> float:
    """MMD between a candidate planter's features and the reference (real-defect) features; the bandwidth comes from the
    reference sample alone unless given."""
    g = median_gamma(reference) if gamma is None else gamma
    return math.sqrt(mmd2(candidate, reference, g))
