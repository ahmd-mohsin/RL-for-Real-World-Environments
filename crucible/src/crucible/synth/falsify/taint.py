"""TAINT: communicability by taint (part2-paper-core §5, claim C3). Executed, no model.

Probes, all property probes on the oracle run's taint map (``program.run``):
- ``literal`` (one per non-structural literal of π*): it must have a source (ι, an earlier observation, the intent card's
  ``given``, or the tool catalogue). Unsourced: confirmed **D5** (success needs information the agent cannot get) unless
  it is **free**: π* with that literal replaced by an unrelated value still runs and still passes C (M4: a search query
  or a note body the oracle's author chose is the agent's own wording, not information it must get; §5.1).
- ``derive`` (one per intent-card ``derive`` value): the agent must work it out, so it must not occur in ι. Found in ι:
  confirmed **D7** (the decision is given away).
- ``r_const`` / ``q_const`` (one per constant of a required pattern or an answer check: a concrete key, an ``eq``/``in``/
  ``approx`` value): it must trace to ι, ``given``, the tool catalogue or a π* value (an argument it sent, a leaf of an
  observation, its answer). Unsourced: confirmed **D5**.
Sites name values, not step locations, so a repaired π* with renamed steps re-opens a finding it did not fix.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

from crucible.synth.canon import cjson
from crucible.synth.contract import Contract, EffectPattern, PApprox, PEq, PIn, is_glob, unglob
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, key_text
from crucible.synth.iface import ToolInfo
from crucible.synth.program import Lit, Program, _loose_eq, is_structural, program_locs, text_spans
from crucible.synth.task import Task


def _pred_consts(p: Any) -> list[Any]:
    if isinstance(p, PEq):
        return [p.v]
    if isinstance(p, PIn):
        return list(p.vs)
    if isinstance(p, PApprox):
        return [p.v]
    return []


def contract_constants(c: Contract) -> Iterator[tuple[str, str, Any]]:
    """(kind, owner id, value) for every constant of R and Q."""
    for p in c.required:
        yield from _pattern_consts(p)
    for q in c.answer:
        if q.ref is None and q.op in ("eq", "approx", "in", "set_eq") and q.value is not None:
            for v in (q.value if isinstance(q.value, list) else [q.value]):
                yield "q_const", q.id, v


def _pattern_consts(p: EffectPattern) -> Iterator[tuple[str, str, Any]]:
    if not is_glob(p.key) and not p.key.startswith(("h:", "#")):
        yield "r_const", p.id, unglob(p.key)
    if p.value is not None:
        for v in _pred_consts(p.value):
            yield "r_const", p.id, v
    for pred in p.where.values():
        for v in _pred_consts(pred):
            yield "r_const", p.id, v


def sourced(v: Any, instruction: str, given: Sequence[Any], tools: Sequence[ToolInfo], values: Sequence[Any]) -> list[str]:
    """Where a contract constant can come from (empty: nowhere)."""
    if is_structural(v):
        return ["structural"]
    out = []
    if text_spans(v, instruction):
        out.append("instruction")
    if any(_loose_eq(v, g) or (isinstance(g, str) and text_spans(v, g, min_len=3)) for g in given):
        out.append("given")
    for t in tools:
        enums = [e for p in (t.input_schema.get("properties") or {}).values() if isinstance(p, dict) for e in p.get("enum", [])]
        if any(_loose_eq(v, e) for e in enums) or (t.description and text_spans(v, t.description, min_len=3)):
            out.append("tool_spec")
            break
    kv = key_text(v)
    if any(_loose_eq(v, x) or (kv is not None and kv == key_text(x)) for x in values):
        out.append("program")
    return out


class TAINT(BaseFalsifier):
    name: ClassVar[str] = "TAINT"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        oracle = await ctx.oracle(task)
        out: list[Probe] = []
        for e in oracle.taint:
            if e.has("structural"):
                continue
            out.append(Probe(cls="TAINT", op="literal", site=f"literal:{cjson(e.value)}", expect="property",
                             data={"loc": e.loc, "value": e.value, "sources": [s.kind for s in e.sources]}))
        for d in task.intent.derive:
            out.append(Probe(cls="TAINT", op="derive", site=f"derive:{cjson(d)}", expect="property", data={"value": d}))
        seen: set[str] = set()
        for kind, owner, v in contract_constants(task.contract):
            site = f"{kind}:{cjson(v)}"
            if site in seen:
                continue
            seen.add(site)
            out.append(Probe(cls="TAINT", op=kind, site=site, expect="property", data={"owner": owner, "value": v}))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        v = probe.data.get("value")
        if probe.op == "literal":
            hit = not probe.data["sources"]
            free = hit and await self.free(task, ctx, probe.data["loc"], v)
            return Outcome(data={**probe.data, "hit": hit and not free, **({"free": True} if free else {})})
        if probe.op == "derive":
            spans = text_spans(v, task.instruction)
            return Outcome(data={"hit": bool(spans), "spans": spans})
        oracle = await ctx.oracle(task)
        tools = await (await ctx.world(task)).tools()
        src = sourced(v, task.instruction, task.intent.given, tools, oracle.values())
        return Outcome(data={"hit": not src, "sources": src})

    async def free(self, task: Task, ctx: FalsifyContext, loc: str, v: Any) -> bool:
        """Does π* still run and pass C with the literal at ``loc`` replaced by an unrelated value?"""
        prog = replace_literal(task.oracle, loc, other_value(v))
        if prog is None:
            return False
        r = await ctx.run(task, prog)
        return r.error is None and not any(c.obs.error for c in r.calls) and ctx.grade(r, task.contract).passed

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if not outcome.data.get("hit"):
            return "clean", None
        return "confirmed", "D7" if probe.op == "derive" else "D5"


def other_value(v: Any) -> Any:
    """An unrelated value of the same type (the free-literal test)."""
    if isinstance(v, bool) or v is None:
        return v
    if isinstance(v, int):
        return v + 7919
    if isinstance(v, float):
        return round(v * 1.37 + 3.1, 6)
    return "unrelated placeholder text qx" if isinstance(v, str) else v


def replace_literal(p: Program, loc: str, value: Any) -> Program | None:
    """``p`` with the scalar literal at taint location ``loc`` replaced (None: no scalar literal there)."""
    q = p.model_copy(deep=True)
    for at, a in program_locs(q):
        if at == loc and isinstance(a, Lit) and not isinstance(a.v, dict | list):
            a.v = value
            return q
    return None
