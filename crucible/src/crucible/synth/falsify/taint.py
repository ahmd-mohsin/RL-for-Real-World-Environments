"""TAINT: communicability by taint (part2-paper-core §5, claim C3). Executed, no model.

Probes, all property probes on the oracle run's taint map (``program.run``):
- ``literal`` (one per non-structural literal of π*): it must have a source (ι, an earlier observation, the intent card's
  ``given``, or the tool catalogue). Unsourced: confirmed **D5** (success needs information the agent cannot get).
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
from crucible.synth.program import _loose_eq, is_structural, text_spans
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
            return Outcome(data={"hit": not probe.data["sources"], **probe.data})
        if probe.op == "derive":
            spans = text_spans(v, task.instruction)
            return Outcome(data={"hit": bool(spans), "spans": spans})
        oracle = await ctx.oracle(task)
        tools = await (await ctx.world(task)).tools()
        src = sourced(v, task.instruction, task.intent.given, tools, oracle.values())
        return Outcome(data={"hit": not src, "sources": src})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if not outcome.data.get("hit"):
            return "clean", None
        return "confirmed", "D7" if probe.op == "derive" else "D5"
