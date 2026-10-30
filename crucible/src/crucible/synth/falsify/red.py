"""RED: red on start (part2-paper-core §5; G4). A required check already true at s₀ is free reward (D4).

Probes:
- ``start_state``: the whole contract graded on the do-nothing episode (s₀, s₀, ∅, no answer); a pass is D4;
- ``pattern`` (one per R pattern): a ``state`` pattern that s₀ already satisfies; a change pattern with ``count == 0``;
  a fully concrete update pattern whose predicates already hold for that cell at s₀ (the "change" is the status quo); a
  concrete insert whose entity already exists with those fields; a concrete delete of an entity already absent;
- ``exec`` (one per ``require`` exec check): the predicate already true on the do-nothing episode. A ``guard`` (a
  pass-to-pass suite) holds at s₀ by design and is not probed.
Change patterns with a wildcard address are never vacuous on s₀ alone (they still need a change) and are left to MUT,
NULL and TWIN. All probes are property probes (re-checked by site on replay).
"""

from __future__ import annotations

import time
from typing import ClassVar

from crucible.synth.canon import Address, Change, cjson, content_hash
from crucible.synth.contract import EffectPattern, PAny, _Ctx, _exec_ok, matches, state_count, state_ok, unglob
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, empty_run
from crucible.synth.task import Task


def pattern_site(p: EffectPattern) -> str:
    return "pattern:" + content_hash(p.model_dump(mode="json", exclude={"id", "why"}), 16)


class RED(BaseFalsifier):
    name: ClassVar[str] = "RED"

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        o = await ctx.oracle(task)
        return "no canonical state" if o.start is None and not task.contract.answer else None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        out = [Probe(cls="RED", op="start_state", site="start_state", expect="property")]
        for p in task.contract.required:
            out.append(Probe(cls="RED", op="pattern", site=pattern_site(p), expect="property", data={"id": p.id}))
        for x in task.contract.exec:
            if x.kind == "guard":  # a preservation check holds at s₀ by design; it rewards nothing on its own
                continue
            out.append(Probe(cls="RED", op="exec", site=f"exec:{x.name}:{cjson(x.args)}", expect="property", data={"id": x.id}))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        oracle = await ctx.oracle(task)
        c = task.contract
        e = empty_run(oracle.start, task.instruction)
        if probe.op == "start_state":
            v = ctx.grade(e, c)
            return Outcome(passed=v.passed, reasons=v.reasons)
        if probe.op == "exec":
            x = next(x for x in c.exec if x.id == probe.data["id"])
            try:
                ok = _exec_ok(x, e, ctx.exec_registry, ctx.deadline_s)
            except Exception:  # noqa: BLE001 - a predicate that cannot even run on s₀ is not vacuous
                ok = False
            return Outcome(passed=ok)
        p = next(p for p in c.required if p.id == probe.data["id"])
        if oracle.start is None:
            return Outcome(valid=False)
        s0 = c.normalizers.apply(oracle.start)
        ctx2 = _Ctx(s0, s0, None, time.monotonic() + ctx.deadline_s)
        return Outcome(passed=vacuous(p, ctx2))

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", "D4") if outcome.passed else ("clean", None)


def vacuous(p: EffectPattern, ctx: _Ctx) -> bool:
    """Is required pattern ``p`` already satisfied by the start state (``ctx.s0 == ctx.sT``)?"""
    if p.kind == "state":
        return state_ok(p, state_count(p, ctx))
    if p.count == 0:
        return True
    if not p.concrete():
        return False
    app, coll, key, field = unglob(p.app), unglob(p.coll), unglob(p.key), unglob(p.field)
    row = ctx.s0.entity(app, coll, key)
    if p.kind == "update":
        # A value-free or ``any`` update constrains no value: it still needs a change, so s₀ cannot satisfy it. Treating it
        # as the status quo labelled an ``eq → any`` relaxation (D3) as D4 (M3 recall matrix).
        if row is None or field not in row or p.value is None or isinstance(p.value, PAny):
            return False
        v = row[field]
        fake = Change(addr=Address(app=app, coll=coll, key=key, field=field), kind="update", old=v, new=v)
        return matches(p, fake, ctx)
    if p.kind == "insert":
        return row is not None and matches(p, Change(addr=Address(app=app, coll=coll, key=key), kind="insert", new=row), ctx)
    if p.kind == "delete":
        return row is None
    return False  # kind "any" still needs a change
