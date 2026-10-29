"""TWIN: literal substitution (part2-paper-core §5, claim C3): does the contract depend on the instruction?

For each literal ℓ of π* that the taint map sources from ι:
1. **Locate** ℓ in s₀: an entity key equal to ℓ, or a cell equal to ℓ (collections π* touches first).
2. **Sibling**: another entity e′ of that collection whose key (or cell, for a cell match) differs from ℓ and does not
   itself occur in ι (so the twin instruction is unambiguous); up to ``twins`` siblings, seeded.
3. **Substitute** σ = {ℓ → ℓ′} (plus e's key → e′'s key when ℓ is a cell that identifies exactly one entity; a value
   several entities share, such as a customer, is a selector, and the keys it selects are re-derived instead) through
   ι (every token span), π* (every literal) and C (concrete pattern keys, predicate constants, answer constants).
   Siblings with the same ℓ′ make one twin.
4. **Re-derive**: a constant of C′ that σ did not map but that equals an argument π* sent is replaced by what π*′ sent
   at the same place (an amount π* read from e is e′'s amount in the twin; a literal π* sends in both is unchanged; a
   constant no argument carries is not touched). The answer constant follows the answer.
5. **Execute** π*′ from s₀ under ι′. If π*′ stops, or a call errors that did not error in π* (e′ cannot be acted on),
   the twin is dead: not a trial (D10 is noted for that instance, never counted against the task).
6. **Label**: π* must fail C′ and π*′ must pass C′. Either violation is confirmed **D9**.
"""

from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from crucible.synth.canon import MISSING, CanonState, cjson
from crucible.synth.contract import Contract, is_glob, unglob
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, delta_of, key_text, written_entities
from crucible.synth.infer import _lit_glob
from crucible.synth.program import Program, Run, _loose_eq, is_structural, scalar_leaves, text_spans
from crucible.synth.task import Task


def _same(x: Any, src: Any) -> bool:
    if is_structural(x):
        return False
    if _loose_eq(x, src):
        return True
    kx, ks = key_text(x), key_text(src)
    return kx is not None and kx == ks


def _conv(dst: Any, like: Any) -> Any:
    if isinstance(like, int) and not isinstance(like, bool) and isinstance(dst, str) and dst.lstrip("-").isdigit():
        return int(dst)
    if isinstance(like, str) and isinstance(dst, int) and not isinstance(dst, bool):
        return str(dst)
    return dst


def substitute(x: Any, sigma: Sequence[tuple[Any, Any]]) -> Any:
    if isinstance(x, list):
        return [substitute(y, sigma) for y in x]
    if isinstance(x, dict):
        return {k: substitute(y, sigma) for k, y in x.items()}
    for src, dst in sigma:
        if _same(x, src):
            return _conv(dst, x)
    return x


def map_program(p: Program, f: Callable[[Any], Any]) -> Program:
    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            if node.get("op") == "lit":
                return {**node, "v": f(node.get("v"))}
            return {k: walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [walk(v) for v in node]
        return node

    return Program.model_validate(walk(p.model_dump(mode="json")))


def map_contract(c: Contract, f: Callable[[str, Any], Any]) -> Contract:
    """Rewrite every constant of a contract: ``f("key", text)`` for a concrete pattern key, ``f("value", v)`` for a
    predicate or answer constant."""

    def pred(d: Any) -> Any:
        if not isinstance(d, dict):
            return d
        if d.get("op") in ("eq", "approx") and "v" in d:
            return {**d, "v": f("value", d["v"])}
        if d.get("op") == "in":
            return {**d, "vs": [f("value", v) for v in d.get("vs", [])]}
        return d

    def pattern(p: dict[str, Any]) -> dict[str, Any]:
        q = dict(p)
        if not is_glob(q.get("key", "*")):
            q["key"] = _lit_glob(str(f("key", unglob(q["key"]))))
        if q.get("value") is not None:
            q["value"] = pred(q["value"])
        q["where"] = {k: pred(v) for k, v in (q.get("where") or {}).items()}
        return q

    d = c.model_dump(mode="json")
    d["required"] = [pattern(p) for p in d["required"]]
    d["allowed"] = [pattern(p) for p in d["allowed"]]
    d["answer"] = [{**q, "value": f("value", q["value"]) if q.get("value") is not None else None} for q in d["answer"]]
    return Contract.model_validate(d)


def rewrite_instruction(ins: str, sigma: Sequence[tuple[Any, Any]]) -> str:
    reps: list[tuple[int, int, str]] = []
    for src, dst in sigma:
        reps += [(a, b, str(dst)) for a, b in text_spans(src, ins)]
    reps.sort(key=lambda r: (r[0], -(r[1] - r[0])))
    out, last = [], 0
    for a, b, t in reps:
        if a < last:
            continue
        out += [ins[last:a], t]
        last = b
    return "".join([*out, ins[last:]])


def locate(v: Any, s0: CanonState, touched: set[tuple[str, str]]) -> list[tuple[str, str, str, str]]:
    """Where ℓ lives in s₀: (app, coll, key, "") for a key, (app, coll, key, field) for a cell."""
    hits: list[tuple[str, str, str, str]] = []
    kt = key_text(v)
    for app, coll, key, row in s0.entities():
        if kt is not None and key == kt:
            hits.append((app, coll, key, ""))
        for f in sorted(row):
            if _same(row[f], v) and not (kt is not None and key == kt):
                hits.append((app, coll, key, f))
    return sorted(hits, key=lambda h: ((h[0], h[1]) not in touched, h[3] != "", h))


def rederive(c: Contract, oracle: Run, twin: Run, fixed: Sequence[Any]) -> Contract:
    """Step 4 of the module docstring."""
    where: list[tuple[str, Any]] = []
    for call in oracle.calls:
        where += list(scalar_leaves(call.args, call.step))
    now = {loc: v for call in twin.calls for loc, v in scalar_leaves(call.args, call.step)}

    def f(kind: str, v: Any) -> Any:
        if is_structural(v) or any(_same(v, x) for x in fixed):
            return v
        if kind == "value" and oracle.answer is not None and twin.answer is not None and _loose_eq(v, oracle.answer):
            return twin.answer
        for loc, x in where:
            if _same(v, x) and loc in now and not _same(now[loc], x):
                return _conv(now[loc], v) if kind == "value" else str(now[loc])
        return v

    return map_contract(c, f)


class TWIN(BaseFalsifier):
    name: ClassVar[str] = "TWIN"

    def __init__(self, twins: int = 2):
        self.twins = twins

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        o = await ctx.oracle(task)
        if o.start is None:
            return "no canonical state"
        if not ctx.grade(o, task.contract).passed:
            return "the oracle fails its own contract (MUT Identity): twins would inherit the failure"
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        oracle = await ctx.oracle(task)
        s0 = oracle.start
        assert s0 is not None
        touched = {(a, c) for a, c, _ in written_entities(delta_of(oracle, task.contract))}
        rng = random.Random(f"TWIN/{task.id}/{task.version}/{ctx.seed}")
        lits: list[Any] = []
        for e in oracle.taint:
            if e.has("instruction") and not e.has("structural") and not any(_loose_eq(e.value, x) for x in lits):
                lits.append(e.value)
        out: list[Probe] = []
        for v in lits:
            cands = locate(v, s0, touched)
            if not cands:
                continue
            app, coll, key, field = cands[0]
            rows = s0.collection(app, coll)
            unique = not field or sum(1 for r in rows.values() if field in r and _same(r[field], v)) == 1
            sibs: list[tuple[str, Any]] = []
            for k2, row2 in sorted(rows.items()):
                if k2 == key:
                    continue
                target = row2.get(field, MISSING) if field else _conv(k2, v)
                if target is MISSING or is_structural(target) or _same(target, v) or text_spans(target, task.instruction):
                    continue
                if any(_same(target, t) for _, t in sibs):
                    continue
                sibs.append((k2, target))
            if len(sibs) > self.twins:
                sibs = sorted(rng.sample(sibs, self.twins), key=lambda s: s[0])
            for k2, target in sibs:
                sigma: list[tuple[Any, Any]] = [(v, target)] + ([(key, k2)] if field and unique else [])
                ins2 = rewrite_instruction(task.instruction, sigma)
                prog2 = map_program(task.oracle, lambda x, sg=sigma: substitute(x, sg))
                c2 = map_contract(task.contract, lambda kind, x, sg=sigma: substitute(x, sg))
                out.append(Probe(cls="TWIN", op="twin", site=f"literal:{cjson(v)}->{cjson(target)}", program=prog2, instruction=ins2, contract=c2,
                                 expect="property", data={"literal": v, "target": target, "entity": [app, coll, key], "twin": [app, coll, k2],
                                                          "sigma": [list(s) for s in sigma]}))
        return out

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        assert probe.program is not None and probe.contract is not None and probe.instruction is not None
        oracle = await ctx.oracle(task)
        r2 = await ctx.run(task, probe.program, instruction=probe.instruction)
        errored_before = {c.step for c in oracle.calls if c.obs.error is not None}
        new_err = [c.step for c in r2.calls if c.obs.error is not None and c.step not in errored_before]
        if r2.error is not None or new_err or len(r2.calls) < len(oracle.calls):
            return Outcome(valid=False, error=r2.error, data={"dead": True, "note": "D10 for this instance", "errored": new_err})
        fixed = [x for pair in probe.data["sigma"] for x in pair]
        c2 = rederive(probe.contract, oracle, r2, fixed)
        g1, g2 = ctx.grade(oracle, c2), ctx.grade(r2, c2)
        return Outcome(data={"oracle_passes_twin": g1.passed, "twin_passes_twin": g2.passed, "g1": g1.reasons, "g2": g2.reasons})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if outcome.data.get("oracle_passes_twin") or not outcome.data.get("twin_passes_twin"):
            return "confirmed", "D9"
        return "clean", None
