"""NULL: degenerate agents that must fail (part2-paper-core §5; G4's null agents).

Every agent is a ``Program`` of literal calls, built from the oracle run (so no taint is involved):
- ``noop``: nothing (D4 when it passes: the contract holds at s₀);
- ``read_all``: π*'s read calls, then every read tool that needs no argument (D4);
- ``answer_only``: the oracle's answer, no calls; proposed only when π* changes state (on a read-only task the answer is
  the task, and the answer-only agent is the oracle) (D4: a bare claim passes);
- ``minus_last_write``: π* without its last write (D1: a half task passes);
- ``write_all``: every write tool on every entity key π* read, other arguments borrowed by name from π* (D8); not
  proposed when it would only repeat π*'s own writes (π* read nothing but its targets);
- ``shortcut``: each write tool π* never calls, aimed at an entity π* changed, its arguments filled from Δ*'s new values
  by field name (D8: a one-call affordance reaches the goal without the skill);
- ``random_k``: a few seeded random calls, arguments drawn from s₀ (keys of entities π* never touches) and schema
  defaults, never π*'s own values (D8).
A pass is confirmed; a failure is clean.
"""

from __future__ import annotations

import random
from typing import Any, ClassVar

from crucible.synth.canon import cjson
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, delta_of, entity_index, key_text, typed_like
from crucible.synth.falsify.mut import key_colls
from crucible.synth.program import Lit, Program, Run, Step
from crucible.synth.task import Task

DEFECT = {"noop": "D4", "read_all": "D4", "answer_only": "D4", "minus_last_write": "D1", "write_all": "D8", "shortcut": "D8", "random_k": "D8"}
MAX_WRITE_ALL = 40
N_RANDOM = 6


def _call(i: int, tool: str, args: dict[str, Any]) -> Step:
    return Step(id=f"n{i}", tool=tool, args={k: Lit(v=v) for k, v in sorted(args.items())})


def borrowed_args(oracle: Run) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for c in oracle.calls:
        out.update(c.args)
    return out


def minus_last_write(task: Task, ro: dict[str, bool]) -> Program | None:
    steps = task.oracle.steps
    w = [i for i, s in enumerate(steps) if not ro.get(s.tool, False)]
    if not w:
        return None
    i = w[-1]
    return task.oracle.model_copy(update={"steps": steps[:i] + steps[i + 1:]}, deep=True)


class NULL(BaseFalsifier):
    name: ClassVar[str] = "NULL"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        oracle = await ctx.oracle(task)
        tools = sorted(await (await ctx.world(task)).tools(), key=lambda t: t.name)
        ro = {t.name: t.read_only for t in tools}
        delta = delta_of(oracle, task.contract)
        base = borrowed_args(oracle)
        idx = entity_index(oracle.start)
        out = [Probe(cls="NULL", op="noop", site="noop", program=Program())]
        reads = [c for c in oracle.calls if c.read_only]
        steps = [_call(i, c.tool, c.args) for i, c in enumerate(reads)]
        for t in tools:
            if t.read_only and not (t.input_schema.get("required") or []):
                steps.append(_call(len(steps), t.name, {}))
        out.append(Probe(cls="NULL", op="read_all", site="read_all", program=Program(steps=steps)))
        if delta and oracle.answer is not None:
            out.append(Probe(cls="NULL", op="answer_only", site="answer_only", program=Program(answer=Lit(v=oracle.answer))))
        mlw = minus_last_write(task, ro)
        if mlw is not None:
            out.append(Probe(cls="NULL", op="minus_last_write", site="minus_last_write", program=mlw))
        # write_all: every write tool x every key pi* read
        read_keys: list[tuple[str, Any, list[tuple[str, str]]]] = []
        for c in reads:
            for name, v in sorted(c.args.items()):
                cols = key_colls(v, c.tool, idx)
                if cols:
                    read_keys.append((name, v, cols))
        wa: list[Step] = []
        for t in tools:
            if t.read_only:
                continue
            props = t.input_schema.get("properties") or {}
            req = t.input_schema.get("required") or []
            for name, v, _ in read_keys:
                if name not in props:
                    continue
                args = {k: base[k] for k in props if k in base}
                args[name] = v
                if all(r in args for r in req) and len(wa) < MAX_WRITE_ALL:
                    wa.append(_call(len(wa), t.name, args))
        oracle_writes = [(c.tool, cjson(c.args)) for c in oracle.calls if not c.read_only]
        if wa and not all((st.tool, cjson({k: a.v for k, a in st.args.items()})) in oracle_writes for st in wa):  # not pi*'s own writes
            out.append(Probe(cls="NULL", op="write_all", site="write_all", program=Program(steps=wa)))
        # shortcut: a write tool pi* never calls, aimed at an entity pi* changed, filled from the new values
        used = {s.tool for s in task.oracle.steps}
        fields: dict[tuple[str, str, str], dict[str, Any]] = {}
        for ch in delta:
            row = fields.setdefault(ch.addr.entity, {})
            if ch.kind == "update":
                row[ch.addr.field] = ch.new
            elif ch.kind == "insert":
                row.update(ch.new or {})
        for t in tools:
            if t.read_only or t.name in used:
                continue
            props = t.input_schema.get("properties") or {}
            req = t.input_schema.get("required") or []
            for (app, coll, key), vals in sorted(fields.items()):
                args = {k: base[k] for k in props if k in base}
                for k in props:
                    if k in vals:
                        args[k] = vals[k]
                    elif k in base and key_colls(base[k], t.name, idx)[:1] == [(app, coll)]:
                        args[k] = typed_like(key, base[k])
                if all(r in args for r in req):
                    out.append(Probe(cls="NULL", op="shortcut", site=f"shortcut:{t.name}:{key}", program=Program(steps=[_call(0, t.name, args)])))
        # random_k: arguments drawn from s0 (a key of the right collection, a value of a same-named field) or the schema
        # default, never from pi*'s own arguments (which would hand a random agent the solution); a call that cannot be
        # filled that way is not made
        rng = random.Random(f"NULL/{task.id}/{task.version}/{ctx.seed}")
        field_values: dict[str, list[Any]] = {}
        if oracle.start is not None:
            for _, _, _, row in oracle.start.entities():
                for f, v in row.items():
                    field_values.setdefault(f, []).append(v)
        touched = {c.addr.entity for c in delta} | {(a, c, key_text(v) or "") for call in oracle.calls for v in call.args.values()
                                                       for a, c in key_colls(v, call.tool, idx)}
        for j in range(N_RANDOM):
            st = []
            for _ in range(rng.randint(2, 5)):
                t = rng.choice(tools)
                args: dict[str, Any] = {}
                props = t.input_schema.get("properties") or {}
                for k in sorted(props):
                    cols = key_colls(base.get(k), t.name, idx)
                    if cols:
                        app, coll = cols[0]
                        off = sorted(x for x in oracle.start.collection(app, coll) if (app, coll, x) not in touched)  # type: ignore[union-attr]
                        if off:  # an entity pi* never touches: a random agent cannot stumble on the target
                            args[k] = typed_like(rng.choice(off), base[k])
                    elif k in field_values:
                        args[k] = rng.choice(sorted(field_values[k], key=cjson))
                    elif "default" in props[k]:
                        args[k] = props[k]["default"]
                if all(r in args for r in t.input_schema.get("required") or []):
                    st.append(_call(len(st), t.name, args))
            out.append(Probe(cls="NULL", op="random_k", site=f"random_k:{j}", program=Program(steps=st)))
        return out

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("confirmed", DEFECT[probe.op]) if outcome.passed else ("clean", None)
