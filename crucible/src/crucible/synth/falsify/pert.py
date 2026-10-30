"""PERT: metamorphic perturbation of the start state (part2-paper-core §5). Needs a ``Perturbable`` World.

Perturbations that preserve the task's intended outcome:
- ``shuffle``: permute the rows of a collection (two seeds per collection with 2+ rows). Row order is never identity.
- ``distractor``: clone an entity π* touches (an argument key of a call, or an entity Δ* changes) under a new key,
  inserted just before the original, with every field that holds an instruction-sourced literal changed (so the clone is
  provably not the entity ι names). Entities with no such field are not cloned: a clone of an entity the task selects by
  a property (``every pending invoice``) could change the intended outcome.
- ``rename``: text fields of an entity whose values π* never saw (no observation, argument or Δ* leaf) get a suffix.

Under each perturbation π* runs with its list refs re-resolved by the oracle run's anchors (``program.run``) and must
still pass; the no-op agent and π* minus its last write must still fail. Only flips count: the class does not apply
when π* already fails its contract, and a null agent that already passes unperturbed is NULL's finding, not PERT's.
Labels: π* fails → confirmed **D11** (shuffle: the verdict depends on row order) or **D6** (distractor, rename: a valid
solution is rejected); π* cannot run (a re-resolution failed, or a call errored that did not before) → suspect D6; a
null agent passes → confirmed D11. Replay re-runs the archived perturbation with the *new* version's oracle.
"""

from __future__ import annotations

import random
from typing import Any, ClassVar

from crucible.synth.canon import cjson, content_hash, is_number
from crucible.synth.falsify.base import BaseFalsifier, Counterexample, FalsifyContext, Label, Outcome, Probe, delta_of, entity_index, key_text, written_entities
from crucible.synth.falsify.mut import key_colls
from crucible.synth.falsify.null import minus_last_write
from crucible.synth.iface import Perturbable, Perturbation
from crucible.synth.program import Program, _loose_eq, leaves, scalar_leaves
from crucible.synth.task import Task

MAX_PROBES = 24


def _perturbed(v: Any) -> Any:
    if isinstance(v, bool):
        return v
    if is_number(v):
        return v + 1000
    if isinstance(v, str):
        return f"{v}-X"
    return v


def _new_key(key: str, keys: list[str]) -> str:
    if key.lstrip("-").isdigit() and all(k.lstrip("-").isdigit() for k in keys):
        return str(max(int(k) for k in keys) + 101)
    k, i = f"{key}-D", 1
    while k in keys:
        i += 1
        k = f"{key}-D{i}"
    return k


class PERT(BaseFalsifier):
    name: ClassVar[str] = "PERT"

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        w = await ctx.world(task)
        if not isinstance(w, Perturbable):
            return "the World cannot edit its start state (no perturb)"
        o = await ctx.oracle(task)
        if o.start is None:
            return "no canonical state"
        if not ctx.grade(o, task.contract).passed:
            return "the oracle fails its own contract (MUT Identity): no metamorphic baseline"
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        oracle = await ctx.oracle(task)
        s0 = oracle.start
        assert s0 is not None
        out: list[Probe] = []
        for app in sorted(s0.apps):
            for coll in sorted(s0.apps[app]):
                if len(s0.apps[app][coll]) >= 2:
                    for seed in (1, 2):
                        p = Perturbation(op="shuffle", app=app, coll=coll, seed=seed)
                        out.append(Probe(cls="PERT", op="shuffle", site=f"shuffle:{app}.{coll}:{seed}", perturb=[p], reresolve=True))
        idx = entity_index(s0)
        touched: set[tuple[str, str, str]] = set(written_entities(delta_of(oracle, task.contract)))
        for c in oracle.calls:
            for _, v in scalar_leaves(c.args, ""):
                for app, coll in key_colls(v, c.tool, idx)[:1]:
                    touched.add((app, coll, key_text(v) or ""))
        ins_lits = [e.value for e in oracle.taint if e.has("instruction") and not e.has("structural")]
        for app, coll, key in sorted(touched):
            row = s0.entity(app, coll, key)
            if row is None:
                continue
            over = {f: _perturbed(v) for f, v in row.items() if any(_loose_eq(v, x) or key_text(v) == key_text(x) is not None for x in ins_lits)}
            if not over and any(key_text(x) == key for x in ins_lits):
                over = {}  # the key itself is the instruction's literal: the clone's new key already differs
            elif not over:
                continue
            keys = sorted(s0.collection(app, coll))
            pos = list(s0.collection(app, coll)).index(key) if key in s0.collection(app, coll) else None
            p = Perturbation(op="clone", app=app, coll=coll, key=key, new_key=_new_key(key, keys), fields=over, position=pos)
            out.append(Probe(cls="PERT", op="distractor", site=f"distractor:{app}.{coll}[{key}]", perturb=[p], reresolve=True))
        seen_vals = [v for c in oracle.calls for _, v in [*scalar_leaves(c.args, ""), *leaves(c.obs.value)]]
        seen_keys = {e for e in touched}
        rng = random.Random(f"PERT/{task.id}/{task.version}/{ctx.seed}")
        for app in sorted(s0.apps):
            for coll in sorted(s0.apps[app]):
                cands = []
                for key, row in sorted(s0.apps[app][coll].items()):
                    if (app, coll, key) in seen_keys or any(_loose_eq(key, v) for v in seen_vals):
                        continue
                    texts = {f: f"{v} (renamed)" for f, v in row.items() if isinstance(v, str) and v and not any(_loose_eq(v, x) for x in seen_vals)}
                    if texts:
                        cands.append((key, texts))
                for key, texts in sorted(rng.sample(cands, min(2, len(cands)))):
                    p = Perturbation(op="set", app=app, coll=coll, key=key, fields=texts)
                    out.append(Probe(cls="PERT", op="rename", site=f"rename:{app}.{coll}[{key}]", perturb=[p], reresolve=True))
        return out[:MAX_PROBES]

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        oracle = await ctx.oracle(task)
        try:
            r = await ctx.run(task, task.oracle, perturb=probe.perturb, anchors=oracle.anchors)
        except (KeyError, ValueError) as e:  # the World refused the perturbation: not a trial
            return Outcome(valid=False, error=f"perturb: {type(e).__name__}")
        v = ctx.grade(r, task.contract)
        errored = {c.step for c in oracle.calls if c.obs.error is not None}
        broke = r.error is not None or any(c.obs.error is not None and c.step not in errored for c in r.calls)
        nulls = {"noop": Program()}
        tools = await (await ctx.world(task)).tools()
        mlw = minus_last_write(task, {t.name: t.read_only for t in tools})
        if mlw is not None:
            nulls["minus_last_write"] = mlw
        passing = []
        for name, prog in nulls.items():
            base = ctx.grade(await ctx.run(task, prog), task.contract).passed  # only a flip is PERT's finding (NULL owns the rest)
            if not base and ctx.grade(await ctx.run(task, prog, perturb=probe.perturb), task.contract).passed:
                passing.append(name)
        return Outcome(passed=v.passed, reasons=v.reasons, error=r.error, data={"oracle_broke": broke, "null_passed": passing})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if outcome.data.get("null_passed"):
            return "confirmed", "D11"
        if outcome.passed:
            return "clean", None
        if outcome.data.get("oracle_broke"):
            return "suspect", "D6"
        return "confirmed", "D11" if probe.op == "shuffle" else "D6"

    async def replay(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool:
        out = await self.execute(cx.probe, task, ctx)
        if not out.valid:
            return False
        lab, _ = self.label(cx.probe, out)
        return lab == "confirmed"


def perturbation_digest(p: Perturbation) -> str:
    return content_hash(cjson(p.model_dump(mode="json")), 12)
