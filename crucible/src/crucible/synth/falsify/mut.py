"""MUT: program mutation (part2-paper-core §5), the extension of ``accept/mutation/actions.py`` to π*'s DSL.

Operators (every mutant is an executable ``Program``; ``data`` says what the operator removed or redirected):
- ``Identity``: π* itself; it must pass. A failure is the oracle rejected by its own contract: D10 when a required
  effect is missing (no tool sequence the builder found reaches it), else D6 (an over-strict frame, trace or answer).
- ``DropStep(i)``, ``DupStep(i)``, ``Truncate(k)`` (keep the first ``k`` steps).
- ``SwapEntity``: an argument whose value is an entity key at s₀ → **every** sibling key of the same collection (up to
  ``MAX_SIBLINGS``, seeded sample beyond it).
- ``PerturbValue``: a numeric argument at several magnitudes (ints ±1, ±10, ±100, ×2, 0; floats ×0.5, ×0.9, ×0.99,
  ×1.01, ×1.1, ×2, ±1); an enum-like string → every other value seen for that name (the tool schema's enum, s₀ cells of
  the same field name, π*'s other arguments of that name), up to ``MAX_ALTS``.
- ``ExtraWrite``: a copy of a write step on **every** distractor entity (a sibling key π* does not write), inserted
  after it; and every write tool π* never calls, with arguments borrowed by name from π*'s calls.
- ``Reorder``: two adjacent writes, neither referring to the other, swapped.
- ``WrongAnswer``: the answer replaced by a plausible wrong one (several magnitudes for a number).
- **The unstated-trace-constraint family** (``TRACE_OPS``; DESIGN §7.7 item 20). Mutants a correct contract cannot tell
  from π*, because they only add or move *reads*: ``InsertRead`` (a call of every read tool, once per entity class its
  arguments can name, at the start and at the end of π*: arguments by name from π*'s calls, else an entity key of a
  collection the parameter or tool names), ``ExtraList`` (a harmless list or search call with only the arguments it
  requires, borrowed from π*), ``SwapReads`` (two adjacent independent reads swapped). Each is equivalent to π* under
  the *stated* constraints, so it must pass: a failure is a confirmed D6 (an unstated ``never``, ``at_most`` or
  ``before`` demand). A non-equivalent one (the read changed what a later step sent) tests nothing and is labelled
  clean. ``MUT(trace=False)`` is the M3 neighbourhood without them. On a correct contract these mutants share π*'s
  effect, so effect deduplication keeps n unchanged.

The neighbourhood scales with the World (M3, §6): a task over a collection of 40 entities gets about 80 swap and
extra-write mutants per written key, so U_MUT ≤ 0.05 (n ≥ 59) is reachable for 1–4 step tasks; a World with few
entities cannot reach it, and the task is admitted only as *provisional* (``certificate``).

Families are interleaved round-robin up to ``ctx.max_mutants`` so no family crowds out the others (accept V18), and
duplicate programs are dropped. After execution, ``finalize`` **deduplicates by normalised effect**
(``base.effect_signature``: Δ under the contract's normalizers, the answer, the reason codes): mutants with one effect
get one verdict, so only the most informative of them (confirmed, then suspect, equivalent, clean; then the earliest)
counts as a trial; the others stay in the result with ``valid = False`` and ``data["duplicate_of"]``. A certificate's
n is the number of *distinct* behaviours the contract was tested on, which is what the rule of three needs.

Label rule (§5): an equivalent mutant must pass (failing: confirmed D6); a non-equivalent one must fail. Survivors of
DropStep on a write, Truncate that removed a write, SwapEntity on a key π* writes, ExtraWrite whose effects fall outside
Δ*, and WrongAnswer are confirmed (D1, D1, D9, D2, and D1 without answer checks / D3 with them); any other survivor is a
suspect D3.
"""

from __future__ import annotations

import random
import re
from typing import Any, ClassVar

from crucible.synth.canon import is_number
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Finding, Label, Outcome, Probe, dedupe_programs, delta_of, entity_index, key_text, typed_like, written_entities
from crucible.synth.iface import ToolInfo
from crucible.synth.program import Fn, Lit, Program, Ref, Run, Step, _loose_eq
from crucible.synth.task import Task

MAX_SIBLINGS = 64
MAX_ALTS = 12
TRACE_OPS = ("InsertRead", "ExtraList", "SwapReads")
LIST_WORDS = ("list", "search", "find", "query", "browse", "lookup")
MAX_READ_ARGS = 4  # argument sets per read tool (one per entity class it can name)
RANK = {"confirmed": 0, "suspect": 1, "equivalent": 2, "clean": 3}


def _refs(step: Step) -> set[str]:
    out: set[str] = set()

    def walk(a: Any) -> None:
        if isinstance(a, Ref):
            out.add(a.path.split(".", 1)[0])
        elif isinstance(a, Fn):
            for x in a.args:
                walk(x)

    for a in step.args.values():
        walk(a)
    return out


def _with_arg(p: Program, i: int, name: str, value: Any) -> Program:
    steps = [s.model_copy(deep=True) for s in p.steps]
    steps[i].args[name] = Lit(v=value)
    return p.model_copy(update={"steps": steps}, deep=True)


def _siblings(s0: Any, app: str, coll: str, key: str, rng: random.Random, exclude: set[str] = frozenset()) -> list[str]:  # type: ignore[assignment]
    keys = sorted(k for k in s0.collection(app, coll) if k != key and k not in exclude)
    if len(keys) > MAX_SIBLINGS:
        keys = sorted(rng.sample(keys, MAX_SIBLINGS))
    return keys


def key_colls(value: Any, tool: str, idx: dict[str, list[tuple[str, str]]]) -> list[tuple[str, str]]:
    """The collections where ``value`` is an entity key, the tool's own App first."""
    k = key_text(value)
    if k is None or k not in idx:
        return []
    app = tool.split(".", 1)[0]
    return sorted(idx[k], key=lambda ac: (ac[0] != app, ac))


def enum_alternatives(name: str, value: Any, tool: ToolInfo | None, oracle: Run, program: Program) -> list[Any]:
    seen: list[Any] = []
    if tool is not None:
        seen += list(((tool.input_schema.get("properties") or {}).get(name) or {}).get("enum") or [])
    if oracle.start is not None:
        for _, _, _, row in oracle.start.entities():
            if name in row and isinstance(row[name], str):
                seen.append(row[name])
    for c in oracle.calls:
        if isinstance(c.args.get(name), str):
            seen.append(c.args[name])
    out: list[Any] = []
    for v in sorted(set(seen), key=str):
        if not _loose_eq(v, value) and v not in out:
            out.append(v)
    return out[:MAX_ALTS]


def number_alternatives(v: Any) -> list[Any]:
    """Several magnitudes, so a tolerance of any width between 1% and 100% has a mutant just outside it."""
    if isinstance(v, int):
        cand = [v + 1, v - 1, v + 10, v - 10, v + 100, v * 2, 0]
    else:
        cand = [round(v * f, 6) for f in (1.01, 0.99, 1.1, 0.9, 2.0, 0.5)] + [v + 1.0, v - 1.0]
    out: list[Any] = []
    for c in cand:
        if c != v and c not in out:
            out.append(c)
    return out


def wrong_answers(a: Any, oracle: Run) -> list[Any]:
    if isinstance(a, bool):
        return [not a]
    if is_number(a):
        out = [a + 1, a * 1.1 if a else 1, a - 1, a + 10, a * 2] if isinstance(a, int) else [round(a * 1.1, 6) if a else 1.0, a + 1.0, round(a * 0.5, 6),
                                                                                              round(a * 1.001, 6), round(a * 1.01, 6), round(a * 2, 6)]
        res: list[Any] = []
        for x in out:
            if x != a and x not in res:
                res.append(x)
        return res
    if isinstance(a, str):
        others = [v for c in oracle.calls for _, v in _str_leaves(c.obs.value) if not _loose_eq(v, a)]
        return [a + " (not)", *sorted(set(others))[:2]]
    if isinstance(a, list):
        return ([a[1:]] if a else []) + [[*a, a[-1] if a else "x"]]
    return []


def touched_keys(oracle: Run, idx: dict[str, list[tuple[str, str]]]) -> set[tuple[str, str, str]]:
    """Entity keys π*'s calls name (arguments that are keys at s₀)."""
    out: set[tuple[str, str, str]] = set()
    for c in oracle.calls:
        for v in c.args.values():
            k = key_text(v)
            if k is not None and k in idx:
                for a, coll in idx[k]:
                    out.add((a, coll, k))
    return out


def _words(name: str) -> set[str]:
    out = set()
    for w in re.split(r"[^a-z0-9]+", re.sub(r"([a-z])([A-Z])", r"\1 \2", name).lower()):
        if w and w not in ("id", "key", "ids", "keys", "get", "read", "by"):
            out.add(w[:-1] if w.endswith("s") and len(w) > 3 else w)
    return out


def read_arg_sets(tool: ToolInfo, borrowed: dict[str, Any], s0: Any, touched: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    """Argument sets for an unused read of ``tool``: the borrowed-only fill, plus one per entity class (collection) a
    parameter can name. A parameter is filled from π*'s arguments by name, else with a key of a collection whose name
    shares a word with the parameter (``ticket_id`` -> ``tickets``; a bare ``id`` with the tool's name, ``get_ticket``),
    π*'s own key in that collection first. A tool whose required parameters cannot be filled gets none."""
    props = tool.input_schema.get("properties") or {}
    req = list(tool.input_schema.get("required") or [])
    colls = sorted({(a, c) for a, c, _, _ in s0.entities()}) if s0 is not None else []
    tool_words = _words(tool.name.split(".", 1)[-1])
    out: list[dict[str, Any]] = []
    base = {r: borrowed[r] for r in req if r in borrowed}
    if len(base) == len(req):
        out.append(base)
    for app, coll in colls:
        cw = _words(coll)
        for p in props:
            pw = _words(p)
            if not (cw & pw or (not pw and cw & tool_words)):
                continue
            keys = sorted(s0.collection(app, coll))
            if not keys:
                continue
            mine = sorted(k for a, c, k in touched if (a, c) == (app, coll))
            k = (mine or keys)[0]
            args = {**base, p: typed_like(k, borrowed.get(p, k))}
            if all(r in args for r in req) and args not in out:
                out.append(args)
            break
        if len(out) >= MAX_READ_ARGS:
            break
    return out


def trace_mutants(p: Program, tools: dict[str, ToolInfo], oracle: Run, touched: set[tuple[str, str, str]]) -> dict[str, list[Probe]]:
    """The unstated-trace-constraint family (module docstring): reads added, list calls added, reads swapped."""
    fam: dict[str, list[Probe]] = {k: [] for k in TRACE_OPS}
    ro = {s.id: tools[s.tool].read_only if s.tool in tools else False for s in p.steps}
    borrowed: dict[str, Any] = {}
    for c in oracle.calls:
        borrowed.update(c.args)
    ids = {s.id for s in p.steps}

    def fresh(stem: str) -> str:
        i = 0
        while f"{stem}{i}" in ids:
            i += 1
        ids.add(f"{stem}{i}")
        return f"{stem}{i}"

    def placed(step: Step, where: str) -> Program:
        steps = [step, *p.steps] if where == "start" else [*p.steps, step]
        return p.model_copy(update={"steps": [s.model_copy(deep=True) for s in steps]}, deep=True)

    for t in sorted(tools.values(), key=lambda t: t.name):
        if not t.read_only:
            continue
        short = t.name.split(".", 1)[-1].lower()
        req = t.input_schema.get("required") or []
        if short.startswith(LIST_WORDS) and all(r in borrowed for r in req):
            args = {r: borrowed[r] for r in req}
            for where in ("start", "end"):
                st = Step(id=fresh("xl"), tool=t.name, args={k: Lit(v=v) for k, v in args.items()})
                fam["ExtraList"].append(Probe(cls="MUT", op="ExtraList", site=f"{t.name}@{where}", program=placed(st, where), data={"tool": t.name}))
        for j, args in enumerate(read_arg_sets(t, borrowed, oracle.start, touched)):
            for where in ("start", "end"):
                st = Step(id=fresh("xr"), tool=t.name, args={k: Lit(v=v) for k, v in args.items()})
                fam["InsertRead"].append(Probe(cls="MUT", op="InsertRead", site=f"{t.name}#{j}@{where}", program=placed(st, where),
                                               data={"tool": t.name, "args": args}))
    for i in range(len(p.steps) - 1):
        a, b = p.steps[i], p.steps[i + 1]
        if not (ro[a.id] and ro[b.id]) or a.id in _refs(b) or b.id in _refs(a):
            continue
        steps = [*p.steps[:i], b, a, *p.steps[i + 2:]]
        fam["SwapReads"].append(Probe(cls="MUT", op="SwapReads", site=f"{a.id}<->{b.id}", program=p.model_copy(update={"steps": steps}, deep=True)))
    return fam


def _str_leaves(v: Any) -> list[tuple[str, str]]:
    from crucible.synth.program import leaves

    return [(p, x) for p, x in leaves(v) if isinstance(x, str) and 0 < len(x) < 40]


class MUT(BaseFalsifier):
    """``trace`` adds the unstated-trace-constraint family (``TRACE_OPS``)."""

    name: ClassVar[str] = "MUT"

    def __init__(self, *, trace: bool = True):
        self.trace = trace

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        p = task.oracle
        oracle = await ctx.oracle(task)
        tools = {t.name: t for t in await (await ctx.world(task)).tools()}
        ro = {s.id: tools[s.tool].read_only if s.tool in tools else False for s in p.steps}
        rng = random.Random(f"MUT/{task.id}/{task.version}/{ctx.seed}")
        idx = entity_index(oracle.start)
        written = written_entities(delta_of(oracle, task.contract))
        written_keys = {(a, c, k) for a, c, k in written}
        fam: dict[str, list[Probe]] = {k: [] for k in ("DropStep", "DupStep", "Truncate", "SwapEntity", "PerturbValue", "ExtraWrite", "Reorder", "WrongAnswer")}
        n = len(p.steps)
        for i, s in enumerate(p.steps):
            fam["DropStep"].append(Probe(cls="MUT", op="DropStep", site=s.id, program=p.model_copy(update={"steps": p.steps[:i] + p.steps[i + 1:]}, deep=True),
                                         data={"write": not ro[s.id]}))
            dup = s.model_copy(update={"id": f"{s.id}_dup"}, deep=True)
            fam["DupStep"].append(Probe(cls="MUT", op="DupStep", site=s.id, program=p.model_copy(update={"steps": [*p.steps[:i + 1], dup, *p.steps[i + 1:]]}, deep=True),
                                        data={"write": not ro[s.id]}))
        for k in range(1, n):
            fam["Truncate"].append(Probe(cls="MUT", op="Truncate", site=str(k), program=p.model_copy(update={"steps": p.steps[:k]}, deep=True),
                                         data={"write": any(not ro[s.id] for s in p.steps[k:])}))
        for i, s in enumerate(p.steps):
            rec = oracle.call(s.id)
            if rec is None:
                continue
            for name in sorted(s.args):
                v = rec.args.get(name)
                colls = key_colls(v, s.tool, idx)
                if colls:
                    app, coll = colls[0]
                    kt = key_text(v)
                    for sib in _siblings(oracle.start, app, coll, kt or "", rng):
                        fam["SwapEntity"].append(Probe(cls="MUT", op="SwapEntity", site=f"{s.id}.{name}->{sib}", program=_with_arg(p, i, name, typed_like(sib, v)),
                                                       data={"written": (app, coll, kt) in written_keys}))
                    if not ro[s.id]:  # ExtraWrite on distractors: the write repeated on a key π* does not write
                        others = {k for (a, c, k) in written_keys if (a, c) == (app, coll)}
                        for j, sib in enumerate(_siblings(oracle.start, app, coll, kt or "", rng, exclude=others)):
                            extra = s.model_copy(deep=True)
                            extra.id = f"{s.id}_x{j}"
                            extra.args[name] = Lit(v=typed_like(sib, v))
                            fam["ExtraWrite"].append(Probe(cls="MUT", op="ExtraWrite", site=f"{s.id}.{name}+{sib}",
                                                           program=p.model_copy(update={"steps": [*p.steps[:i + 1], extra, *p.steps[i + 1:]]}, deep=True)))
                    continue
                if is_number(v):
                    alts = number_alternatives(v)
                elif isinstance(v, str):
                    alts = enum_alternatives(name, v, tools.get(s.tool), oracle, p)
                else:
                    alts = []
                for a in alts:
                    fam["PerturbValue"].append(Probe(cls="MUT", op="PerturbValue", site=f"{s.id}.{name}={a}", program=_with_arg(p, i, name, a)))
        used = {s.tool for s in p.steps}
        borrowed: dict[str, Any] = {}
        for c in oracle.calls:
            borrowed.update(c.args)
        for t in sorted(tools.values(), key=lambda t: t.name):
            if t.read_only or t.name in used:
                continue
            props = t.input_schema.get("properties") or {}
            req = t.input_schema.get("required") or []
            if any(r not in borrowed for r in req):
                continue
            args = {k: borrowed[k] for k in props if k in borrowed}
            extra = Step(id="xw", tool=t.name, args={k: Lit(v=v) for k, v in args.items()})
            fam["ExtraWrite"].append(Probe(cls="MUT", op="ExtraWrite", site=f"tool:{t.name}", program=p.model_copy(update={"steps": [*p.steps, extra]}, deep=True)))
        for i in range(n - 1):
            a, b = p.steps[i], p.steps[i + 1]
            if ro[a.id] or ro[b.id] or a.id in _refs(b) or b.id in _refs(a):
                continue
            steps = [*p.steps[:i], b, a, *p.steps[i + 2:]]
            fam["Reorder"].append(Probe(cls="MUT", op="Reorder", site=f"{a.id}<->{b.id}", program=p.model_copy(update={"steps": steps}, deep=True)))
        if getattr(self, "trace", True):
            fam.update(trace_mutants(p, tools, oracle, written_keys | touched_keys(oracle, idx)))
        if p.answer is not None and oracle.answer is not None:
            for w in wrong_answers(oracle.answer, oracle):
                fam["WrongAnswer"].append(Probe(cls="MUT", op="WrongAnswer", site=f"answer={w}", program=p.model_copy(update={"answer": Lit(v=w)}, deep=True),
                                                data={"q": bool(task.contract.answer)}))
        out = [Probe(cls="MUT", op="Identity", site="-", program=p, expect="pass")]
        seen = {p.digest()}
        queues = [dedupe_programs(v, set()) for v in fam.values()]
        while any(queues) and len(out) < ctx.max_mutants:
            for q in queues:
                if q and len(out) < ctx.max_mutants:
                    pr = q.pop(0)
                    d = pr.program.digest() if pr.program is not None else pr.site
                    if d not in seen:
                        seen.add(d)
                        out.append(pr)
        return out

    def finalize(self, findings: list[Finding]) -> list[Finding]:
        """Deduplicate by normalised effect (see the module docstring)."""
        groups: dict[str, list[int]] = {}
        for i, f in enumerate(findings):
            e = f.outcome.data.get("effect")
            if f.outcome.valid and e is not None:
                groups.setdefault(e, []).append(i)
        out = list(findings)
        for idx in groups.values():
            keep = min(idx, key=lambda i: (RANK.get(findings[i].label, 9), i))
            for i in idx:
                if i != keep:
                    o = findings[i].outcome.model_copy(update={"valid": False, "data": {**findings[i].outcome.data, "duplicate_of": findings[keep].site}})
                    out[i] = findings[i].model_copy(update={"outcome": o})
        return out

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        if probe.op == "Identity":  # the oracle rejected: a required effect it cannot produce (D10), else over-strict (D6)
            if outcome.passed:
                return "equivalent", None
            return "confirmed", "D10" if any(r.startswith("MISSING_REQUIRED") for r in outcome.reasons) else "D6"
        if outcome.equivalent:
            return ("equivalent", None) if outcome.passed else ("confirmed", "D6")
        if not outcome.passed:
            return "clean", None
        op, d = probe.op, probe.data
        if op == "DropStep" and d.get("write"):
            return "confirmed", "D1"
        if op == "Truncate" and d.get("write"):
            return "confirmed", "D1"
        if op == "SwapEntity" and d.get("written"):
            return "confirmed", "D9"
        if op == "ExtraWrite" and outcome.data.get("outside"):
            return "confirmed", "D2"
        if op == "WrongAnswer":
            return "confirmed", "D3" if d.get("q") else "D1"
        if op in TRACE_OPS:  # a read that changed what π* sent is no trace-only mutant: a trial of nothing
            return "clean", None
        return "suspect", "D3"
