"""The falsifier protocol (part2-paper-core §5) and what every class shares.

A falsifier class implements ``propose(task, ctx) -> probes``, ``execute(probe, task, ctx) -> outcome`` and
``label(probe, outcome) -> (label, defect)`` with label ∈ {confirmed, suspect, clean, equivalent}:
- **confirmed**: an executed counterexample whose label rule needs no model (it goes to the archive);
- **suspect**: a probe whose verdict needs adjudication (M4's panel);
- **clean**: a trial that found nothing; **equivalent**: a probe equivalent to the oracle that passed, as it must.
A probe that could not be evaluated (a dead twin, a perturbation the World refused) is ``valid = False`` and is not a
trial: it counts in no certificate. ``replay(cx, task, ctx)`` re-runs an archived counterexample against another version
of its task and says whether it **re-opened** (the monotone-archive rule of §7).

**Equivalence** (MUT, §5): a run is equivalent to the oracle's when its normalised diff equals Δ* (``canon.same_diff``
under the contract's normalizers), its answer equals the oracle's (``contract.answer_equal``; both absent counts as
equal), and every trace constraint the instruction *states* (``TraceConstraint.stated`` found in ι) has the same truth
value on both. Unstated constraints are left out on purpose: an equivalent run that violates one is an over-strict
contract (D6), which is how an unstated order or format demand is caught.

**Effects and scheduling (M3).** ``execute`` records each program probe's ``effect_signature`` (normalised Δ, answer,
reason codes): two probes with one signature get one verdict, so a class may ``finalize`` its findings by keeping one
trial per effect (MUT does). ``run_classes`` is the one scheduler every caller uses (CEGIS, recall, the library): REP
first, and when REP confirms nondeterminism MUT, TWIN and PERT are recorded as not applicable instead of being run
(§5.1 rule 3). With ``cache=True`` it keeps one result per class, configuration and task version.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, ClassVar, Literal, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import CanonState, Change, cjson, content_hash, diff, norm_text, same_diff
from crucible.synth.contract import EXEC, Contract, ExecFn, Verdict, _trace_ok, answer_equal, grade
from crucible.synth.iface import Perturbation, SynthWorld
from crucible.synth.program import Program, Run, run
from crucible.synth.task import Task

Label = Literal["confirmed", "suspect", "clean", "equivalent"]

DEFECTS: dict[str, str] = {
    "D1": "LAX-REQ", "D2": "LAX-FRAME", "D3": "LAX-VALUE", "D4": "VACUOUS", "D5": "HIDDEN", "D6": "OVER", "D7": "LEAK",
    "D8": "AFFORD", "D9": "INSENS", "D10": "UNSOLV", "D11": "NONDET",
}
CLASSES = ("MUT", "NULL", "RED", "TAINT", "TWIN", "PERT", "REP", "ALT", "ATK", "ONL", "LEAK")
DETERMINISTIC = ("MUT", "NULL", "RED", "TAINT", "TWIN", "PERT", "REP")


class Probe(CrucibleModel):
    cls: str
    op: str
    site: str
    program: Program | None = None  # the agent to run (a mutant, a null agent, the oracle)
    instruction: str | None = None  # a twin's instruction (None: the task's)
    contract: Contract | None = None  # a twin's contract (None: the task's)
    perturb: list[Perturbation] = Field(default_factory=list)
    reresolve: bool = False  # resolve list refs by the oracle run's anchors
    expect: Literal["fail", "pass", "property"] = "fail"
    data: dict[str, Any] = Field(default_factory=dict)

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json", exclude={"data"}) | {"site": self.site}, 32)


class Outcome(CrucibleModel):
    valid: bool = True
    passed: bool | None = None
    reasons: list[str] = Field(default_factory=list)
    equivalent: bool | None = None
    error: str | None = None
    data: dict[str, Any] = Field(default_factory=dict)


class Finding(CrucibleModel):
    cls: str
    op: str
    site: str
    label: Label
    defect: str | None = None
    probe: Probe
    outcome: Outcome

    @property
    def counted(self) -> bool:
        return self.outcome.valid


class ClassResult(CrucibleModel):
    cls: str
    applicable: bool = True
    reason: str | None = None  # why the class does not apply (a black-box World, no perturbation support, ...)
    findings: list[Finding] = Field(default_factory=list)

    @property
    def n(self) -> int:
        return sum(1 for f in self.findings if f.outcome.valid)

    @property
    def confirmed(self) -> list[Finding]:
        return [f for f in self.findings if f.outcome.valid and f.label == "confirmed"]

    @property
    def suspects(self) -> list[Finding]:
        return [f for f in self.findings if f.outcome.valid and f.label == "suspect"]

    @property
    def c(self) -> int:
        return len(self.confirmed)


class Counterexample(CrucibleModel):
    """An archived counterexample: the probe that found it, which task version it was found on, the defect, and the
    verdict the task *should* give (``expect``; ``property`` for a static finding such as an unsourced literal)."""

    id: str
    domain: str
    task_id: str
    task_version: int
    cls: str
    op: str
    site: str
    defect: str | None = None
    label: Label = "confirmed"
    probe: Probe
    reasons_seen: list[str] = Field(default_factory=list)

    @classmethod
    def of(cls, task: Task, f: Finding) -> Counterexample:
        probe = f.probe
        trace = f.outcome.data.get("program")
        if probe.program is None and isinstance(trace, dict) and f.outcome.passed is not None:  # an agent's attempt: replay its trace
            probe = probe.model_copy(update={"program": Program.model_validate(trace), "expect": "fail" if f.outcome.passed else "pass"})
        elif probe.expect != "property" and f.outcome.passed is not None:  # the verdict the task gave was the wrong one
            probe = probe.model_copy(update={"expect": "fail" if f.outcome.passed else "pass"})
        cid = content_hash({"task": task.id, "cls": f.cls, "op": f.op, "site": f.site, "probe": probe.digest(), "defect": f.defect}, 24)
        return cls(id=f"cx-{cid}", domain=task.domain, task_id=task.id, task_version=task.version, cls=f.cls, op=f.op, site=f.site,
                   defect=f.defect, label=f.label, probe=probe, reasons_seen=list(f.outcome.reasons))


# ---------------------------------------------------------------------------------------------------------- context

WorldMaker = Callable[[Task], SynthWorld | Awaitable[SynthWorld]]


class FalsifyContext:
    """What falsifiers share for one batch of tasks: World construction (one World per ``WorldRef``, reset before every
    run), the oracle run cache, the exec registry, the grader deadline and budgets."""

    def __init__(self, make_world: WorldMaker | None = None, *, exec_registry: dict[str, ExecFn] | None = None, deadline_s: float = 30.0,
                 max_mutants: int = 240, seed: int = 0):
        self.make_world = make_world
        self.exec_registry = EXEC if exec_registry is None else exec_registry
        self.deadline_s = deadline_s
        self.max_mutants = max_mutants
        self.seed = seed
        self._worlds: dict[str, SynthWorld] = {}
        self._oracle: dict[str, Run] = {}
        self._odelta: dict[tuple[str, str], list[Change]] = {}
        self.class_cache: dict[tuple[str, ...], ClassResult] = {}

    async def world(self, task: Task) -> SynthWorld:
        k = task.world.model_dump_json()
        if k not in self._worlds:
            if self.make_world is not None:
                w = self.make_world(task)
                self._worlds[k] = await w if inspect.isawaitable(w) else w  # type: ignore[assignment]
            else:
                self._worlds[k] = await task.world.build()
        return self._worlds[k]

    async def run(self, task: Task, program: Program, *, instruction: str | None = None, perturb: Sequence[Perturbation] = (),
                  anchors: dict[str, list[list[Any]]] | None = None, taint: bool = False, reset: bool = True) -> Run:
        w = await self.world(task)

        async def prep(world: SynthWorld) -> None:
            for p in perturb:
                await world.perturb(p)  # type: ignore[attr-defined]

        return await run(w, program, task.instruction if instruction is None else instruction, seed=task.seed, given=task.intent.given,
                         prepare=prep if perturb else None, anchors=anchors, taint=taint, reset=reset)

    async def oracle_delta(self, task: Task, contract: Contract | None = None) -> list[Change]:
        """Δ* of the task's oracle run under a contract's normalizers (default: the task's), cached."""
        c = contract or task.contract
        k = (task.digest(), c.normalizers.model_dump_json())
        if k not in self._odelta:
            self._odelta[k] = delta_of(await self.oracle(task), c)
        return self._odelta[k]

    async def oracle(self, task: Task) -> Run:
        k = task.digest()
        if k not in self._oracle:
            self._oracle[k] = await self.run(task, task.oracle, taint=True)
        return self._oracle[k]

    def grade(self, r: Run, contract: Contract) -> Verdict:
        return grade(r, contract, registry=self.exec_registry, deadline_s=self.deadline_s)

    async def close(self) -> None:
        for w in self._worlds.values():
            c = getattr(w, "close", None)
            if c is not None:
                res = c()
                if inspect.isawaitable(res):
                    await res
        self._worlds.clear()


# ---------------------------------------------------------------------------------------------------------- shared helpers

def delta_of(r: Run, contract: Contract) -> list[Change]:
    if r.start is None or r.end is None:
        return []
    return diff(r.start, r.end, contract.normalizers)


def stated(contract: Contract, instruction: str) -> list[Any]:
    ins = norm_text(instruction)
    return [m for m in contract.trace if m.stated and norm_text(m.stated) in ins]


def equivalent(r: Run, oracle: Run, contract: Contract, instruction: str, *, d_run: Sequence[Change] | None = None,
               d_oracle: Sequence[Change] | None = None) -> bool:
    """See the module docstring. ``d_run`` / ``d_oracle`` are the two diffs when the caller already has them."""
    if (r.start is None) != (oracle.start is None):
        return False
    if r.start is not None and not same_diff(delta_of(r, contract) if d_run is None else d_run, delta_of(oracle, contract) if d_oracle is None else d_oracle):
        return False
    if not ((r.answer is None and oracle.answer is None) or (r.answer is not None and oracle.answer is not None and answer_equal(r.answer, oracle.answer))):
        return False
    for m in stated(contract, instruction):
        try:
            if _trace_ok(m, r) != _trace_ok(m, oracle):
                return False
        except Exception:  # noqa: BLE001 - an unevaluable constraint is not evidence of equivalence
            return False
    return True


def effect_signature(delta: Sequence[Change], answer: Any, verdict: Verdict) -> str:
    """What a probe *did*, as the grader sees it: the normalised diff, the answer and the verdict's reason codes (which
    carry the truth of every trace constraint and exec check). Two runs with one signature get the same verdict under
    the contract, so they are one trial, not two."""
    ch = sorted([str(c.addr), c.kind, cjson(c.old), cjson(c.new)] for c in delta)
    return content_hash({"delta": ch, "answer": cjson(answer), "passed": verdict.passed, "reasons": sorted(verdict.reasons)}, 24)


def empty_run(start: CanonState | None, instruction: str) -> Run:
    """The do-nothing episode (s₀, s₀, ∅, no answer)."""
    return Run(instruction=instruction, start=start, end=start)


def written_entities(delta: Sequence[Change]) -> set[tuple[str, str, str]]:
    return {c.addr.entity for c in delta}


def entity_index(s: CanonState | None) -> dict[str, list[tuple[str, str]]]:
    """Key text -> the (app, coll) pairs where it is an entity key."""
    out: dict[str, list[tuple[str, str]]] = {}
    if s is None:
        return out
    for app, colls in s.apps.items():
        for coll, rows in colls.items():
            for k in rows:
                out.setdefault(k, []).append((app, coll))
    for v in out.values():
        v.sort()
    return out


def typed_like(target: str, like: Any) -> Any:
    """A key text in the type of the value it replaces (an int argument gets an int)."""
    if isinstance(like, int) and not isinstance(like, bool) and target.lstrip("-").isdigit():
        return int(target)
    return target


def key_text(v: Any) -> str | None:
    if isinstance(v, bool) or v is None:
        return None
    if isinstance(v, int | str):
        return str(v)
    return None


@runtime_checkable
class Falsifier(Protocol):
    name: ClassVar[str]

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]: ...

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome: ...

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]: ...

    async def replay(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool: ...


class BaseFalsifier:
    """Program probes: run the probe's program (with its instruction, contract and perturbations) and grade it.
    ``applies`` returns a reason when the class cannot run on this task's World."""

    name: ClassVar[str] = "?"

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        return None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        raise NotImplementedError

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        assert probe.program is not None
        oracle = await ctx.oracle(task)
        c = probe.contract or task.contract
        ins = probe.instruction if probe.instruction is not None else task.instruction
        r = await ctx.run(task, probe.program, instruction=ins, perturb=probe.perturb, anchors=oracle.anchors if probe.reresolve else None)
        v = ctx.grade(r, c)
        d = delta_of(r, c)
        d0 = await ctx.oracle_delta(task, c)
        eq = equivalent(r, oracle, c, ins, d_run=d, d_oracle=d0)
        outside = sorted({str(ch.addr) for ch in d} - {str(ch.addr) for ch in d0})
        return Outcome(passed=v.passed, reasons=v.reasons, equivalent=eq, error=r.error, data={"outside": outside, "effect": effect_signature(d, r.answer, v)})

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        raise NotImplementedError

    async def replay(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool:
        """Default: a program probe is re-run against the new version and must now give the verdict it should have
        given (``expect``); a property probe re-opens when the class confirms the same site again."""
        if cx.probe.program is not None and cx.probe.expect in ("fail", "pass") and cx.probe.contract is None:
            probe = cx.probe.model_copy(update={"instruction": None})
            out = await self.execute(probe, task, ctx)
            want_pass = cx.probe.expect == "pass"
            return bool(out.valid and out.passed is not None and out.passed != want_pass)
        return await self.reopens_site(cx, task, ctx)

    async def reopens_site(self, cx: Counterexample, task: Task, ctx: FalsifyContext) -> bool:
        k = (self.name, task.digest())
        if k not in ctx.class_cache:  # one run of the class per task version serves every archived site
            ctx.class_cache[k] = await falsify(task, self, ctx)  # type: ignore[arg-type]
        return any(f.site == cx.site and f.op == cx.op for f in ctx.class_cache[k].confirmed)


async def falsify(task: Task, f: Falsifier, ctx: FalsifyContext) -> ClassResult:
    """Run one falsifier class on one task: propose, execute, label, then the class's ``finalize`` (MUT drops mutants
    whose normalised effect repeats an earlier one: they are not independent trials)."""
    why = await f.applies(task, ctx) if hasattr(f, "applies") else None  # type: ignore[attr-defined]
    if why is not None:
        return ClassResult(cls=f.name, applicable=False, reason=why)
    out = ClassResult(cls=f.name)
    for p in await f.propose(task, ctx):
        o = await f.execute(p, task, ctx)
        lab, d = f.label(p, o) if o.valid else ("clean", None)
        out.findings.append(Finding(cls=f.name, op=p.op, site=p.site, label=lab, defect=d, probe=p, outcome=o))
    fin = getattr(f, "finalize", None)
    if fin is not None:
        out.findings = fin(out.findings)
    return out


# REP runs first and gates the classes whose label rules assume a deterministic World (§5.1 rule 3): in a
# nondeterministic World every mutant looks non-equivalent and every twin or perturbation looks like a flip, so their
# survivors would be mislabelled.
GATED_BY_REP = ("MUT", "MUT-HO", "TWIN", "PERT")


def _config_key(f: Falsifier) -> str:
    try:
        return cjson({k: (v.model_dump(mode="json") if isinstance(v, CrucibleModel) else repr(v)) for k, v in sorted(vars(f).items())})
    except (TypeError, ValueError):
        return repr(sorted(vars(f)))


async def run_classes(task: Task, falsifiers: Sequence[Falsifier], ctx: FalsifyContext, *, cache: bool = False) -> list[ClassResult]:
    """Every class on one task in the scheduled order: REP first (stable otherwise); when REP confirms a counterexample
    the gated classes are recorded as not applicable, with the reason, instead of being run. ``cache=True`` keeps one
    result per (class, configuration, task version) in ``ctx.class_cache`` (recall matrices and fictitious play
    re-measure the same plants under several populations)."""
    order = sorted(falsifiers, key=lambda f: f.name != "REP")
    out: list[ClassResult] = []
    nondet = False
    for f in order:
        if nondet and f.name in GATED_BY_REP:
            out.append(ClassResult(cls=f.name, applicable=False, reason="gated: REP confirmed nondeterminism"))
            continue
        k = (f.name, task.digest(), _config_key(f))
        if cache and k in ctx.class_cache:
            r = ctx.class_cache[k]  # type: ignore[index]
        else:
            r = await falsify(task, f, ctx)
            if cache:
                ctx.class_cache[k] = r  # type: ignore[index]
        out.append(r)
        if f.name == "REP" and r.c > 0:
            nondet = True
    return out


def dedupe_programs(probes: list[Probe], seen: set[str] | None = None) -> list[Probe]:
    seen = set() if seen is None else seen
    out = []
    for p in probes:
        k = cjson(p.program.model_dump(mode="json")) if p.program is not None else p.site
        if k in seen:
            continue
        seen.add(k)
        out.append(p)
    return out
