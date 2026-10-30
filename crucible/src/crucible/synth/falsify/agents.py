"""ALT and ATK: agent falsifiers (part2-paper-core §5), their LLM runners, and the adjudication panel (M4).

An ``AgentRunner`` is ``async (view, world) -> answer``: it sees only the instruction and the tool catalogue
(``AgentView``; never the contract), acts through ``world.call`` and returns its final answer. The World it gets records
every call (``RecordingWorld``), so each attempt becomes an ordinary ``Run`` for the trusted grader, and its trace becomes
a literal ``Program`` (``trace_program``) that the archive can replay like any mutant.

- **ALT** (alternative solvers from families that built nothing in the task): a pass is a clean D6 trial; a failure is a
  **suspect D6** (adjudicated: was it a valid solution the contract rejects?).
- **ATK** (attackers told to pass without doing the skill): a pass is a **suspect D8** (the panel decides D7/D8); a
  failure is clean.

**LLM runners** (``llm_agent``): a tool-calling loop over a ``roles.RoleClient`` with a call budget (``AgentBudget``).
Each turn the model sees the instruction, the tool catalogue and the history of its calls (observations cut to
``obs_chars``), and replies with one JSON object, a call or ``{"done": true, "final": ...}`` (``roles.AgentTurn``): a
JSON protocol, so every model family works the same way whatever its native tool format. The attacker additionally gets
the attack-strategy library (``roles.ATTACK_STRATEGIES``) and never the contract (``roles.ATTACKER.never``).

**Adjudication** (``Panel``, the ``cegis.Adjudicator``): two or more models of families that built nothing in the task
read the task, the intent card, the probe's trace (calls, arguments, observations), the answer and the grader's reason
codes; never the gate code, the contract or which falsifier made the probe (``roles.PANEL``). Each says whether the
attempt is a valid solution, with a confidence. The panel decides when every member agrees and the lowest confidence is
at least **λ**; otherwise it abstains and the probe goes to the **human queue** (a JSONL file) and stays a suspect.
A decided suspect is **confirmed** when the panel's verdict and the grader's disagree (a valid attempt the grader
failed: D6; an invalid attempt the grader passed: the falsifier's defect, D8 for ATK), else **clean**; the finding's
label is updated in place, so the certificate counts it. ``calibrate(panel, anchors)`` is the trust-or-escalate hook:
anchors with executed truth (an oracle trace is valid, a no-op or a mutant the falsifiers killed is not) choose the
smallest λ on a grid whose decided accuracy reaches the target, and report accuracy and coverage per λ.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, ClassVar

from pydantic import Field

from crucible.actors.llm import BudgetExceeded
from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import diff
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Finding, Label, Outcome, Probe
from crucible.synth.iface import Obs, SynthWorld, ToolInfo
from crucible.synth.program import CallRecord, Lit, Program, Run, Step
from crucible.synth.roles import ALT_SOLVER, ATTACK_STRATEGIES, ATTACKER, PANEL, AgentTurn, PanelOut, Role, RoleClient, compact, tool_catalogue
from crucible.synth.task import Task
from crucible.verify.calibrate import clopper_pearson_upper


class AgentView(CrucibleModel):
    instruction: str
    tools: list[ToolInfo]


AgentRunner = Callable[[AgentView, SynthWorld], Awaitable[Any]]


@dataclass
class AgentSpec:
    name: str
    family: str
    runner: AgentRunner


class RecordingWorld:
    """A ``SynthWorld`` proxy that records calls (with per-call effects when the World has a canon)."""

    def __init__(self, world: SynthWorld, tools: list[ToolInfo]):
        self.world = world
        self.ro = {t.name: t.read_only for t in tools}
        self.calls: list[CallRecord] = []
        self._last: Any = None

    async def reset(self, seed: int) -> Any:
        raise RuntimeError("an agent may not reset its World")

    async def tools(self) -> list[ToolInfo]:
        return await self.world.tools()

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        if self._last is None:
            self._last = await self.world.canon()
        obs = await self.world.call(tool, dict(args))
        rec = CallRecord(step=f"a{len(self.calls)}", tool=tool, args=dict(args), obs=obs, read_only=self.ro.get(tool, False))
        if not rec.read_only and self._last is not None:
            now = await self.world.canon()
            rec.effects = sorted({c.addr.entity for c in diff(self._last, now)}) if now is not None else None  # type: ignore[misc]
            self._last = now
        self.calls.append(rec)
        return obs

    async def snapshot(self) -> Any:
        raise RuntimeError("an agent may not snapshot its World")

    async def restore(self, ref: Any) -> None:
        raise RuntimeError("an agent may not restore its World")

    async def canon(self) -> None:
        return None  # the policy never sees the verifier's view


def trace_program(calls: list[CallRecord], answer: Any) -> Program:
    return Program(steps=[Step(id=f"a{i}", tool=c.tool, args={k: Lit(v=v) for k, v in sorted(c.args.items())}) for i, c in enumerate(calls)],
                   answer=None if answer is None else Lit(v=answer))


def trace_view(calls: Sequence[CallRecord], obs_chars: int = 1200) -> list[dict[str, Any]]:
    """A trace as a reviewer reads it: each call with its arguments and its (cut) observation or error."""
    return [{"tool": c.tool, "args": c.args, **({"error": c.obs.error[:400]} if c.obs.error else {"result": compact(c.obs.value, obs_chars)})}
            for c in calls]


async def run_agent(agent: AgentSpec, task: Task, ctx: FalsifyContext) -> Run:
    w = await ctx.world(task)
    await w.reset(task.seed)
    tools = await w.tools()
    start = await w.canon()
    rec = RecordingWorld(w, tools)
    err = None
    answer = None
    try:
        answer = await agent.runner(AgentView(instruction=task.instruction, tools=tools), rec)  # type: ignore[arg-type]
    except Exception as e:
        if isinstance(e, BudgetExceeded):  # the run's spend cap: stop, never a failed attempt
            raise
        err = f"agent: {type(e).__name__}"
    return Run(instruction=task.instruction, seed=task.seed, calls=rec.calls, answer=answer, start=start, end=await w.canon(),
               read_only={t.name: t.read_only for t in tools}, error=err)


class _AgentFalsifier(BaseFalsifier):
    def __init__(self, agents: list[AgentSpec] | None = None, *, exclude_families: set[str] | None = None):
        self.agents = list(agents or [])
        self.exclude = set(exclude_families or ())

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        return None if [a for a in self.agents if a.family not in self.exclude] else "no agent runners injected"

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        return [Probe(cls=self.name, op="agent", site=f"agent:{a.name}", expect="property", data={"agent": a.name, "family": a.family})
                for a in self.agents if a.family not in self.exclude]

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        agent = next(a for a in self.agents if a.name == probe.data["agent"])
        r = await run_agent(agent, task, ctx)
        if r.error is not None and r.error.startswith("agent:"):  # the model or its client failed: no attempt, not a trial
            return Outcome(valid=False, error=r.error, data={"family": agent.family, "calls": len(r.calls)})
        v = ctx.grade(r, task.contract)
        return Outcome(passed=v.passed, reasons=v.reasons, error=r.error,
                       data={"program": trace_program(r.calls, r.answer).model_dump(mode="json"), "family": agent.family,
                             "trace": trace_view(r.calls), "answer": r.answer, "calls": len(r.calls)})


class ALT(_AgentFalsifier):
    name: ClassVar[str] = "ALT"

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("clean", None) if outcome.passed else ("suspect", "D6")


class ATK(_AgentFalsifier):
    name: ClassVar[str] = "ATK"

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("suspect", "D8") if outcome.passed else ("clean", None)


# ---------------------------------------------------------------------------------------------------------- LLM runners

@dataclass(frozen=True)
class AgentBudget:
    max_calls: int = 24  # tool calls (the call budget)
    max_turns: int = 30  # model turns (calls plus refused or malformed turns)
    obs_chars: int = 2500


def llm_agent(client: RoleClient, role: Role = ALT_SOLVER, *, budget: AgentBudget | None = None, strategies: Sequence[str] = ATTACK_STRATEGIES,
              tag: str = "") -> AgentRunner:
    """An ``AgentRunner`` that drives ``client`` through the JSON tool protocol of ``role`` (``ALT_SOLVER`` or
    ``ATTACKER``). The budget ends the attempt; the answer is whatever the model finalised (None when it ran out)."""
    b = budget or AgentBudget()
    attacker = role.name == ATTACKER.name

    async def runner(view: AgentView, world: SynthWorld) -> Any:
        tools = tool_catalogue(view.tools)
        names = {t.name for t in view.tools}
        history: list[dict[str, Any]] = []
        calls, nudged = 0, False
        for _ in range(b.max_turns):
            inputs: dict[str, Any] = {"instruction": view.instruction, "tools": tools,
                                      "history": history or "(no calls yet: start with the tools you need)"}
            if attacker:
                inputs["strategies"] = list(strategies)
            doc = await client.complete_json(role.render(**inputs), schema=AgentTurn, max_tokens=role.max_tokens, temperature=role.temperature,
                                             tag=tag or role.name)
            turn = AgentTurn.model_validate(doc)
            if turn.is_final():
                if calls == 0 and not nudged:  # finishing before any call reads nothing and changes nothing: say so once
                    nudged = True
                    history.append({"note": "You have not called any tool yet: nothing has been read or changed. Work through the tools first."})
                    continue
                return turn.final
            if calls >= b.max_calls:
                return None
            call = turn.call or {}
            tool, args = call.get("tool"), call.get("args") or {}
            if not isinstance(tool, str) or tool not in names:
                history.append({"call": call, "error": f"unknown tool {tool!r}: use one of the listed names"})
                continue
            if not isinstance(args, dict):
                history.append({"call": call, "error": "args must be an object"})
                continue
            obs = await world.call(tool, args)
            calls += 1
            history.append({"call": {"tool": tool, "args": args}, **({"error": obs.error[:600]} if obs.error else {"result": compact(obs.value, b.obs_chars)})})
        return None

    return runner


def llm_alt(name: str, client: RoleClient, **kw: Any) -> AgentSpec:
    return AgentSpec(name=name, family=client.family, runner=llm_agent(client, ALT_SOLVER, **kw))


def llm_atk(name: str, client: RoleClient, **kw: Any) -> AgentSpec:
    return AgentSpec(name=name, family=client.family, runner=llm_agent(client, ATTACKER, **kw))


# ---------------------------------------------------------------------------------------------------------- the panel

@dataclass
class PanelMember:
    name: str
    client: RoleClient

    @property
    def family(self) -> str:
        return self.client.family


class Vote(CrucibleModel):
    member: str
    family: str
    valid: bool | None = None
    confidence: float = 0.0
    why: str = ""
    error: str | None = None


class PanelVerdict(CrucibleModel):
    decision: str  # valid | invalid | abstain
    votes: list[Vote] = Field(default_factory=list)
    lam: float = 0.0


class Attempt(CrucibleModel):
    """What the panel judges: one recorded attempt at a task (never which falsifier made it)."""

    trace: list[dict[str, Any]] = Field(default_factory=list)
    answer: Any = None
    reasons: list[str] = Field(default_factory=list)


def attempt_of(f: Finding) -> Attempt:
    d = f.outcome.data
    trace = d.get("trace")
    if trace is None and isinstance(d.get("program"), dict):
        trace = [{"tool": s["tool"], "args": {k: a.get("v") for k, a in s.get("args", {}).items()}} for s in d["program"].get("steps", [])]
    return Attempt(trace=list(trace or []), answer=d.get("answer"), reasons=list(f.outcome.reasons))


async def attempt_for(probe: Probe, outcome: Outcome, task: Task, ctx: FalsifyContext | None) -> Attempt:
    """The attempt behind a finding: an agent's recorded trace, or (a program probe: a mutant, a null agent) its program
    executed again for the calls and observations it made."""
    d = outcome.data
    if d.get("trace") is not None or probe.program is None or ctx is None:
        return Attempt(trace=list(d.get("trace") or []), answer=d.get("answer"), reasons=list(outcome.reasons))
    oracle = await ctx.oracle(task)
    r = await ctx.run(task, probe.program, instruction=probe.instruction, perturb=probe.perturb, anchors=oracle.anchors if probe.reresolve else None)
    return Attempt(trace=trace_view(r.calls), answer=r.answer, reasons=list(outcome.reasons))


def resolve(grader_passed: bool | None, decision: str, defect: str | None) -> tuple[Label, str | None]:
    """A decided suspect: confirmed when the panel and the grader disagree (a valid attempt failed: D6; an invalid one
    passed: the falsifier's own defect), else clean."""
    grader = bool(grader_passed)
    if (decision == "valid") != grader:
        return "confirmed", ("D6" if not grader else defect)
    return "clean", None


@dataclass
class Panel:
    """The adjudication panel (module docstring). ``queue`` is the human-queue JSONL file; ``show_reasons`` shows the
    grader's reason codes (part2-paper-core §5) — off, the panel judges blind to any verdict (evidence doc §5.7 rule 3)."""

    members: list[PanelMember]
    lam: float = 0.7
    queue: Path | None = None
    show_reasons: bool = True
    tag: str = "adjudicate"
    log: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if len({m.family for m in self.members}) < PANEL.min_families:
            raise ValueError(f"the panel needs >= {PANEL.min_families} families, has {sorted({m.family for m in self.members})}")

    async def judge(self, task: Task, attempt: Attempt, *, tag: str | None = None) -> PanelVerdict:
        inputs = {"task": task.instruction, "intent": task.intent.model_dump(mode="json"), "trace": attempt.trace, "answer": attempt.answer}
        if self.show_reasons:
            inputs["reason_codes"] = attempt.reasons
        msgs = PANEL.render(**inputs)

        async def one(m: PanelMember) -> Vote:
            try:
                doc = await m.client.complete_json(msgs, schema=PanelOut, max_tokens=PANEL.max_tokens, temperature=PANEL.temperature, tag=tag or self.tag)
                o = PanelOut.model_validate(doc)
                return Vote(member=m.name, family=m.family, valid=o.valid, confidence=o.confidence, why=o.why[:400])
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001 - a member that cannot answer is an abstention, never a vote
                return Vote(member=m.name, family=m.family, error=f"{type(e).__name__}: {str(e)[:200]}")

        votes = list(await asyncio.gather(*(one(m) for m in self.members)))
        vals = {v.valid for v in votes}
        if None in vals or len(vals) != 1 or min(v.confidence for v in votes) < self.lam:
            return PanelVerdict(decision="abstain", votes=votes, lam=self.lam)
        return PanelVerdict(decision="valid" if vals.pop() else "invalid", votes=votes, lam=self.lam)

    async def decide(self, task: Task, probe: Probe, outcome: Outcome, ctx: FalsifyContext | None = None) -> PanelVerdict:
        """Judge one suspect, log it, and queue an abstention for a human."""
        att = await attempt_for(probe, outcome, task, ctx)
        v = await self.judge(task, att)
        row = {"task": task.id, "version": task.version, "cls": probe.cls, "op": probe.op, "site": probe.site, "grader_passed": outcome.passed,
               "decision": v.decision, "votes": [x.model_dump(mode="json") for x in v.votes], "lam": v.lam}
        self.log.append(row)
        if v.decision == "abstain" and self.queue is not None:
            self.queue.parent.mkdir(parents=True, exist_ok=True)
            with self.queue.open("a") as fh:
                fh.write(json.dumps({**row, "attempt": att.model_dump(mode="json"), "instruction": task.instruction}, default=str) + "\n")
        return v

    async def __call__(self, task: Task, suspects: list[Finding]) -> list[Finding]:
        """The ``cegis.Adjudicator``: decided suspects are relabelled in place; the confirmed ones are returned."""
        confirmed: list[Finding] = []
        for f in suspects:
            v = await self.decide(task, f.probe, f.outcome)
            if v.decision == "abstain":
                continue
            f.label, f.defect = resolve(f.outcome.passed, v.decision, f.defect)
            if f.label == "confirmed":
                confirmed.append(f)
        return confirmed


class Adjudicated(BaseFalsifier):
    """A falsifier whose suspects the panel decides as they are found (inside ``execute``), so a suspect never stays
    open because a later re-run of the class skipped adjudication; an abstention stays a suspect (human queue). The
    wrapped class keeps its name, probes, replay and archive identity."""

    def __init__(self, inner: BaseFalsifier, panel: Panel):
        self.inner, self.panel = inner, panel
        self.name = inner.name  # type: ignore[misc]
        self._decided: dict[tuple[str, str], str] = {}  # (task digest, effect signature) -> the panel's decision

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        f = getattr(self.inner, "applies", None)
        return await f(task, ctx) if f is not None else None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        return await self.inner.propose(task, ctx)

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        o = await self.inner.execute(probe, task, ctx)
        if o.valid and self.inner.label(probe, o)[0] == "suspect":
            eff = o.data.get("effect")
            k = (task.digest(), str(eff)) if eff is not None else None
            if k is not None and k in self._decided:  # one effect is one trial (``finalize``): the panel judges it once
                o.data["panel"] = self._decided[k]
                o.data["panel_reused"] = True
                return o
            v = await self.panel.decide(task, probe, o, ctx)
            o.data["panel"] = v.decision
            if k is not None:
                self._decided[k] = v.decision
        return o

    def finalize(self, findings: list[Finding]) -> list[Finding]:
        """The wrapped class's own finalisation (MUT keeps one trial per normalised effect). Without it a wrapped MUT
        counted every duplicate mutant as a trial and its certificate bound was too tight (found in the M4 runs:
        finance_ap t02 reported n_MUT = 66 for 20 distinct effects)."""
        fin = getattr(self.inner, "finalize", None)
        return fin(findings) if fin is not None else findings

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        lab, d = self.inner.label(probe, outcome)
        dec = outcome.data.get("panel")
        if lab == "suspect" and dec in ("valid", "invalid"):
            return resolve(outcome.passed, dec, d)
        return lab, d

    async def replay(self, cx: Any, task: Task, ctx: FalsifyContext) -> bool:
        return await self.inner.replay(cx, task, ctx)


class Anchor(CrucibleModel):
    """A calibration item with executed truth: an oracle trace (valid), a no-op or a killed mutant (invalid)."""

    task: Task
    attempt: Attempt
    valid: bool
    kind: str = ""


class CalibrationReport(CrucibleModel):
    lam: float
    target: float
    calibrated: bool
    rows: list[dict[str, Any]] = Field(default_factory=list)  # per λ: decided, correct, accuracy, coverage, upper error bound
    votes: list[dict[str, Any]] = Field(default_factory=list)


async def calibrate(panel: Panel, anchors: Sequence[Anchor], *, target: float = 0.9, grid: Sequence[float] = (0.5, 0.6, 0.7, 0.8, 0.9, 0.95),
                    floor: float = 0.0) -> CalibrationReport:
    """Choose λ (trust or escalate): the smallest grid value whose decided anchors are at least ``target`` accurate;
    none reaches it → the largest, flagged uncalibrated; never below ``floor`` (few anchors bound the error loosely). Sets ``panel.lam``."""
    saved = panel.lam
    panel.lam = 0.0
    try:
        verdicts = [await panel.judge(a.task, a.attempt, tag="calibrate") for a in anchors]
    finally:
        panel.lam = saved
    rows = []
    chosen = None
    for lam in grid:
        dec = cor = 0
        for a, v in zip(anchors, verdicts, strict=True):
            vals = {x.valid for x in v.votes}
            if None in vals or len(vals) != 1 or min(x.confidence for x in v.votes) < lam:
                continue
            dec += 1
            cor += int(vals.pop() == a.valid)
        acc = cor / dec if dec else 0.0
        rows.append({"lam": lam, "decided": dec, "correct": cor, "accuracy": round(acc, 4), "coverage": round(dec / max(1, len(anchors)), 4),
                     "error_upper95": round(clopper_pearson_upper(dec - cor, dec, 0.05), 4) if dec else 1.0})
        if chosen is None and dec and acc >= target:
            chosen = lam
    panel.lam = max(floor, chosen if chosen is not None else max(grid))
    votes = [{"kind": a.kind, "truth": a.valid, "votes": [x.model_dump(mode="json") for x in v.votes]} for a, v in zip(anchors, verdicts, strict=True)]
    return CalibrationReport(lam=panel.lam, target=target, calibrated=chosen is not None, rows=rows, votes=votes)
