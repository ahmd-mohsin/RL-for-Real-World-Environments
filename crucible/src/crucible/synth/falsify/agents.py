"""ALT and ATK: agent falsifiers (part2-paper-core §5). M2 ships the protocol and a runner-injected implementation; the
LLM wiring (families, prompts, budgets, adjudication) is M4.

An ``AgentRunner`` is ``async (view, world) -> answer``: it sees only the instruction and the tool catalogue
(``AgentView``; never the contract), acts through ``world.call`` and returns its final answer. The World it gets records
every call (``RecordingWorld``), so each attempt becomes an ordinary ``Run`` for the trusted grader, and its trace becomes
a literal ``Program`` (``trace_program``) that the archive can replay like any mutant.

- **ALT** (alternative solvers from families that built nothing in the task): a pass is a clean D6 trial; a failure is a
  **suspect D6** (adjudicated: was it a valid solution the contract rejects?).
- **ATK** (attackers told to pass without doing the skill): a pass is a **suspect D8** (the panel decides D7/D8); a
  failure is clean.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import diff
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe
from crucible.synth.iface import Obs, SynthWorld, ToolInfo
from crucible.synth.program import CallRecord, Lit, Program, Run, Step
from crucible.synth.task import Task


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
    except Exception as e:  # noqa: BLE001 - a crashing agent is a failed attempt, graded on the state it reached
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
        v = ctx.grade(r, task.contract)
        return Outcome(passed=v.passed, reasons=v.reasons, error=r.error,
                       data={"program": trace_program(r.calls, r.answer).model_dump(mode="json"), "family": agent.family})


class ALT(_AgentFalsifier):
    name: ClassVar[str] = "ALT"

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("clean", None) if outcome.passed else ("suspect", "D6")


class ATK(_AgentFalsifier):
    name: ClassVar[str] = "ATK"

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        return ("suspect", "D8") if outcome.passed else ("clean", None)
