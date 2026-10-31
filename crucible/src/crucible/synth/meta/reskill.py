"""RESKILL: re-label a valid task with the skill it actually exercises (lead, 2026-10-30; ``rsi.TaskLoop``).

A task can be good and still be mislabelled: too hard because it exercises a more advanced skill than its label says, or
labelled "derive X" when the instruction states X (lookup-and-apply). RESKILL re-labels the intent card from the task's
own evidence, and the task is kept only if it is a good task under the new label.

**The derivation** (``INTENT_DERIVER``, a role of a non-builder family; ``swarm_deriver``) reads the oracle program, its
taint map (where every argument value came from: the instruction, an observation, the tool catalogue, a given) and the
executed behaviour (the effects and the tools used), plus the domain's skill list. It names the skill the task requires
and the given and derive values. It never sees the student, a rollout, a pass rate or a band verdict (``never``), and it
does not see the current label either, so it cannot anchor on it. ``derivation_evidence`` builds its input.

**Checks on the proposal** (``check_proposal``): the skill must be one of the domain's ``DomainSpec`` skills (no
invented skill); every derive value must be something π* writes or answers; a value the instruction states cannot be a
derive value (stating it makes the skill lookup-and-apply, and the label must say so).
"""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import cjson
from crucible.synth.falsify.base import FalsifyContext, delta_of
from crucible.synth.program import leaves, text_spans
from crucible.synth.roles import BUILDERS, Role, ask, tool_catalogue
from crucible.synth.task import IntentCard, Task


class IntentOut(BaseModel):
    model_config = ConfigDict(extra="ignore")
    skill: str
    given: list[Any] = Field(default_factory=list)
    derive: list[Any] = Field(default_factory=list)
    why: str = ""


class IntentProposal(CrucibleModel):
    skill: str
    given: list[Any] = Field(default_factory=list)
    derive: list[Any] = Field(default_factory=list)
    why: str = ""
    by: str = ""  # the deriver's family

    def card(self, notes: str = "") -> IntentCard:
        return IntentCard(skill=self.skill, given=list(self.given), derive=list(self.derive), notes=notes)


INTENT_DERIVER = Role(
    name="intent_deriver", stage="rsi", output=IntentOut, temperature=0.0, max_tokens=2048,
    sees=("skills", "instruction", "tools", "oracle_program", "taint", "effects", "answer", "written_values"),
    never=("student", "rollouts", "pass_rate", "band", "current_skill", "contract", "gate_code"),
    differs_from=BUILDERS, independent=True,
    system=("You label what a task really asks of an agent. You see the domain's skills (id and description), the task's "
            "instruction, its tools, a reference solution as a program, where every value the solution uses came from "
            "(the instruction, an observation it made, the tool catalogue, a given), and what the solution changed. Choose the ONE "
            "skill id from the list that the solution actually needs, judging only from this evidence. List as derive the values "
            "the agent must work out itself (values it writes or answers that the instruction does not state, copied exactly from "
            "written_values: values such as a status or a reason code, never a field name or an id), and as given the "
            "values handed to it outside the instruction. A value the instruction states is never a derive value: if the task only "
            "looks up and applies what it is told, pick the skill that says so. "
            'Reply {"skill": "<id>", "given": [...], "derive": [...], "why": "<one sentence>"}.'))


async def derivation_evidence(task: Task, ctx: FalsifyContext) -> dict[str, Any]:
    """The deriver's input: the oracle program, its taint map, the executed behaviour (no student information)."""
    r = await ctx.oracle(task)
    tools = await (await ctx.world(task)).tools()
    taint = [{"value": e.value, "at": e.loc, "from": sorted({s.kind for s in e.sources}) or ["nowhere"]} for e in r.taint][:60]
    effects = [{"change": str(ch.addr), "kind": ch.kind, "new": ch.new} for ch in delta_of(r, task.contract)][:60]
    return {"instruction": task.instruction, "tools": tool_catalogue(tools), "oracle_program": task.oracle.model_dump(mode="json"), "taint": taint,
            "effects": effects, "answer": r.answer}


def written_values(evidence: dict[str, Any]) -> set[str]:
    out = {cjson(e["new"]) for e in evidence.get("effects", []) if e.get("new") is not None}
    for _p, v in leaves(evidence.get("answer")):
        out.add(cjson(v))
    if evidence.get("answer") is not None:
        out.add(cjson(evidence["answer"]))
    return out


def check_proposal(p: IntentProposal | IntentOut, skills: Sequence[str], evidence: dict[str, Any]) -> list[str]:
    """RS1 (an existing skill) and the derive rules (module docstring); [] when the proposal stands."""
    why = []
    if p.skill not in set(skills):
        why.append(f"RS1: skill {p.skill!r} is not a skill of the domain ({sorted(skills)[:8]})")
    wrote = written_values(evidence)
    ins = str(evidence.get("instruction", ""))
    for v in p.derive:
        if cjson(v) not in wrote:
            why.append(f"RS2: derive value {v!r} is nothing the solution writes or answers")
        elif text_spans(v, ins):
            why.append(f"RS2: derive value {v!r} is stated in the instruction (lookup-and-apply, not derivation)")
    return why


Deriver = Callable[[Task, FalsifyContext], Awaitable[IntentProposal | None]]


def swarm_deriver(client: Any, skills_of: Callable[[Task], Sequence[dict[str, str]]], *, tag: str = "reskill") -> Deriver:
    """``INTENT_DERIVER`` on one ``roles.RoleClient`` (a non-builder family). ``skills_of(task)`` → the domain's skills
    as ``{"id", "description"}``; an answer that fails ``check_proposal`` goes back to the model as feedback once."""

    async def derive(task: Task, ctx: FalsifyContext) -> IntentProposal | None:
        ev = await derivation_evidence(task, ctx)
        skills = list(skills_of(task))
        ids = [s["id"] for s in skills]

        def check(o: IntentOut) -> None:
            bad = check_proposal(o, ids, ev)
            if bad:
                raise ValueError("; ".join(bad)[:400])

        wv = sorted({json.dumps(e["new"]) for e in ev.get("effects", []) if e.get("new") is not None})[:30]  # taskloop-1 §2: field names were proposed
        out = await ask(client, INTENT_DERIVER, tag=f"{tag}:{task.id}@{task.version}", check=check, retries=2, skills=skills,
                        written_values=[json.loads(x) for x in wv], **ev)
        return IntentProposal(skill=out.skill, given=list(out.given), derive=list(out.derive), why=out.why[:300], by=getattr(client, "family", ""))

    return derive


def domain_skills(spec: Any) -> list[dict[str, str]]:
    """``{"id", "description"}`` of a ``domain.DomainSpec``'s skills (the only labels RESKILL may use)."""
    return [{"id": s.id, "description": s.description} for s in getattr(spec, "skills", [])]


# ---------------------------------------------------------------------------------------------------------- the default checks


def p3_checker(ctx: FalsifyContext, *, pool: Sequence[Task] = (), agents: Sequence[Any] = (),
               evidence: bool = False) -> Callable[[Task], Awaitable[tuple[bool | None, list[str]]]]:
    """RS3's P3 under the new card: ``intent.p3`` (EVID, SWAP, TAINT, TWIN). ``evidence``: EVID through the SQL evidence
    channel (``intent.EvidenceContext``; DESIGN §7.7 item 25)."""
    from crucible.synth.intent import p3

    async def check(task: Task) -> tuple[bool | None, list[str]]:
        r = await p3(task, ctx, pool=pool, agents=agents, evidence=evidence)
        why = list(r.reasons)
        if r.on_intent is None and r.not_applicable:
            why += [f"{k} n/a: {v}" for k, v in sorted(r.not_applicable.items())]
        return r.on_intent, why

    return check


def leak_checker(ctx: FalsifyContext, *, extra: Sequence[Any] = ()) -> Callable[[Task], Awaitable[list[str]]]:
    """RS3's leak check with the new derive set: TAINT's layered leak matching (``falsify.leak``) reads the intent
    card's derive values, plus any further leak class (``extra``: e.g. the L2 ``LeakProbe``). Every confirmed D7."""
    from crucible.synth.falsify import TAINT
    from crucible.synth.falsify.base import run_classes

    async def check(task: Task) -> list[str]:
        res = await run_classes(task, [TAINT(), *extra], ctx)
        return [f"{f.cls}:{f.op}:{f.site}" for r in res if r.applicable for f in r.confirmed if f.defect == "D7"]

    return check


def novelty_checker(ctx: FalsifyContext) -> Callable[[Task, str, Sequence[Task]], Awaitable[tuple[bool, list[str]]]]:
    """RS4's novelty in the new cell: ``novelty.novel`` of the re-labelled task against the pool tasks of that cell."""
    from crucible.synth.novelty import novel, signature

    async def check(task: Task, cell: str, pool: Sequence[Task]) -> tuple[bool, list[str]]:
        sig = signature(task, await ctx.oracle(task), cell=cell)
        sigs = [signature(t, await ctx.oracle(t), cell=cell) for t in pool]
        n = novel(sig, sigs)
        return n.novel, list(n.reasons)

    return check
