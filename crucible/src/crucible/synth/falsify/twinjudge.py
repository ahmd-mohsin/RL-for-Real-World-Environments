"""TWIN suspects decided inline by the calibrated panel (DESIGN §7.7 item 18, compose-2).

compose-1 left TWIN suspects open: the pipeline adjudicated MUT inline only, so a TWIN suspect blocked admission until a
person looked. A TWIN suspect is a twin whose verdict TWIN cannot derive (``falsify.twin``): σ cannot carry the literal
into C (no write sent it, C′ = C), or π*′ wandered onto entities neither π* nor σ touches. Both are one question about
the **twin task** τ′ = (ι′, the intent card under σ), which the panel can answer from a trace:

- **π* passes C′** (the contract did not move with the literal): is π*, the original program, a valid solution of ι′?
  *Invalid* → the contract accepts a run that ignores what the instruction names: confirmed **D9**. *Valid* → the
  literal steers nothing the contract must pin (a read's filter with the same result): clean.
- **π*′ fails C′** (the diverged case): is π*′ a valid solution of ι′? *Valid* → C′ rejects a valid twin solution, so the
  contract pins something the instruction does not say: confirmed **D9**. *Invalid* → π*′ is no twin solution: clean.

``TwinAdjudicated`` wraps TWIN (its name, probes, replay and archive identity are TWIN's) and asks ``Panel.judge`` with
the twin task as the task; the decision and the grader's verdict on that attempt ride on the outcome, so the label is
recomputed from them. An abstention stays a suspect and goes to the human queue.

**Strict calibration (item 20's ``calibrate``).** ``twin_anchors`` builds anchors with executed truth from *clean*
twins (π* fails C′, π*′ passes it) whose π*′ writes other entities than π*: on τ′, π*′ is valid, π* (it acts on the
entities ι no longer names) is invalid, and the do-nothing attempt is invalid. (First pilot: requiring σ to re-key a
written entity found no anchor on four it_helpdesk links, where a ticket subject two tickets share is a selector.) ``calibrate_twin_panel`` runs ``calibrate`` with ``min_decided``,
``require_both`` and ``refuse_uncalibrated``: with too few decided anchors, or anchors of one truth only, the TWIN panel
decides nothing (λ above 1) and every suspect stays open.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from crucible.synth.falsify.agents import Anchor, Attempt, CalibrationReport, Panel, calibrate, trace_view
from crucible.synth.falsify.base import BaseFalsifier, FalsifyContext, Label, Outcome, Probe, delta_of, written_entities
from crucible.synth.falsify.twin import TWIN, substitute
from crucible.synth.task import Task


def twin_task(task: Task, probe: Probe) -> Task:
    """τ′: the task with the twin's instruction and its intent card under σ (what the panel judges against)."""
    sigma = [tuple(x) for x in probe.data.get("sigma") or []]
    intent = task.intent.model_copy(update={"given": substitute(list(task.intent.given), sigma), "derive": substitute(list(task.intent.derive), sigma)})
    return task.model_copy(update={"id": f"{task.id}~twin", "instruction": probe.instruction or task.instruction, "intent": intent}, deep=True)


async def twin_attempt(probe: Probe, outcome: Outcome, task: Task, ctx: FalsifyContext) -> tuple[Attempt, bool]:
    """The attempt to judge and the grader's verdict on it (module docstring): π* when it passes C′, else π*′."""
    d = outcome.data
    if d.get("oracle_passes_twin"):
        r = await ctx.oracle(task)
        return Attempt(trace=trace_view(r.calls), answer=r.answer, reasons=list(d.get("g1") or [])), True
    assert probe.program is not None
    r = await ctx.run(task, probe.program, instruction=probe.instruction)
    return Attempt(trace=trace_view(r.calls), answer=r.answer, reasons=list(d.get("g2") or [])), bool(d.get("twin_passes_twin"))


class TwinAdjudicated(BaseFalsifier):
    """TWIN with its suspects decided by ``panel`` on the twin task (module docstring)."""

    def __init__(self, inner: BaseFalsifier, panel: Panel):
        self.inner, self.panel = inner, panel
        self.name = inner.name  # type: ignore[misc]
        self._decided: dict[tuple[str, str], tuple[str, bool]] = {}

    async def applies(self, task: Task, ctx: FalsifyContext) -> str | None:
        f = getattr(self.inner, "applies", None)
        return await f(task, ctx) if f is not None else None

    async def propose(self, task: Task, ctx: FalsifyContext) -> list[Probe]:
        return await self.inner.propose(task, ctx)

    async def execute(self, probe: Probe, task: Task, ctx: FalsifyContext) -> Outcome:
        o = await self.inner.execute(probe, task, ctx)
        if not (o.valid and self.inner.label(probe, o)[0] == "suspect"):
            return o
        k = (task.digest(), probe.digest())
        if k not in self._decided:
            tt = twin_task(task, probe)
            att, graded = await twin_attempt(probe, o, task, ctx)
            v = await self.panel.judge(tt, att)
            row = {"task": task.id, "version": task.version, "cls": "TWIN", "op": probe.op, "site": probe.site, "grader_passed": graded,
                   "judged": "oracle" if o.data.get("oracle_passes_twin") else "twin", "decision": v.decision,
                   "votes": [x.model_dump(mode="json") for x in v.votes], "lam": v.lam}
            self.panel.log.append(row)
            if v.decision == "abstain" and self.panel.queue is not None:
                import json

                self.panel.queue.parent.mkdir(parents=True, exist_ok=True)
                with self.panel.queue.open("a") as fh:
                    fh.write(json.dumps({**row, "attempt": att.model_dump(mode="json"), "instruction": tt.instruction}, default=str) + "\n")
            self._decided[k] = (v.decision, graded)
        o.data["panel"], o.data["panel_grader"] = self._decided[k]
        return o

    def label(self, probe: Probe, outcome: Outcome) -> tuple[Label, str | None]:
        lab, d = self.inner.label(probe, outcome)
        dec = outcome.data.get("panel")
        if lab != "suspect" or dec not in ("valid", "invalid"):
            return lab, d
        graded = bool(outcome.data.get("panel_grader"))
        return ("confirmed", "D9") if (dec == "valid") != graded else ("clean", None)

    def finalize(self, findings: list[Any]) -> list[Any]:
        fin = getattr(self.inner, "finalize", None)
        return fin(findings) if fin is not None else findings

    async def replay(self, cx: Any, task: Task, ctx: FalsifyContext) -> bool:
        return await self.inner.replay(cx, task, ctx)


async def twin_anchors(tasks: Sequence[Task], ctx: FalsifyContext, *, per_task: int = 2) -> list[Anchor]:
    """Anchors with executed truth from clean twins whose σ re-keys an entity π* writes (module docstring)."""
    out: list[Anchor] = []
    tw = TWIN()
    for task in tasks:
        try:
            if await tw.applies(task, ctx) is not None:
                continue
            oracle = await ctx.oracle(task)
            wrote = written_entities(delta_of(oracle, task.contract))
            got = 0
            for p in await tw.propose(task, ctx):
                if got >= per_task:
                    break
                o = await tw.execute(p, task, ctx)
                if not o.valid or tw.label(p, o)[0] != "clean" or p.program is None:
                    continue
                tt = twin_task(task, p)
                r2 = await ctx.run(task, p.program, instruction=p.instruction)
                # executed truth: π*′ acts on other entities than π* (a selector re-derives the keys; a key σ maps directly)
                if not wrote or written_entities(delta_of(r2, task.contract)) == wrote:
                    continue
                # π*′ is a twin SOLUTION only when nothing is derived from the entity's data: on a decision task it applies
                # π*'s decision to the sibling, whose right decision depends on its own records (compose-2: the panel called
                # all 5 such anchors invalid, and it was right)
                if not task.intent.derive:
                    out.append(Anchor(task=tt, attempt=Attempt(trace=trace_view(r2.calls), answer=r2.answer, reasons=[]), valid=True, kind="twin_solution"))
                out.append(Anchor(task=tt, attempt=Attempt(trace=trace_view(oracle.calls), answer=oracle.answer, reasons=list(o.data.get("g1") or [])),
                                  valid=False, kind="original_on_twin"))
                out.append(Anchor(task=tt, attempt=Attempt(trace=[], answer=None, reasons=[]), valid=False, kind="noop"))
                got += 1
        except Exception:  # noqa: BLE001, S112 - a task whose twins cannot run gives no anchors
            continue
    return out


async def calibrate_twin_panel(panel: Panel, anchors: Sequence[Anchor], *, min_decided: int = 6, target: float = 0.9) -> CalibrationReport:
    """Item 20's strict calibration on twin anchors (module docstring): sets ``panel.lam`` (above 1: decides nothing)."""
    if not anchors:
        panel.lam = 1.01
        return CalibrationReport(lam=panel.lam, target=target, calibrated=False, rows=[], votes=[])
    return await calibrate(panel, anchors, target=target, min_decided=min_decided, require_both=True, refuse_uncalibrated=True)


__all__ = ["TwinAdjudicated", "calibrate_twin_panel", "twin_anchors", "twin_attempt", "twin_task"]
