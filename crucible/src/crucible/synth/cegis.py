"""The per-task CEGIS loop (part2-paper-core §7) with injected roles.

    synthesize(cell, domain):
      task ← propose(cell, domain)                       # task proposer
      π*   ← build_oracle(task)                          # oracle programmer; executed, dead → drop
      C    ← generalise(task, skeleton(run(π*)), run)    # contract writer (relaxes the exact skeleton)
      refine(task)

    refine(task):
      for round in 1..max_rounds:
          findings ← cheap falsifiers (REP first, gating MUT/TWIN/PERT; RED, TAINT, NULL); reopened ← archive regressions
          X_new    ← confirmed ∪ adjudicate(suspects)
          if X_new = ∅ and nothing reopened: findings += paid falsifiers (ALT, ATK); X_new ← …
          if X_new = ∅ and nothing reopened: break
          archive.add(X_new)
          candidate ← repair(task, X_new ∪ reopened)     # sees counterexamples and reason codes only
          if candidate re-opens a counterexample the current version closes: refuse it (the monotone archive)
          else task ← candidate
      if the last round's repair was accepted: re-run every class on that version and ADJUDICATE its suspects (archived)
      certify(task) → certified | provisional (c = 0, n too small for the U bound) | reject(reasons)

The roles are ``Protocol``s (M4 binds them to model families through ``factory/synth_wire.py``; tests pass scripted
ones). A repair that only partly fixes the round's counterexamples is accepted: what it leaves open is found again next
round. What it may never do is re-open a counterexample the current version already rejects. The round cap is
``max_rounds`` (3). The report records every round: probe counts, the counterexamples, the repair decision.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.archive import Archive, regress
from crucible.synth.certificate import AdmissionPolicy, Certificate, certify
from crucible.synth.contract import Contract
from crucible.synth.falsify import cheap
from crucible.synth.falsify.base import BaseFalsifier, ClassResult, Counterexample, FalsifyContext, Finding, run_classes
from crucible.synth.infer import skeleton
from crucible.synth.program import Program, Run
from crucible.synth.task import Task


class Proposer(Protocol):
    async def __call__(self, cell: Any, domain: str) -> Task | None: ...


class OracleBuilder(Protocol):
    async def __call__(self, task: Task) -> Program | None: ...


class Generaliser(Protocol):
    async def __call__(self, task: Task, skeleton: Contract, run: Run) -> Contract: ...


class Repairer(Protocol):
    async def __call__(self, task: Task, counterexamples: list[Counterexample]) -> Task | None: ...


class Adjudicator(Protocol):
    async def __call__(self, task: Task, suspects: list[Finding]) -> list[Finding]: ...


class RoundReport(CrucibleModel):
    round: int
    task_version: int
    probes: dict[str, int] = Field(default_factory=dict)
    confirmed: dict[str, int] = Field(default_factory=dict)
    counterexamples: list[str] = Field(default_factory=list)
    archived: list[str] = Field(default_factory=list)
    reopened: list[str] = Field(default_factory=list)
    suspects: int = 0
    paid: bool = False
    repair: Literal["none", "accepted", "refused", "declined"] = "none"
    refused_for: list[str] = Field(default_factory=list)


class FinalPass(CrucibleModel):
    """The certification pass on the final version: ``rerun`` when that version had not been falsified yet (the last
    round's repair was accepted). Its suspects are adjudicated there (``adjudicated`` were decided, the rest abstained
    and stay open) and the confirmed ones are archived."""

    rerun: bool = False
    suspects: int = 0
    adjudicated: int = 0
    confirmed: list[str] = Field(default_factory=list)
    archived: list[str] = Field(default_factory=list)


class CegisReport(CrucibleModel):
    task_id: str | None = None
    domain: str
    status: Literal["admitted", "provisional", "rejected", "dropped"]  # admitted = certified
    reasons: list[str] = Field(default_factory=list)
    rounds: list[RoundReport] = Field(default_factory=list)
    final: FinalPass = Field(default_factory=FinalPass)
    certificate: Certificate | None = None
    task: Task | None = None


async def _round(task: Task, falsifiers: Sequence[BaseFalsifier], ctx: FalsifyContext) -> list[ClassResult]:
    return await run_classes(task, falsifiers, ctx)  # REP first; it gates MUT, TWIN and PERT (§5.1 rule 3)


async def _counterexamples(task: Task, results: Sequence[ClassResult], adjudicate: Adjudicator | None) -> tuple[list[Counterexample], int]:
    confirmed = [f for r in results for f in r.confirmed]
    suspects = [f for r in results for f in r.suspects]
    adj = list(await adjudicate(task, suspects)) if adjudicate is not None and suspects else []
    out, seen = [], set()
    for f in [*confirmed, *adj]:
        cx = Counterexample.of(task, f)
        if cx.id not in seen:
            seen.add(cx.id)
            out.append(cx)
    return out, len(suspects)


async def refine(task: Task, *, repair: Repairer, ctx: FalsifyContext, archive: Archive, falsifiers: Sequence[BaseFalsifier] | None = None,
                 paid: Sequence[BaseFalsifier] = (), adjudicate: Adjudicator | None = None, policy: AdmissionPolicy | None = None,
                 max_rounds: int = 3) -> CegisReport:
    cheap_f = list(falsifiers) if falsifiers is not None else cheap()
    every = [*cheap_f, *paid]
    rounds: list[RoundReport] = []
    last: tuple[int, list[ClassResult]] | None = None
    for rnd in range(1, max_rounds + 1):
        results = await _round(task, cheap_f, ctx)
        reg = await regress(archive, task, ctx, every)
        xs, n_susp = await _counterexamples(task, results, adjudicate)
        rr = RoundReport(round=rnd, task_version=task.version, reopened=list(reg.reopened))
        if not xs and not reg.reopened and paid:
            presults = await _round(task, paid, ctx)
            results += presults
            xs, n_susp = await _counterexamples(task, results, adjudicate)
            rr.paid = True
        rr.probes = {r.cls: r.n for r in results}
        rr.confirmed = {r.cls: r.c for r in results if r.c}
        rr.suspects = n_susp
        rr.counterexamples = [c.id for c in xs]
        last = (task.version, results)
        rounds.append(rr)
        if not xs and not reg.reopened:
            break
        rr.archived = archive.add_all(xs)
        reopened_cx = [c for c in archive.records(task.domain, task.id) if c.id in set(reg.reopened)]
        candidate = await repair(task, [*xs, *reopened_cx])
        if candidate is None:
            rr.repair = "declined"
            break
        creg = await regress(archive, candidate, ctx, every)
        open_now = {c.id for c in xs} | set(reg.reopened)
        broken = sorted(set(creg.reopened) - open_now)
        if broken:
            rr.repair, rr.refused_for = "refused", broken
            continue
        rr.repair = "accepted"
        task = candidate
    final = FinalPass()
    if last is None or last[0] != task.version:
        # the version being certified was never falsified (the last round's repair was accepted): re-run every class on
        # it and decide its suspects here, or a suspect that a last-round repair introduced would stay open
        results = await _round(task, cheap_f, ctx)
        if paid:
            results += await _round(task, paid, ctx)
        last = (task.version, results)
        final.rerun = True
        suspects = [f for r in results for f in r.suspects]
        final.suspects = len(suspects)
        if adjudicate is not None and suspects:
            decided = list(await adjudicate(task, suspects))  # relabels in place, so the certificate counts the decisions
            final.adjudicated = len(suspects) - sum(len(r.suspects) for r in results)
            cxs = [Counterexample.of(task, f) for f in decided if f.label == "confirmed"]
            final.confirmed = [c.id for c in cxs]
            final.archived = archive.add_all(cxs)
    reg = await regress(archive, task, ctx, every)
    cert = certify(task, last[1], policy, reopened=reg.reopened)
    status = {"certified": "admitted", "provisional": "provisional"}.get(cert.status, "rejected")
    return CegisReport(task_id=task.id, domain=task.domain, status=status, reasons=list(cert.reasons) + list(cert.provisional_for), rounds=rounds,
                       final=final, certificate=cert, task=task)


async def synthesize(cell: Any, domain: str, *, propose: Proposer, build_oracle: OracleBuilder, generalise: Generaliser, repair: Repairer,
                     ctx: FalsifyContext, archive: Archive, falsifiers: Sequence[BaseFalsifier] | None = None, paid: Sequence[BaseFalsifier] = (),
                     adjudicate: Adjudicator | None = None, policy: AdmissionPolicy | None = None, max_rounds: int = 3) -> CegisReport:
    task = await propose(cell, domain)
    if task is None:
        return CegisReport(domain=domain, status="dropped", reasons=["propose:none"])
    prog = await build_oracle(task)
    if prog is None or prog.problems():
        return CegisReport(task_id=task.id, domain=domain, status="dropped", reasons=["oracle:none" if prog is None else "oracle:invalid"])
    task = task.model_copy(update={"oracle": prog}, deep=True)
    r = await ctx.run(task, prog, taint=True)
    if r.error is not None or any(c.obs.error is not None for c in r.calls):
        return CegisReport(task_id=task.id, domain=domain, status="dropped", reasons=["oracle:dead"])
    sk = skeleton(r, task.contract.normalizers)
    c = await generalise(task, sk, r)
    task = task.model_copy(update={"contract": c}, deep=True)
    return await refine(task, repair=repair, ctx=ctx, archive=archive, falsifiers=falsifiers, paid=paid, adjudicate=adjudicate, policy=policy,
                        max_rounds=max_rounds)
