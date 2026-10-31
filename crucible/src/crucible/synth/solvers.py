"""The ALT solver pool, measured at equal spend (DESIGN §7.7 item 18, compose-2).

The ALT rule stays as it is. A non-builder family must pass the *whole* task from its instruction. The solver sees the
instruction, the tool catalogue and its own history; it never sees the contract, the oracle or the intent card. What
may change is which solver runs and how it is prompted. This module measures that.

A ``SolverConfig`` is a model client (one family) and a harness:
- ``v2``: ``roles.ALT_SOLVER_V2`` with the finish nudge, as in yield-1;
- ``v3``: ``roles.ALT_SOLVER_V3``, with a plan and a scratchpad, a progress check before finishing, and a windowed
  history for long tasks.

The call budget scales with the oracle's horizon (``compose.agent_budget_for``).

``evaluate`` runs every configuration on every task and keeps two measurements apart.
- **Per attempt.** Pass rate, calls, USD.
- **At equal spend.** Each (configuration, task) gets attempts until its spend on that task reaches ``budget_usd`` (at
  least one attempt, at most ``max_attempts``), and the task counts as solved when any of those attempts passes.

A cheap model can therefore try more often within the same dollars. Every attempt has its own call tag
(``ALT:<task>#<config>~<k>``), so a wire that seeds by tag (``WireClient(seed_by_tag=True)``) makes repeated attempts
independent samples. ``cost_of(tag_prefix)`` is the caller's spend lookup (the wire's usage rows), so this module knows
no model. A pass is the trusted grader's verdict on the attempt under the task's contract (``FalsifyContext.grade``).
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.falsify.agents import AgentBudget, AgentSpec, llm_agent, run_agent
from crucible.synth.falsify.base import FalsifyContext
from crucible.synth.roles import RoleClient, variant
from crucible.synth.task import Task

MAX_MALFORMED = 4  # replies in a row that are neither a call nor a finish end a solver's attempt (pilot-10x10)


@dataclass
class SolverConfig:
    name: str
    client: RoleClient
    harness: str = "v2"  # v2 | v3
    per_call: float = 1.5
    slack: int = 6
    window: int | None = 16

    @property
    def family(self) -> str:
        return self.client.family

    def budget(self, horizon: int) -> AgentBudget:
        from crucible.synth.compose import agent_budget_for

        return agent_budget_for(horizon, per_call=self.per_call, slack=self.slack)

    def agent(self, tag: str, horizon: int, log: Callable[[dict[str, Any]], None] | None = None) -> AgentSpec:
        v3 = self.harness == "v3"
        role = variant("alt_solver", "v3" if v3 else "v2")
        return AgentSpec(name=self.name, family=self.family,
                         runner=llm_agent(self.client, role, budget=self.budget(horizon), tag=tag, finish_nudge=True, progress_check=v3,
                                          window=self.window if v3 else None, log=log, max_malformed=MAX_MALFORMED))


class AttemptRow(CrucibleModel):
    task: str
    set: str
    config: str
    family: str
    harness: str
    k: int
    horizon: int
    passed: bool = False
    reasons: list[str] = Field(default_factory=list)
    calls: int = 0
    writes: int = 0
    usd: float = 0.0
    seconds: float = 0.0
    error: str | None = None


async def attempt(task: Task, cfg: SolverConfig, ctx: FalsifyContext, *, k: int, horizon: int, set_name: str,
                  cost_of: Callable[[str], float] | None = None, log_dir: Path | None = None,
                  sink: Callable[[Any], None] | None = None) -> AttemptRow:
    """One attempt on the whole task. ``sink`` receives the agent's run (its trace), for adjudicating a failed attempt."""
    tag = f"ALT:{task.id}#{cfg.name}~{k}"
    turns: list[dict[str, Any]] = []
    t0 = time.time()
    before = cost_of(tag) if cost_of else 0.0
    r = await run_agent(cfg.agent(tag, horizon, log=turns.append), task, ctx)
    if sink is not None:
        sink(r)
    v = ctx.grade(r, task.contract)
    row = AttemptRow(task=task.id, set=set_name, config=cfg.name, family=cfg.family, harness=cfg.harness, k=k, horizon=horizon,
                     passed=bool(v.passed) and r.error is None, reasons=list(v.reasons)[:8], calls=len(r.calls),
                     writes=sum(1 for c in r.calls if not c.read_only and not c.obs.error), seconds=round(time.time() - t0, 1), error=r.error,
                     usd=round((cost_of(tag) if cost_of else 0.0) - before, 6))
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        safe = f"{task.id.replace('/', '_')}__{cfg.name}__{k}.json"
        (log_dir / safe).write_text(json.dumps({"row": row.model_dump(mode="json"), "turns": turns,
                                                "calls": [{"tool": c.tool, "args": c.args, "error": c.obs.error} for c in r.calls]},
                                               default=str)[:400_000])
    return row


async def evaluate(tasks: Sequence[tuple[str, Task, int]], configs: Sequence[SolverConfig], *, budget_usd: float, max_attempts: int = 3,
                   cost_of: Callable[[str], float] | None = None, concurrency: int = 6, out: Path | None = None,
                   stop: Callable[[], bool] | None = None) -> list[AttemptRow]:
    """Every configuration on every task (``(set name, task, oracle horizon)``), attempts until the equal per-task budget
    is spent (module docstring). Rows are appended to ``out/attempts.jsonl`` as they finish."""
    from crucible.actors.llm import BudgetExceeded

    sem = asyncio.Semaphore(concurrency)
    rows: list[AttemptRow] = []
    capped: list[str] = []

    async def one(set_name: str, task: Task, hz: int, cfg: SolverConfig) -> None:
        spent, k = 0.0, 0
        async with sem:
            ctx = FalsifyContext()
            try:
                while k < max_attempts and (k == 0 or spent < budget_usd):
                    if capped or (stop is not None and stop()):
                        return
                    k += 1
                    try:
                        row = await attempt(task, cfg, ctx, k=k, horizon=hz, set_name=set_name, cost_of=cost_of,
                                            log_dir=out / "attempts" if out is not None else None)
                    except BudgetExceeded as e:  # the run's cap: stop every configuration, keep what finished (a cut attempt is no trial)
                        capped.append(str(e)[:200])
                        return
                    rows.append(row)
                    spent += row.usd
                    if out is not None:
                        with (out / "attempts.jsonl").open("a") as fh:
                            fh.write(row.model_dump_json() + "\n")
                    if row.passed:
                        return
            finally:
                await ctx.close()

    await asyncio.gather(*(one(s, t, h, c) for s, t, h in tasks for c in configs))
    return rows


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return max(0.0, (c - h) / d), min(1.0, (c + h) / d)


def summarize(rows: Sequence[AttemptRow], budget_usd: float) -> list[dict[str, Any]]:
    """Per (set, configuration): first-attempt pass rate, every-attempt pass rate, solved within the equal per-task
    budget (Wilson intervals), USD per attempt and per solved task."""
    out = []
    keys = sorted({(r.set, r.config) for r in rows})
    for s, cfgname in keys:
        rs = [r for r in rows if r.set == s and r.config == cfgname]
        tasks = sorted({r.task for r in rs})
        first = [next(r for r in rs if r.task == t and r.k == 1) for t in tasks if any(r.task == t and r.k == 1 for r in rs)]
        solved = [t for t in tasks if any(r.task == t and r.passed for r in rs)]
        usd = sum(r.usd for r in rs)
        p1 = sum(1 for r in first if r.passed)
        out.append({"set": s, "config": cfgname, "family": rs[0].family, "harness": rs[0].harness, "tasks": len(tasks), "attempts": len(rs),
                    "pass_first": p1, "pass_first_rate": round(p1 / max(1, len(first)), 3), "pass_first_ci": [round(x, 3) for x in wilson(p1, len(first))],
                    "pass_any_attempt": sum(1 for r in rs if r.passed), "solved_at_budget": len(solved),
                    "solved_rate": round(len(solved) / max(1, len(tasks)), 3), "solved_ci": [round(x, 3) for x in wilson(len(solved), len(tasks))],
                    "usd": round(usd, 4), "usd_per_attempt": round(usd / max(1, len(rs)), 4), "usd_per_solved": round(usd / len(solved), 4) if solved else None,
                    "mean_calls": round(sum(r.calls for r in rs) / max(1, len(rs)), 1), "mean_writes": round(sum(r.writes for r in rs) / max(1, len(rs)), 1),
                    "budget_usd": budget_usd})
    return out


def report_md(summary: Sequence[dict[str, Any]], title: str) -> str:
    head = ("| set | config | family | harness | tasks | attempts | pass@1 (95%) | solved at equal spend (95%) | USD | USD/attempt | "
            "USD/solved | calls | writes |")
    lines = [f"# {title}", "", head, "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary:
        lines.append(f"| {r['set']} | {r['config']} | {r['family']} | {r['harness']} | {r['tasks']} | {r['attempts']} | {r['pass_first']}/{r['tasks']} "
                     f"({r['pass_first_ci'][0]:.2f}–{r['pass_first_ci'][1]:.2f}) | {r['solved_at_budget']}/{r['tasks']} ({r['solved_ci'][0]:.2f}–"
                     f"{r['solved_ci'][1]:.2f}) | {r['usd']:.3f} | {r['usd_per_attempt']:.4f} | {r['usd_per_solved']} | {r['mean_calls']} | {r['mean_writes']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------- the solver ladder (R-ALT-whole)

@dataclass
class LadderRung:
    name: str
    configs: list[str]  # names the caller's configuration map resolves (the wire knows the models)
    per_call: float  # the call budget per oracle call (``compose.agent_budget_for``)


def ladder_rungs() -> list[LadderRung]:
    """R-ALT-whole's ladder, from the policy ledger (DESIGN §7.7 item 24): the stronger configurations compose-2 measured."""
    from crucible.synth import policy

    return [LadderRung(name=r["name"], configs=list(r["configs"]), per_call=float(r["per_call"])) for r in policy.param("R-ALT-whole", "ladder_rungs")]


class LadderOutcome(CrucibleModel):
    task: str
    passed: bool = False
    rung: int | None = None  # the rung that passed
    rungs_tried: int = 0
    rows: list[AttemptRow] = Field(default_factory=list)
    verdicts: dict[int, bool | None] = Field(default_factory=dict)  # failed attempt index -> judged valid (D6) / invalid / undecided
    skipped: list[str] = Field(default_factory=list)  # configurations a rung could not use, with the reason
    reason: str = ""
    policy_version: str = ""


async def escalate_alt(task: Task, horizon: int, configs: dict[str, SolverConfig], *, start: int = 0, builders: Sequence[str] = (),
                       judge: Callable[[Task, Any], Any] | None = None, budget_usd: float | None = None, max_attempts: int | None = None,
                       cost_of: Callable[[str], float] | None = None, stop: Callable[[], bool] | None = None, out: Path | None = None,
                       rungs: Sequence[LadderRung] | None = None) -> LadderOutcome:
    """R-ALT-whole (lead, 2026-10-30): a task with no ALT pass climbs the solver ladder before it is rejected.

    - The WHOLE task, from its instruction (``attempt`` takes the task; nothing here segments it).
    - Rung by rung (from ``start``), every configuration of a rung at the same per-task USD budget (equal spend), each
      attempt on its own tag (``ALT:<task>#<config>@L<rung>~<k>``), so a wire seeding by tag draws independent samples.
    - No builder family and no student family solves (P-family); such a configuration is skipped with the reason.
    - A failed attempt goes to ``judge`` (an adjudicator: True = a valid attempt the contract rejects, a D6) when given;
      without one it stays undecided, so ``certificate.with_ladder`` keeps it as a suspect and nothing is admitted on it.
    - It stops at the first rung with a pass. With none, the reason is ``ALT:no_pass after the solver ladder``: the
      rule never changes."""
    from crucible.synth import policy

    pol_params = policy.current().rule("R-ALT-whole").params
    budget = float(budget_usd if budget_usd is not None else pol_params["usd_per_config_task"])
    tries = int(max_attempts if max_attempts is not None else pol_params["max_attempts"])
    banned = set(builders) | set(policy.param("R-family", "student_families"))
    res = LadderOutcome(task=task.id, policy_version=policy.tag())
    ladder = list(rungs if rungs is not None else ladder_rungs())
    runs: dict[int, Any] = {}
    for i in range(start, len(ladder)):
        rung = ladder[i]
        if stop is not None and stop():
            res.reason = "stopped before the ladder finished"
            return res
        use: list[SolverConfig] = []
        for name in rung.configs:
            cfg = configs.get(name)
            if cfg is None:
                res.skipped.append(f"L{i} {name}: not wired")
            elif cfg.family in banned:
                res.skipped.append(f"L{i} {name}: family {cfg.family} built the task or is the student's (P-family)")
            else:
                use.append(dataclasses.replace(cfg, name=f"{cfg.name}@L{i}", per_call=rung.per_call))
        res.rungs_tried = i + 1
        if not use:
            continue

        async def one(cfg: SolverConfig, i: int = i) -> list[tuple[AttemptRow, Any]]:
            got: list[tuple[AttemptRow, Any]] = []
            spent, k = 0.0, 0
            ctx = FalsifyContext()
            try:
                while k < tries and (k == 0 or spent < budget):
                    k += 1
                    box: list[Any] = []
                    row = await attempt(task, cfg, ctx, k=k, horizon=horizon, set_name=f"ladder-L{i}", cost_of=cost_of,
                                        log_dir=out / "ladder" if out is not None else None, sink=box.append)
                    got.append((row, box[0] if box else None))
                    spent += row.usd
                    if row.passed:
                        break
            finally:
                await ctx.close()
            return got

        for got in await asyncio.gather(*(one(c) for c in use)):
            for row, run in got:
                runs[len(res.rows)] = run
                res.rows.append(row)
        if any(r.passed for r in res.rows):
            res.passed, res.rung = True, i
            break
    if judge is not None:
        for j, row in enumerate(res.rows):
            if not row.passed and row.error is None and runs.get(j) is not None:
                v = judge(task, runs[j])
                res.verdicts[j] = (await v) if asyncio.iscoroutine(v) else v
    if not res.passed and not res.reason:
        res.reason = "ALT:no_pass after the solver ladder"
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        (out / f"ladder-{task.id.replace('/', '_')}.json").write_text(res.model_dump_json(indent=1))
    return res


__all__ = ["AttemptRow", "LadderOutcome", "LadderRung", "SolverConfig", "attempt", "escalate_alt", "evaluate", "ladder_rungs", "report_md",
           "summarize", "wilson"]
