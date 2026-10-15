"""The runner (research m9-m10/02 §3.2, §3.4): tasks × policies × harnesses × k samples, journaled, resumable, budgeted,
with stop rules; every failure classified, every infrastructure failure masked and retried, never scored 0.

``Runner.run(spec)`` is idempotent: re-running the same spec resumes. Each planned sample has a ``SampleKey`` with a
derived seed; each attempt ``<sample>-a<N>`` gets its own directory. Per attempt:

1. admission: ledger reservation (the harness's estimate); a denial is the budget stop rule;
2. the harness runs the phases (seed -> agent -> verification -> cleanup) and returns a result;
3. the record is built: a scored reward, a policy failure (scored 0), or a masked failure (``reward=None``);
4. a masked failure is retried while its ``RetryRule`` allows (infra/harness/task blame only), with a fresh attempt.

Stop rules, checked before every dispatch: budget exhaustion (dispatch halts; resumable after ``increase_limit``),
wall clock, mask fraction > 0.2 after 20 terminal samples (the environment is broken), unknown-blame fraction > 0.05.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.run import Failure, RunLock, RunRecord
from crucible.runtime.budget import BudgetLedger, Exhausted
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.infra import Health, RetryRule, coverage, mean_reward, pass_at_k
from crucible.runtime.journal import RunJournal
from crucible.runtime.keys import SampleKey, attempt_id, derive_seed, identity


@dataclass
class RunSpec:
    run_id: str
    tasks: Sequence[Any]  # RunTask adapters (id, open, grade)
    policies: Sequence[Any]  # PolicyRef
    harnesses: Sequence[Any]  # Harness
    k: int = 1
    run_seed: int = 0
    concurrency: int = 4
    tier: str = "full"
    retry: RetryRule = field(default_factory=RetryRule)
    budget_usd: float | None = None
    wall_clock_s: float | None = None
    max_mask_fraction: float = 0.2
    max_unknown_fraction: float = 0.05
    min_terminal_for_health: int = 20
    rollout_mode: str = "eval"  # "train": a rollout without complete token records is masked as crucible:capture_incomplete
    sampling: dict = field(default_factory=dict)  # recorded on every rollout turn (temperature, top_p, max_tokens)
    attempt_timeout_s: float | None = None  # a harness that runs past this is masked (harness blame); sweep G4 D3: a hang ignored every limit
    retry_backoff_s: float = 0.25  # a retry waits backoff * 2**(n-1) outside the concurrency slot (sweep G2 D2: retries were immediate)
    allow_change: Sequence[str] = ()  # dotpath globs of lock settings a resume may change (runtime.lock; recorded in lock.json)

    def identity_doc(self) -> dict:
        return {"run_id": self.run_id, "tasks": [t.id for t in self.tasks], "policies": [p.id for p in self.policies],
                "harnesses": [h.id for h in self.harnesses], "k": self.k, "run_seed": self.run_seed, "tier": self.tier,
                **({"rollout_mode": self.rollout_mode} if self.rollout_mode != "eval" else {})}


def _lock(journal: RunJournal, spec: RunSpec) -> RunLock:
    """Write or check the run's lock, then record this launch, before anything is dispatched (R-1): a changed setting
    raises RunLockMismatch."""
    from crucible.runtime import lock

    rl, status = lock.ensure(journal.dir, spec)
    if status != "same":
        journal.event("run_lock", status=status, fingerprint=rl.fingerprint, changed=rl.changes[-1].changed if status == "changed" else [],
                      opaque=rl.opaque)
    ln = lock.launch(journal.dir, spec, rl, status)  # R-1 step 2: how this invocation ran (argv, operational settings, code)
    journal.event("launch", n=ln.n, git_commit=ln.build.git_commit, git_dirty=ln.build.git_dirty)
    return rl


def _digest_now(task) -> str | None:
    """The task's content digest now (R-2); one that cannot be taken (a file changing mid-read) is reported, never raised."""
    from crucible.runtime.lock import task_digest

    try:
        return task_digest(task)
    except Exception as ex:  # noqa: BLE001 - an unreadable task cannot be the one the lock describes
        return f"unreadable: {type(ex).__name__}: {ex}"[:200]


def _task_changed(task_id: str, before: str | None, after: str | None, when: str, hr: HarnessResult | None = None) -> HarnessResult:
    """R-2: a task whose content changed is never scored as itself (P16: not the policy's failure). Masked with task blame,
    never retried; a grade taken over the changed task is discarded, its cost kept."""
    f = Failure(kind="crucible:task_changed", stage="verification" if hr else "admission", blame="task", retryable=False,
                delivery="delivered" if hr else "not_sent", reason=f"task {task_id} changed {when} ({before} -> {after})")
    return HarnessResult(failure=f, steps=hr.steps if hr else 0, cost_usd=hr.cost_usd if hr else 0.0, cost_uncertain_usd=hr.cost_uncertain_usd if hr else 0.0)


@dataclass
class RunSummary:
    run_id: str
    planned: int
    terminal: int
    scored: int
    masked_terminal: int
    mean_reward: float | None
    pass_at_1: float | None
    coverage: float
    stopped: str | None
    attempts: int
    spend: dict
    by_failure: dict[str, int]


class Runner:
    def __init__(self, root: Path):
        self.root = Path(root)

    def plan(self, spec: RunSpec) -> list[tuple[SampleKey, Any, Any, Any]]:
        """Every planned sample. Duplicate task, policy or harness ids in one run are refused (M13 audit B7: two Harbor
        tasks named after the same directory shared one sample id, the second attempt moved the first's live work dir
        to ``aborted/`` and the report kept one of the two datasets): namespace ids by source before running."""
        for what, items in (("task", spec.tasks), ("policy", spec.policies), ("harness", spec.harnesses)):
            ids = [str(getattr(x, "id", x)) for x in items]
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ValueError(f"duplicate {what} ids in run {spec.run_id}: {dup}; give each a distinct id (e.g. prefix the dataset or path)")
        out = []
        for t in spec.tasks:
            for p in spec.policies:
                for h in spec.harnesses:
                    for i in range(spec.k):
                        sk = SampleKey(task=t.id, world=getattr(t, "world_id", ""), policy=p.id, harness=h.id, sample_idx=i,
                                       seed=derive_seed(spec.run_seed, t.id, p.id, h.id, i), tier=spec.tier)  # type: ignore[arg-type]
                        out.append((sk, t, p, h))
        return out

    async def run(self, spec: RunSpec) -> RunSummary:
        journal = RunJournal(self.root, spec.run_id, identity(spec.identity_doc()), spec.identity_doc())
        rl = _lock(journal, spec)
        planned = {t.id: rl.settings.get(f"tasks.{i}.digest") for i, t in enumerate(spec.tasks)}
        ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})
        plan = self.plan(spec)
        done = journal.terminal()
        health = Health(planned=len(plan))
        for r in done.values():
            health.add(r)
        journal.event("resume" if done else "planned", planned=len(plan), done=len(done))
        state: dict[str, Any] = {"stopped": None, "attempts": 0, "max_cost": 0.0}
        t_start = time.monotonic()
        sem = asyncio.Semaphore(spec.concurrency)

        def stop_check() -> str | None:
            if state["stopped"]:
                return state["stopped"]
            if spec.wall_clock_s is not None and time.monotonic() - t_start > spec.wall_clock_s:
                state["stopped"] = "wall_clock"
            if not state["stopped"] and ledger.over_uncertain_tolerance():  # V4: never checked before
                state["stopped"] = f"budget: uncertain spend above {ledger.uncertain_tolerance:.0%} of the run's spend"
            reason = health.stop_reason(max_mask_fraction=spec.max_mask_fraction, max_unknown_fraction=spec.max_unknown_fraction,
                                        min_terminal=spec.min_terminal_for_health)
            if reason:
                state["stopped"] = f"health: {reason}"
            if state["stopped"]:
                journal.event("stop_rule", reason=state["stopped"])
            return state["stopped"]

        async def attempt_once(sk: SampleKey, task, policy, harness) -> tuple[str, int]:
            """One attempt: ("done" | "retry" | "stop", attempt index). Nothing raised by a harness, its record or its
            rollout escapes (M10 audit V11: one exception aborted the whole run and left every reservation open)."""
            sid = sk.id()
            if stop_check():
                return "stop", 0
            n = journal.next_attempt(sid)
            aid = attempt_id(sid, n)
            adir = journal.attempt_dir(sid, n)
            adir.mkdir(parents=True, exist_ok=True)
            journal.live.add(adir)  # B7: a running attempt's dir is never moved to aborted/ by another attempt's bookkeeping
            try:
                return await _attempt(sk, task, policy, harness, n, aid, adir)
            finally:
                journal.live.discard(adir)

        async def _attempt(sk: SampleKey, task, policy, harness, n: int, aid: str, adir: Path) -> tuple[str, int]:
            scopes = ["run", f"run/policy/{policy.id}", f"run/task/{task.id}"]
            est = float(getattr(policy, "estimate_usd", 0.0) or 0.0)
            if spec.budget_usd is not None:
                # an unpriced policy holds its share of the cap, and every reservation is at least the dearest attempt
                # settled so far (M10 audit V4: reservations of 0 let concurrent attempts spend 3x the cap). The first
                # wave of an unpriced policy can still overrun by concurrency x (cost - share): declare ``estimate_usd``.
                est = max(est if est > 0 else spec.budget_usd / max(1, spec.concurrency), state["max_cost"])
            try:
                ledger.reserve(aid, scopes, est)
            except Exhausted as ex:
                state["stopped"] = f"budget: {ex}"
                journal.event("stop_rule", reason=state["stopped"])
                import shutil

                shutil.rmtree(adir, ignore_errors=True)  # never dispatched: no attempt
                return "stop", n
            want = planned.get(task.id)  # R-2: the task's content digest when the run's lock was taken
            d0 = _digest_now(task) if want is not None else None
            state["attempts"] += 1
            if want is not None and d0 != want:  # edited since the lock was taken: never dispatched, never graded
                hr = _task_changed(task.id, want, d0, "since the run's lock was taken")
            else:
                journal.event("dispatched", attempt=aid, harness=harness.id, policy=policy.id, task=task.id, seed=sk.seed)
                attempt = Attempt(sample=sk, id=aid, n=n, dir=adir, task=task, policy=policy, ledger=ledger, scopes=scopes)
                try:
                    coro = harness.run(attempt)
                    hr: HarnessResult = await (asyncio.wait_for(coro, spec.attempt_timeout_s) if spec.attempt_timeout_s else coro)
                except asyncio.CancelledError:
                    raise
                except Exception as ex:  # noqa: BLE001 -- the harness's own fault: masked, its spend unknown (held as uncertain)
                    # M13 audit G1 D13: a run past attempt_timeout_s stays masked. The clock wraps the whole harness (seed,
                    # agent, verification, cleanup), so it cannot tell a slow policy from a hung container or verifier
                    # (M10 taxonomy: unclear blame masks). A policy that exhausts its own episode limits is decided inside the
                    # harness by its LimitPolicy (scored, penalized or masked). The detail lets bench report this mask apart.
                    timeout = isinstance(ex, TimeoutError)
                    hr = HarnessResult(failure=Failure(kind="agent_run_error", stage="agent", blame="harness",
                                                       reason=f"harness {harness.id} raised {type(ex).__name__}: {ex}"[:500]
                                                       if not timeout else f"harness {harness.id} ran past {spec.attempt_timeout_s}s",
                                                       detail="attempt_timeout" if timeout else None),
                                       cost_uncertain_usd=est)
                if d0 is not None and (d1 := _digest_now(task)) != d0:  # edited during the attempt: its grade is discarded
                    hr = _task_changed(task.id, d0, d1, "during the attempt", hr)
            if hr.failure is not None and hr.failure.delivery == "not_sent" and not hr.cost_usd:
                ledger.release(aid)  # the request never left: it cost nothing
            else:
                ledger.settle(aid, hr.cost_usd, uncertain=hr.cost_uncertain_usd)
            state["max_cost"] = max(state["max_cost"], hr.cost_usd + hr.cost_uncertain_usd)
            try:
                rec = self._record(spec, sk, aid, task, policy, harness, hr, d0)
                self._rollout(spec, sk, rec, hr, adir)
            except Exception as ex:  # noqa: BLE001 -- a record or rollout that cannot be built: masked, never a crash
                rec = self._record(spec, sk, aid, task, policy, harness, HarnessResult(failure=Failure(
                    kind="agent_run_error", stage="collect", blame="harness", reason=f"record/rollout: {type(ex).__name__}: {ex}"[:500])), d0)
            retry = rec.mask_sample and rec.failure is not None and spec.retry.allows(rec.failure, n)
            if rec.failure and rec.failure.kind == "crucible:credits_exhausted":
                state["stopped"] = "budget: provider credits exhausted"
                journal.event("stop_rule", reason=state["stopped"])
                retry = False
                terminal = False  # resumable: the sample stays owed
            else:
                terminal = not retry
            journal.finish(rec, terminal=terminal)
            if terminal:
                health.add(rec)
            if retry:
                journal.event("retry_scheduled", attempt=aid, kind=rec.failure.kind)  # type: ignore[union-attr]
                return "retry", n
            return "done", n

        async def one(sk: SampleKey, task, policy, harness) -> None:
            while True:
                async with sem:
                    what, n = await attempt_once(sk, task, policy, harness)
                if what != "retry":
                    return
                await asyncio.sleep(spec.retry_backoff_s * 2 ** max(0, n - 1))  # outside the slot: a backoff blocks no one

        await asyncio.gather(*(one(*row) for row in plan if row[0].id() not in done))
        recs = journal.records()
        summ = RunSummary(run_id=spec.run_id, planned=len(plan), terminal=len(recs), scored=sum(1 for r in recs if r.scored_reward() is not None),
                          masked_terminal=sum(1 for r in recs if r.mask_sample), mean_reward=mean_reward(recs), pass_at_1=pass_at_k(recs, 1),
                          coverage=round(coverage(recs, len(plan)), 4), stopped=state["stopped"], attempts=state["attempts"], spend=ledger.summary(),
                          by_failure=health.by_kind)
        journal.event("summary", **{k: v for k, v in summ.__dict__.items() if k != "spend"})
        return summ

    async def regrade(self, spec: RunSpec, select=None, *, max_regrades: int = 2) -> RunSummary:
        """Re-grade terminal samples from their stored artifacts (Harbor's regrade pattern), e.g. after the node that
        contaminated their perf grades is quiet again. Default selection: masked at verification. Only harnesses with a
        ``regrade(attempt, prior_dir)`` method qualify; a sample re-graded ``max_regrades`` times is left as it is."""
        journal = RunJournal(self.root, spec.run_id, identity(spec.identity_doc()), spec.identity_doc())
        _lock(journal, spec)
        ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})
        select = select or (lambda r: r.mask_sample and r.failure is not None and r.failure.stage == "verification")
        plan = {sk.id(): (sk, t, p, h) for sk, t, p, h in self.plan(spec)}
        n_done = 0
        for sid, r in journal.terminal().items():
            if sid not in plan or not select(r):
                continue
            sk, task, policy, harness = plan[sid]
            sdir = journal.dir / "samples" / sid
            prior = [d for d in sdir.iterdir() if d.is_dir() and (d / "artifacts" / "model.patch").exists()] if sdir.exists() else []
            regraded = sum(1 for d in sdir.glob("a*/record.json") if (json.loads(d.read_text()).get("meta") or {}).get("stop") == "regraded")
            if not hasattr(harness, "regrade") or not prior or regraded >= max_regrades:
                continue
            src = max(prior, key=lambda d: int(d.name[1:]) if d.name[1:].isdigit() else -1)
            n = journal.next_attempt(sid)
            aid = attempt_id(sid, n)
            adir = journal.attempt_dir(sid, n)
            adir.mkdir(parents=True, exist_ok=True)
            journal.event("regrade", attempt=aid, source=src.name)
            hr = await harness.regrade(Attempt(sample=sk, id=aid, n=n, dir=adir, task=task, policy=policy, ledger=ledger), src)
            rec = self._record(spec, sk, aid, task, policy, harness, hr)
            journal.finish(rec, terminal=True)  # the newest terminal row of a sample wins
            n_done += 1
        recs = journal.records()
        plan_n = len(plan)
        return RunSummary(run_id=spec.run_id, planned=plan_n, terminal=len(recs), scored=sum(1 for r in recs if r.scored_reward() is not None),
                          masked_terminal=sum(1 for r in recs if r.mask_sample), mean_reward=mean_reward(recs), pass_at_1=pass_at_k(recs, 1),
                          coverage=round(coverage(recs, plan_n), 4), stopped=None, attempts=n_done, spend=ledger.summary(), by_failure={})

    @staticmethod
    def _rollout(spec: RunSpec, sk: SampleKey, rec: RunRecord, hr: HarnessResult, adir: Path) -> None:
        """Write ``rollout.json`` (crucible.rollout/v1). In train mode an incomplete token record masks the attempt."""
        from crucible.core.registry import load_impl
        from crucible.runtime.record import build

        parse = load_impl("crucible.interop.harbor.atif:parse_trajectory")  # ATIF models are Harbor's: an interop concern
        trajs = {a: parse((adir / f).read_text()) for k, f in hr.artifacts.items()
                 if k.startswith("trajectory.") and (a := k.split(".", 1)[1]) and (adir / f).exists()}
        prog = adir / "progress.jsonl"
        ro = build(rec, calls=hr.calls, trajectories=trajs, sampling=spec.sampling, group=f"{sk.task}|{sk.harness}", mode=spec.rollout_mode,
                   progress=prog if prog.exists() else None, segments=hr.segments)
        ro.flags = list(hr.flags)
        for t in ro.turns:
            t.format_errors = int(hr.format_errors.get(t.call_id, 0))
        if hr.flags:
            ro.reward_components["hack/intercepted"] = float(sum(f["rule"].startswith("intercepted:") for f in hr.flags))
        if ro.failure_kind == "crucible:capture_incomplete" and not rec.mask_sample:
            rec.mark_failure(Failure(kind="crucible:capture_incomplete", stage="collect", reason=ro.stop or "token record incomplete"))
        (adir / "rollout.json").write_text(ro.model_dump_json(by_alias=True, exclude_none=True))

    @staticmethod
    def _record(spec: RunSpec, sk: SampleKey, aid: str, task, policy, harness, hr: HarnessResult, task_digest: str | None = None) -> RunRecord:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = RunRecord(run_id=spec.run_id, task=task.id, world=getattr(task, "world_id", "") or task.id, policy=policy.id, seed=sk.seed, started_at=now,
                        ended_at=now, steps=hr.steps, outcome="completed", reward=hr.reward, cost_usd=hr.cost_usd, cost_uncertain_usd=hr.cost_uncertain_usd,
                        sample=sk.model_dump(mode="json"), attempt=aid,
                        meta={"harness": harness.id, "stop": hr.stop, "truncated": hr.truncated, "gates": hr.gates, "timings": hr.timings,
                              "tree_changed": hr.tree_changed, "artifacts": hr.artifacts, **({"task_digest": task_digest} if task_digest else {}),
                              **({"progress": hr.grade["progress"]} if hr.grade.get("progress") else {}),
                              **({"task_steps": hr.grade["steps"], "step_reward": hr.grade.get("step_reward")} if hr.grade.get("steps") is not None else {}),
                              **({"seat_rewards": hr.grade["seat_rewards"]} if hr.grade.get("seat_rewards") else {}),
                              **({"trajectory_tokens": hr.trajectory_tokens} if hr.trajectory_tokens else {})})
        if hr.failure is not None:
            rec.mark_failure(hr.failure)
        elif hr.reward is None:  # nothing graded and nothing classified: unknown, masked, counted by the stop rule
            rec.mark_failure(Failure(kind="agent_run_error", stage="verification", reason="no grade and no classified failure", blame="unknown"))
        return rec
