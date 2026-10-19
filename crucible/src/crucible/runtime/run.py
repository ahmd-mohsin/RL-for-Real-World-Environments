"""The runner (research m9-m10/02 §3.2, §3.4): tasks × policies × harnesses × k samples, journaled, resumable, budgeted,
with stop rules; every failure classified, every infrastructure failure masked and retried, never scored 0.

``Runner.run(spec)`` is idempotent: re-running the same spec resumes. Each planned sample has a ``SampleKey`` with a
derived seed; each attempt ``<sample>-a<N>`` gets its own directory. Per attempt:

1. admission: ledger reservation (the harness's estimate); a denial is the budget stop rule;
2. the harness runs the phases (seed -> agent -> verification -> cleanup) and returns a result;
3. the record is built: a scored reward, a policy failure (scored 0), or a masked failure (``reward=None``); a training
   rollout whose capture does not reconcile with the agent's ATIF (``interop.harbor.reconcile`` level fatal) is masked
   ``crucible:capture_incomplete`` (M10 lab audit R-1); a masked failure without evidence gets ``{probe: unprobed,
   control: none, status}`` (M10 lab audit R-3, ``_evidence``);
4. a masked failure is retried while its ``RetryRule`` allows (infra/harness/task blame only), with a fresh attempt. A
   failure after the agent phase (stage ``verification``) is retried only when the harness persisted the work
   (``HarnessResult.regradable``), and the retry re-grades that work without running the agent (M10 lab audit R-3:
   ``grade_only`` on its ``dispatched`` and ``retry_scheduled`` events); otherwise the attempt ends terminal-masked. A
   run past ``attempt_timeout_s`` after the agent phase (``Attempt.phase``) is such a failure too, at that stage, and is
   retryable only when the work was persisted (``Attempt.regradable``): a collect-phase timeout before the patch is
   stored ends terminal-masked. The clock is the harness's too (``Attempt.clock``): its verification phase opens the
   grading's own window on it (``grading.grading_window``), so a grade that hangs ends inside the attempt as
   ``verifier_timeout`` and is decided by the task's control, never cut short by a clock sized for the agent.

Stop rules, checked before every dispatch: budget exhaustion (dispatch halts; resumable after ``increase_limit``),
wall clock, mask fraction > 0.2 after 20 terminal samples (the environment is broken), unknown-blame fraction > 0.05.

M10 lab audit R-4: one writer per run directory. ``run`` and ``regrade`` open the run's journal first, which takes its lease
(``run.lock``) before ``run.json`` is read, before the ledger opens and before any attempt index is chosen, and hold it for
the whole call (every exit path, an exception included, and only after every attempt has ended: an exception that escapes
one attempt cancels the others and waits for them before it propagates); a second caller on the same ``run_id`` raises
``RunBusy`` naming the owner and changes nothing. A summary counts the plan's own samples only (``RunSummary.superseded``:
rows an accepted harness-configuration drift left under the old keys). An attempt index is never reused: a reservation the budget refuses keeps its index
(``refused.json``), is no attempt, and does not use up one of the sample's retries (``RunJournal.dispatched``).

M10 lab audit R-4, identity: ``RunSpec.resolved_doc`` resolves every task, policy and harness to a JSON document
(``runtime.identity``: what an object declares through ``identity_config()``, or the plain data it holds; a closure or a
stateful object with no declaration raises ``Undeclared`` before the run directory is touched), and a resume is refused
(``IdentityDrift``, the dotpaths named) when any leaf but the operational denylist differs. ``RunSpec.accept_drift`` accepts
named paths and opens a new identity epoch instead; code versions open one too and never refuse. Records written in an epoch
after the first carry ``meta.epoch``. The ids that name a harness in the ``SampleKey`` carry a digest of its configuration.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime.budget import BudgetLedger, Exhausted, OpReused
from crucible.runtime.grading import HANDOFF_REF, deciding_record
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime import identity as ident  # the module, called through its attribute (a test patches ``code_versions``)
from crucible.runtime.infra import Health, RetryRule, coverage, mean_reward, pass_at_k
from crucible.runtime.journal import RunJournal
from crucible.runtime.keys import SampleKey, attempt_id, derive_seed


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
    # M10 lab audit R-4: dotpaths (or fnmatch patterns, ``identity.accepts``) of configuration changes a resume of this run accepts:
    # it opens a new identity epoch instead of being refused. Operational, like the fields above: never part of the identity
    accept_drift: tuple[str, ...] = ()

    def identity_doc(self) -> dict:
        """The ids the run is made of (``run.json`` ``spec``, which bench reads). Not the identity: see ``resolved_doc``."""
        return {"run_id": self.run_id, "tasks": [t.id for t in self.tasks], "policies": [p.id for p in self.policies],
                "harnesses": [h.id for h in self.harnesses], "k": self.k, "run_seed": self.run_seed, "tier": self.tier,
                **({"rollout_mode": self.rollout_mode} if self.rollout_mode != "eval" else {})}

    def resolved_doc(self) -> dict:
        """M10 lab audit R-4: the run's identity v2: every task, policy and harness resolved to JSON by ``identity.describe``
        (what it declares through ``identity_config()``, or the plain data it holds), keyed by id (a harness by
        ``name@version``, so a changed configuration is a leaf under the same key), plus the retry rule, the sampling, the
        rollout mode and the run's operational fields (recorded; ``identity.DENY`` drops them from the comparison). Raises
        ``Undeclared`` naming the dotpath of an object that cannot be inspected: nothing is written before it is."""
        return {"v": 2, "run_id": self.run_id, "k": self.k, "run_seed": self.run_seed, "tier": self.tier, "rollout_mode": self.rollout_mode,
                "retry": ident.describe(self.retry, "retry"), "sampling": ident.describe(self.sampling, "sampling"),
                "tasks": {str(t.id): ident.describe(t, f"tasks.{t.id}") for t in self.tasks},
                "policies": {str(p.id): ident.describe(p, f"policies.{p.id}") for p in self.policies},
                "harnesses": {key: ident.describe(h, f"harnesses.{key}") for key, h in zip(ident.harness_keys(self.harnesses), self.harnesses, strict=True)},
                "concurrency": self.concurrency, "budget_usd": self.budget_usd, "wall_clock_s": self.wall_clock_s, "attempt_timeout_s": self.attempt_timeout_s,
                "retry_backoff_s": self.retry_backoff_s, "max_mask_fraction": self.max_mask_fraction, "max_unknown_fraction": self.max_unknown_fraction,
                "min_terminal_for_health": self.min_terminal_for_health}


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
    superseded: int = 0  # M10 lab audit R-4 review: terminal rows under keys this plan does not contain (an earlier epoch's harness ids): not counted above


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
        hids = [h.id for h in spec.harnesses]  # M10 lab audit R-4: a harness id is a digest of its configuration, computed once per plan
        for t in spec.tasks:
            for p in spec.policies:
                for h, hid in zip(spec.harnesses, hids, strict=True):
                    for i in range(spec.k):
                        sk = SampleKey(task=t.id, world=getattr(t, "world_id", ""), policy=p.id, harness=hid, sample_idx=i,
                                       seed=derive_seed(spec.run_seed, t.id, p.id, hid, i), tier=spec.tier)  # type: ignore[arg-type]
                        out.append((sk, t, p, h))
        return out

    def _open(self, spec: RunSpec) -> RunJournal:
        """The run's journal, with its lease (M10 lab audit R-4). The one place a Runner opens a run directory. The identity
        document is resolved first, so an object that cannot be inspected (``identity.Undeclared``) is refused before any
        file is created; the journal then compares it with the stored one under the lease (``IdentityDrift``, epochs)."""
        doc = spec.resolved_doc()
        return RunJournal(self.root, spec.run_id, ident.fingerprint(doc), spec.identity_doc(), resolved=ident.normalize(doc), code=ident.code_versions(),
                          accept_drift=spec.accept_drift)

    async def run(self, spec: RunSpec) -> RunSummary:
        """Run (or resume) ``spec``. M10 lab audit R-4: the run's lease is held for the whole call and released on every exit
        path; a run id another process (or another call of this one) has open raises ``RunBusy`` before anything is touched.
        The plan is built first, so a duplicate id is refused before the run directory is."""
        plan = self.plan(spec)
        journal = self._open(spec)
        try:
            ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})  # after the lease
            try:
                return await self._run(spec, journal, plan, ledger)
            finally:
                ledger.db.close()  # before the lease is released: the next writer opens the ledger on a closed file
        finally:
            journal.close()

    async def _run(self, spec: RunSpec, journal: RunJournal, plan: list[tuple[SampleKey, Any, Any, Any]], ledger: BudgetLedger) -> RunSummary:
        # M10 lab audit R-4 review: only the plan's own samples count. After an accepted harness-configuration drift every sample has
        # a new key, and the earlier epoch's rows (under the old keys) are neither done nor this plan's results: they would count
        # toward the health rules, ``terminal`` could pass ``planned`` and the means would average two configurations
        plan_ids = {row[0].id() for row in plan}
        done = {sid: r for sid, r in journal.terminal().items() if sid in plan_ids}
        health = Health(planned=len(plan))
        for r in done.values():
            health.add(r)
        journal.event("resume" if done else "planned", planned=len(plan), done=len(done), **({"epoch": journal.epoch} if journal.epoch else {}))
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
            except (Exhausted, OpReused) as ex:
                # M10 lab audit R-4: never dispatched, so no attempt and no record, but the index stays taken (refused.json):
                # removing the dir let a later attempt reuse its id and the ledger op of the same name (audit: crash a1,
                # refused a2 reused index 1 and overwrote a1's uncertain row). An op id that already exists is the same kind of
                # stop (it cannot be a fresh attempt), loud as a stop reason: raised, it would leave gather with the other
                # attempts still running while the run's lease is released
                state["stopped"] = f"budget: {ex}" if isinstance(ex, Exhausted) else f"ledger: {ex}"
                journal.event("stop_rule", reason=state["stopped"], attempt=aid)
                journal.refuse(adir, aid, state["stopped"])
                return "stop", n
            # M10 lab audit R-3: a grade-only retry is dispatched too, but runs no agent; the journal tells it apart (taken from
            # the previous attempt's record, which a retry always follows, so a resumed run marks it the same way)
            extra = {"grade_only": True} if self._grade_only_retry(adir, n) else {}
            journal.event("dispatched", attempt=aid, harness=harness.id, policy=policy.id, task=task.id, seed=sk.seed, **extra)
            state["attempts"] += 1
            attempt = Attempt(sample=sk, id=aid, n=n, dir=adir, task=task, policy=policy, ledger=ledger, scopes=scopes,
                              timeout_s=spec.attempt_timeout_s or None)
            try:
                hr: HarnessResult
                if spec.attempt_timeout_s:
                    # M10 lab audit R-3 review round 2: the clock is handed to the harness (Attempt.clock), which opens the
                    # grading's own window on it (grading.grading_window); asyncio.wait_for kept it out of reach
                    async with asyncio.timeout(spec.attempt_timeout_s) as clock:
                        attempt.clock = clock
                        hr = await harness.run(attempt)
                else:
                    hr = await harness.run(attempt)
            except asyncio.CancelledError:
                raise
            except Exception as ex:  # noqa: BLE001 -- the harness's own fault: masked, its spend unknown (held as uncertain)
                # M13 audit G1 D13: a run past attempt_timeout_s stays masked. The clock wraps the whole harness (seed,
                # agent, verification, cleanup), so it cannot tell a slow policy from a hung container or verifier
                # (M10 taxonomy: unclear blame masks). A policy that exhausts its own episode limits is decided inside the
                # harness by its LimitPolicy (scored, penalized or masked). The detail lets bench report this mask apart.
                timeout = isinstance(ex, TimeoutError)
                # M10 lab audit R-3: the clock also covers the in-attempt re-grades and the control. Past the agent phase
                # (the harness's Attempt.phase) a timeout is the grade's: recorded at that stage, regradable only when the
                # work was persisted (Attempt.regradable), so RetryRule answers it with a re-grade or ends it terminal-
                # masked. Read as the agent's, it was retried as a fresh agent run, and an answer that hung the tests past
                # the clock escaped its 0. A policy that runs past the clock during its agent phase stays stage agent
                stage = str(getattr(attempt, "phase", "agent") or "agent") if timeout else "agent"
                stage = stage if stage in ("collect", "verification") else "agent"  # a phase a third-party harness made up reads as before
                why = f"harness {harness.id} ran past {spec.attempt_timeout_s}s" + ("" if stage == "agent" else f" after the agent phase (during {stage})")
                # M10 lab audit R-3 review round 2: past the agent phase with nothing persisted (a clock firing during the
                # workspace collect or the World's close, before _grade stores the patch; an in-place grade) there is no
                # work to re-grade, so the attempt ends terminal-masked. Left retryable, RetryRule (which refuses only stage
                # verification) retried a collect-stage timeout, and the next attempt found no stored patch and ran the
                # agent again. RetryRule itself is unchanged: refusing stage collect there would drop the re-collect of an
                # empty_collection
                regradable = stage != "agent" and bool(getattr(attempt, "regradable", False))
                hr = HarnessResult(failure=Failure(kind="agent_run_error", stage=stage, blame="harness", retryable=stage == "agent" or regradable,
                                                   reason=f"harness {harness.id} raised {type(ex).__name__}: {ex}"[:500] if not timeout else why,
                                                   detail="attempt_timeout" if timeout else None),
                                   cost_uncertain_usd=est, regradable=regradable)
            if hr.failure is not None and hr.failure.delivery == "not_sent" and not hr.cost_usd:
                ledger.release(aid)  # the request never left: it cost nothing
            else:
                ledger.settle(aid, hr.cost_usd, uncertain=hr.cost_uncertain_usd)
            state["max_cost"] = max(state["max_cost"], hr.cost_usd + hr.cost_uncertain_usd)
            try:
                rec = self._record(spec, sk, aid, task, policy, harness, hr, epoch=journal.epoch)
                self._rollout(spec, sk, rec, hr, adir)
            except Exception as ex:  # noqa: BLE001 -- a record or rollout that cannot be built: masked, never a crash
                rec = self._record(spec, sk, aid, task, policy, harness, HarnessResult(failure=Failure(
                    kind="agent_run_error", stage="collect", blame="harness", reason=f"record/rollout: {type(ex).__name__}: {ex}"[:500])), epoch=journal.epoch)
            self._evidence(rec)
            # M10 lab audit R-3: a verification-stage failure is retried only when the harness persisted the work for a re-grade
            # M10 lab audit R-4: the rule counts dispatched attempts, not the index: a refused reservation took an index and ran
            # nothing, so a budget stop and resume must not use up a retry the sample never had
            retry = rec.mask_sample and rec.failure is not None and spec.retry.allows(rec.failure, journal.dispatched(sk.id(), n), regrade=hr.regradable)
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
                journal.event("retry_scheduled", attempt=aid, kind=rec.failure.kind, grade_only=rec.failure.stage == "verification")  # type: ignore[union-attr]
                return "retry", n
            return "done", n

        async def one(sk: SampleKey, task, policy, harness) -> None:
            while True:
                async with sem:
                    what, n = await attempt_once(sk, task, policy, harness)
                if what != "retry":
                    return
                tries = journal.dispatched(sk.id(), n)  # M10 lab audit R-4: refused reservations are no tries
                await asyncio.sleep(spec.retry_backoff_s * 2 ** max(0, tries - 1))  # outside the slot: a backoff blocks no one

        # M10 lab audit R-4 review: the lease is released when this call ends, so it never ends while an attempt is still running or
        # unwinding. ``gather`` alone re-raises the first exception and leaves the other attempts running (and a second
        # ``Runner.run`` would take the lease, move their live dirs to aborted/ and dispatch the same samples again): on any
        # exception (or a cancellation of this call) every attempt is cancelled and awaited first
        tasks = [asyncio.ensure_future(one(*row)) for row in plan if row[0].id() not in done]
        try:
            await asyncio.gather(*tasks)
        except BaseException:
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        final = journal.terminal()
        recs = [r for sid, r in final.items() if sid in plan_ids]
        summ = RunSummary(run_id=spec.run_id, planned=len(plan), terminal=len(recs), scored=sum(1 for r in recs if r.scored_reward() is not None),
                          masked_terminal=sum(1 for r in recs if r.mask_sample), mean_reward=mean_reward(recs), pass_at_1=pass_at_k(recs, 1),
                          coverage=round(coverage(recs, len(plan)), 4), stopped=state["stopped"], attempts=state["attempts"], spend=ledger.summary(),
                          by_failure=health.by_kind, superseded=len(final) - len(recs))
        journal.event("summary", **{k: v for k, v in summ.__dict__.items() if k != "spend"})
        return summ

    async def regrade(self, spec: RunSpec, select=None, *, max_regrades: int = 2) -> RunSummary:
        """Re-grade terminal samples from their stored artifacts (Harbor's regrade pattern), e.g. after the node that
        contaminated their perf grades is quiet again. Default selection: masked at verification. Only harnesses with a
        ``regrade(attempt, prior_dir)`` method qualify, and only samples whose work was persisted: a stored
        ``artifacts/model.patch`` or (M10 lab audit R-3) a native separate verifier's ``verifier/handoff_ref.json``; the
        newest such attempt is the source. A sample re-graded ``max_regrades`` times is left as it is. In train
        mode a FATAL capture-vs-ATIF reconcile carried by the re-grade masks it as in ``_rollout`` (M10 lab audit R-1).

        M10 lab audit R-4: it takes the run's lease like ``run`` and holds it for the whole call (a re-grade writes attempt
        dirs, ledger rows and results rows, so it is a writer too): a run another process has open raises ``RunBusy``."""
        plan = {sk.id(): (sk, t, p, h) for sk, t, p, h in self.plan(spec)}
        journal = self._open(spec)
        try:
            ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})  # after the lease
            try:
                return await self._regrade(spec, journal, plan, select, max_regrades, ledger)
            finally:
                ledger.db.close()
        finally:
            journal.close()

    async def _regrade(self, spec: RunSpec, journal: RunJournal, plan: dict[str, tuple[SampleKey, Any, Any, Any]], select, max_regrades: int,
                       ledger: BudgetLedger) -> RunSummary:
        select = select or (lambda r: r.mask_sample and r.failure is not None and r.failure.stage == "verification")
        n_done = 0
        for sid, r in journal.terminal().items():
            if sid not in plan or not select(r):
                continue
            sk, task, policy, harness = plan[sid]
            sdir = journal.dir / "samples" / sid
            # the work an attempt persisted for a re-grade: a stored patch (workspace) or a quiesced hand-off (native separate
            # verifier, M10 lab audit R-3); an in-place grade leaves neither (its World is gone), so it is never re-graded here
            prior = [d for d in sdir.iterdir() if d.is_dir() and ((d / "artifacts" / "model.patch").exists() or (d / "verifier" / HANDOFF_REF).exists())
                     ] if sdir.exists() else []
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
            rec = self._record(spec, sk, aid, task, policy, harness, hr, epoch=journal.epoch)
            self._reconcile_mask(spec, rec, hr)  # M10 lab audit R-1: no _rollout on this path; the re-attached FATAL reconcile masks as on a grade-only retry
            self._evidence(rec)
            journal.finish(rec, terminal=True)  # the newest terminal row of a sample wins
            n_done += 1
        final = journal.terminal()
        recs = [r for sid, r in final.items() if sid in plan]  # M10 lab audit R-4 review: this plan's samples only (see ``_run``)
        plan_n = len(plan)
        return RunSummary(run_id=spec.run_id, planned=plan_n, terminal=len(recs), scored=sum(1 for r in recs if r.scored_reward() is not None),
                          masked_terminal=sum(1 for r in recs if r.mask_sample), mean_reward=mean_reward(recs), pass_at_1=pass_at_k(recs, 1),
                          coverage=round(coverage(recs, plan_n), 4), stopped=None, attempts=n_done, spend=ledger.summary(), by_failure={},
                          superseded=len(final) - len(recs))

    @staticmethod
    def _reconcile_mask(spec: RunSpec, rec: RunRecord, hr: HarnessResult) -> tuple[dict, dict] | None:
        """The capture-vs-ATIF reconcile report and its first FATAL finding when ``hr.grade["reconcile"]`` is level fatal,
        else None; in train mode a FATAL report masks an unmasked ``rec`` crucible:capture_incomplete (M10 lab audit R-1).
        Shared by ``_rollout`` and ``regrade``: a re-grade re-attaches the source attempt's report, and ``Runner.regrade``
        never reaches ``_rollout``, so it scored a sample whose grade had been masked (and so never reconcile-masked)."""
        rc = hr.grade.get("reconcile")
        if not isinstance(rc, dict) or rc.get("level") != "fatal":
            return None
        fatal = next((f for f in rc.get("findings") or [] if isinstance(f, dict) and f.get("level") == "fatal"), None) or {}
        if spec.rollout_mode == "train" and not rec.mask_sample:
            # M10 lab audit R-1: two independent measurements of the rollout (the proxy's calls, the agent's ATIF) disagree
            # and nothing says which is wrong; training on it is what the reconcile exists to prevent (OpenEnv
            # atif.py:232-240). Eval keeps the grade and meta.reconcile. Unknown blame, never retried: the ATIF is derived
            # from files the policy can write, so a policy that learns to break it stops the run through the 5%
            # unknown-blame rule instead of being re-run until it reconciles (survivorship); subagent runs stop loudly too.
            rec.mark_failure(Failure(kind="crucible:capture_incomplete", stage="collect", blame="unknown", retryable=False, detail="reconcile_fatal",
                                     reason=f"capture reconcile {rc.get('code')}: {fatal.get('detail') or ''}"[:500]))
        return rc, fatal

    @staticmethod
    def _grade_only_retry(adir: Path, n: int) -> bool:
        """M10 lab audit R-3: attempt ``n`` follows a verification-stage mask (``RetryRule`` retries one only when its work
        was persisted, so the harness re-grades it and runs no agent). The deciding attempt is the harnesses' own
        (``grading.deciding_record``: an attempt the process died in, or a grade-only retry that reached no verdict, is
        looked past), so the flag and what the harness does agree on a resumed run too."""
        if n <= 1:
            return False
        got = deciding_record(adir.parent, n)
        return got is not None and bool(got[1].get("mask_sample")) and (got[1].get("failure") or {}).get("stage") == "verification"

    @staticmethod
    def _evidence(rec: RunRecord) -> None:
        """M10 lab audit R-3: every masked failure in a record carries its evidence ``{probe, control, status}``. A harness
        that knows more (a liveness probe, a control grade) fills it; anything else masked here (a harness crash, an
        attempt timeout, a reconcile or capture mask, an ungraded result) is recorded as unprobed and uncontrolled, so a
        reader (bench, the R-5 task breaker) never has to guess whether a non-policy blame was checked. Evidence that lacks
        a key (a budget expiry's ``{budget_s, tries}`` on a failure no ``settle`` saw) keeps what it has and gets the rest."""
        f = rec.failure
        if f is not None and f.masks and not {"probe", "control", "status"} <= set(f.evidence or {}):
            base = {"probe": "unprobed", "control": "none", "status": f.detail or f.kind}
            rec.mark_failure(f.model_copy(update={"evidence": {**base, **(f.evidence or {})}}))

    @staticmethod
    def _rollout(spec: RunSpec, sk: SampleKey, rec: RunRecord, hr: HarnessResult, adir: Path) -> None:
        """Write ``rollout.json`` (crucible.rollout/v1). In train mode an incomplete token record masks the attempt, and so
        does a FATAL capture-vs-ATIF reconcile (``hr.grade["reconcile"]``, M10 lab audit R-1); in eval the report is only
        recorded (``meta.reconcile``) and flagged on the rollout (``reconcile_fatal``)."""
        from crucible.core.registry import load_impl
        from crucible.runtime.record import build

        got = Runner._reconcile_mask(spec, rec, hr)  # marked before build, so rollout.json is masked as well
        parse = load_impl("crucible.interop.harbor.atif:parse_trajectory")  # ATIF models are Harbor's: an interop concern
        trajs = {a: parse((adir / f).read_text()) for k, f in hr.artifacts.items()
                 if k.startswith("trajectory.") and (a := k.split(".", 1)[1]) and (adir / f).exists()}
        prog = adir / "progress.jsonl"
        ro = build(rec, calls=hr.calls, trajectories=trajs, sampling=spec.sampling, group=f"{sk.task}|{sk.harness}", mode=spec.rollout_mode,
                   progress=prog if prog.exists() else None, segments=hr.segments)
        ro.flags = list(hr.flags)
        if got is not None:  # a rollout-level flag in both modes: datasets never take an uncorroborated rollout as a positive
            rc, fatal = got
            ro.flags.append({"rule": "reconcile_fatal", "category": "capture_integrity", "code": rc.get("code"), "evidence": str(fatal.get("detail") or "")[:300]})
        for t in ro.turns:
            t.format_errors = int(hr.format_errors.get(t.call_id, 0))
        if hr.flags:
            ro.reward_components["hack/intercepted"] = float(sum(f["rule"].startswith("intercepted:") for f in hr.flags))
        if ro.failure_kind == "crucible:capture_incomplete" and not rec.mask_sample:
            rec.mark_failure(Failure(kind="crucible:capture_incomplete", stage="collect", reason=ro.stop or "token record incomplete"))
        (adir / "rollout.json").write_text(ro.model_dump_json(by_alias=True, exclude_none=True))

    @staticmethod
    def _record(spec: RunSpec, sk: SampleKey, aid: str, task, policy, harness, hr: HarnessResult, epoch: int = 0) -> RunRecord:
        """The attempt's record. M10 lab audit R-4: written in an identity epoch after the first, it carries ``meta.epoch`` (an
        epoch-0 record is byte-identical to before), so bench splits a run that accepted drift or changed code by it."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = RunRecord(run_id=spec.run_id, task=task.id, world=getattr(task, "world_id", "") or task.id, policy=policy.id, seed=sk.seed, started_at=now,
                        ended_at=now, steps=hr.steps, outcome="completed", reward=hr.reward, cost_usd=hr.cost_usd, cost_uncertain_usd=hr.cost_uncertain_usd,
                        sample=sk.model_dump(mode="json"), attempt=aid,
                        meta={"harness": harness.id, "stop": hr.stop, "truncated": hr.truncated, "gates": hr.gates, "timings": hr.timings,
                              "tree_changed": hr.tree_changed, "artifacts": hr.artifacts,
                              **({"epoch": epoch} if epoch else {}),
                              **({"progress": hr.grade["progress"]} if hr.grade.get("progress") else {}),
                              **({"task_steps": hr.grade["steps"], "step_reward": hr.grade.get("step_reward")} if hr.grade.get("steps") is not None else {}),
                              **({"seat_rewards": hr.grade["seat_rewards"]} if hr.grade.get("seat_rewards") else {}),
                              # M10 lab audit R-1: an external agent's identity (agent, config digest, Harbor version) travels on
                              # every record, its evidence (upstream verdict) and the capture-vs-ATIF reconcile report on every
                              # record whose agent phase ran
                              **({"agent": hr.grade["agent"]} if hr.grade.get("agent") else {}),
                              **({"reconcile": hr.grade["reconcile"]} if hr.grade.get("reconcile") else {}),
                              **({"trajectory_tokens": hr.trajectory_tokens} if hr.trajectory_tokens else {})})
        if hr.failure is not None:
            rec.mark_failure(hr.failure)
        elif hr.reward is None:  # nothing graded and nothing classified: unknown, masked, counted by the stop rule
            rec.mark_failure(Failure(kind="agent_run_error", stage="verification", reason="no grade and no classified failure", blame="unknown"))
        return rec
