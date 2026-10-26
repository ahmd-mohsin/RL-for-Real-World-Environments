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
4. a masked failure is retried while its ``RetryRule`` allows (infra/harness/task blame only), with a fresh attempt after
   ``RetryRule.backoff`` (M10 lab audit R-5: jittered, capped, never below the failure's ``retry_after``). A
   failure after the agent phase (stage ``verification``) is retried only when the harness persisted the work
   (``HarnessResult.regradable``), and the retry re-grades that work without running the agent (M10 lab audit R-3:
   ``grade_only`` on its ``dispatched`` and ``retry_scheduled`` events); otherwise the attempt ends terminal-masked. A
   run past ``attempt_timeout_s`` after the agent phase (``Attempt.phase``) is such a failure too, at that stage, and is
   retryable only when the work was persisted (``Attempt.regradable``): a collect-phase timeout before the patch is
   stored ends terminal-masked. The clock is the harness's too (``Attempt.clock``): its verification phase opens the
   grading's own window on it (``grading.grading_window``), so a grade that hangs ends inside the attempt as
   ``verifier_timeout`` and is decided by the task's control, never cut short by a clock sized for the agent. M10 lab audit
   R-5: an infra-masked attempt whose breaker key is open (``runtime.breaker``) is never made terminal either: it is
   deferred (``meta.breaker``, a ``deferred`` event), stays owed, and uses up none of the sample's retries
   (``RunJournal.tries``).

Stop rules, checked before every dispatch: budget exhaustion (dispatch halts; resumable after ``increase_limit``),
wall clock, mask fraction > 0.2 after 20 terminal samples (the environment is broken), unknown-blame fraction > 0.05. A
sample waiting on an open breaker (R-5, below) checks them too, each time something changes and when the wall clock would
expire, so a run whose every sample waits still stops on time; so does one in a retry backoff (R-5 review round 1: the wait
ends at the first stop rule and at the wall clock, so a provider's 300 s Retry-After never holds the run, or its lease, past
either).

M10 lab audit R-5, an outage pauses the run instead of masking its samples (``runtime.breaker``): a breaker per provider
endpoint (``PolicyRef.endpoint``), per host and per (task, harness) opens after consecutive failures blamed on infra, harness
or task (``RunSpec.breaker``; None disables them). A sample whose key is open waits before dispatch, outside the concurrency
slot, holding no index and no ledger reservation; after a cool-down one probe attempt runs and its verdict closes the key. A
key still failing after ``BreakerRule.quarantine_s`` is quarantined: journaled with its reason, its samples left owed
(``RunSummary.quarantined``, ``quarantined_samples``) while the rest of the run continues. The breakers are in memory and
start closed on every ``run``; every transition is a journal event. On a resume (item 2) a masked-terminal sample whose failure was an
infra outage at the admission, seed or agent stage that the run's ``RetryRule`` allows (``Runner._reopens``) is dispatched again with a
fresh retry budget (the ``resume`` event lists it under ``reopened``; its attempts before the terminal one are not counted:
``RunJournal.tries(since=)``), and the stop rules count only the samples that end in this call (``Health`` is not seeded with the
earlier windows); ``by_failure`` still reports every masked kind of the plan's final rows.

M10 lab audit R-5 review round 1: the run-global ``Health`` is not fed the terminal masks of a (task, harness) or host key that is
open or quarantined (``RunSummary.breaker_masked``): a broken grader's masks are what its breaker already pauses, and counted there
they would stop the whole campaign on the mask-fraction rule. A grade-only retry needs no provider key (``SampleKeys.provider`` is
None: it neither waits on, probes nor closes the endpoint's breaker), and the breakers' ``failures`` is clamped to
``RetryRule.max_attempts``. A sample a quarantine left owed is not a terminal row of the summary, even when a masked row of an
earlier call is still its newest.

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

M10 lab audit R-6, sandbox hygiene (``world.lifecycle``): every container Crucible starts has an owner and an expiry, and every exit
path reclaims it. Each dispatched attempt (and each ``regrade`` attempt) runs in a ``lifecycle.Scope`` (``lifecycle.attempt``) that
labels what starts under it (``crucible.run``, ``crucible.attempt``, ``crucible.lease`` = this run's ``run.lock``, owner and expiry),
renews the renewable keepers while the attempt lives (the container lifetime is the attempt backstop plus ``RunSpec.hygiene.margin_s``:
R-3's grading window can move the clock past ``attempt_timeout_s``), and on exit, however it ends (scored, masked, past the clock,
cancelled), closes the Worlds still open, removes the containers the attempt started (by name and by label) and writes
``attempt/teardown.json`` ``{terminated, failed, still_up, released}`` before the record (a ``teardown`` event when something is
``failed`` or ``still_up``). A reservation the budget refused is no attempt and has none. The warm container pools belong to the run:
``close_pools`` ends them before the lease is released (a ``pools_closed`` event). After a crash, kill -9 or OOM kill the next start
reclaims (``Runner._sweep``, a ``sweep`` event): ``run`` and ``regrade`` sweep right after taking the lease (so a resume is just another
start), over the local daemon, every docker host the run used before (``RunJournal.sandbox_hosts``) and every one the plan names
(``Runner._plan_hosts``: a task's ``docker_hosts()``), and remove this run's own containers, those of a run on this host whose
``run.lock`` is free or gone, and any container past its expiry (a renewable one by its real deadline: its label is a floor); it
decides by the run lock, never by pid (no ``os.kill(pid, 0)`` on Windows) and never fails the run. ``close_pools`` is keyed by the
run's lease and bounded on the normal path (``lifecycle.POOLS_CLOSE_S``). ``RunSpec.hygiene`` is operational
(``identity.DENY``): the TTL margin, the lifetime without a backstop and ``sweep``.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from crucible.core.schema.run import NON_RETRYABLE_KINDS, Failure, RunLock, RunRecord
from crucible.core.schema.usage import Timing
from crucible.runtime import identity as ident  # the module, called through its attribute (a test patches ``code_versions``)
from crucible.runtime.breaker import BreakerRule, Breakers, SampleKeys, sample_keys
from crucible.runtime.budget import BudgetLedger, Exhausted, OpReused
from crucible.runtime.grading import HANDOFF_REF, deciding_record
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.infra import Health, RetryRule, coverage, mean_reward, pass_at_k
from crucible.runtime.journal import BREAKER_META, LOCK, RunJournal, atomic_write
from crucible.runtime.keys import SampleKey, attempt_id, derive_seed
from crucible.world import hostreach, lifecycle
from crucible.world.lifecycle import Hygiene
from crucible.world.pool import close_pools

OUTAGE_META = "host_outages"  # record.meta: the attempt's remote-host outages (``hostreach.Outages.to_json``)


async def _run_harness(harness: Any, attempt: Attempt, timeout_s: float | None, outages: hostreach.Outages) -> HarnessResult:
    """``harness.run(attempt)`` under the attempt clock (``timeout_s``; none without it), with ``outages`` as the attempt's
    ledger and the clock registered with ``world.hostreach``: a remote host's outage holds it while the episode is paused
    (2026-10-26), so a pause never runs an attempt out of time."""
    with hostreach.episode(outages):
        if not timeout_s:
            return await harness.run(attempt)
        # M10 lab audit R-3 review round 2: the clock is handed to the harness (Attempt.clock), which opens the grading's own
        # window on it (grading.grading_window); asyncio.wait_for kept it out of reach
        async with asyncio.timeout(timeout_s) as clock:
            attempt.clock = clock
            with hostreach.clock(clock):
                return await harness.run(attempt)


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
    # a retry waits ``retry.backoff(failure, tries, retry_backoff_s)`` outside the concurrency slot (sweep G2 D2: retries were
    # immediate); M10 lab audit R-5 item 3: the base of a jittered, capped doubling that is never below a provider's Retry-After
    retry_backoff_s: float = 0.25
    # M10 lab audit R-5 item 1: breakers keyed by provider endpoint, host and (task, harness) (``runtime.breaker``): an outage pauses the
    # samples that need the dependency (deferred, probed after a cool-down, quarantined if it stays down) instead of masking them. None:
    # no breakers, the behaviour before R-5. Operational (``identity.DENY``): when a run pauses, not what it measures
    breaker: BreakerRule | None = field(default_factory=BreakerRule)
    # M10 lab audit R-4: dotpaths (or fnmatch patterns, ``identity.accepts``) of configuration changes a resume of this run accepts:
    # it opens a new identity epoch instead of being refused. Operational, like the fields above: never part of the identity
    accept_drift: tuple[str, ...] = ()
    # M10 lab audit R-6: container lifecycle settings (``world.lifecycle.Hygiene``): the TTL margin and floor added to ``attempt_timeout_s``,
    # the lifetime without an attempt backstop, and whether a Runner start sweeps orphans. Operational (``identity.DENY``): how long an
    # orphan lives, not what a run measures, so changing it never drifts a run
    hygiene: Hygiene = field(default_factory=Hygiene)

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
                "retry_backoff_s": self.retry_backoff_s, "breaker": ident.describe(self.breaker, "breaker"),
                "hygiene": ident.describe(self.hygiene, "hygiene"),
                "max_mask_fraction": self.max_mask_fraction, "max_unknown_fraction": self.max_unknown_fraction,
                "min_terminal_for_health": self.min_terminal_for_health}


def _lock(journal: RunJournal, spec: RunSpec) -> tuple[RunLock, dict[str, Any]]:
    """Write or update the run's lock, then record this launch, before anything is dispatched (R-1): a changed setting
    is recorded (the run identity refused any it did not accept). Returns the lock and the stamp every record of this
    launch carries (R-1 step 3)."""
    from crucible.runtime import lock

    rl, status = lock.ensure(journal.dir, spec)
    if status != "same":
        journal.event("run_lock", status=status, fingerprint=rl.fingerprint, changed=rl.changes[-1].changed if status == "changed" else [],
                      opaque=rl.opaque)
    ln = lock.launch(journal.dir, spec, rl, status)  # R-1 step 2: how this invocation ran (argv, operational settings, code)
    journal.event("launch", n=ln.n, git_commit=ln.build.git_commit, git_dirty=ln.build.git_dirty)
    return rl, {"lock": rl.fingerprint, "launch": ln.n}


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
    if hr is None:
        return HarnessResult(failure=f)
    return HarnessResult(failure=f, steps=hr.steps, cost_usd=hr.cost_usd, cost_uncertain_usd=hr.cost_uncertain_usd, usage=hr.usage, model_s=hr.model_s,
                         timings=hr.timings)


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
    # M10 lab audit R-5: breaker keys quarantined in this call (key -> reason) and the samples left owed by them: neither terminal nor lost,
    # they run again on a resume (the breakers start closed)
    quarantined: dict[str, str] = field(default_factory=dict)
    quarantined_samples: int = 0
    # M10 lab audit R-5 review round 1: terminal masks of this call that ``Health`` did not count because their (task, harness) or host
    # breaker key was open or quarantined (the breaker pauses what they point at; the stop rules would read a broken grader as a broken run)
    breaker_masked: int = 0


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
            rl, stamp = _lock(journal, spec)  # the run lock (R-1), under the lease
            planned = {t.id: rl.settings.get(f"tasks.{i}.digest") for i, t in enumerate(spec.tasks)}
            await self._sweep(spec, journal)  # M10 lab audit R-6: only under the lease; a resume is just another start
            ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})  # after the lease
            try:
                return await self._run(spec, journal, plan, ledger, stamp, planned)
            finally:
                ledger.db.close()  # before the lease is released: the next writer opens the ledger on a closed file
        finally:
            try:
                await self._close_pools(spec, journal)  # M10 lab audit R-6: the warm pool belongs to the run, and ends with it
            finally:
                journal.close()

    @staticmethod
    async def _sweep(spec: RunSpec, journal: RunJournal) -> None:
        """M10 lab audit R-6: reclaim what a crashed run left (``lifecycle.sweep``), at the start of ``run`` and ``regrade`` and only
        once the run's lease is held (a resume is just another start). It covers the local daemon (when a docker CLI exists) and every
        docker host this run has already started containers on (``RunJournal.sandbox_hosts``), and removes: containers carrying this
        run's lease (no attempt of this run is live yet), containers of a run on this host whose ``run.lock`` is free or gone, and
        any ``crucible.owner`` container past its expiry (a renewable one by its real deadline). It decides by the run lock, never by pid
        (``os.kill(pid, 0)`` has no meaning on Windows). Journaled as a ``sweep`` event; ``RunSpec.hygiene.sweep`` turns it off. Never
        fails the run: an error is the event's ``errors``. Review round 1: the docker hosts the plan names (``_plan_hosts``) are swept
        too, so a fresh run id reaches the lab host where an earlier run was killed mid-grade; a host that is neither named nor used
        is covered by the container TTL and the expiry rule only."""
        if not spec.hygiene.sweep or not lifecycle.docker_available():
            return
        try:
            hosts = journal.sandbox_hosts() | {""} | Runner._plan_hosts(spec)
            rep = await lifecycle.sweep(hosts, own_lock=str((journal.dir / LOCK).resolve()), reclaim_own=True)
            journal.event("sweep", **rep.as_dict())
        except Exception as ex:  # noqa: BLE001 -- a sweep that cannot run is journaled, never a failed run
            with contextlib.suppress(Exception):
                journal.event("sweep", hosts=[], removed=[], failed=[], kept=0, errors={"*": f"{type(ex).__name__}: {ex}"[:300]})

    @staticmethod
    def _plan_hosts(spec: RunSpec) -> set[str]:
        """The docker hosts the plan names, for the start sweep (M10 lab audit R-6 review round 1): what each task adapter says through
        an optional ``docker_hosts()`` (``HarborDirTask``: its grading host; ``HarborInPlaceTask``: its World's host). A fresh run id has
        no ``sandbox_host`` events, so without this a lab host whose grade a killed run left running was never swept by another run.
        Never raises: an adapter that cannot say names none."""
        out: set[str] = set()
        for t in spec.tasks:
            fn = getattr(t, "docker_hosts", None)
            if callable(fn):
                with contextlib.suppress(Exception):
                    out |= {str(h) for h in fn() if h}
        return out

    @staticmethod
    async def _close_pools(spec: RunSpec, journal: RunJournal) -> None:
        """M10 lab audit R-6: the run's warm container pools (``world.pool``) end with the run, before its lease is released
        (``close_pools`` had no caller: a pool outlived the run that built it), keyed by the run's lease (``run.lock`` path: a run id is
        unique only within a root). Shielded from a cancel and bounded on the normal path too (``POOLS_CLOSE_S``: a pool refill stuck in
        ``docker run`` must not hold the lease; the close keeps going in the background); what it removed or failed to remove is a
        ``pools_closed`` event. Never raises: a pool that cannot be closed is left to its TTL and the next sweep."""
        try:
            fut = lifecycle.hold(close_pools(owner=str((journal.dir / LOCK).resolve())))
            fut.add_done_callback(lambda f: f.cancelled() or f.exception())  # retrieved: a close that outlives its wait is nobody's error
            try:
                done, _ = await asyncio.wait({fut}, timeout=lifecycle.POOLS_CLOSE_S)
            except asyncio.CancelledError:
                await lifecycle.finish(fut, lifecycle.POOLS_CLOSE_S)
                raise
            if not done:
                journal.event("pools_closed", terminated=[], failed=[], unfinished=0, timed_out=True)
                return
            rep = fut.result()
            if rep["terminated"] or rep["failed"] or rep["unfinished"]:
                journal.event("pools_closed", **rep)
        except Exception:  # noqa: BLE001 -- cleanup never replaces the run's own result or error
            pass

    @staticmethod
    def _scope_base(spec: RunSpec, journal: RunJournal) -> tuple[str, int]:
        """What every attempt scope of this call shares: the absolute path of the run's lease (``crucible.lease``: the sweep decides a
        container's owner is gone by this lock, never by pid) and the container lifetime, the attempt backstop plus a margin
        (``lifecycle.lifetime_for``; ``Hygiene``)."""
        h = spec.hygiene
        return (str((journal.dir / LOCK).resolve()),
                lifecycle.lifetime_for(spec.attempt_timeout_s, margin_s=h.margin_s, floor_s=h.floor_s, default_s=h.default_s))

    @staticmethod
    def _teardown(journal: RunJournal, adir: Path, aid: str, scope: lifecycle.Scope) -> None:
        """M10 lab audit R-6: ``teardown.json`` of an attempt that ended (any way: scored, masked, timed out, cancelled), written
        atomically before its ``record.json``: ``{terminated, failed, still_up, released}``. A reservation the budget refused is no
        attempt and has none. A container the attempt could not confirm removed is also a ``teardown`` event. Then the scope stops
        journaling: nothing may be written to the run after its lease is released."""
        rep = scope.report()
        try:
            atomic_write(adir / "teardown.json", json.dumps(rep, default=str))
        except OSError:
            pass
        if rep["failed"] or rep["still_up"]:
            journal.event("teardown", attempt=aid, failed=len(rep["failed"]), still_up=rep["still_up"])
        scope.event = None

    async def _run(self, spec: RunSpec, journal: RunJournal, plan: list[tuple[SampleKey, Any, Any, Any]], ledger: BudgetLedger,
                   stamp: dict[str, Any], planned: dict[str, str | None]) -> RunSummary:
        # M10 lab audit R-4 review: only the plan's own samples count. After an accepted harness-configuration drift every sample has
        # a new key, and the earlier epoch's rows (under the old keys) are neither done nor this plan's results: they would count
        # toward the health rules, ``terminal`` could pass ``planned`` and the means would average two configurations
        plan_ids = {row[0].id() for row in plan}
        ended = {sid: r for sid, r in journal.terminal().items() if sid in plan_ids}
        # M10 lab audit R-5 item 2: a masked-terminal sample an outage left (a retryable infra kind before the verification phase) is
        # dispatched again on a resume, with a fresh retry budget: ``base`` is the index of its terminal attempt, and
        # ``RunJournal.tries(..., since=base)`` counts only the attempts after it. An owed sample (no terminal row) counts from 0, so a
        # resume after a crash is unchanged. A reopened sample that ends this call deferred, quarantined or stopped keeps its older
        # masked row as its newest (``journal.terminal``: the newest row wins), and the next resume reopens it again
        # (review round 1: and only when the run's own ``RetryRule`` would retry that failure: a rule that excludes the kind or the blame
        # means "never retried", and a resume must not spend a fresh paid attempt on it. ``_reopens`` guarantees ``r.failure``)
        reopened = {sid: r for sid, r in ended.items() if self._reopens(r) and spec.retry.allows(r.failure, 0)}  # type: ignore[arg-type]
        done = {sid: r for sid, r in ended.items() if sid not in reopened}
        base = {sid: self._attempt_index(r) for sid, r in reopened.items()}
        # M10 lab audit R-5 item 2: the stop rules count this call's window only. Seeded with the earlier windows' rows, a resume after an
        # outage (3 masked of 3 terminal) stopped at its first dispatch on the mask-fraction rule, and the run could never recover the
        # samples the outage had cost it
        health = Health(planned=len(plan))
        journal.event("resume" if ended else "planned", planned=len(plan), done=len(done), **({"reopened": sorted(reopened)} if reopened else {}),
                      **({"epoch": journal.epoch} if journal.epoch else {}))
        state: dict[str, Any] = {"stopped": None, "attempts": 0, "max_cost": 0.0}
        t_start = time.monotonic()
        sem = asyncio.Semaphore(spec.concurrency)
        # M10 lab audit R-5 item 1: the breakers (in memory: they start closed on every call, a resume probes again) and the samples a
        # quarantine left owed. ``journal.event`` is how every transition is recorded
        # review round 1: ``failures`` never above the retries a sample has, so no sample ends masked-terminal on a key still closed
        breakers = (Breakers(replace(spec.breaker, failures=max(1, min(spec.breaker.failures, spec.retry.max_attempts))), journal.event)
                    if spec.breaker is not None else None)
        held: set[str] = set()
        stop_ev = asyncio.Event()  # set by the first stop rule: a retry backoff sleeps on it (review round 1), never past a stop
        state_masked = {"breaker": 0}  # terminal masks Health did not count (``RunSummary.breaker_masked``)
        # M10 lab audit R-6: what every attempt's lifecycle scope shares: the lease path, the container lifetime and the docker hosts this
        # run has used (journaled once per host as ``sandbox_host``)
        lock, ttl = self._scope_base(spec, journal)
        run_hosts: set[str] = journal.sandbox_hosts()  # the hosts an earlier call used: ``sandbox_host`` is journaled once per host per run

        def wake_all() -> None:
            """A stop rule fired: a sample in a retry backoff and one waiting on an open key both see it at once."""
            stop_ev.set()
            if breakers is not None:
                breakers.wake()

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
                wake_all()  # M10 lab audit R-5: a sample waiting on an open key or in a backoff sees the stop at once
            return state["stopped"]

        def wall_left() -> float | None:
            return None if spec.wall_clock_s is None else spec.wall_clock_s - (time.monotonic() - t_start)

        async def attempt_once(sk: SampleKey, task, policy, harness, keys: SampleKeys) -> tuple[str, int, Failure | None]:
            """One attempt: ("done" | "retry" | "deferred" | "wait" | "stop", attempt index, the failure a retry follows: M10 lab
            audit R-5 item 3, its Retry-After sets the wait). "wait": a breaker key the sample needs closed on it between its
            admission and its slot (nothing was taken: no index, no reservation); "deferred": the attempt ran and an open
            breaker kept it from being terminal. Nothing raised by a harness, its record or its rollout escapes (M10 audit
            V11: one exception aborted the whole run and left every reservation open)."""
            sid = sk.id()
            if stop_check():
                return "stop", 0, None
            if breakers is not None and breakers.check(sid, keys)[0] not in ("go", "probe"):
                return "wait", 0, None  # in the slot, before ``next_attempt`` and the reservation: an open key costs a waiting sample nothing
            n = journal.next_attempt(sid)
            aid = attempt_id(sid, n)
            adir = journal.attempt_dir(sid, n)
            adir.mkdir(parents=True, exist_ok=True)
            journal.live.add(adir)  # B7: a running attempt's dir is never moved to aborted/ by another attempt's bookkeeping
            try:
                return await _attempt(sk, task, policy, harness, n, aid, adir, keys)
            finally:
                journal.live.discard(adir)

        async def _attempt(sk: SampleKey, task, policy, harness, n: int, aid: str, adir: Path, keys: SampleKeys) -> tuple[str, int, Failure | None]:
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
                wake_all()
                return "stop", n, None
            # M10 lab audit R-3: a grade-only retry is dispatched too, but runs no agent; the journal tells it apart (taken from
            # the previous attempt's record, which a retry always follows, so a resumed run marks it the same way)
            want = planned.get(task.id)  # R-2: the task's content digest when the run's lock was taken
            d0 = _digest_now(task) if want is not None else None
            extra = {"grade_only": True} if self._grade_only_retry(adir, n) else {}
            outages = hostreach.Outages()  # 2026-10-26: the remote-host outages this attempt paused for (``world.hostreach``)
            state["attempts"] += 1
            if want is not None and d0 != want:  # edited since the lock was taken: never dispatched, never graded
                hr = _task_changed(task.id, want, d0, "since the run's lock was taken")
            else:
                journal.event("dispatched", attempt=aid, harness=harness.id, policy=policy.id, task=task.id, seed=sk.seed, **extra)
                attempt = Attempt(sample=sk, id=aid, n=n, dir=adir, task=task, policy=policy, ledger=ledger, scopes=scopes,
                                  timeout_s=spec.attempt_timeout_s or None)
                # M10 lab audit R-6: every dispatched attempt runs in a lifecycle scope. Containers started under it (a World sandbox, a
                # grade container, a borrowed pooled one) are recorded and, while the attempt lives, renewed; on exit, however it ends
                # (scored, masked, past the clock, cancelled), the scope closes any World still open, removes what the attempt started
                # (by name and by its ``crucible.attempt`` label) and writes ``teardown.json``. Renewable, because R-3's grading window
                # can move the attempt clock past ``attempt_timeout_s`` (``grading.grading_window``) and a fixed TTL would end an
                # in-place grade or its control. The reclaim runs after the clock's block ends, so it is never cut short by it
                scope = lifecycle.Scope(run=spec.run_id, attempt=aid, lock=lock, ttl_s=ttl, renewable=True, event=journal.event, hosts=run_hosts)
                try:
                    async with lifecycle.attempt(scope):
                        try:
                            hr: HarnessResult
                            hr = await _run_harness(harness, attempt, spec.attempt_timeout_s, outages)
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
                finally:
                    self._teardown(journal, adir, aid, scope)
                if d0 is not None and (d1 := _digest_now(task)) != d0:  # edited during the attempt: its grade is discarded
                    hr = _task_changed(task.id, d0, d1, "during the attempt", hr)
            if hr.failure is not None and hr.failure.delivery == "not_sent" and not hr.cost_usd:
                ledger.release(aid)  # the request never left: it cost nothing
            else:
                ledger.settle(aid, hr.cost_usd, uncertain=hr.cost_uncertain_usd)
            state["max_cost"] = max(state["max_cost"], hr.cost_usd + hr.cost_uncertain_usd)
            try:
                rec = self._record(spec, sk, aid, task, policy, harness, hr, d0, epoch=journal.epoch, **stamp)
                self._rollout(spec, sk, rec, hr, adir)
            except Exception as ex:  # noqa: BLE001 -- a record or rollout that cannot be built: masked, never a crash
                rec = self._record(spec, sk, aid, task, policy, harness, HarnessResult(failure=Failure(
                    kind="agent_run_error", stage="collect", blame="harness", reason=f"record/rollout: {type(ex).__name__}: {ex}"[:500])), d0, epoch=journal.epoch, **stamp)
            if outages.rows:  # every pause of this attempt for a remote host, resumed or not: the record and the run journal carry them
                rec.meta[OUTAGE_META] = outages.to_json()
                journal.event("host_outage", attempt=aid, hosts=rec.meta[OUTAGE_META]["hosts"], paused_s=rec.meta[OUTAGE_META]["paused_s"],
                              outages=len(outages.rows), resumed=rec.meta[OUTAGE_META]["resumed"], masked=bool(rec.mask_sample),
                              failure=rec.failure.kind if rec.failure else None, detail=rec.failure.detail if rec.failure else None)
            self._evidence(rec)
            # M10 lab audit R-3: a verification-stage failure is retried only when the harness persisted the work for a re-grade
            # M10 lab audit R-4: the rule counts dispatched attempts, not the index: a refused reservation took an index and ran
            # nothing, so a budget stop and resume must not use up a retry the sample never had
            # M10 lab audit R-5 item 1: and not the attempts an open breaker deferred (``RunJournal.tries``): they failed because a
            # dependency was down, so they say nothing of whether another attempt would
            open_keys = breakers.record(sk.id(), keys, rec) if breakers is not None else []
            tries = journal.tries(sk.id(), n, since=base.get(sk.id(), 0))  # R-5 item 2: a reopened sample's earlier attempts are not its tries
            retry = rec.mask_sample and rec.failure is not None and spec.retry.allows(rec.failure, tries, regrade=hr.regradable)
            deferred = False
            if rec.failure and rec.failure.kind == "crucible:credits_exhausted":
                state["stopped"] = "budget: provider credits exhausted"
                journal.event("stop_rule", reason=state["stopped"])
                wake_all()
                retry = False
                terminal = False  # resumable: the sample stays owed
            elif (open_keys and rec.failure is not None and rec.failure.who == "infra" and rec.failure.detail != "attempt_timeout"
                  and spec.retry.allows(rec.failure, 0, regrade=hr.regradable)):
                # M10 lab audit R-5 item 1: an infra-masked attempt whose key is open (it opened it, or it was already open, probing
                # or quarantined) never becomes terminal: the dependency is down, not the sample. It stays owed, waits at admission
                # until the key's probe closes it, and its sample's retry budget is untouched (``meta.breaker`` is read by
                # ``RunJournal.tries``; set before ``finish`` so record.json carries it). Only a failure a retry could have changed
                # is deferred: a non-retryable kind, or a verification failure with no persisted work, ends as ``RetryRule`` says
                # (its sample's later samples wait). A harness- or task-blamed failure counts toward opening the key but is never
                # deferred either: its attempt ran the policy, and running it again for as long as a key is open would pick the
                # samples that finish (survivorship, P23); a run past ``attempt_timeout_s`` is that too (M13 audit G1 D13)
                retry = False
                terminal = False
                deferred = True
                rec.meta[BREAKER_META] = {"deferred": open_keys}
            else:
                terminal = not retry
            journal.finish(rec, terminal=terminal)
            if terminal:
                if any(not k.startswith("provider:") for k in open_keys):
                    # M10 lab audit R-5 review round 1: a terminal mask counted against a (task, harness) or host key that is open or
                    # quarantined is what that key's breaker already pauses (a grader that fails its control is task-blamed and never
                    # deferred: the attempts before the key opened, every failed probe after it did). Fed to the run-global Health
                    # they were the mask-fraction rule's, and one broken grader stopped the campaign. A provider key is left out of
                    # this: its outage is deferred, never terminal, so an infra mask that reaches here is the sample's own
                    state_masked["breaker"] += 1
                else:
                    health.add(rec)
            if deferred:
                journal.event("deferred", attempt=aid, keys=open_keys, kind=rec.failure.kind)  # type: ignore[union-attr]
                return "deferred", n, rec.failure
            if retry:
                journal.event("retry_scheduled", attempt=aid, kind=rec.failure.kind, grade_only=rec.failure.stage == "verification")  # type: ignore[union-attr]
                return "retry", n, rec.failure
            return "done", n, None

        async def one(sk: SampleKey, task, policy, harness) -> None:
            sid = sk.id()
            planned_keys = sample_keys(sk, task, policy, harness)
            while True:
                keys = planned_keys
                if breakers is not None and keys.provider is not None:
                    # M10 lab audit R-5 review round 1: the keys are the next attempt's, not the sample's: a grade-only retry (R-3) runs
                    # no agent and calls no model, so it neither waits on nor probes nor closes the provider key (a clean regrade says
                    # nothing of an endpoint that is still down). Decided as the harness and the journal decide it
                    n_next = journal.next_attempt(sid)
                    if self._grade_only_retry(journal.attempt_dir(sid, n_next), n_next):
                        keys = planned_keys.without_provider()
                if breakers is not None:
                    # M10 lab audit R-5 item 1: a sample whose key is open waits here, outside the slot (it holds no concurrency, no
                    # index, no reservation), until the key's probe closes it; one whose key was quarantined is left owed
                    got = await breakers.admit(sid, keys, stopped=lambda: bool(stop_check()), remaining=wall_left)
                    if got == "quarantined":
                        held.add(sid)
                        journal.event("quarantined", sample=sid, reasons={k: v for k, v in breakers.quarantined().items() if k in keys.all()})
                        return
                    if got == "stopped":
                        return
                async with sem:
                    what, n, f = await attempt_once(sk, task, policy, harness, keys)
                if breakers is not None and what in ("wait", "stop"):
                    breakers.release(sid)  # a probe this sample claimed but did not run is the next sample's
                if what in ("wait", "deferred"):
                    continue
                if what != "retry":
                    return
                # M10 lab audit R-4: refused reservations are no tries; R-5: nor are deferred attempts, nor a reopened sample's earlier ones
                tries = journal.tries(sid, n, since=base.get(sid, 0))
                # M10 lab audit R-5 item 3: jittered, capped, and never below the provider's Retry-After (``RetryRule.backoff``)
                delay = spec.retry.backoff(f, tries, spec.retry_backoff_s)
                left = wall_left()
                if left is not None:
                    # review round 1: never past the wall clock (a 300 s Retry-After is a wait, not a stop); 10 ms over, so the timer
                    # cannot fire a clock tick early and let ``stop_check`` below read the clock as not yet expired
                    delay = min(delay, max(0.0, left) + 0.01)
                if delay > 0:
                    try:
                        await asyncio.wait_for(stop_ev.wait(), timeout=delay)  # outside the slot: a backoff blocks no one; a stop wakes it
                    except TimeoutError:
                        pass
                if stop_check():
                    return

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
        # review round 1: a sample a quarantine left owed is owed, not terminal, even when a masked row of an earlier call (a reopened
        # sample) is still its newest: counted in ``quarantined_samples`` only, never also in ``terminal`` / ``masked_terminal`` / ``by_failure``
        recs = [r for sid, r in final.items() if sid in plan_ids and sid not in held]
        # M10 lab audit R-5 item 2: the masked kinds of the plan's final records, not ``health.by_kind`` (Health counts this call's window
        # only, so a resumed run's summary would have dropped every earlier window's masks while ``masked_terminal`` still counted them)
        by_failure: dict[str, int] = {}
        for r in recs:
            if r.mask_sample:
                k = r.failure.kind if r.failure else "?"
                by_failure[k] = by_failure.get(k, 0) + 1
        summ = RunSummary(run_id=spec.run_id, planned=len(plan), terminal=len(recs), scored=sum(1 for r in recs if r.scored_reward() is not None),
                          masked_terminal=sum(1 for r in recs if r.mask_sample), mean_reward=mean_reward(recs), pass_at_1=pass_at_k(recs, 1),
                          coverage=round(coverage(recs, len(plan)), 4), stopped=state["stopped"], attempts=state["attempts"], spend=ledger.summary(),
                          by_failure=by_failure, superseded=sum(1 for sid in final if sid not in plan_ids),
                          quarantined=breakers.quarantined() if breakers is not None else {}, quarantined_samples=len(held),
                          breaker_masked=state_masked["breaker"])
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
            _, stamp = _lock(journal, spec)  # the run lock (R-1), under the lease
            await self._sweep(spec, journal)  # M10 lab audit R-6: a regrade is a start too (it takes the lease and starts containers)
            ledger = BudgetLedger(journal.dir / "ledger.sqlite", {"run": spec.budget_usd} if spec.budget_usd is not None else {})  # after the lease
            try:
                return await self._regrade(spec, journal, plan, select, max_regrades, ledger, stamp)
            finally:
                ledger.db.close()
        finally:
            try:
                await self._close_pools(spec, journal)  # M10 lab audit R-6
            finally:
                journal.close()

    async def _regrade(self, spec: RunSpec, journal: RunJournal, plan: dict[str, tuple[SampleKey, Any, Any, Any]], select, max_regrades: int,
                       ledger: BudgetLedger, stamp: dict[str, Any]) -> RunSummary:
        select = select or (lambda r: r.mask_sample and r.failure is not None and r.failure.stage == "verification")
        lock, ttl = self._scope_base(spec, journal)  # M10 lab audit R-6: a regrade attempt runs in a lifecycle scope like a dispatched one
        run_hosts: set[str] = journal.sandbox_hosts()
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
            scope = lifecycle.Scope(run=spec.run_id, attempt=aid, lock=lock, ttl_s=ttl, renewable=True, event=journal.event, hosts=run_hosts)
            try:
                async with lifecycle.attempt(scope):
                    hr = await harness.regrade(Attempt(sample=sk, id=aid, n=n, dir=adir, task=task, policy=policy, ledger=ledger), src)
            finally:
                self._teardown(journal, adir, aid, scope)  # M10 lab audit R-6: a regrade attempt dir has its teardown.json too
            rec = self._record(spec, sk, aid, task, policy, harness, hr, epoch=journal.epoch, **stamp)
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
    def _reopens(r: RunRecord) -> bool:
        """M10 lab audit R-5 item 2: whether a terminal row is a sample an outage left masked, so a resume dispatches it again with
        a fresh retry budget. An outage that outlasted the retries ended its samples terminal-masked at the stages before the
        verdict, and nothing reopened them: their coverage was lost for good (``Runner.regrade`` covers verification only). True
        for a masked, infra-blamed, retryable failure of a kind a retry can change, at the admission, seed or agent stage (the
        caller adds the run's own rule: ``spec.retry.allows(r.failure, 0)``, review round 1: a kind or blame the ``RetryRule``
        excludes is never retried, and a resume does not spend a paid attempt on it). Not
        reopened: a verification or collect failure (R-3: the policy is never sampled again because a grade or a collect failed;
        ``regrade`` re-grades the stored work), a harness-, task- or unknown-blamed one (not an outage: the same attempt would fail
        the same way, and re-running a policy that hangs a harness picks the samples that finish), a ``NON_RETRYABLE_KINDS`` kind
        and exhausted provider credits (a retry cannot fix them)."""
        f = r.failure
        return (r.mask_sample and f is not None and f.who == "infra" and f.retryable and f.kind not in NON_RETRYABLE_KINDS
                and f.kind != "crucible:credits_exhausted" and f.stage in ("admission", "seed", "agent"))

    @staticmethod
    def _attempt_index(r: RunRecord) -> int:
        """The index ``N`` of a record's attempt ``<sample>-a<N>`` (0 when it names none): where a reopened sample's fresh retry budget
        starts counting (``RunJournal.tries(..., since=N)``)."""
        try:
            return int((r.attempt or "").rsplit("-a", 1)[1])
        except (IndexError, ValueError):
            return 0

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
        reader (bench, the (task, harness) breaker's events) never has to guess whether a non-policy blame was checked. Evidence that lacks
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
        ro = build(rec, calls=hr.calls, trajectories=trajs, group=f"{sk.task}|{sk.harness}", mode=spec.rollout_mode,
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
    def _record(spec: RunSpec, sk: SampleKey, aid: str, task, policy, harness, hr: HarnessResult, task_digest: str | None = None, epoch: int = 0, *,
                lock: str | None = None, launch: int | None = None) -> RunRecord:
        """The attempt's record. M10 lab audit R-4: written in an identity epoch after the first, it carries ``meta.epoch`` (an
        epoch-0 record is byte-identical to before), so bench splits a run that accepted drift or changed code by it."""
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        rec = RunRecord(run_id=spec.run_id, task=task.id, task_digest=task_digest, world=getattr(task, "world_id", "") or task.id, policy=policy.id,
                        seed=sk.seed, started_at=now, ended_at=now, steps=hr.steps, outcome="completed", reward=hr.reward, cost_usd=hr.cost_usd,
                        cost_uncertain_usd=hr.cost_uncertain_usd, usage=hr.usage, timing=Timing.from_stages(hr.timings, model=hr.model_s),
                        sample=sk.model_dump(mode="json"), attempt=aid, lock=lock, launch=launch,
                        meta={"harness": harness.id, "stop": hr.stop, "truncated": hr.truncated, "gates": hr.gates,
                              **({"gate_kinds": hr.gate_kinds} if hr.gate_kinds else {}),
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
