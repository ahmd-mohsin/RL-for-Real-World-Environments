"""The workspace harness: a policy works in a shell workspace World; its work is *collected as a patch* and graded by
the task from a pristine tree (the coding seam; the shape of a Harbor trial with a separate verifier).

Phases: seed (open) -> agent (run_episode; broken-environment check) -> collect (tolerant or legacy; an empty patch from
a changed tree is ``crucible:empty_collection``) -> verification (``task.grade_patch``) -> cleanup.

A collection failure preserves the workspace in the attempt directory. The next attempt of the same sample then
**re-collects only** (tolerant hook, preserved tree) instead of re-running the agent: the agent's work was fine, the
harness lost it (research m9-m10/02 §3.3 row 1). Likewise a grade masked for infrastructure (node contention, a
transport failure) is retried as a **re-grade of the stored patch** (``regrade``; Harbor's regrade-from-artifacts):
re-running the agent would sample a different solution and silently replace the one being measured. M10 lab audit R-3:
the stored patch is what makes the attempt ``regradable`` (``RetryRule`` retries a verification failure only then), and a
failure the patch itself can cause (a verifier timeout, a checker outage, a non-finite reward) is graded again inside the
attempt at once and, when it repeats, decided there instead of masked away by a later retry: by the task's control
(``grading.settle``; ``FibTask``'s oracle patch, ``HarborDirTask``'s ``solution/model.patch``), graded in the same
``PerfGuard`` slot: control passes -> ``crucible:policy_error`` / ``grader_failed_on_answer`` scored 0, fails -> a
task-blamed mask, none -> a mask with flag ``no_control``. A re-grade (``regrade``) ends the same way.

Tasks that declare ``perf = True`` grade inside a ``PerfGuard`` slot (one perf grade per host, quiet host first).

M17 checkpoint credit (opt-in): the harness's ``sidecar(attempt)`` or the task's ``episode_sidecar(attempt)``
(``native.EpisodeSidecar``; ``factory.checkpoint``) observes the episode, grades workspace snapshots out of band and writes
``progress.jsonl``, which the record builder turns into ``Turn.advantage``. Off (the default): no observer, no change.
"""

from __future__ import annotations

import json
import shutil
import time
from typing import Any

from crucible.actors.policy import run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected, classify, collect, repo_broken
from crucible.runtime.grading import answer_sensitive, decided, deciding_record, grade_in_place, grade_summary, grading_window, seen_before, settle, write_reports
from crucible.runtime.harness.base import Attempt, HarnessResult, settle_usage
from crucible.runtime.harness.native import (
    LIMIT_STOPS,
    NativeHarness,
    _keep_tries,
    _take_findings,
    broke_world,
    episode_sidecar,
    misbehaved,
    probed,
    save_episode,
)
from crucible.runtime.infra import StageFailure, boundary, classify_exception, episode_failure
from crucible.verify.patchio import read_patch, write_patch
from crucible.world.exec import HostUnreachable, _run_subprocess


class WorkspaceHarness(NativeHarness):
    def __init__(self, name: str = "native:workspace", version: str = "1", *, collect_mode: str = "tolerant", perf_guard=None, host: str = "local", **kw):
        super().__init__(name, version, **kw)
        self.collect_mode, self.perf_guard, self.host = collect_mode, perf_guard, host

    def _id_config(self) -> dict[str, Any]:
        """M10 lab audit R-4: the collect mode decides what a patch holds, so it is part of the id; the perf guard and the host
        are placement (``host`` is not in the run's identity)."""
        return {**super()._id_config(), "collect_mode": self.collect_mode}

    def identity_config(self) -> dict[str, Any]:
        # ``host`` is left out: where the work is collected is placement (a perf task declares its own measuring host in its
        # document, ``HarborDirTask.identity_config``), and ``*.host`` is no longer in the denylist (M10 lab audit R-4 review)
        return super().identity_config()

    def _prior(self, attempt: Attempt) -> tuple[str, object] | None:
        """What an earlier attempt of this sample left that a retry can reuse instead of re-running the agent. The latest
        attempt decides *what* (re-collect or re-grade); the work itself is found by looking back to the attempt that ran
        the agent (M10 audit V1/V3: a re-collect or a train-mode re-grade preserved nothing, so the next retry re-ran the
        agent and silently replaced the solution being measured). M10 lab audit R-3: the latest attempt is the one whose
        record decides (``grading.deciding_record``: an attempt the process died in, moved to ``aborted/`` without a record,
        is looked past; ``a{n-1}`` alone re-ran the agent after a crash during a re-grade)."""
        if attempt.n <= 1:
            return None
        got = deciding_record(attempt.dir.parent, attempt.n)
        if got is None:
            return None
        r = got[1]
        f = r.get("failure") or {}
        if not r.get("mask_sample"):
            return None
        for k in range(attempt.n - 1, 0, -1):
            d = attempt.dir.parent / f"a{k}"
            if f.get("kind") == "crucible:empty_collection" and (d / "preserved_ws").exists():
                return "recollect", d / "preserved_ws"
            if f.get("stage") in ("verification", "collect") and f.get("kind") != "crucible:empty_collection" and (d / "artifacts" / "model.patch").exists():
                return "regrade", d
        return None

    def _limit_penalty(self, res: HarnessResult, stop: str | None) -> None:
        """m13 C-8 (A2-24): a limit stop under ``penalty:<c>`` gets the configured reward instead of its partial grade, as
        ``NativeHarness.run`` applies it (harness-reward audit 2026-10-28: the workspace harness never did, though its id
        digests the limit policy)."""
        pen = self.limit_policy.penalty(stop) if stop in LIMIT_STOPS else None
        if pen is not None and res.reward is not None:
            res.grade["limit_penalty"] = pen
            res.reward = pen

    async def regrade(self, attempt: Attempt, prior) -> HarnessResult:
        """Grade the patch stored by an earlier attempt (no agent, no spend)."""
        attempt.phase = "verification"  # M10 lab audit R-3: no agent here; an attempt clock firing is this grade's (run.py)
        res = HarnessResult(stop="regraded")
        t: dict[str, float] = {}
        meta = json.loads((prior / "record.json").read_text()).get("meta") or {} if (prior / "record.json").exists() else {}
        patch = read_patch(prior / "artifacts" / "model.patch")  # M7 R3: the stored bytes, CR and all
        try:
            # M10 lab audit R-3 review round 2: the answer-sensitive grades the deciding attempt already ran on this patch, so
            # a repeat here is decided by the control (an answer-sensitive failure graded once was left regradable)
            await self._grade(attempt, Collected(patch=patch, mode="stored", tree_changed=bool(meta.get("tree_changed", True))), res, t,
                              seen=seen_before(attempt.dir.parent, attempt.n))
            self._limit_penalty(res, meta.get("stop"))  # the agent attempt's limit stop keeps its penalty (NativeHarness.regrade does the same)
        except StageFailure as sf:
            res.failure = sf.failure
        self._inherit_calls(res, prior)
        res.artifacts["regraded_from"] = prior.name
        res.timings = {k: round(v, 3) for k, v in t.items()}
        return res

    async def _grade(self, attempt: Attempt, c: Collected, res: HarnessResult, t: dict, *, seen: int = 0) -> list:
        """Grade the patch on a pristine tree (``task.grade_patch``). M10 lab audit R-3: the stored patch makes the attempt
        ``regradable``, so a failure another attempt can answer (a contaminated node, a transport failure) is re-graded
        from it by the next attempt (``regrade``, never an agent re-run); an answer-sensitive failure (a hang or a checker
        outage on this patch) is graded again at once (``grading.grade_in_place``, try ``i`` into ``regrade<i>/grade``) and,
        when it repeats, is decided in this attempt (no longer regradable) by the task's control (``grading.settle``,
        inside the same ``PerfGuard`` slot for a perf task). ``attempt.phase`` / ``attempt.regradable`` follow (the Runner
        reads them when its attempt clock cancels the harness), and a later try that raises keeps the reports before it.

        Review round 2: the grading's window opens on the Runner's clock (``grading.grading_window``); ``seen`` is the
        answer-sensitive grades an earlier attempt ran on this patch (``grading.seen_before``); only a failure that
        repeated or ran out the budget (``grading.decided``) is settled here, one graded once (``grade_retries=0``) stays
        regradable; ``attempt.regradable`` is cleared before an answer-sensitive failure is graded again."""
        attempt.phase = "verification"
        # the window bounds the PerfGuard slot's wait too; it is opened again once the slot is held (graded)
        grading_window(attempt, attempt.task, budget_s=self.scoring_budget_s, control_s=self.control_timeout_s)
        (attempt.dir / "artifacts").mkdir(exist_ok=True)
        write_patch(attempt.dir / "artifacts" / "model.patch", c.patch)
        res.artifacts["patch"] = "artifacts/model.patch"
        res.tree_changed = c.tree_changed
        res.regradable = attempt.regradable = True
        t0 = time.monotonic()
        reps: list = []

        async def grade(d):
            return await attempt.task.grade_patch(c.patch, d)

        async def graded():
            # M10 lab audit R-3 review round 3: the window opened again once the PerfGuard slot is held (never moved earlier):
            # the slot's lock and quiet-host wait ran on the clock and used up the time reserved for the grade and the
            # control, so a decided hang ended as an attempt_timeout mask during its control, never scored 0
            budget = grading_window(attempt, attempt.task, budget_s=self.scoring_budget_s, control_s=self.control_timeout_s)
            with boundary("verification"):
                _, f = await grade_in_place(grade, attempt.dir / "grade", retries=self.grade_retries, budget_s=budget,
                                            again=answer_sensitive, out=reps, seen=seen, on_again=self._on_again(attempt), attempt=attempt)
            if f is not None and decided(f):
                # graded again on this patch already (or the scoring budget ran out during its first grade): decided here,
                # never a third grade in a fresh attempt. M10 lab audit R-3 item 2: by the task's control (grading.settle;
                # the policy's World is closed before any grade, so there is no sandbox to probe), whatever RetryRule
                # allows: with max_attempts=1 (the R3 gate) a hang still scores 0. Review round 2: one answer-sensitive
                # grade alone stays regradable (a grade-only retry grades it again, seen)
                res.regradable = attempt.regradable = False
                f = await settle(attempt.task, f, workdir=attempt.dir / "control", timeout_s=self.control_timeout_s, attempt=attempt)
            attempt.regradable = res.regradable
            return reps, f

        try:
            if self.perf_guard is not None and getattr(attempt.task, "perf", False):
                # the control runs in the same slot (critic amendment 5: one perf grade per host; the slot's lock is not
                # reentrant, so it is never taken twice)
                async with self.perf_guard.slot(self.host) as g:
                    (attempt.dir / "perf_guard.json").write_text(json.dumps(g, default=str))
                    reps, f = await graded()
            else:
                reps, f = await graded()
        except StageFailure:
            _keep_tries(attempt.dir / "verifier", reps, res)  # M10 lab audit R-3: the tries that came back before a later one raised
            raise
        t["verification"] = time.monotonic() - t0
        write_reports(attempt.dir / "verifier", reps)
        res.grade = {**res.grade, **grade_summary(reps, f)}
        _take_findings(reps, res)  # M7 R5: the grade's flags and a policy verdict (as the native harness takes them)
        if f is not None:
            raise StageFailure(f)
        res.reward, res.gates, res.gate_kinds = reps[-1].reward, reps[-1].gate_bits(), reps[-1].gate_kinds()
        return reps

    async def run(self, attempt: Attempt) -> HarnessResult:
        t: dict[str, float] = {}
        res = HarnessResult()
        world = None
        policies: dict[str, Any] = {}
        side = None
        spend0 = attempt.policy.spend()
        prior = self._prior(attempt)
        if prior is not None and prior[0] == "regrade":
            return await self.regrade(attempt, prior[1])
        try:
            if prior is not None:  # re-collect only: never re-run an agent whose work the harness lost
                # the base recorded by the attempt that ran the agent (M10 audit V2: this read the *new* attempt's dir and
                # fell back to the agent-movable ``base`` tag, so a committed fix collected as an empty patch)
                base_file = prior[1].parent / "base_sha"
                if not base_file.exists():
                    raise StageFailure(Failure(kind="crucible:empty_collection", stage="collect", retryable=False,
                                               reason="no recorded base commit for the preserved workspace"))
                res.stop = "recollected"
                self._inherit_calls(res, prior[1])
                c = await collect(prior[1], base=base_file.read_text().strip(), mode="tolerant")
                if c.stalled:
                    # M10 lab audit R-5 review round 1: the first collection of this workspace stalled (a masked empty_collection, the
                    # host's) and so does the re-collect of the same preserved tree: what answers or not is the repository's own
                    # config and attributes (a filter driver or ``textconv`` the policy planted), not the host. The policy's
                    # outcome, scored 0: a masked retry loop would drop the policy's 0 from mean and pass@k
                    raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason=(
                        "git does not answer in the agent's repository (the collection of the preserved workspace ran past its timeout again): "
                        + "; ".join(c.errors))[:400]))
                f = classify(c)
                if f:
                    raise StageFailure(f)
                await self._grade(attempt, c, res, t)
                return res
            t0 = time.monotonic()
            with boundary("seed"):
                world = await attempt.task.open(attempt.dir / "work")
            t["seed"] = time.monotonic() - t0
            t0 = time.monotonic()
            # the commit the agent starts from, recorded before it runs: the ``base`` tag is in the agent's own repository
            # and can be deleted or moved (M7 audit)
            base_sha = await _base_sha(await _workspace(attempt.task, world, "seed"))
            if base_sha:
                (attempt.dir / "base_sha").write_text(base_sha + "\n")
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                attempt.policies = policies  # async-RL audit R3: Runner.inflight_min_version reads them
                side = episode_sidecar(self.sidecar, attempt)  # M17 checkpoint credit (opt-in; None = off, nothing changes)
                ep = await run_episode(world, probed(policies, side, world), limits=self.limits, observers=[side] if side is not None else (),
                                       sample_key=attempt.id)
                if side is not None:  # M7 R7: a sidecar that raised leaves the episode graded, its progress unavailable
                    from crucible.verify.progress import observed_progress

                    res.grade["progress"] = observed_progress(ep.observed[0], attempt.dir / "progress.jsonl")
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            from crucible.credit.hack_detector import rule_flags
            from crucible.credit.shaping import format_errors_by_call

            # Part 1 A1 (2026-10-29): the level-1 rules and the malformed-call counts, as NativeHarness.run takes them. This
            # override skipped both, so the student path's rollouts carried only the grade's flags and Turn.format_errors
            # stayed 0. Recorded only: nothing here changes the reward (the level-2 judge has no caller; the penalty is 0)
            trace = world.to_trace()
            res.flags = [*rule_flags(trace), *res.flags]
            res.format_errors = format_errors_by_call(trace, ep.model_calls)
            save_episode(attempt, ep, res, self.id, policies)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                # M10 lab audit R-3 (b): the episode's blame, mask and retryable are kept (an unclassified policy exception
                # stays unknown: never retried, counted by Health.unknown); the Runner adds the unprobed evidence
                raise StageFailure(episode_failure(f, "agent"))
            pe = next((x for x in ep.failures if x.get("kind") == "crucible:policy_error"), None)
            if pe is not None:  # the policy's own code or output raised: its outcome, scored 0 (M10 audit V6)
                raise StageFailure(Failure(kind="crucible:policy_error", stage="agent", reason=pe.get("reason", "")))
            if (mb := misbehaved(ep) or broke_world(ep)) is not None:  # survivors could still write the tree collected next; or it killed its World
                raise mb
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            ws = await _workspace(attempt.task, world, "collect")
            broken += [b for b in await repo_broken(ws, base_sha) if b not in broken]
            if broken:
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            if ep.stop_condition in LIMIT_STOPS and self.limit_policy.for_stop(ep.stop_condition) == "mask":
                # harness-reward audit 2026-10-28: the limit policy is part of this harness's id, but only NativeHarness applied it
                raise StageFailure(Failure(kind="agent_timeout", stage="agent", reason=ep.stop_condition, mask=True, blame="harness"))
            t0 = time.monotonic()
            attempt.phase = "collect"  # M10 lab audit R-3: the agent phase is over (run.py reads it when its attempt clock fires)
            c = await collect(ws, base=base_sha or "base", mode=self.collect_mode)
            t["collect"] = time.monotonic() - t0
            f = classify(c)
            if f:  # keep the agent's tree as it is (a rename: unreadable files survive) for a re-collect-only retry
                shutil.move(str(ws), str(attempt.dir / "preserved_ws"))
                raise StageFailure(f)
            # the policy's World is closed before grading: a perf grade must not count the attempt's own sandbox or the
            # processes it started as foreign load (M10 audit V5: they masked the grade as node contention)
            closing, world = world, None
            try:
                await closing.close()
            except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                pass
            reps = await self._grade(attempt, c, res, t)
            if side is not None and hasattr(side, "reconcile") and reps:
                # one verdict on the end state (factory.checkpoint, "One verdict on the end state"): the final grade's milestone
                # vector replaces the sidecar's own for the collected patch, so Φ, per-turn credit and the reward agree
                rep = reps[-1]
                fm = (rep.meta or {}).get("milestones") if (rep.infra or {}).get("status") == "ok" else None
                got = side.reconcile(c.patch, fm)
                if got is not None:
                    res.grade["progress"] = got
            self._limit_penalty(res, ep.stop_condition)
        except StageFailure as sf:
            res.failure = sf.failure
        finally:
            if world is not None:
                try:
                    await world.close()
                except Exception:  # noqa: BLE001, S110 -- cleanup never overrides a result
                    pass
            usd, unc = attempt.policy.spend()
            res.cost_usd, res.cost_uncertain_usd = max(0.0, usd - spend0[0]), max(0.0, unc - spend0[1])
            # the attempt's own model calls when it ran the agent (NativeHarness does the same): a client shared by concurrent
            # attempts makes the spend delta count theirs too (M17 report 11: 10 concurrent attempts booked 7.3x the real cost)
            own = [c for cs in (res.calls or {}).values() for c in cs if c.get("cost_usd") is not None]
            if own and not res.artifacts.get("calls_from"):
                res.cost_usd = round(sum(float(c["cost_usd"]) for c in own), 9)
            if policies:  # an agent ran here: a re-collect's calls are those of the attempt that ran it
                settle_usage(res, res.calls or {a: list(getattr(p, "model_calls", [])) for a, p in policies.items()})
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res


async def _workspace(task: Any, world: Any, stage: str) -> Any:
    """The task's workspace directory for ``world``: ``await task.aworkspace(world)`` when the task has one (a remote World
    mirrors its repository here off the event loop and pauses for its host), else ``task.workspace(world)``.
    A host that did not come back is the typed infrastructure failure of ``stage`` (``world.hostreach``), not a harness crash."""
    try:
        return await task.aworkspace(world) if callable(getattr(task, "aworkspace", None)) else task.workspace(world)
    except HostUnreachable as e:
        raise StageFailure(classify_exception(e, stage)) from e


async def _base_sha(ws) -> str | None:
    """The commit the agent starts from (the ``base`` tag), read through ``world.exec._run_subprocess``: off the event loop,
    bounded, its session killed past the bound (M10 lab audit R-5 item 4). None when git is missing, fails or does not answer
    (the re-collect then refuses to guess a base)."""
    try:
        out = await _run_subprocess(["git", "-C", str(ws), "rev-parse", "--verify", "-q", "base^{commit}"], cwd=None, env=None, timeout_s=60, stdin=None)
    except OSError:
        return None
    return (out.stdout.strip() or None) if out.ok else None
