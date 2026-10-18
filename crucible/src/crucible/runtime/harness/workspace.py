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
"""

from __future__ import annotations

import json
import shutil
import time

from crucible.actors.policy import run_episode
from crucible.core.schema.run import Failure
from crucible.runtime.collect import Collected, classify, collect, repo_broken
from crucible.runtime.grading import answer_sensitive, decided, deciding_record, grade_in_place, grade_summary, grading_window, seen_before, settle, write_reports
from crucible.runtime.harness.base import Attempt, HarnessResult
from crucible.runtime.harness.native import LIMIT_STOPS, NativeHarness, _keep_tries, save_episode
from crucible.runtime.infra import StageFailure, boundary, episode_failure


class WorkspaceHarness(NativeHarness):
    def __init__(self, name: str = "native:workspace", version: str = "1", *, collect_mode: str = "tolerant", perf_guard=None, host: str = "local", **kw):
        super().__init__(name, version, **kw)
        self.collect_mode, self.perf_guard, self.host = collect_mode, perf_guard, host

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

    async def regrade(self, attempt: Attempt, prior) -> HarnessResult:
        """Grade the patch stored by an earlier attempt (no agent, no spend)."""
        attempt.phase = "verification"  # M10 lab audit R-3: no agent here; an attempt clock firing is this grade's (run.py)
        res = HarnessResult(stop="regraded")
        t: dict[str, float] = {}
        meta = json.loads((prior / "record.json").read_text()).get("meta") or {} if (prior / "record.json").exists() else {}
        patch = (prior / "artifacts" / "model.patch").read_text()
        try:
            # M10 lab audit R-3 review round 2: the answer-sensitive grades the deciding attempt already ran on this patch, so
            # a repeat here is decided by the control (an answer-sensitive failure graded once was left regradable)
            await self._grade(attempt, Collected(patch=patch, mode="stored", tree_changed=bool(meta.get("tree_changed", True))), res, t,
                              seen=seen_before(attempt.dir.parent, attempt.n))
        except StageFailure as sf:
            res.failure = sf.failure
        self._inherit_calls(res, prior)
        res.artifacts["regraded_from"] = prior.name
        res.timings = {k: round(v, 3) for k, v in t.items()}
        return res

    async def _grade(self, attempt: Attempt, c: Collected, res: HarnessResult, t: dict, *, seen: int = 0) -> None:
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
        (attempt.dir / "artifacts" / "model.patch").write_text(c.patch)
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
        if f is not None:
            raise StageFailure(f)
        res.reward, res.gates = reps[-1].reward, reps[-1].gate_bits()

    async def run(self, attempt: Attempt) -> HarnessResult:
        t: dict[str, float] = {}
        res = HarnessResult()
        world = None
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
                c = collect(prior[1], base=base_file.read_text().strip(), mode="tolerant")
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
            base_sha = _base_sha(attempt.task.workspace(world))
            if base_sha:
                (attempt.dir / "base_sha").write_text(base_sha + "\n")
            with boundary("agent"):
                policies = attempt.policy.make(attempt)
                ep = await run_episode(world, policies, limits=self.limits, sample_key=attempt.id)
            t["agent"] = time.monotonic() - t0
            res.steps, res.stop, res.truncated = ep.steps, ep.stop_condition, ep.truncated or ep.stop_condition in LIMIT_STOPS
            save_episode(attempt, ep, res, self.id, policies)
            if ep.masked:
                f = next((x for x in ep.failures if x.get("kind")), {"kind": "agent_run_error", "reason": "episode masked"})
                # M10 lab audit R-3 (b): the episode's blame, mask and retryable are kept (an unclassified policy exception
                # stays unknown: never retried, counted by Health.unknown); the Runner adds the unprobed evidence
                raise StageFailure(episode_failure(f, "agent"))
            pe = next((x for x in ep.failures if x.get("kind") == "crucible:policy_error"), None)
            if pe is not None:  # the policy's own code or output raised: its outcome, scored 0 (M10 audit V6)
                raise StageFailure(Failure(kind="crucible:policy_error", stage="agent", reason=pe.get("reason", "")))
            broken = await attempt.task.broken(world) if hasattr(attempt.task, "broken") else []
            ws = attempt.task.workspace(world)
            broken += [b for b in repo_broken(ws, base_sha) if b not in broken]
            if broken:
                raise StageFailure(Failure(kind="crucible:agent_broke_env", stage="agent", reason="; ".join(broken)[:400]))
            t0 = time.monotonic()
            attempt.phase = "collect"  # M10 lab audit R-3: the agent phase is over (run.py reads it when its attempt clock fires)
            c = collect(ws, base=base_sha or "base", mode=self.collect_mode)
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
            await self._grade(attempt, c, res, t)
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
            res.timings = {k: round(v, 3) for k, v in t.items()}
        return res


def _base_sha(ws) -> str | None:
    import subprocess

    try:
        r = subprocess.run(["git", "-C", str(ws), "rev-parse", "--verify", "-q", "base^{commit}"], capture_output=True, text=True, timeout=60, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    return r.stdout.strip() or None
