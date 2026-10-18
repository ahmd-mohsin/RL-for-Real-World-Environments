"""M10 lab audit R-3, slice B: grade-only retries. A failed grade is graded again on the same work; the policy is never
sampled again because a grade failed.

- ``RetryRule`` refuses a verification-stage retry unless the attempt persisted its work (``HarnessResult.regradable``).
- A native separate verifier persists its quiesced hand-off (``verifier/handoff_ref.json``); the next attempt (and
  ``Runner.regrade``) re-grades it in a fresh verifier World without running the agent (``NativeHarness.regrade``); a
  ref that cannot be read, or whose spec is not the snapshot's, is a harness mask, never an agent re-run.
- In place (native, Harbor's installed-agent harness) the grade is retried inside the attempt before the World closes
  (``grading.grade_in_place``: ``regrade<i>/grade``, ``grades.json``, one scoring budget) and the attempt then ends
  terminal-masked; Harbor's in-place re-grade gets its own quiesce receipt and never flags the first grade's leftovers.
- The workspace harness grades an answer-sensitive failure (a checker outage, a timeout) of its patch again at once, and
  one that repeats is not left to a later retry; a node contention still waits for a grade-only retry.
- A non-finite reward, a grader's own ``TimeoutError`` and the scoring budget's expiry are typed grade failures, never a
  harness crash that the Runner would retry as a fresh agent run. A multi-step step is graded again in place unless its
  task declares ``step_grades_once`` (then graded once).

Review round 1: the Runner's attempt clock firing during a grade is a verification failure (``Attempt.phase``), never an
agent re-run; a budget expiry on a later try keeps the earlier failure; a grade that raises a transport error is graded
again in place; the verifier World's own start is classified by ``boundary`` (re-graded from the hand-off); a crash
during a re-grade is looked past (``grading.deciding_record``); earlier tries' reports survive a later try's raise; a
hand-off with a missing or short trace is ``handoff_unreadable``.

Review round 2 (more in ``test_m10_lab_audit_r3_round2.py``): the attempt clock's grading window makes a hang a settled
``verifier_timeout`` (the clock still ends an infrastructure wait); a step graded once is flagged ``not_repeated``.

No Docker, no network, no Harbor (one test stands in for Harbor's ``Verifier`` and skips without it).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.core.schema.run import Failure
from crucible.runtime.grading import answer_sensitive, grade_in_place, settle, try_dir
from crucible.runtime.harness.base import Attempt
from crucible.runtime.harness.native import NativeHarness, PolicyRef, TaskStep
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import RetryRule, StageFailure
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import GradeReport
from tests.unit.test_verifier_separate import GOOD, HIDDEN, CustodyTask


def _report(tid: str, reward: float = 0.0, infra: dict | None = None) -> GradeReport:
    return GradeReport(task=tid, tier="full", checks=[], gates={}, reward=reward, outcome=0.0 if reward != reward else reward,
                       infra=infra or {"status": "ok"}, trusted=True, elapsed_s=0.0)


def _masked(tid: str, reason: str, kind: str = "check_infra") -> GradeReport:
    return _report(tid, 0.0, {"status": "masked", "failure_kind": kind, "reason": reason})


def _counting(make_one) -> tuple[PolicyRef, list[str]]:
    made: list[str] = []

    def make(a):
        made.append(a.id)
        return make_one(a)

    return PolicyRef(id="scripted:counted", make=make), made


def _rows(run: Path, name: str) -> list[dict]:
    p = run / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _adir(run: Path, n: int = 1) -> Path:
    return next((run / "samples").glob(f"*/a{n}"))


@pytest.fixture
def seed(tmp_path):
    d = tmp_path / "seed"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_hidden.py").write_text(HIDDEN)
    return d


# ---------------------------------------------------------------- RetryRule


def test_retry_rule_refuses_verification_without_regrade():
    rr = RetryRule()
    nc = Failure(kind="crucible:node_contention", stage="verification", reason="contaminated")
    assert not rr.allows(nc, 1)  # the default fails closed: no persisted work, no retry
    assert rr.allows(nc, 1, regrade=True) and not rr.allows(nc, 3, regrade=True)
    assert rr.allows(Failure(kind="transport_unreachable", stage="agent"), 1)  # the agent stage is unchanged
    assert not rr.allows(Failure(kind="crucible:policy_misbehavior", stage="verification"), 1, regrade=True)
    assert not rr.allows(Failure(kind="crucible:stale_image", stage="verification"), 1, regrade=True)
    assert not rr.allows(Failure(kind="verifier_error", stage="verification", blame="unknown"), 1, regrade=True)


# ---------------------------------------------------------------- grading.grade_in_place


async def test_grade_in_place_dirs_rules_and_budget(tmp_path):
    g0 = tmp_path / "a1" / "grade"
    assert try_dir(g0, 0) == g0 and try_dir(g0, 2) == tmp_path / "a1" / "regrade2" / "grade"
    seen: list[Path] = []

    async def outage(d):
        seen.append(Path(d))
        return _masked("t", "judge endpoint down")

    reps, f = await grade_in_place(outage, g0, retries=2)
    assert seen == [g0, tmp_path / "a1" / "regrade1" / "grade", tmp_path / "a1" / "regrade2" / "grade"] and len(reps) == 3
    assert (f.kind, f.detail, f.stage) == ("verifier_error", "check_infra", "verification") and answer_sensitive(f)

    seen.clear()

    async def contended(d):
        seen.append(Path(d))
        return _masked("t", "perf rounds contaminated: candidate")

    reps, f = await grade_in_place(contended, g0, retries=2, again=answer_sensitive)
    assert len(seen) == 1 and f.kind == "crucible:node_contention"  # left to a grade-only retry in a fresh attempt
    seen.clear()

    async def stale(d):
        seen.append(Path(d))
        return _masked("t", "stale image: digest differs")

    _reps, f = await grade_in_place(stale, g0, retries=2)
    assert len(seen) == 1 and f.kind == "crucible:stale_image"  # no grade fixes it

    async def unsupported(d):
        raise StageFailure(Failure(kind="crucible:backend_error", stage="verification", retryable=False, blame="unknown", detail="quiesce_respawned"))

    with pytest.raises(StageFailure) as ei:  # R-2's verdicts propagate unchanged
        await grade_in_place(unsupported, g0)
    assert ei.value.failure.detail == "quiesce_respawned"

    async def slow(d):
        await asyncio.sleep(5)
        return _report("t", 1.0)

    reps, f = await grade_in_place(slow, g0, retries=3, budget_s=0.05)
    assert reps == [] and (f.kind, f.detail) == ("verifier_error", "verifier_timeout") and "scoring budget" in f.reason

    calls = {"n": 0}

    async def raises_timeout(d):
        calls["n"] += 1
        raise RuntimeError("judge call failed") from TimeoutError("read timed out")

    reps, f = await grade_in_place(raises_timeout, g0, retries=1)
    assert calls["n"] == 2 and reps == [] and (f.kind, f.detail) == ("verifier_error", "verifier_timeout")

    async def crashes(d):
        raise ValueError("grader bug")

    with pytest.raises(ValueError):  # unknown blame: the caller's boundary's to classify, never graded again
        await grade_in_place(crashes, g0)

    # review round 1: a raised transport error is classified as boundary() would and graded again in place
    n = {"calls": 0}

    async def reset_once(d):
        n["calls"] += 1
        if n["calls"] == 1:
            raise ConnectionResetError("judge connection reset")
        return _report("t", 1.0)

    reps, f = await grade_in_place(reset_once, g0, retries=1)
    assert n["calls"] == 2 and len(reps) == 1 and f is None

    n["calls"] = 0

    async def always_reset(d):
        n["calls"] += 1
        raise ConnectionResetError("judge connection reset")

    reps, f = await grade_in_place(always_reset, g0, retries=1)
    assert n["calls"] == 2 and reps == [] and f.kind == "transport_unreachable" and f.stage == "verification"
    n["calls"] = 0
    with pytest.raises(ConnectionResetError):  # a persisted path: left to a grade-only retry in a fresh attempt, unchanged
        await grade_in_place(always_reset, g0, retries=1, again=answer_sensitive)
    assert n["calls"] == 1


async def test_a_budget_expiry_on_a_later_grade_keeps_the_earlier_failure(tmp_path):
    g0 = tmp_path / "a1" / "grade"
    n = {"grades": 0, "controls": 0}

    async def contended_then_slow(d):
        n["grades"] += 1
        if n["grades"] == 1:
            return _masked("t", "perf rounds contaminated: candidate")  # node_contention: infra, regradable in place
        await asyncio.sleep(5)
        return _report("t", 1.0)

    reps, f = await grade_in_place(contended_then_slow, g0, retries=2, budget_s=0.2)
    assert n["grades"] == 2 and len(reps) == 1
    assert f.kind == "crucible:node_contention" and not answer_sensitive(f) and "scoring budget" in f.reason

    class Task:
        async def grade_control(self, workdir):
            n["controls"] += 1
            return _report("t", 1.0)

    s = await settle(Task(), f, workdir=tmp_path / "control", timeout_s=5)
    assert n["controls"] == 0 and s.kind == "crucible:node_contention" and s.masks and s.evidence["control"] == "none"


async def test_grade_in_place_keeps_the_reports_of_tries_before_a_later_raise(tmp_path):
    g0 = tmp_path / "a1" / "grade"
    n = {"grades": 0}

    async def hung_then_uninterruptible(d):
        n["grades"] += 1
        if n["grades"] == 1:
            return _masked("t", "test.sh hung", "verifier_timeout")
        raise StageFailure(Failure(kind="crucible:backend_error", stage="verification", retryable=False, blame="unknown", detail="quiesce_uninterruptible"))

    out: list = []
    with pytest.raises(StageFailure):
        await grade_in_place(hung_then_uninterruptible, g0, retries=1, out=out)
    assert n["grades"] == 2 and len(out) == 1 and out[0].infra["failure_kind"] == "verifier_timeout"


# ---------------------------------------------------------------- native, separate verifier: re-graded from the hand-off


class SeparateTask(CustodyTask):
    """``CustodyTask`` (a separate verifier over the policy's ``out/``) whose grade comes back contaminated while
    ``contaminated`` grades remain, then is the custody grade. ``answers``: what each grade read."""

    def __init__(self, seed: Path, *, contaminated: int = 1, name: str = "regrade"):
        super().__init__(seed)
        self.id = f"t/custody/{name}@1"
        self.contaminated = contaminated
        self.answers: list[str | None] = []

    async def grade(self, world, workdir):
        p = Path(world.apps["ws"].state_params["root"]) / "out" / "answer.txt"
        self.answers.append(p.read_text().strip() if p.is_file() else None)
        if self.contaminated > 0:
            self.contaminated -= 1
            self.graded.append(world)
            return _masked(self.id, "perf rounds contaminated: candidate")
        return await super().grade(world, workdir)


def _good() -> tuple[PolicyRef, list[str]]:
    return _counting(lambda a: {"agent": ScriptedPolicy("agent", list(GOOD))})


def _native() -> NativeHarness:
    return NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20))


async def test_native_separate_masked_grade_is_regraded_from_the_handoff(tmp_path, seed):
    task = SeparateTask(seed)
    pol, made = _good()
    run = tmp_path / "runs" / "sep"
    s = await Runner(tmp_path / "runs").run(RunSpec(run_id="sep", tasks=[task], policies=[pol], harnesses=[_native()], k=1, retry_backoff_s=0.0))
    a1 = _adir(run, 1)
    ref = json.loads((a1 / "verifier" / "handoff_ref.json").read_text())
    assert ref["source"] == "a1" and ref["final"]["world"] and ref["log_len"] > 0 and Path(ref["store_root"]).is_dir()
    r1 = json.loads((a1 / "record.json").read_text())
    assert r1["mask_sample"] and r1["failure"]["kind"] == "crucible:node_contention" and r1["failure"]["stage"] == "verification"
    (row,) = _rows(run, "results.jsonl")
    assert row["attempt"].endswith("-a2") and row["reward"] == 1.0 and not row["mask_sample"], row
    assert row["meta"]["stop"] == "regraded" and row["meta"]["artifacts"]["regraded_from"] == "a1"
    assert len(made) == 1 and s.attempts == 2  # the agent ran once: a2 graded the persisted hand-off
    assert len(task.graded) == 2 and task.answers == ["42", "42"] and len(task.opened) == 1
    assert task.graded[1] is not task.opened[0] and task.graded[1].closed  # a fresh verifier World, closed after its grade
    ev = _rows(run, "journal.jsonl")
    (retry,) = [e for e in ev if e["event"] == "retry_scheduled"]
    assert retry["grade_only"] is True
    dispatched = [e for e in ev if e["event"] == "dispatched"]
    assert len(dispatched) == 2 and [bool(e.get("grade_only")) for e in dispatched] == [False, True]
    stops = [json.loads(p.read_text())["meta"]["stop"] for p in sorted((a1.parent).glob("a*/record.json"))]
    assert sum(1 for x in stops if x != "regraded") == 1  # one agent dispatch
    a2 = _adir(run, 2)
    assert json.loads((a2 / "verifier" / "handoff_ref.json").read_text())["source"] == "a1"  # carried on for a later re-grade


async def test_runner_regrade_reaches_a_native_separate_sample(tmp_path, seed):
    task = SeparateTask(seed, contaminated=10**6, name="later")
    pol, made = _good()
    spec = RunSpec(run_id="rg", tasks=[task], policies=[pol], harnesses=[_native()], k=1, retry=RetryRule(max_attempts=1), retry_backoff_s=0.0)
    s = await Runner(tmp_path).run(spec)
    assert s.masked_terminal == 1 and s.mean_reward is None
    task.contaminated = 0  # the node is quiet again
    s2 = await Runner(tmp_path).regrade(spec)
    assert s2.attempts == 1 and s2.scored == 1 and s2.mean_reward == 1.0 and len(made) == 1
    assert task.answers[-1] == "42"


@pytest.mark.parametrize("damage", ["unreadable", "spec", "no_trace", "short_trace"])
async def test_a_bad_handoff_ref_is_a_harness_mask_never_an_agent_rerun(tmp_path, seed, damage):
    task = SeparateTask(seed, contaminated=10**6, name=f"bad-{damage}")
    pol, made = _good()
    spec = RunSpec(run_id=damage, tasks=[task], policies=[pol], harnesses=[_native()], k=1, retry=RetryRule(max_attempts=1), retry_backoff_s=0.0)
    await Runner(tmp_path).run(spec)
    a1 = _adir(tmp_path / damage, 1)
    p = a1 / "verifier" / "handoff_ref.json"
    if damage == "unreadable":
        p.write_text("{not json")
    elif damage == "spec":
        doc = json.loads(p.read_text())
        doc["spec"]["world"] = "t/custody-other"  # a spec that is not the snapshot's world
        p.write_text(json.dumps(doc))
    elif damage == "no_trace":  # review round 1: a pruned trace is never an empty episode (a trace check would pass on nothing)
        (a1 / "trace.jsonl").unlink()
    else:  # shorter than the hand-off's log_len (> 1: the reset event, then the episode's)
        t = a1 / "trace.jsonl"
        t.write_text(t.read_text().splitlines(keepends=True)[0])
    task.contaminated = 0
    s2 = await Runner(tmp_path).regrade(spec)
    row = _rows(tmp_path / damage, "results.jsonl")[-1]
    f = row["failure"]
    assert s2.attempts == 1 and row["mask_sample"] and f["stage"] == "verification" and f["blame"] == "harness" and f["retryable"] is False
    assert f["detail"] == ("handoff_spec_mismatch" if damage == "spec" else "handoff_unreadable"), f
    assert len(made) == 1 and len(task.graded) == 1  # never graded from a ref it cannot trust, never re-sampled


async def test_a_busy_verifier_world_is_infrastructure_regraded_from_the_handoff(tmp_path, seed, monkeypatch):
    """Review round 1: the verifier World's own start (a serial host lock that stayed busy) is classified by boundary as
    transport_timeout, never as this answer's verifier_timeout: the persisted hand-off is re-graded by a fresh attempt,
    and no control is graded (the control takes no World)."""
    from crucible.runtime.harness import native as native_mod

    real = native_mod.verifier_world
    starts = {"n": 0}

    @contextlib.asynccontextmanager
    async def busy_once(world, final, **kw):
        starts["n"] += 1
        if starts["n"] == 1:
            raise TimeoutError("world t/custody is serial and stayed busy")
        async with real(world, final, **kw) as vw:
            yield vw

    monkeypatch.setattr(native_mod, "verifier_world", busy_once)

    class Controlled(SeparateTask):
        controls = 0

        async def grade_control(self, workdir):
            type(self).controls += 1
            return _report(self.id, 1.0)

    task = Controlled(seed, contaminated=0, name="vw-busy")
    pol, made = _good()
    run = tmp_path / "runs" / "busy"
    await Runner(tmp_path / "runs").run(RunSpec(run_id="busy", tasks=[task], policies=[pol], harnesses=[_native()], k=1, retry_backoff_s=0.0))
    a1 = _adir(run, 1)
    r1 = json.loads((a1 / "record.json").read_text())
    f1 = r1["failure"]
    assert r1["mask_sample"] and (f1["kind"], f1["stage"]) == ("transport_timeout", "verification") and f1["retryable"], f1
    assert (a1 / "verifier" / "handoff_ref.json").exists()
    (row,) = _rows(run, "results.jsonl")
    assert row["attempt"].endswith("-a2") and row["reward"] == 1.0 and row["meta"]["stop"] == "regraded", row
    assert len(made) == 1 and Controlled.controls == 0 and task.answers == ["42"]


def _record(d: Path, **doc) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "record.json").write_text(json.dumps(doc))


def _ref(d: Path, source: str) -> None:
    (d / "verifier").mkdir(parents=True, exist_ok=True)
    (d / "verifier" / "handoff_ref.json").write_text(json.dumps({"source": source}))


def test_a_crash_or_timeout_during_a_regrade_still_regrades(tmp_path):
    """Review round 1 (``grading.deciding_record``): an attempt the process died in (moved to ``aborted/`` without a
    record) or a grade-only retry recorded at the agent stage (it carried the hand-off of an earlier attempt) is looked past,
    so the next attempt re-grades instead of running the agent; the journal's ``grade_only`` flag agrees."""
    R = Runner  # noqa: N806

    def at(sdir: Path, n: int) -> Attempt:
        return Attempt(sample=None, id=f"s-a{n}", n=n, dir=sdir / f"a{n}", task=None, policy=None)  # type: ignore[arg-type]

    masked_v = {"mask_sample": True, "failure": {"stage": "verification", "kind": "crucible:node_contention"}}
    # the process died during a2 (the journal moved it to aborted/)
    crash = tmp_path / "crash"
    _record(crash / "a1", **masked_v)
    _ref(crash / "a1", "a1")
    (crash / "a1" / "artifacts").mkdir()
    (crash / "a1" / "artifacts" / "model.patch").write_text("diff --git a/x b/x\n")
    (crash / "aborted" / "a2").mkdir(parents=True)
    (crash / "a3").mkdir()
    assert NativeHarness()._regrade_source(at(crash, 3)) == crash / "a1"
    assert R._grade_only_retry(crash / "a3", 3) is True
    assert WorkspaceHarness()._prior(at(crash, 3)) == ("regrade", crash / "a1")

    # a2 was a grade-only retry the Runner recorded at the agent stage: it carried a1's hand-off
    timed = tmp_path / "timed"
    _record(timed / "a1", **masked_v)
    _ref(timed / "a1", "a1")
    _record(timed / "a2", mask_sample=True, failure={"stage": "agent", "kind": "agent_run_error", "detail": "attempt_timeout"})
    _ref(timed / "a2", "a1")
    (timed / "a3").mkdir()
    assert NativeHarness()._regrade_source(at(timed, 3)) is not None and R._grade_only_retry(timed / "a3", 3) is True

    # an agent-stage mask of an attempt that ran the agent (no carried hand-off): the agent runs again, as before
    agent = tmp_path / "agent"
    _record(agent / "a1", **masked_v)
    _ref(agent / "a1", "a1")
    _record(agent / "a2", mask_sample=True, failure={"stage": "agent", "kind": "transport_unreachable"})
    (agent / "a3").mkdir()
    assert NativeHarness()._regrade_source(at(agent, 3)) is None and R._grade_only_retry(agent / "a3", 3) is False


async def test_the_attempt_clock_during_a_grade_never_reruns_the_agent(tmp_path, seed, monkeypatch):
    """Review round 1: the Runner's attempt clock wraps the in-attempt re-grades and the control too. Firing after the
    agent phase (``Attempt.phase``), it is a verification failure: a persisted hand-off is re-graded by a grade-only retry
    (``Attempt.regradable``), and in place the attempt ends terminal-masked; read as the agent's, it re-sampled the
    policy.

    Review round 2 (``grading.grading_window``): the verification phase opens the grading's own window on that clock (the
    scoring budget defaults to the clock's length; the clock moves past it and the control's), so a grade that hangs
    ends inside the attempt as ``verifier_timeout`` and is settled (here: no control, flags ``not_repeated`` +
    ``no_control``), never cut short as an ``attempt_timeout`` mask. The clock still ends an infrastructure wait (a
    verifier World that never starts, outside the scoring budget), and a persisted hand-off is then re-graded."""

    def no_retry_scheduled(run: Path) -> bool:
        return not [e for e in _rows(run, "journal.jsonl") if e["event"] == "retry_scheduled"]

    # (a) separate verifier, the grade hangs on the answer: the scoring budget (the clock's 3 s) ends it, settled in a1
    class SlowFirst(SeparateTask):
        async def grade(self, world, workdir):
            if not self.answers:
                self.answers.append("slow")
                await asyncio.sleep(60)  # the scoring budget runs out here
            return await super().grade(world, workdir)

    task = SlowFirst(seed, contaminated=0, name="clock")
    pol, made = _good()
    run = tmp_path / "runs" / "clock"
    s = await Runner(tmp_path / "runs").run(RunSpec(run_id="clock", tasks=[task], policies=[pol], harnesses=[_native()], k=1, retry_backoff_s=0.0,
                                                    attempt_timeout_s=3.0))
    (row,) = _rows(run, "results.jsonl")
    f = row["failure"]
    assert row["attempt"].endswith("-a1") and row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "verifier_timeout", "verification"), f
    assert f["retryable"] is False and f["evidence"]["flags"] == ["not_repeated", "no_control"] and f["evidence"]["budget_s"] == 3.0
    assert len(made) == 1 and s.attempts == 1 and no_retry_scheduled(run)

    # (b) in place, the grade hangs: the same, inside the attempt
    class Hangs(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            await asyncio.sleep(60)
            return await super().grade(world, workdir)

    hangs = Hangs("clock")
    pol2, made2 = _oracle()
    s2, rows, _ = await _run_one(tmp_path, "clock-inplace", hangs, NativeHarness("native:scripted"), pol2, retry=RetryRule(max_attempts=3),
                                 attempt_timeout_s=3.0)
    (row2,) = rows
    f2 = row2["failure"]
    assert row2["mask_sample"] and (f2["kind"], f2["detail"], f2["stage"]) == ("verifier_error", "verifier_timeout", "verification") and row2["attempt"].endswith("-a1"), f2
    assert f2["evidence"]["flags"] == ["not_repeated", "no_control"] and f2["retryable"] is False
    assert s2.attempts == 1 and len(made2) == 1 and len(hangs.grades) == 1 and no_retry_scheduled(tmp_path / "clock-inplace")

    # (c) in place, with a control that grades cleanly within that budget: the hang is the answer's, scored 0
    class HangsControlled(Hangs):
        controls = 0

        async def grade_control(self, workdir):
            type(self).controls += 1
            return _report(self.id, 1.0)

    hc = HangsControlled("clock-ctl")
    pol3, made3 = _oracle()
    s3, rows3, _ = await _run_one(tmp_path, "clock-ctl", hc, NativeHarness("native:scripted"), pol3, retry=RetryRule(max_attempts=3), attempt_timeout_s=3.0)
    (row3,) = rows3
    f3 = row3["failure"]
    assert (f3["kind"], f3["detail"]) == ("crucible:policy_error", "grader_failed_on_answer") and row3["reward"] == 0.0 and not row3["mask_sample"], f3
    assert HangsControlled.controls == 1 and s3.attempts == 1 and len(made3) == 1

    # (d) the clock firing while the verifier World never starts (infrastructure, outside the scoring budget): a verification
    # failure on a persisted hand-off, re-graded by a grade-only retry, never an agent re-run
    from crucible.runtime.harness import native as native_mod

    real = native_mod.verifier_world
    starts = {"n": 0}

    @contextlib.asynccontextmanager
    async def stuck_once(world, final, **kw):
        starts["n"] += 1
        if starts["n"] == 1:
            await asyncio.sleep(60)  # the attempt clock fires here (3 s + the 0.5 s control budget + the 3 s margin)
        async with real(world, final, **kw) as vw:
            yield vw

    monkeypatch.setattr(native_mod, "verifier_world", stuck_once)
    task4 = SeparateTask(seed, contaminated=0, name="clock-world")
    pol4, made4 = _good()
    run4 = tmp_path / "runs" / "clock-world"
    harness4 = NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20), control_timeout_s=0.5)
    s4 = await Runner(tmp_path / "runs").run(RunSpec(run_id="clock-world", tasks=[task4], policies=[pol4], harnesses=[harness4], k=1,
                                                     retry_backoff_s=0.0, attempt_timeout_s=3.0))
    r1 = json.loads((_adir(run4, 1) / "record.json").read_text())
    f1 = r1["failure"]
    assert r1["mask_sample"] and (f1["kind"], f1["detail"], f1["stage"]) == ("agent_run_error", "attempt_timeout", "verification") and f1["retryable"], f1
    (row4,) = _rows(run4, "results.jsonl")
    assert row4["attempt"].endswith("-a2") and row4["reward"] == 1.0 and row4["meta"]["stop"] == "regraded", row4
    assert len(made4) == 1 and s4.attempts == 2 and task4.answers == ["42"]
    assert [bool(e.get("grade_only")) for e in _rows(run4, "journal.jsonl") if e["event"] == "dispatched"] == [False, True]


# ---------------------------------------------------------------- native, in place: re-graded inside the attempt


class _Wrapped:
    """A ``HelpdeskTask`` (graded in place) whose grade a test replaces; ``grades``: the dirs it was graded into."""

    world_id = "toy/helpdesk"

    def __init__(self, name: str):
        from crucible_pack_toy.runtime import HelpdeskTask

        self.inner = HelpdeskTask()
        self.id = f"toy/helpdesk/{name}@1"
        self.grades: list[Path] = []

    async def open(self, workdir):
        return await self.inner.open(workdir)

    async def grade(self, world, workdir):
        self.grades.append(Path(workdir))
        return await self.inner.grade(world, workdir)


def _oracle() -> tuple[PolicyRef, list[str]]:
    from crucible_pack_toy.runtime import ORACLE

    return _counting(ORACLE.make)


async def _run_one(tmp_path: Path, name: str, task, harness, pol, **kw):
    s = await Runner(tmp_path).run(RunSpec(run_id=name, tasks=[task], policies=[pol], harnesses=[harness], k=1, retry_backoff_s=0.0, **kw))
    return s, _rows(tmp_path / name, "results.jsonl"), _rows(tmp_path / name, "failures.jsonl")


async def test_in_place_masked_grade_is_regraded_inside_the_attempt_never_re_sampled(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.runtime.inject import FlakyGradeTask

    pol, made = _oracle()
    s, rows, fails = await _run_one(tmp_path, "inplace", FlakyGradeTask(HelpdeskTask(), "perf rounds contaminated: candidate/kinfu"),
                                    NativeHarness("native:scripted"), pol, retry=RetryRule(max_attempts=3))
    assert s.attempts == 1 and len(made) == 1 and len(fails) == 1
    (row,) = rows
    assert row["attempt"].endswith("-a1") and row["mask_sample"] and row["reward"] is None and row["failure"]["kind"] == "crucible:node_contention"
    a1 = _adir(tmp_path / "inplace")
    grades = json.loads((a1 / "verifier" / "grades.json").read_text())
    assert len(grades) == 2 and all(g["infra"]["status"] == "masked" for g in grades)
    assert (a1 / "verifier" / "report.r0.json").exists() and (a1 / "verifier" / "report.r1.json").exists()
    assert not (a1 / "verifier" / "handoff_ref.json").exists()  # nothing persisted: an in-place World is gone after the attempt


async def test_in_place_regrade_recovers_inside_the_attempt(tmp_path):
    class Once(_Wrapped):
        async def grade(self, world, workdir):
            if not self.grades:
                self.grades.append(Path(workdir))
                return _masked(self.id, "judge endpoint down")
            return await super().grade(world, workdir)

    task = Once("once")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "once", task, NativeHarness("native:scripted"), pol)
    (row,) = rows
    assert row["reward"] == 1.0 and not row["mask_sample"] and row["attempt"].endswith("-a1") and s.attempts == 1 and len(made) == 1, row
    a1 = _adir(tmp_path / "once")
    assert task.grades == [a1 / "grade", a1 / "regrade1" / "grade"]
    assert len(json.loads((a1 / "verifier" / "grades.json").read_text())) == 2
    assert json.loads((a1 / "verifier" / "report.json").read_text())["reward"] == 1.0  # report.json is the grade that decided


async def test_non_finite_reward_is_a_masked_verifier_error_not_a_harness_crash(tmp_path):
    class NaN(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            return _report(self.id, float("nan"))

    task = NaN("nan")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "nan", task, NativeHarness("native:scripted"), pol, retry=RetryRule(max_attempts=3))
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "non_finite_reward", "verification") and row["mask_sample"], f
    assert s.attempts == 1 and len(made) == 1 and len(task.grades) == 2  # graded again in place, never an agent re-run


async def test_a_timeout_error_raised_by_the_grader_is_a_verifier_timeout(tmp_path):
    class Hangs(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            raise TimeoutError("test.sh did not finish")

    task = Hangs("hangs")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "hangs", task, NativeHarness("native:scripted"), pol, retry=RetryRule(max_attempts=3))
    (row,) = rows
    f = row["failure"]
    # the grader's, answer-sensitive: never boundary()'s transport_timeout (the infrastructure's, masked and unanswerable)
    assert (f["kind"], f["detail"]) == ("verifier_error", "verifier_timeout") and row["mask_sample"], f
    assert s.attempts == 1 and len(made) == 1 and len(task.grades) == 2


async def test_the_scoring_budget_ends_the_grade_as_verifier_timeout(tmp_path):
    class Slow(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            await asyncio.sleep(5)
            return await super().grade(world, workdir)

    task = Slow("slow")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "slow", task, NativeHarness("native:scripted", scoring_budget_s=0.05), pol, retry=RetryRule(max_attempts=3))
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "verifier_timeout", "verification") and "scoring budget" in f["reason"], f
    assert s.attempts == 1 and len(made) == 1 and len(task.grades) == 1  # the budget is spent: no second try, no crash
    assert not (_adir(tmp_path / "slow") / "verifier" / "report.json").exists()  # no report came back


async def _steps_run(tmp_path: Path, name: str, first, *, once: bool = False, separate: bool = False):
    """A native two-step ``TicketTask`` whose first step is graded by ``first(n, world, dir)`` (``n``: its grade count)
    and whose second step always passes; ``graded``: every grade dir, in order. ``separate``: graded in a fresh verifier
    World per step (``verifier.mode: separate``)."""
    from tests.unit.test_multistep import GOOD as STEPS
    from tests.unit.test_multistep import Counting, TicketTask

    task = TicketTask(separate=separate)
    if once:
        task.step_grades_once = True
    graded: list[str] = []

    async def step1(world, d):
        graded.append(str(d))
        return await first(sum("00-open" in g for g in graded), world, d)

    async def step2(world, d):
        graded.append(str(d))
        return _report(task.id, 1.0)

    filed = task.steps[1].setup  # TicketTask's own setup of the second step: it files T2
    task.steps = [TaskStep("open", "Open ticket T1.", step1), TaskStep("close", "A customer filed T2. Close it.", step2, setup=filed)]
    pol = Counting(STEPS)
    s, rows, _ = await _run_one(tmp_path, name, task, NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=10)), pol.ref(),
                                retry=RetryRule(max_attempts=3))
    return s, rows, graded, pol


async def test_a_native_multi_step_step_is_graded_again_in_place(tmp_path):
    """Review round 1: a native step grade has no per-step state a second grade would use up (only ``step_grades_once``
    tasks do: ``HarborMultiStepTask``), so it is graded again in place like a single-step grade."""

    async def outage(n, world, d):
        return _masked("t", "judge endpoint down")

    s, rows, graded, pol = await _steps_run(tmp_path, "steps-down", outage)
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "check_infra", "verification") and row["mask_sample"], f
    assert len(graded) == 2 and all("00-open" in g for g in graded) and Path(graded[1]).parts[-2:] == ("regrade1", "grade")
    assert f["evidence"]["flags"] == ["no_control"] and s.attempts == 1 and pol.made == 1  # TicketTask declares no control

    async def once_down(n, world, d):
        return _masked("t", "judge endpoint down") if n == 1 else _report("t", 1.0)

    s2, rows2, graded2, pol2 = await _steps_run(tmp_path, "steps-once", once_down)
    (row2,) = rows2
    assert not row2["mask_sample"] and row2["reward"] == 1.0 and s2.attempts == 1 and pol2.made == 1, row2
    assert sum("00-open" in g for g in graded2) == 2 and sum("01-close" in g for g in graded2) == 1  # the step scored, the next ran


async def test_step_grades_once_steps_are_graded_once(tmp_path):
    async def outage(n, world, d):
        return _masked("t", "judge endpoint down")

    s, rows, graded, pol = await _steps_run(tmp_path, "steps-one", outage, once=True)
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "check_infra", "verification") and row["mask_sample"], f
    assert len(graded) == 1 and "00-open" in graded[0]  # the first step, graded once; the second step never ran
    # review round 2: graded once, so it never repeated (not_repeated); a step_grades_once task has no control either
    assert f["evidence"]["flags"] == ["not_repeated", "no_control"] and f["evidence"]["tries"] == 1 and s.attempts == 1 and pol.made == 1


async def test_an_in_place_grade_that_raises_a_transport_error_is_graded_again(tmp_path):
    class Reset(_Wrapped):
        async def grade(self, world, workdir):
            if not self.grades:
                self.grades.append(Path(workdir))
                raise ConnectionResetError("judge connection reset by peer")
            return await super().grade(world, workdir)

    task = Reset("reset")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "reset", task, NativeHarness("native:scripted"), pol, retry=RetryRule(max_attempts=3))
    (row,) = rows
    assert row["reward"] == 1.0 and not row["mask_sample"] and row["attempt"].endswith("-a1") and s.attempts == 1 and len(made) == 1, row
    a1 = _adir(tmp_path / "reset")
    assert task.grades == [a1 / "grade", a1 / "regrade1" / "grade"]


async def test_an_in_place_regrade_that_raises_keeps_the_first_grades_report_and_flags(tmp_path):
    flag = {"rule": "policy_process_killed", "category": "grader_integrity", "evidence": {"pid": 4242}}

    class HungThenStuck(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            if len(self.grades) == 1:
                return GradeReport(task=self.id, tier="full", checks=[], gates={}, reward=0.0, outcome=0.0, trusted=True, elapsed_s=0.0,
                                   infra={"status": "masked", "failure_kind": "verifier_timeout", "reason": "test.sh hung"}, meta={"flags": [flag]})
            raise StageFailure(Failure(kind="crucible:backend_error", stage="verification", retryable=False, blame="unknown",
                                       detail="quiesce_uninterruptible", reason="only uninterruptible processes left"))

    task = HungThenStuck("stuck")
    pol, made = _oracle()
    s, rows, _ = await _run_one(tmp_path, "stuck", task, NativeHarness("native:scripted"), pol, retry=RetryRule(max_attempts=3))
    (row,) = rows
    f = row["failure"]
    assert row["mask_sample"] and (f["detail"], f["blame"], f["retryable"]) == ("quiesce_uninterruptible", "unknown", False), f
    assert s.attempts == 1 and len(made) == 1 and len(task.grades) == 2
    a1 = _adir(tmp_path / "stuck")
    assert json.loads((a1 / "verifier" / "report.json").read_text())["infra"]["failure_kind"] == "verifier_timeout"  # the first grade's
    ro = json.loads((a1 / "rollout.json").read_text())
    assert [x for x in ro.get("flags") or [] if x.get("rule") == "policy_process_killed"] == [flag]  # its flag reached the rollout, once


# ---------------------------------------------------------------- workspace: answer-sensitive failures graded again at once


async def test_workspace_answer_sensitive_failure_is_graded_again_in_the_attempt(tmp_path):
    from crucible_pack_toy.runtime import FibTask, fib_actions

    from crucible.runtime.inject import FlakyGradeTask

    class Once(FibTask):
        id = "toy/workspace/fib-once@1"

        def __init__(self):
            self.patch_grades: list[Path] = []

        async def grade_patch(self, patch, gdir):
            self.patch_grades.append(Path(gdir))
            if len(self.patch_grades) == 1:
                return _masked(self.id, "judge endpoint down")
            return await super().grade_patch(patch, gdir)

    task = Once()
    pol, made = _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})
    s, rows, _ = await _run_one(tmp_path, "ws-once", task, WorkspaceHarness(), pol)
    (row,) = rows
    assert row["reward"] == 1.0 and row["attempt"].endswith("-a1") and s.attempts == 1 and len(made) == 1, row
    a1 = _adir(tmp_path / "ws-once")
    assert task.patch_grades == [a1 / "grade", a1 / "regrade1" / "grade"]

    # an outage on every grade of this patch: decided in the attempt (terminal), never re-graded or re-run by a retry
    pol2, made2 = _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})
    s2, rows2, _ = await _run_one(tmp_path, "ws-down", FlakyGradeTask(FibTask(), "judge endpoint down"), WorkspaceHarness(), pol2,
                                  retry=RetryRule(max_attempts=3))
    (row2,) = rows2
    assert row2["mask_sample"] and row2["failure"]["kind"] == "verifier_error" and row2["failure"]["detail"] == "check_infra"
    assert s2.attempts == 1 and len(made2) == 1 and row2["attempt"].endswith("-a1")


# ---------------------------------------------------------------- Harbor's installed-agent harness, in place


async def test_harbor_in_place_grade_mask_never_reruns_the_agent(tmp_path):
    from tests.unit.test_m10_lab_audit_r1_harbor import Env, FakeTask, _one_call

    class TimeoutTask(FakeTask):
        async def grade(self, world, gdir):
            self.events.append("grade")
            return self.report(0.0, {"status": "masked", "failure_kind": "verifier_timeout", "reason": "test.sh hung"})

    env = Env(tmp_path)
    env.hooks["run"] = _one_call
    s, rows = await env.run(TimeoutTask(env.events), retry=RetryRule(max_attempts=3), retry_backoff_s=0.0)
    assert s.attempts == 1 and env.events.count("run") == 1 and env.events.count("grade") == 2, env.events
    (row,) = rows
    f = row["failure"]
    assert row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "verifier_timeout", "verification"), f
    assert len(json.loads((env.adir("r", row) / "verifier" / "grades.json").read_text())) == 2


async def test_an_in_place_regrade_has_its_own_receipt_and_never_flags_the_first_grades_leftovers(tmp_path, monkeypatch):
    """Critic amendment 9 (R-2 interaction): the re-grade's quiesce receipt is ``<attempt>/regrade1/verifier/quiesce.json``,
    never written over the first grade's; and a server the first grade's test.sh left running is listed before the
    re-grade (``_regrade_leftovers``) and killed as a leftover, never flagged ``policy_process_killed``."""
    pytest.importorskip("harbor.verifier.verifier")
    from tests.unit.test_m10_lab_audit_r2_quiesce import BASE_ROWS, PID1, WorldBackend, _base, _inplace, _Verifier, _world, q, top

    monkeypatch.setattr("harbor.verifier.verifier.Verifier", _Verifier)
    monkeypatch.setattr(q, "PAUSE_S", 0.0)
    old = (5150, PID1, "S", 30, "http.server # FIRST-GRADE")
    # the first grade: clean; before the re-grade: the server it left; the re-grade's quiesce: it, then gone once killed
    be = WorldBackend([top(*BASE_ROWS), top(*BASE_ROWS, old), top(*BASE_ROWS, old), top(*BASE_ROWS)])
    w = _world(be, base=_base())
    t = _inplace()
    first = await t.grade(w, tmp_path / "grade")
    assert first.reward == 1.0 and first.meta["quiesce"]["verdict"] == "clean" and w.__dict__["_inplace_swept"] is True
    again = await t.grade(w, try_dir(tmp_path / "grade", 1))
    assert again.reward == 1.0 and again.meta["quiesce"]["verdict"] == "clean" and again.meta["quiesce"]["killed"] == 1
    assert again.meta["flags"] == []  # the first grade's leftover is not the policy's
    r0 = json.loads((tmp_path / "verifier" / "quiesce.json").read_text())
    r1 = json.loads((tmp_path / "regrade1" / "verifier" / "quiesce.json").read_text())
    assert r0["killed"] == [] and [e["pid"] for e in r1["leftovers"]] == [5150] and [k["pid"] for k in r1["killed"]] == [5150]
