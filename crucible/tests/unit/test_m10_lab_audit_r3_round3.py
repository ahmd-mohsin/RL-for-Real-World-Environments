"""M10 lab audit R-3, review round 3: the gaps the third review found in grade-only retries and the control.

- Harbor multi-step: a step grade whose Verifier started and raised (a reward-file error scores the step 0 and the task
  goes on) leaves no pending guard window across the next step's agent phase (``HarborMultiStepTask._setup_step``
  closes it), so that phase's ``~/.local/bin`` shadow is still ``grader_tampered`` and a pre-placed grader-only tool is
  still hidden as ``grader_tool_preinstalled``.
- The attempt clock: a hold of ``grading.unbudgeted`` gives the clock back the time it took, a control gets its own
  window (``grading.control_window``), and ``WorkspaceHarness`` opens the grading window again once its ``PerfGuard``
  slot is held, so a decided hang reaches its control and is scored 0, never an ``attempt_timeout`` mask.
- An answer-dependent host check (``unbudgeted(answer=True)``, Harbor in place: the grader guard's scan) holds the
  scoring budget at most ``grading.HOLD_CAP_S``: past it the answer's ``verifier_timeout`` is decided and controlled
  before the attempt clock fires.
- A later try the scoring budget's remainder cut is no grade of the answer (``budget_cut_s``, never ``budget_s``): it
  decides nothing and is never controlled.
- ``grade_in_place(on_again=)`` wiring: the attempt clock firing during the infrastructure of a repeat on a persisted
  hand-off (``NativeHarness``) or a stored patch (``WorkspaceHarness``) ends the attempt, never a grade-only retry.
- ``grading.agent_clock``: a multi-step task's later agent phase gets only what the attempt clock had left.

No Docker, no network, no paid calls; one test stands in for Harbor's ``Verifier`` and skips without Harbor.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.runtime import grading
from crucible.runtime.grading import agent_clock, decided, grade_in_place, grading_window, settle, unbudgeted
from crucible.runtime.harness.native import NativeHarness, TaskStep
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import RetryRule
from crucible.runtime.perf_guard import PerfGuard
from tests.unit.test_m10_lab_audit_r3_regrade import SeparateTask, _adir, _counting, _good, _masked, _oracle, _report, _run_one, _Wrapped
from tests.unit.test_m10_lab_audit_r3_round2 import _Ctl, _no_retry_scheduled, _stuck_world_once
from tests.unit.test_verifier_separate import HIDDEN


@pytest.fixture
def seed(tmp_path):
    d = tmp_path / "seed"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_hidden.py").write_text(HIDDEN)
    return d


# ---------------------------------------------------------------- Harbor multi-step: no guard window across an agent phase


@pytest.mark.parametrize("plant", ["shadow", "preplaced"])
async def test_a_step_grade_that_raised_never_judges_the_next_steps_agent_phase_as_its_window(tmp_path, monkeypatch, plant):
    """Step a's Verifier installs ``~/.local/bin/uv`` and writes no reward (``RewardFileNotFoundError``: the step scores 0,
    the task goes on, and its grade window is left pending). Step b's setup closes that window (its advance absorbs step
    a's install). Step b's own agent phase then plants a ``~/.local/bin/pytest`` shadow (``grader_tampered``) or a
    pre-placed grader-only ``uv`` (hidden for the grade, ``grader_tool_preinstalled``). Judged at step b's grade as step
    a's window, the after-grade relaxations let both through and the baseline moved past them: an unearned 1.0."""
    pytest.importorskip("harbor.verifier.verifier")
    from crucible.interop.harbor import grader_guard
    from crucible.interop.harbor import quiesce as q
    from tests.unit.test_m10_lab_audit_r2_quiesce import BASE_ROWS, WorldBackend, _base, _multistep, _world, top

    monkeypatch.setattr(q, "PAUSE_S", 0.0)
    now: dict[str, str] = {}  # the container's docker diff

    async def diff(backend, cid):
        return dict(now)

    async def fetch(backend, cid, path, *, follow=True, limit=64 << 20):
        return b"#!/bin/sh\n", 0o100755  # an executable script

    monkeypatch.setattr(grader_guard, "diff", diff)
    monkeypatch.setattr(grader_guard, "fetch", fetch)

    class RewardFileNotFoundError(Exception):  # Harbor's, matched by name (inplace.REWARD_FILE_ERRORS)
        pass

    class Verifier:
        """Harbor's ``Verifier`` stand-in: step a's test.sh runs the uv installer, then writes no reward; step b's passes."""

        calls = 0

        def __init__(self, **kw):
            pass

        async def verify(self):
            type(self).calls += 1
            if type(self).calls == 1:
                now["/root/.local/bin/uv"] = "A"
                raise RewardFileNotFoundError("reward.txt was not written")
            return SimpleNamespace(rewards={"reward": 1.0}, model_dump_json=lambda: "{}")

    monkeypatch.setattr("harbor.verifier.verifier.Verifier", Verifier)
    t, (a, b) = _multistep(tmp_path)
    t.grader_integrity = "verify"
    w = _world(WorldBackend([top(*BASE_ROWS)]), base=_base())
    surf = grader_guard.GraderSurface(container="c", user=None, path=["/usr/local/bin", "/usr/bin"], home="/root",
                                      names={"pytest": ["/usr/local/bin/pytest"], "uv": [], "sh": [], "bash": [], "env": [], "mv": []},
                                      grader_only_absent=["uv"], files={}, trees={}, hook_dirs=[], sourced=[], needs_network=False, hosts=[],
                                      configs={}, baseline={})
    w.__dict__["_grader_surface"] = surf

    ra = await t._grade_step(a, w, tmp_path / "steps" / "00-a" / "grade")
    assert ra.reward == 0.0 and ra.infra == {"status": "ok"} and ra.meta["failure"]["kind"] == "crucible:RewardFileNotFoundError", ra.meta
    assert "_guard_pending" in w.__dict__  # the started Verifier's window, left for a re-grade of the same work
    await t._setup_step(b, w)
    assert "_guard_pending" not in w.__dict__ and "/root/.local/bin/uv" in surf.baseline  # closed by the step's setup

    if plant == "shadow":
        now["/root/.local/bin/pytest"] = "A"  # step b's agent phase: shadows /usr/local/bin/pytest
        rb = await t._grade_step(b, w, tmp_path / "steps" / "01-b" / "grade")
        assert rb.reward == 0.0 and rb.meta["failure"]["kind"] == "crucible:grader_tampered", rb.meta
        assert "shadows the grader's pytest" in rb.meta["failure"]["reason"] and Verifier.calls == 1
    else:
        now["/usr/local/bin/uv"] = "A"  # step b's agent phase: a uv placed before the grade installs its own
        rb = await t._grade_step(b, w, tmp_path / "steps" / "01-b" / "grade")
        assert rb.reward == 1.0 and Verifier.calls == 2, rb.meta
        assert "grader_tool_preinstalled" in {f["rule"] for f in rb.meta["flags"]}  # hidden for the grade, never baselined
        assert "/usr/local/bin/uv" in surf.rejudge


# ---------------------------------------------------------------- the attempt clock reaches the control


async def test_a_finished_hold_gives_its_time_back_to_the_attempt_clock(tmp_path):
    async with asyncio.timeout(50.0) as clock:
        att = SimpleNamespace(clock=clock, timeout_s=50.0, agent_left=None)
        before = clock.when()

        async def slow_infra(d):
            with unbudgeted():
                await asyncio.sleep(0.2)  # a host lock, a container start, a transfer
            return _report("t", 1.0)

        reps, f = await grade_in_place(slow_infra, tmp_path / "grade", retries=0, budget_s=5.0, attempt=att)
        assert f is None and len(reps) == 1 and clock.when() >= before + 0.19
        before = clock.when()
        reps, f = await grade_in_place(slow_infra, tmp_path / "grade2", retries=0, budget_s=5.0)  # no attempt: the clock is untouched
        assert f is None and clock.when() == before


async def test_infrastructure_inside_an_in_place_grade_never_costs_the_control_its_window(tmp_path):
    """The grading window moves the clock to about 0.3 + 1.0 + 3 s past its opening. A host wait inside the grade (3.5 s,
    unbudgeted) used that up: the answer's hang ran the 0.3 s scoring budget out at about 3.8 s and the control (0.8 s)
    was cut by the clock, a terminal ``attempt_timeout`` mask. The hold now gives its time back to the clock and the
    control opens its own window: the hang is the answer's, scored 0."""

    class SlowHost(_Wrapped):
        controls = 0

        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            with unbudgeted():
                await asyncio.sleep(3.5)  # a serial host lock, a slow quiesce
            await asyncio.sleep(60)  # then the answer hangs the tests
            return await super().grade(world, workdir)

        async def grade_control(self, workdir):
            type(self).controls += 1
            await asyncio.sleep(0.8)  # within its 1.0 s budget
            return _report(self.id, 1.0)

    task = SlowHost("slow-host")
    pol, made = _oracle()
    harness = NativeHarness("native:scripted", scoring_budget_s=0.3, control_timeout_s=1.0)
    s, rows, _ = await _run_one(tmp_path, "slow-host", task, harness, pol, attempt_timeout_s=3.0)
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"]) == ("crucible:policy_error", "grader_failed_on_answer") and row["reward"] == 0.0 and not row["mask_sample"], f
    assert f["evidence"]["control"] == "passed" and f["evidence"]["flags"] == ["not_repeated"] and f["evidence"]["budget_s"] == 0.3
    assert SlowHost.controls == 1 and s.attempts == 1 and len(made) == 1 and len(task.grades) == 1


async def test_a_perf_guard_slot_wait_never_costs_the_control_its_window(tmp_path):
    """A perf workspace task: the grading window opens (about 0.3 + 2.0 + 5 s on the clock), then the host stays busy for
    6.5 s of the ``PerfGuard`` slot's quiet-host wait. The patch hangs the tests (the 0.3 s scoring budget ends it) and
    the control takes 0.9 s: before, the clock fired during it (an ``attempt_timeout`` mask); now the window is opened
    again once the slot is held and the control opens its own, so the hang is scored 0."""
    from crucible_pack_toy.runtime import FibTask, fib_actions

    first: list[float] = []

    def probe(cmd: str) -> tuple[int, str]:
        t = time.monotonic()
        if not first:
            first.append(t)
        if "utilization.gpu" in cmd and t - first[0] < 6.5:
            return 0, "99\n"  # another job's GPU load: wait for a quiet host
        return 1, ""

    class HangsOnAPerfHost(FibTask):
        id = "toy/workspace/fib-perf-hang@1"
        perf = True
        controls = 0

        async def grade_patch(self, patch, gdir):
            await asyncio.sleep(60)  # the patch hangs the tests: the scoring budget ends it
            return await super().grade_patch(patch, gdir)

        async def grade_control(self, workdir):
            type(self).controls += 1
            await asyncio.sleep(0.9)
            return _report(self.id, 1.0)

    guard = PerfGuard(probe=probe, poll_s=0.1, wait_s=60.0)
    pol, made = _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})
    harness = WorkspaceHarness(perf_guard=guard, scoring_budget_s=0.3, control_timeout_s=2.0)
    s, rows, _ = await _run_one(tmp_path, "perf-slot", HangsOnAPerfHost(), harness, pol, attempt_timeout_s=5.0)
    (row,) = rows
    f = row["failure"]
    assert (f["kind"], f["detail"]) == ("crucible:policy_error", "grader_failed_on_answer") and row["reward"] == 0.0 and not row["mask_sample"], f
    assert f["evidence"]["control"] == "passed" and HangsOnAPerfHost.controls == 1 and s.attempts == 1 and len(made) == 1
    assert (_adir(tmp_path / "perf-slot") / "perf_guard.json").exists()  # graded inside the slot


# ---------------------------------------------------------------- an answer-dependent host check holds the budget at most HOLD_CAP_S


async def test_an_answer_dependent_check_past_its_allowance_is_the_answers_timeout(tmp_path, monkeypatch):
    monkeypatch.setattr(grading, "HOLD_CAP_S", 0.05)

    async def scan(d):
        with unbudgeted(answer=True):
            await asyncio.sleep(10)  # one docker cp per file the policy planted
        return _report("t", 1.0)

    reps, f = await asyncio.wait_for(grade_in_place(scan, tmp_path / "grade", retries=1, budget_s=0.05), 2)
    assert reps == [] and f.detail == "verifier_timeout" and f.evidence["budget_s"] == 0.05 and decided(f)

    # within the allowance it is outside the budget, like any infrastructure; the allowance is per grade_in_place call
    monkeypatch.setattr(grading, "HOLD_CAP_S", 1.0)

    async def short_scan(d):
        with unbudgeted(answer=True):
            await asyncio.sleep(0.2)
        return _report("t", 1.0)

    reps, f = await grade_in_place(short_scan, tmp_path / "grade2", retries=0, budget_s=0.05)
    assert f is None and len(reps) == 1


async def test_an_answer_dependent_check_ends_before_the_attempt_clock_and_is_controlled(tmp_path, monkeypatch):
    """The clock (0.2 s) is moved by the grading window to about 0.6 s; the uncapped scan ran past it, and the 0 became a
    terminal ``attempt_timeout`` mask. Capped, the scoring budget ends the scan as the answer's ``verifier_timeout``
    (the clock covers the allowance while it runs) and the control decides it inside the attempt."""
    monkeypatch.setattr(grading, "HOLD_CAP_S", 0.05)
    t_s = 0.2

    async def scan(d):
        with unbudgeted(answer=True):
            await asyncio.sleep(10)
        return _report("t", 1.0)

    ctl = _Ctl()
    async with asyncio.timeout(t_s) as clock:
        att = SimpleNamespace(clock=clock, timeout_s=t_s, agent_left=None)
        budget = grading_window(att, ctl, budget_s=None, control_s=None)
        reps, f = await grade_in_place(scan, tmp_path / "grade", retries=1, budget_s=budget, attempt=att)
        s = await settle(ctl, f, workdir=tmp_path / "control", timeout_s=None, attempt=att)
        assert not clock.expired()
    assert reps == [] and (s.kind, s.detail) == ("crucible:policy_error", "grader_failed_on_answer") and ctl.controls == 1, s


# ---------------------------------------------------------------- a later try cut by the budget's remainder decides nothing


async def test_a_later_grade_cut_by_the_budgets_remainder_is_never_controlled(tmp_path):
    n = {"g": 0}

    async def grade(d):
        n["g"] += 1
        if n["g"] == 1:
            await asyncio.sleep(0.3)  # a stall that ends as the grader's own timeout
            return _masked("t", "test.sh hung", "verifier_timeout")
        await asyncio.sleep(5)  # an honest grade that needs more than the ~0.1 s left
        return _report("t", 1.0)

    reps, f = await grade_in_place(grade, tmp_path / "grade", retries=1, budget_s=0.4)
    assert n["g"] == 2 and len(reps) == 1 and f.detail == "verifier_timeout" and f.evidence["tries"] == 1, f
    assert "budget_s" not in f.evidence and 0 < f.evidence["budget_cut_s"] < 0.4 and not decided(f)
    ctl = _Ctl()
    s = await settle(ctl, f, workdir=tmp_path / "control", timeout_s=5.0)
    assert (s.kind, s.detail, s.masks, s.retryable) == ("verifier_error", "verifier_timeout", True, False), s
    assert s.evidence["control"] == "none" and s.evidence["flags"] == ["not_repeated"] and ctl.controls == 0
    # the same outcome as grading once (grade_retries=0, or a budget spent before the second try started)
    n["g"] = 0
    _reps0, f0 = await grade_in_place(grade, tmp_path / "grade0", retries=0, budget_s=0.4)
    s0 = await settle(ctl, f0, workdir=tmp_path / "control0", timeout_s=5.0)
    assert (s0.kind, s0.detail, s0.evidence["flags"]) == (s.kind, s.detail, s.evidence["flags"]) and ctl.controls == 0


# ---------------------------------------------------------------- on_again: the attempt clock during a repeat's infrastructure


async def test_the_attempt_clock_during_a_repeat_on_a_handoff_ends_the_attempt(tmp_path, seed, monkeypatch):
    """``NativeHarness._on_again``: grade 1 of the persisted hand-off is a masked ``verifier_timeout``, so
    ``attempt.regradable`` is cleared before grade 2; that grade's verifier World never starts (outside the scoring
    budget) and the attempt clock fires: the attempt ends terminal-masked, never a grade-only retry."""
    starts = _stuck_world_once(monkeypatch, at=2)

    class HungOnce(SeparateTask):
        async def grade(self, world, workdir):
            if not self.answers:
                self.answers.append(str(workdir))
                return _masked(self.id, "test.sh hung", "verifier_timeout")
            return await super().grade(world, workdir)

    task = HungOnce(seed, contaminated=0, name="hung-once")
    pol, made = _good()
    harness = NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20), control_timeout_s=0.5)
    s, rows, _ = await _run_one(tmp_path, "hung-once", task, harness, pol, attempt_timeout_s=3.0)
    (row,) = rows
    f = row["failure"]
    assert row["attempt"].endswith("-a1") and row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("agent_run_error", "attempt_timeout", "verification"), f
    assert f["retryable"] is False and starts["n"] == 2 and len(task.answers) == 1
    assert s.attempts == 1 and len(made) == 1 and _no_retry_scheduled(tmp_path / "hung-once")


async def test_the_attempt_clock_during_a_repeat_on_a_stored_patch_ends_the_attempt(tmp_path):
    from crucible_pack_toy.runtime import FibTask, fib_actions

    class HungThenStuck(FibTask):
        """Grade 1 of the patch is a masked ``verifier_timeout``; grade 2's host transfer (``unbudgeted``) never returns."""

        id = "toy/workspace/fib-hung-stuck@1"
        grades = 0

        async def grade_patch(self, patch, gdir):
            if Path(gdir).name != "control":
                type(self).grades += 1
                if type(self).grades == 1:
                    return _masked(self.id, "test.sh hung", "verifier_timeout")
                with unbudgeted():
                    await asyncio.sleep(60)  # the attempt clock fires here
            return await super().grade_patch(patch, gdir)

    pol, made = _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})
    s, rows, _ = await _run_one(tmp_path, "ws-repeat-clock", HungThenStuck(), WorkspaceHarness(control_timeout_s=0.5), pol, attempt_timeout_s=5.0)
    (row,) = rows
    f = row["failure"]
    assert row["attempt"].endswith("-a1") and row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("agent_run_error", "attempt_timeout", "verification"), f
    assert f["retryable"] is False and HungThenStuck.grades == 2 and s.attempts == 1 and len(made) == 1
    assert _no_retry_scheduled(tmp_path / "ws-repeat-clock")


# ---------------------------------------------------------------- agent_clock: a later step's agent phase


async def test_a_later_step_agent_phase_gets_only_what_the_clock_had_left(tmp_path):
    """Step 1's grading window moves the attempt clock past the scoring and control budgets (about 3 + 1800 + 3 s); step
    2's agent phase gets back only what the clock had left (``grading.agent_clock``), so its slow setup is the agent's
    ``attempt_timeout``."""
    from tests.unit.test_multistep import GOOD, Counting, TicketTask

    task = TicketTask()

    async def ok(world, d):
        return _report(task.id, 1.0)

    async def slow(world):
        await asyncio.sleep(10)  # past the ~3 s the clock had left; inside the moved clock without agent_clock

    task.steps = [TaskStep("open", "Open ticket T1.", ok), TaskStep("close", "A customer filed T2. Close it.", ok, setup=slow)]
    pol = Counting(GOOD)
    s, rows, _ = await _run_one(tmp_path, "agent-clock", task, NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=10)), pol.ref(),
                                retry=RetryRule(max_attempts=1), attempt_timeout_s=3.0)
    (row,) = rows
    f = row["failure"]
    assert row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("agent_run_error", "attempt_timeout", "agent"), f
    assert s.attempts == 1 and pol.made == 1
    assert (_adir(tmp_path / "agent-clock") / "steps" / "00-open" / "verifier" / "report.json").exists()  # step 1 was graded: its window opened


async def test_agent_clock_reschedules_to_the_recorded_remainder():
    loop = asyncio.get_running_loop()
    async with asyncio.timeout(50.0) as clock:
        a = SimpleNamespace(clock=clock, timeout_s=3.0, agent_left=None)
        assert grading_window(a, SimpleNamespace(), budget_s=None, control_s=1800.0) == 3.0
        assert a.agent_left is not None and 49.0 < a.agent_left <= 50.0 and clock.when() > loop.time() + 1800
        agent_clock(a)
        assert a.agent_left is None and clock.when() <= loop.time() + 50.0
        before = clock.when()
        agent_clock(a)  # nothing recorded: a no-op
        assert clock.when() == before
