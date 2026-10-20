"""M10 lab audit R-3, review round 2: the gaps the second review found in grade-only retries and the control.

- ``RLMHarness`` runs ``NativeHarness`` on the Runner's own ``Attempt`` (the policy swapped in place), so the attempt
  clock reads the phase and the persisted hand-off it set: a clock firing during the grade is never an agent re-run.
- Acceptance check A11 holds the R-3 contract (a masked grade is graded again on the same work, the policy sampled once)
  for a task graded in place.
- The scoring budget counts only the grade (``grading.unbudgeted``): a slow verifier World start, R-2's quiesce, a
  ``HarborDirTask``'s transfers never run it out, so infrastructure is never the answer's ``verifier_timeout``.
- ``grade_in_place(on_again=)``: ``attempt.regradable`` is cleared before an answer-sensitive failure on a persisted
  hand-off or patch is graded again, so the attempt clock firing during that grade's infrastructure ends the attempt
  (never a grade-only retry that grades it twice more; ``test_m10_lab_audit_r3_round3.py``). Separately, the grading
  window on the attempt clock (``grading.grading_window``, exercised in ``test_m10_lab_audit_r3_regrade.py``): a hang on
  the second grade of a persisted hand-off is cut by what was left of the scoring budget (review round 3: no grade of the
  answer, so it stays regradable) and decided by the grade-only retry.
- Only an answer-sensitive failure that repeated is controlled (``evidence.tries``, ``grading.decided``): one after a
  transport failure or a contaminated node is masked ``not_repeated``; a stored patch graded once (``grade_retries=0``)
  is graded again by a grade-only retry (``seen_before``) before any control decides.
- Harbor in place: a re-grade after a Verifier timeout judges the first grade's window first (``_guard_window``), so the
  first grade's own installs are never the policy's tampering.
- A clock firing during the workspace collect (nothing persisted) ends the attempt terminal-masked.
- A separate-verifier multi-step step is graded again inside the attempt (a contaminated node, a verifier World that
  did not start).

No Docker, no network, no paid calls; one test stands in for Harbor's ``Verifier`` and skips without Harbor.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.core.schema.run import Failure
from crucible.runtime.grading import decided, grade_in_place, settle, try_dir, unbudgeted
from crucible.runtime.harness.base import Attempt
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import RetryRule, StageFailure
from crucible.runtime.run import Runner, RunSpec
from tests.unit.test_m10_lab_audit_r3_regrade import (
    SeparateTask,
    _adir,
    _counting,
    _good,
    _masked,
    _native,
    _oracle,
    _report,
    _rows,
    _run_one,
    _steps_run,
    _Wrapped,
)
from tests.unit.test_verifier_separate import HIDDEN


@pytest.fixture
def seed(tmp_path):
    d = tmp_path / "seed"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test_hidden.py").write_text(HIDDEN)
    return d


def _no_retry_scheduled(run: Path) -> bool:
    return not [e for e in _rows(run, "journal.jsonl") if e["event"] == "retry_scheduled"]


def _stuck_world_once(monkeypatch, delay: float = 60.0, at: int = 1):
    """``verifier_world`` whose start number ``at`` hangs ``delay`` seconds before delegating (a serial host lock, an App
    that is never ready); every other start is the real one."""
    from crucible.runtime.harness import native as native_mod

    real = native_mod.verifier_world
    starts = {"n": 0}

    @contextlib.asynccontextmanager
    async def stuck_once(world, final, **kw):
        starts["n"] += 1
        if starts["n"] == at:
            await asyncio.sleep(delay)
        async with real(world, final, **kw) as vw:
            yield vw

    monkeypatch.setattr(native_mod, "verifier_world", stuck_once)
    return starts


# ---------------------------------------------------------------- RLMHarness keeps the Runner's Attempt


def _rlm(**attrs):
    """An ``RLMHarness`` that starts no kernel: scripted policies (``act``, no ``llm``) pass through ``wrap`` unchanged."""
    from crucible.runtime.harness.rlm import RLMConfig, RLMHarness

    h = RLMHarness(RLMConfig(kernel="local", allow_host=True), launcher=lambda a, actor: None, limits=EpisodeLimits(max_steps=20))
    for k, v in attrs.items():
        setattr(h, k, v)
    return h


async def test_an_rlm_harness_runs_on_the_runners_own_attempt(tmp_path, seed):
    task = SeparateTask(seed, contaminated=0, name="rlm-direct")
    pol, made = _good()
    adir = tmp_path / "s" / "a1"
    adir.mkdir(parents=True)
    att = Attempt(sample=None, id="s-a1", n=1, dir=adir, task=task, policy=pol)  # type: ignore[arg-type]
    res = await _rlm().run(att)
    assert res.reward == 1.0 and res.failure is None, res
    # what NativeHarness set is on this Attempt (the one the Runner reads when its clock fires), and its policy is restored
    assert att.phase == "verification" and att.regradable is True and att.policy is pol and len(made) == 1


async def test_the_attempt_clock_during_an_rlm_grade_never_reruns_the_agent(tmp_path, seed, monkeypatch):
    # separate verifier: the clock fires while the verifier World never starts (outside the scoring budget). The persisted
    # hand-off is re-graded by a grade-only retry; through a copy of the Attempt the Runner read phase "agent" and re-ran it
    _stuck_world_once(monkeypatch)
    task = SeparateTask(seed, contaminated=0, name="rlm-clock")
    pol, made = _good()
    run = tmp_path / "runs" / "rlm-clock"
    s = await Runner(tmp_path / "runs").run(RunSpec(run_id="rlm-clock", tasks=[task], policies=[pol], harnesses=[_rlm(control_timeout_s=0.5)], k=1,
                                                    retry_backoff_s=0.0, attempt_timeout_s=3.0))
    r1 = json.loads((_adir(run, 1) / "record.json").read_text())
    f1 = r1["failure"]
    assert r1["mask_sample"] and (f1["kind"], f1["detail"], f1["stage"]) == ("agent_run_error", "attempt_timeout", "verification"), f1
    (row,) = _rows(run, "results.jsonl")
    assert row["attempt"].endswith("-a2") and row["reward"] == 1.0 and row["meta"]["stop"] == "regraded", row
    assert len(made) == 1 and s.attempts == 2
    assert [bool(e.get("grade_only")) for e in _rows(run, "journal.jsonl") if e["event"] == "dispatched"] == [False, True]

    # in place (M10 lab audit R-3 review round 3): an infrastructure wait inside the grade (Harbor's R-2 quiesce, unbudgeted)
    # hangs, so the clock fires after phase "verification" with nothing persisted: the attempt ends terminal-masked at that
    # stage. Through a copy of the Attempt the Runner read phase "agent", retried it and sampled the policy again (a hang on
    # the answer's own grade could not tell them apart: the copy shared the clock, so the scoring budget ended it either way)
    class Hangs(_Wrapped):
        async def grade(self, world, workdir):
            self.grades.append(Path(workdir))
            with unbudgeted():
                await asyncio.sleep(60)  # the attempt clock fires here (0.5 s scoring + 0.5 s control + the 3 s margin)
            return await super().grade(world, workdir)

    hangs = Hangs("rlm-clock")
    pol2, made2 = _oracle()
    s2, rows, _ = await _run_one(tmp_path, "rlm-clock-inplace", hangs, _rlm(scoring_budget_s=0.5, control_timeout_s=0.5), pol2,
                                 retry=RetryRule(max_attempts=3), attempt_timeout_s=3.0)
    (row2,) = rows
    f2 = row2["failure"]
    assert row2["attempt"].endswith("-a1") and row2["mask_sample"] and (f2["kind"], f2["detail"], f2["stage"]) == ("agent_run_error", "attempt_timeout", "verification"), f2
    assert f2["retryable"] is False and s2.attempts == 1 and len(made2) == 1 and len(hangs.grades) == 1
    assert _no_retry_scheduled(tmp_path / "rlm-clock-inplace")


# ---------------------------------------------------------------- A11 on a task graded in place


async def test_a11_holds_the_regrade_contract_for_a_task_graded_in_place(tmp_path):
    from crucible_pack_toy.runtime import HelpdeskTask

    from crucible.accept.checks.extended import a11_harness

    t = SimpleNamespace(runtime_task=HelpdeskTask, oracle=HelpdeskTask().oracle)  # HelpdeskTask grades in place
    out = await a11_harness(t, SimpleNamespace(workdir=tmp_path))  # a11 reads only ctx.workdir
    cases, runs = out.measured["cases"], out.measured["runs"]
    assert cases["contaminated grade masked as node_contention, re-graded on the same work, policy not re-sampled"], runs
    assert cases["grader crash masked as verifier_error, re-graded on the same work, policy not re-sampled"], runs
    assert cases["control: no-op is scored 0, not masked"] and cases["control: oracle is scored 1"], runs
    for name in ("verifier_contention", "grader_crash"):
        assert runs[name]["agent_dispatches"] == 1 and runs[name]["attempts"] == 1, runs[name]
        adir = next((tmp_path / "a11" / name / "samples").glob("*/a1"))
        assert len(json.loads((adir / "verifier" / "grades.json").read_text())) == 2  # graded again inside the attempt


# ---------------------------------------------------------------- the scoring budget counts only the grade


async def test_infrastructure_inside_a_grade_is_outside_the_scoring_budget(tmp_path):
    g0 = tmp_path / "a1" / "grade"

    async def slow_infra(d):
        with unbudgeted():
            await asyncio.sleep(0.3)  # a host lock, a container start, a transfer
        return _report("t", 1.0)

    reps, f = await grade_in_place(slow_infra, g0, retries=1, budget_s=0.05)
    assert f is None and len(reps) == 1

    async def slow_grade(d):
        await asyncio.sleep(0.3)  # the grade itself
        return _report("t", 1.0)

    reps, f = await grade_in_place(slow_grade, g0, retries=1, budget_s=0.05)
    assert reps == [] and (f.detail, f.evidence["tries"], f.evidence["budget_s"]) == ("verifier_timeout", 1, 0.05)

    # the time held is given back to the shared budget: the second try still runs
    n = {"g": 0}

    async def infra_then_outage_once(d):
        n["g"] += 1
        with unbudgeted():
            await asyncio.sleep(0.2)
        return _masked("t", "judge endpoint down") if n["g"] == 1 else _report("t", 1.0)

    reps, f = await grade_in_place(infra_then_outage_once, g0, retries=1, budget_s=0.1)
    assert f is None and n["g"] == 2 and len(reps) == 2

    with unbudgeted(), unbudgeted():  # outside a grade, and nested: no-ops
        pass


async def test_a_slow_verifier_world_start_never_costs_the_answer_its_budget(tmp_path, seed, monkeypatch):
    """Before: the budget ran out while the verifier World started (a serial lock), the answer's grade was relabelled
    ``verifier_timeout``, and a control that needs no verifier World passed and scored a correct answer 0."""
    _stuck_world_once(monkeypatch, delay=0.3)

    class Controlled(SeparateTask):
        controls = 0

        async def grade_control(self, workdir):
            type(self).controls += 1
            return _report(self.id, 1.0)

    task = Controlled(seed, contaminated=0, name="vw-slow")
    pol, made = _good()
    harness = NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20), scoring_budget_s=0.1)
    s, rows, _ = await _run_one(tmp_path, "vw-slow", task, harness, pol)
    (row,) = rows
    assert row["reward"] == 1.0 and not row["mask_sample"] and not row.get("failure"), row
    assert Controlled.controls == 0 and len(made) == 1 and s.attempts == 1 and task.answers == ["42"]


async def test_a_budget_never_preempts_the_r2_quiesce_verdict(tmp_path):
    """Harbor in place: the liveness probe and R-2's quiesce run outside the scoring budget, so its ``uninterruptible``
    verdict (unknown blame) comes out unchanged, never the answer's ``verifier_timeout`` and never a control."""
    from tests.unit.test_m10_lab_audit_r2_quiesce import BASE_ROWS, WorldBackend, _base, _inplace, _world, top

    t = _inplace()

    async def alive(world, tid, t0):
        return None

    async def stuck(world, *, tid, t0, receipt_dir, setup=None, leftovers=None):
        await asyncio.sleep(0.3)  # the D_WAIT rounds
        raise StageFailure(Failure(kind="crucible:backend_error", stage="verification", retryable=False, blame="unknown",
                                   detail="quiesce_uninterruptible", reason="only uninterruptible processes left"))

    t._liveness, t._quiesce = alive, stuck  # type: ignore[method-assign]
    w = _world(WorldBackend([top(*BASE_ROWS)]), base=_base())
    with pytest.raises(StageFailure) as ei:
        await grade_in_place(lambda d: t.grade(w, d), tmp_path / "grade", retries=1, budget_s=0.05)
    assert ei.value.failure.detail == "quiesce_uninterruptible" and ei.value.failure.who == "unknown"


# ---------------------------------------------------------------- on_again: a repeat on a persisted hand-off


async def test_on_again_is_called_before_each_further_grade(tmp_path):
    seen: list[str | None] = []

    async def outage(d):
        return _masked("t", "judge endpoint down")

    reps, f = await grade_in_place(outage, tmp_path / "grade", retries=2, on_again=lambda f: seen.append(f.detail))
    assert seen == ["check_infra", "check_infra"] and len(reps) == 3 and f.evidence["tries"] == 3


async def test_a_hang_on_the_second_grade_of_a_handoff_is_decided_by_the_grade_only_retry(tmp_path, seed):
    """Grade 1 of the persisted hand-off is a masked ``verifier_timeout``, grade 2 hangs and is cut by what was left of the
    scoring budget (the attempt clock's length). Review round 3: a later try the budget's remainder cut is no grade of the
    answer (``budget_cut_s``, never ``budget_s``), so a1 stays regradable; the grade-only retry a2 grades the hand-off
    with ``seen=1``, its first grade runs out the whole budget, and a2 decides it (no control: ``no_control``). The agent
    never runs again."""

    class HungAgain(SeparateTask):
        async def grade(self, world, workdir):
            self.answers.append(str(workdir))
            if len(self.answers) == 1:
                return _masked(self.id, "test.sh hung", "verifier_timeout")
            await asyncio.sleep(60)
            return _report(self.id, 1.0)

    task = HungAgain(seed, contaminated=0, name="hung-again")
    pol, made = _good()
    s, rows, _ = await _run_one(tmp_path, "hung-again", task, _native(), pol, attempt_timeout_s=3.0)
    r1 = json.loads((_adir(tmp_path / "hung-again", 1) / "record.json").read_text())
    f1 = r1["failure"]
    assert r1["mask_sample"] and f1["detail"] == "verifier_timeout" and f1["retryable"] and f1["evidence"]["tries"] == 1, f1
    assert "budget_s" not in f1["evidence"] and 0 < f1["evidence"]["budget_cut_s"] <= 3.0
    (row,) = rows
    f = row["failure"]
    assert row["attempt"].endswith("-a2") and row["mask_sample"] and (f["kind"], f["detail"]) == ("verifier_error", "verifier_timeout"), f
    assert f["retryable"] is False and f["evidence"]["budget_s"] == 3.0 and f["evidence"]["tries"] == 2 and f["evidence"]["control"] == "none"
    assert f["evidence"]["flags"] == ["no_control"] and len(task.answers) == 3
    assert s.attempts == 2 and len(made) == 1


# ---------------------------------------------------------------- only a repeated answer-sensitive failure is controlled


class _Ctl:
    def __init__(self):
        self.controls = 0

    async def grade_control(self, workdir):
        self.controls += 1
        return _report("t", 1.0)


@pytest.mark.parametrize("first", ["reset", "contaminated"])
async def test_one_answer_sensitive_grade_after_an_infra_failure_is_never_controlled(tmp_path, first):
    n = {"g": 0}

    async def grade(d):
        n["g"] += 1
        if n["g"] == 1:
            if first == "reset":
                raise ConnectionResetError("judge connection reset by peer")  # transport_unreachable: graded again in place
            return _masked("t", "perf rounds contaminated: candidate")  # node_contention: graded again in place
        return _masked("t", "test.sh hung", "verifier_timeout")

    reps, f = await grade_in_place(grade, tmp_path / "grade", retries=1)
    assert n["g"] == 2 and (f.kind, f.detail) == ("verifier_error", "verifier_timeout") and f.evidence["tries"] == 1 and not decided(f)
    ctl = _Ctl()
    s = await settle(ctl, f, workdir=tmp_path / "control", timeout_s=5.0)
    assert (s.kind, s.detail, s.masks, s.retryable) == ("verifier_error", "verifier_timeout", True, False), s
    assert s.evidence["control"] == "none" and s.evidence["flags"] == ["not_repeated"] and ctl.controls == 0


@pytest.mark.parametrize("second", ["check_infra", "verifier_timeout"])
async def test_two_answer_sensitive_grades_are_controlled(tmp_path, second):
    n = {"g": 0}

    async def grade(d):
        n["g"] += 1
        if n["g"] == 1 or second == "check_infra":
            return _masked("t", "judge endpoint down")
        return _masked("t", "test.sh hung", "verifier_timeout")

    reps, f = await grade_in_place(grade, tmp_path / "grade", retries=1)
    assert len(reps) == 2 and f.detail == second and f.evidence["tries"] == 2 and decided(f)
    ctl = _Ctl()
    s = await settle(ctl, f, workdir=tmp_path / "control", timeout_s=5.0)
    assert (s.kind, s.detail) == ("crucible:policy_error", "grader_failed_on_answer") and ctl.controls == 1, s
    assert "not_repeated" not in (s.evidence.get("flags") or [])


async def test_a_stored_patch_graded_once_is_graded_again_by_a_grade_only_retry(tmp_path):
    from crucible_pack_toy.runtime import FibTask, fib_actions

    class Blip(FibTask):
        """The judge is down for the first ``outages`` grades of the policy's patch (never for the control's)."""

        id = "toy/workspace/fib-blip@1"

        def __init__(self, outages: int):
            self.outages, self.patch_grades = outages, []

        async def grade_patch(self, patch, gdir):
            if Path(gdir).name != "control":
                self.patch_grades.append(Path(gdir))
                if len(self.patch_grades) <= self.outages:
                    return _masked(self.id, "judge endpoint down")
            return await super().grade_patch(patch, gdir)

    def agent():
        return _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})

    # one outage: graded once (grade_retries=0), so left regradable and graded again on the grade-only retry, scored
    task = Blip(1)
    pol, made = agent()
    s, rows, _ = await _run_one(tmp_path, "blip", task, WorkspaceHarness(grade_retries=0), pol)
    (row,) = rows
    assert row["attempt"].endswith("-a2") and row["reward"] == 1.0 and row["meta"]["stop"] == "regraded" and not row["mask_sample"], row
    r1 = json.loads((_adir(tmp_path / "blip", 1) / "record.json").read_text())
    assert r1["failure"]["detail"] == "check_infra" and r1["failure"]["evidence"]["tries"] == 1 and r1["failure"]["retryable"], r1
    assert len(made) == 1 and s.attempts == 2

    # two outages: the repeat on the grade-only retry is decided there by the control (the oracle patch grades cleanly)
    task2 = Blip(2)
    pol2, made2 = agent()
    s2, rows2, _ = await _run_one(tmp_path, "blip2", task2, WorkspaceHarness(grade_retries=0), pol2)
    (row2,) = rows2
    f2 = row2["failure"]
    assert row2["attempt"].endswith("-a2") and (f2["kind"], f2["detail"]) == ("crucible:policy_error", "grader_failed_on_answer") and row2["reward"] == 0.0, f2
    assert f2["evidence"]["tries"] == 2 and f2["evidence"]["control"] == "passed" and len(made2) == 1 and s2.attempts == 2


# ---------------------------------------------------------------- Harbor in place: the guard's window after a Verifier timeout


async def test_an_in_place_regrade_after_a_verifier_timeout_never_blames_the_first_grades_installs(tmp_path, monkeypatch):
    pytest.importorskip("harbor.verifier.verifier")
    from tests.unit.test_m10_lab_audit_r2_quiesce import BASE_ROWS, WorldBackend, _base, _inplace, _world, top

    from crucible.interop.harbor import grader_guard
    from crucible.interop.harbor import quiesce as q

    monkeypatch.setattr(q, "PAUSE_S", 0.0)
    now: dict[str, str] = {}  # the container's docker diff

    async def diff(backend, cid):
        return dict(now)

    async def fetch(backend, cid, path, *, follow=True, limit=64 << 20):
        return b"#!/bin/sh\n", 0o100755  # an executable script

    monkeypatch.setattr(grader_guard, "diff", diff)
    monkeypatch.setattr(grader_guard, "fetch", fetch)

    class Verifier:
        """Harbor's ``Verifier`` stand-in: the first grade's test.sh runs the uv installer (``~/.local/bin/uv``), then
        pytest hangs; the next grade passes."""

        calls = 0

        def __init__(self, **kw):
            pass

        async def verify(self):
            type(self).calls += 1
            if type(self).calls == 1:
                now["/root/.local/bin/uv"] = "A"
                raise TimeoutError("pytest hung")
            return SimpleNamespace(rewards={"reward": 1.0}, model_dump_json=lambda: "{}")

    monkeypatch.setattr("harbor.verifier.verifier.Verifier", Verifier)

    async def scenario(name: str, *, forget: bool):
        now.clear()
        Verifier.calls = 0
        t = _inplace()
        t.grader_integrity = "verify"
        w = _world(WorldBackend([top(*BASE_ROWS)]), base=_base())
        surf = grader_guard.GraderSurface(container="c", user=None, path=["/usr/local/bin", "/usr/bin"], home="/root",
                                          names={"uv": ["/usr/local/bin/uv"], "sh": [], "bash": [], "env": [], "mv": []}, grader_only_absent=[],
                                          files={}, trees={}, hook_dirs=[], sourced=[], needs_network=False, hosts=[], configs={}, baseline={})
        w.__dict__["_grader_surface"] = surf
        gdir = tmp_path / name / "grade"
        first = await t.grade(w, gdir)
        if forget:  # what a re-grade saw before review round 2: the first grade's window never judged
            w.__dict__.pop("_guard_pending", None)
        again = await t.grade(w, try_dir(gdir, 1))
        return first, again, surf

    first, again, surf = await scenario("judged", forget=False)
    assert first.infra["status"] == "masked" and first.infra["failure_kind"] == "verifier_timeout", first.infra
    assert again.reward == 1.0 and again.infra == {"status": "ok"} and not again.meta.get("failure"), again.meta
    assert not {x["rule"] for x in again.meta.get("flags") or []} & {"grader_tampered", "grader_tool_preinstalled"}
    assert "/root/.local/bin/uv" in surf.baseline  # the baseline moved past the first grade's window

    _, blamed, _ = await scenario("unjudged", forget=True)
    assert blamed.reward == 0.0 and blamed.meta["failure"]["kind"] == "crucible:grader_tampered"


# ---------------------------------------------------------------- the attempt clock during collect


async def test_the_attempt_clock_during_collect_never_reruns_the_agent(tmp_path):
    """The workspace harness's agent phase is over (``collect``) but nothing is persisted yet (the patch is stored by
    ``_grade``): a clock firing here (a World close that hangs) ends the attempt terminal-masked. Retried, the next
    attempt found no stored patch and ran the agent again."""
    from crucible_pack_toy.runtime import FibTask, fib_actions

    class SlowClose(FibTask):
        id = "toy/workspace/fib-slowclose@1"

        async def open(self, workdir):
            w = await super().open(workdir)
            close = w.close

            async def slow_close():
                try:
                    await asyncio.sleep(60)  # the attempt clock fires here
                finally:
                    await close()

            w.close = slow_close
            return w

    pol, made = _counting(lambda a: {"agent": ScriptedPolicy("agent", fib_actions())})
    s, rows, _ = await _run_one(tmp_path, "ws-collect-clock", SlowClose(), WorkspaceHarness(), pol, retry=RetryRule(max_attempts=3), attempt_timeout_s=5.0)
    (row,) = rows
    f = row["failure"]
    assert row["attempt"].endswith("-a1") and row["mask_sample"] and (f["kind"], f["detail"], f["stage"]) == ("agent_run_error", "attempt_timeout", "collect"), f
    assert f["retryable"] is False and s.attempts == 1 and len(made) == 1 and _no_retry_scheduled(tmp_path / "ws-collect-clock")


# ---------------------------------------------------------------- a separate-verifier step is graded again in the attempt


async def test_a_separate_verifier_step_is_graded_again_after_a_contaminated_grade(tmp_path):
    async def contaminated_once(n, world, d):
        return _masked("t", "perf rounds contaminated: candidate") if n == 1 else _report("t", 1.0)

    s, rows, graded, pol = await _steps_run(tmp_path, "steps-sep", contaminated_once, separate=True)
    (row,) = rows
    assert not row["mask_sample"] and row["reward"] == 1.0 and s.attempts == 1 and pol.made == 1, row
    assert sum("00-open" in g for g in graded) == 2 and sum("01-close" in g for g in graded) == 1
    assert Path([g for g in graded if "00-open" in g][1]).parts[-2:] == ("regrade1", "grade")


async def test_a_separate_verifier_step_whose_world_did_not_start_is_graded_again(tmp_path, monkeypatch):
    from crucible.runtime.harness import native as native_mod

    real = native_mod.verifier_world
    starts = {"n": 0}

    @contextlib.asynccontextmanager
    async def busy_once(world, final, **kw):
        starts["n"] += 1
        if starts["n"] == 1:
            raise TimeoutError("world t/tickets is serial and stayed busy")  # boundary: transport_timeout, infra
        async with real(world, final, **kw) as vw:
            yield vw

    monkeypatch.setattr(native_mod, "verifier_world", busy_once)

    async def passes(n, world, d):
        return _report("t", 1.0)

    s, rows, graded, pol = await _steps_run(tmp_path, "steps-sep-busy", passes, separate=True)
    (row,) = rows
    assert row["reward"] == 1.0 and not row["mask_sample"] and s.attempts == 1 and pol.made == 1, row
    assert starts["n"] == 3 and len(graded) == 2  # the first step: refused once, then started and graded; the second: once
