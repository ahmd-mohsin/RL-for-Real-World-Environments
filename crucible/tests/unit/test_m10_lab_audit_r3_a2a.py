"""M10 lab audit R-3, review round 1: the A2A harness grades in place like the native harness (``grading.grade_in_place``
then ``grading.settle``). A non-finite reward is a masked ``verifier_error`` / ``non_finite_reward`` (it raised
``HarnessResult``'s ``allow_inf_nan=False``, escaped ``run()`` and the Runner re-ran the agent as ``agent_run_error``); a
failed grade is graded again before the World closes, and a repeated answer-sensitive failure is decided by the task's
control: one agent dispatch, never a re-sample.

No agentenv: the fake protocol module, the fake agent's ``/ext`` endpoints and the real capture proxy of
``test_m10_lab_audit_r1_a2a`` (its ``proto`` fixture, ``Env`` and ``FakeTask``).
"""

from __future__ import annotations

import json

from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.infra import RetryRule
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import GradeReport
from tests.unit.test_m10_lab_audit_r1_a2a import COMPLETED, Env, FakeTask, proto  # noqa: F401 -- proto is a fixture


def _report(tid: str, reward: float, infra: dict | None = None) -> GradeReport:
    return GradeReport(task=tid, tier="full", checks=[], gates={}, reward=reward, outcome=0.0 if reward != reward else reward,
                       infra=infra or {"status": "ok"}, trusted=True, elapsed_s=0.0)


class NaNTask(FakeTask):
    """Every grade returns a NaN reward with infra ok, or the ``masked`` infra when given; no control."""

    def __init__(self, *, masked: dict | None = None):
        super().__init__()
        self.masked, self.dirs, self.controls = masked, [], 0

    def identity_config(self) -> dict:  # M10 lab audit R-4: the configuration (the injected infra status), never the counters or the dirs
        return {**super().identity_config(), "masked": self.masked}

    async def grade(self, world, workdir):
        self.graded += 1
        self.dirs.append(workdir)
        return _report(self.id, 0.0, self.masked) if self.masked else _report(self.id, float("nan"))


class Controlled(NaNTask):
    """``NaNTask`` with a control that grades cleanly (``controls``: how often it ran)."""

    async def grade_control(self, workdir):
        self.controls += 1
        return _report(self.id, 1.0)


def _completed(p) -> None:
    async def poll():
        return COMPLETED

    p.script["poll"] = poll


async def test_a_nan_reward_is_a_masked_grade_failure_never_an_agent_rerun(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)
    _completed(proto)
    task = NaNTask()
    s = await Runner(tmp_path / "runs").run(RunSpec(run_id="nan", tasks=[task], policies=[PolicyRef(id="a2a:fake", make=lambda a: {}, config={})],
                                                    harnesses=[env.harness()], k=1, retry=RetryRule(max_attempts=3), retry_backoff_s=0.0))
    rows = [json.loads(x) for x in (tmp_path / "runs" / "nan" / "results.jsonl").read_text().splitlines() if x.strip()]
    (row,) = rows
    f = row["failure"]
    assert row["mask_sample"] and row["reward"] is None and (f["kind"], f["detail"], f["stage"]) == ("verifier_error", "non_finite_reward", "verification"), f
    assert f["retryable"] is False and f["evidence"]["flags"] == ["no_control"]
    assert s.attempts == 1 and len(proto.calls["send"]) == 1 and task.graded == 2  # graded twice in place, the agent asked once
    assert [d.parts[-2:] for d in task.dirs] == [(row["attempt"].rsplit("-", 1)[-1], "grade"), ("regrade1", "grade")]
    ev = [json.loads(x) for x in (tmp_path / "runs" / "nan" / "journal.jsonl").read_text().splitlines() if x.strip()]
    assert len([e for e in ev if e["event"] == "dispatched"]) == 1 and not [e for e in ev if e["event"] == "retry_scheduled"]


async def test_a_repeated_nan_reward_scores_zero_when_the_control_passes(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)
    _completed(proto)
    task = Controlled()
    res = await env.run(env.harness(), task)
    f = res.failure
    assert (f.kind, f.detail, f.who) == ("crucible:policy_error", "grader_failed_on_answer", "policy") and not f.masks, f
    assert f.evidence["control"] == "passed" and task.controls == 1 and task.graded == 2 and res.reward is None


async def test_a_repeated_verifier_timeout_is_decided_by_the_control(tmp_path, monkeypatch, proto):  # noqa: F811
    env = Env(tmp_path, monkeypatch)
    _completed(proto)
    task = Controlled(masked={"status": "masked", "failure_kind": "verifier_timeout", "reason": "test.sh hung"})
    res = await env.run(env.harness(), task)
    f = res.failure
    assert (f.kind, f.detail) == ("crucible:policy_error", "grader_failed_on_answer") and f.evidence["status"] == "verifier_timeout", f
    assert task.graded == 2 and task.controls == 1
    assert (tmp_path / "a1" / "verifier" / "grades.json").exists() and (tmp_path / "a1" / "control" / "report.json").exists()
