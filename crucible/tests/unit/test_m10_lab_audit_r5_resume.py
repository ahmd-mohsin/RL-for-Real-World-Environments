"""M10 lab audit R-5, item 2 (Acceptance 2): a resume dispatches the samples an outage left masked-terminal, with a fresh retry
budget, and the stop rules count only the current window.

An outage that outlasted the retries ended its samples terminal-masked at the stages before the verdict, and nothing reopened them
(``Runner.regrade`` covers verification only); the resume then seeded ``Health`` with those rows, so the run stopped on the
mask-fraction rule at its first dispatch. ``Runner._reopens`` picks the retryable infra masks of the admission, seed and agent
stages; ``RunJournal.tries(since=)`` counts a reopened sample's attempts from after its terminal one; ``Health`` is not seeded.

Nothing here needs a network, Docker or a paid call.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime.breaker import BreakerRule
from crucible.runtime.harness.base import HarnessResult
from crucible.runtime.infra import RetryRule
from crucible.runtime.run import Runner, RunSpec


@dataclass
class _T:
    id: str = "t@1"


@dataclass
class _P:
    id: str = "p"
    estimate_usd: float = 0.0


class Down:
    """A stateless harness over a service: ``provider_unavailable`` while ``Down.up`` is False, and at the attempt indices in
    ``Down.fail_at``; it scores otherwise."""

    id = "svc@1"
    up = False
    fail_at: tuple[int, ...] = ()

    async def run(self, attempt):
        if not Down.up or attempt.n in Down.fail_at:
            return HarnessResult(failure=Failure(kind="provider_unavailable", stage="agent", reason="HTTP 503"))
        return HarnessResult(reward=1.0, steps=1)


def _events(run_dir) -> list[dict]:
    return [json.loads(x) for x in (run_dir / "journal.jsonl").read_text().splitlines() if x.strip()]


def _rows(run_dir, name: str) -> list[dict]:
    p = run_dir / name
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _rec(failure: Failure | None = None, reward: float = 1.0, attempt: str | None = "s-a3") -> RunRecord:
    rec = RunRecord(run_id="r", world="w", policy="p", started_at="2026-10-19T00:00:00Z", attempt=attempt, reward=reward, outcome="completed")
    if failure is not None:
        rec.mark_failure(failure)
    return rec


@pytest.fixture(autouse=True)
def _reset_down(monkeypatch):
    monkeypatch.setattr(Down, "up", False)
    monkeypatch.setattr(Down, "fail_at", ())


async def test_resume_dispatches_samples_an_outage_left_masked_terminal(tmp_path, monkeypatch):
    spec = RunSpec(run_id="o", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=3, retry_backoff_s=0.0, breaker=None, max_mask_fraction=1.0)
    s1 = await Runner(tmp_path).run(spec)  # the outage outlasts the retries: every sample is masked-terminal after 3 attempts
    assert s1.masked_terminal == 3 and s1.scored == 0 and s1.attempts == 9 and s1.by_failure == {"provider_unavailable": 3}

    # the service is back; the first attempt of each reopened sample (a4) still fails once, and its retry (a5) scores
    monkeypatch.setattr(Down, "up", True)
    monkeypatch.setattr(Down, "fail_at", (4,))
    # the stop rules are operational (identity.DENY): a resume may tighten them. Seeded with the three old masks, 3/3 > 0.2 at
    # min_terminal 1 stopped the run before its first dispatch
    spec2 = RunSpec(run_id="o", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=3, retry_backoff_s=0.0, breaker=None, max_mask_fraction=0.2,
                    min_terminal_for_health=1)
    s2 = await Runner(tmp_path).run(spec2)
    assert s2.stopped is None
    assert s2.scored == 3 and s2.masked_terminal == 0 and s2.terminal == 3 and s2.by_failure == {}
    # a4 failed and was retried as a5: a fresh budget (the three earlier attempts were already spent, and 3 are all a sample gets)
    assert s2.attempts == 6

    run_dir = tmp_path / "o"
    sids = sorted(row[0].id() for row in Runner(tmp_path).plan(spec2))
    resume = [e for e in _events(run_dir) if e["event"] == "resume"]
    assert len(resume) == 1 and resume[0]["reopened"] == sids and resume[0]["done"] == 0 and resume[0]["planned"] == 3
    # the newest terminal row of each sample is the score, under its newest attempt
    results = _rows(run_dir, "results.jsonl")
    assert len(results) == 6 and sum(1 for r in results if r["mask_sample"]) == 3
    assert all(r["attempt"].endswith("-a5") and r["reward"] == 1.0 for r in results[3:])


async def test_a_reopened_sample_gets_one_fresh_budget_each_resume(tmp_path):
    spec = RunSpec(run_id="again", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=1, retry_backoff_s=0.0, breaker=None, max_mask_fraction=1.0)
    s1 = await Runner(tmp_path).run(spec)
    assert s1.attempts == 3 and s1.masked_terminal == 1
    # still down: the sample is reopened and spends exactly one fresh budget (max_attempts=3), not the 1 it would have had
    # counting its earlier attempts (a4 > max_attempts: the old count refused every retry)
    s2 = await Runner(tmp_path).run(spec)
    assert s2.attempts == 3 and s2.masked_terminal == 1 and s2.scored == 0 and s2.by_failure == {"provider_unavailable": 1}
    s3 = await Runner(tmp_path).run(spec)
    assert s3.attempts == 3 and s3.masked_terminal == 1  # counted from its newest terminal attempt (a6), not from a1
    sdir = next((tmp_path / "again" / "samples").iterdir())
    assert sorted(p.name for p in sdir.iterdir() if p.is_dir()) == [f"a{i}" for i in range(1, 10)]
    assert sum(1 for e in _events(tmp_path / "again") if e["event"] == "resume" and e.get("reopened")) == 2


async def test_a_resume_never_reopens_what_the_runs_own_retry_rule_would_not_retry(tmp_path):
    """``Runner._reopens`` is the outage's shape; the run's ``RetryRule`` is its definition of retryable. With a rule that retries only
    node contention, a ``provider_unavailable`` mask is final: a resume must not spend a fresh (paid) attempt on it. The default rule
    is the control: the same masks are reopened."""
    rule = RetryRule(kinds=("crucible:node_contention",))
    spec = RunSpec(run_id="rule", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=3, retry_backoff_s=0.0, breaker=None, max_mask_fraction=1.0, retry=rule)
    s1 = await Runner(tmp_path).run(spec)
    assert s1.masked_terminal == 3 and s1.attempts == 3  # one attempt each: the rule does not retry this kind
    s2 = await Runner(tmp_path).run(spec)
    assert s2.attempts == 0 and s2.masked_terminal == 3 and s2.terminal == 3
    resume = [e for e in _events(tmp_path / "rule") if e["event"] == "resume"]
    assert len(resume) == 1 and "reopened" not in resume[0] and resume[0]["done"] == 3  # they stay done

    ctl = RunSpec(run_id="rule-ctl", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=3, retry_backoff_s=0.0, breaker=None, max_mask_fraction=1.0)
    await Runner(tmp_path).run(ctl)
    c2 = await Runner(tmp_path).run(ctl)
    assert c2.attempts == 9  # the default rule: all three reopened, a fresh budget of three each
    assert len([e for e in _events(tmp_path / "rule-ctl") if e["event"] == "resume" and e.get("reopened")]) == 1


async def test_the_stop_rules_count_only_this_calls_window_and_the_summary_still_reports_every_masked_kind(tmp_path):
    """Health is not seeded with earlier windows (item 2), and ``by_failure`` is the plan's final rows, not ``Health.by_kind``.
    Call 1 leaves three verification-stage masks (never reopened: the policy is not sampled again because a grade failed) and two
    owed samples, stopped by the mask-fraction rule. Call 2 tightens the rule to ``min_terminal 1``: seeded with the three masks it
    would stop at its first dispatch (3/3 > 0.2); counted from its own window it dispatches both samples."""

    class H:
        id = "h@1"

        async def run(self, attempt):
            if attempt.task.id.startswith("bad"):
                return HarnessResult(failure=Failure(kind="verifier_error", stage="verification", blame="task", retryable=False, reason="grader broke"))
            return HarnessResult(reward=1.0, steps=1)

    tasks = [_T(id="bad0"), _T(id="bad1"), _T(id="bad2"), _T(id="good0"), _T(id="good1")]
    s1 = await Runner(tmp_path).run(RunSpec(run_id="win", tasks=tasks, policies=[_P()], harnesses=[H()], k=1, concurrency=1, retry_backoff_s=0.0, breaker=None,
                                            min_terminal_for_health=3, max_mask_fraction=0.2))
    assert s1.stopped is not None and s1.stopped.startswith("health") and s1.terminal == 3 and s1.attempts == 3, s1
    s2 = await Runner(tmp_path).run(RunSpec(run_id="win", tasks=tasks, policies=[_P()], harnesses=[H()], k=1, concurrency=1, retry_backoff_s=0.0, breaker=None,
                                            min_terminal_for_health=1, max_mask_fraction=0.2))
    assert s2.stopped is None, s2.stopped  # this window's two samples both score
    assert s2.attempts == 2 and s2.scored == 2 and s2.terminal == 5
    assert s2.masked_terminal == 3 and s2.by_failure == {"verifier_error": 3}  # the earlier window's masks are still reported


async def test_a_reopened_sample_a_quarantine_holds_is_owed_not_terminal(tmp_path):
    """A reopened sample keeps its older masked row as its newest terminal one; if this call quarantines the key it waits on, the
    summary counts it as held only (not also terminal and masked), and ``run_info`` does not call the run complete."""
    spec = RunSpec(run_id="held", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=5, retry_backoff_s=0.0, breaker=None, max_mask_fraction=1.0)
    s1 = await Runner(tmp_path).run(spec)  # the outage outlasts the retries: five masked-terminal samples
    assert s1.masked_terminal == 5
    # a resume with breakers (operational: ``breaker`` is not part of the identity): the five are reopened, the service is still down,
    # the provider key opens, its probes fail, and it is quarantined with every sample waiting
    spec2 = RunSpec(run_id="held", tasks=[_T()], policies=[_P()], harnesses=[Down()], k=5, concurrency=2, retry_backoff_s=0.0, max_mask_fraction=1.0,
                    breaker=BreakerRule(failures=2, cooldown_s=0.05, quarantine_s=0.1))
    s2 = await Runner(tmp_path).run(spec2)
    assert list(s2.quarantined) == ["provider:p"], s2.quarantined
    assert s2.quarantined_samples == 5 and s2.terminal == 0 and s2.masked_terminal == 0 and s2.by_failure == {}
    assert s2.terminal + s2.quarantined_samples <= s2.planned
    from crucible.bench.metrics import run_info

    info = run_info(tmp_path / "held")
    assert info["complete"] is False and info["quarantined"] == s2.quarantined


def test_only_retryable_infra_masks_before_verification_are_reopened():
    def reopens(f: Failure | None, reward: float = 1.0) -> bool:
        return Runner._reopens(_rec(f, reward))

    assert reopens(Failure(kind="provider_unavailable", stage="agent"))
    assert reopens(Failure(kind="crucible:backend_error", stage="seed"))
    assert reopens(Failure(kind="transport_timeout", stage="admission"))
    # a verification or collect failure is never reopened: a re-dispatch would sample the policy again because a grade (or a
    # collect) failed (R-3); Runner.regrade covers the stored work
    assert not reopens(Failure(kind="crucible:node_contention", stage="verification"))
    assert not reopens(Failure(kind="crucible:env_container_lost", stage="collect"))
    assert not reopens(Failure(kind="provider_unavailable", stage="cleanup"))
    # not an outage: harness- and unknown-blamed failures, and the policy's own
    assert not reopens(Failure(kind="agent_run_error", stage="agent", blame="harness"))
    assert not reopens(Failure(kind="agent_run_error", stage="agent", blame="unknown"))
    assert not reopens(Failure(kind="verifier_error", stage="agent", blame="task"))
    assert not reopens(Failure(kind="crucible:policy_misbehavior", stage="agent"), 0.0)
    # a retry cannot change these
    assert not reopens(Failure(kind="crucible:stale_image", stage="seed"))
    assert not reopens(Failure(kind="crucible:credits_exhausted", stage="agent"))
    assert not reopens(Failure(kind="provider_unavailable", stage="agent", retryable=False))
    # and a scored record has nothing to reopen
    assert not reopens(None)


def test_the_attempt_index_of_a_terminal_row_is_where_its_fresh_budget_starts():
    assert Runner._attempt_index(_rec()) == 3
    assert Runner._attempt_index(_rec(attempt=None)) == 0
    assert Runner._attempt_index(_rec(attempt="no-index")) == 0
    assert Runner._attempt_index(_rec(attempt="sample-a-b-a12")) == 12  # a sample id may itself contain "-a": the last one splits


async def test_a_finished_run_with_policy_zeros_still_dispatches_nothing(tmp_path):
    class Mixed:
        id = "mixed@1"

        async def run(self, attempt):
            if attempt.task.id == "bad":
                return HarnessResult(failure=Failure(kind="crucible:policy_misbehavior", stage="agent", reason="refused the task"))
            return HarnessResult(reward=1.0, steps=1)

    spec = RunSpec(run_id="done", tasks=[_T(id="bad"), _T(id="good")], policies=[_P()], harnesses=[Mixed()], k=1, retry_backoff_s=0.0, breaker=None)
    s1 = await Runner(tmp_path).run(spec)
    assert s1.scored == 2 and s1.masked_terminal == 0 and s1.attempts == 2 and s1.mean_reward == 0.5  # a scored 0 and a scored 1
    s2 = await Runner(tmp_path).run(spec)
    assert s2.attempts == 0 and s2.terminal == 2 and s2.scored == 2 and s2.stopped is None  # a policy's 0 is an answer, not an outage
    resume = [e for e in _events(tmp_path / "done") if e["event"] == "resume"]
    assert len(resume) == 1 and "reopened" not in resume[0] and resume[0]["done"] == 2
