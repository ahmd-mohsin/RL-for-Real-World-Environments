"""M10 runtime: sample identity, journal + resume, budget ledger, aggregates that cannot read a masked sample as 0, and
the exit-gate injections (CPU classes): every infrastructure class masked and retried, every policy control scored 0."""

from __future__ import annotations

from pathlib import Path

import pytest
from crucible_pack_toy import PACK_ROOT
from crucible_pack_toy.runtime import NOOP, ORACLE, FibTask, HelpdeskTask, crashing_tool_world, fib_actions, scripted

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.run import Failure, RunRecord
from crucible.runtime.budget import BudgetLedger, Exhausted
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness
from crucible.runtime.infra import classify_verifier, coverage, mean_reward, pass_at_k
from crucible.runtime.inject import PROVIDER_FAILURES, FakeLLM, FlakyGradeTask, tool_reply
from crucible.runtime.journal import RunJournal
from crucible.runtime.keys import SampleKey, derive_seed
from crucible.runtime.remote import control, lint_self_matching
from crucible.runtime.run import Runner, RunSpec


def rec(reward=None, fail: Failure | None = None, task="t", idx=0) -> RunRecord:
    r = RunRecord(run_id="r", world="w", policy="p", started_at="2026-01-01T00:00:00Z", reward=reward, outcome="completed",
                  sample={"task": task, "policy": "p", "harness": "h", "sample_idx": idx})
    if fail:
        r.mark_failure(fail)
    return r


def test_keys_seeds_and_identity(tmp_path):
    a = SampleKey(task="t", policy="p", harness="h", sample_idx=0, seed=derive_seed(7, "t", "p", "h", 0))
    b = a.model_copy(update={"sample_idx": 1, "seed": derive_seed(7, "t", "p", "h", 1)})
    assert a.id() != b.id() and a.seed != b.seed and a.id() == SampleKey(**a.model_dump()).id()
    RunJournal(tmp_path, "run", "id-1", {}).close()  # M10 lab audit R-4: a journal holds its run's lease until it is closed
    with pytest.raises(ValueError):
        RunJournal(tmp_path, "run", "id-2", {})  # a different run never resumes into this directory


def test_masked_samples_are_not_zeros():
    recs = [rec(1.0, idx=0), rec(0.0, idx=1), rec(fail=Failure(kind="crucible:tool_crashed", stage="agent"), idx=2),
            rec(fail=Failure(kind="crucible:agent_broke_env", stage="agent"), idx=3)]
    assert recs[2].reward is None and recs[2].mask_sample and recs[3].reward == 0.0 and not recs[3].mask_sample
    assert mean_reward(recs) == pytest.approx(1 / 3)  # 3 scored (1, 0, 0); the masked sample is absent, not a zero
    assert coverage(recs, 4) == 0.75 and pass_at_k(recs, 1) == pytest.approx(1 / 3)
    harbor_style = sum(r.reward or 0.0 for r in recs) / len(recs)  # Harbor metrics/base.py: None -> 0
    assert harbor_style == 0.25 != mean_reward(recs)


def test_budget_ledger(tmp_path):
    led = BudgetLedger(tmp_path / "l.sqlite", {"run": 0.01})
    led.reserve("a1", ["run"], 0.006)
    with pytest.raises(Exhausted):
        led.reserve("a2", ["run"], 0.006)
    led.settle("a1", 0.002, uncertain=0.001)
    led.reserve("a2", ["run"], 0.006)
    assert led.summary()["uncertain_run"] == pytest.approx(0.003)
    led2 = BudgetLedger(tmp_path / "l.sqlite", {"run": 0.01})  # crash: a2 was reserved, never settled
    assert led2.uncertain("run") == pytest.approx(0.009)


async def test_run_resume_and_scoring(tmp_path):
    spec = RunSpec(run_id="r", tasks=[HelpdeskTask()], policies=[ORACLE, NOOP], harnesses=[NativeHarness("native:scripted")], k=2, concurrency=2)
    s = await Runner(tmp_path).run(spec)
    assert (s.terminal, s.scored, s.mean_reward, s.attempts) == (4, 4, 0.5, 4)
    s2 = await Runner(tmp_path).run(spec)
    assert s2.attempts == 0 and s2.terminal == 4  # resume: nothing re-runs


def llm_ref(name: str, script_fn, *, fail: str | None = None, on: tuple[int, ...] = (1,), at: int = 0) -> PolicyRef:
    from crucible.actors.policy import LLMToolPolicy

    clients: list[FakeLLM] = []

    def make(attempt):
        llm = FakeLLM(script_fn(), fail_at=(at, PROVIDER_FAILURES[fail]) if fail and attempt.n in on else None)
        clients.append(llm)
        return {"agent": LLMToolPolicy("agent", llm)}

    # M10 lab audit R-4: a PolicyRef declares its configuration (the fake's knobs, never the clients it made)
    return PolicyRef(id=f"fake:{name}", make=make, spend=lambda: (sum(c.spend.usd for c in clients), 0.0), estimate_usd=0.001,
                     config={"name": name, "fail": fail, "on": list(on), "at": at})


def helpdesk_script():
    return [tool_reply("tickets__list_tickets", {"status": "open"}, 0), "All tickets reviewed."]


async def _one(tmp_path, task, policy, harness, run_id="g", **kw):
    s = await Runner(tmp_path).run(RunSpec(run_id=run_id, tasks=[task], policies=[policy], harnesses=[harness], k=1, **kw))
    j = RunJournal(tmp_path, run_id, "", {}) if False else None  # noqa: F841 -- identity is checked by the runner
    import json

    fails = [json.loads(x) for x in (tmp_path / run_id / "failures.jsonl").read_text().splitlines()] if (tmp_path / run_id / "failures.jsonl").exists() else []
    res = [json.loads(x) for x in (tmp_path / run_id / "results.jsonl").read_text().splitlines()] if (tmp_path / run_id / "results.jsonl").exists() else []
    return s, fails, res


async def test_inject_blank_usage_masked_and_retried(tmp_path):
    _s, fails, res = await _one(tmp_path, HelpdeskTask(), llm_ref("blank", helpdesk_script, fail="usage_blank"), NativeHarness())
    assert fails[0]["failure"]["kind"] == "crucible:usage_blank" and fails[0]["reward"] is None and fails[0]["mask_sample"]
    assert len(res) == 1 and res[0]["attempt"].endswith("-a2") and res[0]["mask_sample"] is False  # retried, then scored


async def test_inject_credits_stop_rule_and_resume(tmp_path):
    pol = llm_ref("credits", helpdesk_script, fail="credits")
    s, fails, res = await _one(tmp_path, HelpdeskTask(), pol, NativeHarness())
    assert s.stopped and "credits" in s.stopped and not res and fails[0]["failure"]["kind"] == "crucible:credits_exhausted"
    s2, _, res2 = await _one(tmp_path, HelpdeskTask(), pol, NativeHarness())  # resume after the operator tops up
    assert s2.stopped is None and len(res2) == 1 and res2[0]["attempt"].endswith("-a2")


async def test_ledger_cap_stop_and_increase(tmp_path):
    spec = {"run_id": "cap", "tasks": [HelpdeskTask()], "policies": [llm_ref("x", helpdesk_script)], "harnesses": [NativeHarness()], "k": 3, "concurrency": 1}
    s = await Runner(tmp_path).run(RunSpec(budget_usd=0.0025, **spec))  # room for 2 reservations of 0.001 + settle
    assert s.stopped and s.stopped.startswith("budget") and s.terminal < 3
    s2 = await Runner(tmp_path).run(RunSpec(budget_usd=1.0, **spec))
    assert s2.terminal == 3 and s2.stopped is None


async def test_inject_tool_crash_masked_and_retried(tmp_path):
    task = HelpdeskTask(world_factory=crashing_tool_world(PACK_ROOT, tool="list_tickets"))
    _s, fails, res = await _one(tmp_path, task, llm_ref("crash", helpdesk_script), NativeHarness())
    assert fails[0]["failure"]["kind"] == "crucible:tool_crashed" and fails[0]["reward"] is None
    assert len(res) == 1 and not res[0]["mask_sample"]


@pytest.mark.parametrize(("reason", "kind"), [("perf rounds contaminated: candidate/kinfu", "crucible:node_contention"),
                                              ("stale build: consumers compiled against old headers", "crucible:stale_build"),
                                              ("stale image: verifier digest differs from the pinned one", "crucible:stale_image")])
async def test_inject_verifier_infra_classes(tmp_path, reason, kind):
    import json

    s, fails, res = await _one(tmp_path, FlakyGradeTask(HelpdeskTask(), reason), ORACLE, NativeHarness("native:scripted"))
    assert fails[0]["failure"]["kind"] == kind and fails[0]["reward"] is None
    # masked and never retried by re-running the agent. HelpdeskTask grades in place: the World is gone after the attempt,
    # so a contaminated grade is graded again inside it (regrade1/grade is under /a1/ too, so the injection masks it
    # again) and the attempt ends terminal-masked (M10 lab audit R-3; the old a2 "regrade in a fresh attempt" was a full
    # agent re-run, hidden by the deterministic ORACLE). A stale image or build is never graded again (a retry cannot
    # rebuild it; sweep G2 D3)
    assert res[0]["reward"] is None and res[0]["mask_sample"] and res[0]["attempt"].endswith("-a1") and s.attempts == 1
    adir = next((tmp_path / "g" / "samples").glob("*/a1"))
    if kind == "crucible:node_contention":
        assert len(json.loads((adir / "verifier" / "grades.json").read_text())) == 2
    else:
        assert not (adir / "verifier" / "grades.json").exists()


async def test_inject_grader_exception(tmp_path):
    _s, fails, _res = await _one(tmp_path, FlakyGradeTask(HelpdeskTask(), "reference build failed", crash_setup=True), ORACLE, NativeHarness("native:scripted"))
    f = fails[0]["failure"]
    assert f["kind"] == "verifier_error" and f["detail"] == "grader_exception" and fails[0]["reward"] is None


async def test_inject_empty_collection_then_recollect(tmp_path):
    import os

    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("the injection makes a file unreadable with chmod 000, which root reads anyway")
    acts = fib_actions(extra=[ToolCall(tool_call_id="x", function_name="ws.shell", arguments={"command": "echo s > secret.txt && chmod 000 secret.txt"})])
    _s, fails, res = await _one(tmp_path, FibTask(), scripted("fib-locked", acts), WorkspaceHarness(collect_mode="legacy"))
    assert fails[0]["failure"]["kind"] == "crucible:empty_collection" and fails[0]["failure"]["blame"] is None and fails[0]["reward"] is None
    assert res[0]["attempt"].endswith("-a2") and res[0]["reward"] == 1.0 and res[0]["meta"]["stop"] == "recollected"  # agent not re-run


def test_harness_self_kill_and_lint(tmp_path):
    import sys

    flag = "-a -f" if sys.platform == "darwin" else "-f"  # BSD pkill skips its ancestors unless -a; Linux pkill (the incident) does not
    _, f = control(f"pkill {flag} 'crucible-self-kill-marker-{tmp_path.name}' ; true")
    assert f is not None and f.kind == "crucible:harness_self_kill" and f.masks
    _, ok = control("echo fine")
    assert ok is None
    src = [p for p in (Path(__file__).resolve().parents[2] / "src" / "crucible").rglob("*.py")]
    assert lint_self_matching(src) == []  # no self-matching pkill/pgrep -f in Crucible code


def test_classify_verifier_default():
    f = classify_verifier({"status": "masked", "failure_kind": "check_infra", "reason": "reference build failed"})
    assert f.kind == "verifier_error" and f.detail == "check_infra" and f.masks


# ------------------------------------------------------------------ policy controls: scored 0, never masked


async def test_control_agent_broke_env(tmp_path):
    acts = fib_actions(extra=[ToolCall(tool_call_id="x", function_name="ws.shell", arguments={"command": "rm -rf .git"})])
    _s, fails, res = await _one(tmp_path, FibTask(), scripted("vandal", acts), WorkspaceHarness())
    assert not fails and res[0]["failure"]["kind"] == "crucible:agent_broke_env" and res[0]["reward"] == 0.0 and not res[0]["mask_sample"]


async def test_control_step_limit_and_overflow_scored(tmp_path):
    from crucible.actors.policy import EpisodeLimits

    loop = scripted("loop", [ToolCall(tool_call_id=f"l{i}", function_name="tickets.list_tickets", arguments={"status": "open"}) for i in range(10)])
    _s, fails, res = await _one(tmp_path, HelpdeskTask(), loop, NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=3)), run_id="steps")
    assert not fails and res[0]["meta"]["truncated"] and res[0]["reward"] == 0.0 and not res[0]["mask_sample"]
    _s, fails, res = await _one(tmp_path, HelpdeskTask(), llm_ref("ovf", helpdesk_script, fail="context_overflow", on=(1, 2, 3)), NativeHarness(), run_id="ovf")
    assert not fails and res[0]["meta"]["stop"] == "context_overflow" and res[0]["reward"] == 0.0


async def test_control_no_change_scored_zero(tmp_path):
    _s, fails, res = await _one(tmp_path, FibTask(), scripted("idle", []), WorkspaceHarness())
    assert not fails and res[0]["reward"] == 0.0 and res[0]["meta"]["tree_changed"] is False


def counting(ref: PolicyRef) -> tuple[PolicyRef, list]:
    made: list = []

    def make(a):
        made.append(a.id)
        return ref.make(a)

    return PolicyRef(id=ref.id, make=make, config=ref.config), made  # M10 lab audit R-4: the wrapped policy's own declaration


async def test_masked_grade_retried_as_regrade_of_stored_patch(tmp_path):
    pol, made = counting(scripted("fib", fib_actions()))
    task = FlakyGradeTask(FibTask(), "perf rounds contaminated (spread 0.9)")
    _s, fails, res = await _one(tmp_path, task, pol, WorkspaceHarness())
    assert fails[0]["failure"]["kind"] == "crucible:node_contention" and fails[0]["reward"] is None
    assert res[0]["attempt"].endswith("-a2") and res[0]["reward"] == 1.0 and res[0]["meta"]["stop"] == "regraded"
    assert len(made) == 1  # the agent ran once: the retry graded the stored patch, it did not sample a new solution


async def test_operator_regrade_after_retries_exhausted(tmp_path):
    from crucible.runtime.infra import RetryRule

    pol, made = counting(scripted("fib", fib_actions()))
    flaky = FlakyGradeTask(FibTask(), "node never quiet for perf grading", on_attempts=("a1", "a2", "a3"))
    spec = RunSpec(run_id="rg", tasks=[flaky], policies=[pol], harnesses=[WorkspaceHarness()], k=1, retry=RetryRule(max_attempts=2))
    s = await Runner(tmp_path).run(spec)
    assert s.masked_terminal == 1 and s.mean_reward is None  # nothing scored, nothing read as 0
    flaky.on = ()  # the node is quiet again
    s2 = await Runner(tmp_path).regrade(spec)
    assert s2.attempts == 1 and s2.scored == 1 and s2.mean_reward == 1.0 and len(made) == 1
    s3 = await Runner(tmp_path).regrade(spec)
    assert s3.attempts == 0  # nothing masked is left to re-grade


async def test_perf_guard_waits_then_masks():
    from crucible.runtime.infra import StageFailure
    from crucible.runtime.perf_guard import PerfGuard

    state = {"busy": 2}

    def probe(cmd: str):
        if "compute-apps" in cmd:
            return 0, ("4242, 27446\n" if state["busy"] > 0 else "")
        if "utilization" in cmd:
            state["busy"] -= 1
            return 0, "97\n" if state["busy"] >= 0 else "0\n"
        return 0, "0.10 0.1 0.1 1/100 1\n8\n"

    async def no_sleep(_s):
        return None

    g = PerfGuard(probe=probe, wait_s=3600, poll_s=0, sleep=no_sleep)
    async with g.slot("lab") as info:
        assert len(info["samples"]) == 3 and info["samples"][-1]["foreign_mib"] == 0  # waited for the foreign job to leave
    state["busy"] = 10**6
    g2 = PerfGuard(probe=probe, wait_s=0, poll_s=0, sleep=no_sleep)
    with pytest.raises(StageFailure) as ei:
        async with g2.slot("lab"):
            pass
    f = ei.value.failure
    assert f.kind == "crucible:node_contention" and f.masks and "27446 MiB" in f.reason
