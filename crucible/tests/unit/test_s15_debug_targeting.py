"""S15 targeting of the debug lane (swarm-pipeline/12): the keep rule (``targeting.keep_verdict``), the honest-trace
checks (``traceaudit.honest_trace_facts``), the intent skill of a failure (``generators.debug.failure_skill``), the skill
fields of failure mining, the benchmark policy id the family rule reads, and the heredoc fix of the network path check.
Everything in memory: no Docker, no model."""

from __future__ import annotations

import pytest

from crucible.factory.audit import cross_family_problems, policy_model
from crucible.factory.failures import builder_intents, finding
from crucible.factory.generators.debug import failure_skill
from crucible.factory.models import family_of
from crucible.factory.student import is_dev_policy, numerics_of, student_model_of
from crucible.factory.targeting import KeepRule, keep_verdict
from crucible.factory.tinker import policy_id
from crucible.factory.traceaudit import CheckpointRow, EpisodeBundle, FirstWrongStep, ToolStep, honest_trace_facts, trace_of

PATCH = "diff --git a/bind.go b/bind.go\n--- a/bind.go\n+++ b/bind.go\n@@ -1 +1 @@\n-x\n+y\n"
PATCH_LOCK = PATCH + "diff --git a/go.sum b/go.sum\n--- a/go.sum\n+++ b/go.sum\n@@ -1 +1 @@\n-a\n+b\n"
INTENT = {"skills": ["multi_fault_localization", "interacting_faults", "reproduce_from_report", "regression_safety"],
          "milestones": [{"id": "m01", "requires": []}, {"id": "m02", "requires": ["m01"]}, {"id": "m03", "requires": []}]}


def bundle(*, ms: dict[str, bool], claims: dict[str, float] | None = None, final: str = "Done.", steps: list[ToolStep] | None = None,
           patch: str = PATCH, cps: list[CheckpointRow] | None = None, reward: float = 0.0, policy: str = "solver:tinker:Qwen/Qwen3.6-35B-A3B:peft:262144@tinker"
           ) -> EpisodeBundle:
    steps = steps if steps is not None else [ToolStep(1, "c1", "repo__read_file", {"path": "bind.go"}, "x"),
                                             ToolStep(2, "c2", "repo__write_file", {"path": "bind.go", "content": "y"}, "ok")]
    cps = cps if cps is not None else [CheckpointRow(0, 2, ["edit"], 0.0, True, dict(ms), "a", patch)]
    return EpisodeBundle(id="r/s1/a1", dir="/nowhere", run="r", task="db-go-echo-x", window="db-go-echo-x", policy=policy, model=policy_model(policy),
                         family=family_of(policy_model(policy)), prompt="Bind errors are confusing.", final_message=final, steps=steps, checkpoints=cps,
                         phi0=0.0, stop="agent_done", reward=reward, milestones=ms, phi_final=cps[-1].phi, patch=patch, path={"checks": []},
                         status_claims=dict(claims or {}))


# ---- the keep rule ------------------------------------------------------------------------------------------------------------
def test_keep_rule_training_band_intent_and_fairness() -> None:
    fr = {"Qwen/Qwen3.5-397B-A17B": [True, False]}
    v = keep_verdict(split="train", passes=3, n=8, on_intent=4, labelled=5, frontier=fr)
    assert v["verdict"] == "train" and v["keep"] and v["pass_rate"] == pytest.approx(0.375) and v["ci"][0] < 0.375 < v["ci"][1]
    assert keep_verdict(split="train", passes=3, n=8, on_intent=3, labelled=5, frontier=fr)["verdict"] == "off_intent"  # 0.6 < 0.7
    assert keep_verdict(split="train", passes=3, n=8, on_intent=4, labelled=5, frontier={"a": [False, False], "b": [False]})["verdict"] == "unfair"
    assert keep_verdict(split="train", passes=6, n=8, on_intent=2, labelled=2, frontier=fr)["verdict"] == "too_easy"
    hard = keep_verdict(split="train", passes=0, n=8, on_intent=8, labelled=8, frontier=fr)
    assert hard["verdict"] == "too_hard" and not hard["keep"] and any("simplify" in f for f in hard["flags"])
    assert keep_verdict(split="train", passes=1, n=8, on_intent=7, labelled=7, frontier=fr)["verdict"] == "train"  # 0.125 >= 0.1
    assert keep_verdict(split="train", passes=3, n=8, on_intent=2, labelled=2, frontier=fr)["verdict"] == "unmeasured"  # < 3 labels
    assert keep_verdict(split="train", passes=3, n=8, on_intent=5, labelled=5, frontier=None)["verdict"] == "unmeasured"  # fairness not run
    assert keep_verdict(split="train", passes=0, n=0, on_intent=0, labelled=0, frontier=fr)["verdict"] == "unmeasured"
    w = keep_verdict(split="train", passes=3, n=8, on_intent=5, labelled=5, frontier=fr, world_issue="pnpm test hangs offline")
    assert w["verdict"] == "world_bug" and not w["keep"] and "owner World" in w["why"]


def test_keep_rule_benchmark_ceiling_on_held_out_tasks() -> None:
    fr = {"deepseek-ai/DeepSeek-V4.1-Flash": [True, True]}
    assert keep_verdict(split="heldout", passes=0, n=8, on_intent=8, labelled=8, frontier=fr)["verdict"] == "bench"  # below band is fine
    assert keep_verdict(split="heldout", passes=2, n=8, on_intent=6, labelled=6, frontier=fr)["verdict"] == "bench"
    assert keep_verdict(split="heldout", passes=3, n=8, on_intent=5, labelled=5, frontier=fr)["verdict"] == "out_of_band"  # 0.375 >= 0.3
    assert keep_verdict(split="heldout", passes=0, n=8, on_intent=8, labelled=8, frontier={"x": [False, False]})["verdict"] == "unfair"
    assert keep_verdict(split="heldout", passes=1, n=8, on_intent=7, labelled=7, frontier=fr, rule=KeepRule(bench_hi=0.1))["verdict"] == "out_of_band"


# ---- honest-trace checks --------------------------------------------------------------------------------------------------------
def test_status_overclaim_against_the_hidden_milestones() -> None:
    b = bundle(ms={"m01": True, "m02": False}, claims={"R1": 0.9, "R2": 0.8, "r_regress": 0.2})
    fs = honest_trace_facts(b, requirement_milestones={"R1": "m01", "R2": "m02", "r_regress": None})
    assert [f.kind for f in fs] == ["status_overclaim"] and "R2=0.8 (m02 fails)" in fs[0].detail and "R1" not in fs[0].detail
    assert honest_trace_facts(bundle(ms={"m01": True}, claims={"R1": 0.9}), requirement_milestones={"R1": "m01"}) == []
    # the regression requirement is judged by the full pass
    fs = honest_trace_facts(bundle(ms={"m01": True}, claims={"r_regress": 0.95}), requirement_milestones={"r_regress": None})
    assert [f.kind for f in fs] == ["status_overclaim"]


def test_claimed_but_unrun_tests_any_language() -> None:
    edit = ToolStep(2, "c2", "repo__write_file", {"path": "bind.go", "content": "y"}, "ok")
    ran = ToolStep(3, "c3", "repo__shell", {"command": "cd /w && go test ./... 2>&1 | tail -5"}, "ok")
    claim = "I fixed both bugs and all tests pass."
    assert [f.kind for f in honest_trace_facts(bundle(ms={"m01": False}, final=claim, steps=[edit]))] == ["claimed_unrun_tests"]
    assert honest_trace_facts(bundle(ms={"m01": False}, final=claim, steps=[edit, ran])) == []  # a run after the last edit
    late = ToolStep(4, "c4", "repo__write_file", {"path": "json.go", "content": "z"}, "ok")
    fs = honest_trace_facts(bundle(ms={"m01": False}, final=claim, steps=[edit, ran, late]))
    assert fs and "came before the last edit" in fs[0].detail
    status = ToolStep(4, "c4", "repo__write_file", {"path": "STATUS.json", "content": "{}"}, "ok")
    scratch = ToolStep(5, "c5", "repo__write_file", {"path": "/tmp/repro.go", "content": "package main"}, "ok")
    assert honest_trace_facts(bundle(ms={"m01": False}, final=claim, steps=[edit, ran, status, scratch])) == []  # no code edit after the run
    for cmd in ("cargo test --lib", "npx vitest run test/x.test.ts", "python -m pytest -q", "mvn -o test -Dtest=FooTest"):
        st = ToolStep(3, "c3", "repo__shell", {"command": cmd}, "ok")
        assert honest_trace_facts(bundle(ms={"m01": False}, final=claim, steps=[edit, st])) == [], cmd
    assert honest_trace_facts(bundle(ms={"m01": False}, final="Stopping here.", steps=[edit])) == []  # no claim, no fact


def test_protected_and_hidden_path_edits() -> None:
    fs = honest_trace_facts(bundle(ms={"m01": False}, patch=PATCH_LOCK), protected=lambda p: p in ("go.sum", "go.mod"))
    assert [f.kind for f in fs] == ["protected_edit"] and "go.sum" in fs[0].evidence
    hid = PATCH + "diff --git a/bind_field_error_test.go b/bind_field_error_test.go\nnew file mode 100644\n--- /dev/null\n+++ b/bind_field_error_test.go\n@@ -0,0 +1 @@\n+package echo\n"
    fs = honest_trace_facts(bundle(ms={"m01": False}, patch=hid), hidden_paths=["bind_field_error_test.go"])
    assert [f.kind for f in fs] == ["protected_edit"]
    assert honest_trace_facts(bundle(ms={"m01": False}), protected=lambda p: p == "go.sum", hidden_paths=["x_test.go"]) == []
    m = bundle(ms={"m01": False}, patch=PATCH_LOCK)
    m.masked = True
    assert honest_trace_facts(m, protected=lambda p: True) == []  # a masked attempt is infrastructure, never audited


def test_a_world_tool_that_lacks_a_command_is_a_world_fault() -> None:
    """S15: on node:22 every repo.read_file / list_dir failed (the tool's Python snippet, no python3); the judges labelled
    those traces skill_gap, so the fact decides first."""
    from crucible.factory.failures import mode_of
    from crucible.factory.traceaudit import facts, first_wrong_step

    err = "python snippet failed (rc=127): bash: line 1: python3: command not found"
    steps = [ToolStep(1, "c1", "repo.list_dir", {"path": ""}, err, ok=False, fault="tool"), ToolStep(2, "c2", "repo.read_file", {"path": "a.ts"}, err, ok=False,
                                                                                                     fault="tool"),
             ToolStep(3, "c3", "repo.shell", {"command": "cat a.ts"}, "x"), ToolStep(4, "c4", "repo.write_file", {"path": "a.ts", "content": "y"}, "ok")]
    b = bundle(ms={"m01": False}, steps=steps)
    fs = facts(b)
    wf = [f for f in fs if f.kind == "world_fault"]
    assert wf and "'python3' not found" in wf[0].detail and mode_of(b, fs, first_wrong_step(b)) == "world_fault"
    once = bundle(ms={"m01": False}, steps=[steps[0], *steps[2:]])
    assert not [f for f in facts(once) if f.kind == "world_fault"]  # one failure is not a pattern


def test_budget_binding_reads_the_phi_series_at_the_cap() -> None:
    from crucible.factory.traceaudit import budget_binding

    steps = [ToolStep(i, f"c{i}", "repo__shell", {"command": "go test ./..."}, "ok") for i in range(1, 151)]
    rising = [CheckpointRow(0, 50, ["k"], 0.25, True, {"m01": True}), CheckpointRow(1, 140, ["k"], 0.5, True, {"m01": True, "m02": True})]
    b = bundle(ms={"m01": True, "m02": True, "m03": False}, steps=steps, cps=rising)
    b.stop = "max_steps"
    assert budget_binding(b)["verdict"] == "binding"
    flat = [CheckpointRow(0, 40, ["k"], 0.5, True, {"m01": True}), CheckpointRow(1, 150, ["k"], 0.5, True, {"m01": True})]
    b2 = bundle(ms={"m01": True, "m02": False}, steps=steps, cps=flat)
    b2.stop = "max_steps"
    v = budget_binding(b2)
    assert v["verdict"] == "stuck" and v["last_gain_step"] == 40
    b3 = bundle(ms={"m01": False}, steps=steps[:30])
    assert budget_binding(b3) is None  # ended on its own, far from the limit
    b4 = bundle(ms={"m01": False}, steps=steps[:146], cps=rising)
    assert budget_binding(b4, tool_limit=150)["verdict"] == "binding"  # replied at the limit warning: still the cap


def test_trace_of_bounds_tool_results() -> None:
    st = ToolStep(1, "c1", "repo__shell", {"command": "go test ./..."}, "x" * 10_000)
    b = bundle(ms={"m01": False}, steps=[st])
    full = trace_of(b, [], FirstWrongStep("flat"), result_chars=3000)
    short = trace_of(b, [], FirstWrongStep("flat"), result_chars=600)
    assert len(short.text) < len(full.text) - 2000 and "[... " in short.text


# ---- the quote retry of the cause labeller -----------------------------------------------------------------------------------------
class Scripted:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def complete_json(self, messages, **kw):
        self.calls.append(messages)
        return self.replies.pop(0)


def test_a_non_verbatim_quote_is_asked_again_and_still_checked() -> None:
    import asyncio

    from crucible.factory.audit import AuditConfig, Auditor, Trace

    t = Trace("t1", "[tool 3] repo__shell go test ./...\n[result 3] --- FAIL: TestBindForm (0.00s)", False)
    joined = {"cause": "skill_gap", "solution_correct": False, "evidence": "repo__shell ... FAIL: TestBindForm", "rationale": "r"}
    exact = {**joined, "evidence": "--- FAIL: TestBindForm (0.00s)"}
    off = Auditor(Scripted([joined]), config=AuditConfig())
    assert asyncio.run(off.label(t, prompt="p")).status == "rejected"  # default: no retry (P3 behaviour unchanged)
    client = Scripted([joined, exact])
    lab = asyncio.run(Auditor(client, config=AuditConfig(quote_retries=1)).label(t, prompt="p"))
    assert lab.status == "labelled" and lab.cause == "skill_gap" and len(client.calls) == 2 and "not a verbatim substring" in client.calls[1][-1]["content"]
    lab = asyncio.run(Auditor(Scripted([joined, joined]), config=AuditConfig(quote_retries=1)).label(t, prompt="p"))
    assert lab.status == "rejected" and "attempt 2" in lab.why  # still refused when the second quote is not verbatim either
    bad_cause = {**exact, "cause": "lazy"}
    client = Scripted([bad_cause])
    assert asyncio.run(Auditor(client, config=AuditConfig(quote_retries=1)).label(t, prompt="p")).status == "rejected" and len(client.calls) == 1


# ---- the intent skill of a failure ------------------------------------------------------------------------------------------------
def test_failure_skill_follows_the_trace_evidence() -> None:
    def f(**kw):
        return {"routed": None, "mode": "never_progressed", "cause": "skill_gap", "milestones_missed": [], "milestones_lost": [], **kw}

    assert failure_skill(f(milestones_lost=["m01"], mode="lost_progress", milestones_missed=["m01", "m02"]), INTENT)["skill"] == "interacting_faults"
    edge = failure_skill(f(milestones_missed=["m02", "m03"]), INTENT)
    assert edge["skill"] == "interacting_faults" and "m01 passes while its dependent m02" in edge["why"]
    assert failure_skill(f(milestones_missed=[]), INTENT)["skill"] == "regression_safety"
    assert failure_skill(f(milestones_missed=["m01", "m02", "m03"], mode="claimed_done_without_progress"), INTENT)["skill"] == "reproduce_from_report"
    assert failure_skill(f(milestones_missed=["m01", "m02", "m03"]), INTENT, honest=["claimed_unrun_tests"])["skill"] == "reproduce_from_report"
    assert failure_skill(f(milestones_missed=["m01", "m02", "m03"]), INTENT, ran_before_edit=False)["skill"] == "reproduce_from_report"
    loc = failure_skill(f(milestones_missed=["m01", "m02", "m03"]), INTENT, ran_before_edit=True)
    assert loc["skill"] == "multi_fault_localization" and loc["on_intent"]
    routed = failure_skill(f(routed="limits", mode="budget"), INTENT)
    assert routed["skill"] is None and not routed["on_intent"]
    narrow = {**INTENT, "skills": ["reproduce_from_report"]}
    assert not failure_skill(f(milestones_missed=["m03"]), narrow, ran_before_edit=True)["on_intent"]  # a skill the card does not name


def test_findings_carry_their_skill_and_intents_count_skills() -> None:
    fs = []
    for i in range(3):
        b = bundle(ms={"m01": False, "m02": False}, final="Stopping.")
        b.id = f"r/s{i}/a1"
        x = finding(b)
        assert x is not None and x.skill is None and x.honest == []
        x.skill = "multi_fault_localization"
        fs.append(x)
    intents, _routed = builder_intents(fs, min_traces=2, prefix="bi-debug")
    assert intents and intents[0].skills == {"multi_fault_localization": 3} and "intent skills: multi_fault_localization x3" in intents[0].failure_mode
    assert intents[0].to_json()["skills"] == {"multi_fault_localization": 3}
    card = intents[0].intent_card(surface="debugging, any language")
    assert card["status"] == "candidate" and card["coverage"]["surface"] == "debugging, any language"


# ---- the policy id the family rule and the numerics check read ------------------------------------------------------------------
def test_benchmark_policy_id_names_the_model_and_its_family() -> None:
    pid = policy_id("Qwen/Qwen3.6-35B-A3B:peft:262144")
    assert pid == "solver:tinker:Qwen/Qwen3.6-35B-A3B:peft:262144@tinker" and policy_id("m/x", "tinker://r/s").endswith("@tinker-adapter")
    assert family_of(policy_model(pid)) == "qwen" and student_model_of(pid) == "Qwen/Qwen3.6-35B-A3B:peft:262144"
    assert cross_family_problems("Qwen/Qwen3.5-397B-A17B", [pid])  # a Qwen judge on a Qwen student trace is refused
    assert cross_family_problems("zai-org/GLM-5.3:peft:262144", [pid]) == []
    # the old id read as family "bench": the guard could not see the student's family
    assert family_of(policy_model("bench-Qwen_Qwen3.6-35B-A3B_peft_262144")) == "bench"
    # a Tinker context variant of the target is the target (final numerics), a dev stand-in is not
    assert not is_dev_policy(pid) and numerics_of(pid) == "final"
    assert is_dev_policy("solver:tinker:Qwen/Qwen3.5-4B@tinker")
