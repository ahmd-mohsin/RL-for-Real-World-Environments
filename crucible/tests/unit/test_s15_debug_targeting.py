"""S15 targeting of the debug lane (swarm-pipeline/12): the keep rule (``targeting.keep_verdict``), the honest-trace
checks (``traceaudit.honest_trace_facts``), the intent skill of a failure (``generators.debug.failure_skill``), the skill
fields of failure mining, the benchmark policy id the family rule reads, and the heredoc fix of the network path check.
Everything in memory: no Docker, no model."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

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
    assert keep_verdict(split="train", passes=3, n=8, on_intent=2, labelled=5, frontier=fr)["verdict"] == "off_intent"  # 0.4 < 0.5
    assert keep_verdict(split="train", passes=1, n=8, on_intent=4, labelled=7, frontier=fr)["verdict"] == "train"  # 4 skill gaps, 3 budget stops: 0.57
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
    # the bar is 0.5, inclusive (the lead raised it from 0.3, 2026-10-27: pass@1 is reported either way)
    assert keep_verdict(split="heldout", passes=3, n=8, on_intent=5, labelled=5, frontier=fr)["verdict"] == "bench"
    assert keep_verdict(split="heldout", passes=4, n=8, on_intent=4, labelled=4, frontier=fr)["verdict"] == "bench"  # 4/8: kept
    five = keep_verdict(split="heldout", passes=5, n=8, on_intent=3, labelled=3, frontier=fr)
    assert five["verdict"] != "bench" and not five["keep"]  # 5/8 = 0.625: not a benchmark task
    assert keep_verdict(split="heldout", passes=5, n=10, on_intent=5, labelled=5, frontier=fr)["verdict"] == "bench"  # 0.5 exactly
    assert keep_verdict(split="heldout", passes=6, n=10, on_intent=4, labelled=4, frontier=fr)["verdict"] == "out_of_band"  # 0.6
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


# ---- S15 at 300 tool calls: the lane's env budget, the h3 World, distinct run ids ------------------------------------------
def _target_script() -> Any:
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_debug_target.py"
    spec = importlib.util.spec_from_file_location("lh_debug_target_t", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_lane_env_budget_is_300_calls_with_scaled_timeouts_and_phi_cadence() -> None:
    from crucible.factory.envs import Timeouts
    from crucible.factory.generators.debug import ENV_BUDGET

    b = ENV_BUDGET
    assert b.tool_limit == 300 and b.decided == "2026-10-20"
    # proportional to the 150-call run (2700 / 4500 s, Phi every 10 calls)
    assert (b.agent_s, b.rollout_s) == (2700.0 * 2, 4500.0 * 2) and b.checkpoint_every == 10 * 2
    Timeouts(agent_s=b.agent_s, rollout_s=b.rollout_s)  # grading fits between the two
    # the periodic Phi points reach the last 30% of the budget, which traceaudit.budget_binding reads (the end has its own slot)
    assert (b.checkpoint_cap - 1) * b.checkpoint_every >= 0.9 * b.tool_limit
    # at 150 calls the cap of 12 stopped the periodic points at call 110: at 300 calls the same cap would stop them at 37%
    assert (12 - 1) * 10 < 0.4 * b.tool_limit


def test_the_targeting_script_runs_under_the_lane_budget_and_records_it() -> None:
    t = _target_script()
    a = t.parse_args(["report"])
    assert (a.tool_limit, a.agent_s, a.rollout_s, a.ckpt_every, a.ckpt_cap, a.seed_offset) == (300, 5400.0, 9000.0, 20, 15, 0)
    eb = t.env_budget(a)
    assert eb["is_lane_budget"] and eb["tool_limit"] == 300 and eb["lane"]["tool_limit"] == 300
    assert not t.env_budget(t.parse_args(["report", "--tool-limit", "150"]))["is_lane_budget"]


def test_a_gates_run_of_some_gates_adds_to_the_record_and_keeps_what_it_replaced() -> None:
    """gates --gates S3,S10 after the cheap ones completes the record; a rerun S10 keeps the failure it replaced."""
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_debug.py"
    spec = importlib.util.spec_from_file_location("lh_debug_t", p)
    lh = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lh)
    order = ("S6", "S5", "S8", "S3", "S10")
    cheap = {"ok": True, "passed": ["S6", "S5", "S8"], "gates": {g: {"ok": True} for g in ("S6", "S5", "S8")}}
    slow = {"ok": True, "passed": ["S3", "S10"], "gates": {"S10": {"ok": True}, "S3": {"ok": True}}, "seconds": 9}
    full = lh.merge_gates(cheap, slow, order)
    assert full["ok"] and full["passed"] == list(order) and list(full["gates"]) == list(order) and "earlier" not in full
    failed = {**full, "ok": False, "passed": list(order[:4]), "gates": {**full["gates"], "S10": {"ok": False, "findings": ["1 in 60"]}}}
    rerun = lh.merge_gates(failed, {"ok": True, "passed": ["S10"], "gates": {"S10": {"ok": True}}}, order)
    assert rerun["ok"] and rerun["earlier"] == [{"gate": "S10", "ok": False, "findings": ["1 in 60"]}]
    crashed = lh.merge_gates(cheap, {"ok": False, "error": "boom", "gates": {}}, order)
    assert not crashed["ok"] and crashed["passed"] == ["S6", "S5", "S8"]
    assert not lh.merge_gates(cheap, {"ok": True, "gates": {"S3": {"ok": None}}}, order)["ok"]  # a gate not run is no pass


def test_a_partial_gate_record_is_measured_only_by_a_quick_pass(tmp_path: Path) -> None:
    """lh_debug.py gates --gates S6,S5,S8 passes a variant to a quick pass; a full run still needs all five gates."""
    t = _target_script()
    rows = [{"id": "full", "gates_ok": True, "gates_run_ok": True}, {"id": "cheap", "gates_ok": False, "gates_run_ok": True},
            {"id": "failed", "gates_ok": False, "gates_run_ok": False}, {"id": "unlinted", "gates_ok": True, "gates_run_ok": True}]
    (tmp_path / "candidates.json").write_text(json.dumps(rows))
    (tmp_path / "authored").mkdir()
    for r in rows:
        (tmp_path / "authored" / f"{r['id']}.json").write_text(json.dumps({"findings": ["x"] if r["id"] == "unlinted" else []}))
    assert [r["id"] for r in t.candidates(tmp_path)] == ["full"]
    assert [r["id"] for r in t.candidates(tmp_path, cheap_gates=True)] == ["full", "cheap"]
    assert not t.parse_args(["student"]).cheap_gates and t.parse_args(["student", "--cheap-gates"]).cheap_gates


def test_node_world_pnpm_is_offline_without_its_pre_run_install() -> None:
    from crucible.factory import langs
    from crucible.factory.langs.node import PNPM_OFFLINE
    from crucible.factory.langs.runtime import policy_world_dockerfile

    node = langs.get("node")
    assert node.recipe == "node-5"
    assert "pnpm_config_verify_deps_before_run=false" in PNPM_OFFLINE and "pnpm_config_offline=true" in PNPM_OFFLINE
    df = policy_world_dockerfile("crucible-lh-d:node-h3-x", node)
    assert df is not None and df.startswith("FROM crucible-lh-d:node-h3-x\n") and f"ENV {PNPM_OFFLINE}\n" in df
    # the recipe sets it after the dependency install (the build itself installs online, once)
    rec = node.dockerfile(Path("."), {"manager": "pnpm", "project": "h3"})
    assert rec.index(PNPM_OFFLINE) > rec.index("pnpm install --frozen-lockfile")


def test_a_spec_defect_sends_the_task_back_to_its_builder_whatever_the_labels() -> None:
    """S15 at 300 calls: every excelize episode (student and frontier) failed one milestone whose table test also checks a
    reverted fix the report never describes; both judges labelled the failures skill gaps (they never see the hidden
    tests). An executed disclosure check records a spec defect, and the keep rule sends the task to its builder."""
    ev = {"split": "heldout", "passes": 0, "n": 8, "on_intent": 8, "labelled": 8, "frontier": {"ds": [True, False]}}
    assert keep_verdict(**ev)["verdict"] == "bench"
    v = keep_verdict(**ev, spec_issue="m04 needs the undisclosed VALUE fault")
    assert v["verdict"] == "spec_defect" and not v["keep"] and "builder" in v["why"]
    assert keep_verdict(**ev, world_issue="pnpm hangs offline", spec_issue="x")["verdict"] == "world_bug"  # the World first


def test_disclosure_check_finds_a_milestone_that_needs_a_silent_fault() -> None:
    """B without the silent unit (a reverted fix with no milestone) is graded through the task's own grader: the milestone
    whose table test also checks that fix fails, so its report must describe the fix."""
    import asyncio
    from types import SimpleNamespace

    from crucible.factory.generators.debug import disclosure_check, silent_units

    units = [{"sha": "u1", "subject": "fix PERCENTILE.EXC"}, {"sha": "u2", "subject": "fix nested func args"}, {"sha": "u3", "subject": "fix COUNT"}]
    window = {"units": units, "milestones": [{"id": "m01", "unit": "u1", "tests": ["a"]}, {"id": "m02", "unit": "u3", "requires": ["m01"], "tests": ["b"]}]}
    graded: list[list[str]] = []

    async def grade(steps: list[dict[str, Any]]) -> tuple[Any, str]:
        got = [s["arguments"]["unit"] for s in steps]
        graded.append(got)
        ms = {"m01": "u1" in got, "m02": "u3" in got and "u2" in got}  # m02's table test also checks u2's behaviour
        return SimpleNamespace(infra={"status": "ok"}, meta={"milestones": ms, "apply_errors": []}), "fp"

    task = SimpleNamespace(task_id="db-go-x", window=window, grade=grade,
                           oracle=[{"function_name": "apply_unit", "arguments": {"unit": u["sha"], "patch": "p"}} for u in units])
    assert [u["sha"] for u in silent_units(window)] == ["u2"]
    r = asyncio.run(disclosure_check(task))
    assert graded == [["u1", "u3"]] and r["measured"] and r["needs"] == {"m02": ["fix nested func args"]}
    # every fix has a milestone: nothing to grade
    w2 = {**window, "units": [units[0], units[2]]}
    assert asyncio.run(disclosure_check(SimpleNamespace(task_id="y", window=w2, grade=grade, oracle=[])))["needs"] == {} and len(graded) == 1
    # a simplified start (the oracle as files): unmeasured, never a verdict
    t3 = SimpleNamespace(task_id="z", window=window, grade=grade, oracle=[{"function_name": "apply_files", "arguments": {"files": {}}}])
    assert asyncio.run(disclosure_check(t3))["measured"] is False


def test_the_committed_summary_seals_held_out_details() -> None:
    t = _target_script()
    s = {"tasks": [{"task": "db-go-x", "split": "heldout", "keep": {"verdict": "spec_defect", "why": "m04 TestHidden needs fix Y", "flags": ["f"]}},
                   {"task": "db-go-y", "split": "train", "keep": {"verdict": "too_hard", "why": "pass rate 0.00", "flags": ["g"]}}],
         "issues": {"db-go-x": {"kind": "spec_defect", "status": "open", "owner": "builder", "issue": "TestHidden checks fix Y", "fix_needed": "z"}}}
    out = t.sealed(s)
    assert "TestHidden" not in json.dumps(out) and out["tasks"][0]["keep"]["verdict"] == "spec_defect"
    assert out["tasks"][1]["keep"]["why"] == "pass rate 0.00" and out["issues"]["db-go-x"] == {"kind": "spec_defect", "status": "open", "owner": "builder"}
    assert "TestHidden" in json.dumps(s)  # the local copy keeps everything


def test_a_re_measurement_never_reuses_an_earlier_phases_run_ids() -> None:
    """The Weave trace key is ``<run id>/<sample>/<attempt>`` and the run id is ``bench-<purpose>-s<seed>-<task>``: a
    second phase on the same tasks and seeds would be skipped as already exported (and its audits attached to the first
    phase's traces). ``--seed-offset`` gives every arm fresh samples and distinct run ids."""
    t = _target_script()
    a = t.parse_args(["report", "--seed-offset", "300"])
    seeds = {t.SEEDS["student"] + a.seed_offset, *(t.SEEDS[m] + a.seed_offset for m in t.FRONTIER)}
    assert seeds.isdisjoint({t.SEEDS["student"], *(t.SEEDS[m] for m in t.FRONTIER)}) and len(seeds) == 3


def test_unknown_task_ids_are_refused_before_any_episode() -> None:
    mod = _target_script()
    assert mod.unknown_tasks(["a", "b-c1-x"], {"a": {}}) == ["b-c1-x"]
    assert mod.unknown_tasks(["a"], {"a": {}}) == []


def test_report_before_mine_is_unmeasured_not_off_intent() -> None:
    mod = _target_script()
    assert mod.unmined([{"final": {"cause": "skill_gap"}}, {"final": {"cause": "skill_gap"}}])
    assert not mod.unmined([{"final": {"cause": "skill_gap"}, "skill": {"on_intent": False}}])
    assert not mod.unmined([])
