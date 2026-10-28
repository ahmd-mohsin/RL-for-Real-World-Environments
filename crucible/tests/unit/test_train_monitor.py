"""The training monitor (factory.train_monitor, observe.train_monitor): general modes from traces, general vs task-specific,
trends and alerts, proposals and intents (training split only), the capped judge, and ingest of a Runner attempt."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.factory import train_monitor as tm
from crucible.factory.traceaudit import CheckpointRow, EpisodeBundle, ToolStep


def bundle(**kw) -> EpisodeBundle:
    base = {"id": "r/s/a1", "dir": "/x", "run": "r", "task": "db-go-carbon-93371b6f", "window": "db-go-carbon-93371b6f", "policy": "tinker:Qwen/Qwen3.6-35B-A3B",
            "model": "Qwen/Qwen3.6-35B-A3B", "family": "qwen", "patch": "diff --git a/x.go b/x.go\n+x\n"}
    base.update(kw)
    return EpisodeBundle(**base)


def shell(i: int, cmd: str, result: str = "", ok: bool = True) -> ToolStep:
    return ToolStep(i, f"c{i}", "shell", {"command": cmd}, result=result, ok=ok)


def edit(i: int, path: str = "x.go") -> ToolStep:
    return ToolStep(i, f"c{i}", "write_file", {"path": path, "content": "y"})


def ep(task: str, *, lang: str = "go", lane: str = "debug", modes: dict | None = None, passed: bool = False, split: str = "train", step: int | None = None,
       hack: bool = False, reward: float | None = None, brier: float | None = None, i: int = 0) -> tm.Episode:
    return tm.Episode(id=f"{task}/{i}", dir="/x", task=task, lane=lane, lang=lang, split=split, cohort="c", step=step, policy="tinker:Qwen/Qwen3.6-35B-A3B",
                      family="qwen", masked=False, passed=passed, reward=reward if reward is not None else (1.0 if passed else 0.0), phi0=0.0,
                      phi_final=1.0 if passed else 0.5, stop="agent_done", tool_calls=50, compactions=1, milestones={"m01": True, "m02": passed},
                      modes=dict(modes or {}), hack=hack, brier=brier)


# ---- modes of one bundle ----
def test_a_partial_fix_that_stops_by_itself_is_a_stall_but_a_limit_stop_is_not():
    b = bundle(steps=[edit(1)], milestones={"m01": True, "m02": False}, stop="agent_done", reward=0.25)
    assert "stalls_after_partial_fix" in tm.modes_of(b)
    b2 = bundle(steps=[edit(1)], milestones={"m01": True, "m02": False}, stop="max_steps", reward=0.25)
    assert "stalls_after_partial_fix" not in tm.modes_of(b2)


def test_a_late_first_edit_is_slow_and_an_early_one_is_not():
    late = bundle(steps=[shell(i, "rg foo") for i in range(1, 60)] + [edit(60)], milestones={"m01": False}, stop="agent_done", reward=0.0)
    assert "slow_first_edit" in tm.modes_of(late)
    early = bundle(steps=[edit(1)] + [shell(i, "go test ./...") for i in range(2, 60)], milestones={"m01": False}, stop="agent_done", reward=0.0)
    assert "slow_first_edit" not in tm.modes_of(early)


def test_the_same_failing_command_three_times_without_an_edit_is_a_loop_and_an_edit_resets_it():
    loop = bundle(steps=[shell(i, "go test ./...", "--- FAIL: TestX") for i in range(1, 4)], reward=0.0)
    assert "loops_failing_command" in tm.modes_of(loop)
    broken = bundle(steps=[shell(1, "go test ./...", "--- FAIL"), shell(2, "go test ./...", "--- FAIL"), edit(3), shell(4, "go test ./...", "--- FAIL")],
                    reward=0.0)
    assert "loops_failing_command" not in tm.modes_of(broken)


def test_status_claims_the_grader_verified_false_are_overclaims():
    b = bundle(steps=[edit(1)], milestones={"m01": True, "m02": False}, reward=0.25)
    assert "overclaims_status" in tm.modes_of(b, honesty={"scored": {"R1": [0.9, 1], "R2": [0.9, 0]}, "defaulted": []})
    assert "overclaims_status" not in tm.modes_of(b, honesty={"scored": {"R1": [0.9, 1], "R2": [0.2, 0]}, "defaulted": []})


def test_a_regression_counts_only_when_the_episode_moved():
    moved = bundle(steps=[edit(1)], milestones={"m01": True, "m02": False}, reward=0.25)
    assert "breaks_regressions" in tm.modes_of(moved, checks={"c_regress": "fail"})
    still = bundle(steps=[edit(1)], milestones={"m01": False}, phi0=0.0, phi_final=0.0, reward=0.0)
    assert "breaks_regressions" not in tm.modes_of(still, checks={"c_regress": "fail"})


def test_phi_below_the_pre_compaction_peak_right_after_a_compaction_is_lost_progress():
    cps = [CheckpointRow(0, 20, ["periodic"], 0.5, True, {}), CheckpointRow(1, 30, ["compaction"], 0.5, True, {}),
           CheckpointRow(2, 40, ["periodic"], 0.0, True, {})]
    assert "lost_after_compaction" in tm.modes_of(bundle(steps=[edit(1)], checkpoints=cps, phi0=0.0, reward=0.0))
    ok = [CheckpointRow(0, 20, ["periodic"], 0.5, True, {}), CheckpointRow(1, 30, ["compaction"], 0.5, True, {}), CheckpointRow(2, 40, ["end"], 0.5, True, {})]
    assert "lost_after_compaction" not in tm.modes_of(bundle(steps=[edit(1)], checkpoints=ok, phi0=0.0, reward=0.0))


def test_a_failing_path_check_is_environment_misuse_and_a_masked_episode_has_no_modes():
    b = bundle(steps=[edit(1)], reward=0.0, path={"checks": [{"id": "path:install", "status": "fail", "severity": "gate"}]})
    assert tm.modes_of(b)["env_misuse"] == "path:install"
    assert tm.modes_of(bundle(masked=True, path={"checks": [{"id": "path:install", "status": "fail"}]})) == {}


# ---- task metadata and splits ----
def test_task_ids_give_lane_language_and_repo_and_held_out_repos_split_out(monkeypatch):
    from crucible.factory.lanes import windows

    monkeypatch.setitem(windows.HELD_OUT_REPOS, "zzheld", "D")  # a made-up held-out repository (no real held-out id in git)
    assert tm.task_meta("db-go-mvdan-sh-04056a74-k1-1c2768") == ("debug", "go", "mvdan-sh")
    assert tm.split_of_task("db-go-zzheld-0000aaaa-d1") == "heldout"
    assert tm.split_of_task("db-go-go-toml-a02114e3-e1b") == "train"
    assert tm.placeholder("db-go-zzheld-0000aaaa").startswith("heldout-") and "zzheld" not in tm.placeholder("db-go-zzheld-0000aaaa")


# ---- general vs task-specific ----
def test_a_mode_across_three_tasks_and_two_languages_is_general_and_one_in_one_task_is_specific():
    eps = [ep("db-go-a-00000001", modes={"stalls_after_partial_fix": "x"}, i=1), ep("db-go-b-00000002", modes={"stalls_after_partial_fix": "x"}, i=2),
           ep("db-python-c-00000003", lang="python", modes={"stalls_after_partial_fix": "x", "loops_failing_command": "y"}, i=3),
           ep("db-python-c-00000003", lang="python", i=4)]
    sp = tm.general_split(eps, min_tasks=3)
    assert "stalls_after_partial_fix" in sp["general"] and "loops_failing_command" not in sp["general"]
    row = next(r for r in sp["modes"] if r["mode"] == "stalls_after_partial_fix")
    assert row["k"] == 3 and row["n"] == 4 and 0 < row["lo"] < row["p"] < row["hi"] <= 1
    same_lang = [ep(f"db-go-t{k}-0000000{k}", modes={"budget_flat": "z"}, i=k) for k in range(1, 4)]
    assert "budget_flat" not in tm.general_split(same_lang)["general"]  # three tasks, one language and one lane: task-family specific
    assert any(t["task"] == "db-python-c-00000003" for t in sp["task_specific"])


# ---- learning and alerts ----
def test_a_mode_that_falls_over_training_is_learned_and_one_that_rises_is_regressing():
    eps = []
    for s in range(1, 7):
        for k in range(10):
            m = {}
            if s <= 2 and k < 8:
                m["stalls_after_partial_fix"] = "x"
            if s >= 5 and k < 8:
                m["env_misuse"] = "y"
            eps.append(ep(f"db-go-t{k}-0000000{k % 9}", step=s, modes=m, i=s * 100 + k))
    v = tm.trends(eps)["verdicts"]
    assert v["stalls_after_partial_fix"]["verdict"] == "learned" and v["env_misuse"]["verdict"] == "regressing"


def test_unordered_cohorts_get_no_trend_and_no_trend_alerts():
    eps = [ep("db-go-a-00000001", hack=(k > 5), i=k) for k in range(10)]
    for k, e in enumerate(eps):
        e.cohort = "deepseek" if k > 5 else "student"
    tr = tm.trends(eps)
    assert tr["verdicts"] == {} and not tr["ordered"]
    assert not [a for a in tm.alerts(eps, tr) if a["kind"] == "hack_rate_up"]


def test_alerts_fire_on_hacks_rising_reward_without_solves_kl_and_flat_groups():
    eps = []
    for s in range(1, 7):
        for k in range(10):
            eps.append(ep(f"db-go-t{k}-0000000{k % 9}", step=s, hack=(s >= 5 and k < 3), reward=0.1 if s <= 2 else 0.5, i=s * 100 + k))
    steps = [{"step": s, "kl": 0.2 if s == 6 else 0.001, "groups": 4, "flat_groups": 4 if s >= 5 else 0, "entropy": 1.0 if s < 6 else 0.2} for s in range(1, 7)]
    kinds = {a["kind"] for a in tm.alerts(eps, tm.trends(eps), steps)}
    assert {"hack_rate_up", "reward_up_full_pass_flat", "kl_spike", "entropy_collapse", "flat_groups"} <= kinds


def test_the_band_empties_when_every_task_is_clearly_solved_or_clearly_failed():
    eps = [ep("db-go-a-00000001", step=3, passed=True, i=k) for k in range(12)] + [ep("db-go-b-00000002", step=3, i=20 + k) for k in range(40)]
    assert tm.in_band_tasks(eps)["in_band"] == []
    assert "band_emptying" in {a["kind"] for a in tm.alerts(eps, tm.trends(eps))}


# ---- proposals, intents and the held-out discipline ----
def test_proposals_and_intents_refuse_held_out_episodes():
    eps = [ep("db-go-zzheld-0000aaaa-d1", split="heldout", modes={"stalls_after_partial_fix": "x"})]
    sp = tm.general_split(eps)
    with pytest.raises(tm.HeldOutUse):
        tm.proposals(eps, sp)
    with pytest.raises(tm.HeldOutUse):
        tm.swarm_intents(eps, sp)


def test_the_report_builds_proposals_from_training_only_and_shows_held_out_as_placeholders():
    train = [ep(f"db-{lang}-r{k}-0000000{k}", lang=lang, modes={"stalls_after_partial_fix": "x"}, i=k) for k, lang in enumerate(["go", "go", "python"], 1)]
    held = [ep("db-go-zzheld-0000aaaa-d1", split="heldout", modes={"stalls_after_partial_fix": "x"}, i=9)]
    r = tm.build_report(train + held)
    assert r["proposals"] and all(p["applied"] is False for p in r["proposals"])
    assert all("zzheld" not in json.dumps(p) for p in r["proposals"] + r["swarm_intents"])
    assert r["heldout"]["episodes"] == 1 and all(t.startswith("heldout-") for t in r["heldout"]["tasks"])
    assert "zzheld" not in json.dumps(r["heldout"])


def test_a_learned_general_mode_gets_no_proposal():
    eps = []
    for s in range(1, 7):
        for k, lang in enumerate(["go", "go", "python"] * 4):
            eps.append(ep(f"db-{lang}-r{k}-0000000{k % 9}", lang=lang, step=s, modes={"stalls_after_partial_fix": "x"} if s <= 2 else {}, i=s * 100 + k))
    r = tm.build_report(eps)
    assert r["trends"]["verdicts"]["stalls_after_partial_fix"]["verdict"] == "learned"
    assert not [p for p in r["proposals"] if p["mode"] == "stalls_after_partial_fix"]


# ---- the judge ----
class FakeJudge:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    async def complete_json(self, messages, **kw):
        self.calls += 1
        return self.reply(messages[1]["content"])


def test_the_judge_is_never_qwen_and_a_quote_not_in_the_trace_is_rejected():
    b = bundle(steps=[edit(1)], milestones={"m01": False}, reward=0.0, final_message="I fixed the parser")
    e = ep("db-go-carbon-93371b6f")
    e.id = b.id
    with pytest.raises(ValueError):
        asyncio.run(tm.judge([e], {b.id: b}, FakeJudge(lambda d: {}), judge_model="Qwen/Qwen3.5-397B-A17B", n=1, cap_usd=1.0))
    good = FakeJudge(lambda d: {"cause": "skill_gap", "mode": "symptom_only", "quote": "I fixed the parser"})
    r = asyncio.run(tm.judge([e], {b.id: b}, good, judge_model="openai/gpt-oss-120b", n=1, cap_usd=1.0))
    assert r["labelled"] == 1 and e.judged and e.cause == "skill_gap" and "symptom_only" in e.modes
    e2 = ep("db-go-carbon-93371b6f")
    e2.id = b.id
    bad = FakeJudge(lambda d: {"cause": "skill_gap", "mode": "none", "quote": "a sentence that is not there"})
    assert asyncio.run(tm.judge([e2], {b.id: b}, bad, judge_model="openai/gpt-oss-120b", n=1, cap_usd=1.0))["rejected"] == 1 and not e2.judged


def test_the_judge_stops_at_its_cap_and_skips_held_out_failures():
    b = bundle(steps=[edit(1)], reward=0.0, final_message="done")
    eps = [ep("db-go-carbon-93371b6f", i=k) for k in range(3)] + [ep("db-go-zzheld-0000aaaa", split="heldout", i=9)]
    bundles = {e.id: b for e in eps}
    j = FakeJudge(lambda d: {"cause": "skill_gap", "mode": "none", "quote": "done"})
    r = asyncio.run(tm.judge(eps, bundles, j, judge_model="openai/gpt-oss-120b", n=10, cap_usd=1.0, spent=lambda: 5.0, concurrency=1))
    assert r["stopped_at_cap"] and j.calls == 0
    j2 = FakeJudge(lambda d: {"cause": "skill_gap", "mode": "none", "quote": "done"})
    asyncio.run(tm.judge(eps, bundles, j2, judge_model="openai/gpt-oss-120b", n=10, cap_usd=1.0))
    assert j2.calls == 3  # the held-out failure is never sent to the judge


# ---- ingest of a Runner attempt and the views ----
def write_attempt(root: Path, run: str, sample: str, *, task: str, milestones: dict, reward: float, stop: str = "agent_done") -> Path:
    a = root / "rollouts" / run / "samples" / sample / "a1"
    (a / "verifier").mkdir(parents=True)
    (a / "record.json").write_text(json.dumps({"task": task, "policy": "tinker:Qwen/Qwen3.6-35B-A3B", "reward": reward,
                                               "meta": {"stop": stop, "progress": {"honesty": {"brier": 0.4, "scored": {"R1": [0.9, 0]}}}}}))
    (a / "verifier" / "report.json").write_text(json.dumps({"checks": [{"id": "c_regress", "status": "fail"}],
                                                            "meta": {"milestones": milestones, "phi": 0.5, "full_pass": False,
                                                                     "path": {"checks": [{"id": "path:install", "status": "pass"}]}}}))
    (a / "trace.json").write_text(json.dumps({"events": [{"type": "tool_call", "data": {"id": "c1", "function": "write_file",
                                                                                      "arguments": {"path": "x.go", "content": "y"}}},
                                                         {"type": "tool_result", "data": {"id": "c1", "ok": True}, "text": "ok"}]}))
    return a


def test_ingest_reads_a_training_run_and_maps_groups_to_steps(tmp_path):
    write_attempt(tmp_path, "g1", "s1", task="db-go-carbon-93371b6f", milestones={"m01": True, "m02": False}, reward=0.25)
    write_attempt(tmp_path, "g2", "s1", task="db-go-carbon-93371b6f", milestones={"m01": True, "m02": True}, reward=1.0)
    (tmp_path / "steps.jsonl").write_text(json.dumps({"step": 1, "groups": ["g1"], "rewards": [[0.25, 0.25]]}) + "\n"
                                          + json.dumps({"step": 2, "groups": ["g2"], "rewards": [[1.0, 0.0]], "kl_sample_vs_train": 0.001}) + "\n")
    eps, _ = tm.ingest([tmp_path])
    by = {e.step: e for e in eps}
    assert set(by) == {1, 2} and by[1].brier == 0.4 and "breaks_regressions" in by[1].modes and "overclaims_status" in by[1].modes
    sh = tm.step_health(tmp_path)
    assert sh[0]["flat_groups"] == 1 and sh[1]["kl"] == 0.001


def test_the_markdown_report_names_general_and_task_specific_sections(tmp_path):
    from crucible.observe.train_monitor import html, markdown

    eps = [ep(f"db-{lang}-r{k}-0000000{k}", lang=lang, modes={"stalls_after_partial_fix": "x"}, i=k) for k, lang in enumerate(["go", "go", "python"], 1)]
    md = markdown(tm.build_report(eps))
    assert "## General failures" in md and "stalls_after_partial_fix" in md and "## Task-specific" in md and "## Proposals" in md
    assert html(tm.build_report(eps)).startswith("<!doctype html>")


def test_swarm_diagnose_carries_the_monitor_s_general_failures(tmp_path):
    from crucible.factory.swarm_journal import diagnose, render

    eps = [ep(f"db-{lang}-r{k}-0000000{k}", lang=lang, modes={"stalls_after_partial_fix": "x"}, i=k) for k, lang in enumerate(["go", "go", "python"], 1)]
    d = diagnose(tmp_path, monitor=tm.build_report(eps))
    assert d["general_failures"]["modes"][0]["mode"] == "stalls_after_partial_fix" and d["general_failures"]["intents"]
    assert "General failures of the student" in render(d)
