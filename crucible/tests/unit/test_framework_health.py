"""Framework health (factory.framework_health): agent vs framework, the component, the cluster and its impact, the pause
decision, the fix journal and versions, the metadata and tuning proposals, held-out redaction, and the CLI's exit code.
Every case is a defect we already met (the scratch install, the lost Node setup, the unquoted workspace path, a stale
visible test, the flock-masked grades, the liveness probe on the wrong Docker daemon) or its agent-side look-alike."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from crucible.factory import framework_health as fh
from crucible.factory import train_monitor as tm
from crucible.factory.traceaudit import EpisodeBundle, ToolStep

HERE = Path(__file__).resolve().parents[2]


def bundle(i: int = 0, task: str = "db-go-carbon-93371b6f", **kw) -> EpisodeBundle:
    base = {"id": f"r/s{i}/a1", "dir": "/x", "run": "r", "task": task, "window": task, "policy": "tinker:Qwen/Qwen3.6-35B-A3B",
            "model": "Qwen/Qwen3.6-35B-A3B", "family": "qwen", "patch": "diff --git a/x.go b/x.go\n+x\n"}
    base.update(kw)
    return EpisodeBundle(**base)


def view(i: int = 0, *, task: str = "db-go-carbon-93371b6f", family: str = "qwen", model: str = "Qwen/Qwen3.6-35B-A3B", masked: bool = False,
         reward: float | None = 0.0, milestones: dict | None = None, steps: list | None = None, rec: dict | None = None, rep: dict | None = None,
         cps: list | None = None, calls: list | None = None, split: str = "train", group: str = "g", stop: str = "agent_done", step: int | None = None) -> fh.View:
    ms = dict(milestones if milestones is not None else {"m01": True, "m02": False})
    b = bundle(i, task, steps=list(steps or []), milestones=ms, masked=masked, reward=reward, stop=stop, family=family, model=model)
    lane, lang, _ = tm.task_meta(task)
    e = tm.Episode(id=b.id, dir=f"/nonexistent/{i}", task=task, lane=lane, lang=lang, split=split, cohort="c", step=step, policy=f"tinker:{model}", family=family,
                   masked=masked, passed=bool(ms) and all(ms.values()) and not masked, reward=reward, phi0=0.0, phi_final=0.5, stop=stop, tool_calls=len(steps or []),
                   compactions=0, milestones=ms, model=model)
    return fh.View(e, b, dict(rec or {}), dict(rep or {}), list(cps or []), list(calls or []), None, group)


def tool(i: int, fn: str, result: str, *, fault: str | None = "tool", ok: bool = False, args: dict | None = None) -> ToolStep:
    return ToolStep(i, f"c{i}", fn, args or {"path": "x"}, result=result, ok=ok, error="tool_error" if fault == "tool" else None, fault=fault)


def kinds(views: list[fh.View]) -> dict[str, fh.Defect]:
    return {d.kind + ":" + d.signature: d for d in fh.cluster([h for v in views for h in fh.per_attempt(v)], views)}


# ---- agent vs framework ----
def test_a_tool_error_from_the_agents_own_arguments_is_agent_behaviour_and_never_a_pause():
    vs = [view(k, steps=[tool(1, "repo.edit_file", "ValueError: old_string not found")]) for k in range(4)]
    d = next(iter(kinds(vs).values()))
    assert d.cls == "agent" and d.severity == "LOW"
    assert not fh.health_report(vs)["pause"]


def test_a_missing_core_command_in_most_of_a_tasks_episodes_is_a_critical_world_defect_and_one_fact():
    """The World tools' own interpreter missing (a Node World without python3): every tool fails, the rewards are the World's."""
    res = "python snippet failed (rc=127): bash: line 1: python3: command not found"
    vs = [view(k, steps=[tool(1, "repo.list_dir", res), tool(2, "repo.read_file", res)], reward=0.25) for k in range(4)]
    ds = kinds(vs)
    world = [d for d in ds.values() if d.kind == "world_fault"]
    assert len(world) == 1 and world[0].severity == "CRITICAL" and world[0].component == "world" and "python3" in world[0].signature
    assert not [d for d in ds.values() if d.kind == "tool_error"]  # the same fact seen through two tools: one defect
    assert world[0].impact["reward_at_risk"] == 3.0 and "advantage_if_corrected" in world[0].impact


def test_a_lost_node_setup_is_a_world_fault_and_a_made_up_command_is_the_agents():
    lost = view(1, task="db-node-zzfake-0000beef", steps=[tool(1, "repo.shell", "sh: 1: vitest: not found", fault=None, ok=False)])
    made_up = view(2, steps=[tool(1, "repo.shell", "sh: 1: foo: not found", fault=None, ok=False)] * 2)
    ds = kinds([lost, made_up])
    assert any(d.kind == "world_fault" and "vitest" in d.signature and d.cls == "framework" for d in ds.values())
    assert not any("foo" in d.signature and d.cls == "framework" for d in ds.values())


def test_a_convenience_tool_missing_everywhere_is_high_not_critical():
    vs = [view(k, steps=[tool(1, "repo.shell", "bash: xxd: command not found", fault=None)]) for k in range(4)]
    d = next(d for d in kinds(vs).values() if "xxd" in d.signature)
    assert d.severity == "HIGH"


# ---- masked attempts, host and sandbox ----
def test_the_liveness_probe_on_the_wrong_daemon_is_a_sandbox_defect_that_costs_samples():
    f = {"kind": "crucible:env_container_lost", "stage": "agent", "reason": "the World's sandbox died (after step 1)", "retryable": True}
    vs = [view(k, masked=True, reward=None, rec={"failure": f}) for k in range(2)] + [view(9, reward=1.0, milestones={"m01": True})]
    d = next(d for d in kinds(vs).values() if d.kind == "masked_failure")
    assert d.component == "sandbox" and d.severity == "HIGH" and d.impact["samples_lost"] == 2 and d.impact["groups_without_signal"] == 1


def test_the_unquoted_workspace_path_masks_at_seed_as_a_world_defect():
    f = {"kind": "agent_run_error", "stage": "seed", "reason": "sh: 1: cd: can't cd to /Users/someone/Desktop/RL"}
    d = next(d for d in kinds([view(k, masked=True, reward=None, rec={"failure": f}) for k in range(3)]).values() if d.kind == "masked_failure")
    assert d.component == "world" and "<path>" in d.signature and len(d.episodes) == 3


def test_flock_masked_grades_show_as_a_lock_defect_and_unmeasured_checkpoints():
    reason = "Traceback (most recent call last):\n  File \"run_gpu.py\", line 58\nflock: failed to execute cd: No such file or directory"
    cps = [{"index": 1, "turn": 20, "triggers": ["periodic"], "measured": False, "reason": f"RuntimeError: milestones unmeasured: {reason}"}]
    vs = [view(k, masked=True, reward=None, rec={"failure": {"kind": "agent_run_error", "stage": "verification", "reason": reason}}, cps=cps) for k in range(2)]
    ds = kinds(vs)
    assert any(d.kind == "host_resource" and d.signature.startswith("lock") for d in ds.values())
    assert any(d.kind == "unmeasured_checkpoint" and d.component == "grader" for d in ds.values())
    assert any(d.kind == "masked_failure" and d.component == "grader" for d in ds.values())


def test_a_job_log_error_outside_any_episode_is_found(tmp_path):
    (tmp_path / "plumbing.jsonl").write_text('{"step": "grade", "reason": "flock: failed to execute cd: No such file or directory"}\n'
                                             "Traceback (most recent call last):\n  File \"x.py\", line 1\nImportError: cannot import name 'kernel_profiles'\n")
    hs = fh.log_hits([tmp_path])
    assert {h.kind for h in hs} == {"host_resource", "job_log_error"} and any("kernel_profiles" in h.signature for h in hs)


# ---- grader defects that corrupt reward ----
def test_a_patch_graded_two_ways_by_the_sidecar_is_nondeterministic_and_a_sidecar_final_split_is_a_credit_defect(tmp_path):
    cps = [{"index": 0, "turn": 20, "measured": True, "patch_sha256": "a" * 64, "milestones": {"m01": True}},
           {"index": 1, "turn": 40, "measured": True, "patch_sha256": "a" * 64, "milestones": {"m01": False}}]
    assert any(h.kind == "nondeterministic_grade" and h.severity == "CRITICAL" for h in fh.per_attempt(view(1, cps=cps)))
    adir = tmp_path / "a1"
    (adir / "artifacts").mkdir(parents=True)
    (adir / "artifacts" / "model.patch").write_text("diff\n")
    import hashlib

    sha = hashlib.sha256(b"diff\n").hexdigest()
    v = view(2, milestones={"m01": True}, cps=[{"index": 0, "turn": 20, "measured": True, "patch_sha256": sha, "milestones": {"m01": False}}])
    v.ep.dir = str(adir)
    assert any(h.kind == "credit_grade_mismatch" and h.severity == "CRITICAL" for h in fh.per_attempt(v))


def test_a_stale_visible_test_fails_in_the_solving_episodes_of_two_families_and_its_full_passes_are_at_risk():
    rep = {"checks": [{"id": "c_regress", "status": "fail", "detail": {"failing": ["pkg/x_test.go::TestEscapes"]}}]}
    vs = [view(k, milestones={"m01": True, "m02": True}, reward=0.5, rep=rep, family=("qwen" if k % 2 else "deepseek"),
               model=("Qwen/Qwen3.6-35B-A3B" if k % 2 else "deepseek-ai/DeepSeek-V4.1-Flash")) for k in range(4)]
    hs = fh.stale_tests(vs)
    assert len(hs) == 4 and all(h.kind == "stale_visible_test" and h.severity == "CRITICAL" for h in hs)
    d = fh.cluster(hs, vs)[0]
    assert d.impact["reward_at_risk"] == 2.0
    one_family = [view(k, milestones={"m01": True, "m02": True}, reward=0.5, rep=rep) for k in range(4)]
    assert not fh.stale_tests(one_family)  # one family breaking the same test is the agent's regression


def test_a_stale_test_tied_to_one_milestone_is_found_even_without_full_solves():
    rep_fail = {"checks": [{"id": "c_regress", "status": "fail", "detail": {"failing": ["t::TestOld"]}}]}
    vs = [view(k, milestones={"m01": True, "m02": False}, rep=rep_fail, family=f, model=m) for k, (f, m) in
          enumerate([("qwen", "Q"), ("deepseek", "D"), ("qwen", "Q"), ("deepseek", "D")])]
    vs += [view(10 + k, milestones={"m01": False, "m02": False}) for k in range(4)]
    hs = fh.stale_tests(vs)
    assert hs and all("whenever m01 passes" in h.signature for h in hs)


def test_a_moved_reward_from_a_path_rule_change_corrupts_advantages_and_is_measured():
    vs = [view(k, reward=r, group="g1") for k, r in enumerate([0.25, 0.0, 1.0])]
    hits = [fh.Hit("path_false_positive", "grader", "path:install: fail -> pass", "CRITICAL", vs[0].ep.id, vs[0].ep.task, reward_new=1.0)]
    d = fh.cluster(hits, vs)[0]
    # the stored 0.25 sat below the group mean (a negative advantage); corrected to 1.0 it sits above it: the gradient pushed the wrong way
    assert d.impact["reward_moved"] == 0.75 and d.impact["groups"] == 1 and d.impact["sum_abs_delta_advantage"] > 0 and d.impact["sign_flips"] == 1


def test_the_scratch_install_regrade_is_read_through_repath(tmp_path, monkeypatch):
    """The scratch ``go mod tidy`` case: repath under the current rules flips path:install and lifts the reward."""
    from crucible.factory import repath as rp

    store = tmp_path / "store"
    (store / "train" / "tasks" / "db-go-carbon-93371b6f" / "base").mkdir(parents=True)
    (store / "train" / "tasks" / "db-go-carbon-93371b6f" / "window.json").write_text("{}")
    v = view(1, reward=0.5)
    monkeypatch.setattr(rp, "repath", lambda a, w, b: rp.Repath(str(a), v.ep.task, "student", {"reward": 0.5, "gates": {"path:install": "fail"}},
                                                               {"reward": 1.0, "gates": {"path:install": "pass"}, "evidence": {}}, ["path:install"]))
    hs = fh.repath_flips([v], store)
    assert hs and hs[0].kind == "path_false_positive" and hs[0].reward_new == 1.0 and hs[0].severity == "CRITICAL"


def test_s9b_false_fails_and_low_agreement_are_grader_defects(tmp_path):
    (tmp_path / "fairness").mkdir()
    (tmp_path / "fairness" / "db-go-echo-61eca05b.json").write_text(json.dumps({"task": "db-go-echo-61eca05b", "false_fail": [{"id": "alt/x/a1"}],
                                                                               "agreement": {"kappa": 0.4, "disagreements": [{}, {}]}}))
    hs = fh.fairness([tmp_path])
    assert {h.severity for h in hs} == {"CRITICAL", "HIGH"} and all(h.component == "grader" for h in hs)


def test_a_codec_bug_is_one_models_malformed_calls_far_above_the_others():
    bad = [ToolStep(i, f"c{i}", "repo.x", {}, ok=False, error="bad_tool_name", fault="policy") for i in range(60)]
    good = [ToolStep(i, f"c{i}", "repo.x", {}, ok=True) for i in range(60)]
    vs = [view(1, steps=bad, model="moonshotai/Kimi-K2.6", family="kimi"), view(2, steps=good), view(3, steps=good, model="glm", family="glm")]
    hs = fh.codec(vs)
    assert hs and all(h.component == "sampler" and "Kimi" in h.signature for h in hs)


def test_trainer_nan_kl_spikes_and_drops_are_trainer_defects(tmp_path):
    (tmp_path / "steps.jsonl").write_text("\n".join(json.dumps(r) for r in [{"step": 1, "loss": float("nan"), "rewards": []},
                                                                            {"step": 2, "loss": 0.1, "kl_sample_vs_train": 0.2, "rewards": [], "drops": {"g1": "flat"}}]))
    hs = fh.trainer_hits([tmp_path])
    assert {(h.severity, h.signature.split(" ")[0]) for h in hs} >= {("CRITICAL", "loss"), ("HIGH", "sampler-learner"), ("HIGH", "groups")}


# ---- triage: pause, fix journal, versions ----
def test_an_open_critical_defect_pauses_and_a_recorded_fix_unpauses(tmp_path):
    res = "python snippet failed (rc=127): bash: line 1: python3: command not found"
    vs = [view(k, steps=[tool(1, "repo.list_dir", res)], step=1) for k in range(3)]
    h = fh.health_report(vs)
    assert h["pause"] and h["pause_reason"]
    did = next(d["id"] for d in h["defects"] if d["severity"] == "CRITICAL")
    fh.record_fix(tmp_path, defect=did, component="world", version="world-node-2", commit="abc1234", after_step=1, note="python3 in the Node World")
    h2 = fh.health_report(vs, journal=fh.read_journal([tmp_path]))
    d2 = next(d for d in h2["defects"] if d["id"] == did)
    assert d2["status"] == "fixed@world-node-2" and not h2["pause"]
    later = vs + [view(9, steps=[tool(1, "repo.list_dir", res)], step=2)]
    h3 = fh.health_report(later, journal=fh.read_journal([tmp_path]))
    assert next(d for d in h3["defects"] if d["id"] == did)["status"] == "recurring" and h3["pause"]


def test_two_path_rule_versions_in_one_run_without_a_fix_record_is_a_defect():
    a = view(1, step=1, rep={"meta": {"path": {"rules_version": "path-1"}}})
    b = view(2, step=2, rep={"meta": {"path": {"rules_version": "path-2"}}})
    h = fh.health_report([a, b])
    assert any(d["kind"] == "version_mix" for d in h["defects"]) and h["versions"]["step 1"]["path_rules"] == {"path-1": 1}
    h2 = fh.health_report([a, b], journal=[{"component": "grader", "defect": "x", "version": "path-2"}])
    assert not any(d["kind"] == "version_mix" for d in h2["defects"])


def test_regrade_plans_name_the_offline_tool_and_never_a_training_reward():
    d = fh.Defect("fd-1", "path_false_positive", "grader", "framework", "CRITICAL", "s", ["r/s0/a1"], ["t"], {}, {}, {})
    assert fh.regrade_plan(d, [view(0)])["tool"] == "repath"
    d2 = fh.Defect("fd-2", "stale_visible_test", "task_spec", "framework", "CRITICAL", "s", [], [], {}, {}, {})
    assert "OfflineGrade" in fh.regrade_plan(d2, [])["command"]


# ---- held-out discipline ----
def test_a_held_out_defect_keeps_component_and_counts_and_withholds_tasks_episodes_and_content():
    rep = {"checks": [{"id": "c_regress", "status": "fail", "detail": {"failing": ["secret_test.go::TestSecret"]}}]}
    vs = [view(k, task="db-go-zzheld-0000aaaa", split="heldout", milestones={"m01": True}, reward=0.5, rep=rep, family=f, model=m)
          for k, (f, m) in enumerate([("qwen", "Q"), ("deepseek", "D"), ("qwen", "Q")])]
    h = fh.health_report(vs)
    d = next(d for d in h["defects"] if d["kind"] == "stale_visible_test")
    blob = json.dumps(d)
    assert d["heldout"] and d["component"] == "task_spec" and "TestSecret" not in blob and "zzheld" not in blob and d["repro"] == {"withheld": "held-out"}
    assert d["tasks"][0].startswith("heldout-")


def test_tuning_reads_training_traces_only():
    bad = [tool(i, "repo.search", "search failed: re.error: unterminated character set at position 3 SECRETQUERY") for i in range(1, 30)]
    vs = [view(k, task="db-go-zzheld-0000aaaa", split="heldout", steps=bad, family=f) for k, f in enumerate(["qwen", "deepseek"])]
    assert "SECRETQUERY" not in json.dumps(fh.health_report(vs)["tuning"])


# ---- metadata and tuning ----
def test_group_size_follows_the_spread_probability_of_in_band_tasks():
    assert abs(fh.spread_probability(0.5, 8) - (1 - 2 * 0.5 ** 8)) < 1e-12
    vs = [view(k, task=f"db-go-t{t}-0000000{t}", milestones={"m01": k % 8 == 0}, reward=1.0 if k % 8 == 0 else 0.0) for t in range(3) for k in range(8)]
    md = fh.metadata(vs)
    props = fh.tuning(md, vs, k=4)
    assert any(p["lever"] == "group_size" for p in props)


def test_checkpoint_overhead_proposes_a_bounded_cadence_and_useful_samples_are_counted():
    cps = [{"index": i, "turn": 20 * i, "measured": True, "graded_s": 60.0, "milestones": {}} for i in range(5)]
    vs = [view(k, reward=float(k % 2), cps=cps, rec={"timing": {"agent": 1000.0, "model": 400.0}, "usage": {"calls": 10, "input_tokens": 1000}, "cost_usd": 1.0})
          for k in range(4)]
    md = fh.metadata(vs)
    lane = md["lanes"]["debug"]
    assert lane["checkpoint_share_of_agent_time"] == 0.3 and lane["useful_samples"] == 4 and lane["usd_per_useful_sample"] == 1.0
    cad = [p for p in fh.tuning(md, vs) if p["lever"] == "checkpoint_cadence"]
    assert cad and "40" in cad[0]["text"] and cad[0]["applied"] is False


def test_an_agent_input_heavy_tool_gets_an_affordance_proposal():
    bad = [tool(i, "repo.search", "search failed: re.error: missing )") for i in range(1, 11)] + [tool(i, "repo.search", "ok", fault=None, ok=True) for i in range(11, 31)]
    vs = [view(1, steps=bad), view(2, steps=bad, family="deepseek", model="D")]
    assert any(p["lever"] == "tool_affordance" and "repo.search" in p["text"] for p in fh.tuning(fh.metadata(vs), vs))


def test_an_attempt_no_view_can_be_read_from_is_reported_not_skipped(tmp_path):
    a = tmp_path / "rollouts" / "r1" / "samples" / "s0" / "a1"
    a.mkdir(parents=True)
    (a / "record.json").write_text("{not json")
    hs = fh.unreadable_hits([tmp_path], [])
    assert len(hs) == 1 and hs[0].kind == "unreadable_attempt" and hs[0].component == "harness"


# ---- the CLI ----
def test_health_cli_exits_3_on_an_open_critical_defect(tmp_path, monkeypatch):
    run = tmp_path / "run"
    for k in range(3):
        a = run / "rollouts" / "r1" / "samples" / f"s{k}" / "a1"
        (a / "verifier").mkdir(parents=True)
        (a / "record.json").write_text(json.dumps({"task": "db-go-carbon-93371b6f", "policy": "tinker:Qwen/Qwen3.6-35B-A3B", "reward": 0.0, "meta": {"stop": "agent_done"}}))
        (a / "verifier" / "report.json").write_text(json.dumps({"checks": [], "meta": {"milestones": {"m01": False}, "phi": 0.0, "path": {"checks": []}}}))
        res = "python snippet failed (rc=127): bash: line 1: python3: command not found"
        (a / "trace.json").write_text(json.dumps({"events": [{"type": "tool_call", "data": {"id": "c1", "function": "repo.list_dir", "arguments": {}}},
                                                             {"type": "tool_result", "data": {"id": "c1", "ok": False, "fault": "tool", "error": "tool_error"}, "text": res}]}))
    sys.path.insert(0, str(HERE / "scripts"))
    import train_monitor as cli

    rc = cli.main(["health", "--run", str(run), "--out", str(tmp_path / "out")])
    assert rc == 3 and (tmp_path / "out" / "health.md").read_text().count("PAUSE") == 1
    h = json.loads((tmp_path / "out" / "health.json").read_text())
    did = next(d["id"] for d in h["defects"] if d["severity"] == "CRITICAL")
    assert cli.main(["fix", "--run", str(run), "--defect", did, "--component", "world", "--version", "w2", "--commit", "abc", "--after-step", "0"]) == 0
    assert cli.main(["health", "--run", str(run), "--out", str(tmp_path / "out2")]) == 0
