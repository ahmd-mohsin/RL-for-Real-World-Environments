"""P3 (swarm-pipeline/00 §8.2-§8.5): facts first, the first wrong step, pass audits, typed proposals and their executed
acceptance (§8.3) with fake ports, intent candidates for a human, versions with lineage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.traceaudit import (
    SEEDED_REQUIREMENTS,
    AcceptConfig,
    CheckpointRow,
    EpisodeBundle,
    FakeAlternatives,
    Proposal,
    ProposedCheck,
    TaskContext,
    ToolStep,
    Verdict,
    accept,
    accept_false_fail,
    accept_spec_defect,
    accept_verifier_gap,
    accept_world_bug,
    facts,
    fault_regex,
    first_wrong_step,
    intent_candidates,
    load_bundle,
    missing_names,
    pass_audit,
    propose_spec_defects,
    repro_command,
    seeded_prompt,
    seeded_rec,
    version_accepted,
    write_intent_candidates,
)
from crucible.factory.versioning import VersionStore

PATCH_A = "diff --git a/m.py b/m.py\n--- a/m.py\n+++ b/m.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
PATCH_B = PATCH_A + "diff --git a/n.py b/n.py\n--- a/n.py\n+++ b/n.py\n@@ -1 +1 @@\n-y = 1\n+y = 3\n"


def bundle(phis: list[float], *, steps: int = 12, edits: dict[int, str] | None = None, patch: str = PATCH_A, reward: float = 0.0, **kw: Any) -> EpisodeBundle:
    edits = edits or {}
    st = [ToolStep(i, f"c{i}", "repo.write_file" if i in edits else "repo.read_file", {"path": edits.get(i, "m.py")}, turn=i // 2) for i in range(1, steps + 1)]
    cps = [CheckpointRow(k, (k + 1) * (steps // len(phis)), ["periodic"], p, True, {"m1": p >= 0.5, "m2": p >= 1.0},
                         patch=PATCH_A if k < 2 else PATCH_B) for k, p in enumerate(phis)]
    base = {"id": kw.pop("id", "r/s/a1"), "dir": "", "run": "r", "task": "w1~x", "window": "w1", "policy": kw.pop("policy", "solver:qwen/qwen3.7-flash"),
            "model": "qwen/qwen3.7-flash", "family": kw.pop("family", "qwen"), "steps": st, "checkpoints": cps, "phi0": 0.0, "reward": reward, "patch": patch,
            "stop": "agent_done"}
    base.update(kw)
    return EpisodeBundle(**base)


# ---- facts --------------------------------------------------------------------------------------------------------------------
def test_facts_decide_masked_limit_reply_before_action_world_fault_and_misuse():
    assert [f.cause for f in facts(bundle([0.0], masked=True))] == ["env_flake"]
    lim = facts(bundle([0.0], stop="max_steps"))
    assert any(f.kind == "limit_stop" and f.cause == "budget" for f in lim)
    rba = facts(bundle([0.0, 0.0], final_message="All requirements are implemented.", status_claims={"R1": 0.9}))
    assert any(f.kind == "reply_before_action" and "All requirements are implemented" in f.evidence for f in rba)
    moved = facts(bundle([0.5, 0.5], final_message="All requirements are implemented."))
    assert not any(f.kind == "reply_before_action" for f in moved)  # a milestone moved: no reply-before-action
    b = bundle([0.0], project_modules=["durations"])
    b.steps[1] = ToolStep(2, "c2", "repo.shell", {"command": "sh tools/run_tests.sh"}, "E   ModuleNotFoundError: No module named 'durations'")
    b.steps[3] = ToolStep(4, "c4", "repo.shell", {"command": "make test"}, "sh: 1: make: not found")
    b.steps[5] = ToolStep(6, "c6", "repo.shell", {"command": "make"}, "sh: 1: make: not found")
    b.steps[7] = ToolStep(8, "c8", "repo.shell", {"command": "repo__list_dir ."}, "sh: 1: repo__list_dir: not found")
    b.steps[9] = ToolStep(10, "c10", "repo.shell", {"command": "repo__list_dir"}, "sh: 1: repo__list_dir: not found")
    wf = [f for f in facts(b) if f.kind == "world_fault"]
    assert {f.detail.split(" (x")[0] for f in wf} == {"project module 'durations' not importable through repo.shell", "command 'make' not found in the World"}
    mis = bundle([0.0])
    for i in range(6):
        mis.steps[i].fault, mis.steps[i].error = "policy", "unknown_tool"
    assert any(f.kind == "harness_misuse" for f in facts(mis))
    nm = bundle([0.0])
    nm.steps[0] = ToolStep(1, "c1", "repo.shell", {"command": "git log --oneline -5"})
    assert any(f.kind == "path_near_miss" and "history-adjacent" in f.detail for f in facts(nm))
    assert any(f.kind == "no_edit" for f in facts(bundle([0.0], patch="")))


def test_fault_regex_and_repro_command():
    assert fault_regex("command 'make' not found in the World").search("sh: 1: make: not found")
    assert fault_regex("module 'yaml' missing in the World").search("ModuleNotFoundError: No module named 'yaml'")
    assert repro_command("python -m pytest glom/test -v 2>&1 | head -150 | tail -5") == "python -m pytest glom/test -v 2>&1"


# ---- first wrong step ------------------------------------------------------------------------------------------------------------
def test_first_wrong_step_lost_plateau_flat_and_divergence():
    lost = first_wrong_step(bundle([0.5, 0.5, 0.0, 0.0], steps=12, edits={7: "n.py"}))
    assert lost.kind == "lost" and lost.step == 7 and lost.files == ["n.py"] and lost.lost == ["m1"] and lost.turn == 3 and 0 < lost.position <= 1
    plateau = first_wrong_step(bundle([0.0, 0.5, 0.5], steps=12))
    assert plateau.kind == "plateau" and plateau.step == 8 and plateau.peak == 0.5
    flat = first_wrong_step(bundle([0.0, 0.0], steps=10, edits={3: "m.py"}), oracle_files=["m.py"])
    assert flat.kind == "flat" and flat.files == ["m.py"] and flat.tool_calls == [3]
    peer = bundle([0.5, 1.0], id="r/peer/a1", reward=1.0, patch=PATCH_A)
    div = first_wrong_step(bundle([0.0, 0.0], steps=10, edits={2: "other.py"}), [peer, bundle([0.5, 1.0], id="r/p2/a1", reward=1.0)])
    assert div.kind == "divergence" and div.step == 2 and "other.py" in div.detail and div.reference == "2 passing episode(s)"
    assert first_wrong_step(peer).kind == "none"


def test_pass_audit_flags_short_paths_outside_files_and_near_misses():
    peers = [bundle([1.0], id=f"r/p{i}/a1", reward=1.0, steps=20) for i in range(2)]
    lucky = bundle([1.0], id="r/lucky/a1", reward=1.0, steps=3, patch=PATCH_B)
    why = pass_audit(lucky, oracle_patch=PATCH_A * 1, oracle_files=["m.py"], peers=[*peers, lucky])
    assert any(w.startswith("short_path: 3 tool calls") for w in why) and any(w.startswith("outside_files: ['n.py']") for w in why)
    assert pass_audit(bundle([0.0]), oracle_files=["m.py"]) == []  # failures are not pass-audited


# ---- spec defects (D8) -------------------------------------------------------------------------------------------------------------
def test_missing_names_skip_name_errors():
    tail = "E   ImportError: cannot import name 'format_duration' from 'durations'\nNameError: name 'Any' is not defined\nAttributeError: type object 'V' has no attribute 'isvalid'"
    assert missing_names(tail) == ["format_duration", "isvalid"]


def _base(tmp_path: Path) -> Path:
    base = tmp_path / "base"
    (base / "tests").mkdir(parents=True)
    (base / "durations.py").write_text("def seconds(m):\n    return m * 60\n")
    return base


def _spec(names: list[str], tests: list[str]) -> Proposal:
    return Proposal("spec_defect", "seeded-durations", ["t1"], {"type": "test_id", "tests": tests}, {"prompt_edit": {"names": names, "tests": tests}}, "grader", "x")


def test_spec_defect_accepted_only_on_test_source_lint_and_ledger(tmp_path):
    rec, prompt, reqs, base = seeded_rec(), seeded_prompt(), [dict(r) for r in SEEDED_REQUIREMENTS], _base(tmp_path)
    tests = ["tests/test_format.py::test_format", "tests/test_format.py::test_round_trip"]
    ok = accept_spec_defect(_spec(["format_duration"], tests), rec=rec, prompt=prompt, requirements=reqs, base_dir=base)
    assert ok.status == "accepted" and ok.evidence["counterfactual_rerun_needed"] and ok.evidence["lint_new"] == [] and ok.evidence["ledger_new"] == []
    assert "`format_duration`" in ok.change["prompt"] and ok.evidence["edits"][0]["requirement"] == "R2"
    guess = accept_spec_defect(_spec(["render_compact"], tests), rec=rec, prompt=prompt, requirements=reqs, base_dir=base)
    assert guess.status == "rejected" and "auditor's guess" in guess.reasons[0]
    (base / "durations.py").write_text("def format_duration(t):\n    return str(t)\n")  # A already has it: the end state lost it
    old = accept_spec_defect(_spec(["format_duration"], tests), rec=rec, prompt=prompt, requirements=reqs, base_dir=base)
    assert old.status == "rejected" and "already exist at the start state" in old.reasons[0]
    free = accept_spec_defect(_spec(["format_duration"], ["the auditor thinks a renderer is needed"]), rec=rec, prompt=prompt, requirements=reqs, base_dir=base)
    assert free.status == "rejected" and "D8" in free.reasons[0]
    again = accept_spec_defect(_spec(["format_duration"], tests), rec=rec, prompt=ok.change["prompt"], requirements=ok.change["requirements"], base_dir=base)
    assert again.status == "rejected" and "already states" in again.reasons[0]


def test_spec_defect_proposals_come_from_the_grader_and_the_prompt():
    b = bundle([0.0], prompt="Add parse_duration.", grader_tail="E   ImportError: cannot import name 'format_duration' from 'durations'",
               failing_tests=["tests/test_format.py::test_format"])
    ps = propose_spec_defects(b)
    assert [p.change["prompt_edit"]["names"] for p in ps] == [["format_duration"]] and ps[0].artifact["grader_line"].startswith("ImportError")
    b.prompt = "Add parse_duration and format_duration."
    assert propose_spec_defects(b) == []  # stated: a skill gap, not a spec defect


# ---- verifier gaps and false fails with fake ports ------------------------------------------------------------------------------
class FakeVerifier:
    """Solutions are strings; ``truth`` says which really work. The current verifier passes ``demo`` (a lucky pass); a
    check passes the solutions in its ``passes`` list."""

    task_id = "w1"

    def __init__(self, *, flaky: bool = False, mutant_passes: bool = False):
        self.flaky, self.mutant_passes, self.calls = flaky, mutant_passes, 0

    async def verdict(self, solution: Any, *, add: Any = (), drop: Any = ()) -> Verdict:
        if solution in ("oracle", "alt"):
            ok = True
        elif solution == "demo":
            ok = not add
        elif solution == "mutant":
            ok = bool(drop) or (self.mutant_passes and bool(add))
        elif solution == "end":  # a correct end state the current verifier fails
            ok = bool(drop)
        else:
            ok = False
        return Verdict(ok, {"m1": ok}, [] if ok else ["t::x"])

    async def check(self, check: ProposedCheck, solution: Any) -> bool | None:
        self.calls += 1
        if self.flaky and solution == "oracle" and self.calls % 7 == 0:
            return False
        return solution in json.loads(check.source)["passes"]

    def oracle(self) -> Any:
        return "oracle"

    def alternatives(self) -> list[tuple[str, Any]]:
        return [("alt", "alt")]

    def mutants(self) -> list[tuple[str, Any]]:
        return [("revert_m1", "mutant")]

    def floors(self) -> list[tuple[str, Any]]:
        return [("no_op", "noop")]

    def content(self) -> dict[str, Any]:
        return {"checks": {"c_m1": ["t::x"]}}


def gap(passes: list[str]) -> Proposal:
    chk = ProposedCheck("new", "tests/test_new.py", json.dumps({"passes": passes}), ["tests/test_new.py::t"])
    return Proposal("verifier_gap", "w1", ["r/lucky/a1"], {"type": "end_state", "patch": "demo"}, {"add_check": chk.__dict__}, "pass_audit", "lucky")


FAST = AcceptConfig(flake_reruns=30)


async def test_verifier_gap_accepted_only_on_executed_evidence():
    alts = FakeAlternatives({"w1": [("p2_alt", "alt")]})
    good = await accept_verifier_gap(gap(["oracle", "alt"]), FakeVerifier(), alts, config=FAST)
    assert good.status == "accepted" and good.evidence["flake"]["n"] == 60 and good.evidence["flake"]["upper"] <= 0.05
    assert good.evidence["alternatives"] == {"alt": True, "p2_alt": True} and good.evidence["mutants"] == {"revert_m1": False}
    fake = await accept_verifier_gap(gap(["alt"]), FakeVerifier(), alts, config=FAST)
    assert fake.status == "rejected" and "fails the oracle" in fake.reasons[0]
    unfair = await accept_verifier_gap(gap(["oracle"]), FakeVerifier(), alts, config=FAST)
    assert unfair.status == "rejected" and "valid alternative" in unfair.reasons[0]
    lax = await accept_verifier_gap(gap(["oracle", "alt", "demo"]), FakeVerifier(), alts, config=FAST)
    assert lax.status == "rejected" and "does not fail the demonstrating end state" in lax.reasons[0]
    loose = await accept_verifier_gap(gap(["oracle", "alt"]), FakeVerifier(mutant_passes=True), alts, config=FAST)
    assert loose.status == "rejected" and "mutants:revert_m1" in loose.reasons[0]
    flaky = await accept_verifier_gap(gap(["oracle", "alt"]), FakeVerifier(flaky=True), alts, config=FAST)
    assert flaky.status == "rejected" and "flake" in flaky.reasons[0]
    p = gap(["oracle", "alt"])
    p.artifact["patch"] = "end"  # the current verifier already fails it: nothing to tighten
    assert (await accept_verifier_gap(p, FakeVerifier(), alts, config=FAST)).status == "rejected"
    empty = Proposal("verifier_gap", "w1", ["x"], {"type": "end_state", "patch": "demo"}, {}, "pass_audit", "lucky")
    assert (await accept_verifier_gap(empty, FakeVerifier(), alts, config=FAST)).status == "needs_input"


class Second:
    def __init__(self, v: bool | None):
        self.v = v

    async def verdict(self, task_id: str, solution: Any) -> bool | None:
        return self.v


async def test_false_fail_needs_the_second_verifier_and_keeps_mutants_failing():
    def ff() -> Proposal:
        return Proposal("false_fail", "w1", ["r/e/a1"], {"type": "end_state", "patch": "end"}, {"relax": {"drop_tests": ["t::x"]}}, "label", "x")

    assert (await accept_false_fail(ff(), FakeVerifier(), Second(None))).status == "rejected"  # no P2 second verifier yet: never relaxed
    assert (await accept_false_fail(ff(), FakeVerifier(), Second(False))).status == "rejected"

    class Strict(FakeVerifier):
        async def verdict(self, solution: Any, *, add: Any = (), drop: Any = ()) -> Verdict:
            return Verdict(solution in ("oracle", "end") and bool(drop) or solution == "oracle")

    ok = await accept_false_fail(ff(), Strict(), Second(True))
    assert ok.status == "accepted" and ok.evidence["mutants"] == {"revert_m1": False} and ok.evidence["floors"] == {"no_op": False}
    loose = await accept_false_fail(ff(), FakeVerifier(), Second(True))  # dropping the test lets the mutant pass
    assert loose.status == "rejected" and "mutants:revert_m1" in loose.reasons[0]


# ---- World bugs ---------------------------------------------------------------------------------------------------------------------
class FakeWorld:
    def __init__(self, *, broken: bool = True, fix_works: bool = True, s3_ok: bool = True):
        self.broken, self.fix_works, self.s3_ok = broken, fix_works, s3_ok

    async def reproduce(self, repro: Any, *, state: Any, fix: Any = None) -> dict[str, Any]:
        fails = self.broken and not (fix and self.fix_works)
        return {"failed": fails, "exit": 1 if fails else 0, "output": "No module named 'durations'" if fails else "1 passed"}

    async def s3(self, *, fix: Any = None) -> dict[str, Any]:
        return {"ok": self.s3_ok, "why": "" if self.s3_ok else "existing tests fail"}

    def content(self) -> dict[str, Any]:
        return {}


async def test_world_bug_needs_a_reproduction_a_fix_and_s3():
    def wb(fix: Any = None) -> Proposal:
        return Proposal("world_bug", "w1", ["t"], {"type": "repro", "function": "repo.shell", "arguments": {"command": "sh tools/run_tests.sh"},
                                                  "signature": "project module 'durations' not importable"}, {"world_fix": fix}, "fact", "x")

    assert (await accept_world_bug(wb(), FakeWorld(broken=False), oracle="o")).status == "rejected"
    assert (await accept_world_bug(wb(), FakeWorld(), oracle="o")).status == "reproduced_needs_fix"
    assert (await accept_world_bug(wb({"files": {}}), FakeWorld(fix_works=False), oracle="o")).status == "rejected"
    assert (await accept_world_bug(wb({"files": {}}), FakeWorld(s3_ok=False), oracle="o")).status == "rejected"
    ok = await accept_world_bug(wb({"files": {"tools/run_tests.sh": "x"}}), FakeWorld(), oracle="o")
    assert ok.status == "accepted" and ok.evidence["before_fix_on_oracle"]["failed"] and not ok.evidence["after_fix_on_oracle"]["failed"]


# ---- intent drift, dispatch, versions -------------------------------------------------------------------------------------------------
async def test_intent_drift_is_a_candidate_for_a_human_never_applied(tmp_path):
    rows = [{"trace": f"t{i}", "family": fam, "signature": "skill_gap|flat|harness_misuse", "on_intent": False} for i, fam in enumerate(["qwen", "qwen", "llama"])]
    rows += [{"trace": f"u{i}", "family": "qwen", "signature": "skill_gap|flat|no_edit", "on_intent": False} for i in range(5)]  # one family only
    rows += [{"trace": f"v{i}", "family": f, "signature": "skill_gap|lost|none", "on_intent": True} for i, f in enumerate(["qwen", "llama", "openai"])]
    cands = intent_candidates(rows, intent={"id": "I1", "behaviour": "keeps earlier work"})
    assert [c["signature"] for c in cands] == ["skill_gap|flat|harness_misuse"] and cands[0]["status"] == "awaiting_human" and not cands[0]["applied"]
    write_intent_candidates(tmp_path / "ic.jsonl", cands)
    assert json.loads((tmp_path / "ic.jsonl").read_text())["families"] == ["llama", "qwen"]
    p = Proposal("intent_drift", "I1", ["t0"], {"type": "cluster"}, {"intent": "candidate"}, "cluster", "x")
    assert (await accept(p, None)).status == "awaiting_human"


async def test_dispatch_without_ports_is_needs_input_and_versions_record_lineage(tmp_path):
    p = Proposal("world_bug", "w1", ["t"], {"type": "repro"}, {"world_fix": None}, "fact", "x")
    assert (await accept(p, TaskContext("w1"))).status == "needs_input"
    store = VersionStore(tmp_path / "v.jsonl")
    v = FakeVerifier()
    g = await accept_verifier_gap(gap(["oracle", "alt"]), v, FakeAlternatives({}), config=FAST)
    ctx = TaskContext("w1", verifier=v, prompt="P")
    ver = version_accepted(g, store, ctx)
    assert ver is not None and ver.proposal_id == g.id and ver.parent is not None and g.version == ver.id
    assert "new" in ver.content["added"] and store.verify() == []
    rejected = await accept_verifier_gap(gap(["alt"]), v, FakeAlternatives({}), config=FAST)
    assert version_accepted(rejected, store, ctx) is None


def test_load_bundle_reads_a_runner_attempt(tmp_path):
    d = tmp_path / "runs" / "solvers" / "samples" / "s1" / "a1"
    (d / "checkpoints").mkdir(parents=True)
    (d / "artifacts").mkdir()
    (d / "verifier").mkdir()
    (d / "record.json").write_text(json.dumps({"task": "w1~abc", "policy": "solver:openai/gpt-oss-20b", "reward": 0.0, "mask_sample": False, "steps": 2,
                                               "meta": {"stop": "max_steps", "progress": {"phi0": 0.0, "honesty": {"scored": {"R1": [0.9, 0]}, "defaulted": [],
                                                                                                                  "claimed": True}}}}))
    (d / "calls.json").write_text(json.dumps({"agent": [{"call_id": "agent:c1", "actions": ["tool:a", "tool:b"], "finish": "tool_calls"}]}))
    (d / "trajectory.agent.json").write_text(json.dumps({"steps": [{"source": "user", "message": "do R1"}, {"source": "agent", "message": "on it"}]}))
    (d / "trace.json").write_text(json.dumps({"events": [
        {"type": "tool_call", "actor": "agent", "data": {"function": "repo.shell", "arguments": {"command": "git log --all"}, "id": "a"}},
        {"type": "tool_result", "actor": "agent", "text": "commit 1", "data": {"id": "a", "ok": True}},
        {"type": "tool_call", "actor": "agent", "data": {"function": "repo.write_file", "arguments": {"path": "m.py"}, "id": "b"}},
        {"type": "tool_result", "actor": "agent", "text": "ok", "data": {"id": "b", "ok": True}}]}))
    (d / "checkpoints" / "h1.patch").write_text(PATCH_A)
    (d / "checkpoints.jsonl").write_text(json.dumps({"index": 0, "turn": 2, "triggers": ["end"], "patch_sha256": "h1", "state": "done", "milestones": {"m1": False},
                                                     "phi": 0.0, "measured": True}) + "\n")
    (d / "artifacts" / "model.patch").write_text(PATCH_A)
    (d / "verifier" / "report.json").write_text(json.dumps({"checks": [{"id": "c_m1", "detail": {"failing": ["t::y"]}}], "meta": {
        "milestones": {"m1": False}, "phi": 0.0, "failing_tests": ["t::x"], "tail": "E assert 1 == 2"}}))
    b = load_bundle(d, run="lhrun:solvers")
    assert b.window == "w1" and b.family == "openai" and b.role == "solver" and b.prompt == "do R1" and b.status_claims == {"R1": 0.9}
    assert [s.turn for s in b.steps] == [0, 0] and b.steps[0].result == "commit 1" and b.checkpoints[0].patch == PATCH_A
    assert b.failing_tests == ["t::x", "t::y"] and b.path["recomputed"] and "path:history" in b.path["violations"]
    kinds = {f.kind for f in facts(b)}
    assert {"limit_stop", "path_violation"} <= kinds


@pytest.mark.parametrize("policy,role", [("hacker:deepseek/deepseek-v4-flash", "hacker"), ("rlm:qwen/qwen3.7-flash", "rlm"),
                                         ("solver:qwen/qwen3.7-flash@route", "solver"), ("qwen/qwen3.7-flash", "solver")])
def test_roles(policy, role):
    assert bundle([0.0], policy=policy).role == role
