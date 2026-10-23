"""S9b (factory.alternatives_gate) and the multi-language second verifier (factory.verifier2 under a language profile)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from crucible.factory.alternatives_gate import (
    Candidate,
    LiveAlternatives,
    agreement,
    classify_one,
    family_of_policy,
    gate,
    harvest,
    kappa,
)
from crucible.factory.generators.debug import scope
from crucible.factory.verifier2 import check_shadow_path, leaks, parse_shadow, repo_context, shadow_brief

REQ_MS = {"R1": "m01", "R2": "m02"}
MS_TESTS = {"m01": ["a_test.go::TestA"], "m02": ["b_test.go::TestB"]}


def cand(fam: str = "qwen", cid: str = "c1") -> Candidate:
    return Candidate(cid, "solver", f"solver:tinker:{'Qwen/Qwen3.5-397B-A17B' if fam == 'qwen' else 'deepseek-ai/DeepSeek-V4.1-Flash'}@tinker", fam, "patch")


def v1(full: bool, m01: bool = True, m02: bool = True) -> dict:
    return {"reward": 1.0 if full else 0.0, "milestones": {"m01": m01, "m02": m02},
            "statuses": {"a_test.go::TestA": "passed" if m01 else "failed", "b_test.go::TestB": "passed" if m02 else "failed"}}


def test_family_of_a_run_records_policy() -> None:
    assert family_of_policy("solver:tinker:deepseek-ai/DeepSeek-V4.1-Flash:peft:262144@tinker") == "deepseek"
    assert family_of_policy("student:tinker:Qwen/Qwen3.6-35B-A3B@tinker") == "qwen"
    assert family_of_policy("scripted") == ""


def test_the_classes_follow_both_verifiers_and_independence() -> None:
    ok = {"R1": True, "R2": True}
    # verifier 1 rejects a fix that an independent verifier 2 passes on every requirement: a false-fail candidate
    ff = classify_one(cand("qwen"), v1=v1(False, m02=False), v2=ok, v2_family="deepseek", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert ff.cls == "false_fail" and ff.failing_hidden == ["b_test.go::TestB"]
    # the same verdict from verifier 2 of the candidate's own family is not independent: nothing confirms it
    same = classify_one(cand("deepseek"), v1=v1(False, m02=False), v2=ok, v2_family="deepseek", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert same.cls == "unknown" and not same.v2_independent and set(same.v2.values()) == {None}
    # verifier 1 passes, verifier 2 fails a requirement: a lucky pass for the audit
    lp = classify_one(cand(), v1=v1(True), v2={"R1": True, "R2": False}, v2_family="deepseek", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert lp.cls == "lucky_pass"
    acc = classify_one(cand(), v1=v1(True), v2=ok, v2_family="deepseek", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert acc.cls == "accepted"
    wrong = classify_one(cand(), v1=v1(False, m02=False), v2={"R1": True, "R2": False}, v2_family="deepseek", repro=None, req_milestone=REQ_MS,
                         milestone_tests=MS_TESTS)
    assert wrong.cls == "wrong"
    # a reproduction can stand in for verifier 2 on a requirement; every requirement must be covered for a false fail
    rp = classify_one(cand(), v1=v1(False, m02=False), v2={"R1": True, "R2": None}, v2_family="deepseek", repro={"R2": True},
                      req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert rp.cls == "false_fail"
    half = classify_one(cand(), v1=v1(False, m02=False), v2={"R1": True, "R2": None}, v2_family="deepseek", repro=None,
                        req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    assert half.cls == "wrong" or half.cls == "unknown"
    assert classify_one(cand(), v1=None, v2=ok, v2_family="deepseek", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS).cls == "unknown"


def test_the_gate_fails_on_a_false_fail_and_needs_two_alternatives_one_from_another_family() -> None:
    ok = {"R1": True, "R2": True}

    def acc(fam: str, cid: str):
        return classify_one(cand(fam, cid), v1=v1(True), v2=ok, v2_family="nemotron", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)

    assert gate([acc("qwen", "a")], writer_family="anthropic")["status"] == "unmeasured"  # one alternative is never a pass
    assert gate([acc("qwen", "a"), acc("deepseek", "b")], writer_family="anthropic")["status"] == "pass"
    assert gate([acc("qwen", "a"), acc("qwen", "b")], writer_family="qwen")["status"] == "unmeasured"  # none from another family
    ff = classify_one(cand("qwen", "f"), v1=v1(False, m02=False), v2=ok, v2_family="nemotron", repro=None, req_milestone=REQ_MS, milestone_tests=MS_TESTS)
    g = gate([acc("qwen", "a"), acc("deepseek", "b"), ff], writer_family="anthropic")
    assert g["status"] == "fail" and g["counts"]["false_fail"] == 1 and g["false_fail"][0]["id"] == "f"


def test_kappa_and_the_disagreements_for_the_tie_break() -> None:
    assert kappa([(True, True), (False, False)]) == 1.0
    assert kappa([(True, False), (False, True)]) == -1.0
    assert kappa([(True, True)]) is None and kappa([(True, True), (True, True)]) is None  # no variance: undefined
    rows = [{"id": "oracle", "kind": "oracle", "v1_milestones": {"m01": True, "m02": True}, "v2": {"R1": True, "R2": True}},
            {"id": "start", "kind": "start", "v1_milestones": {"m01": False, "m02": False}, "v2": {"R1": False, "R2": True}}]
    a = agreement(rows, REQ_MS)
    assert a["pairs"] == 4 and a["agree"] == 3 and a["disagreements"] == [{"state": "start", "kind": "start", "requirement": "R2", "milestone": "m02",
                                                                          "v1": False, "v2": True}]


def test_harvest_reads_graded_progress_and_drops_hacks_masks_and_excluded_families(tmp_path: Path) -> None:
    def ep(arm: str, sid: str, *, phi: float, full: bool = False, hack: bool = False, masked: bool = False, policy: str) -> None:
        a = tmp_path / "arms" / arm / "t1" / "rollouts" / "run" / "samples" / sid / "a1"
        (a / "verifier").mkdir(parents=True)
        (a / "artifacts").mkdir()
        (a / "artifacts" / "model.patch").write_text(f"diff {sid}")
        (a / "record.json").write_text(json.dumps({"policy": policy}))
        (a / "verifier" / "report.json").write_text(json.dumps({"reward": 1.0 if full else 0.0, "infra": {"status": "masked" if masked else "ok"},
                                                                "meta": {"phi": phi, "full_pass": full, "hack_suspect": hack}}))

    q, d = "solver:tinker:Qwen/Qwen3.5-397B-A17B@tinker", "solver:tinker:deepseek-ai/DeepSeek-V4.1-Flash@tinker"
    ep("frontier", "s1", phi=1.0, full=True, policy=q)
    ep("frontier", "s2", phi=0.25, policy=q)  # no progress: left out
    ep("frontier", "s3", phi=1.0, full=True, hack=True, policy=q)
    ep("frontier", "s4", phi=1.0, full=True, masked=True, policy=q)
    ep("alt-deepseek", "s5", phi=0.5, policy=d)
    got = harvest("t1", [tmp_path])
    assert sorted((c.arm, c.family, c.source) for c in got) == [("alt-deepseek", "deepseek", "session"), ("frontier", "qwen", "solver")]
    assert [c.family for c in harvest("t1", [tmp_path], exclude_families=["qwen"])] == ["deepseek"]


def test_live_alternatives_reads_the_accepted_patches(tmp_path: Path) -> None:
    (tmp_path / "t1.json").write_text(json.dumps({"accepted_patches": {"frontier/s1/a1": "diff x"}}))
    assert asyncio.run(LiveAlternatives(tmp_path).alternatives("t1~chec-abc")) == [("frontier/s1/a1", "diff x")]
    assert asyncio.run(LiveAlternatives(tmp_path).alternatives("nope")) == []


@pytest.fixture
def go_repo(tmp_path: Path) -> Path:
    (tmp_path / "go.mod").write_text("module example.com/bind\n")
    (tmp_path / "bind.go").write_text("package bind\n\n// Bind binds a value.\nfunc Bind(v string) (int, error) { return 0, nil }\n")
    (tmp_path / "bind_test.go").write_text("package bind\n\nimport \"testing\"\n\nfunc TestBindOld(t *testing.T) {}\n")
    (tmp_path / "sub").mkdir()
    return tmp_path


def test_a_go_shadow_suite_places_new_test_files_and_refuses_bad_places(go_repo: Path) -> None:
    with scope("go"):
        assert "testing package" in shadow_brief() and "_test.go" in shadow_brief()
        hidden = {"bind_test.go": "package bind\nfunc TestBindHidden(t *testing.T) {}\n"}
        assert check_shadow_path("crucible_shadow_r1_test.go", base_dir=go_repo, hidden=hidden) is None
        assert "existing file" in check_shadow_path("bind_test.go", base_dir=go_repo, hidden={})
        assert "hidden test file" in check_shadow_path("bind_test.go", base_dir=go_repo, hidden=hidden)
        assert "not a go test file" in check_shadow_path("crucible_shadow_r1.go", base_dir=go_repo, hidden={})
        assert "no directory" in check_shadow_path("nope/x_test.go", base_dir=go_repo, hidden={})
        assert check_shadow_path("../x_test.go", base_dir=go_repo, hidden={}) == "not a repository-relative path"
        code = "package bind\n\nimport \"testing\"\n\nfunc TestShadowR1Field(t *testing.T) {\n\tif _, err := Bind(\"10a\"); err == nil {\n\t\tt.Fatal(\"want error\")\n\t}\n}\n"
        out = {"tests": [{"requirement": "R1", "path": "crucible_shadow_r1_test.go", "code": code},
                         {"requirement": "R2", "path": "bind_test.go", "code": code},
                         {"requirement": "R9", "path": "x_test.go", "code": code}]}
        files, reqs, probs = parse_shadow(out, ["R1", "R2"], base_dir=go_repo, hidden=hidden)
        assert reqs == {"R1": "crucible_shadow_r1_test.go"} and set(files) == {"crucible_shadow_r1_test.go"}
        assert any("R2" in p and "hidden test file" in p for p in probs) and any("R9" in p for p in probs)
        ctx = repo_context(go_repo, "`Bind` returns an error that names the field")
        assert ctx["language"] == "go" and "bind.go" in ctx["modules"] and ctx["test_style_example"] == {"bind_test.go": (go_repo / "bind_test.go").read_text()}


def test_the_go_payload_is_proven_blind_to_the_oracle_and_the_hidden_tests(go_repo: Path) -> None:
    with scope("go"):
        oracle = {"bind.go": "package bind\n\nfunc Bind(v string) (int, error) { return 0, fmt.Errorf(\"field %s: %w\", name, err) }\n"}
        hidden = {"bind_test.go": "package bind\n\nfunc TestBindNamesTheFailingField(t *testing.T) {}\n"}
        clean = {"request": "Bind should name the field", "repository": repo_context(go_repo, "Bind")}
        assert leaks(clean, oracle_files=oracle, hidden_tests=hidden, base_dir=go_repo) == []
        dirty = {**clean, "hint": "func Bind(v string) (int, error) { return 0, fmt.Errorf(\"field %s: %w\", name, err) }", "x": "TestBindNamesTheFailingField"}
        bad = leaks(dirty, oracle_files=oracle, hidden_tests=hidden, base_dir=go_repo)
        assert any(b.startswith("oracle line") for b in bad) and any("TestBindNamesTheFailingField" in b for b in bad)
