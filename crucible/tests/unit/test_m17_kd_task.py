"""Kernel debugging lane, the task (swarm-pipeline/10 §2, §4, §7): A'..B' synthesis on a synthetic repository (B' == B,
fixes applying alone, reverts alone), milestones and claimed edges, the LongTask contract, masking without a GPU,
outcome gates from the kit's report, the prompt lint (no faulty names, files, hidden cases, hashes), the ledger, the intent
card, simplify, and the GPU calibration math. git only, no Docker, no GPU."""

from __future__ import annotations

import asyncio
import importlib.util
import subprocess
from pathlib import Path

import pytest

from crucible.factory.generators import base as gb
from crucible.factory.lanes import kernel_debug as kd
from crucible.factory.lanes import kernel_debug_mine as kdm
from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.longtask import check_longtask

SOFT_A = "static __global__ void soft_max_kernel(float * y) {\n    float v = y[0];\n    y[1] = v;\n}\n"
NORM_A = "static __global__ void norm_kernel(float * x, int n) {\n    int i = blockIdx.y * 65535;\n    x[i] = 0;\n}\n"
HARN = "// op test\nint main() {\n    // cases\n    return 0;\n}\n"


def git(d: Path, *a: str, date: str) -> str:
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(d)}
    return subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True, env=env).stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> tuple[Path, list[kdm.KernelChange]]:
    d = tmp_path / "src"
    (d / "ggml/src/ggml-cuda").mkdir(parents=True)
    (d / "tests").mkdir()
    (d / "ggml/src/ggml-cuda/softmax.cu").write_text(SOFT_A)
    (d / "ggml/src/ggml-cuda/norm.cu").write_text(NORM_A)
    (d / "tests/test-backend-ops.cpp").write_text(HARN)
    git(d, "init", "-q", date="2026-05-02T00:00:00+00:00")
    git(d, "add", "-A", date="2026-05-02T00:00:00+00:00")
    git(d, "commit", "-qm", "init", date="2026-05-02T00:00:00+00:00")
    out = []
    steps = [("CUDA: fix race in soft max", "softmax.cu", SOFT_A.replace("    y[1] = v;", "    __syncthreads();\n    y[1] = v;"), "    // cases\n    // soft max race case\n",
              "2026-06-01T00:00:00+00:00", "race", ["SOFT_MAX"]),
             ("CUDA: fix norm for more than 65535 rows", "norm.cu", NORM_A.replace("blockIdx.y * 65535", "(int64_t) blockIdx.y * 65535"), "    // norm grid case\n",
              "2026-07-01T00:00:00+00:00", "grid_limit", ["NORM", "SOFT_MAX"])]
    for subj, f, text, case, date, cls, ops in steps:
        (d / "ggml/src/ggml-cuda" / f).write_text(text)
        h = (d / "tests/test-backend-ops.cpp").read_text().replace("    return 0;", case.rstrip("\n") + "\n    return 0;")
        (d / "tests/test-backend-ops.cpp").write_text(h)
        git(d, "commit", "-qam", subj, date=date)
        sha = git(d, "rev-parse", "HEAD", date=date)
        out.append(kdm.KernelChange(repo="r", sha=sha, parent=git(d, "rev-parse", "HEAD^", date=date), date=date, author_date=date, subject=subj, body="",
                                    files=[f"ggml/src/ggml-cuda/{f}", "tests/test-backend-ops.cpp"], kind="bug", fault_class=cls,
                                    detectors=list(kdm.CLASSES[cls]["detectors"]), functions=[f.split(".")[0] + "_kernel" if f != "softmax.cu" else "soft_max_kernel"],
                                    kernels=[], ops=ops, adds_cases=True, licence="MIT"))
    return d, out


def build_rec(tmp_path: Path, repo: tuple[Path, list[kdm.KernelChange]]) -> dict:
    d, faults = repo
    derived = gb.derived_clone(d, tmp_path / "derived")
    syn = kd.synthesize(derived, faults, ko.GGML_ORG, rid="kd-bug-r-test")
    assert syn.ok and syn.b == faults[-1].sha and [f.sha for f in syn.faults] == [f.sha for f in faults]
    assert all(syn.alone.values()) and all(syn.revert_alone.values())
    a_tree = kd.export(derived, syn.a, tmp_path / "a")
    assert "__syncthreads" not in (a_tree / "ggml/src/ggml-cuda/softmax.cu").read_text()  # both fixes out of A'
    assert "norm grid case" not in (a_tree / "tests/test-backend-ops.cpp").read_text()  # the fixes' own cases out of the agent's copy
    ms = kd.milestones_of(syn.faults, {f.sha: [f"OP{i}(x)" for i in range(3)] for f in faults}, {faults[1].sha: ["NORM(rows=70000)"]})
    rec = {"id": "kd-bug-r-test", "family": kd.FAMILIES["bug"], "kind": "bug", "repo": "o/r", "split": "train", "a": syn.a, "b": syn.b, "b_date": faults[-1].date,
           "layout": {"cuda_dir": ko.GGML_ORG.cuda_dir, "dispatch_file": ko.GGML_ORG.dispatch_file}, "harness": {"dialect": "ggml-2025"},
           "milestones": ms, "oracle_patch": kd.diff_patch(derived, syn.a, syn.b, ko.GGML_ORG.paths()), "tree_a": "a",
           "mutants": [{"id": "revert-m1", "kind": "revert", "milestone": "m1", "patch": "x"}], "world": {"base_dir": str(a_tree)}}
    return rec


def test_synthesis_milestones_and_claimed_edges(tmp_path: Path, repo):
    rec = build_rec(tmp_path, repo)
    m1, m2 = rec["milestones"]
    assert (m1["class"], m1["checks"], m1["repeat"]) == ("race", ["golden", "racecheck"], 5)
    assert m2["checks"] == ["golden"] and m2["claimed_requires"] == ["m1"]  # they share SOFT_MAX: claimed, measured on the GPU
    assert m2["requires"] == [] and m2["cases_added"] == ["NORM(rows=70000)"]
    assert "__syncthreads" in rec["oracle_patch"] and "tests/" not in rec["oracle_patch"]  # the oracle is the backend change only
    spec = kd.kdspec(rec)
    assert spec["keep_paths"] == ["ggml/src/ggml-cuda/"] and "fault" not in str(spec["milestones"])  # no fault identity reaches the grader


def authored(rec: dict) -> dict:
    return {"builder": kd.BUILDER, "opening": "Since the last update two ops give me wrong numbers on our H100 boxes.",
            "requirements": {"m1": "SOFT_MAX sometimes returns different numbers on repeated runs of the same input.",
                             "m2": "NORM is wrong when the tensor has more than 65,535 rows in a dimension."},
            "closing": "Both used to match the CPU results."}


def test_task_contract_masking_and_outcome_gates(tmp_path: Path, repo):
    rec = build_rec(tmp_path, repo)
    res = kd.author(rec, authored(rec))
    assert res["findings"] == [], res["findings"]
    t = kd.KernelDebugTask(rec)
    assert check_longtask(t, long_horizon=False) == []
    assert any("short" in p for p in check_longtask(t, long_horizon=True))  # edges are claimed until the GPU prober executes them
    rep = asyncio.run(t.grade_patch(t.oracle_patch()))
    assert rep.infra["failure_kind"] == "no_gpu" and rep.meta["phi"] is None  # V8: masked, never a 0

    class Fake:
        async def grade(self, r, patch):
            return {"reward": 0.0, "gates": {"build": 1}, "components": {"check/m1": 1.0, "check/m2": 0.0}, "infra": {"status": "ok"},
                    "gate_status": {"build": "pass", "m1": "pass", "m2": "fail"}}

    rep = asyncio.run(kd.KernelDebugTask(rec, Fake()).grade_patch("p"))
    assert rep.meta["milestones"] == {"m1": True, "m2": False} and rep.reward == pytest.approx(0.25)
    assert t.card().family == "kernel-bug"
    loc = t.simplify("locate")
    assert "softmax.cu" in loc.prompt and loc.task_id.endswith("~locate")
    pre = t.simplify("prerequisites", pre={"m1"})
    assert [m["id"] for m in pre.rec["milestones"]] == ["m2"]
    intent = kd.task_intent(rec)
    assert {"kernel_fault_localization", "gpu_concurrency", "index_arithmetic", "interacting_faults"} <= set(intent["skills"])
    assert intent["targets"]["student"] == "Qwen/Qwen3.6-35B-A3B" and intent["targets"]["benchmark_max"] == 0.30


def test_prompt_lint_refuses_the_answer(tmp_path: Path, repo):
    rec = build_rec(tmp_path, repo)
    bad = authored(rec)
    bad["requirements"]["m1"] = "soft_max_kernel in softmax.cu needs a barrier (see 1bb2b9fcb, #30103)."
    bad["requirements"]["m2"] = "the case NORM(rows=70000) fails"
    f = kd.author(rec, bad)["findings"]
    assert any("soft_max_kernel" in x for x in f) and any("softmax.cu" in x for x in f) and any("1bb2b9fcb" in x for x in f)
    assert any("#30103" in x for x in f) and any("NORM(rows=70000)" in x for x in f)
    ok = authored(rec)
    ok["opening"] = "SOFT_MAX gives different numbers each run on 1048576 elements."  # an op name and a plain number are fine
    assert kd.author(rec, ok)["findings"] == []
    assert kd.instruction("sm_{ARCH} within {PERF_BAR}", "120", "1.08x") == "sm_120 within 1.08x"


def test_calibration_pool_keeps_added_cases_and_strides_the_rest():
    from crucible.factory.lanes import kernel_debug_plan as kdp

    pool = [f"FA(x={i})" for i in range(10_000)]
    got = kdp.pool_sample(pool, ["FA(x=9999)", "NEW(x)"], cap=100)
    assert got[:2] == ["FA(x=9999)", "NEW(x)"] and len(got) == 100 and len(set(got)) == 100
    assert got[2] == "FA(x=0)" and got[-1].startswith("FA(x=98")  # spread over the whole pool
    assert kdp.pool_sample(pool[:5], []) == pool[:5]


def load_gpu_runner():
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_debug_gpu.py"
    spec = importlib.util.spec_from_file_location("lh_kernel_debug_gpu", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_gpu_calibration_math():
    g = load_gpu_runner()
    b = {"A": True, "B": True, "C": False, "D": True}
    rv = {"A": False, "B": True, "C": False, "D": False}
    assert g.owned_cases(b, rv, ["D"]) == ["D", "A"]  # B must pass and the revert break it; the fix's own case first
    edges = g.executed_edges({"m1": ["A"], "m2": ["X"]}, {"m1": {"A"}, "m2": {"A", "Y"}})
    assert edges == {"m1": ["m2"], "m2": []}  # m1's case also breaks under fault 2's revert: m1 requires m2
    ok = g.perf_bar([1.0, 1.01, 0.99], 1.6)
    assert ok["ok"] and ok["bar"] == pytest.approx(1.0 + max(3 * (1.01 - 0.99) / 0.99, 0.05), abs=1e-3)
    assert not g.perf_bar([1.0, 1.2], 1.25)["ok"]  # noise eats the headroom: no valid bar
    assert g.cp_upper(0, 60) < 0.05 and g.cp_upper(3, 60) > 0.05


def test_host_seeding_verdict():
    g = load_gpu_runner()
    s1 = [{"case": "A", "out": b"1", "ref": b"x"}]
    s2 = [{"case": "A", "out": b"2", "ref": b"y"}]
    assert g.seeding_verdict(s1, list(s1), s2, {"A": "OK"}, ["A"])["ok"]
    assert not g.seeding_verdict(s1, s2, s2, {"A": "OK"}, ["A"])["same_seed_identical"]  # the same seed must repeat exactly
    assert not g.seeding_verdict(s1, list(s1), list(s1), {"A": "OK"}, ["A"])["new_seed_differs"]  # a new seed must change the inputs
    assert not g.seeding_verdict(s1, list(s1), s2, {"A": "FAIL"}, ["A"])["cpu_reference_ok"]
    assert not g.seeding_verdict([], [], [], {}, ["A"])["ok"]


def test_partial_mutant_whitespace_equivalence_and_pool_reuse(tmp_path):
    g = load_gpu_runner()
    a = "--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-  int i = 0;\n+  int  i = 0;\n"
    b = "--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-  int i = 0;\n+int i = 0;\n"
    c = "--- a/x.cu\n+++ b/x.cu\n@@ -1 +1 @@\n-  int i = 0;\n+  int i = 1;\n"
    assert g.added_lines(a) == g.added_lines(b) != g.added_lines(c)
    # a pool grade is reused only for the same patch, spec and tag (its key is written next to the report)
    d = tmp_path / "t"
    (d / "grades" / "cal-oracle").mkdir(parents=True)
    patch, spec = d / "oracle.patch", d / "kdspec.pooled.json"
    patch.write_text(c)
    spec.write_text("{}")
    import hashlib

    key = hashlib.sha256(b"\0".join([patch.read_bytes(), spec.read_bytes(), b"cal-oracle", b"True", b"", g.kit_digest()])).hexdigest()
    out = d / "grades" / "cal-oracle"
    (out / "inputs.sha256").write_text(key)
    (out / "report.json").write_text('{"gates": {"build": "pass"}, "checks": []}')
    (out / "crucible_reward.json").write_text('{"reward": 1.0, "infra": {"status": "ok"}}')
    assert g.grade(d, patch, spec, "docker", "120", "cal-oracle", pool=True)["reward"] == 1.0  # no container started


def test_pool_cap_zero_means_every_case():
    from crucible.factory.lanes import kernel_debug_plan as kdp

    # the runtime JSON-decodes parameters: pool_cap=0 arrives as the integer 0 and must not fall back to the default cap
    assert kdp.pool_cap(None) == kdp.POOL_CAP and kdp.pool_cap("") == kdp.POOL_CAP
    assert kdp.pool_cap(0) == 0 and kdp.pool_cap("0") == 0 and kdp.pool_cap(250) == 250
    assert kdp.perf_groups("m1_ms", ["A", "B"], False) == {"m1_ms": ["A", "B"]}
    assert kdp.perf_groups("m1_ms", ["A", "B"], True) == {"m1_ms#0": ["A"], "m1_ms#1": ["B"]}


def test_start_drop_and_flake_verdict():
    g = load_gpu_runner()
    bug = {"milestones": [{"id": "m1", "kind": "bug"}, {"id": "m2", "kind": "bug"}]}
    perf = {"milestones": [{"id": "m1", "kind": "perf"}, {"id": "m2", "kind": "perf"}]}
    assert g.start_not_real(bug, True, {"m1": [], "m2": []}, {"m1:racecheck": "pass"})
    assert not g.start_not_real(bug, True, {"m1": ["A"], "m2": []}, {})  # one case fails with every fault in: real
    assert not g.start_not_real(bug, True, {"m1": [], "m2": []}, {"m2:memcheck": "fail"})  # a sanitizer catches it: real
    assert not g.start_not_real(bug, False, {"m1": [], "m2": []}, {})  # an unbuilt start says nothing
    assert not g.start_not_real(perf, True, {"m1": [], "m2": []}, {})  # a perf start is correct by design: never dropped here
    ok, masked, bad = {"reward": 1.0, "infra": {"status": "ok"}}, {"reward": None, "infra": {"status": "masked"}}, {"reward": 0.5, "infra": {"status": "ok"}}
    v = g.flake_verdict([ok] * 59 + [masked])
    assert v["measured"] == 59 and v["false_negatives"] == 0 and v["cp_upper"] <= 0.05
    assert g.flake_verdict([ok] * 30 + [masked] * 30)["cp_upper"] > 0.05  # masked grades are not passes
    assert g.flake_verdict([ok] * 59 + [bad])["cp_upper"] > 0.05


def test_pool_reuse_key_includes_the_grader_kit(tmp_path, monkeypatch):
    g = load_gpu_runner()
    monkeypatch.setattr(g, "HERE", tmp_path)
    f = tmp_path / "kit" / "crucible" / "factory" / "lanes" / "kernel_debug_plan.py"
    f.parent.mkdir(parents=True)
    f.write_text("POOL_CAP = 1000\n")
    before = g.kit_digest()
    f.write_text("POOL_CAP = 1000  # fixed\n")
    assert g.kit_digest() != before  # a grader fix invalidates every cached pool grade



def test_sanitizer_cases_launch_the_fault_kernels():
    g = load_gpu_runner()
    assert g.launches_kernel("_Z20cpy_scalar_transposeIfEvPKcPclllllllllllll", "cpy_scalar_transpose")
    assert not g.launches_kernel("_Z20cpy_scalar_transposeIfEvPKcPclllllllllllll", "cpy_scalar")  # a longer name is another kernel
    assert g.launches_kernel("cpy_scalar(const char*)", "cpy_scalar") and not g.launches_kernel("_Z110cpy_scalarv", "cpy_scalar")
    prof = {"CPY(contig)": ["_Z21cpy_scalar_contiguousIffEvPKcPcl"], "CPY(contig,2)": ["_Z21cpy_scalar_contiguousIffEvPKcPcl"],
            "CPY(transposed,long-name)": ["_Z20cpy_scalar_transposeIfEvPKcPcl", "_Z21cpy_scalar_contiguousIffEvPKcPcl"],
            "CPY(t)": ["_Z20cpy_scalar_transposeIfEvPKcPcl"], "ADD(x)": ["_Z5k_addv"]}
    got = g.sanitizer_cases(prof, ["cpy_scalar_transpose", "cpy_scalar_contiguous"], k=2)
    assert got[0] == "CPY(t)"  # the rarer kernel first, the shorter case first
    assert set(got) <= set(prof) - {"ADD(x)"} and len(got) == 2
    assert g.sanitizer_cases(prof, ["no_such_kernel"]) == []


def test_fault_kernels_follow_the_call_graph(tmp_path):
    d = tmp_path / "ggml" / "src" / "ggml-cuda"
    d.mkdir(parents=True)
    (d / "a.cuh").write_text("static __device__ void helper(float * x) { x[0] = 1.0f; }\n"
                             "static __global__ void kern_a(float * x) { helper(x); }\n"
                             "static __global__ void kern_b(float * x) { x[0] = 2.0f; }\n"
                             "void launch_b(float * x) { kern_b<<<1, 1>>>(x); }\n")
    (d / "ggml-cuda.cu").write_text("static bool ggml_cuda_compute_forward(int op) { return true; }\n")
    assert ko.fault_kernels(tmp_path, ["helper"]) == ["kern_a"]  # a kernel that reaches a changed device function
    assert ko.fault_kernels(tmp_path, ["launch_b"]) == ["kern_b"]  # the kernel a changed launcher launches
    assert ko.fault_kernels(tmp_path, ["kern_b"]) == ["kern_b"] and ko.fault_kernels(tmp_path, ["missing"]) == []


def test_grader_cases_join_the_eval_list_and_their_pools():
    from crucible.factory.lanes import kernel_harness as kh

    src = ("static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n"
           "    test_cases.emplace_back(new test_get_rows());\n    return test_cases;\n}\n"
           "static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    return test_cases;\n}\n")
    line = "test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 32, 8, 70000, 1, false));"
    out = kh.add_cases(src, [line])
    assert out.index(line) < out.index("return test_cases;") and out.count(line) == 1  # eval mode only, before its return
    assert kh.add_cases(out, [line]) == out and kh.add_cases(src, []) == src
    with pytest.raises(kh.HarnessPatchError):
        kh.add_cases(src, ["system(\"rm -rf /\");"])  # only case lines, never other code
    with pytest.raises(kh.HarnessPatchError):
        kh.add_cases(src, ["test_cases.emplace_back(new test_get_rows()); system(\"x\");"])
    assert "{4096, 512, 1, 1}" in kh.add_cases(src, ["test_cases.emplace_back(new test_rms_norm(GGML_TYPE_F32, {4096, 512, 1, 1}, false, 1e-6f));"])
    g = load_gpu_runner()
    spec = {"harness_extra": [{"milestone": "m1", "code": line}],
            "milestones": [{"id": "m1", "ops": ["GET_ROWS"], "case_pool": ["GET_ROWS(a)"]}, {"id": "m2", "ops": ["MUL_MAT"], "case_pool": ["MUL_MAT(b)"]}]}
    got = g.adopt_extra_cases(spec, ["GET_ROWS(a)", "GET_ROWS(big)", "MUL_MAT(b)", "MUL_MAT(new)"])
    assert got == {"m1": ["GET_ROWS(big)"]}  # only the milestone the lines were written for, only its ops
    assert spec["milestones"][0]["case_pool"][0] == "GET_ROWS(big)" and spec["milestones"][0]["cases_added"] == ["GET_ROWS(big)"]
    assert "MUL_MAT(new)" not in spec["milestones"][1]["case_pool"]


def test_prompt_discloses_grader_cases():
    rec = {"milestones": [{"id": "m1"}], "harness_extra": [{"milestone": "m1", "code": "x"}]}
    text, _r, _s = kd.compose({"opening": "o", "requirements": {"m1": "r"}}, rec)
    assert "plus the grader's own cases" in text
    assert "plus the grader's own cases" not in kd.compose({"opening": "o", "requirements": {"m1": "r"}}, {"milestones": [{"id": "m1"}]})[0]


def test_borderline_cases_leave_the_calibrated_set():
    from crucible.factory.lanes import kernel_debug_plan as kdp

    g = load_gpu_runner()
    assert kdp.case_ratio({"nmse_vs_cpu": 1e-4, "max_err": 5e-4}) == pytest.approx(0.2)
    assert kdp.case_ratio({"error": "missing"}) == float("inf")
    grade = {"checks": [{"id": "m1:golden", "detail": {"per_case_ratio": {"A": 0.02, "B": 0.9, "C": None}}}]}
    assert g.borderline(grade, "m1") == {"B", "C"}  # near the tolerance or unmeasured: a fresh seed would fail it


def test_perf_ratio_reads_passing_and_failing_checks():
    g = load_gpu_runner()
    passing = {"checks": [{"id": "m1:perf", "detail": {"pass": True, "metrics": {"m1_ms#0": {"ratio": 1.02}}}}]}
    failing = {"checks": [{"id": "m1:perf", "detail": {"attempts": [{"metrics": {"m1_ms#0": {"ratio": 3.0}}}, {"metrics": {"m1_ms#0": {"ratio": 1.7}}}]}}]}
    assert g.perf_ratio(passing, "m1:perf", "m1_ms#0") == 1.02  # a pass reports its last attempt at the top (it read None)
    assert g.perf_ratio(failing, "m1:perf", "m1_ms#0") == 1.7 and g.perf_ratio(failing, "m1:perf", "m2_ms") is None


def test_grader_cases_in_both_layouts_and_by_name():
    from crucible.factory.lanes import kernel_harness as kh

    two = ("static std::vector<std::unique_ptr<test_case>> make_test_cases_eval() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n    return test_cases;\n}\n"
           "static std::vector<std::unique_ptr<test_case>> make_test_cases_perf() {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n    return test_cases;\n}\n")
    t = "test_cases.emplace_back(new test_get_rows(GGML_TYPE_F32, 32, 8, 70000, 1, false));"
    p = "test_cases.emplace_back(new test_mul_mat_id(GGML_TYPE_IQ4_KS, GGML_TYPE_F32, 8, 2, false, 2048, 512, 2048));"
    out = kh.add_cases(two, [{"code": t}, {"code": p, "mode": "perf"}])
    perf_at = out.index("make_test_cases_perf")
    assert out.index(t) < perf_at < out.index(p)  # each mode's list gets its own lines
    old = "static bool test_backend(ggml_backend_t backend, test_mode mode, const char * op_name) {\n    std::vector<std::unique_ptr<test_case>> test_cases;\n" \
          "    test_cases.emplace_back(new test_get_rows());\n    // run tests\n    if (mode == MODE_TEST) {\n        return true;\n    }\n}\n"
    out = kh.add_cases(old, [{"code": p, "mode": "perf"}])
    assert out.index(p) < out.index("if (mode == MODE_TEST)")  # the 2024 layout: one list both modes run
    with pytest.raises(kh.HarnessPatchError):
        kh.add_cases(two, [{"code": t, "mode": "bench"}])
    g = load_gpu_runner()
    name = "MUL_MAT_ID(type_a=iq4_ks,type_b=f32,n_mats=8,n_used=2,b=0,m=2048,n=512,k=2048)"
    spec = {"harness_extra": [{"milestone": "m1", "code": p, "mode": "perf", "name": name}, {"milestone": "m1", "code": p, "mode": "perf", "name": "X(y)"}],
            "milestones": [{"id": "m1", "ops": ["MUL_MAT_ID"], "checks": ["golden", "perf"], "case_pool": [], "perf_cases": ["MUL_MAT_ID(f32)"]}]}
    got = g.adopt_extra_cases(spec, [name], [name])
    assert spec["milestones"][0]["perf_cases"][0] == name and got["missing"] == ["X(y)"]
    assert spec["milestones"][0]["case_pool"] == []  # a perf-mode case does not join the correctness pool


def test_blackwell_consumer_builds_its_arch_specific_target():
    from crucible.factory.lanes import kernel_harness as kh

    assert kh.cuda_arch_flag(["120"]) == "120a-real" and kh.cuda_arch_flag(["90", "121"]) == "90-real;121a-real"


def test_apptainer_hosts_command_and_definition():
    from crucible.factory.lanes import kernel_plan as kp

    assert kp.sif_name("crucible-lh-k:base-12.8.1") == "crucible-lh-k_base-12.8.1.sif"
    user = kp.apptainer_argv("/s/x.sif", [("/h/a", "/k/a", False), ("/h/o", "/out", True)], ["python3", "-V"], env=(("CRUCIBLE_CUDA_ARCH", "90"),),
                             root="user", tmp="/n/t1")
    assert user[:4] == ["apptainer", "exec", "--nv", "--containall"] and "--fakeroot" not in user  # DeltaAI: one uid, no fake root
    assert user[user.index("--network") + 1] == "none" and "CRUCIBLE_CUDA_ARCH=90" in user
    assert ["-B", "/n/t1:/tmp"] == user[user.index("/n/t1:/tmp") - 1:user.index("/n/t1:/tmp") + 1]  # a build fits node disk, not the session tmpfs
    assert "/h/a:/k/a:ro" in user and "/h/o:/out" in user and user[-3:] == ["/s/x.sif", "python3", "-V"]
    assert "--fakeroot" in kp.apptainer_argv("/s/x.sif", [], ["true"])  # the default keeps the grader root inside
    with pytest.raises(ValueError):
        kp.apptainer_argv("/s/x.sif", [], ["true"], root="sudo")
    d = kp.apptainer_def("FROM nvidia/cuda:12.8.1-devel-ubuntu24.04\nENV DEBIAN_FRONTEND=noninteractive\n"
                         "RUN apt-get update && apt-get install -y \\\n    ccache\nENV PATH=/usr/local/cuda/bin:$PATH\n")
    assert d.startswith("Bootstrap: docker\nFrom: nvidia/cuda:12.8.1-devel-ubuntu24.04\n")
    post, env = d.split("%environment")
    assert "APT::Sandbox::User" in post and "apt-get install -y      ccache" in post and post.index("DEBIAN_FRONTEND") < post.index("apt-get")
    assert "export PATH=/usr/local/cuda/bin:$PATH" in env
    with pytest.raises(ValueError):
        kp.apptainer_def("FROM x\nCOPY a b\n")


def test_gpu_runner_apptainer_user_mode_drops_require_root(monkeypatch, tmp_path):
    g = load_gpu_runner()
    seen = []
    monkeypatch.setattr(g, "sh", lambda cmd, **kw: seen.append(cmd) or subprocess.CompletedProcess(cmd, 0, "", ""))
    monkeypatch.setattr(g, "APPTAINER_ROOT", "user")
    monkeypatch.setattr(g, "APPTAINER_TMP", tmp_path / "t")
    monkeypatch.setattr(g, "SIF_DIR", tmp_path / "sif")
    monkeypatch.setattr(g, "USE_CCACHE", False)
    d = tmp_path / "task"
    (d / "ref").mkdir(parents=True)
    patch, spec = tmp_path / "p.patch", tmp_path / "s.json"
    patch.write_text("")
    spec.write_text("{}")
    r = g.grade(d, patch, spec, "apptainer", "90", "oracle")
    cmd = seen[-1]
    assert cmd[0] == "apptainer" and "--fakeroot" not in cmd and "--require-root" not in cmd and str(tmp_path / "sif" / "crucible-lh-k_base-12.8.1.sif") in cmd
    assert r["infra"]["status"] == "masked"  # no reward file: masked, never a pass
    assert not list((tmp_path / "t").iterdir())  # each container's /tmp is removed afterwards
    monkeypatch.setattr(g, "APPTAINER_ROOT", "fakeroot")
    g.grade(d, patch, spec, "apptainer", "90", "oracle")
    assert "--fakeroot" in seen[-1] and "--require-root" in seen[-1]
    g.grade(d, patch, spec, "docker", "120", "oracle")
    assert seen[-1][:3] == ["docker", "run", "--rm"] and "--require-root" in seen[-1]  # the 5090 path is unchanged
    with pytest.raises(RuntimeError, match="build-sif"):
        g.ensure_image("apptainer")


def test_start_gate_compiles_untouched_units_that_use_what_a_fault_takes_away(tmp_path):
    lay = ko.Layout(cuda_dir="g/cuda/", dispatch_file="g/cuda/main.cu")
    for side, extra in (("b", "static __device__ uint32_t fastdiv(uint32_t n, uint3 v) { return n; }\n"), ("a", "")):
        d = tmp_path / side / "g" / "cuda"
        d.mkdir(parents=True)
        (d / "common.cuh").write_text("// fastdiv: see the paper\n#pragma once\n" + extra)
        (d / "main.cu").write_text('#include "common.cuh"\nint main() { return 0; }\n')
        (d / "norm.cu").write_text('#include "common.cuh"\n__global__ void k(uint32_t n, uint3 v) { n = ' + ("fastdiv(n, v)" if side == "b" else "n / v.z") + "; }\n")
        (d / "binbcast.cu").write_text('#include "common.cuh"\n__global__ void b(uint32_t n, uint3 v) { n = fastdiv(n, v); }\n')  # no fault touches it
        (d / "other.cu").write_text('#include "common.cuh"\n// fastdiv would help here\n__global__ void o() {}\n')
    got = kd.start_dependents(tmp_path / "a", tmp_path / "b", ["g/cuda/common.cuh", "g/cuda/norm.cu"], lay)
    assert got == ["g/cuda/binbcast.cu"]  # a mention in a comment is not a use; the touched unit is compiled anyway
    assert kd.start_dependents(tmp_path / "a", tmp_path / "b", ["g/cuda/norm.cu"], lay) == []  # no header change: nothing taken away


def test_the_kernel_lint_refuses_a_diagnosis_in_a_symptom_report():
    from crucible.factory.lanes.kernel_debug import CAUSE

    bad = ("FLASH_ATTN_EXT fails because the kernel incorrectly includes out-of-bounds values in the softmax maximum reduction",
           "ARGSORT fails for large inputs due to integer overflow in temporary buffer size calculations")
    good = ("FLASH_ATTN_EXT reports NMSE above the 5e-4 threshold when the KV length is not a multiple of 256",
            "ARGSORT returns wrong indices for 70000 x 40000 inputs and is correct for smaller ones")
    assert all(CAUSE.search(x) for x in bad) and not any(CAUSE.search(x) for x in good)


def test_ownership_needs_a_break_on_the_fixed_seeds():
    """2026-10-23: cases owned on a break seen on one fresh seed of the calibration let the same revert pass its
    milestone at S8 (kd-bug-llamacpp-a45699aa m3): ownership reads the fixed-seed verdicts."""
    g = load_gpu_runner()
    grade = {"checks": [{"id": "m1:golden", "detail": {"per_case": {"A": False, "B": False}, "per_case_fixed": {"A": False, "B": True}}}]}
    assert g.per_case_fixed(grade, "m1") == {"A": False, "B": True}
    assert g.owned_cases({"A": True, "B": True}, g.per_case_fixed(grade, "m1"), []) == ["A"]
    old = {"checks": [{"id": "m1:golden", "detail": {"per_case": {"A": False}}}]}
    assert g.per_case_fixed(old, "m1") == {"A": False}  # a grade from before the field: every seed


def test_a_sanitizer_dirty_on_b_keeps_no_milestone():
    g = load_gpu_runner()
    assert g.sanitizer_discriminates({"clean_on_b": True, "dirty_on_revert": True})
    assert not g.sanitizer_discriminates({"clean_on_b": False, "dirty_on_revert": True})  # held-out 44b81230 m1 (2026-10-23)
    assert not g.sanitizer_discriminates({})


def test_exact_attention_reference_only_where_two_computations_agree_against_the_cpu():
    g = load_gpu_runner()
    tol = 5e-4
    ev = {"cpu_wrong": {"tol": tol, "applied": True, "gpu_vs_cpu": 0.25, "gpu_vs_exact": 1e-7, "cpu_vs_exact": 0.2, "gpu_vs_gpu": 0.0},
          "agree": {"tol": tol, "applied": True, "gpu_vs_cpu": 1e-5, "gpu_vs_exact": 1e-7, "cpu_vs_exact": 1e-5, "gpu_vs_gpu": 0.0},
          "gpu_bad": {"tol": tol, "applied": True, "gpu_vs_cpu": 0.25, "gpu_vs_exact": 0.3, "cpu_vs_exact": 1e-6, "gpu_vs_gpu": 0.0},
          "noisy": {"tol": tol, "applied": True, "gpu_vs_cpu": 0.25, "gpu_vs_exact": 1e-7, "cpu_vs_exact": 0.2, "gpu_vs_gpu": 1e-4},
          "sinks": {"tol": tol, "applied": False, "gpu_vs_cpu": 0.25, "gpu_vs_exact": 0.25, "cpu_vs_exact": 0.0, "gpu_vs_gpu": 0.0}}
    assert {c: g.fa_reference_verdict(e) for c, e in ev.items()} == {
        "cpu_wrong": "cpu_wrong", "agree": "agree", "gpu_bad": "gpu_disagrees", "noisy": "nondeterministic", "sinks": "unmodelled"}
    few = g.fa_exact_choice(ev)
    assert not few["validated"] and few["exact"] == []  # the exact computation agreed with the CPU on too few cases
    many = {**ev, **{f"ok{i}": dict(ev["agree"]) for i in range(8)}}
    choice = g.fa_exact_choice(many)
    assert choice["validated"] and choice["exact"] == ["cpu_wrong"] and choice["verdicts"]["agree"] == 9


def load_migration_runner():
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_gpu.py"
    spec = importlib.util.spec_from_file_location("lh_kernel_gpu", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_migration_s8_reads_the_report_gates_and_the_mutant_s_own_milestones(tmp_path):
    """2026-10-23: S8 read the reward document's ``gates``, which this plan leaves empty, so no fault mutant ever counted
    as killed. Gates now come from the report (or the check components), and a mutant is killed when a milestone of the
    program it is aimed at fails."""
    m = load_migration_runner()
    (tmp_path / "report.json").write_text('{"gates": {"c02": "pass", "r02": "fail", "r03": "pass"}}')
    assert m.gates_of(tmp_path, {"gates": {}}) == {"c02": "pass", "r02": "fail", "r03": "pass"}
    assert m.gates_of(tmp_path / "none", {"components": {"check/r02": 0.0, "check/c02": 1.0, "time/x": 3.0}}) == {"r02": "fail", "c02": "pass"}
    spec = {"milestones": [{"id": "c02", "programs": ["p_soft"]}, {"id": "r02", "programs": ["p_soft"]}, {"id": "r03", "programs": ["p_moe"]}]}
    assert m.mutant_targets("mutant_fault-shfl_down_to_up-p_soft", spec) == ["c02", "r02"]
    killed = {"gates": {"c02": "pass", "r02": "fail", "r03": "pass"}}
    elsewhere = {"gates": {"c02": "pass", "r02": "pass", "r03": "fail"}}
    assert m.mutant_killed("mutant_fault-x-p_soft", killed, spec) and not m.mutant_killed("mutant_fault-x-p_soft", elsewhere, spec)


def test_migration_s10_leaves_masked_grades_out():
    m = load_migration_runner()
    ok, masked = {"reward": 1.0, "infra": {"status": "ok"}}, {"reward": None, "infra": {"status": "masked"}}
    v = m.flake_verdict([ok] * 59 + [masked])
    assert v["false_negatives"] == 0 and v["masked"] == 1 and v["measured"] == 59 and v["cp_upper"] < 0.05
    assert m.flake_verdict([ok] * 59 + [{"reward": 0.0, "infra": {"status": "ok"}}])["cp_upper"] > 0.05


def test_perf_calibration_rejects_timings_that_measured_nothing():
    g = load_gpu_runner()
    assert g.plausible_timing(0.97, 1.03, 1.6)
    assert not g.plausible_timing(1.0, 0.0, 1.6)  # B against itself at 0: the run timed nothing
    assert not g.plausible_timing(1.15, 1007.2, 8551.7)  # GH200 2026-10-24: kept with a bar of 3019 before this rule
    assert not g.plausible_timing(1.0, 1.0, 2902.0)  # a 2902x regression is a broken timing, not a speed-up undone
    assert not g.plausible_timing(0.49, 0.62, 0.31)


def test_a_report_never_asks_for_a_milestone_the_gpu_calibration_dropped():
    """2026-10-25: the GPU ingest dropped cef66d4a's m2 and kept the old prompt, which still asked for R2 (V5). The ingest
    now re-authors from prompts/<id>.json, and the kernel targeting preflight refuses a report with an ungraded requirement."""
    from crucible.factory.lanes import kernel_debug as kd

    rec = {"milestones": [{"id": "m1"}, {"id": "m3"}],
           "requirements": [{"id": "R1", "milestone": "m1"}, {"id": "R2", "milestone": "m2"}, {"id": "R3", "milestone": "m3"}]}
    assert kd.ungraded_requirements(rec) == ["R2"]
    assert kd.ungraded_requirements({**rec, "requirements": [q for q in rec["requirements"] if q["id"] != "R2"]}) == []
    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_debug.py"
    src = p.read_text()
    assert "reauthor(rec)\n        kd.save(store, rec)" in src  # the ingest re-authors before it saves


def test_s7k_the_report_states_what_sets_a_milestone_s_cases_apart():
    """S7k on 1b8b8fb1's shape (2026-10-25): R2 said FLASH_ATTN_EXT "picks a kernel for some K/V layouts that it cannot
    handle" while all of m2's cases have head sizes 576/512; a sharpened requirement passes; ops, sanitizer symptoms and
    the two-way map are checked too."""
    from crucible.factory.lanes import kernel_debug as kd

    fa = "FLASH_ATTN_EXT(hsk={h},hsv={v},nh=4,nr23=[{r},1],kv=512,nb={b},mask=1,prec=f32,type_KV={t},permute=[0,1,2,3])"
    rec = {"id": "kd-x", "family": "kernel-bug", "milestones": [
        {"id": "m1", "checks": ["golden", "racecheck"], "cases": ["CUMSUM(type=f32,ne=[10,5,4,3])"]},
        {"id": "m2", "checks": ["golden"], "cases": [fa.format(h=576, v=512, r=4, b=1, t=t) for t in ("q4_0", "q8_0")]},
        {"id": "m3", "checks": ["golden"], "cases": [fa.format(h=128, v=128, r=12, b=3, t=t) for t in ("f16", "f32")]}],
        "requirements": [{"id": "R1", "milestone": "m1", "text": "CUMSUM differs between runs and compute-sanitizer's race check flags it."},
                         {"id": "R2", "milestone": "m2", "text": "FLASH_ATTN_EXT picks a kernel for some K/V layouts that it cannot handle."},
                         {"id": "R3", "milestone": "m3", "text": "FLASH_ATTN_EXT is wrong when query heads per KV head is not a power of two."}]}
    got = kd.report_gate(rec)
    assert len(got) == 1 and got[0].startswith("S7k undisclosed: m2's FLASH_ATTN_EXT") and "hsk=576" in got[0]
    sharp = {**rec, "requirements": [*rec["requirements"][:1], {"id": "R2", "milestone": "m2", "text": "FLASH_ATTN_EXT is wrong for head size 576 with a value size of 512."}, rec["requirements"][2]]}
    assert kd.report_gate(sharp) == []
    no_race = {**rec, "requirements": [{"id": "R1", "milestone": "m1", "text": "CUMSUM is sometimes wrong."}, *sharp["requirements"][1:]]}
    assert any(f.startswith("S7k check: m1") for f in kd.report_gate(no_race))
    missing = {**sharp, "requirements": sharp["requirements"][:2]}
    assert "S7k unasked: milestone m3 has no requirement" in kd.report_gate(missing)
    assert kd.parse_case(fa.format(h=1, v=2, r=3, b=4, t="f16"))[1]["nr23"] == "[3,1]"


def test_k6_states_only_parameters_the_milestone_s_cases_have():
    """A K6 revision may only name values its hidden cases carry (the cases are the GPU-executed reproduction)."""
    g = load_gpu_lane_script()
    rec = {"milestones": [{"id": "m2", "cases": ["FLASH_ATTN_EXT(hsk=576,hsv=512,nh=4,nr23=[4,1],kv=512,type_KV=q4_0)"]}]}
    assert g.k6_claims_ok(rec, "m2", "FLASH_ATTN_EXT is wrong for head size 576 with value size 512 and a quantized cache.") == []
    assert g.k6_claims_ok(rec, "m2", "FLASH_ATTN_EXT is wrong for head size 256.") == ["K6 claim: 256 is no parameter of m2's cases"]
    # a list value is not one number (cef66d4a's first K6 said bs=[1,1], read as 11), a thousands separator is
    mm = {"milestones": [{"id": "m3", "cases": ["MUL_MAT(type_a=q4_0,m=16,n=70000,k=256,bs=[1,1],nr=[1,1])"]}]}
    assert g.k6_claims_ok(mm, "m3", "MUL_MAT fails for n=70,000 with bs=[1,1] and nr=[1,1].") == []


def load_gpu_lane_script():
    import importlib.util

    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_debug.py"
    spec = importlib.util.spec_from_file_location("lh_kernel_debug_k6", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_a_revision_never_points_at_the_location(tmp_path, monkeypatch):
    """Finding the fault is the skill (2026-10-26 guard): the location probe refuses location words and any word of the
    requirement that a grep of the start tree's CUDA backend finds only in the milestone's fault files."""
    g = load_gpu_lane_script()
    be = tmp_path / "store" / "train" / "t" / "a" / "ggml" / "src" / "ggml-cuda"
    be.mkdir(parents=True)
    (be / "fattn-mma-f16.cuh").write_text("void flash_attn_ext_f16_process_tile() { __syncthreads(); } // tile_barrier_rows\n")
    (be / "fattn.cu").write_text("void dispatch() { /* FLASH_ATTN_EXT */ }\n")
    monkeypatch.setattr(g, "ROOT", tmp_path)
    rec = {"id": "t", "split": "train", "tree_a": "train/t/a", "milestones": [{"id": "m2", "files": ["ggml/src/ggml-cuda/fattn-mma-f16.cuh"],
                                                                         "functions": ["flash_attn_ext_f16_process_tile"]}]}
    assert g.location_probe(rec, "m2", "FLASH_ATTN_EXT with f16 caches and head size 128 hangs.") == []
    got = g.location_probe(rec, "m2", "FLASH_ATTN_EXT hangs on tile_barrier_rows; the dispatcher picks the wrong path.")
    assert any(x.startswith("location grep: 'tile_barrier_rows'") for x in got) and any("dispatcher" in x for x in got)


def test_a_migration_window_has_an_intent_card():
    """2026-10-26: the kernel targeting mine stage raised KeyError 'class' on a migration window (its milestones are
    compile/run checks of programs, with no fault class)."""
    from crucible.factory.lanes import kernel_debug as kd

    own = {"intent": "migrate", "skills": ["toolkit_migration", "kernel_fault_localization"]}
    card = kd.task_intent({"family": "kernel-migration", "intent": own, "milestones": [{"id": "c01"}, {"id": "r01"}]})
    assert card == own
