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
