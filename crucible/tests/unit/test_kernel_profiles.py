"""Kernel harness profiles (swarm-pipeline/10 §4.1): the ggml profile is ``kernel_harness`` unchanged (recorded
verdicts below were produced by ``kernel_harness`` itself before profiles existed), and the Catch2 profile reads CUB's
self-checking tests."""

from __future__ import annotations

import json
import struct

import pytest

from crucible.factory.lanes import kernel_harness as kh
from crucible.factory.lanes import kernel_profiles as kp


def rec(case: str, node: int, op: str, max_err: float, out: list[float], ref: list[float]) -> bytes:
    c, o = case.encode(), op.encode()
    return (b"CRD1" + struct.pack("<I", len(c)) + c + struct.pack("<I", len(o)) + o + struct.pack("<idQ", node, max_err, len(out))
            + struct.pack(f"<{len(out)}f", *out) + struct.pack(f"<{len(ref)}f", *ref))


CONSOLE = ("  MUL_MAT(type_a=f16,type_b=f32,m=16,n=1,k=256): OK\n"
           "  MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256): \x1b[1;31mFAIL\x1b[0m\n"
           "  FLASH_ATTN_EXT(hs=64,nh=4): not supported [CUDA0]\n"
           "  ARGSORT(type=f32,ne=[70000,1,1,1]): ")
LISTING = "CRUCIBLE_CASE A(x)\nCRUCIBLE_CASE B(y)\nother line\nCRUCIBLE_CASE A(x)\n"
REF = rec("A(x)", 0, "A", 1e-6, [1.0, 2.0], [1.0, 2.0]) + rec("B(y)", 0, "B", 5e-4, [5.0], [5.0])
CAND = rec("A(x)", 0, "A", 9.9, [1.0, 2.0], [7.0, 7.0]) + rec("B(y)", 0, "B", 9.9, [6.0], [0.0])

# recorded with kernel_harness at 5ab1f1bb, before kernel_profiles existed
RECORDED = {
    "status": {"MUL_MAT(type_a=f16,type_b=f32,m=16,n=1,k=256)": "OK", "MUL_MAT(type_a=q4_0,type_b=f32,m=16,n=8,k=256)": "FAIL",
               "FLASH_ATTN_EXT(hs=64,nh=4)": "not supported"},
    "crashed": "ARGSORT(type=f32,ne=[70000,1,1,1])",
    "listed": ["A(x)", "B(y)"],
    "compare": {"A(x)": {"pass": True, "nmse_vs_cpu": 0.0, "nmse_vs_b": 0.0, "max_err": 1e-6},
                "B(y)": {"pass": False, "nmse_vs_cpu": 0.04, "nmse_vs_b": 0.04, "max_err": 5e-4},
                "C(z)": {"pass": False, "error": "reference did not run the case (infra)", "infra": True}},
}


def test_the_ggml_profile_returns_kernel_harness_s_recorded_verdicts():
    g = kp.GGML
    assert g.status(CONSOLE) == kh.case_status(CONSOLE) == RECORDED["status"]
    assert g.crashed(CONSOLE) == kh.crashed_case(CONSOLE) == RECORDED["crashed"]
    assert g.listed(LISTING) == kh.listed_cases(LISTING) == RECORDED["listed"]
    assert g.read_dump(REF) == kh.read_dump(REF)
    got = g.compare({"recs": g.read_dump(CAND)}, {"recs": g.read_dump(REF)}, ["A(x)", "B(y)", "C(z)"])
    assert got == kh.compare_dumps(kh.read_dump(CAND), kh.read_dump(REF), cases=["A(x)", "B(y)", "C(z)"])
    assert json.dumps(got, sort_keys=True) == json.dumps(RECORDED["compare"], sort_keys=True)
    assert g.cases_file(["A(x)", "B(y)"]) == kh.cases_file(["A(x)", "B(y)"]) == "A(x)\nB(y)\n"


def test_the_ggml_profile_builds_and_runs_exactly_as_the_grader_did():
    g = kp.GGML
    assert (g.harness, g.target, g.exe, g.device_glob) == (kh.HARNESS, kh.TARGET, "bin/test-backend-ops", "libggml-cuda*")
    assert g.cmake_cuda(["120"]) == kh.cmake_cuda(["120"]) and g.cmake_cpu() == kh.CMAKE_CPU
    assert g.argv("/k/ref/bin/test-backend-ops", "perf", "CUDA0", "/x/cases.txt") == ["/k/ref/bin/test-backend-ops", "perf", "-b", "CUDA0"]
    assert g.list_argv("/k/ref/bin/test-backend-ops", "test") == (["/k/ref/bin/test-backend-ops", "test", "-b", "CUDA0"], {"CRUCIBLE_LIST": "1"})
    # the grader's environment before profiles, key for key and in order
    assert list(g.env(7, "/d/cases.txt", "/d/ref.dump", reps=3).items()) == [
        ("CRUCIBLE_SEED", "7"), ("CRUCIBLE_CASES", "/d/cases.txt"), ("GGML_CUDA_DISABLE_GRAPHS", "1"), ("CUDA_DISABLE_PTX_JIT", "1"),
        ("CRUCIBLE_DUMP", "/d/ref.dump"), ("CRUCIBLE_REPS", "3")]
    assert g.env(7, "/d/c", None) == {"CRUCIBLE_SEED": "7", "CRUCIBLE_CASES": "/d/c", "GGML_CUDA_DISABLE_GRAPHS": "1", "CUDA_DISABLE_PTX_JIT": "1"}
    assert g.host_flags({"ggml/src/ggml-cuda.cu": True}) == kh.HOST_CPU_FLAGS["ggml/src/ggml-cuda.cu"] and g.host_flags({}) == ()
    assert g.FA_REF_ENV == kh.FA_REF_ENV and g.EXACT_LINE is kh.EXACT_LINE


def test_a_spec_without_a_harness_is_ggml_and_unknown_profiles_are_refused():
    assert kp.of_spec({}) is kp.GGML and kp.of_spec({"harness": "ggml"}) is kp.GGML
    with pytest.raises(KeyError):
        kp.profile("no-such-harness")


CATCH2_XML = """<?xml version="1.0" encoding="UTF-8"?>
<Catch2TestRun name="cub.test.device.reduce_by_key.lid_0.types_0" rng-seed="1">
  <TestCase name="Device reduce-by-key works - uint8_t" tags="[by_key]" filename="t.cu" line="10">
    <OverallResult success="true" skips="0"/>
  </TestCase>
  <TestCase name="Device reduce-by-key compares keys before converting them to the output key type" tags="[by_key]">
    <Expression success="false" type="REQUIRE"><Original>num_unique_keys[0] == 4</Original></Expression>
    <OverallResult success="false" skips="0"/>
  </TestCase>
  <TestCase name="a &lt;b&gt; &amp; c" tags="[x]">
    <OverallResult success="true"/>
  </TestCase>
  <TestCase name="died here" tags="[x]">
"""


def test_the_catch2_profile_reads_catch2_s_own_verdict_per_case():
    h = kp.cccl(["cub/test/catch2_test_device_reduce_by_key.cu"])
    st = h.status(CATCH2_XML)
    assert st == {"Device reduce-by-key works - uint8_t": "OK",
                  "Device reduce-by-key compares keys before converting them to the output key type": "FAIL", "a <b> & c": "OK"}
    assert h.crashed(CATCH2_XML) == "died here"
    ref = {"status": {**st, "Device reduce-by-key compares keys before converting them to the output key type": "OK"}}
    cmp = h.compare({"status": st}, ref, [*st, "never ran"])
    assert cmp["Device reduce-by-key works - uint8_t"]["pass"] and not cmp["Device reduce-by-key compares keys before converting them to the output key type"]["pass"]
    assert cmp["never ran"]["infra"]  # the reference must pass a case before it is graded
    assert not h.compare({"status": {}}, {"status": {"x": "OK"}}, ["x"])["x"]["pass"]  # a case the candidate never reported fails
    assert h.compare_mode == "status" and not h.seeded and not h.perf


def test_catch2_specs_match_names_literally():
    assert kp.catch2_spec("plain name") == "plain name"
    assert kp.catch2_spec("a [tag], b*") == r"a \[tag\]\, b\*"
    assert kp.catch2_spec("~not negated") == r"\~not negated" and kp.catch2_spec('say "hi"') == r'say \"hi\"'
    h = kp.cccl(["cub/test/catch2_test_device_reduce_by_key.cu"])
    assert h.cases_file(["x,y", "z"]) == "x\\,y\nz\n"
    assert h.argv("/b/bin/t", "test", "CUDA0", "/w/cases.txt") == ["/b/bin/t", "-r", "xml", "--input-file", "/w/cases.txt"]
    with pytest.raises(ValueError):
        h.argv("/b/bin/t")
    assert h.listed("Device a\nDevice b\nDevice a\n") == ["Device a", "Device b"]


def test_cub_targets_follow_the_configured_build_s_names():
    # names as `ninja -t targets all` printed them on the 5090 (CCCL 3.7, 2026-10-24)
    assert kp.cccl_target("cub/test/catch2_test_device_reduce_by_key.cu") == ("cub.test.device.reduce_by_key.lid_0.types_0",
                                                                              "bin/cub.test.device.reduce_by_key.lid_0.types_0")
    assert kp.cccl_target("cub/test/catch2_test_device_histogram.cu", lid=1, types=None)[0] == "cub.test.device.histogram.lid_1"
    ninja = ("cub.test.device.reduce_by_key: phony\ncub.test.device.reduce_by_key.lid_0.types_0: CUDA_EXECUTABLE_LINKER__x\n"
             "cub.test.device.reduce_by_key.lid_2.types_3: CUDA_EXECUTABLE_LINKER__x\ncub.test.device.reduce.lid_0.types_0: x\n")
    assert kp.ninja_targets(ninja, "cub/test/catch2_test_device_reduce_by_key.cu") == ["cub.test.device.reduce_by_key.lid_0.types_0",
                                                                                       "cub.test.device.reduce_by_key.lid_2.types_3"]
    h = kp.of_spec({"harness": "catch2-cccl", "harness_tests": {"cub/test/catch2_test_device_scan.cu": ["cub.test.device.scan.lid_1.types_2",
                                                                                                         "bin/cub.test.device.scan.lid_1.types_2"]}})
    assert (h.target, h.exe) == ("cub.test.device.scan.lid_1.types_2", "bin/cub.test.device.scan.lid_1.types_2")
    assert all(f in h.cmake_cuda(["90"]) for f in ("-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++", "-DCMAKE_CUDA_ARCHITECTURES=90-real"))


def test_the_catch2_profile_writes_b_s_test_sources_into_the_grader_s_tree(tmp_path):
    h = kp.cccl(["cub/test/catch2_test_device_reduce_by_key.cu"])
    (tmp_path / "cub" / "test").mkdir(parents=True)
    (tmp_path / "cub" / "test" / "catch2_test_device_reduce_by_key.cu").write_text("// the agent's copy")
    h.prepare(tmp_path, json.dumps({"cub/test/catch2_test_device_reduce_by_key.cu": "// B's test"}))
    assert (tmp_path / "cub" / "test" / "catch2_test_device_reduce_by_key.cu").read_text() == "// B's test"


def test_held_out_repositories_come_from_the_local_table_only(tmp_path):
    assert set(kp.REPOS.values()) <= {"ggml", "catch2-cccl"} and kp.kind_for("cccl", local={}) == "catch2-cccl"
    assert kp.kind_for("llamacpp", local={}) == "ggml"
    table = tmp_path / "harness_repos.json"
    table.write_text(json.dumps({"some-heldout": "pytest-jit"}))
    assert kp.local_repos(table) == {"some-heldout": "pytest-jit"}
    assert kp.kind_for("some-heldout", local=kp.local_repos(table)) == "pytest-jit"
    with pytest.raises(KeyError):
        kp.kind_for("some-heldout", local={})
    assert kp.local_repos(tmp_path / "missing.json") == {}


def test_a_ggml_reference_keeps_its_identity():
    """The reference's stamp hashes ``kernel_harness`` only for ggml (references built before profiles stay valid); any
    other profile adds its own code and description."""
    import hashlib
    import importlib.util
    from pathlib import Path

    p = Path(__file__).resolve().parents[2] / "scripts" / "lh_kernel_debug_gpu.py"
    spec = importlib.util.spec_from_file_location("lh_kernel_debug_gpu", p)
    g = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g)
    old = "harness " + hashlib.sha256(Path(kh.__file__).read_bytes()).hexdigest()[:16]
    assert g.harness_digest() == g.harness_digest(kp.GGML) == old
    assert g.harness_digest(kp.cccl(["cub/test/catch2_test_device_reduce_by_key.cu"])) != old


def test_a_ggml_kdspec_s_harness_table_is_not_a_profile_name():
    """2026-10-25: every ggml kdspec carries ``harness`` as {file, dialect} (kernel_debug.kdspec); of_spec read the table as a
    profile name and every kernel-bug episode grade raised TypeError (all masked)."""
    assert kp.of_spec({"harness": {"file": "tests/test-backend-ops.cpp", "dialect": "2025"}}) is kp.GGML
    assert kp.of_spec({"harness": {"file": "x", "dialect": "2025", "profile": "ggml"}}) is kp.GGML
    from pathlib import Path

    from crucible.factory.lanes import kernel_debug as kd

    store = Path(__file__).resolve().parents[2] / "data" / "kernel_debug" / "store"
    recs = list(kd.records(store)) if store.exists() else []
    for rec in recs[:5]:
        assert kp.of_spec(kd.kdspec(rec)) is kp.GGML


def test_a_ggml_spec_whose_harness_is_the_op_test_descriptor_is_a_ggml_task():
    """DeltaAI smoke 3462338: calibrated ggml specs carry ``harness: {"dialect", "file"}`` (the op test's descriptor, older
    than profiles); ``of_spec`` read it as a profile name and failed every episode grade with an unhashable dict."""
    spec = {"harness": {"dialect": "ggml-2025", "file": "tests/test-backend-ops.cpp"}}
    assert kp.of_spec(spec) is kp.profile("ggml")
    assert kp.of_spec({}) is kp.profile("ggml")
