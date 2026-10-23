"""Kernel debugging lane, hardware reachability (swarm-pipeline/10 §3.1): which of our GPUs (GH200 sm_90, RTX 5090
sm_120) compile and dispatch a change, the message rules (speed-only, launch-shape-only, precision-only selection,
hardware we do not have), op-test coverage, and retirement. The synthetic snippets mirror the retired tasks' faults; the
store test runs the filter on the real retired and passing tasks when the lane's clones are present. No GPU, no Docker."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from crucible.factory.lanes import kernel_debug as kd
from crucible.factory.lanes import kernel_debug_mine as kdm
from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.lanes import kernel_debug_reach as kr

COMMON = """#pragma once
#define GGML_CUDA_CC_PASCAL          600
#define GGML_CUDA_CC_VOLTA           700
#define GGML_CUDA_CC_TURING          750
#define GGML_CUDA_CC_AMPERE          800
#define GGML_CUDA_CC_ADA_LOVELACE    890
#define GGML_CUDA_CC_HOPPER          900
#define GGML_CUDA_CC_BLACKWELL       1200
#define GGML_CUDA_CC_OFFSET_AMD      0x1000000
#define GGML_CUDA_CC_OFFSET_MTHREADS 0x0100000
#define GGML_CUDA_CC_IS_NVIDIA(cc)   (cc < GGML_CUDA_CC_OFFSET_MTHREADS)
#define GGML_CUDA_CC_RDNA3      (GGML_CUDA_CC_OFFSET_AMD + 0x1100)
#define GGML_CUDA_CC_IS_AMD(cc)     (cc >= GGML_CUDA_CC_OFFSET_AMD)
#define GGML_CUDA_CC_IS_RDNA3(cc)   (cc >= GGML_CUDA_CC_RDNA3)
#if !defined(GGML_USE_HIP) && __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE
#define AMPERE_MMA_AVAILABLE
#endif
static bool turing_mma_available(const int cc) { return GGML_CUDA_CC_IS_NVIDIA(cc) && cc >= GGML_CUDA_CC_TURING; }
"""
HARNESS = "static ggml_tensor * build(ggml_context * ctx) { return ggml_soft_max(ctx, a); }\nstruct test_norm { ggml_tensor * f() { return ggml_norm(ctx, a, eps); } };\n"


def targets(before: str, after: str) -> list[str]:
    return [name for name, cc in kr.TARGETS.items() if kr.changed_on(before, after, COMMON, COMMON, cc)]


def test_an_amd_only_selection_reaches_neither_gpu() -> None:
    # 7ac249cc m2: the NVIDIA branch is untouched; only the AMD fallthrough moves inside an IS_AMD test
    before = ("int max_batch(int type, int cc) {\n    if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {\n        return 8;\n    }\n"
              "    if (GGML_CUDA_CC_IS_RDNA3(cc)) {\n        return rdna3(type);\n    }\n    return 8;\n}\n")
    after = ("int max_batch(int type, int cc) {\n    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {\n        if (cc == GGML_CUDA_CC_VOLTA || cc >= GGML_CUDA_CC_ADA_LOVELACE) {\n"
             "            return 8;\n        }\n        return pascal(type);\n    }\n    // AMD\n    if (GGML_CUDA_CC_IS_AMD(cc)) {\n"
             "        if (GGML_CUDA_CC_IS_RDNA3(cc)) {\n            return rdna3(type);\n        }\n    }\n    return 8;\n}\n")
    assert targets(before, after) == []


def test_a_change_in_the_nvidia_branch_reaches_both() -> None:
    before = "int max_batch(int cc) {\n    if (GGML_CUDA_CC_IS_NVIDIA(cc)) {\n        return 8;\n    }\n    return 4;\n}\n"
    after = before.replace("return 8;", "return 16;")
    assert targets(before, after) == ["sm_90", "sm_120"]


def test_a_pre_ampere_device_path_reaches_neither_and_the_ampere_path_both() -> None:
    tmpl = ("__global__ void k_add(float * z, const float * x) {\n#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE\n    z[0] = {NEW};\n#else\n"
            "    z[0] = {OLD};\n#endif\n}\n")
    base = tmpl.replace("{NEW}", "x[0] + x[1]").replace("{OLD}", "x[0]")
    assert targets(base, base.replace("= x[0];", "= x[0] + x[1];")) == []
    assert targets(base, base.replace("x[0] + x[1]", "x[1] + x[0]")) == ["sm_90", "sm_120"]


def test_a_blackwell_only_path_reaches_the_5090_only() -> None:
    before = "__global__ void k(float * y) {\n#if __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL\n    y[0] = 1;\n#endif\n}\n"
    assert targets(before, before.replace("y[0] = 1;", "y[0] = 2;")) == ["sm_120"]


def test_hip_guards_and_feature_macros_follow_the_gpu() -> None:
    hip = "void f() {\n#if defined(GGML_USE_HIP)\n    a();\n#endif\n}\n"
    assert targets(hip, hip.replace("a();", "b();")) == []
    mma = "__device__ void f() {\n#ifdef AMPERE_MMA_AVAILABLE\n    a();\n#else\n    c();\n#endif\n}\n"
    assert targets(mma, mma.replace("c();", "d();")) == []  # both GPUs have Ampere MMA in the device pass
    assert targets(mma, mma.replace("a();", "b();")) == ["sm_90", "sm_120"]


def test_runtime_predicates_from_common_fold() -> None:
    before = "void f(int cc) {\n    if (!turing_mma_available(cc)) {\n        slow();\n    } else {\n        fast();\n    }\n}\n"
    assert targets(before, before.replace("slow();", "slower();")) == []
    assert targets(before, before.replace("fast();", "faster();")) == ["sm_90", "sm_120"]


def test_a_macro_value_and_an_undecided_guard_count_as_reached() -> None:
    m = "#define FATTN_KQ_MAX_OFFSET 0.6931f\nvoid f() { g(FATTN_KQ_MAX_OFFSET); }\n"
    assert targets(m, m.replace("0.6931f", "(3.0f*0.6931f)")) == ["sm_90", "sm_120"]
    cccl = "void f() {\n#ifdef STRIDED_ITERATOR_AVAILABLE\n    a();\n#else\n    b(n + 1);\n#endif\n}\n"  # set by the toolkit's CCCL version
    assert targets(cccl, cccl.replace("b(n + 1);", "b(n + 2);")) == ["sm_90", "sm_120"]


def test_identical_branches_merge_and_unbraced_returns_stay_conditional() -> None:
    before = "__global__ void k(T * z, const T * x) {\n    z[0] = x[0];\n}\n"
    after = ("__global__ void k(T * z, const T * x) {\n    if constexpr (std::is_same_v<T, nv_bfloat16>) {\n#if __CUDA_ARCH__ >= GGML_CUDA_CC_AMPERE\n"
             "        z[0] = x[0];\n#else\n        z[0] = cvt(x[0]);\n#endif\n    } else {\n        z[0] = x[0];\n    }\n}\n")
    assert targets(before, after) == []  # only the pre-Ampere bf16 branch differs
    loop = 'int f(int cc, graph * g) {\n    printf("{ %d", cc);\n    if (!g) return 0;\n    while (more()) {\n        if (stop()) break;\n    }\n    return 1;\n}\n'
    assert targets(loop, loop.replace("if (stop()) break;", "if (stop() || done(cc)) break;")) == ["sm_90", "sm_120"]


def test_message_rules() -> None:
    speed = kr.message_reasons("CUDA: Fix loop unrolling for BW in the stream-k fixup", "By providing stride_* variables as size_t the compiler can unroll the loops; some perf on BW",
                               "bug", "overflow", "-const int s,\n+const size_t s,\n", False)
    assert any("speed change" in r for r in speed)
    launch = kr.message_reasons("CUDA: fix thread/block count in quantized cpy kernel launches", "tests: add uneven block count cpy case", "bug",
                                "launch_config", "-    const int64_t num_blocks = ne;\n+    const int64_t num_blocks = (ne + 255) / 256;\n"
                                "-    k<<<num_blocks, 1, 0, stream>>>\n+    k<<<num_blocks, 256, 0, stream>>>\n", True)
    assert any("launch-shape" in r for r in launch)
    args = kr.message_reasons("Fix get_rows", "", "bug", "logic", "-    k<<<grid, 256, 0, s>>>(dst->ne[0],\n+    k<<<grid, 256, 0, s>>>(row_size/ts,\n", False)
    assert not args  # the launch's own arguments changed: not a reshape
    sel = kr.message_reasons("CUDA: fix MMV kernel being used for FP16 src1", "", "bug", "precision",
                             "-    if (!split && src0->type == GGML_TYPE_F16 && src1->ne[1] == 1) {\n+    if (!split && use_mul_mat_vec) {\n", False)
    assert any("selection between two kernels" in r for r in sel)
    assert kr.message_reasons("cuda : fix buffer type check with integrated GPUs", "", "bug", "logic", "", False)
    assert kr.message_reasons("Fix fp16 path for arch < Ampere", "", "bug", "logic", "", False)
    assert kr.message_reasons("CUDA: fix --split-mode row for MMQ", "", "bug", "logic", "", False)
    for subject in ("cuda: fix race condition in cumsum", "CUDA: fix norm family kernels when ne[2]/ne[3] exceed grid dim limits",
                    "ggml-cuda: fix divergent barrier in f16 flash attention", "CUDA: fix alignment check for FA"):
        assert kr.message_reasons(subject, "", "bug", "logic", "-a = b;\n+a = c;\n", False) == [], subject


def test_op_coverage_and_the_reach_verdict() -> None:
    assert kr.covered_ops(["SOFT_MAX", "INDEXER_TOPK"], HARNESS) == ["SOFT_MAX"]
    files = {"softmax.cu": ("void f() { a(); }\n", "void f() { b(); }\n")}
    ok = kr.reach(subject="CUDA: fix race in soft_max", body="", kind="bug", fault_class="race", diff="-a();\n+b();\n", adds_cases=False,
                  files=files, common=(COMMON, COMMON), ops=["SOFT_MAX"], harness=HARNESS)
    assert ok.qualifies and ok.targets == ["sm_90", "sm_120"]
    no = kr.reach(subject="Fix race in indexer topk", body="", kind="bug", fault_class="race", diff="-a();\n+b();\n", adds_cases=False,
                  files=files, common=(COMMON, COMMON), ops=["INDEXER_TOPK"], harness=HARNESS)
    assert not no.qualifies and "op test builds none" in no.reasons[0]
    assert no.to_json()["qualifies"] is False


def test_retire_moves_the_record_out_of_the_store(tmp_path: Path) -> None:
    store = tmp_path / "store"
    rec = {"id": "kd-bug-x-1", "split": "train", "family": "kernel-bug", "repo": "o/r", "milestones": []}
    kd.save(store, rec)
    (store / "train" / "kd-bug-x-1" / "a").mkdir()
    work = tmp_path / "work" / "kd-bug-x-1"
    work.mkdir(parents=True)
    prompt = tmp_path / "prompts" / "kd-bug-x-1.json"
    prompt.parent.mkdir()
    prompt.write_text("{}")
    dest = kd.retire(store, rec, "speed-only faults", extra={"work": work, "prompt.json": prompt, "gpu_bundle_task": tmp_path / "absent"}, at="t")
    assert kd.records(store) == [] and not work.exists() and not prompt.exists()
    assert (dest / "record" / "record.json").exists() and (dest / "record" / "a").is_dir() and (dest / "work").is_dir() and (dest / "prompt.json").exists()
    meta = kd.retired(store)
    assert meta[0]["status"] == kd.RETIRED and meta[0]["reason"] == "speed-only faults" and set(meta[0]["moved_from"]) == {"record", "work", "prompt.json"}


ROOT = Path(__file__).resolve().parents[2] / "data" / "kernel_debug"
RETIRED = ("kd-bug-llamacpp-7ac249cc", "kd-bug-llamacpp-4b8c156e", "kd-bug-llamacpp-b377258d", "kd-bug-ikllamacpp-5b19ab0c", "kd-bug-ikllamacpp-4da3f014")
PASSING = ("kd-bug-llamacpp-1b8b8fb1", "kd-bug-llamacpp-ca21f3a1", "kd-bug-llamacpp-efe7d4b1")


def _record(rid: str) -> dict | None:
    for p in (ROOT / "retired" / rid / "record" / "record.json", *ROOT.glob(f"store/*/{rid}/record.json")):
        if p.exists():
            return json.loads(p.read_text())
    return None


@pytest.mark.skipif(not (ROOT / "clones" / "ggml-org_llama.cpp").is_dir() or not (ROOT / "clones" / "ikawrakow_ik_llama.cpp").is_dir(),
                    reason="the kernel lane's clones are not on this machine")
@pytest.mark.parametrize("rid", RETIRED + PASSING)
def test_the_filter_on_the_real_tasks(rid: str) -> None:
    rec = _record(rid)
    if rec is None:
        pytest.skip(f"{rid} is not on this machine")
    repo = ROOT / "clones" / rec["source_clone"]
    try:
        subprocess.run(["git", "-C", str(repo), "cat-file", "-e", rec["faults"][0]["sha"]], check=True, capture_output=True)
    except subprocess.CalledProcessError:
        pytest.skip("the clone lacks the fault commits")
    faults = {f["sha"]: f for f in rec["faults"]}
    good = 0
    for m in rec["milestones"]:
        f = faults[m["fault"]]
        r = kdm.reach_of(repo, sha=f["sha"], parent=f["parent"], subject=f["subject"], body=f["body"], kind=f["kind"], fault_class=f["fault_class"],
                         files=f["files"], ops=m["ops"], adds_cases=f["adds_cases"], layout=ko.Layout(**rec["layout"]))
        good += r.qualifies
    assert (good >= 2) == (rid in PASSING), (rid, good)
