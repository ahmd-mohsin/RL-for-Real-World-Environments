"""Kernel debugging lane, S1 (swarm-pipeline/10 §3): the CUDA language profile, the op map from the backend's dispatch,
the change filters and fault classes, grouping, and mining a synthetic repository end to end. git only, no Docker."""

from __future__ import annotations

import subprocess
from pathlib import Path

from crucible.factory.lanes import kernel_debug_mine as kdm
from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.langs import cuda

KERNELS = """#include "common.cuh"
// a comment with a fake definition: void not_a_function() { }
template <int D>
static __global__ void __launch_bounds__(128, 1) soft_max_kernel(const float * x, float * y, const int n) {
    __shared__ float buf[32];
    if (n > 0) { buf[threadIdx.x] = x[threadIdx.x]; }
    const char * s = "}{";
    y[threadIdx.x] = block_reduce(buf);
}
void ggml_cuda_op_soft_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    soft_max_kernel<64><<<1, 32>>>(nullptr, nullptr, 1);
}
"""
COMMON = "#pragma once\nstatic __device__ float block_reduce(float * b) {\n    return b[0];\n}\n#define CALL_NORM(x) norm_kernel<<<1, 1>>>(x)\n"
NORM = '#include "common.cuh"\nstatic __global__ void norm_kernel(float * x) { x[0] = block_reduce(x); }\nvoid ggml_cuda_op_norm(ggml_backend_cuda_context & ctx, ggml_tensor * dst) { CALL_NORM(nullptr); }\n'
DISPATCH = """#include "softmax.cuh"
static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        default:
            return false;
    }
    return true;
}
"""


def write(root: Path, files: dict[str, str]) -> Path:
    for f, t in files.items():
        p = root / f
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(t)
    return root


def test_cuda_profile_paths_and_definitions():
    p = cuda.PROFILE
    assert p.is_source("ggml/src/ggml-cuda/softmax.cu") and p.is_kernel_source("ggml/src/ggml-cuda/common.cuh")
    assert p.is_protected("tests/test-backend-ops.cpp") and p.is_protected("ggml/src/ggml-cuda/CMakeLists.txt") and p.is_protected("cmake/x.cmake")
    assert not p.is_source("tests/test-backend-ops.cpp") and not p.is_source("examples/x.cu")
    names = [n for n, _a, _b in cuda.find_funcs(KERNELS)]
    assert names == ["soft_max_kernel", "ggml_cuda_op_soft_max"]  # comments, strings and launch bounds do not confuse it
    assert p.kernels(KERNELS) == ["soft_max_kernel"]
    after = KERNELS.replace("buf[threadIdx.x] = x[threadIdx.x]; }", "buf[threadIdx.x] = x[threadIdx.x]; }\n    __syncthreads();")
    assert cuda.changed_names(KERNELS, after) == ["soft_max_kernel"]
    gt = "TEST(Suite, Works) {\n  EXPECT_EQ(1, 1);\n}\n    test_cases.emplace_back(new test_norm(GGML_TYPE_F32, {64, 5, 4, 3}));\n"
    keys = list(p.test_defs(gt))
    assert keys[0] == "Suite.Works" and keys[1].startswith("test_norm#")
    assert "TEST(Suite" not in (p.strip_tests(gt, {"Suite.Works"}) or "")


def test_op_map_reaches_through_calls_launches_and_macros(tmp_path: Path):
    t = write(tmp_path, {"ggml/src/ggml-cuda/ggml-cuda.cu": DISPATCH, "ggml/src/ggml-cuda/softmax.cu": KERNELS, "ggml/src/ggml-cuda/common.cuh": COMMON,
                         "ggml/src/ggml-cuda/norm.cu": NORM})
    m = ko.op_map(t)
    assert set(m) == {"SOFT_MAX", "NORM", "GELU"}
    assert {"soft_max_kernel", "block_reduce"} <= m["SOFT_MAX"]["functions"]
    assert "norm_kernel" in m["NORM"]["functions"]  # reached only through the CALL_NORM macro
    assert ko.ops_for(m, ["ggml/src/ggml-cuda/common.cuh"], ["block_reduce"]) == ["NORM", "SOFT_MAX"]
    assert ko.ops_for(m, ["ggml/src/ggml-cuda/norm.cu"], []) == ["NORM"]  # no changed function found: by file
    assert ko.layout_of(["ggml/src/ggml-cuda.cu", "ggml/src/ggml-cuda/x.cu"]) == ko.GGML_2024
    assert ko.layout_of(["ggml/src/ggml-cuda/ggml-cuda.cu"]) == ko.GGML_ORG


def test_change_filters_and_fault_classes():
    assert kdm.kind_of("CUDA: fix shared-memory WAR hazard in flash_attn_ext_vec (#30573)", "") == "bug"
    assert kdm.kind_of("CUDA/HIP: fix race in tile kernel", "") == "bug"  # CUDA named with HIP: kept
    assert kdm.kind_of("HIP: fix template skip for mfma kernels", "") is None
    assert kdm.kind_of("CUDA: fix 2 broken Volta FA cases", "") is None
    assert kdm.kind_of("cuda : fix compiler warnings in mmq", "") is None
    assert kdm.kind_of("cuda : fix -Wtautological-compare in fattn vec dispatch", "") is None
    assert kdm.kind_of("CUDA: faster MMVQ for low-bit types", "") == "perf"
    assert kdm.kind_of("Revert CUDA concat change", "") is None
    assert kdm.classify("CUDA: fix race in X", "", "", "bug")[0] == "race"
    assert kdm.classify("fix divergent barrier in f16 flash attention", "", "", "bug") == ("divergent_barrier", ["golden_repeat", "synccheck"])
    assert kdm.classify("fix MMQ out-of-bounds reads", "", "", "bug")[1] == ["golden", "memcheck"]
    assert kdm.classify("fix get_rows for more than 65535 rows", "", "", "bug")[0] == "grid_limit"
    assert kdm.classify("fix something", "", "+    __syncthreads();\n", "bug")[0] == "race"  # the diff decides when the message does not
    assert kdm.classify("fix norm grid limits", "", "+    __syncthreads();\n", "bug")[0] == "grid_limit"  # the message first
    assert kdm.classify("faster", "", "", "perf") == ("perf", ["golden", "perf_bar"])


def ch(sha: str, date: str, ops: list[str], files: list[str], kind: str = "bug") -> kdm.KernelChange:
    return kdm.KernelChange(repo="r", sha=sha, parent=sha + "p", date=date, author_date=date, subject="fix", body="", files=files, kind=kind,
                            fault_class="race", detectors=["golden"], ops=ops)


def test_groups_seed_and_absorb_post_release_first():
    cs = [ch("a", "2026-06-01T00:00:00+00:00", ["FA"], ["x.cu"]), ch("b", "2026-06-10T00:00:00+00:00", ["NORM"], ["n.cu"]),
          ch("c", "2026-07-01T00:00:00+00:00", ["FA"], ["y.cu"]), ch("d", "2026-08-01T00:00:00+00:00", ["NORM"], ["n.cu"]),
          ch("e", "2026-03-01T00:00:00+00:00", ["FA"], ["x.cu"]), ch("f", "2026-03-05T00:00:00+00:00", ["FA"], ["z.cu"]),
          ch("p", "2026-06-02T00:00:00+00:00", ["FA"], ["x.cu"], kind="perf")]
    gs = kdm.group_changes(cs, kind="bug")
    assert [[c.sha for c in g] for g in gs] == [["a", "c"], ["b", "d"], ["e", "f"]]  # interleaved faults find their partners; pre-cut apart
    assert kdm.interactions(gs[0])[0]["ops"] == ["FA"]
    far = [ch("a", "2026-01-01T00:00:00+00:00", ["FA"], ["x.cu"]), ch("b", "2026-09-01T00:00:00+00:00", ["FA"], ["x.cu"])]
    assert kdm.group_changes(far, kind="bug", since="2025-01-01") == []  # more than 120 days apart


def git(d: Path, *a: str, date: str = "2026-06-01T00:00:00+00:00") -> str:
    env = {"GIT_AUTHOR_DATE": date, "GIT_COMMITTER_DATE": date, "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin", "HOME": str(d)}
    return subprocess.run(["git", "-C", str(d), "-c", "user.name=t", "-c", "user.email=t@t", *a], capture_output=True, text=True, check=True, env=env).stdout


def test_mine_a_synthetic_backend(tmp_path: Path):
    d = tmp_path / "r"
    write(d, {"ggml/src/ggml-cuda/ggml-cuda.cu": DISPATCH, "ggml/src/ggml-cuda/softmax.cu": KERNELS, "ggml/src/ggml-cuda/common.cuh": COMMON,
              "ggml/src/ggml-cuda/norm.cu": NORM, "LICENSE": "MIT License\n\nPermission is hereby granted, free of charge, to any person\n",
              "tests/test-backend-ops.cpp": "int main() {}\n", "ggml/src/ggml-cpu/ops.cpp": "int x;\n"})
    git(d, "init", "-q")
    git(d, "add", "-A")
    git(d, "commit", "-qm", "init", date="2026-05-02T00:00:00+00:00")
    (d / "ggml/src/ggml-cuda/softmax.cu").write_text(KERNELS.replace("y[threadIdx.x] = block_reduce(buf);", "__syncthreads();\n    y[threadIdx.x] = block_reduce(buf);"))
    git(d, "commit", "-qam", "CUDA: fix race in soft max", date="2026-06-01T00:00:00+00:00")
    (d / "ggml/src/ggml-cuda/norm.cu").write_text(NORM.replace("x[0] = block_reduce(x);", "if (x) { x[0] = block_reduce(x); }"))
    (d / "ggml/src/ggml-cpu/ops.cpp").write_text("int y;\n")
    git(d, "commit", "-qam", "fix norm on CPU and CUDA", date="2026-06-02T00:00:00+00:00")
    (d / "ggml/src/ggml-cuda/norm.cu").write_text(NORM.replace("x[0] = block_reduce(x);", "x[0] = 2 * block_reduce(x);"))
    git(d, "commit", "-qam", "HIP: fix norm on gfx1100", date="2026-06-03T00:00:00+00:00")
    got, drops = kdm.mine(d, "r", since="2026-05-01", cache=tmp_path / "cache")
    assert [(c.subject, c.fault_class, c.ops) for c in got] == [("CUDA: fix race in soft max", "race", ["SOFT_MAX"])]
    assert got[0].functions == ["soft_max_kernel"] and got[0].kernels == ["soft_max_kernel"] and got[0].licence == "MIT"
    reasons = {d_["subject"]: d_["reason"] for d_ in drops}
    assert reasons["fix norm on CPU and CUDA"].startswith("touches code outside the CUDA backend")
    assert reasons["HIP: fix norm on gfx1100"].startswith("platform-specific")
