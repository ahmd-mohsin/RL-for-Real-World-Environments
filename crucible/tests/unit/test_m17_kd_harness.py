"""Kernel debugging lane, the grader-owned harness (swarm-pipeline/10 §4): the test-backend-ops patch in both dialects
(seeded inputs, case selection, listing, dumps, repetitions, the CPU switch), anchors refused loudly, the dump format, the
out-of-process comparison and the console parser. No compiler, no GPU."""

from __future__ import annotations

import math
import struct

import pytest

from crucible.factory.lanes import kernel_harness as kh

NEW = """#include <ggml.h>
#include <random>

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
        auto init_thread = [&](size_t start, size_t end) {
            thread_local std::default_random_engine gen(std::random_device{}());
        };
}
static void init_mask() { std::random_device rd; std::mt19937 gen(rd()); }
struct test_case {
    test_status_t eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_names_filter, printer * output_printer) {
        if (!matches_filter(out, op_names_filter)) {
            return test_status_t::SKIPPED;
        }
        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            double err = ud->tc->err(f1.data(), f2.data(), f1.size());
            if (err > ud->tc->max_err(ud->backend1)) { ud->ok = false; }
            return true;
        };
        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud, nullptr, 0);
    }
    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {
        if (!matches_filter(out, op_names_filter)) {
            return true;
        }
        do {
            total_runs += n_runs;
        } while (total_time_us < 1000*1000); // run for at least 1 second
    }
};
int main() {
        if (backend_filter == NULL && ggml_backend_dev_type(dev) == GGML_BACKEND_DEVICE_TYPE_CPU && mode != MODE_GRAD) {
        }
}
"""

OLD = """#include <ggml.h>
#include <random>

static void init_tensor_uniform(ggml_tensor * tensor, float min = -1.0f, float max = 1.0f) {
    static std::vector<std::default_random_engine> generators = []() {
        std::random_device rd;
        return std::vector<std::default_random_engine>{};
    }();
}
struct test_case {
    bool eval(ggml_backend_t backend1, ggml_backend_t backend2, const char * op_name) {
        if (op_name != nullptr && op_desc(out) != op_name) {
            return true;
        }
        auto callback = [](int index, ggml_tensor * t1, ggml_tensor * t2, void * user_data) -> bool {
            double err = nmse(f1.data(), f2.data(), f1.size());
            if (err > ud->max_err) { ud->ok = false; }
            return true;
        };
        const bool cmp_ok = ggml_backend_compare_graph_backend(backend1, backend2, gf, callback, &ud);
    }
    bool eval_perf(ggml_backend_t backend, const char * op_name) {
        int64_t start_time = ggml_time_us();
        ggml_backend_graph_compute(backend, gf);
        ggml_backend_synchronize(backend);
    }
};
int main() {
        if (backend_filter == NULL && ggml_backend_is_cpu(backend)) {
        }
}
"""


@pytest.mark.parametrize("text,dialect", [(NEW, "ggml-2025"), (OLD, "ggml-2024")])
def test_patch_hooks_every_anchor_once(text: str, dialect: str):
    assert kh.dialect_of(text) == dialect
    out = kh.patch_harness(text, need_perf=True)
    assert kh.MARK in out and kh.patch_harness(out) == out  # idempotent
    assert "std::random_device" not in out and "crucible_random_device rd" in out  # every random draw is seeded
    assert out.count("crucible_dump(crucible_case_name, index") == 1
    assert out.count("crucible_case_name = ") == 1 and "crucible_reps()" in out
    assert "crucible_case_selected(op_desc(out)" in out  # case selection (and listing) in every mode's filter
    assert 'getenv("CRUCIBLE_ALLOW_CPU")' in out
    assert "CRUCIBLE_LIST" in out and "CRUCIBLE_CASES" in out and "CRUCIBLE_DUMP" in out and "CRUCIBLE_SEED" in out
    if dialect == "ggml-2025":
        assert "crucible_call ^ ((uint64_t) start" in out and "thread_local std::default_random_engine gen(std::random_device" not in out
        assert "total_runs / std::max(1, n_runs) < crucible_reps()" in out
    else:
        assert "for (int crucible_i = 0; crucible_i < std::max(1, crucible_reps()); ++crucible_i) ggml_backend_graph_compute(backend, gf);" in out


def test_missing_anchor_is_refused_not_silently_graded():
    with pytest.raises(kh.HarnessPatchError, match="dialect unknown"):
        kh.patch_harness("int main() {}\n")
    broken = NEW.replace("const bool cmp_ok = ggml_backend_compare_graph_backend(", "bool cmp_ok = compare(")
    with pytest.raises(kh.HarnessPatchError, match="compare"):
        kh.patch_harness(broken)
    no_perf = NEW.replace("} while (total_time_us < 1000*1000); // run for at least 1 second", "}")
    kh.patch_harness(no_perf)  # a bug task does not need the perf loop
    with pytest.raises(kh.HarnessPatchError, match="perf"):
        kh.patch_harness(no_perf, need_perf=True)


def rec(case: str, node: int, op: str, tol: float, out: list[float], ref: list[float]) -> bytes:
    n = len(out)
    return (b"CRD1" + struct.pack("<I", len(case)) + case.encode() + struct.pack("<I", len(op)) + op.encode() + struct.pack("<idQ", node, tol, n)
            + struct.pack(f"<{n}f", *out) + struct.pack(f"<{n}f", *ref))


def test_dump_round_trip_and_truncated_tail():
    data = rec("NORM(ne=[4,1,1,1])", 2, "NORM", 1e-7, [1.0, 2.0], [1.0, 2.0]) + rec("NORM(ne=[8,1,1,1])", 2, "NORM", 1e-7, [3.0], [3.5])
    recs = kh.read_dump(data + b"CRD1\x05\x00")
    assert [r["case"] for r in recs] == ["NORM(ne=[4,1,1,1])", "NORM(ne=[8,1,1,1])"]
    assert recs[1]["out"] == (3.0,) and recs[1]["ref"] == (3.5,) and recs[0]["max_err"] == pytest.approx(1e-7)


def test_nmse_matches_ggml():
    assert kh.nmse([1.0, 2.0], [1.0, 2.0]) == 0.0
    assert kh.nmse([1.0, 3.0], [1.0, 2.0]) == pytest.approx(1 / 5)
    assert kh.nmse([float("nan")], [1.0]) == math.inf and kh.nmse([1.0], [1.0, 2.0]) == math.inf
    assert kh.nmse([float("inf")], [float("inf")]) == 0.0


def test_compare_dumps_judges_against_the_reference_cpu_column_only():
    ref = kh.read_dump(rec("A(x)", 0, "A", 1e-6, [1.0, 2.0], [1.0, 2.0]) + rec("B(x)", 0, "B", 1e-6, [5.0], [5.0]))
    good = kh.read_dump(rec("A(x)", 0, "A", 9.9, [1.0, 2.0], [7.0, 7.0]) + rec("B(x)", 0, "B", 9.9, [5.0], [0.0]))  # own CPU column ignored
    bad = kh.read_dump(rec("A(x)", 0, "A", 1e-6, [1.0, 2.5], [1.0, 2.5]))
    r = kh.compare_dumps(good, ref)
    assert all(v["pass"] for v in r.values())
    r = kh.compare_dumps(bad, ref)
    assert not r["A(x)"]["pass"] and r["B(x)"] == {"pass": False, "error": "missing"}
    assert kh.compare_dumps(good, ref, cases=["C(x)"])["C(x)"]["infra"]


def test_console_status_and_listing():
    out = ("  NORM(type=f32,ne=[64,5,4,3]): \x1b[1;32mOK\x1b[0m\n  NORM(type=f32,ne=[4,1]): \x1b[1;31mFAIL\x1b[0m\n"
           "  ADD(type=f16): not supported [CUDA0]\n  MUL(x=1): \x1b[1;32mSUPPORTED\x1b[0m\nCRUCIBLE_CASE ADD(type=f16)\nCRUCIBLE_CASE ADD(type=f16)\nCRUCIBLE_CASE MUL(x=1)\n")
    st = kh.case_status(out)
    assert st == {"NORM(type=f32,ne=[64,5,4,3])": "OK", "NORM(type=f32,ne=[4,1])": "FAIL", "ADD(type=f16)": "not supported", "MUL(x=1)": "supported"}
    assert kh.listed_cases(out) == ["ADD(type=f16)", "MUL(x=1)"]
    assert kh.cases_file(["A(x)", "B(y)"]) == "A(x)\nB(y)\n"


def test_the_grader_declares_the_op_test_target_when_a_fork_dropped_it(tmp_path):
    assert kh.ensure_target("llama_build_and_test(test-backend-ops.cpp)\n") == "llama_build_and_test(test-backend-ops.cpp)\n"
    shim = kh.ensure_target("llama_build_and_test(test-jinja.cpp)\n")
    assert "add_executable(test-backend-ops test-backend-ops.cpp)" in shim and kh.ensure_target(shim) == shim
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests/CMakeLists.txt").write_text("llama_build_and_test(test-jinja.cpp)\n")
    kh.prepare_tree(tmp_path, "patched")
    assert (tmp_path / kh.HARNESS).read_text() == "patched" and "add_executable" in (tmp_path / "tests/CMakeLists.txt").read_text()


def test_api_drift_repairs_follow_the_trees_own_header():
    header = ("GGML_API size_t ggml_quantize_chunk(enum ggml_type type, const float * src, void * dst, int64_t start, int64_t nrows, int64_t n_per_row,\n"
              "    const float * imatrix, const struct quantize_user_data * ud);\n"
              "GGML_API struct ggml_tensor * ggml_upscale(struct ggml_context * ctx, struct ggml_tensor * a, int scale_factor, enum ggml_scale_mode mode);\n")
    text = ("struct test_upscale : public test_case {\n  ggml_tensor * out = ggml_upscale(ctx, a, scale_factor);\n};\n"
            "  ggml_quantize_chunk(tensor->type, data.data(), dataq.data(), 0, size/tensor->ne[0], tensor->ne[0], im);\n"
            "    test_cases.emplace_back(new test_upscale(GGML_TYPE_F32));\n    test_cases.emplace_back(new test_pad_ext(GGML_TYPE_F32, {1, 1, 1, 1}));\n")
    out, applied = kh.repair_harness(text, header)
    assert "ggml_upscale(ctx, a, scale_factor, GGML_SCALE_MODE_NEAREST)" in out and "tensor->ne[0], im, nullptr)" in out
    assert "test_pad_ext" not in out and "new test_upscale(" in out  # an undefined case class goes, a defined one stays
    assert applied == ["quantize_chunk_user_data x1", "upscale_mode x1", "undefined_case_class x1"]
    assert kh.repair_harness(text, None)[0].count("im)") == 1  # no header: no API repair


def test_grader_cmake_builds_sass_only():
    flags = kh.cmake_cuda(["90", "120"])
    assert "-DCMAKE_CUDA_ARCHITECTURES=90-real;120-real" in flags and "-DGGML_CUDA=ON" in flags and "-DGGML_NATIVE=OFF" in flags
    assert "-DGGML_CUDA=OFF" in kh.CMAKE_CPU
