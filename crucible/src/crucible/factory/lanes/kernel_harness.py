"""Kernel debugging lane, the grader-owned harness (swarm-pipeline/10 §4). Standard library only: the kit plan
(``kernel_debug_plan``) imports it inside the verifier image, and the CPU gates import it here.

**The harness is the repository's own op test, owned by the grader.** ggml's ``tests/test-backend-ops.cpp`` builds a
graph per test case, runs it on a backend and compares every node with the CPU backend's reference implementation, with
a tolerance per case (normalized mean squared error, ``max_err``). The grader always uses the file **as it is at B**
(the fixed release, with every test case the fixes added), never the agent's copy, and applies ``patch_harness``:

- ``CRUCIBLE_SEED``: every random draw of the harness (tensor initialisation, index and mask generators) comes from one
  counter-based stream seeded at grading time, so the reference build and the candidate see the same inputs, and a fresh
  seed gives inputs neither has seen. ``std::random_device`` is replaced by ``crucible_random_device`` (a deterministic
  generator), and the per-thread engines of ``init_tensor_uniform`` are seeded from (seed, call index, slice start), so
  the parallel initialisation is deterministic too;
- ``CRUCIBLE_CASES``: a file of exact case names (``OP(vars)``, one per line); only those cases run (a selection file
  with no line selects nothing);
- ``CRUCIBLE_LIST``: every case of the mode's list is printed (``CRUCIBLE_CASE <name>``) and none runs: the case pool
  of an op, for test mode and perf mode alike, without computing anything;
- ``CRUCIBLE_DUMP``: after each compared node, the backend's output and the CPU reference are appended to a dump file
  (case name, node, op, the case's tolerance, both float vectors). The grader compares dumps **outside** the candidate's
  process: the candidate's GPU output against the CPU reference and the GPU output of B, both computed by the
  reference build (root-only) on the same seed;
- ``CRUCIBLE_REPS``: the graph is computed that many extra times on the tested backend before the comparison, so the
  grader's own wall clock around whole processes (R vs 3R repetitions) measures the kernels (perf gates);
- ``CRUCIBLE_ALLOW_CPU``: the CPU backend may be the tested backend (the CPU gates run the harness this way: the case
  selection, the seeding and the dump format are executed without a GPU).

Two dialects of the file are known (``DIALECTS``): ``ggml-2025`` (ggml-org/llama.cpp since mid 2025) and ``ggml-2024``
(the 2024 layout that forks such as ik_llama.cpp keep). Every rewrite is anchored on a source line; ``patch_harness``
raises ``HarnessPatchError`` on an unknown dialect or a missing anchor, so a B whose harness changed shape is refused at
S5 instead of being graded with a silently unpatched file.
"""

from __future__ import annotations

import math
import re
import struct
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

HARNESS = "tests/test-backend-ops.cpp"
MARK = "// crucible harness hooks"
TARGET = "test-backend-ops"
# the grader's CMake configuration (one source of truth for the CPU gates, the reference build and the candidate build)
CMAKE_COMMON = ("-DGGML_NATIVE=OFF", "-DLLAMA_CURL=OFF", "-DLLAMA_OPENSSL=OFF", "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_BUILD_TESTS=ON",
                "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_TOOLS=OFF")
CMAKE_CUDA = ("-DGGML_CUDA=ON", *CMAKE_COMMON, "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON", "-DGGML_CUDA_FA_ALL_QUANTS=OFF")
CMAKE_CPU = ("-DGGML_CUDA=OFF", *CMAKE_COMMON)
# layouts whose CPU backend does not build without x86 SIMD (ik_llama.cpp's 2024 layout: iqk needs AVX2); the grader's
# reference and candidate builds and the CPU gates all add these, keyed by the layout's dispatch file
HOST_CPU_FLAGS = {"ggml/src/ggml-cuda.cu": ("-DGGML_AVX=ON", "-DGGML_AVX2=ON", "-DGGML_FMA=ON", "-DGGML_F16C=ON")}


def cmake_ccache() -> tuple[str, ...]:
    """Compiler-launcher flags for a build cache (gate runs of the grader's own patches only: a cache shared with
    candidate builds would let one candidate's objects reach another's)."""
    return tuple(f"-DCMAKE_{lang}_COMPILER_LAUNCHER=ccache" for lang in ("C", "CXX", "CUDA"))


TESTS_CMAKE = "tests/CMakeLists.txt"
TARGET_SHIM = (f"\n# crucible: the grader's op-test target (this repository keeps the file but no longer declares it)\n"
               f"add_executable({TARGET} test-backend-ops.cpp)\ntarget_link_libraries({TARGET} PRIVATE ggml)\n")


def ensure_target(tests_cmake: str) -> str:
    """``tests/CMakeLists.txt`` with the op-test target declared (the grader owns build files: forks such as ik_llama.cpp
    dropped the declaration but keep and maintain the source)."""
    return tests_cmake if TARGET in tests_cmake else tests_cmake.rstrip("\n") + "\n" + TARGET_SHIM


def ggml_header(tree: Path) -> str | None:
    """The tree's ``ggml.h`` (what the API-drift repairs read), or None."""
    p = Path(tree) / "ggml" / "include" / "ggml.h"
    return p.read_text(errors="replace") if p.is_file() else None


def prepare_tree(tree: Path, harness_text: str) -> None:
    """The grader's tree: the patched harness in place and its target declared; in trees that generate
    ``common/build-info.cpp`` from git into the (read-only) source directory (llama.cpp before mid 2025), the file is
    written once from its template, newer than the template, so the build never runs git."""
    harness_file(tree).write_text(harness_text)
    cm = Path(tree) / TESTS_CMAKE
    if cm.is_file():
        cm.write_text(ensure_target(cm.read_text(errors="replace")))
    tpl, out = Path(tree) / "common" / "build-info.cpp.in", Path(tree) / "common" / "build-info.cpp"
    ccm = Path(tree) / "common" / "CMakeLists.txt"
    if tpl.is_file() and ccm.is_file():
        text = ccm.read_text(errors="replace")
        gen = re.compile(r'add_custom_command\(\s*OUTPUT\s+"\$\{CMAKE_CURRENT_SOURCE_DIR\}/build-info\.cpp".*?\n\)', re.DOTALL)
        if gen.search(text):
            ccm.write_text(gen.sub("# crucible: build-info.cpp is written by the grader (no git, read-only sources)", text, count=1))
            out.write_text(re.sub(r"@\w+@", "unknown", tpl.read_text(errors="replace").replace("@BUILD_NUMBER@", "0")))


def cuda_arch_flag(arches: Sequence[str]) -> str:
    """``CMAKE_CUDA_ARCHITECTURES`` for the grader: real (SASS) targets only, so no PTX can be JIT-compiled for another GPU."""
    return ";".join(f"{a}-real" for a in arches)


def cmake_cuda(arches: Sequence[str]) -> list[str]:
    return [*CMAKE_CUDA, f"-DCMAKE_CUDA_ARCHITECTURES={cuda_arch_flag(arches)}"]


class HarnessPatchError(RuntimeError):
    pass


PRELUDE = r"""
// crucible harness hooks (grader-owned; swarm-pipeline/10 §4): seeded inputs, case selection, output dumps, repetitions
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <vector>
static uint64_t crucible_splitmix(uint64_t x) {
    x += 0x9E3779B97F4A7C15ull;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
    return x ^ (x >> 31);
}
static uint64_t crucible_seed_base() {
    static const uint64_t s = [] {
        const char * e = getenv("CRUCIBLE_SEED");
        return e && *e ? (uint64_t) strtoull(e, nullptr, 10) : (uint64_t) 0x5EEDull;
    }();
    return s;
}
static std::atomic<uint64_t> crucible_counter{0};
static uint64_t crucible_next() { return crucible_splitmix(crucible_seed_base() ^ crucible_splitmix(crucible_counter.fetch_add(1))); }
struct crucible_random_device {
    using result_type = unsigned int;
    static constexpr result_type min() { return 0u; }
    static constexpr result_type max() { return 0xFFFFFFFFu; }
    result_type operator()() { return (result_type) (crucible_next() & 0xFFFFFFFFull); }
};
static std::mutex crucible_dump_mu;
static FILE * crucible_dump_file() {
    static FILE * f = [] {
        const char * p = getenv("CRUCIBLE_DUMP");
        return p && *p ? fopen(p, "wb") : (FILE *) nullptr;
    }();
    return f;
}
static void crucible_dump(const std::string & name, int node, const char * op, double max_err, const float * a, const float * b, size_t n) {
    FILE * f = crucible_dump_file();
    if (!f) {
        return;
    }
    std::lock_guard<std::mutex> lk(crucible_dump_mu);
    const uint32_t ln = (uint32_t) name.size();
    const uint32_t lo = (uint32_t) strlen(op);
    const uint64_t nn = (uint64_t) n;
    const int32_t nd = (int32_t) node;
    fwrite("CRD1", 1, 4, f);
    fwrite(&ln, 4, 1, f); fwrite(name.data(), 1, ln, f);
    fwrite(&lo, 4, 1, f); fwrite(op, 1, lo, f);
    fwrite(&nd, 4, 1, f); fwrite(&max_err, 8, 1, f); fwrite(&nn, 8, 1, f);
    fwrite(a, 4, n, f); fwrite(b, 4, n, f);
    fflush(f);
}
static int crucible_reps() {
    const char * r = getenv("CRUCIBLE_REPS");
    return r && *r ? atoi(r) : 0;
}
static std::string crucible_case_name;
static bool crucible_case_selected(const std::string & name) {
    static const bool list = getenv("CRUCIBLE_LIST") != nullptr;
    if (list) {  // listing: name every case, run none
        printf("CRUCIBLE_CASE %s\n", name.c_str());
        return false;
    }
    static const std::vector<std::string> sel = [] {
        std::vector<std::string> v;
        const char * p = getenv("CRUCIBLE_CASES");
        FILE * f = p && *p ? fopen(p, "r") : (FILE *) nullptr;
        if (!f) {
            return v;
        }
        std::string cur;
        int c;
        while ((c = fgetc(f)) != EOF) {
            if (c == '\n') {
                if (!cur.empty()) { v.push_back(cur); }
                cur.clear();
            } else {
                cur.push_back((char) c);
            }
        }
        if (!cur.empty()) { v.push_back(cur); }
        fclose(f);
        v.push_back("");  // a selection file was read: an empty selection selects nothing
        return v;
    }();
    if (sel.empty()) {
        return true;
    }
    for (const auto & s : sel) {
        if (!s.empty() && s == name) {
            return true;
        }
    }
    return false;
}
"""

INCLUDE_END = re.compile(r"(?:^#include[^\n]*\n)+", re.MULTILINE)
# The harness changed shape over 2024-2026; each hook has the forms seen in ggml-org/llama.cpp (2025-09 .. 2026-10) and
# in the 2024 layout forks keep. A form not listed is a missing anchor (the task is refused at S5).
ANCHORS: dict[str, tuple[re.Pattern[str], ...]] = {
    "init_fn": (re.compile(r"static void init_tensor_uniform\s*\([^)]*\)\s*\{"),),
    "engine_thread_local": (re.compile(r"thread_local\s+std::default_random_engine\s+gen\s*\(\s*std::random_device\s*\{\s*\}\s*\(\s*\)\s*\)\s*;"),),
    "engine_static": (re.compile(r"static\s+std::vector<std::default_random_engine>\s+generators\s*=\s*\[\]\s*\(\s*\)\s*\{\s*std::random_device\s+rd\s*;"),),
    "err": (re.compile(r"^(?P<ind>[ \t]*)double err = (?:ud->tc->err|nmse)\(f1\.data\(\), f2\.data\(\), f1\.size\(\)\);[ \t]*\n[ \t]*if \(err > "
                       r"(?P<max>ud->tc->max_err\(ud->backend1\)|ud->tc->max_err\(\)|ud->max_err)\)", re.MULTILINE),),
    "compare": (re.compile(r"^(?P<ind>[ \t]*)const bool cmp_ok = ggml_backend_compare_graph_backend\(", re.MULTILINE),),
    "filter": (re.compile(r"if \(!matches_filter\(out, op_names_filter\)"), re.compile(r"if \(op_name != nullptr && op_desc\(out\) != op_name\)")),
    "cpu_skip": (re.compile(r"if \(backend_filter == NULL && ggml_backend_dev_type\(dev\) == GGML_BACKEND_DEVICE_TYPE_CPU"),
                 re.compile(r"if \(backend_filter == NULL && ggml_backend_is_cpu\(backend\)\)")),
    "perf": (re.compile(r"\}\s*while\s*\(\s*total_time_us\s*<\s*1000\s*\*\s*1000\s*\)\s*;"),
             re.compile(r"int64_t start_time = ggml_time_us\(\);\n(?P<ind>[ \t]*)ggml_backend_graph_compute\(backend, gf\);")),
}
PERF_NEW = ("} while (crucible_reps() > 0 ? total_runs / std::max(1, n_runs) < crucible_reps() : total_time_us < 1000*1000);",
            ("int64_t start_time = ggml_time_us();\n\\g<ind>for (int crucible_i = 0; crucible_i < std::max(1, crucible_reps()); ++crucible_i) "
             "ggml_backend_graph_compute(backend, gf);"))
SELECT = 'crucible_case_selected(op_desc(out) + "(" + vars() + ")")'
FILTER_NEW = (f"if (!{SELECT} || !matches_filter(out, op_names_filter)", f"if (!{SELECT} || (op_name != nullptr && op_desc(out) != op_name))")
STATIC_ENGINES = ANCHORS["engine_static"][0]


def _find(text: str, key: str) -> tuple[int, re.Match[str]] | None:
    for i, rx in enumerate(ANCHORS[key]):
        m = rx.search(text)
        if m:
            return i, m
    return None


# Grader repairs of a repository's own harness where it no longer compiles against its own library (ik_llama.cpp's
# flash-attention case generator was synced from upstream while its test class kept the 2024 constructor). Each repair
# is anchored, recorded on the task (``repairs_applied``) and only ever narrows what is generated to what the file's own
# class can build; it never changes how a case is judged.
REPAIRS: tuple[tuple[str, re.Pattern[str], str], ...] = (
    ("ik-fattn-2024-ctor",
     re.compile(r"test_cases\.emplace_back\(new test_flash_attn_ext\(\s*hsk, hsv, nh, \{nr2, nr3\}, kv, nb, mask, sinks, max_bias, logit_softcap, prec, type_KV\)\);"
                r"(?P<rest>\s*// run fewer test cases permuted\s*if \([^\n]*\)\s*\{\s*test_cases\.emplace_back\(new test_flash_attn_ext\(\s*hsk, hsv, nh, \{nr2, nr3\}, kv, nb, mask, "
                r"sinks, max_bias, logit_softcap, prec, type_KV, \{0, 2, 1, 3\}\)\);\s*\})?"),
     ("if (!sinks && hsk == hsv && nr2 == 1 && nr3 == 1 && prec == GGML_PREC_F32) { "
      "test_cases.emplace_back(new test_flash_attn_ext(hsk, nh, kv, nb, mask, max_bias, logit_softcap, type_KV)); }  // crucible repair: the 2024 constructor")),
)
OLD_FA_CTOR = re.compile(r"test_flash_attn_ext\(int64_t hs = 128, int64_t nh = 32, int64_t kv = 96, int64_t nb = 8, bool mask = true, float max_bias = 0\.0f, "
                         r"float softcap = 0\.0f, ggml_type type_KV = GGML_TYPE_F16\)")


def _decl(header: str, fn: str) -> str:
    m = re.search(rf"\b{fn}\s*\((.*?)\)\s*\)?\s*;", header, re.DOTALL)
    return m.group(1) if m else ""


def repair_harness(text: str, header: str | None = None) -> tuple[str, list[str]]:
    """``text`` with the anchored ``REPAIRS`` that apply (only where the file's own class has the old constructor), and,
    given the tree's ``ggml.h``, the API-drift repairs: a call missing a trailing parameter the library added since the
    harness last compiled gets that parameter's neutral value (``ggml_quantize_chunk``'s user data: ``nullptr``;
    ``ggml_upscale``/``ggml_upscale_ext``'s scale mode: nearest, the old behaviour), and a case line constructing a test
    class the file no longer defines is dropped. Returns the text and the repairs applied."""
    applied = []
    for name, rx, new in REPAIRS:
        if name == "ik-fattn-2024-ctor" and not OLD_FA_CTOR.search(text):
            continue
        text, n = rx.subn(new, text)
        if n:
            applied.append(f"{name} x{n}")
    if header:
        if "quantize_user_data" in _decl(header, "ggml_quantize_chunk"):
            text, n = re.subn(r"(ggml_quantize_chunk\((?:[^;()]|\([^;()]*\))*?,\s*im)\)", r"\1, nullptr)", text)
            if n:
                applied.append(f"quantize_chunk_user_data x{n}")
        if "ggml_scale_mode" in _decl(header, "ggml_upscale"):
            text, n = re.subn(r"ggml_upscale\(ctx, a, scale_factor\)", "ggml_upscale(ctx, a, scale_factor, GGML_SCALE_MODE_NEAREST)", text)
            if n:
                applied.append(f"upscale_mode x{n}")
        if "ggml_scale_mode" in _decl(header, "ggml_upscale_ext"):
            text, n = re.subn(r"ggml_upscale_ext\(ctx, a, (ne_tgt\[0\], ne_tgt\[1\],\s*ne_tgt\[2\], ne_tgt\[3\])\)", r"ggml_upscale_ext(ctx, a, \1, GGML_SCALE_MODE_NEAREST)", text)
            if n:
                applied.append(f"upscale_ext_mode x{n}")
    defined = set(re.findall(r"\b(?:struct|class)\s+(test_\w+)\b(?:\s*final)?\s*(?::[^;{]*)?\{", text))
    lines, dropped = [], 0
    for ln in text.splitlines(keepends=True):
        m = re.search(r"test_cases\.emplace_back\(new\s+(test_\w+)\s*\(", ln)
        if m and defined and m.group(1) not in defined:
            dropped += 1
            continue
        lines.append(ln)
    if dropped:
        text = "".join(lines)
        applied.append(f"undefined_case_class x{dropped}")
    return text, applied


def dialect_of(text: str) -> str | None:
    """``ggml-2025`` (per-case filters: ggml-org/llama.cpp since mid 2025) or ``ggml-2024`` (one op filter), when the
    compare callback is recognised; else None."""
    if _find(text, "err") is None:
        return None
    f = _find(text, "filter")
    return "ggml-2024" if f is not None and f[0] == 1 else "ggml-2025"


def patch_harness(text: str, *, need_perf: bool = False, header: str | None = None) -> str:
    """The grader's version of ``test-backend-ops.cpp`` (module doc). Idempotent; raises ``HarnessPatchError`` on an
    unknown dialect or a missing anchor (the perf-loop anchor only with ``need_perf``: a perf task needs it)."""
    if MARK in text:
        return text
    text, _repairs = repair_harness(text, header)
    name = dialect_of(text)
    if name is None:
        raise HarnessPatchError("harness dialect unknown (no compare-callback anchor)")
    keys = ("init_fn", "compare", "filter", "cpu_skip", *(("perf",) if need_perf else ()))
    missing = [k for k in keys if _find(text, k) is None]
    # the initialisation engines: per-call thread_local engines (rewritten), or static per-thread engines seeded once from
    # the random device (deterministic once the device is replaced); anything else is unknown
    engine = _find(text, "engine_thread_local") is not None
    if not engine and _find(text, "engine_static") is None:
        missing.append("engine")
    inc = list(INCLUDE_END.finditer(text))
    if not inc:
        missing.append("includes")
    if missing:
        raise HarnessPatchError(f"harness anchors missing ({name}): {', '.join(missing)}")
    end = inc[0].end()
    out = text[:end] + PRELUDE + text[end:]
    if engine:
        # one call index per init_tensor_uniform call (main thread, deterministic order); each slice's engine from (seed, call, start)
        out = ANCHORS["init_fn"][0].sub(lambda m: m.group(0) + "\n    const uint64_t crucible_call = crucible_next();", out, count=1)
        out = ANCHORS["engine_thread_local"][0].sub(
            "std::default_random_engine gen((unsigned) crucible_splitmix(crucible_call ^ ((uint64_t) start * 0x100000001B3ull)));", out, count=1)
    out = re.sub(r"\bstd::random_device\b", "crucible_random_device", out)
    err = ANCHORS["err"][0]
    out = err.sub(lambda m: m.group(0).replace(
        "\n", f"\n{m.group('ind')}crucible_dump(crucible_case_name, index, ggml_op_desc(t1), {m.group('max')}, f1.data(), f2.data(), f1.size());\n", 1),
        out, count=1)
    case_name = 'current_op_name + "(" + vars() + ")"' if re.search(r"current_op_name\s*=\s*op_desc\(out\)", text) else 'op_desc(out) + "(" + vars() + ")"'

    def compare(m: re.Match[str]) -> str:
        i = m.group("ind")
        return (f"{i}crucible_case_name = {case_name};\n"
                f"{i}for (int crucible_i = 0, crucible_n = crucible_reps(); crucible_i < crucible_n; ++crucible_i) {{\n"
                f"{i}    ggml_backend_graph_compute(backend1, gf);\n{i}}}\n"
                f"{i}if (crucible_reps() > 0) {{\n{i}    ggml_backend_synchronize(backend1);\n{i}}}\n{m.group(0)}")

    out = ANCHORS["compare"][0].sub(compare, out, count=1)
    fi, _ = _find(text, "filter")
    out = ANCHORS["filter"][fi].sub(FILTER_NEW[fi], out)
    pf = _find(text, "perf")
    if pf is not None:
        out = ANCHORS["perf"][pf[0]].sub(PERF_NEW[pf[0]], out, count=1)
    ci, _ = _find(text, "cpu_skip")
    out = ANCHORS["cpu_skip"][ci].sub(lambda m: m.group(0).replace("if (", "if (!getenv(\"CRUCIBLE_ALLOW_CPU\") && ", 1), out, count=1)
    return out


EXTRA_MARK = "// crucible: grader cases"
EXTRA_LINE = re.compile(r"^test_cases\.emplace_back\(new test_\w+\([^;]*\)\);$")  # one statement: braced shapes, no blocks
RETURN_CASES = re.compile(r"^(?P<ind>[ \t]*)return test_cases;", re.MULTILINE)
# where each layout keeps its case lists: ggml-2025 builds one list per mode, the 2024 layout one list inside
# ``test_backend`` that both modes run
LIST_FN = {"test": re.compile(r"\bmake_test_cases_eval\s*\([^)]*\)\s*\{"), "perf": re.compile(r"\bmake_test_cases_perf\s*\([^)]*\)\s*\{")}
LIST_2024 = (re.compile(r"\bstatic bool test_backend\s*\([^)]*\)\s*\{"), re.compile(r"^(?P<ind>[ \t]*)if \(mode == MODE_TEST\) \{", re.MULTILINE))


def add_cases(text: str, extra: Sequence[Mapping[str, str] | str]) -> str:
    """The harness with grader-owned test cases appended to its case lists: a fault the repository's own cases never
    reach on the grading GPU (a launch grid past 65535 rows, a batch past 1024 matrices, a quantization type the list
    omits) gets a case of the file's own test classes at the shape the bug report describes. Each entry is a line (or
    ``{"code", "mode"}``, mode ``test`` (default), ``perf`` or ``both``); only ``test_cases.emplace_back(new test_X(...));``
    lines are accepted (a case, never other code). ggml-2025: before ``return test_cases;`` of ``make_test_cases_eval``
    (test) or ``make_test_cases_perf`` (perf); the 2024 layout: before ``if (mode == MODE_TEST)`` in ``test_backend``."""
    rows = [(e, "test") if isinstance(e, str) else (str(e["code"]), str(e.get("mode") or "test")) for e in extra]
    rows = [(c.strip(), m) for c, m in rows if c.strip()]
    if not rows or EXTRA_MARK in text:
        return text
    bad = [c for c, _m in rows if not EXTRA_LINE.match(c)] + [f"mode {m!r}" for _c, m in rows if m not in ("test", "perf", "both")]
    if bad:
        raise HarnessPatchError(f"not a test-case line: {bad[0][:120]}")

    def block(ind: str, lines: list[str]) -> str:
        return f"{ind}{EXTRA_MARK} (the shapes the task's report describes)\n" + "".join(f"{ind}{ln}\n" for ln in lines)

    if LIST_FN["test"].search(text):
        spots = []
        for mode in ("test", "perf"):
            lines = [c for c, m in rows if m in (mode, "both")]
            if not lines:
                continue
            fn = LIST_FN[mode].search(text)
            ret = RETURN_CASES.search(text, fn.end()) if fn else None
            if ret is None:
                raise HarnessPatchError(f"harness anchors missing: the {mode} case list's return")
            spots.append((ret.start(), block(ret.group("ind"), lines)))
        for pos, b in sorted(spots, reverse=True):
            text = text[:pos] + b + text[pos:]
        return text
    fn = LIST_2024[0].search(text)
    run = LIST_2024[1].search(text, fn.end()) if fn else None
    if run is None:
        raise HarnessPatchError("harness anchors missing: test_backend's case list")
    return text[:run.start()] + block(run.group("ind"), [c for c, _m in rows]) + text[run.start():]


# ---- dumps -----------------------------------------------------------------------------------------------------------------------
def read_dump(data: bytes) -> list[dict]:
    """Records of a ``CRUCIBLE_DUMP`` file: ``{case, node, op, max_err, out: [floats], ref: [floats]}`` (a truncated tail
    record, from a crashed run, is dropped)."""
    recs, i, n = [], 0, len(data)
    while i + 4 <= n and data[i:i + 4] == b"CRD1":
        try:
            i += 4
            (ln,) = struct.unpack_from("<I", data, i)
            i += 4
            case = data[i:i + ln].decode("utf-8", "replace")
            i += ln
            (lo,) = struct.unpack_from("<I", data, i)
            i += 4
            op = data[i:i + lo].decode("utf-8", "replace")
            i += lo
            node, max_err, cnt = struct.unpack_from("<idQ", data, i)
            i += 4 + 8 + 8
            if i + 8 * cnt > n:
                break
            a = struct.unpack_from(f"<{cnt}f", data, i)
            i += 4 * cnt
            b = struct.unpack_from(f"<{cnt}f", data, i)
            i += 4 * cnt
        except struct.error:
            break
        recs.append({"case": case, "node": node, "op": op, "max_err": max_err, "out": a, "ref": b})
    return recs


def nmse(a: Sequence[float], b: Sequence[float]) -> float:
    """ggml's normalized mean squared error (``test_case::err``): sum (a-b)^2 / sum b^2; inf on a length or NaN mismatch."""
    if len(a) != len(b):
        return math.inf
    num = den = 0.0
    for x, y in zip(a, b, strict=True):
        if math.isnan(x) or math.isnan(y):
            return math.inf
        if math.isinf(x) or math.isinf(y):
            if x == y:
                continue
            return math.inf
        num += (x - y) ** 2
        den += y * y
    return num / den if den > 0 else (0.0 if num == 0 else math.inf)


def by_case(recs: Iterable[Mapping]) -> dict[str, list[Mapping]]:
    out: dict[str, list[Mapping]] = {}
    for r in recs:
        out.setdefault(str(r["case"]), []).append(r)
    return out


def compare_dumps(cand: Sequence[Mapping], ref: Sequence[Mapping], *, cases: Sequence[str] | None = None, tol_scale: float = 1.0) -> dict[str, dict]:
    """Per case: the candidate's tested-backend output against the reference build's CPU output (``ref['ref']``) and
    against the reference build's own tested-backend output (``ref['out']``), node by node, each within the case's
    tolerance (``max_err`` as the reference build reported it, scaled by ``tol_scale``). A case the reference ran and the
    candidate did not is a failure (``missing``); the candidate's own CPU column is never read."""
    c, r = by_case(cand), by_case(ref)
    names = list(cases) if cases is not None else sorted(r)
    out: dict[str, dict] = {}
    for name in names:
        rr, cc = r.get(name) or [], c.get(name) or []
        if not rr:
            out[name] = {"pass": False, "error": "reference did not run the case (infra)", "infra": True}
            continue
        if len(cc) != len(rr):
            out[name] = {"pass": False, "error": "missing" if not cc else f"node count {len(cc)} != {len(rr)}"}
            continue
        worst = {"vs_cpu": 0.0, "vs_b": 0.0}
        ok = True
        for x, y in zip(cc, rr, strict=True):
            tol = float(y["max_err"]) * tol_scale
            e_cpu, e_b = nmse(x["out"], y["ref"]), nmse(x["out"], y["out"])
            worst = {"vs_cpu": max(worst["vs_cpu"], e_cpu), "vs_b": max(worst["vs_b"], e_b)}
            ok &= e_cpu <= tol and e_b <= tol
        out[name] = {"pass": ok, "nmse_vs_cpu": worst["vs_cpu"], "nmse_vs_b": worst["vs_b"], "max_err": float(rr[0]["max_err"]) * tol_scale}
    return out


# ---- console output ------------------------------------------------------------------------------------------------------------
CASE_LINE = re.compile(r"^\s*(?P<case>[A-Z][A-Z0-9_]*\(.*?\)):\s*(?:\S*\[[^\]]*\]\s*)*(?:\x1b\[[0-9;]*m)?(?P<status>OK|FAIL|not supported)\b",
                       re.MULTILINE)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def crashed_case(stdout: str) -> str | None:
    """The case the harness was running when its process died: the last case line with no verdict (a backend that
    claims an op and then aborts on it, ik_llama.cpp's CPU flash attention with f32 K/V)."""
    last = None
    for ln in ANSI.sub("", stdout).splitlines():
        m = re.match(r"^\s*([A-Z][A-Z0-9_]*\(.*\)):\s*(.*)$", ln)
        if m:
            last = m.group(1) if not re.search(r"\b(OK|FAIL|SUPPORTED|not supported)\b", m.group(2), re.IGNORECASE) else None
    return last


def case_status(stdout: str) -> dict[str, str]:
    """``case -> OK | FAIL | not supported`` from the harness's console output (informational and for the CPU gates'
    case listing: the GPU verdict is the dump comparison, done outside the candidate's process)."""
    out: dict[str, str] = {}
    for ln in ANSI.sub("", stdout).splitlines():
        m = re.match(r"^\s*([A-Z][A-Z0-9_]*\(.*\)):\s*(.*)$", ln)
        if not m:
            continue
        tail = m.group(2)
        st = ("not supported" if re.search(r"not supported|\bNOT SUPPORTED\b", tail) else "supported" if re.search(r"\bSUPPORTED\b", tail)
              else "FAIL" if re.search(r"\bFAIL\b", tail) else "OK" if re.search(r"\bOK\b", tail) else None)
        if st:
            out[m.group(1)] = st
    return out


def listed_cases(stdout: str) -> list[str]:
    """Case names a ``CRUCIBLE_LIST`` run printed (every case of the mode's list, none run), in order, deduplicated."""
    return list(dict.fromkeys(ln[len("CRUCIBLE_CASE "):].strip() for ln in stdout.splitlines() if ln.startswith("CRUCIBLE_CASE ")))


def cases_file(cases: Sequence[str]) -> str:
    """The ``CRUCIBLE_CASES`` file content for ``cases``."""
    return "".join(f"{c}\n" for c in cases)


def harness_file(tree: Path) -> Path:
    return Path(tree) / HARNESS
