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
STATIC_ENGINES = re.compile(r"static\s+std::vector<std::default_random_engine>\s+generators\s*=\s*\[\]\s*\(\s*\)\s*\{\s*std::random_device\s+rd\s*;")
DIALECTS: dict[str, dict[str, object]] = {
    "ggml-2025": {
        "detect": re.compile(r"ud->tc->max_err\(ud->backend1\)"),
        "engine": re.compile(r"thread_local\s+std::default_random_engine\s+gen\s*\(\s*std::random_device\s*\{\s*\}\s*\(\s*\)\s*\)\s*;"),
        "init_fn": re.compile(r"static void init_tensor_uniform\s*\([^)]*\)\s*\{"),
        "err": re.compile(r"^(?P<ind>[ \t]*)double err = ud->tc->err\(f1\.data\(\), f2\.data\(\), f1\.size\(\)\);[ \t]*$", re.MULTILINE),
        "max_err": "ud->tc->max_err(ud->backend1)",
        "compare": re.compile(r"^(?P<ind>[ \t]*)const bool cmp_ok = ggml_backend_compare_graph_backend\(", re.MULTILINE),
        "case_name": 'current_op_name + "(" + vars() + ")"',
        "filter": re.compile(r"if \(!matches_filter\(out, op_names_filter\)"),
        "filter_new": 'if (!crucible_case_selected(op_desc(out) + "(" + vars() + ")") || !matches_filter(out, op_names_filter)',
        "cpu_skip": re.compile(r"if \(backend_filter == NULL && ggml_backend_dev_type\(dev\) == GGML_BACKEND_DEVICE_TYPE_CPU"),
        # perf mode: with CRUCIBLE_REPS, exactly that many timed iterations instead of "at least one second"
        "perf": re.compile(r"\}\s*while\s*\(\s*total_time_us\s*<\s*1000\s*\*\s*1000\s*\)\s*;"),
        "perf_new": "} while (crucible_reps() > 0 ? total_runs / std::max(1, n_runs) < crucible_reps() : total_time_us < 1000*1000);",
    },
    "ggml-2024": {
        "detect": re.compile(r"double err = nmse\(f1\.data\(\), f2\.data\(\), f1\.size\(\)\);"),
        "engine": None,  # static per-thread engines seeded once from the (replaced) random device: deterministic already
        "init_fn": re.compile(r"static void init_tensor_uniform\s*\([^)]*\)\s*\{"),
        "err": re.compile(r"^(?P<ind>[ \t]*)double err = nmse\(f1\.data\(\), f2\.data\(\), f1\.size\(\)\);[ \t]*$", re.MULTILINE),
        "max_err": "ud->max_err",
        "compare": re.compile(r"^(?P<ind>[ \t]*)const bool cmp_ok = ggml_backend_compare_graph_backend\(", re.MULTILINE),
        "case_name": 'op_desc(out) + "(" + vars() + ")"',
        "filter": re.compile(r"if \(op_name != nullptr && op_desc\(out\) != op_name\)"),
        "filter_new": 'if (!crucible_case_selected(op_desc(out) + "(" + vars() + ")") || (op_name != nullptr && op_desc(out) != op_name))',
        "cpu_skip": re.compile(r"if \(backend_filter == NULL && ggml_backend_is_cpu\(backend\)\)"),
        "perf": re.compile(r"int64_t start_time = ggml_time_us\(\);\n(?P<ind>[ \t]*)ggml_backend_graph_compute\(backend, gf\);"),
        "perf_new": "int64_t start_time = ggml_time_us();\n\\g<ind>for (int crucible_i = 0; crucible_i < std::max(1, crucible_reps()); ++crucible_i) "
                    "ggml_backend_graph_compute(backend, gf);",
    },
}


def _rx(d: Mapping[str, object], key: str) -> re.Pattern[str]:
    v = d[key]
    assert isinstance(v, re.Pattern)
    return v


def dialect_of(text: str) -> str | None:
    """The harness dialect of ``text`` (``DIALECTS``), or None."""
    return next((name for name, d in DIALECTS.items() if _rx(d, "detect").search(text)), None)


def patch_harness(text: str, *, need_perf: bool = False) -> str:
    """The grader's version of ``test-backend-ops.cpp`` (module doc). Idempotent; raises ``HarnessPatchError`` on an
    unknown dialect or a missing anchor (the perf-loop anchor only with ``need_perf``: a perf task needs it)."""
    if MARK in text:
        return text
    name = dialect_of(text)
    if name is None:
        raise HarnessPatchError("harness dialect unknown (no compare-callback anchor)")
    d = DIALECTS[name]
    keys = ("init_fn", "err", "compare", "filter", "cpu_skip", *(("perf",) if need_perf else ()))
    missing = [k for k in keys if d[k] is not None and not _rx(d, k).search(text)]
    # the initialisation engines: per-call thread_local engines (rewritten), or static per-thread engines seeded once from
    # the random device (deterministic once the device is replaced); anything else is unknown
    engine = d["engine"] is not None and _rx(d, "engine").search(text) is not None
    if not engine and not STATIC_ENGINES.search(text):
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
        out = _rx(d, "init_fn").sub(lambda m: m.group(0) + "\n    const uint64_t crucible_call = crucible_next();", out, count=1)
        out = _rx(d, "engine").sub("std::default_random_engine gen((unsigned) crucible_splitmix(crucible_call ^ ((uint64_t) start * 0x100000001B3ull)));",
                                   out, count=1)
    out = re.sub(r"\bstd::random_device\b", "crucible_random_device", out)
    out = _rx(d, "err").sub(lambda m: f"{m.group(0)}\n{m.group('ind')}crucible_dump(crucible_case_name, index, ggml_op_desc(t1), {d['max_err']}, "
                                      "f1.data(), f2.data(), f1.size());", out, count=1)

    def compare(m: re.Match[str]) -> str:
        i = m.group("ind")
        return (f"{i}crucible_case_name = {d['case_name']};\n"
                f"{i}for (int crucible_i = 0, crucible_n = crucible_reps(); crucible_i < crucible_n; ++crucible_i) {{\n"
                f"{i}    ggml_backend_graph_compute(backend1, gf);\n{i}}}\n"
                f"{i}if (crucible_reps() > 0) {{\n{i}    ggml_backend_synchronize(backend1);\n{i}}}\n{m.group(0)}")

    out = _rx(d, "compare").sub(compare, out, count=1)
    out = _rx(d, "filter").sub(str(d["filter_new"]), out)
    out = _rx(d, "perf").sub(str(d["perf_new"]), out, count=1)
    out = _rx(d, "cpu_skip").sub(lambda m: m.group(0).replace("if (", "if (!getenv(\"CRUCIBLE_ALLOW_CPU\") && ", 1), out, count=1)
    return out


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
