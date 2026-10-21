"""task01_prims_omp / task01v2_tessera_omp: the static gate re-expressed with the M7 kit (CUDA -> OpenMP target offload).

Only the **static** gate is re-expressed so far (verified against the recorded history by ``golden_cpu``). The rest of
these graders needs work recorded in DESIGN Appendix B.10/B.12, not yet done:

- the hidden doctest suite is built by the candidate's own CMake and judged by doctest's exit code (a PUBLIC
  ``DOCTEST_CONFIG_DISABLE`` passes with 0 cases run). Fix: grader-owned compile of the hidden suite against the
  candidate's installed library, verdict from doctest's XML reporter into a grader-opened fd with an expected case
  count (``patterns.hidden_tests.doctest_suite``);
- offload-probe and perf rows are parsed from the candidate's stdout (last row wins). Fix: rows through an fd plus
  the grader's clock (``perf.rows_vs_clock``);
- the calibration ran on an RTX 4090 (sm_89) inside the agent's pod, which had internet (A1 "declared environment");
  the lab GPU is sm_120, so the clang offload toolchain needs re-validation.

The static gate here also closes an A13 hole: the seed skipped ``third_party/`` and ``docs/`` when scanning, yet the
candidate's CMake may compile them; skipped directories must now be **unchanged from the base** (``frozen_base``).
"""

from __future__ import annotations

from pathlib import Path

from crucible.verify.runtime.patterns.static import Rule, RuleSet, scan
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict

SOURCE_EXT = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx", ".c", ".inl", ".ipp", ".inc"}
CUDA_LANGUAGE = (r"__global__|__device__|__shared__|__host__|__constant__|<<<|__syncthreads|__syncwarp|__shfl|__ballot|__popc\b|__ffs\b|__ldg|\bcuda[A-Z]\w*|"
                 r"\bcub::|\bthrust::|cuda_runtime|<cuda[./]|\bdim3\b|\bthreadIdx\b|\bblockIdx\b|\bblockDim\b|\bgridDim\b|\bwarpSize\b")
VENDOR = r"\bompx_\w+|\bompx::|__nvvm|__builtin_nvvm|__builtin_amdgcn|\basm\s*(volatile\s*)?\(|__asm__|\bptx\b"
EMULATOR = {"v1": r"\bemu::|tsr/emu|emulator|thread_pool|ThreadPool|TSR_BACKEND|TSR_ENABLE_CUDA",
            "v2": r"\bemu::|tsr/device/emu|tsr/emu|emulator|fiber_context|TSR_BACKEND_EMU|TSR_BACKEND_CUDA|TSR_ENABLE_CUDA|TSR_EMU_"}
HOST_ALGO = (r"std::(ranges::)?(sort|stable_sort|partial_sort|nth_element|partial_sum|inclusive_scan|exclusive_scan|transform_inclusive_scan|"
             r"transform_exclusive_scan|accumulate|reduce|transform_reduce|partition|stable_partition|merge|copy_if|count|count_if|max_element|min_element|"
             r"minmax_element)\b|\bqsort\s*\(")
CMAKE_CUDA = r"LANGUAGES[^)]*\bCUDA\b|enable_language\s*\(\s*CUDA|CUDAToolkit|CUDA::|\bcudart\b|\bnvcc\b|CMAKE_CUDA|FindCUDA|\.cu\b"
SKIP = ("third_party", ".git", "docs")  # tests/ is replaced by the hidden suite: pruned, never scanned


KINDS = {"static": "gate"}  # M7 R6: "instruction:pure ..." is a stated rule; the seed's own tests (test.sh) are outcomes

def rules(version: str, *, frozen_base: str | None = None) -> RuleSet:
    return RuleSet(forbid=[Rule("cuda_language", CUDA_LANGUAGE, suffixes=SOURCE_EXT), Rule("vendor_intrinsics", VENDOR, suffixes=SOURCE_EXT),
                           Rule("emulator", EMULATOR[version], suffixes=SOURCE_EXT), Rule("host_algorithm", HOST_ALGO, suffixes=SOURCE_EXT, roots=("src", "include")),
                           Rule("cmake_cuda", CMAKE_CUDA, suffixes={".cmake", "CMakeLists.txt"})],
                   forbidden_files={".cu", ".cuh", ".ptx", ".cubin", ".fatbin", ".a", ".so", ".o"}, exclude=("tests",), skip=SKIP, frozen_base=frozen_base,
                   runner_tamper=False, text_suffixes=SOURCE_EXT | {".cmake", ".txt"})


def plan(ctx: Context) -> Plan:
    version = ctx.params.get("version", "v2")
    base = ctx.params.get("base")

    def static(c: Context) -> Verdict:
        r = scan(Path(c.candidate), rules(version, frozen_base=base))
        return Verdict(r["pass"], {"violations": r["violations"][:50], "count": r["count"]})

    return Plan(f"task01{'v2' if version == 'v2' else ''}:static", [Gate("static", KINDS["static"])], [Check("static", "static", static, tier="fast")])
