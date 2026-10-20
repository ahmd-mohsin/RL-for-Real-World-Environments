"""Kernel lane grader: a kit plan run as root inside the verifier image (swarm-pipeline/06 §5). Standard library + the kit
runtime only (it is staged by ``verify.stage.bundle``; nothing else of Crucible is importable there).

    python3 -I /crucible_verify/crucible/verify/runtime/run.py --plan crucible.factory.lanes.kernel_plan \
        --out /logs/verifier --base /k/base --patch /k/model.patch --param spec=/k/kspec.json --require-root

**What is graded.** Each *program* (a ``main`` translation unit plus the sources the repository builds with it) is built
by the grader with its own nvcc command for ``sm_<arch>`` (``CRUCIBLE_CUDA_ARCH``), linked with the grader's observation
hooks (``OBSERVE_CPP``: every device-to-host copy is recorded, ``srand`` is mixed with the grade's seed), and run on
grader-chosen arguments in a pristine copy of the start tree's run directory. Its *observation* (normalized stdout, the
device-to-host byte stream, files it wrote) must equal the original's: the start tree built with CUDA 11.8 as
``compute_<legacy>`` PTX and run on the same GPU by driver JIT, root-only in ``/opt/legacy``. A milestone passes when every
one of its programs builds, carries device code for the arch only, launches kernels (CUPTI injection, ``LAUNCH_CPP``),
matches on every case (including a fresh seed drawn now and revealed after the candidate ran), and passes the rung's
sanitizer and perf gates.

**Never trusted** (P26): exit codes, PASS lines, the candidate's build files (the grader owns the command), timing in its
process (perf is the grader's wall clock around whole runs, R vs 3R repetitions so start-up cancels). Candidate code runs
as a fresh unprivileged uid with a fresh HOME/TMPDIR; its outputs are sealed before comparison. PTX JIT is disabled for
candidate runs (``CUDA_DISABLE_PTX_JIT``), so an embedded older-arch PTX cannot stand in for the migration.

Spec (``kspec.json``; written by ``lanes.kernel``)::

    {"task_id", "repo", "kinds": [...], "static_files": [...], "arch_default": "120", "legacy_dir": "/opt/legacy/<task>",
     "programs": {pid: {"sources": [...], "includes": [...], "defines": [...], "libs": [...], "std": "c++17",
                        "run_dir": "<dir in A>", "cases": {name: [args...]}, "fresh": true, "timeout_s": 300,
                        "compare": {"volatile": [regex], "stdout_rtol": 0, "d2h": "bitwise"|"f32"|"f64", "rtol": 0, "atol": 0},
                        "perf": {"repeat_index": i, "r": 8} | null}},
     "milestones": [{"id", "programs": [...], "files": [...], "sanitize": bool, "perf": bool}],
     "perf_bars": {"<GPU name>": {"<pid>_ms": max_ratio}}}
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import struct
import subprocess
from pathlib import Path

from crucible.verify.runtime.patterns.compare import safe_read
from crucible.verify.runtime.patterns.differential import draw_seed
from crucible.verify.runtime.patterns.perf import LoadGuard, PerfBars
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import CandidateError, InfraError

CUDA = os.environ.get("CRUCIBLE_CUDA_HOME", "/usr/local/cuda")
NVCC = f"{CUDA}/bin/nvcc"
SANITIZER = f"{CUDA}/bin/compute-sanitizer"
WRAPS = ("cudaMemcpy", "cudaMemcpyAsync", "cudaMemcpy2D", "cudaMemcpyFromSymbol", "srand")
DEFAULT_VOLATILE = (r"(?i)\b(time|timing|elapsed|ms|msec|millisec|sec|secs|seconds|us|usec|ns|gb/s|gbps|gflops?|tflops?|throughput|bandwidth|"
                    r"speed|speedup|cycles|clock|mhz|ghz|device|driver|runtime|capability|name|pid|date|average|avg|took|perf|rate)\b")
LEGACY = {  # comment-stripped scan of the window's files (kept in step with ``kernel_mine.KINDS``; a unit test checks)
    "texref": r"\btexture\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]|\bcuda(?:Bind|Unbind)Texture\w*\s*\(|\btextureReference\b",
    "surfref": r"\bsurface\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]|\bcudaBindSurfaceToArray\s*\(|\bsurfaceReference\b",
    "warp_sync": r"\b__shfl(?:_up|_down|_xor)?\s*\(|\b__any\s*\(|\b__all\s*\(|\b__ballot\s*\(",
    "thread_api": r"\bcudaThread(?:Synchronize|Exit|SetLimit|GetLimit|SetCacheConfig|GetCacheConfig)\s*\(",
    "lib_api": r"\bcusparse[SDCZ](?:csrmv|csrmm2?|csr2csc|gemmi|csrsv_analysis|csrsv_solve|hybmv)\s*\(",
    # CUDA 13 / CCCL 3 removals (``kernel_debug_migrate.KINDS13``; a unit test keeps the two in step)
    "cccl3_functors": r"\bcub::(?:Sum|Max|Min|ArgMax|ArgMin|Equality|Inequality|CastOp)\b",
    "memloc_api": (r"\bcudaMem(?:PrefetchAsync|Advise)\s*\(\s*[^,;]+,\s*[^,;]+,\s*(?![^,;)]*(?:[Ll]oc|location))"
                   r"(?:cudaCpuDeviceId|\d+|[A-Za-z_]\w*)\s*(?:,\s*[^,;()]+)?\)"),
    "devprop_removed": r"\.\s*(?:clockRate|memoryClockRate|computeMode|deviceOverlap|kernelExecTimeoutEnabled|maxTexture1DLinear|cooperativeMultiDeviceLaunch)\b",
}
JIT_ESCAPE = r"\bcuModuleLoad\w*\s*\(|\bcuLinkCreate\w*\s*\(|\bnvrtcCompileProgram\s*\(|\bnvPTXCompilerCompile\s*\(|\bcudaLibraryLoad\w*\s*\("

# ---- grader-owned sources linked into / injected into every build -------------------------------------------------------
OBSERVE_CPP = r"""// crucible kernel lane: observation hooks (grader-owned; linked into the original and the candidate alike)
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
extern "C" {
cudaError_t __real_cudaMemcpy(void*, const void*, size_t, cudaMemcpyKind);
cudaError_t __real_cudaMemcpyAsync(void*, const void*, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t __real_cudaMemcpy2D(void*, size_t, const void*, size_t, size_t, size_t, cudaMemcpyKind);
cudaError_t __real_cudaMemcpyFromSymbol(void*, const void*, size_t, size_t, cudaMemcpyKind);
void __real_srand(unsigned);
}
static std::mutex g_mu;
static FILE* g_bin = nullptr;
static FILE* g_idx = nullptr;
static bool g_off = false;
static void crucible_open() {
  if (g_bin || g_off) return;
  const char* d = getenv("CRUCIBLE_OBS");
  if (!d || !*d) { g_off = true; return; }
  char p[4096];
  snprintf(p, sizeof p, "%s/d2h.bin", d); g_bin = fopen(p, "wb");
  snprintf(p, sizeof p, "%s/d2h.idx", d); g_idx = fopen(p, "w");
  if (!g_bin || !g_idx) g_off = true;
}
static bool host_dst(void* dst, cudaMemcpyKind kind) {
  if (kind == cudaMemcpyDeviceToHost) return true;
  if (kind != cudaMemcpyDefault) return false;
  cudaPointerAttributes a; memset(&a, 0, sizeof a);
  if (cudaPointerGetAttributes(&a, dst) != cudaSuccess) { cudaGetLastError(); return true; }
  return a.type == cudaMemoryTypeHost || a.type == cudaMemoryTypeUnregistered;
}
static void record(const void* host, size_t width, size_t pitch, size_t rows, const char* what) {
  std::lock_guard<std::mutex> lk(g_mu);
  crucible_open();
  if (g_off) return;
  for (size_t r = 0; r < rows; ++r) fwrite((const char*)host + r * pitch, 1, width, g_bin);
  fprintf(g_idx, "%s %zu\n", what, width * rows);
  fflush(g_bin); fflush(g_idx);
}
extern "C" cudaError_t __wrap_cudaMemcpy(void* dst, const void* src, size_t n, cudaMemcpyKind kind) {
  cudaError_t e = __real_cudaMemcpy(dst, src, n, kind);
  if (e == cudaSuccess && n && host_dst(dst, kind)) record(dst, n, n, 1, "memcpy");
  return e;
}
extern "C" cudaError_t __wrap_cudaMemcpyAsync(void* dst, const void* src, size_t n, cudaMemcpyKind kind, cudaStream_t s) {
  cudaError_t e = __real_cudaMemcpyAsync(dst, src, n, kind, s);
  if (e == cudaSuccess && n && host_dst(dst, kind) && getenv("CRUCIBLE_OBS")) {
    if (cudaStreamSynchronize(s) == cudaSuccess) record(dst, n, n, 1, "memcpy_async");
  }
  return e;
}
extern "C" cudaError_t __wrap_cudaMemcpy2D(void* dst, size_t dp, const void* src, size_t sp, size_t w, size_t h, cudaMemcpyKind kind) {
  cudaError_t e = __real_cudaMemcpy2D(dst, dp, src, sp, w, h, kind);
  if (e == cudaSuccess && w && h && host_dst(dst, kind)) record(dst, w, dp, h, "memcpy2d");
  return e;
}
extern "C" cudaError_t __wrap_cudaMemcpyFromSymbol(void* dst, const void* sym, size_t n, size_t off, cudaMemcpyKind kind) {
  cudaError_t e = __real_cudaMemcpyFromSymbol(dst, sym, n, off, kind);
  if (e == cudaSuccess && n && host_dst(dst, kind)) record(dst, n, n, 1, "from_symbol");
  return e;
}
extern "C" void __wrap_srand(unsigned s) {
  const char* m = getenv("CRUCIBLE_SEED");
  unsigned mix = m ? (unsigned)strtoul(m, nullptr, 10) : 0u;
  __real_srand(s ^ mix);
}
"""

LAUNCH_CPP = r"""// crucible kernel lane: kernel-launch evidence (CUPTI activity records, loaded by the driver via CUDA_INJECTION64_PATH)
#include <cupti.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <map>
#include <string>
static std::mutex g_mu;
static std::map<std::string, unsigned long long> g_counts;
static void CUPTIAPI buf_req(uint8_t** buf, size_t* size, size_t* maxrec) {
  *size = 1 << 20; *buf = (uint8_t*)aligned_alloc(8, *size); *maxrec = 0;
}
static void CUPTIAPI buf_done(CUcontext, uint32_t, uint8_t* buf, size_t, size_t valid) {
  CUpti_Activity* rec = nullptr;
  std::lock_guard<std::mutex> lk(g_mu);
  while (cuptiActivityGetNextRecord(buf, valid, &rec) == CUPTI_SUCCESS) {
    if (rec->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL || rec->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
      const char* n = ((CUpti_ActivityKernel9*)rec)->name;
      g_counts[n ? n : "?"]++;
    }
  }
  free(buf);
}
static void crucible_flush() {
  cuptiActivityFlushAll(1);
  const char* p = getenv("CRUCIBLE_LAUNCH_LOG");
  if (!p) return;
  FILE* f = fopen(p, "w");
  if (!f) return;
  std::lock_guard<std::mutex> lk(g_mu);
  for (auto& kv : g_counts) fprintf(f, "%llu\t%s\n", kv.second, kv.first.c_str());
  fclose(f);
}
extern "C" int InitializeInjection(void) {
  cuptiActivityRegisterCallbacks(buf_req, buf_done);
  cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
  atexit(crucible_flush);
  return 1;
}
"""
LAUNCH_BUILD = ["g++", "-O2", "-shared", "-fPIC", f"-I{CUDA}/include", "crucible_launch.cpp", f"-L{CUDA}/lib64", "-lcupti",
                f"-Wl,-rpath,{CUDA}/lib64", "-o", "libcrucible_launch.so"]


# ---- build commands (shared with ``lanes.kernel_world``: one source of truth) ----------------------------------------------
def nvcc_flags(prog: dict, *, root: Path, arch: str, legacy: bool = False, legacy_arch: str = "60", nvcc: str | None = None) -> list[str]:
    """The grader's compiler and flags. ``legacy``: the original (CUDA 11.8) as ``compute_<legacy_arch>`` PTX only (driver
    JIT on the GPU); otherwise native ``sm_<arch>`` SASS only, no PTX (no JIT escape)."""
    gen = f"arch=compute_{legacy_arch},code=compute_{legacy_arch}" if legacy else f"arch=compute_{arch},code=sm_{arch}"
    cmd = [nvcc or NVCC, f"-std={prog.get('std') or 'c++17'}", "-O3", "-gencode", gen, "-cudart", "static", "-Wno-deprecated-gpu-targets",
           "-diag-suppress", "1444"]
    for inc in prog.get("includes") or []:
        cmd += ["-I", str(root / inc)]
    cmd += [f"-D{d}" for d in prog.get("defines") or []]
    if prog.get("openmp"):
        cmd += ["-Xcompiler", "-fopenmp"]
    return cmd


def nvcc_command(prog: dict, *, root: Path, out: Path, arch: str, observe: Path, legacy: bool = False, legacy_arch: str = "60",
                 nvcc: str | None = None) -> list[str]:
    """The grader's build of one program: its sources, the observation hooks, the link wraps."""
    cmd = nvcc_flags(prog, root=root, arch=arch, legacy=legacy, legacy_arch=legacy_arch, nvcc=nvcc)
    cmd += [str(root / s) for s in prog["sources"]]
    cmd += [str(observe), "-Xlinker", "--wrap=" + ",--wrap=".join(WRAPS), "-o", str(out)]
    # the driver library is on every GPU host; linking it always lets a migration move to driver-API calls
    # (``cuArrayGetDescriptor``) without editing a build file the grader does not read
    cmd += ["-L", f"{CUDA}/lib64/stubs", *[f"-l{lib}" for lib in prog.get("libs") or [] if lib != "cuda"], "-lcuda"]
    if prog.get("openmp"):
        cmd += ["-lgomp"]
    return cmd


def tu_command(prog: dict, tu: str, *, root: Path, out: Path, arch: str, legacy: bool = False, legacy_arch: str = "60", nvcc: str | None = None) -> list[str]:
    """Compile one translation unit (``-c``) with the program's flags: the compile milestone's check (no GPU needed)."""
    return [*nvcc_flags(prog, root=root, arch=arch, legacy=legacy, legacy_arch=legacy_arch, nvcc=nvcc), "-c", str(root / tu), "-o", str(out)]


def strip_c(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r'"(?:\\.|[^"\\\n])*"', '""', text)
    return re.sub(r"//[^\n]*", "", text)


def static_hits(root: Path, files: list[str], kinds: list[str]) -> list[dict]:
    pats = [(k, re.compile(LEGACY[k])) for k in kinds if k in LEGACY]
    jit = re.compile(JIT_ESCAPE)
    hits = []
    for rel in files:
        p = root / rel
        if not p.is_file() or p.is_symlink():
            continue  # a deleted file holds no legacy construct; its programs fail to build if they needed it
        code = strip_c(safe_read(p).decode("utf-8", "replace"))
        for k, rx in pats:
            hits += [{"file": rel, "kind": k, "match": m.group(0)[:100]} for m in rx.finditer(code)]
        hits += [{"file": rel, "kind": "jit_escape", "match": m.group(0)[:100]} for m in jit.finditer(code)]
    return hits


# ---- observations ------------------------------------------------------------------------------------------------------------
def normalize_stdout(text: str, volatile: list[str]) -> list[str]:
    rxs = [re.compile(v) for v in volatile]
    out = []
    for ln in text.splitlines():
        s = " ".join(ln.split())
        if not s or any(r.search(s) for r in rxs):
            continue
        out.append(s)
    return out


NUM = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?|[-+]?(?:nan|inf)", re.IGNORECASE)


def compare_lines(cand: list[str], ref: list[str], rtol: float, atol: float) -> list[dict]:
    if len(cand) != len(ref):
        return [{"stdout": "line count", "cand": len(cand), "ref": len(ref)}]
    fails = []
    for i, (c, r) in enumerate(zip(cand, ref, strict=True)):
        if c == r:
            continue
        if NUM.sub("#", c) != NUM.sub("#", r):
            fails.append({"line": i, "cand": c[:160], "ref": r[:160]})
            continue
        for a, b in zip(NUM.findall(c), NUM.findall(r), strict=True):
            fa, fb = float(a), float(b)
            same = (math.isnan(fa) and math.isnan(fb)) or fa == fb or (math.isfinite(fa) and math.isfinite(fb) and abs(fa - fb) <= atol + rtol * abs(fb))
            if not same:
                fails.append({"line": i, "cand": c[:160], "ref": r[:160]})
                break
        if len(fails) > 12:
            break
    return fails


def compare_stream(cand: bytes, ref: bytes, mode: str, rtol: float, atol: float) -> list[dict]:
    if len(cand) != len(ref):
        return [{"d2h": "length", "cand": len(cand), "ref": len(ref)}]
    if cand == ref:
        return []
    if mode == "bitwise":
        first = next(i for i, (a, b) in enumerate(zip(cand, ref, strict=True)) if a != b)
        return [{"d2h": "bytes differ", "first_offset": first}]
    w, fmt = (4, "f") if mode == "f32" else (8, "d")
    if len(ref) % w:
        return [{"d2h": f"length {len(ref)} not a multiple of {w}"}]
    n = len(ref) // w
    ca, ra = struct.unpack(f"<{n}{fmt}", cand), struct.unpack(f"<{n}{fmt}", ref)
    bad = 0
    worst = (0.0, -1)
    for i, (a, b) in enumerate(zip(ca, ra, strict=True)):
        if a == b or (math.isnan(a) and math.isnan(b)):
            continue
        if not (math.isfinite(a) and math.isfinite(b)) or abs(a - b) > atol + rtol * abs(b):
            bad += 1
            d = abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf
            if d > worst[0]:
                worst = (d, i)
    return [{"d2h": "values differ", "count": bad, "worst_abs": worst[0], "at": worst[1]}] if bad else []


def observation(run_dir: Path, obs_dir: Path, stdout: str, before: set[str]) -> dict:
    """What a run produced: stdout, the d2h stream, files it created in its run directory (sealed already)."""
    out = {"stdout": stdout, "d2h": b"", "files": {}}
    b = obs_dir / "d2h.bin"
    if b.exists():
        out["d2h"] = safe_read(b)
    for p in sorted(run_dir.rglob("*")):
        rel = p.relative_to(run_dir).as_posix()
        if p.is_file() and not p.is_symlink() and rel not in before:
            out["files"][rel] = safe_read(p)
    return out


def compare_obs(cand: dict, ref: dict, cmp: dict) -> list[dict]:
    vol = list(cmp.get("volatile") or [DEFAULT_VOLATILE])
    rtol, atol = float(cmp.get("rtol") or 0.0), float(cmp.get("atol") or 0.0)
    fails = compare_lines(normalize_stdout(cand["stdout"], vol), normalize_stdout(ref["stdout"], vol), float(cmp.get("stdout_rtol") or rtol), atol)
    fails += compare_stream(cand["d2h"], ref["d2h"], cmp.get("d2h") or "bitwise", rtol, atol)
    for name in sorted(set(ref["files"]) | set(cand["files"])):
        if name not in cand["files"] or name not in ref["files"]:
            fails.append({"file": name, "error": "missing" if name not in cand["files"] else "unexpected"})
        elif cand["files"][name] != ref["files"][name]:
            fails += [{"file": name, **f} for f in compare_stream(cand["files"][name], ref["files"][name], cmp.get("d2h") or "bitwise", rtol, atol)]
    return fails


def empty_obs(o: dict) -> bool:
    return not o["d2h"] and not o["files"] and not normalize_stdout(o["stdout"], [DEFAULT_VOLATILE])


SAN_SUMMARY = re.compile(r"(ERROR SUMMARY|RACECHECK SUMMARY|SYNCCHECK SUMMARY|LEAK SUMMARY)[^\n]*")


def sanitizer_clean(rc: int, log: str) -> tuple[bool, list[str]]:
    """Every summary line reports zero and the sanitizer's exit code is 0 (the task04 grader read only the first line)."""
    lines = [m.group(0) for m in SAN_SUMMARY.finditer(log)]
    if rc != 0 or not lines:
        return False, lines[-4:]
    zero = all(re.search(r":\s*0\s+(errors?|hazards?)", ln) or re.search(r"\b0 hazards", ln) or "0 bytes" in ln for ln in lines)
    return zero, lines[-4:]


# ---- the plan --------------------------------------------------------------------------------------------------------------
class KernelGrader:
    def __init__(self, ctx: Context, spec: dict):
        self.ctx, self.spec, self.t = ctx, spec, ctx.trusted
        self.arch = str(ctx.params.get("arch") or os.environ.get("CRUCIBLE_CUDA_ARCH") or spec.get("arch_default") or "90")
        self.base = Path(ctx.params.get("base_tree") or "/k/base")
        self.legacy = Path(ctx.params.get("legacy_dir") or spec.get("legacy_dir") or "/opt/legacy")
        self.hooks = Path(ctx.params.get("hooks_dir") or "/k/hooks")
        self.work = ctx.work
        self.bins: dict[str, Path | None] = {}
        self.build_log: dict[str, dict] = {}
        self.refs: dict[tuple, dict] = {}
        self.seed: int | None = None
        self.gpu = gpu_name()

    # -- runs
    def _rundir(self, pid: str, tag: str) -> tuple[Path, set[str]]:
        prog = self.spec["programs"][pid]
        d = self.work / "runs" / pid / tag
        if d.exists():
            shutil.rmtree(d)
        src = self.base / prog.get("run_dir", ".")
        shutil.copytree(src, d, symlinks=False, ignore=shutil.ignore_patterns("*.cu", "*.cuh", "*.o", ".git"))
        before = {p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file()}
        return d, before

    def run_original(self, pid: str, case: str, args: list, seed: int, *, obs: bool = True) -> dict:
        key = (pid, case, seed, obs)
        if key in self.refs:
            return self.refs[key]
        exe = self.legacy / pid
        if not exe.exists():
            raise InfraError(f"the original build of {pid} is missing from {self.legacy}")
        d, before = self._rundir(pid, f"ref-{case}-{seed}")
        o = self.t.private(self.work / "obs" / pid / f"ref-{case}-{seed}")
        env = {"CRUCIBLE_SEED": str(seed), "OMP_NUM_THREADS": "1", "CUDA_CACHE_PATH": str(self.work / "ref_jit_cache"), "CUDA_CACHE_MAXSIZE": "4294967296"}
        if obs:
            env["CRUCIBLE_OBS"] = str(o)
        r = self.t.run([exe, *args], cwd=d, env=env, timeout=float(self.spec["programs"][pid].get("timeout_s", 300)))
        if r.timed_out or r.rc != 0:
            raise InfraError(f"the original {pid} {case} failed (rc {r.rc}): {r.log[-400:]}")
        res = observation(d, o, r.stdout, before)
        res["wall_s"] = r.wall_s
        if obs and empty_obs(res):
            raise InfraError(f"the original {pid} {case} produced no observation")
        self.refs[key] = res
        return res

    def run_candidate(self, pid: str, case: str, args: list, seed: int, *, obs: bool = True, wrap: list | None = None,
                      launches: bool = False, timeout: float | None = None) -> tuple[dict, object]:
        exe = self.bins.get(pid)
        d, before = self._rundir(pid, f"cand-{case}-{seed}-{len(wrap or [])}")
        self.t.give(d)
        o = self.t.udir(self.work / "obs" / pid / f"cand-{case}-{seed}-{len(wrap or [])}")
        env = {"CRUCIBLE_SEED": str(seed), "OMP_NUM_THREADS": "1", "CUDA_DISABLE_PTX_JIT": "1"}
        if obs:
            env["CRUCIBLE_OBS"] = str(o)
        if launches:
            env["CUDA_INJECTION64_PATH"] = str(self.hooks / "libcrucible_launch.so")
            env["CRUCIBLE_LAUNCH_LOG"] = str(o / "launches.tsv")
        r = self.t.run([*(wrap or []), exe, *args], cwd=d, env=env, timeout=timeout or float(self.spec["programs"][pid].get("timeout_s", 300)) * 3,
                       candidate=True)
        self.t.seal(d)
        self.t.seal(o)
        res = observation(d, o, r.stdout, before)
        res["wall_s"] = r.wall_s
        res["launches"] = read_launches(o / "launches.tsv") if launches else None
        return res, r

    # -- checks
    def build(self, pid: str) -> Verdict:
        prog = self.spec["programs"][pid]
        src = self.ctx.candidate
        out_dir = self.t.udir(self.work / "bin" / pid)
        exe = out_dir / "prog"
        cmd = nvcc_command(prog, root=src, out=exe, arch=self.arch, observe=self.hooks / "crucible_observe.cpp")
        r = self.t.run(cmd, cwd=out_dir, timeout=1800, candidate=True)
        self.t.seal(out_dir)
        errs = [ln for ln in r.log.splitlines() if re.search(r"\berror\b", ln)][:20]
        ok = r.rc == 0 and not r.timed_out and exe.is_file()
        self.bins[pid] = exe if ok else None
        self.build_log[pid] = {"rc": r.rc, "errors": errs}
        return Verdict(ok, {"program": pid, "arch": self.arch, "rc": r.rc, "errors": errs, "log": r.log[-1500:] if not ok else ""})

    def compile_tu(self, pid: str, tu: str) -> Verdict:
        prog = self.spec["programs"][pid]
        out_dir = self.t.udir(self.work / "obj" / re.sub(r"[^\w]+", "_", tu))
        r = self.t.run(tu_command(prog, tu, root=self.ctx.candidate, out=out_dir / "unit.o", arch=self.arch), cwd=out_dir, timeout=900, candidate=True)
        self.t.seal(out_dir)
        errs = [ln for ln in r.log.splitlines() if re.search(r"\berror\b", ln)][:12]
        ok = r.rc == 0 and not r.timed_out and (out_dir / "unit.o").is_file()
        return Verdict(ok, {"unit": tu, "arch": self.arch, "rc": r.rc, "errors": errs})

    def binary(self, pid: str) -> Verdict:
        exe = self.bins[pid]
        elf = subprocess.run([f"{CUDA}/bin/cuobjdump", "--list-elf", str(exe)], capture_output=True, text=True, check=False).stdout
        ptx = subprocess.run([f"{CUDA}/bin/cuobjdump", "--list-ptx", str(exe)], capture_output=True, text=True, check=False).stdout
        sms = sorted(set(re.findall(r"\.(sm_\w+)\.cubin", elf)))
        ptxs = sorted(set(re.findall(r"\.((?:sm|compute)_\w+)\.ptx", ptx)))
        bad = [a for a in sms if not re.fullmatch(rf"sm_{self.arch}[af]?", a)] + [f"ptx:{a}" for a in ptxs if not re.fullmatch(rf"(sm|compute)_{self.arch}[af]?", a)]
        if not sms:
            bad.append("no device code")
        return Verdict(not bad, {"sass": sms, "ptx": ptxs, "issues": bad})

    def golden(self, pid: str) -> Verdict:
        prog = self.spec["programs"][pid]
        cmp = prog.get("compare") or {}
        if self.seed is None:
            self.seed = draw_seed(self.ctx) & 0x7FFFFFFF
        detail, ok = {}, True
        cases = list((prog.get("cases") or {"default": []}).items())
        draws = [(name, args, 0) for name, args in cases]
        if prog.get("fresh", True):
            draws += [(f"fresh-{name}", args, self.seed) for name, args in cases[:1]]
        for name, args, seed in draws:
            ref = self.run_original(pid, name, args, seed)
            cand, r = self.run_candidate(pid, name, args, seed, launches=(name == draws[0][0]))
            if r.timed_out:
                detail[name] = {"error": "timed out"}
                ok = False
                continue
            fails = compare_obs(cand, ref, cmp)
            if cand.get("launches") is not None:
                n = sum(cand["launches"].values())
                if n == 0:
                    fails.append({"launches": 0, "error": "no kernel ran on the GPU"})
            detail[name] = {"pass": not fails, "failures": fails[:8]}
            ok &= not fails
        return Verdict(ok, {"cases": detail, "seed_revealed_after_run": True})

    def sanitize(self, pid: str) -> Verdict:
        prog = self.spec["programs"][pid]
        _name, args = next(iter((prog.get("sanitize_cases") or prog.get("cases") or {"default": []}).items()))
        detail, ok = {}, True
        for tool in ("racecheck", "synccheck", "memcheck"):
            wrap = [SANITIZER, "--tool", tool, "--error-exitcode", "97", "--print-limit", "30"]
            if tool == "memcheck":
                wrap += ["--leak-check", "no"]
            _cand, r = self.run_candidate(pid, f"san-{tool}", args, 0, obs=False, wrap=wrap, timeout=float(prog.get("timeout_s", 300)) * 20)
            clean, summary = sanitizer_clean(r.rc, r.log)
            detail[tool] = {"pass": clean, "rc": r.rc, "summary": summary}
            ok &= clean
        return Verdict(ok, detail)

    def perf_metric(self, pid: str):
        prog = self.spec["programs"][pid]
        p = prog["perf"]
        _name, base_args = next(iter(prog["cases"].items()))
        i, r1 = int(p["repeat_index"]), int(p.get("r", 8))

        def args_for(reps: int) -> list:
            a = list(base_args)
            a[i] = str(reps)
            return a

        def round_fn(rnd: int) -> dict:
            orig, cand = {}, {}
            for reps in (r1, 3 * r1):
                o = self.run_original(pid, f"perf{rnd}-{reps}", args_for(reps), 0, obs=False)
                c, rr = self.run_candidate(pid, f"perf{rnd}-{reps}", args_for(reps), 0, obs=False, launches=True)
                if rr.rc != 0 or rr.timed_out:
                    raise CandidateError(f"timed run of {pid} failed (rc {rr.rc})")
                orig[reps], cand[reps] = o["wall_s"], (c["wall_s"], sum((c.get("launches") or {}).values()))
            per = lambda a, b: max(1e-6, (b - a) / (2 * r1)) * 1e3
            if cand[3 * r1][1] - cand[r1][1] < 2 * r1:
                raise CandidateError(f"{pid}: kernel launches do not scale with repetitions (memoized or skipped work)")
            return {"original": {f"{pid}_ms": per(orig[r1], orig[3 * r1])}, "candidate": {f"{pid}_ms": per(cand[r1][0], cand[3 * r1][0])}}

        return round_fn

    def perf(self, pid: str) -> Verdict:
        bars = (self.spec.get("perf_bars") or {}).get(self.gpu) or {}
        bar = bars.get(f"{pid}_ms")
        if bar is None:
            return Verdict(False, {"gpu": self.gpu}, infra=f"no calibrated perf bar for {pid} on {self.gpu!r}")
        pb = PerfBars(bars={f"{pid}_ms": float(bar)}, round_fn=self.perf_metric(pid), rounds=3, remeasure=1, spread_max=0.25,
                      guard=LoadGuard(max_load_per_core=0.5, max_gpu_util=20, wait_s=1200))
        v = pb()
        v.detail["gpu"] = self.gpu
        return v


def read_launches(p: Path) -> dict[str, int]:
    if not p.exists():
        return {}
    out = {}
    for ln in safe_read(p).decode("utf-8", "replace").splitlines():
        n, _, name = ln.partition("\t")
        if n.strip().isdigit():
            out[name] = out.get(name, 0) + int(n)
    return out


def gpu_name() -> str:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=False).stdout
        return out.strip().splitlines()[0].strip() if out.strip() else "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def plan(ctx: Context) -> Plan:
    spec = json.loads(Path(ctx.params["spec"]).read_text())
    g = KernelGrader(ctx, spec)
    cpu_only = str(ctx.params.get("cpu_only", "")).lower() in ("1", "true", "yes")
    gates = [Gate(m["id"], "outcome", f"{m.get('kind', 'run')} milestone {m['id']}") for m in spec["milestones"]]
    checks = []
    skipped = Verdict(False, {"skipped": "build failed"})
    for m in spec["milestones"]:
        mid = m["id"]
        files, kinds = list(m.get("files") or []), list(spec.get("kinds") or [])
        if m.get("kind") == "compile":

            def static(ctx, files=files, kinds=kinds):
                hits = static_hits(ctx.candidate, files, kinds)
                return Verdict(not hits, {"hits": hits[:20], "count": len(hits)})

            checks.append(Check(f"{mid}:static", mid, static, tier="fast"))
            checks.append(Check(f"{mid}:compile", mid, lambda ctx, m=m: g.compile_tu(m["program"], m["tu"]), tier="fast"))
            continue
        pid = m["program"]
        if cpu_only:
            checks.append(Check(f"{mid}:gpu", mid, lambda ctx: Verdict(False, {}, infra="GPU gates not run (cpu_only)")))
            continue
        checks.append(Check(f"{mid}:build:{pid}", mid, lambda ctx, pid=pid: g.build(pid)))
        checks.append(Check(f"{mid}:binary:{pid}", mid, lambda ctx, pid=pid: g.binary(pid) if g.bins.get(pid) else skipped))
        checks.append(Check(f"{mid}:golden:{pid}", mid, lambda ctx, pid=pid: g.golden(pid) if g.bins.get(pid) else skipped))
        if m.get("sanitize"):
            checks.append(Check(f"{mid}:sanitizer:{pid}", mid, lambda ctx, pid=pid: g.sanitize(pid) if g.bins.get(pid) else skipped))
        if m.get("perf") and spec["programs"][pid].get("perf"):
            checks.append(Check(f"{mid}:perf:{pid}", mid, lambda ctx, pid=pid: g.perf(pid) if g.bins.get(pid) else skipped))
    return Plan(spec["task_id"], gates, checks)
