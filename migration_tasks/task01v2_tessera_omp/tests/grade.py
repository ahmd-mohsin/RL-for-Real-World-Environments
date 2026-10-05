#!/usr/bin/env python3
"""Grader for task01v2_tessera_omp (Tessera renderer, CUDA -> OpenMP target offload).

    grade.py --candidate DIR --reference DIR [--out report.json] [--skip-perf]

--candidate  the migrated repository (agent's work applied to the original)
--reference  the original CUDA repository (pristine), used for the CUDA perf baseline

Gates, all required for reward 1:
  static        no CUDA, no emulator, no vendor intrinsics, no host algorithms
  build_gpu     clean build for --offload-arch=sm_89 with the hidden tests/
  tests_gpu     hidden suite on the GPU (OMP_TARGET_OFFLOAD=MANDATORY)
  build_host    clean build for the host-offload device
  tests_host    hidden suite on the host-offload device (portability)
  offload       every pipeline stage launches >= 1 kernel and stays within its download budget
  link          test binaries do not link the CUDA runtime
  perf          per-case time vs the original CUDA build on the same GPU (1080p frames, stages, multi-view)
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
HIDDEN = HERE / "hidden"
THRESHOLDS = json.loads((HERE / "perf_thresholds.json").read_text())

GPU_ARCH = os.environ.get("TSR_GRADE_ARCH", "sm_89")
HOST_TRIPLE = os.environ.get("TSR_GRADE_HOST_TRIPLE", "x86_64-pc-linux-gnu")
CUDA_HOME = os.environ.get("CUDA_HOME", "/usr/local/cuda")
BUILD_TYPE = "RelWithDebInfo"

PROBE_STAGES = [
    "transform_faces", "bin_mesh_faces", "rasterize_mesh_fine", "shade_surface", "project_gaussians",
    "compute_splat_colors", "bin_splats", "rasterize_splats", "tonemap", "frame",
]

# ---------------------------------------------------------------- static checks

SOURCE_EXT = {".h", ".hpp", ".hh", ".cpp", ".cc", ".cxx", ".c", ".inl", ".ipp", ".inc"}
CMAKE_NAMES = {"CMakeLists.txt"}
SKIP_DIRS = {"tests", "third_party", ".git", "docs"}

CODE_RULES = [
    ("cuda_language", r"__global__|__device__|__shared__|__host__|__constant__|<<<|__syncthreads|__syncwarp|__shfl|__ballot|__popc\b|__ffs\b|__ldg|\bcuda[A-Z]\w*|\bcub::|\bthrust::|cuda_runtime|<cuda[./]|\bdim3\b|\bthreadIdx\b|\bblockIdx\b|\bblockDim\b|\bgridDim\b|\bwarpSize\b"),
    ("vendor_intrinsics", r"\bompx_\w+|\bompx::|__nvvm|__builtin_nvvm|__builtin_amdgcn|\basm\s*(volatile\s*)?\(|__asm__|\bptx\b"),
    ("emulator", r"\bemu::|tsr/device/emu|tsr/emu|emulator|fiber_context|TSR_BACKEND_EMU|TSR_BACKEND_CUDA|TSR_ENABLE_CUDA|TSR_EMU_"),
]
# Host algorithms are forbidden in the library: every primitive must compute on the device.
LIB_RULES = [
    ("host_algorithm", r"std::(ranges::)?(sort|stable_sort|partial_sort|nth_element|partial_sum|inclusive_scan|exclusive_scan|transform_inclusive_scan|transform_exclusive_scan|accumulate|reduce|transform_reduce|partition|stable_partition|merge|copy_if|count|count_if|max_element|min_element|minmax_element)\b|\bqsort\s*\("),
]
CMAKE_RULES = [
    ("cmake_cuda", r"LANGUAGES[^)]*\bCUDA\b|enable_language\s*\(\s*CUDA|CUDAToolkit|CUDA::|\bcudart\b|\bnvcc\b|CMAKE_CUDA|FindCUDA|\.cu\b"),
]


def strip_c_comments(text):
    """Remove // and /* */ comments, keeping string literals."""
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c == '"' or c == "'":
            j = i + 1
            while j < n and text[j] != c:
                j += 2 if text[j] == "\\" else 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            out.append("\n" * text.count("\n", i, n if j < 0 else j))
            i = n if j < 0 else j + 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def strip_cmake_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def static_checks(repo):
    violations = []
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if not path.is_file() or (rel.parts and rel.parts[0] in SKIP_DIRS):
            continue
        if path.suffix in {".cu", ".cuh", ".ptx", ".cubin", ".fatbin"}:
            violations.append({"rule": "cuda_file", "file": str(rel)})
            continue
        if path.suffix in {".a", ".so", ".o"}:
            violations.append({"rule": "prebuilt_binary", "file": str(rel)})
            continue
        if path.suffix in SOURCE_EXT:
            text = strip_c_comments(path.read_text(errors="replace"))
            rules = CODE_RULES + (LIB_RULES if rel.parts[0] in {"src", "include"} else [])
        elif path.name in CMAKE_NAMES or path.suffix == ".cmake":
            text = strip_cmake_comments(path.read_text(errors="replace"))
            rules = CMAKE_RULES
        else:
            continue
        for rule, pattern in rules:
            for m in re.finditer(pattern, text):
                line = text.count("\n", 0, m.start()) + 1
                violations.append({"rule": rule, "file": str(rel), "line": line, "match": m.group(0)})
    return violations


# ---------------------------------------------------------------- helpers

def run(cmd, cwd=None, env=None, timeout=1800):
    full_env = dict(os.environ, OMP_TARGET_OFFLOAD="MANDATORY")
    full_env.update(env or {})
    try:
        p = subprocess.run(cmd, cwd=cwd, env=full_env, timeout=timeout, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        return 124, e.stdout or "", f"timeout after {timeout}s"


def tail(text, lines=40):
    return "\n".join((text or "").splitlines()[-lines:])


def build(src, build_dir, extra):
    cfg = ["cmake", "-S", str(src), "-B", str(build_dir), "-G", "Ninja", "-DCMAKE_CXX_COMPILER=clang++",
           f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}", "-DTSR_BUILD_TESTS=ON", "-DTSR_BUILD_APPS=OFF"] + extra
    rc, out, err = run(cfg)
    if rc != 0:
        return False, tail(out + err)
    rc, out, err = run(["ninja", "-C", str(build_dir), "tsr_tests", "tsr_offload_probe", "tsr_bench"], timeout=3600)
    return rc == 0, tail(out + err)


def find_binary(build_dir, name):
    hits = [p for p in Path(build_dir).rglob(name) if p.is_file() and os.access(p, os.X_OK)]
    return hits[0] if hits else None


def run_doctest(binary):
    rc, out, err = run([str(binary), "--no-intro", "--duration=false"], timeout=1800)
    summary = [l for l in (out + err).splitlines() if l.startswith("[doctest] test cases") or l.startswith("[doctest] assertions")]
    return rc == 0, {"returncode": rc, "summary": summary, "log": tail(out + err, 60)}


def parse_json_lines(text):
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return rows


# ---------------------------------------------------------------- reference

def build_reference_bench(reference, work):
    """Original CUDA library + the hidden benchmark, compiled with nvcc."""
    nvcc = str(Path(CUDA_HOME) / "bin" / "nvcc")
    bdir = work / "ref-build"
    rc, out, err = run(["cmake", "-S", str(reference), "-B", str(bdir), "-G", "Ninja", f"-DCMAKE_BUILD_TYPE={BUILD_TYPE}",
                        "-DTSR_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DCMAKE_CUDA_ARCHITECTURES=89",
                        "-DTSR_BUILD_TESTS=OFF", "-DTSR_BUILD_APPS=OFF"])
    if rc != 0:
        raise RuntimeError("reference configure failed:\n" + tail(out + err))
    rc, out, err = run(["ninja", "-C", str(bdir)], timeout=3600)
    if rc != 0:
        raise RuntimeError("reference build failed:\n" + tail(out + err))
    lib = next(bdir.rglob("libtessera.a"))
    src = work / "bench_ref.cu"
    shutil.copy(HIDDEN / "bench_tessera.cpp", src)
    exe = work / "ref_bench"
    rc, out, err = run([nvcc, "-O2", "-std=c++20", "-arch=sm_89", "--expt-relaxed-constexpr", "--extended-lambda", str(src),
                        f"-I{reference / 'include'}", f"-I{HIDDEN}", str(lib), "-o", str(exe)], timeout=1800)
    if rc != 0:
        raise RuntimeError("reference bench build failed:\n" + tail(out + err))
    return exe


# ---------------------------------------------------------------- gates

def gate_offload(probe):
    rc, out, err = run([str(probe)], timeout=900)
    rows = {r["stage"]: r for r in parse_json_lines(out)}
    problems = []
    if rc != 0:
        problems.append(f"probe exited with {rc}: {tail(err, 10)}")
    for stage in PROBE_STAGES:
        r = rows.get(stage)
        if r is None:
            problems.append(f"{stage}: no probe output")
        elif r["kernels"] < 1:
            problems.append(f"{stage}: no target region launched")
        elif r["d2h"] > r["budget"]:
            problems.append(f"{stage}: {r['d2h']} bytes copied device->host (budget {r['budget']})")
    return not problems, {"rows": list(rows.values()), "problems": problems}


def gate_link(binaries):
    problems = []
    for b in binaries:
        rc, out, _ = run(["ldd", str(b)], timeout=60)
        if re.search(r"libcudart|libcublas|libcuda\.so", out):
            problems.append(f"{b.name} links the CUDA runtime")
    return not problems, {"problems": problems}


def gate_perf(port_bench, ref_bench, rounds):
    """Interleave reference and candidate runs; keep the best median of each case."""
    best = {"ref": {}, "port": {}}
    logs = []
    for _ in range(rounds):
        for tag, exe in (("ref", ref_bench), ("port", port_bench)):
            rc, out, err = run([str(exe)], timeout=1800)
            if rc != 0:
                return False, {"error": f"{tag} bench exited with {rc}", "log": tail(out + err)}
            for r in parse_json_lines(out):
                prev = best[tag].get(r["case"])
                best[tag][r["case"]] = r["median_ms"] if prev is None else min(prev, r["median_ms"])
            logs.append(out)
    cases, ok = [], True
    for case, limit in THRESHOLDS["max_ratio"].items():
        ref, port = best["ref"].get(case), best["port"].get(case)
        if ref is None or port is None:
            cases.append({"case": case, "pass": False, "error": "missing measurement"})
            ok = False
            continue
        ratio = port / ref
        passed = ratio <= limit
        ok &= passed
        cases.append({"case": case, "cuda_ms": ref, "port_ms": port, "ratio": round(ratio, 3), "max_ratio": limit, "pass": passed})
    return ok, {"cases": cases}


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--reference", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--perf-rounds", type=int, default=3)
    ap.add_argument("--keep", action="store_true", help="keep the work directory")
    args = ap.parse_args()

    work = Path(tempfile.mkdtemp(prefix="tsr-grade-"))
    report = {"task": "task01_prims_omp", "gates": {}}

    def record(name, ok, detail):
        report["gates"][name] = {"pass": bool(ok), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        return ok

    try:
        cand = work / "candidate"
        shutil.copytree(args.candidate, cand, ignore=shutil.ignore_patterns(".git", "build*"))
        # The hidden suite replaces whatever tests the candidate shipped.
        shutil.rmtree(cand / "tests", ignore_errors=True)
        shutil.copytree(HIDDEN, cand / "tests", ignore=shutil.ignore_patterns("tools"))
        # golden frames of the original CUDA build
        rc, out, err = run(["tar", "xzf", str(HIDDEN / "data" / "golden.tgz"), "-C", str(cand / "tests" / "data"), "--strip-components=1"])
        if rc != 0:
            raise RuntimeError("cannot unpack golden frames: " + err)

        violations = static_checks(cand)
        all_ok = record("static", not violations, {"violations": violations[:50], "count": len(violations)})

        ok, log = build(cand, work / "build-gpu", [f"-DTSR_OFFLOAD_ARCH={GPU_ARCH}"])
        all_ok &= record("build_gpu", ok, {"log": log})
        gpu_ok = ok
        if gpu_ok:
            ok, detail = run_doctest(find_binary(work / "build-gpu", "tsr_tests"))
            all_ok &= record("tests_gpu", ok, detail)
            ok, detail = gate_offload(find_binary(work / "build-gpu", "tsr_offload_probe"))
            all_ok &= record("offload", ok, detail)
            ok, detail = gate_link([find_binary(work / "build-gpu", n) for n in ("tsr_tests", "tsr_bench")])
            all_ok &= record("link", ok, detail)

        ok, log = build(cand, work / "build-host", ["-DTSR_OFFLOAD_ARCH=", f"-DTSR_OFFLOAD_TARGETS={HOST_TRIPLE}"])
        all_ok &= record("build_host", ok, {"log": log})
        if ok:
            ok, detail = run_doctest(find_binary(work / "build-host", "tsr_tests"))
            all_ok &= record("tests_host", ok, detail)

        if args.skip_perf:
            report["gates"]["perf"] = {"pass": None, "skipped": True}
        elif gpu_ok and all_ok:
            ref = build_reference_bench(args.reference, work)
            ok, detail = gate_perf(find_binary(work / "build-gpu", "tsr_bench"), ref, args.perf_rounds)
            all_ok &= record("perf", ok, detail)
        else:
            all_ok &= record("perf", False, {"skipped": "earlier gate failed"})

        report["reward"] = 1.0 if all_ok and not args.skip_perf else 0.0
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    text = json.dumps(report, indent=2)
    if args.out:
        args.out.write_text(text)
    print(json.dumps({"reward": report["reward"],
                      "gates": {k: v["pass"] for k, v in report["gates"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
