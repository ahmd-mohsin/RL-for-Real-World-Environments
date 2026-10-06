#!/usr/bin/env python3
"""Grader for task02_pcops_triton (pcops: CUDA extension -> Triton).

    grade.py --candidate DIR --originals DIR [--out report.json] [--skip-perf]

--candidate  the migrated pcops repository (agent's patch applied to the original)
--originals  directory holding the pristine pcops, pcnet and pcdata repositories

Gates, all required for reward 1:
  static     no C++/CUDA sources or extension build, no foreign kernel libraries,
             Triton kernels present
  install    `pip install` of the candidate works with no CUDA compiler and
             installs no compiled extension
  ops        hidden operator suite (semantics, dtypes, layouts, edge cases, grads)
  hard       opcheck, determinism, streams, threads, dynamic-shape compile,
             errors, large/skewed inputs, every operator launches Triton kernels
  app        pristine pcnet / pcdata test suites and golden logits / training
             trajectories recorded with the CUDA original
  perf       per-case time vs the original CUDA extension on the same GPU
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

FORBIDDEN_FILES = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".so", ".ptx", ".cubin", ".fatbin"}
FORBIDDEN_CODE = [
    ("native_extension", r"cpp_extension|CUDAExtension|CppExtension|load_inline|ext_modules|torch\.utils\.cpp_extension|TORCH_LIBRARY"),
    ("foreign_kernels", r"\bimport\s+(ctypes|cffi|cupy|numba|pycuda|torch_scatter|torch_cluster|torch_sparse|pointnet2_ops)\b|\bfrom\s+(ctypes|cffi|cupy|numba|pycuda|torch_scatter|torch_cluster|torch_sparse|pointnet2_ops)\b"),
    ("load_library", r"torch\.ops\.load_library|ctypes\.CDLL|cdll\.LoadLibrary"),
]


def run(cmd, cwd=None, env=None, timeout=3600):
    e = dict(os.environ)
    e.update(env or {})
    try:
        p = subprocess.run(cmd, cwd=cwd, env=e, timeout=timeout, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        return 124, ex.stdout or "", f"timeout after {timeout}s"


def tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def strip_py_comments(text):
    out = []
    for line in text.splitlines():
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


def static_checks(repo: Path):
    violations = []
    has_triton = False
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if not path.is_file() or (rel.parts and rel.parts[0] in {".git", "tests", "build", "dist"}) or "egg-info" in str(rel):
            continue
        if path.suffix in FORBIDDEN_FILES:
            violations.append({"rule": "native_source", "file": str(rel)})
            continue
        if path.suffix in {".py", ".toml", ".cfg", ".txt"} or path.name == "setup.py":
            text = path.read_text(errors="replace")
            code = strip_py_comments(text) if path.suffix == ".py" else text
            for rule, pattern in FORBIDDEN_CODE:
                for m in re.finditer(pattern, code):
                    violations.append({"rule": rule, "file": str(rel), "match": m.group(0)})
            if path.suffix == ".py" and re.search(r"@triton\.jit", code):
                has_triton = True
    if not has_triton:
        violations.append({"rule": "no_triton", "file": "-", "match": "no @triton.jit kernel found"})
    return violations


def pytest_gate(files, pythonpath, cwd, timeout=3600):
    rc, out, err = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore", *files],
                       cwd=cwd, env={"PYTHONPATH": pythonpath}, timeout=timeout)
    summary = [l for l in (out + err).splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    failed = [l for l in (out + err).splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    return rc == 0, {"returncode": rc, "summary": summary[-1:] if summary else [], "failed": failed[:40], "log": tail(out + err, 60)}


def parse_json_lines(text):
    rows = {}
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                r = json.loads(line)
                rows[r["case"]] = r["median_ms"]
            except (json.JSONDecodeError, KeyError):
                pass
    return rows


def gate_perf(cand_path, ref_path, rounds):
    best = {"ref": {}, "port": {}}
    for _ in range(rounds):
        for tag, pp in (("ref", ref_path), ("port", cand_path)):
            rc, out, err = run([sys.executable, str(HIDDEN / "bench_pcops.py")], env={"PYTHONPATH": pp}, timeout=3600)
            if rc != 0:
                return False, {"error": f"{tag} benchmark failed", "log": tail(out + err)}
            for k, v in parse_json_lines(out).items():
                best[tag][k] = min(v, best[tag].get(k, v))
    cases, ok = [], True
    for case, limit in THRESHOLDS["max_ratio"].items():
        r, p = best["ref"].get(case), best["port"].get(case)
        if r is None or p is None:
            cases.append({"case": case, "pass": False, "error": "missing"})
            ok = False
            continue
        ratio = p / r
        cases.append({"case": case, "cuda_ms": r, "port_ms": p, "ratio": round(ratio, 3), "max_ratio": limit, "pass": ratio <= limit})
        ok &= ratio <= limit
    return ok, {"cases": cases}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--originals", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--perf-rounds", type=int, default=3)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    args.candidate = args.candidate.resolve()
    args.originals = args.originals.resolve()

    work = Path(tempfile.mkdtemp(prefix="pcops-grade-"))
    report = {"task": "task02_pcops_triton", "gates": {}}

    def record(name, ok, detail):
        report["gates"][name] = {"pass": bool(ok), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        return ok

    try:
        cand = work / "pcops"
        shutil.copytree(args.candidate, cand, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__", "*.so"))
        violations = static_checks(cand)
        all_ok = record("static", not violations, {"violations": violations[:50], "count": len(violations)})

        # install the candidate with no CUDA compiler reachable
        site = work / "site"
        env = {"CUDA_HOME": "/nonexistent", "PATH": "/opt/venv/bin:/usr/bin:/bin", "MAX_JOBS": "8"}
        rc, out, err = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", str(site), str(cand)],
                           env=env, timeout=1800)
        so_files = [str(p.relative_to(site)) for p in site.rglob("*.so")] if site.exists() else []
        all_ok &= record("install", rc == 0 and not so_files, {"log": tail(out + err, 20), "compiled_files": so_files[:10]})

        # pristine consumers
        cons = work / "consumers"
        rc2, out2, err2 = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", str(cons),
                               str(args.originals / "pcdata"), str(args.originals / "pcnet")], timeout=1800)
        if rc2 != 0:
            raise RuntimeError("cannot install pcnet / pcdata: " + tail(out2 + err2))
        pp = f"{site}:{cons}:{HIDDEN}"
        hid = work / "hidden"
        shutil.copytree(HIDDEN, hid, ignore=shutil.ignore_patterns("tools", "__pycache__"))

        if rc == 0:
            ok, d = pytest_gate(["test_hidden_ops.py"], pp, hid)
            all_ok &= record("ops", ok, d)
            ok, d = pytest_gate(["test_hidden_hard.py"], pp, hid)
            all_ok &= record("hard", ok, d)
            app_tests = [str(args.originals / "pcnet" / "tests"), str(args.originals / "pcdata" / "tests"), "test_hidden_app.py"]
            ok, d = pytest_gate(app_tests, pp, hid)
            all_ok &= record("app", ok, d)
        else:
            for g in ("ops", "hard", "app"):
                all_ok &= record(g, False, {"skipped": "install failed"})

        if args.skip_perf:
            report["gates"]["perf"] = {"pass": None, "skipped": True}
        elif all_ok:
            ref_site = work / "ref_site"
            rc3, out3, err3 = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", str(ref_site),
                                   str(args.originals / "pcops")], env={"MAX_JOBS": "16"}, timeout=3600)
            if rc3 != 0:
                raise RuntimeError("cannot build the CUDA original: " + tail(out3 + err3))
            ok, d = gate_perf(f"{site}:{cons}", f"{ref_site}:{cons}", args.perf_rounds)
            if not ok:
                # one re-measurement, so that a transient load on a shared GPU
                # cannot fail a correct solution
                ok, d2 = gate_perf(f"{site}:{cons}", f"{ref_site}:{cons}", args.perf_rounds)
                d = {"first_attempt": d, **d2}
            all_ok &= record("perf", ok, d)
        else:
            all_ok &= record("perf", False, {"skipped": "earlier gate failed"})
        report["reward"] = 1.0 if all_ok and not args.skip_perf else 0.0
    finally:
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"reward": report["reward"], "gates": {k: v["pass"] for k, v in report["gates"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
