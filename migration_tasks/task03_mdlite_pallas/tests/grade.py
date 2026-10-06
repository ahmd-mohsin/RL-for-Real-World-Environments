#!/usr/bin/env python3
"""Grader for task03_mdlite_pallas (mdlite: CUDA C++ engine -> JAX + Pallas).

    grade.py --candidate DIR --originals DIR [--out report.json] [--skip-perf]

--candidate  the migrated mdlite repository (agent's patch applied to the original)
--originals  directory holding the pristine mdlite, mdbuild and mdkit repositories

Gates, all required for reward 1:
  static     no C/C++/CUDA sources or native build, no foreign kernel libraries or host
             callbacks, Pallas kernels present
  install    `pip install` of the candidate works with no CUDA compiler and installs no
             compiled extension
  api        configuration / input validation, error contract, CLI, Pallas usage probe
  forces     forces, energy terms, virial, pressure vs goldens from the CUDA original
  dynamics   trajectories of every integrator / thermostat / barostat, output files,
             bitwise determinism, call-splitting invariance, bitwise restarts
  physics    40 ps NVE drift, 60 ps NPT density, 1M-atom and 96k-atom systems
  app        pristine mdbuild / mdkit test suites and mdlite's own release tests
  fresh      randomized systems and settings from a seed drawn at grading time; references
             computed by the CUDA original into a root-only directory
  perf       ms per step vs the original CUDA engine on the same GPU, with an end-state check

Every step that executes candidate code (install, tests, benchmarks) runs as an
unprivileged user that cannot write the grader, the originals or the Python
environment, and all of its processes are killed before reference measurements.
"""
import argparse
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import fresh_cases

HERE = Path(__file__).resolve().parent
HIDDEN = HERE / "hidden"
THRESHOLDS = json.loads((HERE / "perf_thresholds.json").read_text())

FORBIDDEN_FILES = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".so", ".ptx", ".cubin", ".fatbin", ".a", ".o"}
FORBIDDEN_NAMES = {"CMakeLists.txt", "Makefile"}
FORBIDDEN_CODE = [
    ("native_build", r"ext_modules|Extension\s*\(|cmake|nvcc|cpp_extension|load_inline"),
    ("foreign_kernels", r"(?m)^\s*(import|from)\s+(ctypes|cffi|cupy|numba|pycuda|torch|triton|cuda|tensorflow|warp|taichi)\b"),
    ("host_callback", r"pure_callback|io_callback|host_callback|jax\.ffi|ffi_call|xla_client|custom_call"),
    ("load_library", r"CDLL|cdll|LoadLibrary|dlopen"),
]
GATES = (("api", "test_hidden_api.py", 2400), ("forces", "test_hidden_forces.py", 3600),
         ("dynamics", "test_hidden_dynamics.py", 5400), ("physics", "test_hidden_physics.py", 5400))


UNPRIV = 65534   # candidate code (install, tests, benchmarks) never runs as root
UNPRIV_HOME = None


def as_unpriv(cmd):
    return ["setpriv", f"--reuid={UNPRIV}", f"--regid={UNPRIV}", "--clear-groups", "--"] + list(cmd)


def kill_unpriv():
    """No candidate process may survive into a reference measurement."""
    subprocess.run(["pkill", "-9", "-u", str(UNPRIV)], capture_output=True)


def run(cmd, cwd="/", env=None, timeout=3600, unpriv=False):
    # cwd defaults to "/": `python -m` puts the working directory on sys.path, and the
    # grader's own working directory holds the candidate's files
    e = dict(os.environ)
    e["PYTHONDONTWRITEBYTECODE"] = "1"
    e.pop("PYTHONPATH", None)
    if unpriv:
        e["HOME"] = UNPRIV_HOME
        e["TMPDIR"] = UNPRIV_HOME
        cmd = as_unpriv(cmd)
    e.update(env or {})
    try:
        p = subprocess.run(cmd, cwd=cwd, env=e, timeout=timeout, capture_output=True, text=True)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        return 124, ex.stdout or "", f"timeout after {timeout}s"


def tail(text, n=40):
    if isinstance(text, bytes):
        text = text.decode(errors="replace")
    return "\n".join((text or "").splitlines()[-n:])


def strip_py_comments(text):
    return "\n".join(line.split("#", 1)[0] for line in text.splitlines())


def static_checks(repo: Path):
    violations = []
    has_pallas = False
    for path in sorted(repo.rglob("*")):
        rel = path.relative_to(repo)
        if not path.is_file() or (rel.parts and rel.parts[0] in {".git", "tests", "build", "dist", "docs"}) or "egg-info" in str(rel):
            continue
        if path.suffix in FORBIDDEN_FILES or path.name in FORBIDDEN_NAMES:
            violations.append({"rule": "native_source", "file": str(rel)})
            continue
        if path.suffix in {".py", ".toml", ".cfg", ".txt"}:
            text = path.read_text(errors="replace")
            code = strip_py_comments(text) if path.suffix == ".py" else text
            for rule, pattern in FORBIDDEN_CODE:
                for m in re.finditer(pattern, code):
                    violations.append({"rule": rule, "file": str(rel), "match": m.group(0)})
            if path.suffix == ".py" and re.search(r"\bpallas_call\b", code):
                has_pallas = True
    if not has_pallas:
        violations.append({"rule": "no_pallas", "file": "-", "match": "no pallas_call found"})
    return violations


def pytest_gate(files, pythonpath, cwd, timeout):
    rc, out, err = run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-W", "ignore",
                        f"--basetemp={UNPRIV_HOME}/pt", *files],
                       cwd=cwd, env={"PYTHONPATH": pythonpath, "MDLITE_TEST_CACHE": f"{UNPRIV_HOME}/cache"}, timeout=timeout,
                       unpriv=True)
    kill_unpriv()
    summary = [l for l in (out + err).splitlines() if re.search(r"\d+ (passed|failed|error)", l)]
    failed = [l for l in (out + err).splitlines() if l.startswith("FAILED") or l.startswith("ERROR")]
    return rc == 0, {"returncode": rc, "summary": summary[-1:] if summary else [], "failed": failed[:40], "log": tail(out + err, 60)}


def parse_json_lines(text):
    rows = {}
    for line in text.splitlines():
        if line.startswith("{"):
            try:
                r = json.loads(line)
                rows[r["case"]] = r
            except (json.JSONDecodeError, KeyError):
                pass
    return rows


def gate_perf(cand_pp, ref_pp, bench, rounds):
    best = {"ref": {}, "port": {}}
    last = {"ref": {}, "port": {}}
    for _ in range(rounds):
        for tag, pp in (("ref", ref_pp), ("port", cand_pp)):
            kill_unpriv()
            rc, out, err = run([sys.executable, str(bench)], env={"PYTHONPATH": pp}, timeout=3600, unpriv=(tag == "port"))
            kill_unpriv()
            if rc != 0:
                return False, {"error": f"{tag} benchmark failed", "log": tail(out + err)}
            for k, r in parse_json_lines(out).items():
                best[tag][k] = min(r["ms_per_step"], best[tag].get(k, r["ms_per_step"]))
                last[tag][k] = r
    cases, ok = [], True
    for case in THRESHOLDS["max_ratio"]:
        r, p = last["ref"].get(case), last["port"].get(case)
        if r is None or p is None:
            continue
        # the benchmark must have simulated the same physics (end-state potential and temperature)
        sane = abs(p["pot_per_atom"] - r["pot_per_atom"]) <= 0.03 * abs(r["pot_per_atom"]) + 0.05 and \
            abs(p["temperature"] - r["temperature"]) <= 0.05 * r["temperature"] + 2.0 and p["steps"] == r["steps"]
        if not sane:
            cases.append({"case": case, "pass": False, "error": "benchmark end state differs from the original's",
                          "ref": r, "port": p})
            ok = False
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

    global UNPRIV_HOME
    work = Path(tempfile.mkdtemp(prefix="mdlite-grade-"))
    os.chmod(work, 0o755)
    report = {"task": "task03_mdlite_pallas", "gates": {}}

    def record(name, ok, detail):
        report["gates"][name] = {"pass": bool(ok), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        return ok

    def unpriv_dir(path):
        path.mkdir(parents=True, exist_ok=True)
        os.lchown(path, UNPRIV, UNPRIV)
        return path

    try:
        UNPRIV_HOME = str(unpriv_dir(work / "home"))
        # the CUDA original and the pristine consumers are installed by root, before any candidate code runs
        ref_site = work / "ref_site"
        rc3, out3, err3 = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target",
                               str(ref_site), str(args.originals / "mdlite")], env={"MAX_JOBS": "16"}, timeout=3600)
        if rc3 != 0:
            raise RuntimeError("cannot build the CUDA original: " + tail(out3 + err3))
        cons = work / "consumers"
        rc2, out2, err2 = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", str(cons),
                               str(args.originals / "mdbuild"), str(args.originals / "mdkit")], timeout=1800)
        if rc2 != 0:
            raise RuntimeError("cannot install mdbuild / mdkit: " + tail(out2 + err2))
        hid = work / "hidden"
        shutil.copytree(HIDDEN, hid, ignore=shutil.ignore_patterns("tools", "__pycache__"))
        app_src = work / "app_tests"
        for r in ("mdbuild", "mdkit", "mdlite"):
            shutil.copytree(args.originals / r / "tests", app_src / r)

        cand = work / "mdlite"
        shutil.copytree(args.candidate, cand, ignore=shutil.ignore_patterns(".git", "build", "dist", "*.egg-info", "__pycache__", "*.so"))
        violations = static_checks(cand)
        all_ok = record("static", not violations, {"violations": violations[:50], "count": len(violations)})

        # install the candidate as the unprivileged user, with no CUDA compiler reachable
        for pth in [cand] + list(cand.rglob("*")):
            os.lchown(pth, UNPRIV, UNPRIV)
        site = unpriv_dir(work / "site")
        env = {"CUDA_HOME": "/nonexistent", "CUDACXX": "/nonexistent", "PATH": "/opt/venv/bin:/usr/bin:/bin"}
        rc, out, err = run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir", "-q",
                            "--target", str(site), str(cand)], env=env, timeout=1800, unpriv=True)
        kill_unpriv()
        so_files = [str(p.relative_to(site)) for p in site.rglob("*.so")] if site.exists() else []
        all_ok &= record("install", rc == 0 and not so_files, {"log": tail(out + err, 20), "compiled_files": so_files[:10]})
        pp = f"{site}:{cons}:{hid}"

        if rc == 0:
            for gate, fname, tmo in GATES:
                ok, d = pytest_gate([fname], pp, hid, tmo)
                all_ok &= record(gate, ok, d)
            ok, d = pytest_gate([str(app_src / r) for r in ("mdbuild", "mdkit", "mdlite")], f"{site}:{cons}", work, 2400)
            all_ok &= record("app", ok, d)
            # fresh cases from a secret seed: references by the original (root-only), candidate as the unprivileged user
            fresh = work / "fresh"
            fresh.mkdir()
            seed = secrets.randbits(48)
            rcb, ob, eb = run([sys.executable, str(HERE / "fresh_cases.py"), "build", str(seed), str(fresh)],
                              env={"PYTHONPATH": f"{ref_site}:{cons}"}, timeout=1800)
            refdir = work / "fresh_ref"
            refdir.mkdir(mode=0o700)
            rcr, orr, er = run([sys.executable, str(HERE / "fresh_cases.py"), "run", str(fresh), str(refdir / "ref.npz")],
                               env={"PYTHONPATH": f"{ref_site}:{cons}"}, timeout=1800)
            if rcb != 0 or rcr != 0:
                raise RuntimeError("fresh reference failed: " + tail(ob + eb + orr + er))
            cout = unpriv_dir(work / "fresh_cand")
            rcc, oc, ec = run([sys.executable, str(HERE / "fresh_cases.py"), "run", str(fresh), str(cout / "cand.npz")],
                              env={"PYTHONPATH": f"{site}:{cons}"}, timeout=3600, unpriv=True)
            kill_unpriv()
            if rcc != 0:
                all_ok &= record("fresh", False, {"error": "candidate failed", "log": tail(oc + ec)})
            else:
                terms = ("lj", "lj_14", "disp_corr", "coulomb_sr", "coulomb_14", "coulomb_recip", "bond", "angle", "dihedral",
                         "potential", "kinetic", "total", "conserved", "temperature", "pressure", "volume")
                fails = fresh_cases.compare(refdir / "ref.npz", cout / "cand.npz", terms)
                all_ok &= record("fresh", not fails, {"failures": fails[:30]})
        else:
            for g in [g for g, _, _ in GATES] + ["app", "fresh"]:
                all_ok &= record(g, False, {"skipped": "install failed"})

        if args.skip_perf:
            report["gates"]["perf"] = {"pass": None, "skipped": True}
        elif all_ok:
            bench = hid / "bench_mdlite.py"
            ok, d = gate_perf(f"{site}:{cons}", f"{ref_site}:{cons}", bench, args.perf_rounds)
            if not ok:
                # one re-measurement, so that a transient load on a shared GPU cannot fail a correct solution
                ok, d2 = gate_perf(f"{site}:{cons}", f"{ref_site}:{cons}", bench, args.perf_rounds)
                d = {"first_attempt": d, **d2}
            all_ok &= record("perf", ok, d)
        else:
            all_ok &= record("perf", False, {"skipped": "earlier gate failed"})
        report["reward"] = 1.0 if all_ok and not args.skip_perf else 0.0
    finally:
        kill_unpriv()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)

    if args.out:
        args.out.write_text(json.dumps(report, indent=2))
    print(json.dumps({"reward": report["reward"], "gates": {k: v["pass"] for k, v in report["gates"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
