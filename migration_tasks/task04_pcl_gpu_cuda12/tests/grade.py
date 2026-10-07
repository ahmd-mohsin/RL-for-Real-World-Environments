#!/usr/bin/env python3
"""Grader for task04_pcl_gpu_cuda12 (PCL GPU KinFu / KinFu large scale / people on CUDA 12.8, sm_120).

    grade.py --candidate PCL_DIR --originals DIR [--out report.json] [--skip-perf] [--skip-sanitizer]

--candidate  the migrated PCL tree (the agent's patch applied to the pinned upstream commit)
--originals  directory holding the pristine kinfu_replay and people_eval repositories

The original pre-CUDA-12 stack (CUDA 11.8, compute_86 PTX run through driver JIT) is installed root-only
in /opt/legacy_rt by the verifier image; it computes every reference result and timing at grading time.

Gates, all required for reward 1:
  static      no texture / surface references in the candidate's sources
  build       PCL configures and builds with CUDA 12.8 for sm_120 with no module switches passed;
              gpu_kinfu, gpu_kinfu_large_scale and gpu_people get built
  binary      device code (SASS and PTX) targets sm_120 only, CUDA 12 runtime
  consumers   the pristine kinfu_replay and people_eval build against the candidate's install
  kinfu       KinFu / KinFu large scale workloads vs the original: tracking, TSDF, mesh, cloud, ray cast
  people      body-part labels identical to the original's (OMP_NUM_THREADS=1)
  ncv         NCV / NPP-staging outputs bit-identical to the original's
  concurrency trackers / detectors on several host threads equal the candidate's own solo runs bitwise
  fresh       workloads drawn from a secret seed, references computed by the original
  sanitizer   compute-sanitizer racecheck, synccheck and memcheck report nothing
  perf        per-frame KinFu / KinFu LS, marching cubes, cloud extraction, people: <= 1.10x the original

Every step that executes candidate code (CMake, compiler, the consumers linked against the candidate)
runs as an unprivileged user that cannot read the references, and its processes are killed before
reference measurements.
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
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cases  # noqa: E402

LEGACY = Path(os.environ.get("TASK04_LEGACY_RT", "/opt/legacy_rt"))
THRESHOLDS = json.loads((HERE / "perf_thresholds.json").read_text())
FINGERPRINTS = HERE / "fingerprints.json"
CUDA = "/usr/local/cuda"
SANITIZER = f"{CUDA}/bin/compute-sanitizer"
JOBS = str(min(os.cpu_count() or 8, 16))

UNPRIV = 65534
UNPRIV_HOME = None
CMAKE_FLAGS = [
    "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DWITH_CUDA=ON", "-DBUILD_GPU=ON", "-DCUDA_ARCH_BIN=120",
    "-DBUILD_apps=OFF", "-DBUILD_examples=OFF", "-DBUILD_tools=OFF", "-DBUILD_global_tests=OFF", "-DWITH_OPENNI=OFF",
    "-DWITH_OPENNI2=OFF", "-DWITH_QT=OFF", "-DWITH_PCAP=OFF", "-DWITH_LIBUSB=OFF", "-DWITH_ENSENSO=OFF", "-DWITH_DAVIDSDK=OFF",
    "-DWITH_DSSDK=OFF", "-DWITH_RSSDK=OFF", "-DWITH_RSSDK2=OFF", "-DBUILD_gpu_kinfu_tools=OFF",
    "-DBUILD_gpu_kinfu_large_scale_tools=OFF", "-DBUILD_gpu_people_tools=OFF",
]
REQUIRED_LIBS = ("pcl_gpu_containers", "pcl_gpu_kinfu", "pcl_gpu_kinfu_large_scale", "pcl_gpu_people")
TEXREF = [
    r"\btexture\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]",
    r"\bsurface\s*<[^;{}()]*>\s*[A-Za-z_]\w*\s*[;,=]",
    r"\bcuda(Bind|Unbind)Texture\w*\s*\(",
    r"\bcudaBindSurfaceToArray\s*\(",
    r"\bcudaGet(Texture|Surface)Reference\s*\(",
    r"\bcudaGetTextureAlignmentOffset\s*\(",
    r"\btextureReference\b",
    r"\bsurfaceReference\b",
]


def as_unpriv(cmd):
    return ["setpriv", f"--reuid={UNPRIV}", f"--regid={UNPRIV}", "--clear-groups", "--"] + list(cmd)


def kill_unpriv():
    """No candidate process may survive into a reference measurement."""
    for _ in range(3):
        subprocess.run(["pkill", "-9", "-u", str(UNPRIV)], capture_output=True)
        if subprocess.run(["pgrep", "-u", str(UNPRIV)], capture_output=True).returncode != 0:
            return
        time.sleep(0.5)


def run(cmd, cwd="/", env=None, timeout=3600, unpriv=False):
    e = {"PATH": f"{CUDA}/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}
    if unpriv:
        e["HOME"] = UNPRIV_HOME
        e["TMPDIR"] = UNPRIV_HOME
        cmd = as_unpriv(cmd)
    else:
        e["HOME"] = os.environ.get("HOME", "/root")
    e.update(env or {})
    try:
        p = subprocess.run([str(c) for c in cmd], cwd=cwd, env=e, timeout=timeout, capture_output=True, text=True,
                           errors="replace")
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        out = ex.stdout.decode(errors="replace") if isinstance(ex.stdout, bytes) else (ex.stdout or "")
        return 124, out, f"timeout after {timeout}s"
    finally:
        if unpriv:
            kill_unpriv()


def tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def strip_c_comments(text):
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def static_checks(pcl):
    hits = []
    for path in sorted(pcl.rglob("*")):
        if not path.is_file() or ".git" in path.parts or path.suffix not in {".cu", ".cuh", ".h", ".hpp", ".cpp", ".inl", ".hh"}:
            continue
        rel = path.relative_to(pcl)
        if rel.parts[0] not in {"gpu", "cuda"}:
            continue
        code = strip_c_comments(path.read_text(errors="replace"))
        for pat in TEXREF:
            for m in re.finditer(pat, code):
                hits.append({"file": str(rel), "match": m.group(0)[:120]})
    return hits


def binary_checks(install):
    issues, seen = [], {}
    libs = sorted((install / "lib").glob("libpcl_gpu_*.so*"))
    for name in REQUIRED_LIBS:
        found = [p for p in libs if re.fullmatch(rf"lib{name}\.so(\.[\d.]+)?", p.name)]
        if not found:
            issues.append({"lib": name, "error": "not built"})
    for lib in libs:
        if lib.is_symlink():
            continue
        rc, out, err = run(["cuobjdump", "--list-elf", lib], timeout=300)
        elf = re.findall(r"\.(sm_\w+)\.cubin", out)
        rc2, out2, _ = run(["cuobjdump", "--list-ptx", lib], timeout=300)
        ptx = re.findall(r"\.(sm_\w+|compute_\w+)\.ptx", out2)
        seen[lib.name] = {"elf": sorted(set(elf)), "ptx": sorted(set(ptx))}
        for a in set(elf):
            if not re.fullmatch(r"sm_120[af]?", a):
                issues.append({"lib": lib.name, "error": f"device code for {a}"})
        for a in set(ptx):
            if not re.fullmatch(r"(sm|compute)_120[af]?", a):
                issues.append({"lib": lib.name, "error": f"PTX for {a}"})
        if any(k in lib.name for k in ("kinfu", "people")) and "sm_120" not in " ".join(elf):
            issues.append({"lib": lib.name, "error": "no sm_120 device code"})
        rc3, out3, _ = run(["readelf", "-d", lib], timeout=60)
        for m in re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", out3):
            if re.match(r"lib(cudart|npp\w*|cufft|cublas)\.so\.(\d+)", m) and not re.match(r"lib\w+\.so\.12", m):
                issues.append({"lib": lib.name, "error": f"links {m}"})
        for m in re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r\w*path: \[(.+?)\]", out3):
            for d in m.split(":"):
                if d and not (d.startswith(str(install)) or d.startswith(CUDA) or d.startswith("/usr/lib") or d.startswith("$ORIGIN")):
                    issues.append({"lib": lib.name, "error": f"runpath {d}"})
    return issues, seen


class Grader:
    def __init__(self, args, work):
        self.args = args
        self.work = work
        self.report = {"task": "task04_pcl_gpu_cuda12", "gates": {}}
        self.ok = True

    def record(self, name, ok, detail):
        self.report["gates"][name] = {"pass": bool(ok), **detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name}", flush=True)
        self.ok &= bool(ok)
        return ok

    def udir(self, path):
        path.mkdir(parents=True, exist_ok=True)
        os.lchown(path, UNPRIV, UNPRIV)
        return path

    def chown_tree(self, root, uid=UNPRIV):
        for p in [root] + list(root.rglob("*")):
            os.lchown(p, uid, uid)

    def seal(self, root):
        """Hand a tree back to root (read-only for the candidate): later candidate code cannot alter it."""
        kill_unpriv()
        self.chown_tree(root, 0)
        for p in [root] + list(root.rglob("*")):
            if not p.is_symlink():
                os.chmod(p, (p.stat().st_mode & 0o755) | (0o755 if p.is_dir() else 0o444))

    # ------------------------------------------------------------------ running the tools
    def ref(self, tool, args, out, omp=16, repeat_env=None):
        """The original build, as root, into a root-only directory."""
        out.mkdir(parents=True, exist_ok=True)
        env = {"LD_LIBRARY_PATH": f"{LEGACY}/lib", "OMP_NUM_THREADS": str(omp), "CUDA_CACHE_PATH": str(self.work / "ref" / ".nv"),
               "CUDA_CACHE_MAXSIZE": "4294967296"}
        rc, o, e = run([LEGACY / "bin" / tool, *args, "--out", out], cwd=self.data, env=env, timeout=3600)
        if rc != 0:
            raise RuntimeError(f"original {tool} {' '.join(args)} failed: {tail(o + e)}")
        return out

    def cand(self, tool, args, out, omp=16, timeout=3600, wrap=None):
        """The pristine consumer linked against the candidate, as the unprivileged user."""
        self.udir(out)
        env = {"LD_LIBRARY_PATH": f"{self.install}/lib", "OMP_NUM_THREADS": str(omp)}
        cmd = [self.bins[tool], *args, "--out", out]
        rc, o, e = run((wrap or []) + cmd, cwd=self.data, env=env, timeout=timeout, unpriv=True)
        self.seal(out)
        return rc, o + e

    # ------------------------------------------------------------------ build
    def build(self):
        pcl = self.work / "pcl"
        shutil.copytree(self.args.candidate, pcl, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        hits = static_checks(pcl)
        self.record("static", not hits, {"texture_or_surface_references": hits[:40], "count": len(hits)})
        self.chown_tree(pcl)
        self.install = self.udir(self.work / "install")
        b = self.udir(self.work / "build")
        rc, o, e = run(["cmake", "-S", pcl, "-B", b, f"-DCMAKE_INSTALL_PREFIX={self.install}", *CMAKE_FLAGS], timeout=1800, unpriv=True)
        log = o + e
        if rc == 0:
            rc, o, e = run(["cmake", "--build", b, "-j", JOBS], timeout=7200, unpriv=True)
            log = o + e
            if rc == 0:
                rc, o, e = run(["cmake", "--install", b], timeout=1800, unpriv=True)
                log = o + e
        errors = [l for l in log.splitlines() if re.search(r"error|FAILED", l)][:30]
        if not self.record("build", rc == 0, {"returncode": rc, "errors": errors, "log": tail(log, 30)}):
            return False
        self.seal(self.install)
        self.seal(b)
        issues, seen = binary_checks(self.install)
        self.record("binary", not issues, {"issues": issues[:40], "device_code": seen})
        # pristine consumers against the candidate's install
        self.bins = {}
        cons = self.udir(self.work / "consumers")
        logs, ok = {}, True
        for r in ("kinfu_replay", "people_eval"):
            shutil.copytree(self.args.originals / r, cons / r)
            self.chown_tree(cons / r)
            bd = cons / r / "build"
            rc, o, e = run(["cmake", "-S", cons / r, "-B", bd, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={self.install}"],
                           timeout=900, unpriv=True)
            if rc == 0:
                rc, o, e = run(["cmake", "--build", bd, "-j", JOBS], timeout=1800, unpriv=True)
            logs[r] = tail(o + e, 20)
            ok &= rc == 0
        self.seal(cons)
        for t, r in (("kinfu_replay", "kinfu_replay"), ("people_eval", "people_eval"), ("ncv_check", "people_eval")):
            self.bins[t] = cons / r / "build" / t
            ok &= self.bins[t].exists()
        return self.record("consumers", ok, {"logs": logs})

    # ------------------------------------------------------------------ references
    def references(self):
        R = self.work / "ref"
        R.mkdir(mode=0o700, exist_ok=True)
        self.refs = {"kinfu": {}, "ncv": {}}
        for name, args in cases.KINFU.items():
            self.refs["kinfu"][name] = self.ref("kinfu_replay", args, R / name)
        self.refs["people"] = self.ref("people_eval", self.people_args(cases.PEOPLE_VARIANTS), R / "people", omp=1)
        for s in cases.NCV_SEEDS:
            self.refs["ncv"][s] = self.ref("ncv_check", ["--seed", str(s)], R / f"ncv{s}", omp=1)
        fp = self.fingerprint()
        if FINGERPRINTS.exists():
            want = json.loads(FINGERPRINTS.read_text())
            bad = [k for k in want if k not in fp or not self.fp_match(k, want[k], fp[k])]
            if bad:
                raise RuntimeError(f"the reference build does not reproduce its recorded fingerprints: {bad[:10]}")
        return fp

    def fingerprint(self):
        fp = {}
        for v in cases.PEOPLE_VARIANTS:
            fp[f"people/{v}"] = cases.sha256(self.refs["people"] / f"{v}.labels")
        for s, d in self.refs["ncv"].items():
            for b in sorted(d.glob("*.bin")):
                fp[f"ncv{s}/{b.stem}"] = cases.sha256(b)
        for name, d in self.refs["kinfu"].items():
            r = cases.load(d)
            fp[f"kinfu/{name}"] = [r["trans_err_mean"], r["tsdf"]["observed"], r.get("mesh", {}).get("triangles", 0)]
        return fp

    @staticmethod
    def fp_match(key, want, got):
        if key.startswith("kinfu/"):
            return all(abs(a - b) <= 0.03 * abs(b) + 1e-4 for a, b in zip(got, want))
        return want == got

    def people_args(self, variants, extra=()):
        return ["--pcd", self.data / "test.pcd", "--trees", self.data / "trees", "--variants", ",".join(variants), *extra]

    # ------------------------------------------------------------------ gates
    def gate_kinfu(self):
        C = self.udir(self.work / "cand")
        self.cand_out = {}
        detail, ok = {}, True
        for name, args in cases.KINFU.items():
            rc, log = self.cand("kinfu_replay", args, C / name)
            if rc != 0 or not (C / name / "result.json").exists():
                detail[name] = {"error": "run failed", "log": tail(log, 15)}
                ok = False
                continue
            self.cand_out[name] = C / name
            r = cases.load(self.refs["kinfu"][name])
            try:
                fails = cases.compare_kinfu(C / name, self.refs["kinfu"][name])
                te = round(1e3 * cases.load(C / name)["trans_err_mean"], 4)
            except Exception as ex:  # malformed output
                fails, te = [{"check": "output", "error": repr(ex)}], None
            detail[name] = {"pass": not fails, "failures": fails[:12], "trans_err_mean_mm": [te, round(1e3 * r["trans_err_mean"], 4)]}
            ok &= not fails
        return self.record("kinfu", ok, {"cases": detail})

    def gate_people(self):
        out = self.work / "cand" / "people"
        rc, log = self.cand("people_eval", self.people_args(cases.PEOPLE_VARIANTS), out, omp=1)
        fails = [{"error": "run failed", "log": tail(log, 15)}] if rc != 0 else cases.compare_people(out, self.refs["people"], cases.PEOPLE_VARIANTS)
        self.people_solo = out if rc == 0 else None
        return self.record("people", not fails, {"failures": fails[:20]})

    def gate_ncv(self):
        fails = {}
        for s in cases.NCV_SEEDS:
            out = self.work / "cand" / f"ncv{s}"
            rc, log = self.cand("ncv_check", ["--seed", str(s)], out, omp=1)
            f = [{"error": "run failed", "log": tail(log, 15)}] if rc != 0 else cases.compare_ncv(out, self.refs["ncv"][s])
            if f:
                fails[f"seed{s}"] = f[:25]
        return self.record("ncv", not fails, {"failures": fails})

    def gate_concurrency(self, rounds=2, people_rounds=3):
        C = self.udir(self.work / "conc")
        detail, ok = {}, True
        for name, gargs, spec in cases.CONCURRENT:
            items = spec.split(",")
            solo = []
            for k, item in enumerate(items):
                mode, traj, noise = item.split(":")
                d = C / f"{name}_solo{k}"
                rc, log = self.cand("kinfu_replay", [*gargs, "--mode", mode, "--traj", traj, "--noise", noise], d)
                if rc != 0:
                    detail[name] = {"error": f"solo run {item} failed", "log": tail(log, 15)}
                    ok = False
                    break
                solo.append(d)
            else:
                mism = []
                for rnd in range(rounds):
                    d = C / f"{name}_r{rnd}"
                    rc, log = self.cand("kinfu_replay", [*gargs, "--concurrent", spec], d)
                    if rc != 0:
                        mism.append({"round": rnd, "error": "concurrent run failed", "log": tail(log, 15)})
                        continue
                    for k, item in enumerate(items):
                        try:
                            diff = cases.compare_bitwise_run(d / f"case{k}", solo[k])
                        except Exception as ex:
                            diff = [repr(ex)]
                        if diff:
                            mism.append({"round": rnd, "case": item, "differs": diff})
                detail[name] = {"pass": not mism, "mismatches": mism[:12]}
                ok &= not mism
        # concurrent people detectors vs the solo run
        mism = []
        if self.people_solo is None:
            mism.append({"error": "no solo people run"})
        for rnd in range(people_rounds if self.people_solo else 0):
            d = C / f"people_r{rnd}"
            rc, log = self.cand("people_eval", self.people_args(cases.PEOPLE_VARIANTS, ["--threads", str(cases.PEOPLE_THREADS)]), d, omp=1)
            if rc != 0:
                mism.append({"round": rnd, "error": "run failed", "log": tail(log, 15)})
                continue
            for t in range(cases.PEOPLE_THREADS):
                f = cases.compare_people(d / f"thread{t}", self.people_solo, cases.PEOPLE_VARIANTS)
                if f:
                    mism.append({"round": rnd, "thread": t, "failures": f[:5]})
        detail["people_threads"] = {"pass": not mism, "mismatches": mism[:12]}
        ok &= not mism
        return self.record("concurrency", ok, detail)

    def gate_fresh(self):
        seed = secrets.randbits(40)
        w = cases.fresh_workloads(seed)
        R = self.work / "ref" / "fresh"
        C = self.udir(self.work / "cand_fresh")
        fails = {}
        for k in ("kinfu", "kinfuls"):
            self.ref("kinfu_replay", w[k], R / k)
            rc, log = self.cand("kinfu_replay", w[k], C / k)
            f = [{"error": "run failed", "log": tail(log, 15)}] if rc != 0 else cases.compare_kinfu(C / k, R / k)
            if f:
                fails[k] = f[:12]
        self.ref("people_eval", self.people_args(w["people"]), R / "people", omp=1)
        rc, log = self.cand("people_eval", self.people_args(w["people"]), C / "people", omp=1)
        f = [{"error": "run failed", "log": tail(log, 15)}] if rc != 0 else cases.compare_people(C / "people", R / "people", w["people"])
        if f:
            fails["people"] = f
        self.ref("ncv_check", ["--seed", str(w["ncv"])], R / "ncv", omp=1)
        rc, log = self.cand("ncv_check", ["--seed", str(w["ncv"])], C / "ncv", omp=1)
        f = [{"error": "run failed", "log": tail(log, 15)}] if rc != 0 else cases.compare_ncv(C / "ncv", R / "ncv")
        if f:
            fails["ncv"] = f[:20]
        # the seed is only revealed after the candidate has run
        return self.record("fresh", not fails, {"seed": seed, "failures": fails})

    def gate_sanitizer(self):
        S = self.udir(self.work / "san")
        detail, ok = {}, True
        for tool in ("racecheck", "synccheck", "memcheck"):
            for name, binary, args in cases.SANITIZE:
                if binary == "people_eval":
                    args = self.people_args(args[1].split(","))
                wrap = [SANITIZER, "--tool", tool, "--print-limit", "30", "--error-exitcode", "97"]
                if tool == "memcheck":
                    wrap += ["--leak-check", "no"]
                rc, log = self.cand(binary, args, S / f"{tool}_{name}", omp=1, timeout=5400, wrap=wrap)
                if tool == "racecheck":
                    m = re.search(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", log)
                    clean = rc == 0 and m is not None and int(m.group(1)) == 0 and not re.search(r"ERROR SUMMARY: [1-9]", log)
                else:
                    clean = rc == 0 and "ERROR SUMMARY: 0 errors" in log
                produced = (S / f"{tool}_{name}" / "result.json").exists()
                hazards = [l for l in log.splitlines() if re.search(r"(Error|Warning|Hazard|Invalid|Race)", l)][:15]
                detail[f"{tool}/{name}"] = {"pass": clean and produced, "returncode": rc,
                                            "summary": [l for l in log.splitlines() if "SUMMARY" in l][-3:], "first_reports": hazards}
                ok &= clean and produced
        return self.record("sanitizer", ok, detail)

    def perf_round(self, best, last, rnd):
        P = self.work / "ref" / "perf"
        C = self.udir(self.work / "cand_perf")
        work = [("kinfu", "kinfu_replay", cases.KINFU["kinfu_orbit_clean"], 16),
                ("kinfuls", "kinfu_replay", cases.KINFU["kinfuls_walk_clean"] + ["--skip-mesh"], 16),
                ("people", "people_eval", self.people_args(cases.PEOPLE_VARIANTS[:5], ["--repeat", "15"]), 1)]
        for tag, tool, args, omp in work:
            kill_unpriv()
            r = self.ref(tool, args, P / f"{tag}{rnd}", omp=omp)
            rc, log = self.cand(tool, args, C / f"{tag}{rnd}", omp=omp)
            if rc != 0:
                raise RuntimeError(f"candidate benchmark {tag} failed: {tail(log, 10)}")
            c = C / f"{tag}{rnd}"
            if tag == "people":
                f = cases.compare_people(c, r, cases.PEOPLE_VARIANTS[:5])
                if f:
                    raise RuntimeError(f"candidate benchmark {tag} computed different results: {f[:3]}")
                metrics = {"people_ms": lambda j: sum(v["ms_median"] for v in j.values())}
            else:
                f = cases.compare_kinfu(c, r)
                if f:
                    raise RuntimeError(f"candidate benchmark {tag} computed different results: {f[:3]}")
                metrics = {f"{tag}_frame_ms": lambda j: j["frame_ms_median"]}
                if tag == "kinfu":
                    metrics.update({"marching_cubes_ms": lambda j: j["mc_ms"], "cloud_ms": lambda j: j["cloud_ms"]})
            for side, d in (("original", r), ("candidate", c)):
                j = cases.load(d)
                for m, fn in metrics.items():
                    v = float(fn(j))
                    best[side][m] = min(v, best[side].get(m, v))
                    last[side].setdefault(m, []).append(round(v, 4))

    def gate_perf(self, rounds):
        def attempt():
            best, last = {"original": {}, "candidate": {}}, {"original": {}, "candidate": {}}
            try:
                for rnd in range(rounds):
                    self.perf_round(best, last, rnd)
            except RuntimeError as ex:
                return False, {"error": str(ex)}
            res, ok = {}, True
            for m, limit in THRESHOLDS["max_ratio"].items():
                o, c = best["original"].get(m), best["candidate"].get(m)
                ratio = c / o if o and c is not None else float("inf")
                res[m] = {"original_ms": round(o, 4) if o else None, "candidate_ms": round(c, 4) if c is not None else None,
                          "ratio": round(ratio, 3), "max_ratio": limit, "pass": ratio <= limit}
                ok &= ratio <= limit
            return ok, {"metrics": res, "rounds": last}

        ok, d = attempt()
        if not ok and "error" not in d:
            # one re-measurement, so that a transient load on a shared GPU cannot fail a correct solution
            ok, d2 = attempt()
            d = {"first_attempt": d, **d2}
        return self.record("perf", ok, d)

    # ------------------------------------------------------------------ main
    def main(self):
        global UNPRIV_HOME
        UNPRIV_HOME = str(self.udir(self.work / "home"))
        self.data = self.work / "data"
        (self.data / "trees").mkdir(parents=True)
        src = self.args.originals / "people_eval" / "data"
        shutil.copy(src / "test.pcd", self.data / "test.pcd")
        for t in sorted((src / "trees").glob("*.bz2")):
            rc, o, e = run(["bash", "-c", f"bzcat '{t}' > '{self.data / 'trees' / t.stem}'"])
            if rc != 0:
                raise RuntimeError(f"cannot decompress {t}: {e}")
        for p in [self.data] + list(self.data.rglob("*")):
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        if not self.build():
            for g in ("kinfu", "people", "ncv", "concurrency", "fresh", "sanitizer", "perf"):
                self.record(g, False, {"skipped": "build failed"})
            return
        self.report["reference_fingerprint"] = fp = self.references()
        if self.args.record_fingerprints:
            Path(self.args.record_fingerprints).write_text(json.dumps(fp, indent=1, sort_keys=True))
        self.gate_kinfu()
        self.gate_people()
        self.gate_ncv()
        self.gate_concurrency()
        self.gate_fresh()
        if self.args.skip_sanitizer:
            self.report["gates"]["sanitizer"] = {"pass": None, "skipped": True}
        else:
            self.gate_sanitizer()
        if self.args.skip_perf:
            self.report["gates"]["perf"] = {"pass": None, "skipped": True}
        elif self.ok:
            self.gate_perf(self.args.perf_rounds)
        else:
            self.record("perf", False, {"skipped": "earlier gate failed"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidate", required=True, type=Path)
    ap.add_argument("--originals", required=True, type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--skip-perf", action="store_true")
    ap.add_argument("--skip-sanitizer", action="store_true")
    ap.add_argument("--perf-rounds", type=int, default=3)
    ap.add_argument("--record-fingerprints", help="write the reference fingerprints here (task authoring)")
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()
    args.candidate = args.candidate.resolve()
    args.originals = args.originals.resolve()
    if os.geteuid() != 0:
        sys.exit("grade.py must run as root (it drops privileges for candidate code)")
    work = Path(tempfile.mkdtemp(prefix="pcl-grade-"))
    os.chmod(work, 0o755)
    g = Grader(args, work)
    t0 = time.time()
    try:
        g.main()
    except Exception as ex:  # infrastructure failure or malformed candidate output
        g.record("error", False, {"error": repr(ex)})
    finally:
        kill_unpriv()
        if not args.keep:
            shutil.rmtree(work, ignore_errors=True)
    skipped = args.skip_perf or args.skip_sanitizer
    g.report["reward"] = 1.0 if g.ok and not skipped else 0.0
    g.report["elapsed_s"] = round(time.time() - t0)
    if args.out:
        args.out.write_text(json.dumps(g.report, indent=2, default=str))
    print(json.dumps({"reward": g.report["reward"], "gates": {k: v["pass"] for k, v in g.report["gates"].items()}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
