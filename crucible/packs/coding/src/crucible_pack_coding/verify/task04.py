"""task04_pcl_gpu_cuda12 grader re-expressed with the M7 kit (runs inside ``task04-pcl-gpu-cuda12-verifier``).

Standard library + the kit runtime + the task's own ``tests/cases.py`` (workloads and numpy comparators, which stay
task-owned). Gate for gate the same as ``migration_tasks/task04_pcl_gpu_cuda12/tests/grade.py`` (v3):

    static      kit ``static.scan``: no texture/surface references under gpu/ and cuda/ (comment-stripped)
    build       grader-owned CMake flags, run unprivileged; install sealed
    binary      SASS/PTX sm_120 only, CUDA 12 runtime, sane RUNPATH (cuobjdump/readelf, root, no execution)
    consumers   pristine kinfu_replay / people_eval built against the candidate install, unprivileged, sealed
    kinfu       kit ``differential``: original (root, /opt/legacy_rt) vs candidate, cases.compare_kinfu
    people      differential, OMP_NUM_THREADS=1, cases.compare_people
    ncv         differential over NCV seeds, bit-identical
    concurrency kit ``concurrency_equivalence``: concurrent trackers/detectors == the candidate's own solo runs
    fresh       kit ``fresh``: workloads from a seed drawn at grading time
    sanitizer   racecheck/synccheck/memcheck through the kit's summary rule
    perf        kit ``PerfBars``: best-of-3, alternating, one re-measurement, bars from perf_thresholds.json

Changes from the seed grader (DESIGN Appendix B.10/B.12), none of which changes a verdict on honest submissions:
- every candidate invocation gets a fresh HOME/TMPDIR and its cohort and IPC objects are removed (the seed grader
  shared one HOME for the whole grade: cache-and-replay of perf runs);
- the sanitizer verdict requires *every* summary line to report zero and a zero exit code (the seed grader read the
  *first* RACECHECK summary, which candidate output can forge);
- perf rounds are load-guarded, spread-checked (contaminated rounds are re-graded, never scored), and the candidate's
  whole-process wall time, measured by the grader, must also beat the original (``wall_ratio_max``), so timers
  forged inside the candidate's process cannot buy the bars;
- any grader exception masks the grade (reward 0, ``infra``) instead of reading as a failed gate.
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

from crucible.verify.runtime.patterns import differential as D
from crucible.verify.runtime.patterns.perf import CandidateError, LoadGuard, PerfBars, gpu_memory_settled, oom_under_contention
from crucible.verify.runtime.patterns.record import RecordSpec
from crucible.verify.runtime.patterns.static import Rule, RuleSet, scan
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import InfraError

TASK = "task04_pcl_gpu_cuda12"
CUDA = "/usr/local/cuda"
GATES = ["static", "build", "binary", "consumers", "kinfu", "people", "ncv", "concurrency", "fresh", "sanitizer", "perf"]
# M7 R6: the instruction's prohibitions (no texture/surface references, sm_120 device code only) are hard gates; the build,
# the differential and incident checks and perf are outcomes (perf a score in record mode)
KINDS = {x: "gate" if x in ("static", "binary") else "outcome" for x in GATES}
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
CODE = {".cu", ".cuh", ".h", ".hpp", ".cpp", ".inl", ".hh", ".hxx", ".cc", ".cxx", ".c", ".inc", ".ipp", ".tpp"}  # M7 audit: `texture<>` in an #included .inc passed
RULES = RuleSet(forbid=[Rule(f"texref{i}", p, suffixes=CODE, roots=("gpu", "cuda")) for i, p in enumerate(TEXREF)],
                exclude=(".git",), runner_tamper=False)


# Span scopes (M8 G-span): a derived task grades only its module's workloads, gates and bars. "full" is the seed task.
SCOPES = {
    "full": {"roots": ("gpu", "cuda"), "kinfu": lambda n: True, "people": True, "ncv": True, "conc": lambda n: True, "people_threads": True,
             "fresh": lambda n: True, "sanitize": lambda n: True, "perf": ("kinfu", "kinfuls", "people", "kinfu4")},
    "people": {"roots": ("gpu/people",), "kinfu": lambda n: False, "people": True, "ncv": True, "conc": lambda n: False, "people_threads": True,
               "fresh": lambda n: n.startswith(("people", "ncv")), "sanitize": lambda n: n in ("people", "ncv"), "perf": ("people",)},
    "ncv": {"roots": ("gpu/people/src/cuda/nvidia",), "kinfu": lambda n: False, "people": False, "ncv": True, "conc": lambda n: False, "people_threads": False,
            "fresh": lambda n: n.startswith("ncv"), "sanitize": lambda n: n == "ncv", "perf": ()},
    "kinfu": {"roots": ("gpu/kinfu",), "kinfu": lambda n: n.startswith("kinfu_"), "people": False, "ncv": False, "conc": lambda n: n.endswith("_kinfu"),
              "people_threads": False, "fresh": lambda n: n.startswith("kinfu") and not n.startswith("kinfuls"), "sanitize": lambda n: n == "kinfu",
              "perf": ("kinfu", "kinfu4")},
    "kinfu_large_scale": {"roots": ("gpu/kinfu_large_scale",), "kinfu": lambda n: n.startswith("kinfuls_"), "people": False, "ncv": False,
                          "conc": lambda n: n.endswith("_kinfuls"), "people_threads": False, "fresh": lambda n: n.startswith("kinfuls"),
                          "sanitize": lambda n: n == "kinfuls", "perf": ("kinfuls",)},
}
# one span for the fork pair (m13 G-span closure): the pristine kinfu_replay consumer links both KinFu and KinFu-LS
# (find_package(PCL COMPONENTS gpu_kinfu gpu_kinfu_large_scale)), so a span that leaves either sibling unported can
# never pass ``consumers`` (found by the forge GPU run: the KinFu-LS oracle failed there)
SCOPES["kinfu_family"] = {"roots": ("gpu/kinfu", "gpu/kinfu_large_scale"), "kinfu": lambda n: n.startswith(("kinfu_", "kinfuls_")), "people": False,
                          "ncv": False, "conc": lambda n: n.endswith(("_kinfu", "_kinfuls")), "people_threads": False,
                          "fresh": lambda n: n.startswith("kinfu"), "sanitize": lambda n: n in ("kinfu", "kinfuls"), "perf": ("kinfu", "kinfuls", "kinfu4")}
SCOPE_LIBS = {"full": None, "people": ("pcl_gpu_containers", "pcl_gpu_people"), "ncv": ("pcl_gpu_containers", "pcl_gpu_people"),
              "kinfu": ("pcl_gpu_containers", "pcl_gpu_kinfu"), "kinfu_large_scale": ("pcl_gpu_containers", "pcl_gpu_kinfu_large_scale"),
              "kinfu_family": ("pcl_gpu_containers", "pcl_gpu_kinfu", "pcl_gpu_kinfu_large_scale")}
PERF_METRICS = {"kinfu": ("kinfu_frame_ms", "marching_cubes_ms", "cloud_ms"), "kinfuls": ("kinfuls_frame_ms",), "people": ("people_ms",),
                "kinfu4": ("kinfu_concurrent_wall_ms",)}


def scoped_rules(roots: tuple) -> RuleSet:
    def top_ok(prefixes):  # Rule.roots matches the first path component; deeper scopes filter in scan_scoped
        return tuple(sorted({r.split("/", 1)[0] for r in prefixes}))

    return RuleSet(forbid=[Rule(f"texref{i}", p, suffixes=CODE, roots=top_ok(roots)) for i, p in enumerate(TEXREF)], exclude=(".git",), runner_tamper=False)


def scan_scoped(root: Path, roots: tuple) -> dict:
    r = scan(root, scoped_rules(roots))
    keep = [v for v in r["violations"] if any(v["file"].startswith(x) for x in roots)]
    return {**r, "violations": keep, "count": len(keep), "pass": not keep}


# Record mode (P29): best-known ratios = the v3 oracle's recorded ratios on the lab RTX 5090 (glogs/oracle3, 2026-10-08).
BEST_KNOWN = {"kinfu_frame_ms": 0.571, "marching_cubes_ms": 0.428, "cloud_ms": 0.614, "kinfuls_frame_ms": 0.582, "people_ms": 0.342, "kinfu_concurrent_wall_ms": 0.376}


def record_score(metrics: dict, bars: dict, best: dict, *, floor: str = "all_bars") -> dict:
    """Per metric: normalized = (bar - ratio) / (bar - best_known): 0 at the bar, 1 at the best known, above 1 for a
    faster-than-known port (log tail). ``floor="all_bars"`` keeps the seed's requirement that every bar is met."""
    per, below = {}, []
    for m, bar in bars.items():
        ratio = metrics.get(m, {}).get("ratio")
        if ratio is None or not (isinstance(ratio, (int, float)) and math.isfinite(ratio) and ratio > 0):  # M7 audit V5: -1000 scored 9.38
            below.append(m)
            continue
        spec = RecordSpec(baseline=bar, best_known=min(best.get(m, bar * 0.9), bar * 0.999), direction="min")
        per[m] = {"ratio": ratio, "bar": bar, "best_known": spec.best_known, "normalized": round(spec.normalize(ratio), 4), "reward": round(spec.reward(ratio), 4)}
        if ratio > bar:
            below.append(m)
    valid = not below if floor == "all_bars" else True
    score = sum(v["reward"] for v in per.values()) / len(bars) if per and valid else 0.0
    return {"score": round(score, 4), "valid": valid, "below_bar": below, "metrics": per, "new_records": [m for m, v in per.items() if v["normalized"] > 1]}


def tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def sanitizer_clean(tool: str, rc: int, log: str) -> tuple[bool, list[str]]:
    """Every compute-sanitizer summary line must report zero, at least one must exist, and the exit code must be 0."""
    if tool == "racecheck":
        sums = re.findall(r"RACECHECK SUMMARY: (\d+) hazards? displayed \((\d+) errors?, (\d+) warnings?\)", log)
        counts = [int(a) + int(b) + int(c) for a, b, c in sums]
    else:
        sums = re.findall(r"ERROR SUMMARY: (\d+) errors?", log)
        counts = [int(x) for x in sums]
    errs = re.findall(r"ERROR SUMMARY: (\d+) errors?", log)
    lines = [ln for ln in log.splitlines() if "SUMMARY" in ln][-4:]
    return rc == 0 and bool(counts) and not any(counts) and not any(int(e) for e in errs), lines


class Task04:
    def __init__(self, ctx: Context):
        self.ctx = ctx
        self.t = ctx.trusted
        self.tests = Path(ctx.params["tests"])
        sys.path.insert(0, str(self.tests))
        import cases  # task-owned workloads and comparators (numpy)

        self.cases = cases
        self.legacy = Path(ctx.params.get("legacy", "/opt/legacy_rt"))
        self.originals = Path(ctx.params.get("originals", "/opt/originals"))
        self.thresholds = json.loads((self.tests / "perf_thresholds.json").read_text())
        self.jobs = str(min(os.cpu_count() or 8, 16))
        self.work = ctx.work
        self.data = self.work / "data"
        self.bins: dict[str, Path] = {}
        self.people_solo = None
        self.t.path = f"{CUDA}/bin:" + self.t.path
        self.scope_name = ctx.params.get("scope", "full")
        self.scope = SCOPES[self.scope_name]

    # ------------------------------------------------------------------ setup (root, before any candidate code)
    def setup(self, ctx: Context) -> None:
        (self.data / "trees").mkdir(parents=True)
        src = self.originals / "people_eval" / "data"
        shutil.copy(src / "test.pcd", self.data / "test.pcd")
        for tr in sorted((src / "trees").glob("*.bz2")):
            r = self.t.run(["bash", "-c", f"bzcat '{tr}' > '{self.data / 'trees' / tr.stem}'"])
            if r.rc != 0:
                raise InfraError(f"cannot decompress {tr}: {r.stderr[-300:]}")
        for p in [self.data, *self.data.rglob("*")]:
            os.chmod(p, 0o755 if p.is_dir() else 0o644)
        self.refs = D.References(self.t.private(self.work / "ref"), self._ref_run, self.t)

    def _ref_run(self, name, spec, out):
        tool, args, omp = spec
        env = {"LD_LIBRARY_PATH": f"{self.legacy}/lib", "OMP_NUM_THREADS": str(omp), "CUDA_CACHE_PATH": str(self.work / "ref" / ".nv"),
               "CUDA_CACHE_MAXSIZE": "4294967296"}
        r = self.t.run([self.legacy / "bin" / tool, *args, "--out", out], cwd=self.data, env=env, timeout=3600)
        if r.rc != 0:
            raise InfraError(f"original {tool} {' '.join(map(str, args))} failed: {tail(r.log)}")

    def cand(self, tool, args, out, omp=16, timeout=3600, wrap=None):
        self.t.udir(out)
        env = {"LD_LIBRARY_PATH": f"{self.ctx.artifacts['install']}/lib", "OMP_NUM_THREADS": str(omp)}
        self.t.kill()
        foreign = gpu_memory_settled()  # no candidate process alive and its memory released: what is in use belongs to someone else
        rr = self.t.run([*(wrap or []), self.bins[tool], *args, "--out", out], cwd=self.data, env=env, timeout=timeout, candidate=True)
        if rr.rc != 0 and oom_under_contention(rr.log, foreign):
            self.t.kill()
            after = gpu_memory_settled()  # still held once the candidate is dead? then it really was someone else's
            if oom_under_contention(rr.log, foreign, after_mib=after):
                raise InfraError(f"node_contention: GPU out of memory while other processes held {foreign} MiB before and {after} MiB after ({tool})")
        return rr

    def people_args(self, variants, extra=()):
        return ["--pcd", self.data / "test.pcd", "--trees", self.data / "trees", "--variants", ",".join(variants), *extra]

    # ------------------------------------------------------------------ static / build / binary / consumers
    def static(self, ctx):
        pcl = self.work / "pcl"
        shutil.copytree(ctx.candidate, pcl, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        ctx.artifacts["pcl"] = pcl
        r = scan(pcl, RULES) if self.scope_name == "full" else scan_scoped(pcl, self.scope["roots"])
        return Verdict(r["pass"], {"texture_or_surface_references": r["violations"][:40], "count": r["count"], "scope": self.scope_name})

    def build(self, ctx):
        pcl = ctx.artifacts.get("pcl")
        if pcl is None:  # static did not copy (fast tier ordering): copy now
            self.static(ctx)
            pcl = ctx.artifacts["pcl"]
        self.t.give(pcl)
        install = self.t.udir(self.work / "install")
        b = self.t.udir(self.work / "build")
        ctx.artifacts["install"] = install
        r = self.t.run(["cmake", "-S", pcl, "-B", b, f"-DCMAKE_INSTALL_PREFIX={install}", *CMAKE_FLAGS], timeout=1800, candidate=True)
        if r.rc == 0:
            r = self.t.run(["cmake", "--build", b, "-j", self.jobs], timeout=7200, candidate=True)
            if r.rc == 0:
                r = self.t.run(["cmake", "--install", b], timeout=1800, candidate=True)
        errors = [ln for ln in r.log.splitlines() if re.search(r"error|FAILED", ln)][:30]
        self.t.seal(install)
        self.t.seal(b)
        return Verdict(r.rc == 0, {"returncode": r.rc, "errors": errors, "log": tail(r.log, 30)})

    def binary(self, ctx):
        install = ctx.artifacts["install"]
        issues, seen = [], {}
        libs = sorted((install / "lib").glob("libpcl_gpu_*.so*"))
        for name in SCOPE_LIBS.get(self.scope_name) or REQUIRED_LIBS:
            if not [p for p in libs if re.fullmatch(rf"lib{name}\.so(\.[\d.]+)?", p.name)]:
                issues.append({"lib": name, "error": "not built"})
        for lib in libs:
            if lib.is_symlink():
                continue
            elf = re.findall(r"\.(sm_\w+)\.cubin", self.t.run(["cuobjdump", "--list-elf", lib], timeout=300).stdout)
            ptx = re.findall(r"\.(sm_\w+|compute_\w+)\.ptx", self.t.run(["cuobjdump", "--list-ptx", lib], timeout=300).stdout)
            seen[lib.name] = {"elf": sorted(set(elf)), "ptx": sorted(set(ptx))}
            issues += [{"lib": lib.name, "error": f"device code for {a}"} for a in set(elf) if not re.fullmatch(r"sm_120[af]?", a)]
            issues += [{"lib": lib.name, "error": f"PTX for {a}"} for a in set(ptx) if not re.fullmatch(r"(sm|compute)_120[af]?", a)]
            if any(k in lib.name for k in ("kinfu", "people")) and "sm_120" not in " ".join(elf):
                issues.append({"lib": lib.name, "error": "no sm_120 device code"})
            dyn = self.t.run(["readelf", "-d", lib], timeout=60).stdout
            for m in re.findall(r"\(NEEDED\)\s+Shared library: \[(.+?)\]", dyn):
                if re.match(r"lib(cudart|npp\w*|cufft|cublas)\.so\.(\d+)", m) and not re.match(r"lib\w+\.so\.12", m):
                    issues.append({"lib": lib.name, "error": f"links {m}"})
            for m in re.findall(r"\((?:RPATH|RUNPATH)\)\s+Library r\w*path: \[(.+?)\]", dyn):
                for d in m.split(":"):
                    if d and not d.startswith((str(install), CUDA, "/usr/lib", "$ORIGIN")):
                        issues.append({"lib": lib.name, "error": f"runpath {d}"})
        return Verdict(not issues, {"issues": issues[:40], "device_code": seen})

    def consumers(self, ctx):
        install = ctx.artifacts["install"]
        cons = self.t.udir(self.work / "consumers")
        logs, ok = {}, True
        for r in ("kinfu_replay", "people_eval"):
            shutil.copytree(self.originals / r, cons / r)
            self.t.give(cons / r)
            bd = cons / r / "build"
            res = self.t.run(["cmake", "-S", cons / r, "-B", bd, "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={install}"], timeout=900, candidate=True)
            if res.rc == 0:
                res = self.t.run(["cmake", "--build", bd, "-j", self.jobs], timeout=1800, candidate=True)
            logs[r] = tail(res.log, 20)
            ok &= res.rc == 0
        self.t.seal(cons)
        for tool, r in (("kinfu_replay", "kinfu_replay"), ("people_eval", "people_eval"), ("ncv_check", "people_eval")):
            self.bins[tool] = cons / r / "build" / tool
            ok &= self.bins[tool].exists()
        if ok:
            self._fingerprint_refs()
        return Verdict(ok, {"logs": logs})

    def _fingerprint_refs(self):
        """References for every fixed workload, checked against the recorded fingerprints (a stale or broken verifier
        image is infra, not a candidate failure; Appendix B.5)."""
        c, sc = self.cases, self.scope
        kin = [n for n in c.KINFU if sc["kinfu"](n)]
        for name in kin:
            self.refs.get(name, ("kinfu_replay", c.KINFU[name], 16))
        if sc["people"] or sc["people_threads"]:
            self.refs.get("people", ("people_eval", self.people_args(c.PEOPLE_VARIANTS), 1))
        if sc["ncv"]:
            for s in c.NCV_SEEDS:
                self.refs.get(f"ncv{s}", ("ncv_check", ["--seed", str(s)], 1))
        fp = {}
        if "people" in self.refs.done:
            for v in c.PEOPLE_VARIANTS:
                fp[f"people/{v}"] = c.sha256(self.refs.done["people"] / f"{v}.labels")
        for s in c.NCV_SEEDS if sc["ncv"] else []:
            for b in sorted(self.refs.done[f"ncv{s}"].glob("*.bin")):
                fp[f"ncv{s}/{b.stem}"] = c.sha256(b)
        for name in kin:
            r = c.load(self.refs.done[name])
            fp[f"kinfu/{name}"] = [r["trans_err_mean"], r["tsdf"]["observed"], r.get("mesh", {}).get("triangles", 0)]
        self.ctx.revealed["reference_fingerprint_keys"] = len(fp)
        f = self.tests / "fingerprints.json"
        if f.exists():
            want = json.loads(f.read_text())

            def match(k, w, g):
                return all(abs(a - b) <= 0.03 * abs(b) + 1e-4 for a, b in zip(g, w, strict=False)) if k.startswith("kinfu/") else w == g

            bad = [k for k in want if k in fp and not match(k, want[k], fp[k])] + ([] if self.scope_name != "full" else [k for k in want if k not in fp])
            if bad:
                raise InfraError(f"the reference build does not reproduce its recorded fingerprints: {bad[:10]}")

    # ------------------------------------------------------------------ differential gates
    def kinfu(self, ctx):
        c = self.cases
        cases = {n: ("kinfu_replay", a, 16) for n, a in c.KINFU.items() if self.scope["kinfu"](n)}
        return D.differential(ctx, cases=cases, refs=self.refs, out_root=self.work / "cand",
                              candidate=lambda n, s, out: self.cand("kinfu_replay", s[1], out),
                              success=lambda rr, out: rr.rc == 0 and (out / "result.json").exists(),
                              compare=lambda cd, rd: c.compare_kinfu(cd, rd))

    def people(self, ctx):
        c = self.cases
        out = self.work / "cand" / "people"
        rr = self.cand("people_eval", self.people_args(c.PEOPLE_VARIANTS), out, omp=1)
        self.t.seal(out)
        fails = [{"error": "run failed", "log": tail(rr.log, 15)}] if rr.rc != 0 else c.compare_people(out, self.refs.done["people"], c.PEOPLE_VARIANTS)
        self.people_solo = out if rr.rc == 0 else None
        return Verdict(not fails, {"failures": fails[:20]})

    def ncv(self, ctx):
        c = self.cases
        fails = {}
        for s in c.NCV_SEEDS:
            out = self.work / "cand" / f"ncv{s}"
            rr = self.cand("ncv_check", ["--seed", str(s)], out, omp=1)
            self.t.seal(out)
            f = [{"error": "run failed", "log": tail(rr.log, 15)}] if rr.rc != 0 else c.compare_ncv(out, self.refs.done[f"ncv{s}"])
            if f:
                fails[f"seed{s}"] = f[:25]
        return Verdict(not fails, {"failures": fails})

    def concurrency(self, ctx):
        c = self.cases
        detail, ok = {}, True
        root = self.t.udir(self.work / "conc")
        for name, gargs, spec in [x for x in c.CONCURRENT if self.scope["conc"](x[0])]:
            items = spec.split(",")

            def solo(k, item, out, gargs=gargs):
                mode, traj, noise = item.split(":")
                return self.cand("kinfu_replay", [*gargs, "--mode", mode, "--traj", traj, "--noise", noise], out)

            v = D.concurrency_equivalence(ctx, items=items, solo=solo, out_root=root / name, rounds=2,
                                          together=lambda its, out, gargs=gargs, spec=spec: self.cand("kinfu_replay", [*gargs, "--concurrent", spec], out),
                                          item_dir=lambda d, k: d / f"case{k}", compare=lambda a, b: c.compare_bitwise_run(a, b))
            detail[name] = {"pass": v.passed, **v.detail}
            ok &= v.passed
        mism = []
        if self.scope["people_threads"] and self.people_solo is None:
            mism.append({"error": "no solo people run"})
        for rnd in range(3 if self.people_solo and self.scope["people_threads"] else 0):
            d = root / f"people_r{rnd}"
            rr = self.cand("people_eval", self.people_args(c.PEOPLE_VARIANTS, ["--threads", str(c.PEOPLE_THREADS)]), d, omp=1)
            self.t.seal(d)
            if rr.rc != 0:
                mism.append({"round": rnd, "error": "run failed", "log": tail(rr.log, 15)})
                continue
            for th in range(c.PEOPLE_THREADS):
                f = c.compare_people(d / f"thread{th}", self.people_solo, c.PEOPLE_VARIANTS)
                if f:
                    mism.append({"round": rnd, "thread": th, "failures": f[:5]})
        detail["people_threads"] = {"pass": not mism, "mismatches": mism[:12]}
        return Verdict(ok and not mism, detail)

    def fresh(self, ctx):
        c = self.cases
        seed = D.draw_seed(ctx)
        fails = {}
        for name, (tool, args) in [x for x in c.fresh_workloads(seed).items() if self.scope["fresh"](x[0])]:
            omp = 16 if tool == "kinfu_replay" else 1
            run_args = self.people_args(args) if tool == "people_eval" else args
            ref = self.refs.get(f"fresh/{name}", (tool, run_args, omp))
            out = self.work / "cand_fresh" / name
            rr = self.cand(tool, run_args, out, omp=omp)
            self.t.seal(out)
            if rr.rc != 0:
                f = [{"error": "run failed", "log": tail(rr.log, 15)}]
            elif tool == "kinfu_replay":
                f = c.compare_kinfu(out, ref)
            elif tool == "people_eval":
                f = c.compare_people(out, ref, args)
            else:
                f = c.compare_ncv(out, ref)
            if f:
                fails[name] = f[:12]
        return Verdict(not fails, {"seed": seed, "failures": fails})

    def sanitizer(self, ctx):
        c = self.cases
        detail, ok = {}, True
        for tool in ("racecheck", "synccheck", "memcheck"):
            for name, binary, args in [x for x in c.SANITIZE if self.scope["sanitize"](x[0])]:
                if binary == "people_eval":
                    args = self.people_args(args[1].split(","))
                wrap = [f"{CUDA}/bin/compute-sanitizer", "--tool", tool, "--print-limit", "30", "--error-exitcode", "97"]
                if tool == "memcheck":
                    wrap += ["--leak-check", "no"]
                out = self.work / "san" / f"{tool}_{name}"
                rr = self.cand(binary, args, out, omp=1, timeout=5400, wrap=wrap)
                self.t.seal(out)
                clean, lines = sanitizer_clean(tool, rr.rc, rr.log)
                produced = (out / "result.json").exists()
                hazards = [ln for ln in rr.log.splitlines() if re.search(r"(Error|Warning|Hazard|Invalid|Race)", ln)][:15]
                detail[f"{tool}/{name}"] = {"pass": clean and produced, "returncode": rr.rc, "summary": lines, "first_reports": hazards}
                ok &= clean and produced
        return Verdict(ok, detail)

    # ------------------------------------------------------------------ perf
    def _tp_refs(self):
        c = self.cases
        spec = c.THROUGHPUT_KINFU[-3].split(",")
        gargs = [a for a in c.THROUGHPUT_KINFU if a not in ("--concurrent", c.THROUGHPUT_KINFU[-3], "--mode", "kinfu")]
        out = []
        for k, item in enumerate(spec):
            mode, traj, noise = item.split(":")
            out.append(self.refs.get(f"tp_solo/case{k}", ("kinfu_replay", [*gargs, "--mode", mode, "--traj", traj, "--noise", noise], 16)))
        return out

    def perf_round(self, rnd):
        c = self.cases
        P = self.t.private(self.work / "ref" / "perf")
        C = self.t.udir(self.work / "cand_perf")
        res = {"original": {}, "candidate": {}, "wall": {"original": 0.0, "candidate": 0.0}}
        work = [("kinfu", "kinfu_replay", c.KINFU["kinfu_orbit_clean"], 16),
                ("kinfuls", "kinfu_replay", c.KINFU["kinfuls_walk_clean"] + ["--skip-mesh"], 16),
                ("people", "people_eval", self.people_args(c.PEOPLE_VARIANTS[:5], ["--repeat", "15"]), 1),
                ("kinfu4", "kinfu_replay", c.THROUGHPUT_KINFU, 16)]
        for tag, tool, args, omp in [w for w in work if w[0] in self.scope["perf"]]:
            self.t.kill()
            r = P / f"{tag}{rnd}"
            env = {"LD_LIBRARY_PATH": f"{self.legacy}/lib", "OMP_NUM_THREADS": str(omp), "CUDA_CACHE_PATH": str(self.work / "ref" / ".nv"),
                   "CUDA_CACHE_MAXSIZE": "4294967296"}
            self.t.private(r)
            ro = self.t.run([self.legacy / "bin" / tool, *args, "--out", r], cwd=self.data, env=env, timeout=3600)
            if ro.rc != 0:
                raise InfraError(f"original benchmark {tag} failed: {tail(ro.log, 10)}")
            cdir = C / f"{tag}{rnd}"
            rc_ = self.cand(tool, args, cdir, omp=omp)
            self.t.seal(cdir)
            if rc_.rc != 0:
                raise CandidateError(f"candidate benchmark {tag} failed: {tail(rc_.log, 10)}")
            res["wall"]["original"] += ro.wall_s
            res["wall"]["candidate"] += rc_.wall_s
            if tag == "kinfu4":
                n = len(c.THROUGHPUT_KINFU[-3].split(","))
                for k in range(n):
                    f = c.compare_kinfu(cdir / f"case{k}", self._tp_refs()[k])
                    if f:
                        raise CandidateError(f"candidate concurrent benchmark case {k} computed different results: {f[:3]}")
                for side, d in (("original", r), ("candidate", cdir)):
                    res[side]["kinfu_concurrent_wall_ms"] = max(float(c.load(d / f"case{k}")["track_wall_ms"]) for k in range(n))
                continue
            if tag == "people":
                f = c.compare_people(cdir, r, c.PEOPLE_VARIANTS[:5])
                if f:
                    raise CandidateError(f"candidate benchmark {tag} computed different results: {f[:3]}")
                metrics = {"people_ms": lambda j: sum(v["ms_median"] for v in j.values())}
            else:
                f = c.compare_kinfu(cdir, r)
                if f:
                    raise CandidateError(f"candidate benchmark {tag} computed different results: {f[:3]}")
                metrics = {f"{tag}_frame_ms": lambda j: j["frame_ms_median"]}
                if tag == "kinfu":
                    metrics.update({"marching_cubes_ms": lambda j: j["mc_ms"], "cloud_ms": lambda j: j["cloud_ms"]})
            for side, d in (("original", r), ("candidate", cdir)):
                j = c.load(d)
                for m, fn in metrics.items():
                    res[side][m] = float(fn(j))
        return res

    def perf(self, ctx):
        p = ctx.params
        guard = None if p.get("no_load_guard") else LoadGuard(max_load_per_core=float(p.get("max_load_per_core", 0.5)),
                                                                 max_gpu_util=int(p.get("max_gpu_util", 20)), wait_s=float(p.get("load_wait_s", 1800)))
        metrics = {m for tag in self.scope["perf"] for m in PERF_METRICS[tag]}
        if p.get("perf_mode") == "record":  # deep variant: no ceiling, the oracle is the best known, faster pays more
            from crucible.verify.runtime.patterns.record import record as record_verdict

            # the whole-process wall clock binds record mode too: its per-metric numbers come from JSON the candidate wrote
            measured = PerfBars({m: 10.0 for m in metrics}, self.perf_round, rounds=int(p.get("perf_rounds", 3)), remeasure=0,
                                spread_max=p.get("spread_max", 0.25), wall_ratio_max=p.get("wall_ratio_max", 1.0), guard=guard)(ctx)
            if measured.infra:
                return measured
            bars = {m: b for m, b in self.thresholds["max_ratio"].items() if m in metrics}
            rs = record_score(measured.detail.get("metrics", {}), bars, {**BEST_KNOWN, **(p.get("best_known") or {})}, floor=p.get("record_floor", "all_bars"))
            v = record_verdict(rs["score"], RecordSpec(baseline=0.0, best_known=1.0), valid=rs["valid"], detail=rs)
            v.detail["normalized_reward"] = rs["score"] if rs["valid"] else 0.0
            return v
        bars = PerfBars({m: b for m, b in self.thresholds["max_ratio"].items() if m in metrics}, self.perf_round, rounds=int(p.get("perf_rounds", 3)), remeasure=1,
                        spread_max=p.get("spread_max", 0.25), wall_ratio_max=p.get("wall_ratio_max", 1.0), guard=guard)
        return bars(ctx)


def plan(ctx: Context) -> Plan:
    g = Task04(ctx)
    sc = g.scope
    measured = {"static", "build", "binary", "consumers", "sanitizer", "fresh"}
    measured |= {"kinfu"} if any(sc["kinfu"](n) for n in g.cases.KINFU) else set()
    measured |= {"people"} if sc["people"] or sc["people_threads"] else set()
    measured |= {"ncv"} if sc["ncv"] else set()
    measured |= {"concurrency"} if sc["people_threads"] or any(sc["conc"](x[0]) for x in g.cases.CONCURRENT) else set()
    measured |= {"perf"} if sc["perf"] else set()
    gates = [x for x in GATES if x in measured]
    correctness = tuple(x for x in gates if x != "perf")
    after_build = ("build",)
    checks = [
        Check("static", "static", g.static, tier="fast", provenance="instruction:no texture/surface references"),
        Check("build", "build", g.build, tier="fast", provenance="instruction:CUDA 12.8 sm_120 build"),
        Check("binary", "binary", g.binary, tier="fast", requires=after_build, provenance="instruction:sm_120 device code only"),
        Check("consumers", "consumers", g.consumers, tier="fast", requires=after_build, provenance="reference_differential:pristine consumers"),
        Check("kinfu", "kinfu", g.kinfu, requires=("consumers",), provenance="reference_differential:original build", cost="expensive"),
        Check("people", "people", g.people, tier="fast", requires=("consumers",), provenance="reference_differential:original build"),
        Check("ncv", "ncv", g.ncv, tier="fast", requires=("consumers",), provenance="reference_differential;incident:B.4 NCV guard bands"),
        Check("concurrency", "concurrency", g.concurrency, requires=("consumers",) + (("people",) if "people" in gates else ()),
              provenance="incident:B.3 compare to solo", cost="expensive"),
        Check("fresh", "fresh", g.fresh, requires=("consumers",), provenance="incident:B.2 fresh draws", cost="expensive"),
        Check("sanitizer", "sanitizer", g.sanitizer, requires=("consumers",), provenance="incident:B.4 memcheck exact allocation", cost="expensive"),
        Check("perf", "perf", g.perf, requires=correctness, provenance="calibration:B.1 v3 speedup bars", cost="expensive"),
    ]
    kinds = dict(KINDS)
    if ctx.params.get("perf_mode") == "record":
        kinds["perf"] = "score"
    return Plan(TASK if g.scope_name == "full" else f"{TASK}:{g.scope_name}", [Gate(x, kinds[x]) for x in gates], [c for c in checks if c.gate in gates],
                setup=g.setup)
