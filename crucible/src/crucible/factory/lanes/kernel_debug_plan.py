"""Kernel debugging lane grader: a kit plan run as root inside the verifier image on a GPU host (swarm-pipeline/10 §4).
Standard library + the kit runtime + ``kernel_harness`` and ``kernel_plan`` (bundled next to it).

    python3 -I /crucible_verify/crucible/verify/runtime/run.py --plan crucible.factory.lanes.kernel_debug_plan \
        --out /out --base /k/base --patch /k/patch --param spec=/k/kdspec.json --param ref=/k/ref --param hooks_dir=/k/hooks \
        --param arch=90 --require-root

**What is built.** The grader assembles its own tree: the start tree (A') with the candidate's files under the kept
paths (the CUDA backend) and nothing else of the candidate's patch, plus B's op test patched by ``kernel_harness``.
It configures and builds ``test-backend-ops`` with its own CMake flags (``kernel_harness.cmake_cuda``: SASS for
``sm_<arch>`` only) as an unprivileged uid; the outputs are sealed.

**The reference** (``/k/ref``, root-only, built once per GPU host by the bundle runner): B with the same flags and
harness. It runs as root on the same cases and seeds; its dumps hold the CPU reference (B's CPU backend, which A' shares)
and B's GPU output.

**Checks per milestone** (``spec.milestones[*].checks``):

- ``golden``: every calibrated case, on two fixed seeds and one seed drawn now (revealed after the candidate ran), and
  ``repeat`` more fresh seeds for nondeterministic faults; each case's GPU output must match the reference build's CPU
  output within the case's own tolerance (``kernel_harness.compare_dumps``, outside the candidate's process); the case
  must run (an op the backend no longer supports fails) and the candidate process must launch kernels (CUPTI injection:
  at least one launch per case, no host fallback);
- ``racecheck`` / ``synccheck`` / ``memcheck``: compute-sanitizer on the milestone's first cases; every summary zero;
- ``perf``: the grader's wall clock around whole harness runs in perf mode at R and 3R repetitions (start-up cancels),
  alternating reference and candidate, best of 3 rounds plus one re-measure, a load guard, a spread check (noise ->
  infra), launches must scale with repetitions; the bar is per GPU model (``spec.perf_bars``), calibrated by the bundle
  with a measured noise floor; no bar for this GPU -> infra ``uncalibrated``.

A milestone with no calibrated cases is infra (``uncalibrated``): the task is not gradable until the GPU calibration
ran. ``--param mode=pool`` grades every case of each milestone's pool instead and reports per-case results (the
calibration's input: which cases each single revert breaks). Never trusted: exit codes, console lines, the candidate's
build files, timing inside its process.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from crucible.factory.lanes import kernel_harness as kh
from crucible.factory.lanes import kernel_plan as kp
from crucible.verify.runtime.patterns.compare import safe_read
from crucible.verify.runtime.patterns.differential import draw_seed
from crucible.verify.runtime.patterns.perf import LoadGuard, PerfBars
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import CandidateError, InfraError

CUDA = os.environ.get("CRUCIBLE_CUDA_HOME", "/usr/local/cuda")
SANITIZER = f"{CUDA}/bin/compute-sanitizer"
POOL_CAP = 1000
CHUNK = 96  # cases per harness run: dumps hold every output twice (GPU and CPU), a whole pool would be gigabytes
MAX_DUMP = 200 << 20  # a reference chunk above this is split in two (the candidate's dump stays bounded by safe_read)


class _DumpTooBig(Exception):
    pass


def pool_sample(pool: list[str], first: list[str], cap: int = POOL_CAP) -> list[str]:
    """The calibration's cases: the fixes' own added cases first, then the pool, evenly strided down to ``cap`` (a
    flash-attention pool holds ~10k cases; a stride keeps every shape family represented)."""
    rest = [c for c in pool if c not in set(first)]
    room = max(0, cap - len(first))
    if len(rest) > room:
        step = len(rest) / room if room else len(rest) + 1
        rest = [rest[int(i * step)] for i in range(room)]
    return [*first, *rest][:cap]


class KDGrader:
    def __init__(self, ctx: Context, spec: dict):
        self.ctx, self.spec, self.t = ctx, spec, ctx.trusted
        self.arch = str(ctx.params.get("arch") or os.environ.get("CRUCIBLE_CUDA_ARCH") or spec.get("arch_default") or "90")
        self.ref = Path(ctx.params.get("ref") or "/k/ref")
        self.base = Path(ctx.params.get("base_tree") or "/k/base")
        self.hooks = Path(ctx.params.get("hooks_dir") or "/k/hooks")
        self.mode = str(ctx.params.get("mode") or "grade")
        self.work = ctx.work
        self.exe: Path | None = None
        self.lib_dir: Path | None = None
        self.refs: dict[tuple, list] = {}
        self.fresh: int | None = None
        self.gpu = kp.gpu_name()
        self.pool_seen: dict[int, dict[str, dict]] = {}  # pool mode: seed -> case -> comparison (milestones share ops)

    # -- the grader's tree and build
    def assemble(self) -> Path:
        src = self.work / "kd_src"
        if src.exists():
            shutil.rmtree(src)
        shutil.copytree(self.base, src, symlinks=True, ignore=shutil.ignore_patterns(".git"))
        cand = self.ctx.candidate
        for keep in self.spec["keep_paths"]:
            dst, srcp = src / keep, cand / keep
            if dst.is_dir():
                shutil.rmtree(dst)
            elif dst.exists():
                dst.unlink()
            if srcp.is_dir():
                shutil.copytree(srcp, dst, symlinks=False)
            elif srcp.is_file():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(srcp, dst)
        harness = self.ref / "harness.cpp"
        if not harness.is_file():
            raise InfraError("the reference bundle has no patched harness")
        kh.prepare_tree(src, harness.read_text(errors="replace"))
        self.t.give(src)
        return src

    def build(self) -> Verdict:
        src = self.assemble()
        bld = self.t.udir(self.work / "kd_build")
        jobs = str(max(2, min(32, (os.cpu_count() or 4))))
        nvcc = Path(CUDA) / "bin" / "nvcc"
        if not nvcc.is_file():  # the grader's toolkit, not the candidate's code: never a 0
            raise InfraError(f"no CUDA compiler at {nvcc}")
        # the compiler is named: the trusted runner's PATH has no toolkit directory, and CMake would score its absence
        # static libraries: sealing removes the build's symlinks, so shared libraries' SONAME links would be gone
        flags, env = [*kh.cmake_cuda([self.arch]), f"-DCMAKE_CUDA_COMPILER={nvcc}", "-DBUILD_SHARED_LIBS=OFF", *(self.spec.get("cmake_extra") or [])], {}
        if self.ctx.params.get("ccache"):  # the bundle's gate runs (the grader's own patches) only, never a task grade
            flags += kh.cmake_ccache()
            env = {"CCACHE_DIR": str(self.ctx.params["ccache"]), "CCACHE_BASEDIR": str(self.work), "CCACHE_NOHASHDIR": "1", "CCACHE_UMASK": "000",
                   "CCACHE_MAXSIZE": "30G"}
        cfg = self.t.run(["cmake", "-S", src, "-B", bld, "-G", "Ninja", *flags], cwd=bld, timeout=1800, candidate=True, env=env)
        if cfg.rc != 0:
            self.t.seal(bld)
            return Verdict(False, {"stage": "configure", "rc": cfg.rc, "log": cfg.log[-2000:]})
        r = self.t.run(["cmake", "--build", bld, "--target", kh.TARGET, "-j", jobs], cwd=bld, timeout=5400, candidate=True, env=env)
        self.t.seal(bld)
        exe = bld / "bin" / kh.TARGET
        ok = r.rc == 0 and not r.timed_out and exe.is_file()
        errs = [ln for ln in r.log.splitlines() if re.search(r"\berror\b", ln)][:20]
        if ok:
            self.exe, self.lib_dir = exe, bld / "bin"
        return Verdict(ok, {"stage": "build", "rc": r.rc, "errors": errs, "log": "" if ok else r.log[-2000:]})

    def binary(self) -> Verdict:
        libs = sorted(p for p in (self.lib_dir or Path("/nonexistent")).glob("libggml-cuda*") if p.is_file())
        targets = libs or ([self.exe] if self.exe else [])
        sms, ptxs = set(), set()
        for f in targets:
            elf = subprocess.run([f"{CUDA}/bin/cuobjdump", "--list-elf", str(f)], capture_output=True, text=True, check=False).stdout
            ptx = subprocess.run([f"{CUDA}/bin/cuobjdump", "--list-ptx", str(f)], capture_output=True, text=True, check=False).stdout
            sms |= set(re.findall(r"\.(sm_\w+)\.cubin", elf))
            ptxs |= set(re.findall(r"\.((?:sm|compute)_\w+)\.ptx", ptx))
        bad = [a for a in sorted(sms) if not re.fullmatch(rf"sm_{self.arch}[af]?", a)] + [f"ptx:{a}" for a in sorted(ptxs)]
        if not sms:
            bad.append("no device code")
        return Verdict(not bad, {"sass": sorted(sms), "ptx": sorted(ptxs), "issues": bad, "files": [p.name for p in targets]})

    # -- runs
    def _env(self, seed: int, cases_file: Path, dump: Path | None, *, reps: int = 0) -> dict:
        env = {"CRUCIBLE_SEED": str(seed), "CRUCIBLE_CASES": str(cases_file), "GGML_CUDA_DISABLE_GRAPHS": "1", "CUDA_DISABLE_PTX_JIT": "1"}
        if dump is not None:
            env["CRUCIBLE_DUMP"] = str(dump)
        if reps:
            env["CRUCIBLE_REPS"] = str(reps)
        return env

    def _cases_file(self, d: Path, cases: list[str]) -> Path:
        f = d / "cases.txt"
        f.write_text(kh.cases_file(cases))
        os.chmod(f, 0o644)
        return f

    def run_ref(self, tag: str, cases: list[str], seed: int, *, mode: str = "test", reps: int = 0, dump: bool = True, cache: bool = True) -> dict:
        key = (tag, tuple(cases), seed, mode, reps)
        if cache and key in self.refs:
            return self.refs[key]
        d = self.t.private(self.work / "ref" / f"{tag}-{seed}-{mode}-{reps}")
        cf = self._cases_file(d, cases)
        dump_f = d / "ref.dump" if dump else None
        # the reference's libraries sit next to it (its build directory, which the RPATH names, is gone)
        env = {**self._env(seed, cf, dump_f, reps=reps), "LD_LIBRARY_PATH": str(self.ref / "bin")}
        r = self.t.run([self.ref / "bin" / kh.TARGET, mode, "-b", "CUDA0"], cwd=d, env=env, timeout=3600)
        if r.timed_out:
            raise InfraError(f"the reference timed out on {tag}")
        if dump_f and dump_f.exists() and dump_f.stat().st_size > MAX_DUMP:
            dump_f.unlink()
            raise _DumpTooBig(tag)
        recs = kh.read_dump(safe_read(dump_f)) if dump_f and dump_f.exists() else []
        if dump_f:
            dump_f.unlink(missing_ok=True)
        res = {"recs": recs, "wall_s": r.wall_s, "status": kh.case_status(r.stdout), "rc": r.rc, "crashed": kh.crashed_case(r.stdout) if r.rc else None}
        if dump and cases and not recs and not res["crashed"]:
            raise InfraError(f"the reference produced no output for {tag} (rc {r.rc}): {r.log[-300:]}")
        if cache:
            self.refs[key] = res
        return res

    def run_cand(self, tag: str, cases: list[str], seed: int, *, mode: str = "test", reps: int = 0, dump: bool = True, wrap: list | None = None,
                 launches: bool = True, timeout: float = 3600) -> tuple[dict, object]:
        d = self.t.udir(self.work / "cand" / f"{tag}-{seed}-{mode}-{reps}-{len(wrap or [])}")
        cf = self._cases_file(d, cases)
        dump_f = d / "cand.dump" if dump else None
        env = {**self._env(seed, cf, dump_f, reps=reps), "LD_LIBRARY_PATH": str(self.lib_dir)}
        if launches:
            env["CUDA_INJECTION64_PATH"] = str(self.hooks / "libcrucible_launch.so")
            env["CRUCIBLE_LAUNCH_LOG"] = str(d / "launches.tsv")
        r = self.t.run([*(wrap or []), self.exe, mode, "-b", "CUDA0"], cwd=d, env=env, timeout=timeout, candidate=True)
        self.t.seal(d)
        recs = kh.read_dump(safe_read(dump_f)) if dump_f and dump_f.exists() else []
        if dump_f:
            dump_f.unlink(missing_ok=True)
        n_launch = sum(kp.read_launches(d / "launches.tsv").values()) if launches else None
        return {"recs": recs, "wall_s": r.wall_s, "status": kh.case_status(r.stdout), "launches": n_launch}, r

    # -- checks
    def cases_of(self, m: dict) -> list[str]:
        if self.mode == "pool":
            return pool_sample(list(m.get("case_pool") or []), list(m.get("cases_added") or []))
        cases = list(m.get("cases") or [])
        if not cases:
            raise InfraError(f"milestone {m['id']} has no calibrated cases (run the GPU calibration first)")
        return cases

    def seeds(self, m: dict) -> list[int]:
        if self.fresh is None:
            self.fresh = draw_seed(self.ctx) & 0x7FFFFFFF
        fixed = list((self.spec.get("seeds") or {}).get("fixed") or [7])
        extra = [(self.fresh * 2654435761 + k) & 0x7FFFFFFF for k in range(1, int(m.get("repeat") or 1))]
        return [*fixed, self.fresh, *extra]

    def _pair(self, tag: str, cases: list[str], seed: int) -> dict:
        """The reference, then the candidate, on one chunk of cases (outputs compared outside the candidate's process and
        dropped right after); a chunk whose reference output is too large is halved."""
        try:
            ref = self.run_ref(tag, cases, seed, cache=False)
        except _DumpTooBig:
            if len(cases) == 1:
                raise InfraError(f"one case's reference output is over {MAX_DUMP} bytes: {cases[0]}") from None
            h = len(cases) // 2
            a, b = self._pair(f"{tag}a", cases[:h], seed), self._pair(f"{tag}b", cases[h:], seed)
            return {"cmp": {**a["cmp"], **b["cmp"]}, "launches": None if None in (a["launches"], b["launches"]) else a["launches"] + b["launches"],
                    "rc": a["rc"] + b["rc"], "timed_out": a["timed_out"] or b["timed_out"]}
        if ref["crashed"] in cases:  # the reference died on one case: that case leaves (no reference), the rest run again
            done = {x["case"] for x in ref["recs"]}
            rest = [c for c in cases if c not in done and c != ref["crashed"]]
            head = [c for c in cases if c in done]
            parts = [p for p in (self._pair(f"{tag}h", head, seed) if head else None, self._pair(f"{tag}r", rest, seed) if rest else None) if p]
            cmp = {k: v for p in parts for k, v in p["cmp"].items()}
            cmp[ref["crashed"]] = {"pass": False, "infra": "the reference crashed on this case"}
            return {"cmp": cmp, "launches": None if any(p["launches"] is None for p in parts) else sum(p["launches"] for p in parts),
                    "rc": [rc for p in parts for rc in p["rc"]], "timed_out": any(p["timed_out"] for p in parts)}
        cand, r = self.run_cand(tag, cases, seed)
        return {"cmp": kh.compare_dumps(cand["recs"], ref["recs"], cases=cases), "launches": cand["launches"], "rc": [r.rc], "timed_out": r.timed_out}

    def golden(self, m: dict) -> Verdict:
        cases = self.cases_of(m)
        detail, ok, per_case = {}, True, {}
        for seed in self.seeds(m):
            seen = self.pool_seen.setdefault(seed, {}) if self.mode == "pool" else {}
            todo = [c for c in cases if c not in seen]  # pool mode: a case another milestone ran on this seed is not rerun
            parts = [self._pair(f"{m['id']}-c{i}", todo[i:i + CHUNK], seed) for i in range(0, len(todo), CHUNK)]
            cmp = {k: v for p in parts for k, v in p["cmp"].items()}
            if self.mode == "pool":
                seen.update(cmp)
                cmp = {c: seen[c] for c in cases if c in seen}
            launches = None if any(p["launches"] is None for p in parts) else sum(p["launches"] for p in parts)
            rcs, timed_out = [rc for p in parts for rc in p["rc"]], any(p["timed_out"] for p in parts)
            if self.mode == "pool":  # calibration: a case B's own backend skips (not supported) leaves the pool, it is no evidence
                skipped = [k for k, v in cmp.items() if v.get("infra")]
                cmp = {k: v for k, v in cmp.items() if not v.get("infra")}
                detail["reference_skipped"] = detail.get("reference_skipped", 0) + len(skipped)
            if any(v.get("infra") for v in cmp.values()):
                raise InfraError(f"reference missing cases for {m['id']}: {[k for k, v in cmp.items() if v.get('infra')][:3]}")
            fails = {k: v for k, v in cmp.items() if not v["pass"]}
            for k, v in cmp.items():
                per_case.setdefault(k, []).append(bool(v["pass"]))
            launch_ok = launches is None or launches >= max(1, len(cases) - len(fails))
            detail[f"seed{'_fresh' if seed == self.fresh else ''}_{seed}"] = {"failed": len(fails), "cases": len(cases), "launches": launches,
                                                                               "rc": sorted(set(rcs)), "examples": dict(list(fails.items())[:4])}
            ok &= not fails and launch_ok and not timed_out
            if not launch_ok:
                detail["launch_error"] = "fewer kernel launches than cases: an op ran off the GPU"
        detail["per_case"] = {k: all(v) for k, v in per_case.items()}  # the calibration's input (pool) and its stability check (grade)
        return Verdict(ok, {**detail, "seed_revealed_after_run": True})

    def sanitize(self, m: dict, tool: str) -> Verdict:
        cases = self.cases_of(m)[:4]
        wrap = [SANITIZER, "--tool", tool, "--error-exitcode", "97", "--print-limit", "30", *(["--leak-check", "no"] if tool == "memcheck" else [])]
        _c, r = self.run_cand(f"{m['id']}-{tool}", cases, 7, dump=False, wrap=wrap, launches=False, timeout=7200)
        clean, summary = kp.sanitizer_clean(r.rc, r.log)
        if not summary and "COMPUTE-SANITIZER" not in r.log:  # the tool never started (the grader's toolkit): not the candidate's fault
            raise InfraError(f"compute-sanitizer did not start (rc {r.rc}): {r.log[-300:]}")
        return Verdict(clean, {"tool": tool, "rc": r.rc, "summary": summary, "cases": len(cases), "log": "" if clean else r.log[-1500:]})

    def perf(self, m: dict) -> Verdict:
        bars = (self.spec.get("perf_bars") or {}).get(self.gpu) or {}
        key = f"{m['id']}_ms"
        if key not in bars:
            return Verdict(False, {"gpu": self.gpu}, infra=f"no calibrated perf bar for {m['id']} on {self.gpu!r}")
        cases = list(m.get("perf_cases") or [])
        if not cases:
            return Verdict(False, {}, infra=f"milestone {m['id']} has no calibrated perf cases")
        r1 = int((self.spec.get("perf") or {}).get("r") or 3)

        def round_fn(rnd: int) -> dict:
            orig, cand = {}, {}
            for reps in (r1, 3 * r1):
                o = self.run_ref(f"perf{rnd}", cases, 7, mode="perf", reps=reps, dump=False)
                c, rr = self.run_cand(f"perf{rnd}", cases, 7, mode="perf", reps=reps, dump=False)
                if rr.rc != 0 or rr.timed_out:
                    raise CandidateError(f"timed run of {m['id']} failed (rc {rr.rc})")
                orig[reps], cand[reps] = o["wall_s"], (c["wall_s"], c["launches"] or 0)
            if cand[3 * r1][1] - cand[r1][1] < 2 * r1 * len(cases):
                raise CandidateError(f"{m['id']}: kernel launches do not scale with repetitions (skipped or memoized work)")
            per = lambda a, b: max(1e-6, (b - a) / (2 * r1)) * 1e3
            return {"original": {key: per(orig[r1], orig[3 * r1])}, "candidate": {key: per(cand[r1][0], cand[3 * r1][0])}}

        pb = PerfBars(bars={key: float(bars[key])}, round_fn=round_fn, rounds=3, remeasure=1, spread_max=0.25,
                      guard=LoadGuard(max_load_per_core=0.5, max_gpu_util=20, wait_s=1200))
        v = pb()
        v.detail["gpu"] = self.gpu
        return v


def plan(ctx: Context) -> Plan:
    spec = json.loads(Path(ctx.params["spec"]).read_text())
    g = KDGrader(ctx, spec)
    if g.gpu == "unknown":
        def no_gpu(_ctx):
            return Verdict(False, {}, infra="no GPU visible (nvidia-smi): the grade is masked, never scored")
        gates = [Gate("gpu", "gate", "a GPU is visible")]
        return Plan(spec["task_id"], gates, [Check("gpu:visible", "gpu", no_gpu)])
    gates = [Gate("build", "gate", "the grader builds the candidate's CUDA backend for this GPU")]
    gates += [Gate(m["id"], "outcome", f"{m['kind']} milestone {m['id']}") for m in spec["milestones"]]
    skipped = Verdict(False, {"skipped": "build failed"})
    checks = [Check("build:cuda", "build", lambda ctx: g.build()), Check("build:binary", "build", lambda ctx: g.binary() if g.exe else skipped)]
    for m in spec["milestones"]:
        for c in m["checks"]:
            if c == "golden":
                fn = (lambda ctx, m=m: g.golden(m))
            elif c in ("racecheck", "synccheck", "memcheck"):
                fn = (lambda ctx, m=m, c=c: g.sanitize(m, c))
            elif c == "perf":
                fn = (lambda ctx, m=m: g.perf(m))
            else:
                continue
            checks.append(Check(f"{m['id']}:{c}", m["id"], fn, requires=("build",)))
    only = [c for c in str(ctx.params.get("only") or "").split(",") if c]
    if only:  # calibration: the build plus the named checks (``m1:perf``); a grade with ``only`` is never a task grade
        checks = [c for c in checks if c.gate == "build" or c.id in only]
        keep = {"build"} | {c.gate for c in checks}
        gates = [gt for gt in gates if gt.id in keep]
    return Plan(spec["task_id"], gates, checks)
