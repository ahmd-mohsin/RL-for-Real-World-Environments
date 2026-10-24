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
- ``racecheck`` / ``synccheck`` / ``memcheck``: compute-sanitizer on the milestone's ``sanitizer_cases`` (calibrated: cases
  whose processes launch the fault's kernels) or else its first cases; every summary zero;
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


def pool_cap(v: object) -> int:
    """The ``pool_cap`` parameter: absent -> ``POOL_CAP``; ``0`` -> every case. The runtime JSON-decodes parameter values,
    so ``pool_cap=0`` arrives as the integer 0 (falsy): a plain ``or POOL_CAP`` turned the S1 sweep "of every pool case"
    into a 1000-case stride (found 2026-10-22: 14 starts declared "not real" had been swept on 10-20% of their pools)."""
    return POOL_CAP if v is None or v == "" else int(v)


def case_ratio(v: dict) -> float:
    """A compared case's worst error as a share of its tolerance (``nmse_vs_cpu / max_err``; a missing output is inf):
    the calibration's margin check reads it (a case B passes at 0.9 of its tolerance fails on some other seed)."""
    try:
        return float(v["nmse_vs_cpu"]) / float(v["max_err"]) if float(v["max_err"]) > 0 else float("inf")
    except (KeyError, TypeError, ValueError):
        return float("inf")


def perf_groups(key: str, cases: list[str], each: bool) -> dict[str, list[str]]:
    """The perf check's metrics: one over all the milestone's cases (a grade), or one per case (``perf_each``, the
    calibration: every candidate case timed in one grade instead of one grade per case)."""
    return {f"{key}#{i}": [c] for i, c in enumerate(cases)} if each else {key: list(cases)}


class KDGrader:
    def __init__(self, ctx: Context, spec: dict):
        self.ctx, self.spec, self.t = ctx, spec, ctx.trusted
        self.arch = str(ctx.params.get("arch") or os.environ.get("CRUCIBLE_CUDA_ARCH") or spec.get("arch_default") or "90")
        self.ref = Path(ctx.params.get("ref") or "/k/ref")
        self.base = Path(ctx.params.get("base_tree") or "/k/base")
        self.hooks = Path(ctx.params.get("hooks_dir") or "/k/hooks")
        self.mode = str(ctx.params.get("mode") or "grade")
        self.pool_cap = pool_cap(ctx.params.get("pool_cap"))  # pool mode: 0 = every case of the pool (the S1 sweep of the start)
        self.pool_seeds = str(ctx.params.get("pool_seeds") or "all")  # pool mode: "fixed" = the spec's fixed seeds only
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
                   "CCACHE_MAXSIZE": "12G"}  # one shared gate cache on a shared, nearly full disk
        elif self.ctx.params.get("ccache_ro"):  # an episode grade (swarm-pipeline/10 §9): the gate runs' cache, mounted read-only
            # and opened read-only, so a candidate's build reuses the objects of unchanged units and can neither write nor
            # poison the cache; every unit the candidate changed is compiled from its own source
            flags += kh.cmake_ccache()
            env = {"CCACHE_DIR": str(self.ctx.params["ccache_ro"]), "CCACHE_BASEDIR": str(self.work), "CCACHE_NOHASHDIR": "1", "CCACHE_READONLY": "1",
                   "CCACHE_TEMPDIR": "/tmp/crucible-ccache-tmp"}
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

    def exact_cases(self, cases: list[str]) -> list[str]:
        """The cases of ``cases`` whose CPU column the reference replaces with the exact attention computation: those the
        calibration showed the CPU kernel wrong on (``spec.fa_exact_cases``; the evidence is in the calibration record)."""
        listed = set(self.spec.get("fa_exact_cases") or [])
        return [c for c in cases if c in listed]

    def run_ref(self, tag: str, cases: list[str], seed: int, *, mode: str = "test", reps: int = 0, dump: bool = True, cache: bool = True,
                exact: list[str] | None = None) -> dict:
        exact = self.exact_cases(cases) if exact is None else exact
        key = (tag, tuple(cases), seed, mode, reps, tuple(exact))
        if cache and key in self.refs:
            return self.refs[key]
        d = self.t.private(self.work / "ref" / f"{tag}-{seed}-{mode}-{reps}{'-x' if exact else ''}")
        cf = self._cases_file(d, cases)
        dump_f = d / "ref.dump" if dump else None
        # the reference's libraries sit next to it (its build directory, which the RPATH names, is gone)
        env = {**self._env(seed, cf, dump_f, reps=reps), "LD_LIBRARY_PATH": str(self.ref / "bin")}
        if exact:
            xf = d / "fa_exact.txt"
            xf.write_text(kh.cases_file(exact))
            env[kh.FA_REF_ENV] = str(xf)
        r = self.t.run([self.ref / "bin" / kh.TARGET, mode, "-b", "CUDA0"], cwd=d, env=env, timeout=3600)
        if r.timed_out:
            raise InfraError(f"the reference timed out on {tag}")
        if dump_f and dump_f.exists() and dump_f.stat().st_size > MAX_DUMP:
            dump_f.unlink()
            raise _DumpTooBig(tag)
        recs = kh.read_dump(safe_read(dump_f)) if dump_f and dump_f.exists() else []
        if dump_f:
            dump_f.unlink(missing_ok=True)
        res = {"recs": recs, "wall_s": r.wall_s, "status": kh.case_status(r.stdout), "rc": r.rc, "crashed": kh.crashed_case(r.stdout) if r.rc else None,
               "exact": sorted(set(kh.EXACT_LINE.findall(r.log)))}
        # pool mode: a chunk whose cases this backend does not support at all leaves no output, and those cases leave the pool
        if dump and cases and not recs and not res["crashed"] and not (self.mode == "pool" and res["status"] and set(res["status"].values()) <= {"not supported"}):
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
            pool = list(m.get("case_pool") or [])
            return pool_sample(pool, list(m.get("cases_added") or []), cap=self.pool_cap or len(pool) + len(m.get("cases_added") or []))
        cases = list(m.get("cases") or [])
        if not cases:
            raise InfraError(f"milestone {m['id']} has no calibrated cases (run the GPU calibration first)")
        return cases

    def seeds(self, m: dict) -> list[int]:
        if self.fresh is None:
            self.fresh = draw_seed(self.ctx) & 0x7FFFFFFF
        fixed = list((self.spec.get("seeds") or {}).get("fixed") or [7])
        if self.mode == "pool" and self.pool_seeds == "fixed":
            return fixed
        extra = [(self.fresh * 2654435761 + k) & 0x7FFFFFFF for k in range(1, int(m.get("repeat") or 1))]
        return [*fixed, self.fresh, *extra]

    def _pair(self, tag: str, cases: list[str], seed: int) -> dict:
        """The reference, then the candidate, on one chunk of cases (outputs compared outside the candidate's process and
        dropped right after); a chunk whose reference output is too large is halved."""
        try:
            ref = self.run_ref(tag, cases, seed, cache=False)
        except _DumpTooBig:
            if len(cases) == 1:
                if self.mode == "pool":  # calibration: a case too large to compare leaves the pool
                    return {"cmp": {cases[0]: {"pass": False, "infra": "reference output too large"}}, "launches": 0, "rc": [], "timed_out": False}
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
        # a candidate that takes ten times the reference's wall time plus a minute on a correctness chunk is hung (a faulty
        # kernel that never returns): the chunk fails instead of holding the GPU for the default hour
        cand, r = self.run_cand(tag, cases, seed, timeout=max(120.0, 10 * float(ref.get("wall_s") or 0) + 60))
        return {"cmp": kh.compare_dumps(cand["recs"], ref["recs"], cases=cases), "launches": cand["launches"], "rc": [r.rc], "timed_out": r.timed_out}

    def golden(self, m: dict) -> Verdict:
        cases = self.cases_of(m)
        detail, ok, per_case, ratio, per_fixed = {}, True, {}, {}, {}
        fixed = set((self.spec.get("seeds") or {}).get("fixed") or [7])
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
                if seed in fixed and seed != self.fresh:
                    per_fixed.setdefault(k, []).append(bool(v["pass"]))
                r = case_ratio(v)
                ratio[k] = max(ratio.get(k, 0.0), r)
            # pool mode reruns no case another milestone ran: the launch evidence belongs to the grade, not the calibration
            launch_ok = self.mode == "pool" or launches is None or launches >= max(1, len(cases) - len(fails))
            detail[f"seed{'_fresh' if seed == self.fresh else ''}_{seed}"] = {"failed": len(fails), "cases": len(cases), "launches": launches,
                                                                               "rc": sorted(set(rcs)), "examples": dict(list(fails.items())[:4])}
            ok &= not fails and launch_ok and not timed_out
            if not launch_ok:
                detail["launch_error"] = "fewer kernel launches than cases: an op ran off the GPU"
        detail["per_case"] = {k: all(v) for k, v in per_case.items()}  # the calibration's input (pool) and its stability check (grade)
        # the same on the fixed seeds alone: a revert's break there repeats in every later grade, one seen only on a fresh
        # seed may not (the calibration owns a case only on a fixed-seed break)
        detail["per_case_fixed"] = {k: all(v) for k, v in per_fixed.items()}
        detail["per_case_ratio"] = {k: round(v, 4) if v != float("inf") else None for k, v in ratio.items()}  # worst error / tolerance
        return Verdict(ok, {**detail, "seed_revealed_after_run": True})

    def fa_probe(self, m: dict) -> Verdict:
        """Calibration only (``--param fa_probe=1``): every FLASH_ATTN_EXT case of the milestone's pool, the reference (B)
        run twice per fixed seed, once against its CPU kernel and once against the exact attention computation; no
        candidate runs. Per case: B's GPU against the CPU and against the exact result, the CPU against the exact result,
        and B's GPU against itself across the two processes (determinism). ``fa_reference_verdict`` decides."""
        cases = [c for c in self.cases_of(m) if c.startswith("FLASH_ATTN_EXT(")]
        ev: dict[str, dict] = {}
        for seed in list((self.spec.get("seeds") or {}).get("fixed") or [7]):
            for i in range(0, len(cases), CHUNK):
                part = cases[i:i + CHUNK]
                try:
                    cpu = self.run_ref(f"fap-{m['id']}-c{i}", part, seed, cache=False, exact=[])
                    ex = self.run_ref(f"fax-{m['id']}-c{i}", part, seed, cache=False, exact=part)
                except _DumpTooBig:
                    continue
                c_by, x_by, done = kh.by_case(cpu["recs"]), kh.by_case(ex["recs"]), set(ex["exact"])
                for c in part:
                    a, b = c_by.get(c) or [], x_by.get(c) or []
                    if not a or len(a) != len(b):
                        continue
                    row = {"tol": float(a[0]["max_err"]), "applied": c in done,
                           "gpu_vs_cpu": max(kh.nmse(x["out"], x["ref"]) for x in a),
                           "gpu_vs_exact": max(kh.nmse(y["out"], y["ref"]) for y in b),
                           "cpu_vs_exact": max(kh.nmse(x["ref"], y["ref"]) for x, y in zip(a, b, strict=True)),
                           "gpu_vs_gpu": max(kh.nmse(x["out"], y["out"]) for x, y in zip(a, b, strict=True))}
                    prev = ev.get(c)
                    ev[c] = row if prev is None else {k: (prev[k] and row[k]) if isinstance(row[k], bool) else max(prev[k], row[k]) for k in row}
        clean = {c: {k: (None if v == float("inf") else round(v, 9)) if isinstance(v, float) else v for k, v in r.items()} for c, r in ev.items()}
        return Verdict(True, {"fa_evidence": clean, "cases": len(cases)})

    def sanitize(self, m: dict, tool: str) -> Verdict:
        # the calibration's sanitizer cases launch the fault's kernels (profiled with CUPTI); else the milestone's first cases
        cases = list(m.get("sanitizer_cases") or [])[:4] or self.cases_of(m)[:4]
        wrap = [SANITIZER, "--tool", tool, "--error-exitcode", "97", "--print-limit", "30", *(["--leak-check", "no"] if tool == "memcheck" else [])]
        _c, r = self.run_cand(f"{m['id']}-{tool}", cases, 7, dump=False, wrap=wrap, launches=False, timeout=7200)
        clean, summary = kp.sanitizer_clean(r.rc, r.log)
        if not summary and "COMPUTE-SANITIZER" not in r.log:  # the tool never started (the grader's toolkit): not the candidate's fault
            raise InfraError(f"compute-sanitizer did not start (rc {r.rc}): {r.log[-300:]}")
        return Verdict(clean, {"tool": tool, "rc": r.rc, "summary": summary, "cases": len(cases), "log": "" if clean else r.log[-1500:]})

    def perf(self, m: dict) -> Verdict:
        bars = (self.spec.get("perf_bars") or {}).get(self.gpu) or {}
        key = f"{m['id']}_ms"
        cases = list(m.get("perf_cases") or [])
        if not cases:
            return Verdict(False, {}, infra=f"milestone {m['id']} has no calibrated perf cases")
        groups = perf_groups(key, cases, bool(self.ctx.params.get("perf_each")))
        if any(k not in bars for k in groups):
            return Verdict(False, {"gpu": self.gpu}, infra=f"no calibrated perf bar for {m['id']} on {self.gpu!r}")
        r1 = int((self.spec.get("perf") or {}).get("r") or 3)
        # the launch evidence (CUPTI) slows every kernel it traces, and the reference runs untraced: launches are counted on
        # separate, untimed candidate runs (before 2026-10-22 the traced candidate timed ~10% slower than the same build)
        for gi, (k, cs) in enumerate(groups.items()):
            n = {}
            for reps in (r1, 3 * r1):
                c, rr = self.run_cand(f"perfl-{gi}", cs, 7, mode="perf", reps=reps, dump=False)
                if rr.rc != 0 or rr.timed_out:
                    return Verdict(False, {"error": f"counted run of {m['id']} failed (rc {rr.rc})"})
                n[reps] = c["launches"] or 0
            if n[3 * r1] - n[r1] < 2 * r1 * len(cs):
                return Verdict(False, {"error": f"{m['id']}: kernel launches do not scale with repetitions (skipped or memoized work)", "launches": n})

        def round_fn(rnd: int) -> dict:
            orig, cand = {}, {}
            for gi, (k, cs) in enumerate(groups.items()):
                tag = f"perf{rnd}" if len(groups) == 1 else f"perf{rnd}-{gi}"
                o_s, c_s = {}, {}
                for reps in (r1, 3 * r1):
                    o = self.run_ref(tag, cs, 7, mode="perf", reps=reps, dump=False)
                    c, rr = self.run_cand(tag, cs, 7, mode="perf", reps=reps, dump=False, launches=False)
                    if rr.rc != 0 or rr.timed_out:
                        raise CandidateError(f"timed run of {m['id']} failed (rc {rr.rc})")
                    o_s[reps], c_s[reps] = o["wall_s"], c["wall_s"]
                per = lambda a, b: max(1e-6, (b - a) / (2 * r1)) * 1e3
                orig[k], cand[k] = per(o_s[r1], o_s[3 * r1]), per(c_s[r1], c_s[3 * r1])
            return {"original": orig, "candidate": cand}

        pb = PerfBars(bars={k: float(bars[k]) for k in groups}, round_fn=round_fn, rounds=3, remeasure=1, spread_max=0.25,
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
            if c == "golden" and ctx.params.get("fa_probe") and g.mode == "pool":
                fn = (lambda ctx, m=m: g.fa_probe(m))
            elif c == "golden":
                fn = (lambda ctx, m=m: g.golden(m))
            elif c in ("racecheck", "synccheck", "memcheck"):
                fn = (lambda ctx, m=m, c=c: g.sanitize(m, c))
            elif c == "perf" and g.mode != "pool":  # a pool grade is the calibration's per-case input: no bar exists yet, and its
                fn = (lambda ctx, m=m: g.perf(m))  # infra verdict masked the whole oracle pool grade of every perf task
            else:
                continue
            checks.append(Check(f"{m['id']}:{c}", m["id"], fn, requires=("build",)))
    only = [c for c in str(ctx.params.get("only") or "").split(",") if c]
    if only:  # calibration: the build plus the named checks (``m1:perf``); a grade with ``only`` is never a task grade
        checks = [c for c in checks if c.gate == "build" or c.id in only]
        keep = {"build"} | {c.gate for c in checks}
        gates = [gt for gt in gates if gt.id in keep]
    return Plan(spec["task_id"], gates, checks)
