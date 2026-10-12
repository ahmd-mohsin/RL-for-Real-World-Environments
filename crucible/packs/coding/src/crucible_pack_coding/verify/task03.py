"""task03_mdlite_pallas grader re-expressed with the M7 kit (runs inside ``task03-mdlite-pallas`` with /opt/venv python).

Gate for gate the seed grader (``migration_tasks/task03_mdlite_pallas/tests/grade.py``):

    static     no C/C++/CUDA sources or native build, no foreign kernel libraries or host callbacks, Pallas present
    install    pip install of the candidate (unprivileged, no CUDA compiler reachable), no compiled extension
    api/forces/dynamics/physics   hidden pytest suites (kit ``pytest_suite``: results channel + expected counts)
    app        pristine mdbuild / mdkit / mdlite test suites against the candidate
    fresh      systems from a seed drawn at grading time; references by the CUDA original (root-only)
    perf       ms/step vs the CUDA original, bars from perf_thresholds.json

The B.10 holes in the seed grader, and what the kit does instead:

- pytest gates passed on ``rc == 0`` (``os._exit(0)`` under pytest passed): verdicts now come from the results channel,
  with the expected test counts collected by the grader against the CUDA original before any candidate code runs;
- the candidate came *first* on ``PYTHONPATH`` (``site:consumers:hidden``) and could shadow the consumers: the
  interpreter now runs isolated, grader code first and the candidate after the standard library;
- the CUDA original was installed in a directory the candidate could read (delegation): it is now root-only (0700);
- benchmark timings were printed by the benchmark *inside* the candidate's process: perf now times two runs per case
  (K and 5K steps) from outside the process and takes the difference (JIT, start-up and warm-up cancel); the end state
  of the long run must match the original's (task03's own "same physics" rule), so skipping work is not fast;
- ``fresh_cases.py`` draws systems smaller than twice the cut-off on ~18% of seeds and the grader crashed: the
  reference runs first on each drawn seed, and a seed the *original* cannot run is redrawn (up to 5 times) before
  any candidate code sees it;
- static checks skipped ``tests/``, ``docs/``, ``build/``: those paths are now pruned before the scan and the install,
  so what is scanned is what is installed; dynamic imports of banned modules and runner tampering are flagged.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from crucible.verify.runtime.patterns.hidden_tests import count_pytest, install_plugin, pytest_suite
from crucible.verify.runtime.patterns.perf import CandidateError, LoadGuard, PerfBars
from crucible.verify.runtime.patterns.static import Rule, RuleSet, scan
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import InfraError

TASK = "task03_mdlite_pallas"
GATES = ["static", "install", "api", "forces", "dynamics", "physics", "app", "fresh", "perf"]
SUITES = (("api", "test_hidden_api.py", 2400), ("forces", "test_hidden_forces.py", 3600), ("dynamics", "test_hidden_dynamics.py", 5400),
          ("physics", "test_hidden_physics.py", 5400))
FORBIDDEN_FILES = {".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".so", ".ptx", ".cubin", ".fatbin", ".a", ".o", "CMakeLists.txt", "Makefile"}
BANNED = {"ctypes", "cffi", "cupy", "numba", "pycuda", "torch", "triton", "cuda", "tensorflow", "warp", "taichi"}
TEXT = {".py", ".toml", ".cfg", ".txt"}
RULES = RuleSet(
    forbid=[Rule("native_build", r"ext_modules|Extension\s*\(|cmake|nvcc|cpp_extension|load_inline", suffixes=TEXT),
            Rule("foreign_kernels", r"(?m)^\s*(import|from)\s+(ctypes|cffi|cupy|numba|pycuda|torch|triton|cuda|tensorflow|warp|taichi)\b", suffixes=TEXT),
            Rule("host_callback", r"pure_callback|io_callback|host_callback|jax\.ffi|ffi_call|xla_client|custom_call", suffixes=TEXT),
            Rule("load_library", r"CDLL|cdll|LoadLibrary|dlopen", suffixes=TEXT)],
    require=[Rule("pallas", r"\bpallas_call\b", suffixes={".py"})],
    forbidden_files=FORBIDDEN_FILES, forbid_native=True, banned_modules=BANNED,
    exclude=(".git", "tests", "build", "dist", "docs", "*.egg-info", "__pycache__"), text_suffixes=TEXT)
PY = "/opt/venv/bin/python"


def tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


class Task03:
    def __init__(self, ctx: Context):
        self.ctx, self.t = ctx, ctx.trusted
        self.tests = Path(ctx.params.get("tests", "/tests"))
        self.originals = Path(ctx.params.get("originals", "/opt/originals"))
        self.py = ctx.params.get("python", PY)
        self.thresholds = json.loads((self.tests / "perf_thresholds.json").read_text())
        self.t.path = "/opt/venv/bin:/usr/local/cuda/bin:" + self.t.path
        self.work = ctx.work
        self.counts: dict[str, int] = {}

    # ------------------------------------------------------------------ setup (root, before any candidate code)
    def setup(self, ctx):
        w, t = self.work, self.t
        self.ref_site = t.private(w / "ref_site")  # the CUDA original: root-only, so it cannot be delegated to
        r = t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", self.ref_site, self.originals / "mdlite"],
                  env={"MAX_JOBS": "16"}, timeout=3600)
        if r.rc != 0:
            raise InfraError("cannot build the CUDA original: " + tail(r.log))
        cons_src = w / "consumers_src"
        r = t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", cons_src, self.originals / "mdbuild",
                   self.originals / "mdkit"], timeout=1800)
        if r.rc != 0:
            raise InfraError("cannot install mdbuild / mdkit: " + tail(r.log))
        self.cons = t.owned_copy(cons_src, w / "consumers")
        self.hid = t.owned_copy(self.tests / "hidden", w / "hidden", ignore=shutil.ignore_patterns("tools", "__pycache__"))
        self.fresh_py = t.owned_copy(self.tests / "fresh_cases.py", w / "fresh_tool" / "fresh_cases.py")
        app = w / "app_src"
        for rr in ("mdbuild", "mdkit", "mdlite"):
            shutil.copytree(self.originals / rr / "tests", app / rr)
        self.app = t.owned_copy(app, w / "app_tests")
        shutil.rmtree(app)
        self.plugin = install_plugin(t, w / "plugin")
        self.driver = t.owned_copy(self._write_driver(w / "driver_src"), w / "driver")
        # expected-count manifest: collected by the grader against the CUDA original (never against the candidate)
        for gate, fname, _ in SUITES:
            self.counts[gate] = count_pytest(ctx, args=[fname], cwd=self.hid, pythonpath=[self.hid, self.cons, self.ref_site], plugin_dir=self.plugin, python=self.py)
        self.counts["app"] = count_pytest(ctx, args=[str(self.app / x) for x in ("mdbuild", "mdkit", "mdlite")], cwd=self.work,
                                          pythonpath=[self.cons, self.ref_site], plugin_dir=self.plugin, python=self.py)
        if not all(self.counts.values()):
            raise InfraError(f"empty test manifest: {self.counts}")
        ctx.revealed["expected_counts"] = dict(self.counts)

    def _write_driver(self, d: Path) -> Path:
        d.mkdir(parents=True, exist_ok=True)
        (d / "perf_driver.py").write_text('''"""Grader-owned perf driver: one bench case, warm-up, then K timed-by-the-grader steps; end state to a file."""
import json, os, sys, tempfile
sys.path.insert(0, sys.argv[4])
import bench_mdlite as b
import mdlite as md
name, k, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
fname, cfg, _ = b.CASES[name]
with tempfile.TemporaryDirectory() as o:
    sim = md.Simulation(os.path.join(b.SYS, fname), cfg, output_dir=o)
    sim.run(200)
    sim.run(k)
    sim.finish()
    e = sim.energies()
    n = sim.natoms
    sim.close()
json.dump({"pot_per_atom": e["potential"] / n, "temperature": e["temperature"], "steps": 200 + k}, open(out, "w"))
''')
        return d

    # ------------------------------------------------------------------ gates
    def static(self, ctx):
        cand = self.work / "mdlite"
        shutil.copytree(ctx.candidate, cand, symlinks=True)
        r = scan(cand, RULES)  # prunes tests/ docs/ build/ ... from the copy that will be installed
        ctx.artifacts["cand"] = cand
        return Verdict(r["pass"], {"violations": r["violations"][:50], "count": r["count"], "pruned": r["pruned"][:20]})

    def install(self, ctx):
        if "cand" not in ctx.artifacts:  # the static check prunes and copies; the fast tier may reach install first
            self.static(ctx)
        cand = ctx.artifacts["cand"]
        self.t.give(cand)
        site = self.t.udir(self.work / "site")
        r = self.t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir", "-q", "--target", site, cand],
                       env={"CUDA_HOME": "/nonexistent", "CUDACXX": "/nonexistent", "PATH": "/opt/venv/bin:/usr/bin:/bin"}, timeout=1800, candidate=True)
        self.t.seal(site)
        so = [str(p.relative_to(site)) for p in site.rglob("*") if p.is_file() and p.suffix in (".so", ".pyd", ".dylib")]
        ctx.artifacts["site"] = site
        return Verdict(r.rc == 0 and not so, {"log": tail(r.log, 20), "compiled_files": so[:10]})

    def suite(self, gate, fname, tmo):
        def run(ctx):
            return pytest_suite(ctx, name=gate, args=[fname], cwd=self.hid, pythonpath=[self.hid, self.cons], plugin_dir=self.plugin,
                                candidate_path=(ctx.artifacts["site"],), python=self.py, expected=self.counts[gate], timeout=tmo)
        return run

    def app_gate(self, ctx):
        return pytest_suite(ctx, name="app", args=[str(self.app / x) for x in ("mdbuild", "mdkit", "mdlite")], cwd=self.work, pythonpath=[self.cons],
                            plugin_dir=self.plugin, candidate_path=(ctx.artifacts["site"],), python=self.py, expected=self.counts["app"], timeout=2400)

    def fresh(self, ctx):
        sys.path.insert(0, str(self.fresh_py.parent))
        import secrets

        import fresh_cases  # task-owned comparator (numpy), loaded by the grader

        tries = []
        for _ in range(5):
            seed = secrets.randbits(48)
            d = self.work / f"fresh_{seed}"
            d.mkdir()
            b = self.t.python(script=self.fresh_py, args=["build", seed, d], first=[self.cons, self.ref_site], python=self.py, timeout=1800)
            refdir = self.t.private(self.work / f"fresh_ref_{seed}")
            r = self.t.python(script=self.fresh_py, args=["run", d, refdir / "ref.npz"], first=[self.cons, self.ref_site], python=self.py, timeout=1800) \
                if b.rc == 0 else b
            if b.rc == 0 and r.rc == 0:
                break
            tries.append({"seed": seed, "reason": tail(r.log, 3)})  # the original cannot run this draw: redraw before any candidate run
        else:
            raise InfraError(f"no runnable fresh draw in 5 seeds: {tries}")
        for p in [d, *d.rglob("*")]:
            p.chmod(0o755 if p.is_dir() else 0o644)
        out = self.t.udir(self.work / "fresh_cand")
        c = self.t.python(script=self.fresh_py, args=["run", d, out / "cand.npz"], first=[self.cons], last=[ctx.artifacts["site"]], python=self.py,
                          timeout=3600, candidate=True)
        self.t.seal(out)
        ctx.revealed["fresh_seed"] = seed
        if c.rc != 0 or not (out / "cand.npz").exists():
            return Verdict(False, {"seed": seed, "error": "candidate failed", "log": tail(c.log), "redrawn": tries})
        terms = ("lj", "lj_14", "disp_corr", "coulomb_sr", "coulomb_14", "coulomb_recip", "bond", "angle", "dihedral", "potential", "kinetic", "total",
                 "conserved", "temperature", "pressure", "volume")
        fails = fresh_cases.compare(refdir / "ref.npz", out / "cand.npz", terms)
        return Verdict(not fails, {"seed": seed, "failures": fails[:30], "redrawn": tries})

    def _timed(self, case, k, candidate, tag):
        site = self.ctx.artifacts["site"]
        if candidate:
            d = self.t.udir(self.work / "perf" / f"{tag}_c")
            r = self.t.python(script=self.driver / "perf_driver.py", args=[case, k, d / "end.json", self.hid], first=[self.cons], last=[site],
                              python=self.py, timeout=3600, candidate=True)
            self.t.seal(d)
        else:
            d = self.t.private(self.work / "perf" / f"{tag}_o")
            r = self.t.python(script=self.driver / "perf_driver.py", args=[case, k, d / "end.json", self.hid], first=[self.cons, self.ref_site],
                              python=self.py, timeout=3600)
        if r.rc != 0 or not (d / "end.json").exists():
            if candidate:
                raise CandidateError(f"{case}: candidate timed run failed: {tail(r.log, 6)}")
            raise InfraError(f"{case}: original timed run failed: {tail(r.log, 6)}")
        return r.wall_s, json.loads((d / "end.json").read_text())

    def perf_round(self, rnd):
        sys.path.insert(0, str(self.hid))
        import bench_mdlite as b

        res = {"original": {}, "candidate": {}}
        for case in self.thresholds["max_ratio"]:
            n = b.CASES[case][2]
            for side, cand in (("original", False), ("candidate", True)):
                self.t.kill()
                w1, _ = self._timed(case, n, cand, f"{case}{rnd}s")
                w2, end = self._timed(case, 5 * n, cand, f"{case}{rnd}l")
                per_step = (w2 - w1) / (4 * n) * 1e3  # ms/step from the grader's clock: start-up and JIT cancel
                if per_step <= 0:  # the long run was not longer: start-up depended on the step count (M7 audit V4)
                    raise (CandidateError if cand else InfraError)(f"{case}: the {side} long run took {w2:.3f}s, the short {w1:.3f}s")
                res[side][case] = per_step
                res.setdefault("_end", {})[side] = end
            if res["candidate"][case] < 0.05 * res["original"][case]:  # 20x faster per step than the original: not a measurement
                raise CandidateError(f"{case}: candidate {res['candidate'][case]:.4f} ms/step vs original {res['original'][case]:.4f}: implausibly fast")
            o, c = res["_end"]["original"], res["_end"]["candidate"]
            sane = abs(c["pot_per_atom"] - o["pot_per_atom"]) <= 0.03 * abs(o["pot_per_atom"]) + 0.05 and \
                abs(c["temperature"] - o["temperature"]) <= 0.05 * o["temperature"] + 2.0 and c["steps"] == o["steps"]
            if not sane:
                raise CandidateError(f"{case}: the benchmark's end state differs from the original's: {c} vs {o}")
            res.pop("_end")
        return res

    def perf(self, ctx):
        p = ctx.params
        guard = None if p.get("no_load_guard") else LoadGuard(max_gpu_util=int(p.get("max_gpu_util", 20)), wait_s=float(p.get("load_wait_s", 1800)))
        return PerfBars({k: float(v) for k, v in self.thresholds["max_ratio"].items()}, self.perf_round, rounds=int(p.get("perf_rounds", 3)), remeasure=1,
                        spread_max=p.get("spread_max", 0.25), guard=guard)(ctx)


def plan(ctx: Context) -> Plan:
    g = Task03(ctx)
    checks = [Check("static", "static", g.static, tier="fast", provenance="instruction:pure JAX + Pallas"),
              Check("install", "install", g.install, tier="fast", provenance="instruction:no native build")]
    checks += [Check(gate, gate, g.suite(gate, f, tmo), tier="fast" if gate in ("api", "forces") else "full", requires=("install",),
                     provenance="reference_differential:CUDA original goldens") for gate, f, tmo in SUITES]
    checks += [Check("app", "app", g.app_gate, requires=("install",), provenance="consumers:pristine mdbuild/mdkit"),
               Check("fresh", "fresh", g.fresh, requires=("install",), provenance="incident:B.2 fresh draws"),
               Check("perf", "perf", g.perf, requires=tuple(x for x in GATES if x != "perf"), provenance="calibration:perf bars", cost="expensive")]
    return Plan(TASK, [Gate(x) for x in GATES], checks, setup=g.setup)

