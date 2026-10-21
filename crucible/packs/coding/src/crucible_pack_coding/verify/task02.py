"""task02_pcops_triton grader re-expressed with the M7 kit (runs inside ``task02-pcops-triton`` with /opt/venv python).

Gate for gate the seed grader (``migration_tasks/task02_pcops_triton/tests/grade.py``):

    static   no C++/CUDA sources or extension build, no foreign kernel libraries, Triton kernels present
    install  pip install of the candidate with no CUDA compiler reachable, no compiled extension
    ops      hidden operator suite                    (kit pytest_suite: results channel + expected count)
    hard     opcheck, determinism, streams, threads, compile, errors, large inputs
    app      pristine pcnet / pcdata suites and golden logits / training trajectories
    perf     per-case time vs the original CUDA extension, best of 3 interleaved rounds, one re-measurement

The seed grader's B.10 hole was the worst of the five: it ran the candidate's ``setup.py`` and imports **as root** in the
verifier, then re-ran ``/tests/hidden/bench_pcops.py`` from the same directory, so the candidate could rewrite the
benchmark, slow the reference, write the reward file or leave processes behind. Here every candidate command runs
unprivileged with a fresh HOME, the benchmark and tests are root-owned copies made before the candidate runs, the
CUDA original is root-only, grader code comes first on ``sys.path`` (candidate after the stdlib), pytest verdicts come
from the results channel, and the benchmark's in-process rows are cross-checked against the grader's clock
(``perf.rows_vs_clock``: two runs with different repetition counts, timed from outside).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from crucible.verify.runtime.patterns.hidden_tests import count_pytest, install_plugin, pytest_suite
from crucible.verify.runtime.patterns.perf import CandidateError, LoadGuard, PerfBars, rows_vs_clock
from crucible.verify.runtime.patterns.static import Rule, RuleSet, scan
from crucible.verify.runtime.plan import Check, Context, Gate, Plan, Verdict
from crucible.verify.runtime.trusted import InfraError

TASK = "task02_pcops_triton"
GATES = ["static", "install", "ops", "hard", "app", "perf"]
# M7 R6: the instruction's rules (pure Triton, no native build) are hard gates; correctness, consumers and perf are outcomes
KINDS = {"static": "gate", "install": "gate", "ops": "outcome", "hard": "outcome", "app": "outcome", "perf": "outcome"}
TEXT = {".py", ".toml", ".cfg", ".txt"}
FOREIGN = "ctypes|cffi|cupy|numba|pycuda|torch_scatter|torch_cluster|torch_sparse|pointnet2_ops"
RULES = RuleSet(
    forbid=[Rule("native_extension", r"cpp_extension|CUDAExtension|CppExtension|load_inline|ext_modules|torch\.utils\.cpp_extension|TORCH_LIBRARY", suffixes=TEXT),
            Rule("foreign_kernels", rf"\bimport\s+({FOREIGN})\b|\bfrom\s+({FOREIGN})\b", suffixes=TEXT),
            Rule("load_library", r"torch\.ops\.load_library|ctypes\.CDLL|cdll\.LoadLibrary", suffixes=TEXT)],
    require=[Rule("triton", r"@triton\.jit", suffixes={".py"})],
    forbidden_files={".cu", ".cuh", ".cpp", ".cc", ".cxx", ".c", ".h", ".hpp", ".so", ".ptx", ".cubin", ".fatbin"}, forbid_native=True,
    banned_modules=set(FOREIGN.split("|")), exclude=(".git", "tests", "build", "dist", "*.egg-info", "__pycache__"), text_suffixes=TEXT)
PY = "/opt/venv/bin/python"


def tail(text, n=40):
    return "\n".join((text or "").splitlines()[-n:])


def rows(text: str) -> dict:
    out = {}
    for line in (text or "").splitlines():
        if line.startswith("{"):
            try:
                r = json.loads(line)
                out[r["case"]] = float(r["median_ms"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass
    return out


class Task02:
    def __init__(self, ctx: Context):
        self.ctx, self.t = ctx, ctx.trusted
        self.tests = Path(ctx.params.get("tests", "/tests"))
        self.originals = Path(ctx.params.get("originals", "/opt/originals"))
        self.py = ctx.params.get("python", PY)
        self.thresholds = json.loads((self.tests / "perf_thresholds.json").read_text())
        self.t.path = "/opt/venv/bin:/usr/local/cuda/bin:" + self.t.path
        self.work = ctx.work
        self.counts: dict[str, int] = {}

    def setup(self, ctx):
        w, t = self.work, self.t
        self.ref_site = t.private(w / "ref_site")
        r = t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", self.ref_site, self.originals / "pcops"],
                  env={"MAX_JOBS": "16"}, timeout=3600)
        if r.rc != 0:
            raise InfraError("cannot build the CUDA original: " + tail(r.log))
        src = w / "consumers_src"
        r = t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "-q", "--target", src, self.originals / "pcdata", self.originals / "pcnet"],
                  timeout=1800)
        if r.rc != 0:
            raise InfraError("cannot install pcnet / pcdata: " + tail(r.log))
        self.cons = t.owned_copy(src, w / "consumers")
        self.hid = t.owned_copy(self.tests / "hidden", w / "hidden", ignore=shutil.ignore_patterns("tools", "__pycache__"))
        app = w / "app_src"
        for rr in ("pcnet", "pcdata"):
            shutil.copytree(self.originals / rr / "tests", app / rr)
        self.app = t.owned_copy(app, w / "app_tests")
        shutil.rmtree(app)
        self.plugin = install_plugin(t, w / "plugin")
        first = [self.hid, self.cons, self.ref_site]
        self.counts["ops"] = count_pytest(ctx, args=["test_hidden_ops.py"], cwd=self.hid, pythonpath=first, plugin_dir=self.plugin, python=self.py)
        self.counts["hard"] = count_pytest(ctx, args=["test_hidden_hard.py"], cwd=self.hid, pythonpath=first, plugin_dir=self.plugin, python=self.py)
        self.app_args = [str(self.app / "pcnet"), str(self.app / "pcdata"), str(self.hid / "test_hidden_app.py")]
        self.counts["app"] = count_pytest(ctx, args=self.app_args, cwd=self.hid, pythonpath=first, plugin_dir=self.plugin, python=self.py)
        if not all(self.counts.values()):
            raise InfraError(f"empty test manifest: {self.counts}")
        ctx.revealed["expected_counts"] = dict(self.counts)

    def static(self, ctx):
        cand = self.work / "pcops"
        shutil.copytree(ctx.candidate, cand, symlinks=True)
        r = scan(cand, RULES)
        ctx.artifacts["cand"] = cand
        return Verdict(r["pass"], {"violations": r["violations"][:50], "count": r["count"], "pruned": r["pruned"][:20]})

    def install(self, ctx):
        if "cand" not in ctx.artifacts:
            self.static(ctx)
        cand = ctx.artifacts["cand"]
        self.t.give(cand)
        site = self.t.udir(self.work / "site")
        r = self.t.run([self.py, "-m", "pip", "install", "--no-deps", "--no-build-isolation", "--no-cache-dir", "-q", "--target", site, cand],
                       env={"CUDA_HOME": "/nonexistent", "PATH": "/opt/venv/bin:/usr/bin:/bin", "MAX_JOBS": "8"}, timeout=1800, candidate=True)
        self.t.seal(site)
        so = [str(p.relative_to(site)) for p in site.rglob("*") if p.is_file() and p.suffix in (".so", ".pyd")]
        ctx.artifacts["site"] = site
        return Verdict(r.rc == 0 and not so, {"log": tail(r.log, 20), "compiled_files": so[:10]})

    def suite(self, gate, args, tmo=3600):
        def run(ctx):
            return pytest_suite(ctx, name=gate, args=args() if callable(args) else args, cwd=self.hid, pythonpath=[self.hid, self.cons], plugin_dir=self.plugin,
                                candidate_path=(ctx.artifacts["site"],), python=self.py, expected=self.counts[gate], timeout=tmo)
        return run

    def _bench(self, reps: int, candidate: bool):
        first = [self.hid, self.cons] + ([] if candidate else [self.ref_site])
        last = [self.ctx.artifacts["site"]] if candidate else []
        r = self.t.python(script=self.hid / "bench_pcops.py", args=["--reps", reps], first=first, last=last, python=self.py, timeout=3600, candidate=candidate)
        if r.rc != 0:
            if candidate:
                raise CandidateError(f"candidate benchmark failed: {tail(r.log, 8)}")
            raise InfraError(f"original benchmark failed: {tail(r.log, 8)}")
        return rows(r.stdout), r.wall_s

    def perf_round(self, rnd):
        res = {"original": {}, "candidate": {}}
        for side, cand in (("original", False), ("candidate", True)):
            self.t.kill()
            short, w1 = self._bench(15, cand)
            if cand:  # the candidate's rows must agree with the grader's clock
                _, w2 = self._bench(45, cand)
                x = rows_vs_clock(short, w1, w2, extra_reps=30)
                self.ctx.revealed.setdefault("rows_vs_clock", []).append(x)
                if not x["pass"]:
                    raise CandidateError(f"benchmark rows disagree with the grader's clock: {x}")
            res[side] = short
        return res

    def perf(self, ctx):
        p = ctx.params
        guard = None if p.get("no_load_guard") else LoadGuard(max_gpu_util=int(p.get("max_gpu_util", 20)), wait_s=float(p.get("load_wait_s", 1800)))
        return PerfBars({k: float(v) for k, v in self.thresholds["max_ratio"].items()}, self.perf_round, rounds=int(p.get("perf_rounds", 3)), remeasure=1,
                        spread_max=p.get("spread_max", 0.25), guard=guard)(ctx)


def plan(ctx: Context) -> Plan:
    g = Task02(ctx)
    checks = [Check("static", "static", g.static, tier="fast", provenance="instruction:pure Triton"),
              Check("install", "install", g.install, tier="fast", provenance="instruction:no native build"),
              Check("ops", "ops", g.suite("ops", ["test_hidden_ops.py"]), tier="fast", requires=("install",), provenance="reference_differential:CUDA original"),
              Check("hard", "hard", g.suite("hard", ["test_hidden_hard.py"]), requires=("install",), provenance="reference_differential;construction incidents"),
              Check("app", "app", g.suite("app", lambda: g.app_args), requires=("install",), provenance="consumers:pristine pcnet/pcdata + goldens"),
              Check("perf", "perf", g.perf, requires=("static", "install", "ops", "hard", "app"), provenance="calibration:perf bars", cost="expensive")]
    return Plan(TASK, [Gate(x, KINDS[x]) for x in GATES], checks, setup=g.setup)
