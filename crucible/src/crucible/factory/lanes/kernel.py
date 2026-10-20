"""Kernel-migration lane (swarm-pipeline/06): mined GPU-migration commits as ``LongTask`` windows on a difficulty ladder.

One window = one real migration commit (or one independent file of it), A = its parent, oracle = the commit. The
World expert's work is deterministic here (``discover_programs``): the programs the migrated files affect, how the
repository builds them (Makefile ``source =`` lists, CMake ``add_executable``), and how it runs them (Makefile ``run:``
recipes). Milestones are per migrated file and own the programs whose main file it is (a migrated header with no program
of its own folds into the milestones of the programs that include it); a milestone ``requires`` another when its programs
include the other's file, and the edge counts only once executed (``executed_edges``: the dependent fails to build with
its own file migrated alone and builds with the prerequisite added).

``KernelTask`` is the ``LongTask``: grading goes to a GPU backend (``GpuBackend``) that runs ``kernel_plan`` in the
verifier image; with no backend the grade is masked ``no_gpu`` (V8), never a 0. Rungs (§2) come from the window's shape;
L4 needs calibrated sanitizer and perf gates (GPU), so mined windows are ``l4_candidate`` until calibrated; L5 is the
task04 seed (``SEEDS``).
"""

from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

from crucible.factory.lanes import kernel_mine as km
from crucible.factory.lanes import kernel_plan as kp
from crucible.factory.longtask import GroundCase, LongTaskRef, new_file_patch, status_claim_patch
from crucible.factory.pathcheck import PathRules

LANE = "kernel"
FAMILY = "kernel-migration"
INTENT = "K1"
MINER = "miner:kernel-lane"
MAIN = re.compile(r"\bint\s+main\s*\(")
UNSUPPORTED = {"gl": re.compile(r'#\s*include\s*[<"](GL/|GLFW|glut|GLES|EGL/|cuda_gl_interop)'),
               "opencv": re.compile(r'#\s*include\s*[<"]opencv'), "mpi": re.compile(r'#\s*include\s*[<"]mpi\.h'),
               "x11": re.compile(r'#\s*include\s*[<"]X11/'), "npp": re.compile(r'#\s*include\s*[<"]npp')}
DRIVER_API = re.compile(r"\bcu(?:Init|DeviceGet|CtxCreate|ModuleLoad|MemAlloc|LaunchKernel|FuncGet|MemcpyHtoD|MemcpyDtoH|DriverGetVersion)\w*\s*\(")
CONSUMER_DIRS = ("test", "tests", "testing", "example", "examples", "sample", "samples", "apps", "benchmarks", "bench")
CUDA_LIBS = {"cublas", "cusparse", "curand", "cufft", "cusolver", "cudadevrt"}
INTENT_TEXT = ("Migrates GPU code across toolkit and API generations (removed texture and surface references, warp-synchronous code "
               "to *_sync intrinsics or cooperative groups, deprecated runtime calls, dropped architectures) while preserving exact "
               "results, race-freedom, concurrency safety and speed.")
KIND_TEXT = {"texref": "texture references (removed in CUDA 12)", "surfref": "surface references (removed in CUDA 12)",
             "warp_sync": "warp intrinsics without a member mask (`__shfl*`, `__any`, `__all`, `__ballot`)",
             "thread_api": "the deprecated `cudaThread*` runtime calls", "lib_api": "cuSPARSE calls removed in CUDA 12",
             "volatile_warp": "implicit warp-synchronous shared-memory code"}
ARCH_TOKEN = "{ARCH}"
LANE_SENTENCES = (
    f"Only CUDA 12.8 is installed; build for this machine's GPU, `sm_{ARCH_TOKEN}`.",
    ("Each program must keep its command line and its output: same printed results and same results copied back from the GPU as "
     "the original code produced on older toolkits, on the inputs it already supports."),
    "The work has to run on the GPU; do not replace kernels with host code.",
)
EPILOGUE = ("How this works: you are working in a sandboxed copy of the repository with no network access. After you finish, the "
            "programs named above are rebuilt from your tree with the grader's own compiler command for this GPU and run on inputs "
            "chosen at grading time; their printed results, the results they copy back from the GPU and the files they write are "
            "compared with what the original code produced. Trying to download code or reading version-control history beyond the "
            "current commit scores the run 0. Before you finish, write STATUS.json at the repository root mapping each requirement id "
            "above to your probability (0 to 1) that it is met. If you decide you cannot finish, end your final message with a line "
            "'GIVE_UP: <reason>'; your work is graded as it stands.")


# ---- programs (the World expert's deterministic part) --------------------------------------------------------------------------
@dataclass
class Program:
    id: str
    main: str
    sources: list[str]
    includes: list[str] = field(default_factory=list)
    defines: list[str] = field(default_factory=list)
    libs: list[str] = field(default_factory=list)
    std: str = "c++17"
    run_dir: str = "."
    cases: dict[str, list[str]] = field(default_factory=dict)
    needs: list[str] = field(default_factory=list)
    closure: list[str] = field(default_factory=list)
    timeout_s: int = 300
    perf: dict[str, Any] | None = None
    openmp: bool = False

    def spec(self) -> dict[str, Any]:
        """The grader's view (``kernel_plan`` spec ``programs[pid]``)."""
        return {"sources": self.sources, "includes": self.includes, "defines": self.defines, "libs": self.libs, "std": self.std,
                "run_dir": self.run_dir, "cases": self.cases or {"default": []}, "fresh": True, "timeout_s": self.timeout_s,
                "compare": {}, "perf": self.perf, "openmp": self.openmp}


def read_raw(p: Path) -> str:
    """A file's text with its line endings kept (CRLF sources must round-trip byte for byte into patches)."""
    with open(p, encoding="utf-8", errors="replace", newline="") as f:
        return f.read()


def write_raw(p: Path, text: str) -> None:
    with open(p, "w", encoding="utf-8", newline="") as f:
        f.write(text)


def pid_of(main: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", main.rsplit(".", 1)[0]).strip("_")


def _read(tree: Path, rel: str) -> str:
    p = tree / rel
    try:
        return p.read_text(errors="replace") if p.is_file() else ""
    except OSError:
        return ""


def files_of(tree: Path) -> list[str]:
    return sorted(p.relative_to(tree).as_posix() for p in tree.rglob("*") if p.is_file() and ".git" not in p.parts)


def include_closure(tree: Path, start: Sequence[str], all_files: set[str], *, depth: int = 6) -> list[str]:
    """Repository files reachable through ``#include`` from ``start`` (relative to the including file, then the include
    dirs, then a unique basename match)."""
    seen = set(start)
    frontier = list(start)
    by_base: dict[str, list[str]] = {}
    for f in all_files:
        by_base.setdefault(f.rsplit("/", 1)[-1], []).append(f)
    for _ in range(depth):
        nxt = []
        for f in frontier:
            base = f.rsplit("/", 1)[0] if "/" in f else ""
            for inc in km.INCLUDE.findall(_read(tree, f)):
                cand = km._norm(f"{base}/{inc}" if base else inc)
                hit = cand if cand in all_files else None
                if hit is None:
                    tail = [g for g in all_files if g.endswith("/" + inc) or g == inc]
                    if len(tail) == 1:
                        hit = tail[0]
                    elif not tail and len(by_base.get(inc.rsplit("/", 1)[-1], [])) == 1:
                        hit = by_base[inc.rsplit("/", 1)[-1]][0]
                if hit and hit not in seen:
                    seen.add(hit)
                    nxt.append(hit)
        frontier = nxt
    return sorted(seen)


MAKE_VAR = re.compile(r"^\s*(source|sources|SOURCE|SOURCES|src|SRC|SRCS|srcs|CU_SRCS|CUDA_SRCS|OBJS?)\s*[:+]?=\s*(.+)$", re.MULTILINE)
MAKE_RUN = re.compile(r"^run\s*:[^\n]*\n((?:\t[^\n]*\n?)+)", re.MULTILINE)
CMAKE_EXE = re.compile(r"(?:add_executable|cuda_add_executable)\s*\(\s*([\w.\-${}]+)\s+([^)]*)\)", re.DOTALL)
CMAKE_LINK = re.compile(r"target_link_libraries\s*\(\s*([\w.\-${}]+)\s+([^)]*)\)", re.DOTALL)
CMAKE_INC = re.compile(r"include_directories\s*\(\s*([^)]*)\)", re.DOTALL)
CMAKE_LIB = re.compile(r"add_library\s*\(\s*([\w.\-]+)\s+(?:STATIC|SHARED|OBJECT)?\s*([^)]*)\)", re.DOTALL)


MAKE_TARGET = re.compile(r"^([\w.\-$()]+)\s*:\s*((?:[\w./\-]+\.o\s*)+)$", re.MULTILINE)


def _make_sources(text: str, d: str, main: str | None = None, files: set[str] | None = None) -> list[str] | None:
    text = re.sub(r"\\\n", " ", text)
    if main and files is not None:  # "prog: a.o b.o": the target whose objects include the main's stem
        stem = main.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        for _, objs in MAKE_TARGET.findall(text):
            stems = [o.rsplit(".", 1)[0] for o in objs.split()]
            if stem in stems:
                out = []
                for st in stems:
                    for ext in (".cu", ".cpp", ".cc", ".c", ".cxx"):
                        f = km._norm(f"{d}/{st}{ext}" if d != "." else f"{st}{ext}")
                        if f in files:
                            out.append(f)
                            break
                if out:
                    return out
    for m in MAKE_VAR.finditer(text):
        vals = [v for v in re.split(r"\s+", m.group(2).replace("\\", " ").strip()) if v and not v.startswith("$")]
        vals = [re.sub(r"\.o$", ".cu", v) for v in vals]
        srcs = [km._norm(f"{d}/{v}" if d != "." else v) for v in vals if v.endswith((".cu", ".cpp", ".cc", ".c", ".cxx"))]
        if srcs:
            return srcs
    return None


def _make_flags(text: str) -> tuple[list[str], list[str], list[str]]:
    incs = re.findall(r"-I\s*([^\s$]+)", text)
    defs = re.findall(r"(?<![\w-])-D\s*([A-Za-z_]\w*(?:=[^\s]+)?)", text)
    libs = [lib for lib in re.findall(r"(?<![\w-])-l\s*([A-Za-z_][\w+]*)", text) if lib in CUDA_LIBS or lib in ("m", "pthread", "gomp")]
    return incs, defs, libs


def _make_run(text: str, tree: Path, d: str) -> list[list[str]]:
    text = re.sub(r"\\\n\s*", " ", text)
    m = MAKE_RUN.search(text)
    if not m:
        return []
    out = []
    for ln in m.group(1).splitlines():
        ln = ln.strip().replace("$(LAUNCHER)", "").strip()
        mm = re.match(r"(?:\./)?(?:\$\(program\)|\$\(EXE\)|\$\(TARGET\)|main|\$\(PROGRAM\))\s*(.*)$", ln)
        if not mm:
            continue
        args = re.split(r"\s+", re.sub(r"[;|].*$|>.*$|\\$", "", mm.group(1)).strip())
        args = [a for a in args if a]
        if any("$" in a for a in args):
            continue
        out.append(args)
    return out


def _arg_files_ok(tree: Path, run_dir: str, args: Sequence[str]) -> bool:
    for a in args:
        if "/" in a or re.search(r"\.(txt|dat|bin|csv|fasta|pgm|ppm|png|bmp|raw|mtx|graph|in)$", a):
            p = tree / km._norm(f"{run_dir}/{a}")
            if not p.exists() and not a.startswith("-"):
                return False
    return True


def discover_programs(tree: Path, mains: Sequence[str], *, all_files: set[str] | None = None) -> dict[str, Program]:
    """Build specs for the programs whose main files are ``mains``, from the repository's own build files at ``tree``."""
    all_files = all_files if all_files is not None else set(files_of(tree))
    out: dict[str, Program] = {}
    for main in mains:
        d = main.rsplit("/", 1)[0] if "/" in main else "."
        mk = next((f"{d}/{n}" if d != "." else n for n in ("Makefile", "makefile", "GNUmakefile") if (f"{d}/{n}" if d != "." else n) in all_files), None)
        mtext = _read(tree, mk) if mk else ""
        sources: list[str] | None = _make_sources(mtext, d, main, all_files) if mtext else None
        incs, defs, libs = _make_flags(mtext) if mtext else ([], [], [])
        includes = [km._norm(f"{d}/{i}" if d != "." else i) for i in incs if not i.startswith("/")]
        cm_inc: list[str] = []
        if sources is None:  # CMake: the executable whose sources name this main
            path = d
            texts = []
            while True:
                cm = f"{path}/CMakeLists.txt" if path != "." else "CMakeLists.txt"
                if cm in all_files:
                    texts.append((path, _read(tree, cm)))
                if path == ".":
                    break
                path = path.rsplit("/", 1)[0] if "/" in path else "."
            for path, text in texts:
                for inc in CMAKE_INC.findall(text):
                    cm_inc += [km._norm(f"{path}/{i}" if path != "." else i) for i in re.split(r"\s+", inc.strip()) if i and "$" not in i]
            local = texts[0][1] if texts and texts[0][0] == d else ""
            base = main.rsplit("/", 1)[-1]
            for name, srcs in CMAKE_EXE.findall(local):
                names = [s for s in re.split(r"\s+", srcs.strip()) if s]
                if base in names:
                    sources = [km._norm(f"{d}/{s}" if d != "." else s) for s in names if s.endswith((".cu", ".cpp", ".cc", ".c", ".cxx"))]
                    for tgt, ls in CMAKE_LINK.findall(local):
                        if tgt == name:
                            for lib in re.split(r"\s+", ls.strip()):
                                lib = lib.replace("CUDA::", "").lower()
                                if lib in CUDA_LIBS:
                                    libs.append(lib)
                                else:  # a repository library: compile its sources in
                                    for path, text in texts + [(c.rsplit("/", 1)[0], _read(tree, c)) for c in all_files if c.endswith("CMakeLists.txt")]:
                                        for lname, lsrc in CMAKE_LIB.findall(text):
                                            if lname == lib:
                                                sources += [km._norm(f"{path}/{s}" if path != "." else s) for s in re.split(r"\s+", lsrc.strip())
                                                            if s.endswith((".cu", ".cpp", ".cc", ".c"))]
                    break
        if not sources:
            sources = [main]
        sources = [s for s in dict.fromkeys(sources) if s in all_files]
        if main not in sources:
            sources.insert(0, main)
        closure = include_closure(tree, sources, all_files)
        dirs = {km.module_of(f) for f in all_files}
        includes = [i for i in dict.fromkeys([*(includes or []), *cm_inc, d, *(km.module_of(f) for f in closure)]) if i in dirs or i == "."]
        needs = sorted({k for k, rx in UNSUPPORTED.items() for f in closure if rx.search(_read(tree, f))})
        if any(DRIVER_API.search(km.strip_c_light(_read(tree, f))) for f in closure):
            libs = [*libs, "cuda"]
        openmp = any("omp.h" in inc for f in closure for inc in km.INCLUDE.findall(_read(tree, f)))
        runs = _make_run(mtext, tree, d) if mtext else []
        cases = {f"run{i}": a for i, a in enumerate(runs) if _arg_files_ok(tree, d, a)}
        if runs and not cases:
            needs.append("data")
        if not runs:
            cases = {"default": []}
        pid = pid_of(main)
        out[pid] = Program(id=pid, main=main, sources=sources, includes=includes, defines=list(dict.fromkeys(defs)), libs=list(dict.fromkeys(libs)),
                           run_dir=d, cases=cases, needs=needs, closure=closure, openmp=openmp)
        out[pid].__dict__["_tree"] = str(tree)
    return out


def mains_affected(tree: Path, files: Sequence[str], all_files: set[str] | None = None, *, scope: str | None = None, limit: int = 12) -> dict[str, list[str]]:
    """migrated file -> main files whose include closure holds it (itself when it is a main). ``scope``: only mains under
    this directory prefix (a header's consumers in the same project area)."""
    all_files = all_files if all_files is not None else set(files_of(tree))
    srcs = sorted(f for f in all_files if f.endswith((".cu", ".cpp", ".cc", ".c", ".cxx")))
    mains = [f for f in srcs if MAIN.search(km.strip_c_light(_read(tree, f))) and (scope is None or f.startswith(scope))]
    out: dict[str, list[str]] = {}
    closures = {m: set(include_closure(tree, [m], all_files)) for m in mains}
    for f in files:
        if f in closures:
            out[f] = [f]
            continue
        out[f] = [m for m in mains if f in closures[m]][:limit]
    return out


# ---- windows -------------------------------------------------------------------------------------------------------------------
def build_milestones(files: Sequence[str], affected: Mapping[str, Sequence[str]], programs: Mapping[str, Program]) -> list[dict[str, Any]]:
    """Two milestone kinds over the programs the migrated ``files`` affect:

    - ``compile``: one per translation unit (a source a program compiles) whose include closure holds a migrated file; it
      passes when the unit compiles for the arch with the grader's flags and its migrated files hold no removed construct.
      CPU-gradable;
    - ``run``: one per program; it passes when the program builds, carries device code for the arch only, launches kernels
      and matches the original's observation (GPU). It ``requires`` the compile milestones of its units (claimed; S6
      executes each edge: the program fails with that unit's files at the start and builds with them migrated).

    Programs with unsupported needs (OpenGL, MPI, NPP, missing data) are left out."""
    usable = {pid: p for pid, p in programs.items() if not p.needs}
    mig = set(files)
    tus: dict[str, dict[str, Any]] = {}
    runs: list[dict[str, Any]] = []
    for pid, prog in sorted(usable.items()):
        if not mig & set(prog.closure):
            continue
        need = []
        for tu in prog.sources:
            clo = set(include_closure_cached(prog, tu))
            hit = sorted(mig & clo)
            if not hit:
                continue
            m = tus.setdefault(tu, {"kind": "compile", "tu": tu, "program": pid, "files": hit, "programs": [pid], "requires": [], "requires_evidence": {}})
            if pid not in m["programs"]:
                m["programs"].append(pid)
            need.append(tu)
        runs.append({"kind": "run", "program": pid, "files": sorted(mig & set(prog.closure)), "programs": [pid], "needs_tus": need, "requires": [],
                     "requires_evidence": {}})
    ms = list(tus.values())
    for i, m in enumerate(ms):
        m["id"] = f"c{i + 1:02d}"
    by_tu = {m["tu"]: m["id"] for m in ms}
    for j, r in enumerate(runs):
        r["id"] = f"r{j + 1:02d}"
        r["requires"] = [by_tu[t] for t in r.pop("needs_tus") if t in by_tu]
    return ms + runs


_CLOSURES: dict[tuple[str, str], list[str]] = {}


def include_closure_cached(prog: Program, tu: str) -> list[str]:
    """A unit's closure within its program's closure (the program's closure was computed from its tree; a unit's own
    reach is the subset reachable from it)."""
    key = (prog.id, tu)
    if key not in _CLOSURES:
        tree = Path(prog.__dict__.get("_tree", ""))
        if tree.is_dir():
            _CLOSURES[key] = include_closure(tree, [tu], set(prog.closure) | {tu})
        else:
            _CLOSURES[key] = list(prog.closure)
    return _CLOSURES[key]


def cross_edges(window: Mapping[str, Any]) -> list[tuple[str, str]]:
    """Executed edges that make a window long: a program's run milestone requiring a unit other than its own main file (the
    program is assembled from several migrated units, or consumes a migrated module from elsewhere)."""
    ms = {m["id"]: m for m in window.get("milestones") or []}
    out = []
    for m in ms.values():
        if m.get("kind") != "run":
            continue
        main = programs_main(window, m["program"])
        for r in m.get("requires") or []:
            if (m.get("requires_evidence") or {}).get(r) == "executed" and r in ms and set(ms[r]["files"]) - {main}:
                out.append((m["id"], r))
    return out


def rung_of(window: Mapping[str, Any]) -> str:
    """L1: at most two migrated files in one module, one program. L2: a program that needs two or more migrated units
    (executed). L3: an executed edge into a program outside the migrated module (a downstream consumer). ``L2-short``: more
    than L1's scope with no executed cross-unit edge (long, not long-horizon)."""
    ms = window.get("milestones") or []
    runs = [m for m in ms if m.get("kind") == "run"]
    if not runs:
        return "none"
    files = window.get("migrated_files") or []
    mods = {km.module_of(f) for f in files}
    if len(files) <= 2 and len(mods) == 1 and len(runs) == 1:
        return "L1"
    edges = cross_edges(window)
    if not edges:
        return "L2-short"
    by = {m["id"]: m for m in ms}
    consumer = any(km.module_of(programs_main(window, by[a]["program"])) not in {km.module_of(f) for f in by[b]["files"]} for a, b in edges)
    return "L3" if consumer else "L2"


def programs_main(window: Mapping[str, Any], pid: str) -> str:
    return str((window.get("programs") or {}).get(pid, {}).get("main", ""))


def l4_candidate(window: Mapping[str, Any]) -> bool:
    """Scope of L2/L3 and code the sanitizer can judge (warp-level kinds) or a repetition argument the perf gate can scale."""
    if window.get("rung") not in ("L2", "L3", "L2-short"):
        return False
    warp = bool(set(window.get("kinds") or {}) & {"warp_sync", "volatile_warp"})
    perf = any((p or {}).get("perf") for p in (window.get("programs") or {}).values())
    return warp or perf


# ---- the task ------------------------------------------------------------------------------------------------------------------
class GpuBackend(Protocol):
    """Runs ``kernel_plan`` for one patch on a GPU host and returns the kit's ``crucible_reward.json`` (and report)."""

    async def grade(self, window: Mapping[str, Any], patch: str) -> dict[str, Any]: ...


@dataclass
class GradeReport:
    """The lane's grade (``longtask._row`` reads ``reward``, ``infra`` and ``meta``)."""

    reward: float
    infra: dict[str, Any]
    meta: dict[str, Any]
    raw: dict[str, Any] = field(default_factory=dict)


def lane_reward(gates: Mapping[str, str], milestone_ids: Sequence[str], *, phi_weight: float = 0.5) -> tuple[float, float, dict[str, bool]]:
    status = {m: gates.get(m) == "pass" for m in milestone_ids}
    phi = sum(status.values()) / len(status) if status else 0.0
    full = bool(status) and all(status.values())
    return phi_weight * phi + (1 - phi_weight) * float(full), phi, status


@dataclass
class KernelTask:
    window: dict[str, Any]
    backend: GpuBackend | None = None
    verifier_version: str = "kernel-v1"

    # -- LongTask members
    @property
    def task_id(self) -> str:
        return str(self.window["id"])

    @property
    def world(self) -> dict[str, Any]:
        w = self.window.get("world") or {}
        return {"kind": "kernel_workspace", "repos": [self.window["repo"]], "image": w.get("image", "crucible-k-base:12.8.1"),
                "digest": w.get("digest", ""), "base_dir": w.get("base_dir") or self.window.get("tree_a") or "", "network": "none", "history": "cut",
                "gpu": True, "arch_env": "CRUCIBLE_CUDA_ARCH"}

    @property
    def prompt(self) -> str:
        return str(self.window.get("prompt") or "")

    @property
    def requirements(self) -> list[dict[str, Any]]:
        return list(self.window.get("requirements") or [])

    @property
    def milestones(self) -> list[dict[str, Any]]:
        return [{"id": m["id"], "kind": m.get("kind", "run"), "tu": m.get("tu"), "program": m.get("program"),
                 "tests": [f"{m['id']}:{m.get('kind', 'run')}:{m.get('tu') or m.get('program')}"], "requires": list(m.get("requires") or []),
                 "requires_evidence": dict(m.get("requires_evidence") or {}), "files": list(m["files"]), "programs": list(m.get("programs") or [])}
                for m in self.window.get("milestones") or []]

    @property
    def path_rules(self) -> PathRules:
        return PathRules(grader_paths=(), step_budget=self.window.get("step_budget"))

    @property
    def oracle(self) -> str:
        return self.oracle_patch()

    @property
    def provenance(self) -> list[str]:
        return [f"obs-{self.task_id}"]

    def oracle_patch(self) -> str:
        return str(self.window.get("oracle_patch") or "")

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None) -> GradeReport:
        ids = [m["id"] for m in self.window.get("milestones") or []]
        if self.backend is None:
            return GradeReport(0.0, {"status": "masked", "failure_kind": "no_gpu", "reason": "no GPU backend configured"}, {"phi": None, "milestones": None})
        raw = await self.backend.grade(self.window, str(patch or ""))
        infra = raw.get("infra") or {"status": "ok"}
        gates = {g: ("pass" if int(v) == 1 else "fail") for g, v in (raw.get("gates") or {}).items()}
        gates.update(raw.get("gate_status") or {})
        comp = [m["id"] for m in self.window.get("milestones") or [] if m.get("kind") == "compile"]
        phi_compile = sum(gates.get(c) == "pass" for c in comp) / len(comp) if comp and all(gates.get(c) in ("pass", "fail") for c in comp) else None
        if infra.get("status") == "masked":  # V8 (a CPU-only grade masks: its GPU milestones were not run); compile Phi is still reported
            return GradeReport(0.0, infra, {"phi": None, "milestones": None, "phi_compile": phi_compile,
                                            "compile_milestones": {c: gates.get(c) == "pass" for c in comp}}, raw)
        reward, phi, status = lane_reward(gates, ids)
        return GradeReport(reward, {"status": "ok", "failure_kind": None, "reason": None},
                           {"phi": phi, "milestones": status, "full_pass": all(status.values()) if status else False, "path": {"violations": []},
                            "phi_compile": phi_compile, "compile_milestones": {c: gates.get(c) == "pass" for c in comp}}, raw)

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        rep = await self.grade_patch(patch)
        if rep.infra.get("status") != "ok":
            raise RuntimeError(f"milestones unmeasured: {rep.infra}")
        return dict(rep.meta["milestones"])

    def floors(self) -> list[GroundCase]:
        files = sorted({f for m in self.window.get("milestones") or [] for f in m["files"]})
        stub = "".join(new_file_patch(f"crucible_floor_{i}.txt", "x\n") for i in range(1))
        jit = new_file_patch("crucible_jit_escape.cu", 'extern "C" void* f(){ return 0; }\n// cuModuleLoadData(0, 0);\n')
        return [GroundCase("no_op", "floor", "", note="the start state"), GroundCase("claims_only", "floor", status_claim_patch(self), note="STATUS.json only"),
                GroundCase("unrelated_edit", "floor", stub, note="a file outside the programs"),
                GroundCase("jit_escape", "floor", jit, note=f"a runtime-PTX loader next to unchanged legacy files {files[:3]}")]

    def mutants(self) -> list[Any]:
        return [Mutant(m["id"], m["solution"], m["milestone"], m.get("kind", "revert"), m.get("note", "")) for m in self.window.get("mutants") or []]

    def alternatives(self) -> list[GroundCase]:
        return [GroundCase(a["id"], "alternative", a["patch"]) for a in self.window.get("alternatives") or []]

    def simplify(self, kind: str, **kw: Any) -> KernelTask:
        """``locate``: the prompt names every file and line holding a removed construct. ``prerequisites``: the
        prerequisite milestones' oracle files are part of the start (their milestones leave the task)."""
        w = json.loads(json.dumps(self.window))
        if kind == "locate":
            loc = w.get("locate") or []
            lines = "\n".join(f"- `{x['file']}` line {x['line']}: `{x['text'][:100]}`" for x in loc[:40])
            w["prompt"] = f"{w.get('prompt', '')}\n\nThe code that no longer builds or behaves on CUDA 12 is here:\n{lines}\n"
        elif kind == "prerequisites":
            pre = {r for m in w["milestones"] for r in m.get("requires") or [] if (m.get("requires_evidence") or {}).get(r) == "executed"}
            keep = [m for m in w["milestones"] if m["id"] not in pre]
            w["start_patch_milestones"] = sorted(pre)
            w["milestones"] = [{**m, "requires": [r for r in m.get("requires") or [] if r not in pre]} for m in keep]
            w["requirements"] = [r for r in w.get("requirements") or [] if r.get("milestone") not in pre]
        else:
            raise ValueError(f"kernel windows simplify by locate or prerequisites, not {kind!r}")
        w["simplified_from"] = {"window": self.task_id, "kind": kind}
        w["id"] = f"{self.task_id}~{kind}"
        return KernelTask(w, self.backend, self.verifier_version)

    def card(self, **kw: Any) -> Any:
        from crucible.factory.cards import RequirementLedger, TaskCard, VerifierCard

        led = ledger_of(self.window)[0] if self.requirements else RequirementLedger()
        return TaskCard(id=self.task_id.replace("~", "-"), family=FAMILY, intent=kw.get("intent", INTENT),
                        surface={"domain": "gpu kernels", "world": self.window["repo"], "tools": "shell workspace with a GPU",
                                 "data_shape": "repository", "prompt_style": "maintainer request"},
                        prompt=self.prompt or "(unauthored)", ledger=led, world_ref=f"image:{self.world['image']}",
                        checkset_ref="sha256:" + kp_hash(self.window.get("kspec") or {}), oracle_ref=f"git:{self.window['repo']}@{self.window['b']}",
                        verifier=VerifierCard(verifier_version=self.verifier_version), stats={"rung": self.window.get("rung"),
                                                                                             "milestones": len(self.milestones)},
                        lineage={"generator": "mined_migration", "a": self.window.get("a"), "b": self.window.get("b"),
                                 "simplified_from": self.window.get("simplified_from")},
                        recipe_version=kw.get("recipe_version", "k1"), status=kw.get("status", "draft"))

    def ref(self) -> LongTaskRef:
        return LongTaskRef(lane=LANE, task_id=self.task_id, source_id=str(self.window.get("source_id") or self.task_id), store=str(self.window.get("store") or ""),
                           split=str(self.window.get("split") or "train"), prompt_sha256=kp_hash(self.prompt),
                           checkset_sha256=kp_hash(self.window.get("kspec") or {}), verifier_version=self.verifier_version,
                           simplified=dict(self.window.get("simplified_from") or {}))


@dataclass(frozen=True)
class Mutant:
    id: str
    solution: str
    milestone: str
    kind: str = "revert"
    note: str = ""


def kp_hash(obj: Any) -> str:
    import hashlib

    return hashlib.sha256((obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True)).encode()).hexdigest()


# ---- the grader spec -------------------------------------------------------------------------------------------------------------
def kspec(window: Mapping[str, Any], *, legacy_dir: str | None = None) -> dict[str, Any]:
    progs = window["programs"]
    used = sorted({p for m in window["milestones"] for p in m["programs"]})
    keys = ("sources", "includes", "defines", "libs", "std", "run_dir", "cases", "timeout_s", "perf", "openmp")
    return {"task_id": window["id"], "repo": window["repo"], "kinds": sorted(window.get("graded_kinds") or [k for k in window.get("kinds") or {} if k in kp.LEGACY]),
            "programs": {p: {**{k: progs[p].get(k) for k in keys}, "fresh": True, "compare": (window.get("compare") or {}).get(p) or {}} for p in used},
            "milestones": [{"id": m["id"], "kind": m.get("kind", "run"), "program": m.get("program"), "tu": m.get("tu"), "programs": m["programs"],
                            "files": m["files"], "sanitize": bool(m.get("sanitize")), "perf": bool(m.get("perf"))} for m in window["milestones"]],
            "legacy_dir": legacy_dir or f"/opt/legacy/{window['id']}", "arch_default": "90", "perf_bars": window.get("perf_bars") or {}}


# ---- prompt and ledger ----------------------------------------------------------------------------------------------------------
def requirement_texts(window: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One requirement per milestone, in words the grader checks (V5). Compile milestone: its unit compiles for this GPU
    with CUDA 12.8 and its files no longer use the removed constructs. Run milestone: its program builds, runs on this GPU
    and produces the same results as the original."""
    kinds = [KIND_TEXT[k] for k in sorted(window.get("graded_kinds") or window.get("kinds") or {}) if k in KIND_TEXT]
    out = []
    for i, m in enumerate(window["milestones"]):
        files = ", ".join(f"`{f}`" for f in m["files"])
        if m.get("kind") == "compile":
            text = (f"`{m['tu']}` compiles with CUDA 12.8 for this GPU, and {files} no longer use{'s' if len(m['files']) == 1 else ''} "
                    f"{' or '.join(kinds) or 'the removed APIs'}.")
        else:
            main = window["programs"][m["program"]]["main"]
            text = f"The program built from `{main}` builds and runs on this GPU and produces the same results as the original code."
        out.append({"id": f"R{i + 1}", "milestone": m["id"], "text": text, "kind": m.get("kind", "run")})
    return out


def compose_prompt(opening: str, window: Mapping[str, Any], *, arch_note: bool = True) -> tuple[str, dict[str, list[int]]]:
    reqs = requirement_texts(window)
    parts = [opening.strip(), "", "What needs to work:"]
    spans: dict[str, list[int]] = {}
    text = "\n".join(parts) + "\n"
    for r in reqs:
        line = f"- {r['id']}: {r['text']}\n"
        start = len(text) + 2 + len(r["id"]) + 2
        text += line
        spans[r["id"]] = [start, start + len(r["text"])]
    if arch_note:
        text += "\n" + " ".join(LANE_SENTENCES) + "\n\n" + EPILOGUE + "\n"
    return text, spans


def instruction(prompt: str, arch: str) -> str:
    """The prompt as one World sees it: the GPU architecture of the host filled in."""
    return prompt.replace(ARCH_TOKEN, str(arch))


def ledger_of(window: Mapping[str, Any]) -> tuple[Any, list[str]]:
    from crucible.factory.cards import CheckRef, ObservationRecord, Requirement, RequirementLedger

    reqs = window.get("requirements") or []
    spans = window.get("spans") or {}
    led = RequirementLedger(requirements=[Requirement(id=r["id"], text=r["text"], kind=r.get("kind", "migration"), prompt_span=tuple(spans.get(r["id"]) or (0, 1)))
                                          for r in reqs],
                            checks=[CheckRef(id=f"check-{m['id']}", requirement_ids=[r["id"] for r in reqs if r["milestone"] == m["id"]],
                                             observation_ids=[f"obs-{window['id']}"], gating=True) for m in window["milestones"]])
    obs = ObservationRecord(id=f"obs-{window['id']}".replace("~", "-"), type="expert_end_state", pointer=f"git:{window['repo']}@{window['a']}..{window['b']}",
                            sha256=kp_hash(window.get("oracle_patch") or ""), created=window.get("b_date") or "1970-01-01T00:00:00+00:00", by=MINER)
    problems = led.problems({obs.id: obs}, prompt=window.get("prompt") or "")
    mapped = {r.get("milestone") for r in reqs}
    problems += [f"unmapped_milestone: {m['id']}" for m in window["milestones"] if m["id"] not in mapped]
    return led, problems


LEAKY = re.compile(r"\b[0-9a-f]{12,40}\b|https?://\S+|#\d{2,6}\b|Co-Authored-By:.*|Signed-off-by:.*", re.IGNORECASE)


def scrub(text: str) -> str:
    """No commit hashes, links, issue numbers or trailers in author input (the answer must not be pointed at)."""
    return LEAKY.sub("", text or "").strip()


def lint_prompt(prompt: str, window: Mapping[str, Any]) -> list[str]:
    """Deterministic S7 lint: no oracle pointers, no hint of how (the oracle's new identifiers), every program named."""
    out = []
    if LEAKY.search(prompt):
        out.append("leak: a hash, link, issue number or trailer in the prompt")
    for name in window.get("oracle_new_names") or []:
        if re.search(rf"\b{re.escape(name)}\b", prompt):
            out.append(f"hint: names the oracle's new identifier {name}")
    for p in {p for m in window["milestones"] for p in m["programs"]}:
        if window["programs"][p]["main"] not in prompt:
            out.append(f"unstated: program {window['programs'][p]['main']} is graded but not named")
    return out


def oracle_new_names(window: Mapping[str, Any]) -> list[str]:
    """Identifiers the oracle adds that the start does not have (excluding CUDA's own API names, which the request may
    use): naming them in the prompt would hand over the solution's structure."""
    added = set(re.findall(r"\b[A-Za-z_]\w{3,}\b", "\n".join(ln[1:] for ln in (window.get("oracle_patch") or "").splitlines()
                                                         if ln.startswith("+") and not ln.startswith("+++"))))
    removed = set(re.findall(r"\b[A-Za-z_]\w{3,}\b", "\n".join(ln[1:] for ln in (window.get("oracle_patch") or "").splitlines()
                                                           if ln.startswith("-") and not ln.startswith("---"))))
    code_like = re.compile(r"[a-z][A-Z]|_|\d")  # an identifier, not an English word of a comment
    return sorted(n for n in added - removed if code_like.search(n) and not n.startswith(("cuda", "__", "tex", "surf", "CUDA", "std"))
                  and not n.isupper())[:40]


async def author_opening(client: Any, window: Mapping[str, Any], *, max_tokens: int = 1500) -> str:
    """The prompt author (author family): 2-4 sentences in the maintainer's voice from the commit message and the
    program list, never the diff. The requirement list and the lane sentences are fixed text (V5)."""
    payload = {"repository": window["repo"], "commit_message": scrub(window.get("message") or window.get("subject") or ""),
               "programs": sorted({window["programs"][p]["main"] for m in window["milestones"] for p in m["programs"]}),
               "removed_constructs": [KIND_TEXT[k] for k in sorted(window.get("kinds") or {}) if k in KIND_TEXT]}
    msgs = [{"role": "system", "content": "You write the opening of a request a maintainer files for a teammate. 2 to 4 plain sentences, "
             "first person, no bullet points, no code, no headings. Say why the change is needed (the GPU code no longer builds or is no "
             "longer correct on the current CUDA toolkit and GPU) and which part of the repository it concerns. Never say how to fix it, "
             "never name new functions, types or variables, never mention commits, links, versions of this repository, tests or grading. "
             "Reply with JSON {\"opening\": \"...\"}."},
            {"role": "user", "content": json.dumps(payload)}]
    out = await client.complete_json(msgs, max_tokens=max_tokens, temperature=0.3)
    return str(out.get("opening") or "").strip()


def locate_lines(tree: Path, window: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Every line of the window's files at the start that holds a removed construct (``simplify('locate')``)."""
    rx = [(k, re.compile(kp.LEGACY[k])) for k in window.get("kinds") or {} if k in kp.LEGACY]
    out = []
    for m in window["milestones"]:
        for f in m["files"]:
            for i, ln in enumerate(_read(tree, f).splitlines(), 1):
                if any(r.search(km._strip_comment(ln)) for _, r in rx):
                    out.append({"file": f, "line": i, "text": ln.strip()})
    return out


# ---- fault-class mutants (applied to the oracle; each must fail its milestone on the GPU) ------------------------------------------
FAULTS: list[tuple[str, str, str, str]] = [
    ("addr_clamp_to_border", r"cudaAddressModeClamp", "cudaAddressModeBorder", "texture addressing clamp -> border"),
    ("addr_mirror_to_clamp", r"cudaAddressModeMirror", "cudaAddressModeClamp", "texture addressing mirror -> clamp"),
    ("addr_wrap_to_clamp", r"cudaAddressModeWrap", "cudaAddressModeClamp", "texture addressing wrap -> clamp"),
    ("filter_linear_to_point", r"cudaFilterModeLinear", "cudaFilterModePoint", "filtering linear -> point"),
    ("filter_point_to_linear", r"cudaFilterModePoint", "cudaFilterModeLinear", "filtering point -> linear"),
    ("normalized_flip", r"normalizedCoords\s*=\s*(1|true)", "normalizedCoords = 0", "normalized coordinates dropped"),
    ("readmode_flip", r"cudaReadModeNormalizedFloat", "cudaReadModeElementType", "read mode normalized -> element"),
    ("mask_active", r"\b0x[fF]{8}[uU]?\b(?=\s*,)", "__activemask()", "full shuffle mask -> __activemask()"),
    ("drop_syncwarp", r"__syncwarp\s*\(\s*[^)]*\)\s*;", ";", "a dropped __syncwarp"),
    ("shfl_down_to_up", r"__shfl_down_sync\s*\(", "__shfl_up_sync(", "shuffle direction flipped"),
]


def fault_mutants(oracle_files: Mapping[str, str | None], milestone_of: Callable[[str], str | None]) -> list[dict[str, Any]]:
    """One mutant per (fault class, file) where the class applies to the oracle's version of the file: the whole oracle with
    that file's first match rewritten. They must build (else they are trivially caught) and fail on the GPU."""
    out = []
    for fid, pat, rep, note in FAULTS:
        rx = re.compile(pat)
        for f, text in sorted(oracle_files.items()):
            if not text or not f.endswith(km.CUDA_SRC + km.CXX_SRC):
                continue
            mid = milestone_of(f)
            if mid is None or not rx.search(text):
                continue
            out.append({"id": f"fault-{fid}-{pid_of(f)}", "kind": "fault", "file": f, "milestone": mid, "note": note,
                        "content": rx.sub(rep, text, count=1)})
    return out


# ---- benchmark seeds (L5 / L4) ---------------------------------------------------------------------------------------------------
OPEN_AUDIT = ["consumers build through the candidate's own PCLConfig.cmake (flag injection; B.12)", "arch fixed at sm_120 in the plan (Marlowe is sm_90)",
              "perf bars calibrated on the 5090 only (per-GPU-model bars needed)"]
SEEDS = [  # GPU evidence: docs/modules/reports/m8_coding_gpu.json (lab 5090, kit grader, 17/17 as expected) and m7 golden for task04
    {"id": "task04_pcl_gpu_cuda12", "rung": "L5", "grader": "crucible_pack_coding.verify.task04", "scope": "whole task04 v3", "gpu_evidence": "m7 golden 7/8 graded",
     "open_audit_items": [*OPEN_AUDIT, "KinFu bar headroom 12% < 15% (B.12)", "KinFu-LS relaxed tier not stated in the prompt (P20)"]},
    {"id": "task04_span_people", "rung": "L4", "grader": "crucible_pack_coding.verify.task04", "scope": "M8 span people",
     "gpu_evidence": "solution 1, start 0, 5 unported 0, sol_v3 trap 0", "open_audit_items": list(OPEN_AUDIT)},
    {"id": "task04_span_ncv", "rung": "L4", "grader": "crucible_pack_coding.verify.task04", "scope": "M8 span ncv",
     "gpu_evidence": "solution 1, start 0, 2 unported 0, sol_v3 trap 0", "open_audit_items": list(OPEN_AUDIT)},
    {"id": "task04_span_kinfu_family", "rung": "L5", "grader": "crucible_pack_coding.verify.task04", "scope": "M8 span kinfu + kinfu_large_scale",
     "gpu_evidence": "solution 1, start 0, 2 unported 0", "open_audit_items": [*OPEN_AUDIT, "KinFu bar headroom 12% < 15% (B.12)",
                                                                             "KinFu-LS relaxed tier not stated in the prompt (P20)"]},
]
SEALED_REPOS = ("PointCloudLibrary/pcl",)  # the benchmark seeds' repository: never a training source


def window_json(w: Mapping[str, Any]) -> str:
    return json.dumps(w, indent=1, sort_keys=True, default=lambda o: asdict(o) if hasattr(o, "__dataclass_fields__") else str(o))


def run_sync(coro: Any) -> Any:
    return asyncio.run(coro)


# ---- window construction from a mined commit (S1/S2, deterministic) -------------------------------------------------------------
HARD_KINDS = ("texref", "surfref", "warp_sync", "lib_api", "volatile_warp")


def components(files: Sequence[str], programs: Mapping[str, Program], affected: Mapping[str, Sequence[str]]) -> list[list[str]]:
    """Migrated files grouped by shared programs (a header and the programs that include it are one group): each group is
    one window; independent files are independent windows."""
    parent = {f: f for f in files}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    by_main = {p.main: p for p in programs.values()}
    for f in files:
        for m in affected.get(f, []):
            prog = by_main.get(m)
            if prog is None:
                continue
            for g in files:
                if g != f and (g == m or g in prog.closure):
                    parent[find(g)] = find(f)
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(find(f), []).append(f)
    return [sorted(g) for g in groups.values()]


def oracle_patch_of(tree_a: Path, oracle_files: Mapping[str, str | None]) -> str:
    """The oracle as a patch on the start (``git diff --no-index`` per file, paths made repository-relative)."""
    import subprocess
    import tempfile

    out = []
    with tempfile.TemporaryDirectory() as td:
        for f, text in sorted(oracle_files.items()):
            a = tree_a / f
            b = Path(td) / "b" / f
            if text is not None:
                b.parent.mkdir(parents=True, exist_ok=True)
                write_raw(b, text)
            pa = str(a) if a.exists() else "/dev/null"
            pb = str(b) if text is not None else "/dev/null"
            p = subprocess.run(["git", "diff", "--no-index", "--no-color", "--", pa, pb], capture_output=True, check=False)
            d = p.stdout.decode("utf-8", "replace")  # bytes: text mode would turn CRLF sources into a patch that does not apply
            if not d:
                continue
            d = d.replace(f"a{pa}", f"a/{f}").replace(f"b{pb}", f"b/{f}").replace(f"a{pb}", f"a/{f}").replace(f"b{pa}", f"b/{f}")
            d = re.sub(r"^diff --git .*$", f"diff --git a/{f} b/{f}", d, count=1, flags=re.MULTILINE)
            d = re.sub(r"^--- (?!/dev/null).*$", f"--- a/{f}", d, count=1, flags=re.MULTILINE)
            d = re.sub(r"^\+\+\+ (?!/dev/null).*$", f"+++ b/{f}", d, count=1, flags=re.MULTILINE)
            out.append(d)
    return "".join(out)


def windows_from_commit(commit: Mapping[str, Any], tree_a: Path, tree_b: Path, *, max_programs: int = 3) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Windows of one mined commit, and the reasons for what was dropped. Programs are discovered at A (the candidate's
    start); the oracle of a window is A plus B's version of the window's migrated files."""
    drops: list[dict[str, Any]] = []
    files_a = set(files_of(tree_a))
    files = [f for f in commit["migrated_files"] if f in files_a and f.endswith(km.CUDA_SRC + km.CXX_SRC)]
    if not files:
        return [], [{"commit": commit["sha"], "reason": "no migrated source exists at A"}]
    roots = sorted({"/".join(f.split("/")[:2]) if f.count("/") >= 2 else km.module_of(f) for f in files})
    srcs = [f for f in sorted(files_a) if f.endswith((".cu", ".cpp", ".cc", ".c", ".cxx")) and any(r == "." or f.startswith(r + "/") or km.module_of(f) == r for r in roots)]
    mains = [f for f in srcs if MAIN.search(km.strip_c_light(_read(tree_a, f)))][:80]
    if not mains:  # a library: its consumers are the repository's own tests, examples and samples
        cons = [f for f in sorted(files_a) if f.split("/", 1)[0].lower() in CONSUMER_DIRS and f.endswith((".cu", ".cpp", ".cc"))]
        cons = sorted((f for f in cons if MAIN.search(km.strip_c_light(_read(tree_a, f)))), key=lambda f: (tree_a / f).stat().st_size)
        mains = cons[:60]
    programs = discover_programs(tree_a, mains, all_files=files_a)
    affected = {f: [p.main for p in programs.values() if f in p.closure] for f in files}
    out = []
    for comp in components(files, programs, affected):
        sub_aff = {f: affected.get(f, []) for f in comp}
        sub_mains = sorted({m for f in comp for m in sub_aff[f]})
        sub_progs = {p.id: p for p in programs.values() if p.main in sub_mains}
        ms = build_milestones(comp, sub_aff, sub_progs)
        runs = [m for m in ms if m["kind"] == "run"]
        if len(runs) > max_programs:  # keep the smallest consumers (a library can have dozens): fewer, cheaper observations
            size = {pid: sum(len(_read(tree_a, s)) for s in sub_progs[pid].sources) for pid in sub_progs}
            keep = {m["id"] for m in sorted(runs, key=lambda m: size[m["program"]])[:max_programs]}
            need = {r for m in runs if m["id"] in keep for r in m["requires"]}
            ms = [m for m in ms if (m["kind"] == "run" and m["id"] in keep) or (m["kind"] == "compile" and m["id"] in need)]
        for m in ms:
            m["programs"] = [p for p in m["programs"] if any(r["program"] == p for r in ms if r["kind"] == "run")][:max_programs]
        if not ms:
            needs = sorted({n for p in sub_progs.values() for n in p.needs})
            drops.append({"commit": commit["sha"], "files": comp, "reason": f"no buildable program ({', '.join(needs) or 'none found'})"})
            continue
        used = sorted({p for m in ms for p in m["programs"]})
        # the oracle: B's version of the migrated files and of every other file the commit changed that these programs
        # depend on (a migration commit often fixes a helper header too); nothing else of B
        clos = {f for p in used for f in sub_progs[p].closure}
        dirs = {km.module_of(sub_progs[p].main) for p in used}
        extra = {f for f in commit.get("changed_files") or [] if f.endswith(km.CUDA_SRC + km.CXX_SRC)
                 and (f in clos or (not (tree_a / f).exists() and km.module_of(f) in dirs))}
        oracle_files = {f: read_raw(tree_b / f) if (tree_b / f).is_file() else None
                        for f in sorted({f for m in ms for f in m["files"]} | extra)}
        short = commit["sha"][:7]
        wid = f"k-{commit['repo'].split('_', 1)[-1].lower()[:24]}-{short}" + (f"-{pid_of(comp[0])[-28:]}" if len(comp) < len(files) else "")
        w = {"id": wid, "repo": commit["repo"], "a": commit["parent"], "b": commit["sha"], "b_date": commit["date"], "subject": commit["subject"],
             "licence": commit.get("licence"), "kinds": {k: v for k, v in commit["kinds"].items() if not k.startswith("_")},
             "migrated_files": comp, "programs": {p: asdict(sub_progs[p]) for p in used}, "milestones": ms, "oracle_files": oracle_files,
             "split": "train"}
        w["oracle_patch"] = oracle_patch_of(tree_a, oracle_files)
        w["oracle_new_names"] = oracle_new_names(w)
        out.append(w)
    return out, drops
