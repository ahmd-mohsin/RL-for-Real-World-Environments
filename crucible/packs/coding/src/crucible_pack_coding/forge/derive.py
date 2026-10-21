"""G-span and G-trap for the coding seeds (DESIGN §7.3-7.4; research m7-m8/02 §12.10).

**G-span (task04 v3, H4 -> H2/H3).** One expert end state yields module-scoped tasks: the start state is the base plus
the oracle's changes to every *other* module (they are already migrated), the goal is the module's own changes, and
the hidden checks are task04's behavioural suites restricted to the module (``verify/task04.py`` scopes) plus the
static texture-reference ban on its files. Mined milestones over the module's diff are emitted for Φ (dense credit),
never as gates: M6 showed gold-token milestones are brittle across correct implementations.

**G-trap (every seed with a recorded frontier failure).** The M4 ``coding/agent_run`` mapper turns each hidden gate a
frontier model failed into a signal. The trap's area is the files both the frontier patch and the oracle touch; the
(wrong, corrected) pair is (frontier's version, oracle's version) of those files. C2 (``defect_to_check``) derives
static checks that separate them and pass the expert end state, and the frontier's version becomes the trap-reversion
negative, citing the signal (A8). The instruction states the property the gate grades, never the location.

Each derived task is written as a directory: ``instruction.md``, ``start.patch`` (base -> start), ``solution/solution.patch``
(start -> oracle), ``crucible/{task.yaml, card.yaml, checks.json, milestones.yaml}``, ``crucible/negatives/*.patch`` +
``negatives.json``. CPU acceptance runs here; behavioural acceptance runs in the verifier image (``forge_gpu.py``).
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.milestone import MilestoneSet
from crucible.core.schema.task import CrucibleTask, Difficulty, ForgeRecord, Grounding, Negative, Splits, TaskCard
from crucible.core.store import Store
from crucible.credit.operators.defect_to_check import from_correction
from crucible.forge.schema import CheckSpec, NegativeSpec, StatedProperty
from crucible.signals.extract import extract
from crucible.workflow.filestate import FileState
from crucible.workflow.milestones.evaluate import FileStateView, evaluate
from crucible.workflow.milestones.mine import from_diff
from crucible_pack_coding import PACK_ROOT
from crucible_pack_coding.m56.task_properties import TASKS, base_reader
from crucible_pack_coding.verify.task04 import SCOPES, scan_scoped

CRUCIBLE = PACK_ROOT.parent.parent
HIST = CRUCIBLE / "data" / "m7" / "history"
OUT = CRUCIBLE / "data" / "forge" / "coding"
M3 = CRUCIBLE / "data" / "m3" / "store"


# ------------------------------------------------------------------------------------------------ patch utilities
def split_patch(patch: str) -> dict[str, str]:
    parts = re.split(r"(?m)^(?=diff --git )", patch)
    out = {}
    for p in parts:
        m = re.match(r"diff --git a/(\S+) b/(\S+)", p)
        if m:
            out[m.group(2)] = p if p.endswith("\n") else p + "\n"
    return out


def join(files: dict[str, str], keep: Callable[[str], bool]) -> str:
    return "".join(v for k, v in sorted(files.items()) if keep(k))


def state(base: Callable[[str], str | None], *patches: str) -> FileState:
    fs = FileState(base=base)
    for p in patches:
        if p.strip():
            fs.apply_patch(p)
    return fs


class _CodeOnly:
    """A file state read with comments and docstrings stripped (C-family and Python), for C2 separation: a check that
    separates two versions only through their comments grades words, not code (M8 audit V12: trap gates required the
    absence of ``defines`` and ``schemas`` from a module docstring)."""

    def __init__(self, fs: FileState):
        self._fs = fs

    def __getattr__(self, k: str) -> Any:
        return getattr(self._fs, k)

    def read(self, path: str) -> str | None:
        from crucible.verify.runtime.patterns.static import C_LIKE, PY, strip_c, strip_py

        t = self._fs.read(path)
        if t is None:
            return None
        suf = "." + path.rsplit(".", 1)[-1] if "." in path else ""
        return strip_c(t) if suf in C_LIKE else strip_py(_drop_docstrings(t)) if suf in PY else t


def _drop_docstrings(t: str) -> str:
    """Blank every string-literal statement (module, class and function docstrings): prose, not code."""
    import ast

    try:
        tree = ast.parse(t)
    except (SyntaxError, ValueError):
        return t
    lines = t.split("\n")
    for n in ast.walk(tree):
        if isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            for i in range(n.lineno - 1, (n.end_lineno or n.lineno)):
                lines[i] = ""
    return "\n".join(lines)


def unified(path: str, a: str | None, b: str | None) -> str:
    """A git-style unified diff of one file between two contents (None = absent)."""
    if a == b:
        return ""
    al = (a or "").splitlines(keepends=True)
    bl = (b or "").splitlines(keepends=True)
    for lines in (al, bl):
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
    head = f"diff --git a/{path} b/{path}\n" + ("new file mode 100644\n" if a is None else "") + ("deleted file mode 100644\n" if b is None else "")
    body = "".join(difflib.unified_diff(al, bl, fromfile="/dev/null" if a is None else f"a/{path}", tofile="/dev/null" if b is None else f"b/{path}", n=3))
    return head + body if body else ""


def diff_states(paths: list[str], a: FileState, b: FileState) -> str:
    return "".join(unified(p, a.read(p), b.read(p)) for p in sorted(paths))


def oracle_patch(task: str) -> str:
    return (TASKS / task / "solution" / "solution.patch").read_text(errors="replace")


# ------------------------------------------------------------------------------------------------ signals (M4)
def run_signals() -> tuple[dict[tuple[str, str], list[dict]], set[str]]:
    """(task, run) -> failing-gate signals from the M4 agent_run mapper; plus every resolvable id."""
    ts = TraceStore(Store(M3))
    out: dict[tuple[str, str], list[dict]] = {}
    known: set[str] = set()
    for tid in sorted(i for i in ts.ids() if i.startswith("run:")):
        t = extract(ts.get(tid))
        for s in t.signals:
            known.add(s.id)
            if s.polarity < 0 and s.meta.get("gate"):
                out.setdefault((s.meta["task"], s.meta["run"]), []).append({"id": s.id, "gate": s.meta["gate"], "model": s.meta.get("model"), "trace": tid})
    return out, known


# ------------------------------------------------------------------------------------------------ G-span (task04)
SPAN_TEXT = {
    "people": ("Bring PCL's GPU people detector (gpu/people, including the vendored NCV / NPP-staging primitives) to CUDA 12.8 (RTX 5090)",
               ["People detector body-part labels must match the original's exactly, run with OMP_NUM_THREADS=1.",
                "The NCV / NPP-staging primitives exercised by ncv_check must produce bit-identical buffers.",
                "Several PeopleDetector instances may run at the same time from different host threads, and every result must then be bitwise identical to running each alone.",
                "The people detector (PeopleDetector::process, with OMP_NUM_THREADS=1) may take at most 0.42x of the original's time."], "H2"),
    "ncv": ("Port PCL's vendored NCV / NPP-staging primitives (gpu/people/src/cuda/nvidia) to CUDA 12.8 (RTX 5090)",
            ["The NCV / NPP-staging primitives exercised by ncv_check must produce bit-identical buffers, including at image borders, for unaligned sources and for every filter mode."],
            "H2"),
    "kinfu": ("Bring PCL's GPU KinectFusion (gpu/kinfu) to CUDA 12.8 (RTX 5090)",
              ["KinFu must track and reconstruct as well as the original, measured against ground truth; its TSDF volumes, meshes and extracted clouds must agree with the original's to within floating-point noise.",
               "Several KinfuTracker and MarchingCubes instances may run at the same time from different host threads, and every result must then be bitwise identical to running each alone.",
               "Per-frame KinFu processing may take at most 0.65x of the original's time, marching cubes 0.55x, zero-crossing cloud extraction 0.75x, and four KinFu trackers on four host threads (kinfu_replay --throughput --concurrent) 0.42x."],
              "H3"),
    "kinfu_family": ("Bring PCL's GPU KinectFusion (gpu/kinfu) and its large-scale fork (gpu/kinfu_large_scale) to CUDA 12.8 (RTX 5090)",
                     ["KinFu and KinFu large scale must track and reconstruct as well as the originals, measured against ground truth; their TSDF volumes, meshes and extracted clouds must agree with the originals' to within floating-point noise.",
                      "Several KinfuTracker, kinfuLS::KinfuTracker and MarchingCubes instances may run at the same time from different host threads, and every result must then be bitwise identical to running each alone.",
                      "Per-frame KinFu processing may take at most 0.65x of the original's time, marching cubes 0.55x, zero-crossing cloud extraction 0.75x, four KinFu trackers on four host threads (kinfu_replay --throughput --concurrent) 0.42x, and per-frame KinFu large-scale processing 0.66x."],
                     "H3"),
    "kinfu_large_scale": ("Bring PCL's GPU KinectFusion large scale (gpu/kinfu_large_scale) to CUDA 12.8 (RTX 5090)",
                          ["KinFu large scale must track and reconstruct as well as the original, measured against ground truth, to within floating-point noise.",
                           "Several kinfuLS::KinfuTracker instances may run at the same time from different host threads, and every result must then be bitwise identical to running each alone.",
                           "Per-frame KinFu large-scale processing may take at most 0.66x of the original's time."], "H3"),
}
COMMON = ["pcl configures and builds with CUDA 12.8 for sm_120 with the module enabled by default, and all its device code (SASS and PTX) targets sm_120.",
          "No texture or surface references remain in the module; you may use texture objects or other memory paths.",
          "compute-sanitizer must report nothing on the module's workloads with --tool racecheck, --tool synccheck and --tool memcheck."]
GATE_OF = {"pcl configures": ["build", "binary", "consumers"], "No texture": ["static"], "compute-sanitizer": ["sanitizer"], "labels must match": ["people"],
           "bit-identical": ["ncv"], "track and reconstruct": ["kinfu", "fresh"], "bitwise identical to running each alone": ["concurrency"],
           "of the original's time": ["perf"]}


def properties_for(sentences: list[str]) -> list[StatedProperty]:
    return [StatedProperty(id=f"p{i}", text=s, kind="coding") for i, s in enumerate(sentences)]


def seed_gate_kind(task: str, gate: str) -> str:
    """The kind the seed grader declares for ``gate`` (M7 R6: the A1 gold check fails a check list that disagrees with the
    grade). A gate no kit grader declares is one of the seed's own tests (``test.sh``), an outcome."""
    import importlib

    mod = {"task02_pcops_triton": "task02", "task03_mdlite_pallas": "task03", "task04_pcl_gpu_cuda12": "task04"}.get(task, "task01")
    return importlib.import_module(f"crucible_pack_coding.verify.{mod}").KINDS.get(gate, "outcome")


def suite_checks(scope: str, props: list[StatedProperty], provenance: dict[str, str]) -> list[CheckSpec]:
    from crucible_pack_coding.verify import task04

    sc = SCOPES[scope]
    measured = {"static", "build", "binary", "consumers", "sanitizer"} | ({"fresh"} if scope != "full" else set())
    measured |= {"kinfu"} if scope in ("kinfu", "kinfu_large_scale", "kinfu_family") else set()
    measured |= {"people"} if sc["people"] else set()
    measured |= {"ncv"} if sc["ncv"] else set()
    measured |= {"concurrency"} if sc["people_threads"] or scope in ("kinfu", "kinfu_large_scale", "kinfu_family") else set()
    measured |= {"perf"} if sc["perf"] else set()
    out = []
    for g in [x for x in task04.GATES if x in measured]:
        states = [p.id for p in props if any(g in gates for key, gates in GATE_OF.items() if key in p.text)]
        if g == "fresh":
            states = [p.id for p in props if "bit-identical" in p.text or "labels must match" in p.text or "track and reconstruct" in p.text]
        out.append(CheckSpec(id=f"suite/{g}", gate=g, gate_kind=task04.KINDS[g], kind="suite" if g != "static" else "static", states=states,
                             provenance=provenance.get(g, "reference_differential:original build"), spec={"scope": scope}))
    return out


# sibling forks stay unported in a span's start: KinFu-LS is a fork of KinFu, so the expert's port of one is most of the other
SIBLINGS = {"kinfu": ("gpu/kinfu_large_scale",), "kinfu_large_scale": ("gpu/kinfu",)}
MAX_SIBLING_OVERLAP = 0.2
SPAN_SCOPES = ("people", "ncv", "kinfu_family")
# PCL components -> source roots, for the consumer closure below
COMPONENT_ROOTS = {"gpu_containers": "gpu/containers", "gpu_utils": "gpu/utils", "gpu_kinfu": "gpu/kinfu", "gpu_kinfu_large_scale": "gpu/kinfu_large_scale",
                   "gpu_people": "gpu/people"}


def consumer_needs() -> dict[str, set[str]]:
    """Source roots each pristine consumer of the grader needs installed (its ``find_package(PCL ... COMPONENTS ...)``)."""
    out = {}
    for c in ("kinfu_replay", "people_eval"):
        f = TASKS / "task04_pcl_gpu_cuda12" / "repos" / c / "CMakeLists.txt"
        m = re.search(r"find_package\(PCL[^)]*COMPONENTS([^)]*)\)", f.read_text()) if f.exists() else None
        out[c] = {COMPONENT_ROOTS[x] for x in (m.group(1).split() if m else []) if x in COMPONENT_ROOTS}
    return out


def closure_violations(scope: str, needs: dict[str, set[str]] | None = None) -> list[str]:
    """A span must be closed under the grader's dependencies: every root a consumer the grader always builds needs is
    either ported in the span's start or part of the span. A sibling left unported in both breaks the consumer, so the
    span's oracle can never pass ``consumers`` (m13: the KinFu / KinFu-LS spans)."""
    needs = consumer_needs() if needs is None else needs
    unported = set(SIBLINGS.get(scope, ()))
    return [f"{c} needs {r}, which the {scope} span neither ports in its start (an unported sibling) nor includes" for c, rs in sorted(needs.items())
            for r in sorted(rs & unported)]


def alternative_admissible(alt_patch: str, files: dict[str, str], inside) -> tuple[bool, list[str]]:
    """A recorded alternative is evidence about a span only if it depends on the same outside state as the span's start:
    its changes outside the span must equal the oracle's (an alternative built on its own helper changes, e.g. a
    different TextureBinder API, cannot build against the oracle's helpers in the span start)."""
    alt = split_patch(alt_patch)
    diff = sorted(p for p in set(alt) | set(files) if not inside(p) and alt.get(p) != files.get(p) and p in alt)
    return not diff, diff


async def derive_span(scope: str, files: dict[str, str], base, signals: list[dict]) -> dict[str, Any]:
    roots = SCOPES[scope]["roots"]

    def inside(p: str) -> bool:
        return any(p.startswith(r + "/") or p == r for r in roots)

    def sibling(p: str) -> bool:
        return any(p.startswith(r + "/") for r in SIBLINGS.get(scope, ()))

    start_patch = join(files, lambda p: not inside(p) and not sibling(p))
    agent_oracle = join(files, inside)
    start = state(base, start_patch)
    oracle = state(base, start_patch, agent_oracle)
    title, specific, horizon = SPAN_TEXT[scope]
    sentences = COMMON + specific
    props = properties_for(sentences)
    trap_sigs = [s for s in signals if s["gate"] in ("correctness/ncv", "ncv")] if scope in ("ncv", "people") else []
    prov = {"ncv": trap_sigs[0]["id"]} if trap_sigs else {}
    checks = suite_checks(scope, props, prov)
    # mined milestones for Φ (dense credit), filtered: false at the start, true at the oracle
    cands = [c.milestone for c in from_diff(agent_oracle, start.read, include=inside, max_per_file=6)]
    at_s, at_o = await evaluate(cands, FileStateView(start)), await evaluate(cands, FileStateView(oracle))
    ms = [m.model_copy(update={"states": [props[1].id]}) for m in cands if at_s[m.id] == 0.0 and at_o[m.id] == 1.0]
    negs: list[NegativeSpec] = [NegativeSpec(id="start", task_id=scope, kind="floor", patch="", expected_gate="static", description="the span's start state")]
    span_files = split_patch(agent_oracle)
    from crucible.verify.runtime.patterns.static import strip_c
    from crucible_pack_coding.verify.task04 import TEXREF

    texref = [p for p in sorted(span_files) if any(re.search(rx, strip_c(base(p) or "")) for rx in TEXREF)]
    for k, path in enumerate(texref[:6]):  # defect mutants: the oracle with one module file left on texture references
        if True:
            negs.append(NegativeSpec(id=f"unported_{k}", task_id=scope, kind="defect_mutant", patch=join(span_files, lambda p, path=path: p != path),
                                     expected_gate="static", description=f"{path} left on texture references"))
    tasks_extra: dict[str, Any] = {}
    if trap_sigs:
        tasks_extra = await trap_from_run("task04_pcl_gpu_cuda12", "sol_v3", inside, base, start_patch, files, trap_sigs[0], gate="ncv")
        if tasks_extra.get("negative"):
            negs.append(tasks_extra["negative"])  # caught by the span's behavioural gate (the frontier failed it in history)
            ms += [m.model_copy(update={"states": [props[1].id]}) for m in tasks_extra.get("milestones", [])]
    astra = HIST / "task04" / "patches" / "astra_v3b.patch"
    excluded: dict[str, list[str]] = {}
    admissible, outside_diff = alternative_admissible(astra.read_text(errors="replace"), files, inside) if astra.exists() else (False, [])
    if astra.exists() and not admissible:
        excluded["astra_v3_module"] = outside_diff[:20]
    if astra.exists() and admissible:
        alt = join(split_patch(astra.read_text(errors="replace")), inside)
        kind, exp, gate = ("valid_alternative", "pass", None) if scope in ("people", "ncv") else ("defect_mutant", "fail", "perf")
        negs.append(NegativeSpec(id="astra_v3_module", task_id=scope, kind=kind, patch=alt, expected=exp, expected_gate=gate,
                                 provenance="run:task04_pcl_gpu_cuda12/gpt-6-astra-v3-run2",
                                 description="GPT 6 Astra's v3 port of the module: correct everywhere, slower than the KinFu bars" if gate else
                                 "GPT 6 Astra's v3 port of the module: a different correct implementation (passes every people/NCV gate in history)"))
    return {"scope": scope, "title": title, "horizon": horizon, "instruction": render_instruction(title, sentences), "properties": props,
            "checks": checks, "milestones": ms, "negatives": negs, "start_patch": start_patch, "oracle_patch": agent_oracle, "span_files": sorted(span_files),
            "trap": tasks_extra.get("trap"), "provenance": [s["id"] for s in trap_sigs], "c2": tasks_extra.get("c2"),
            "start": start, "oracle": oracle, "excluded_alternatives": excluded, "closure_violations": closure_violations(scope)}


def render_instruction(title: str, sentences: list[str]) -> str:
    return (f"# {title}\n\n`/workspace/pcl` is the Point Cloud Library, with the rest of its GPU stack already migrated to "
            "CUDA 12.8. The pristine downstream tools `kinfu_replay` and `people_eval` must keep working unchanged; they are rebuilt from their original "
            "sources against your build. Results are compared with the original pre-CUDA-12 code, run on this GPU through PTX JIT.\n\n## Requirements\n\n"
            + "\n".join(f"- {s}" for s in sentences) + "\n\n## Grading\n\nThe grader applies your diff of `/workspace/pcl`, builds it from scratch with CUDA 12.8 "
            "(sm_120), builds the pristine tools against it and runs hidden tests covering every requirement above, including workloads drawn at "
            "grading time. Timing is measured by the tools and, for whole runs, by the grader. All of it must pass.\n")


# ------------------------------------------------------------------------------------------------ G-trap (C2 on a frontier failure)
async def trap_from_run(task: str, run_patch: str, inside: Callable[[str], bool], base, start_patch: str, oracle_files: dict[str, str], sig: dict,
                        *, gate: str) -> dict[str, Any]:
    """C2 on (frontier version, oracle version) of the area: static checks that separate them, and the frontier's
    version (rebased onto the trap task's start state) as the trap-reversion negative."""
    hist = {"task04_pcl_gpu_cuda12": HIST / "task04" / "patches" / f"{run_patch}.patch"}.get(task)
    wrong_files = split_patch((hist or (TASKS / task / "runs" / run_patch / "model.patch")).read_text(errors="replace"))
    area = sorted(p for p in set(wrong_files) & set(oracle_files) if inside(p))
    if not area:
        return {}
    start = state(base, start_patch)
    wrong_state = state(base, start_patch, join(wrong_files, lambda p: p in area))
    oracle_state = state(base, start_patch, join(oracle_files, lambda p: p in area or inside(p)))
    correction = diff_states(area, wrong_state, oracle_state)
    cand = await from_correction(sig["id"], correction, wrong_state.read, end_state=oracle_state.read)
    # C2 checks are Φ milestones, never gates (DESIGN §7.3; M8 audit V12/V13: as gates they graded the oracle's own
    # tokens and docstring words, and no behavioural grader ran them); only those that still separate the oracle from
    # the frontier version with comments and docstrings stripped are kept
    on_o = await evaluate(list(cand.checks), FileStateView(_CodeOnly(oracle_state)))
    on_w = await evaluate(list(cand.checks), FileStateView(_CodeOnly(wrong_state)))
    separating = [m for m in cand.checks if on_o.get(m.id) == 1.0 and on_w.get(m.id) == 0.0]
    milestones = separating[:8]
    dropped = [m.id for m in cand.checks if m not in separating]
    neg = NegativeSpec(id=f"trap_{run_patch}", task_id=task, kind="trap_reversion", patch=diff_states(area, start, wrong_state), expected_gate=gate,
                       provenance=sig["id"], description=f"{sig['model']}'s version of {', '.join(area)} (it failed the hidden {sig['gate']} gate)")
    return {"milestones": milestones, "negative": neg, "trap": f"a property the {sig['gate']} gate grades, which {sig['model']} broke",
            "c2": {"proposed": cand.evidence.get("proposed"), "separating": cand.evidence.get("separating"), "kept": len(milestones),
                   "dropped_comment_only": dropped[:20], "area": area}}


SEED_RUNS = {"task03_mdlite_pallas": "gpt-5.6-sol-run1", "task02_pcops_triton": "gpt-5.6-sol-run1", "task01v2_tessera_omp": "gpt-5.6-sol-run1",
             "task04_pcl_gpu_cuda12": "gpt-5.6-sol-v3-run1"}
SRC = re.compile(r"\.(py|cu|cuh|cpp|cc|h|hpp|c|cxx)$")
TEST = re.compile(r"(^|/)(tests?|testing|docs?)/|(^|/)test_[^/]*\.py$|_test\.py$|conftest\.py$")


def trap_property(instruction: str, gate: str) -> StatedProperty:
    """The seed's stated requirement its ``gate`` grades: the first requirement line that names the gate, else the whole
    requirements section (every line of it is stated)."""
    section = instruction.split("## Requirements", 1)[-1].split("\n## ", 1)[0]
    req = [ln.strip().lstrip("-* ").strip() for ln in section.splitlines() if ln.strip()]
    hit = next((x for x in req if re.search(rf"\b{re.escape(gate)}", x, re.IGNORECASE)), None)
    return StatedProperty(id=f"trap_{gate}", kind="coding", text=hit or " ".join(section.split()))


async def derive_seed_trap(task: str, signals: dict[tuple[str, str], list[dict]]) -> dict[str, Any]:
    """A trap task for a seed: the area the frontier failed in, everything else of the oracle already applied."""
    run = SEED_RUNS.get(task)
    sigs = signals.get((task, run), []) if run else []
    if not sigs:
        return {"task": task, "skipped": "no recorded frontier failure for this seed (its only frontier run passed)"}
    base = base_reader(task)
    files = split_patch(oracle_patch(task))
    run_dir = TASKS / task / "runs" / run.replace("-run1", "-run1")
    wrong_path = run_dir / "model.patch" if run_dir.exists() else None
    if task == "task04_pcl_gpu_cuda12":
        wrong_files = split_patch((HIST / "task04" / "patches" / "sol_v3.patch").read_text(errors="replace"))
    elif wrong_path and wrong_path.exists():
        wrong_files = split_patch(wrong_path.read_text(errors="replace"))
    else:
        cand = sorted((TASKS / task / "runs").glob("*/model.patch"))
        if not cand:
            return {"task": task, "skipped": "frontier patch not found"}
        wrong_files = split_patch(cand[0].read_text(errors="replace"))
    gate = next((s["gate"] for s in sigs if s["gate"] != "perf"), sigs[0]["gate"])
    sig = next(s for s in sigs if s["gate"] == gate)
    area = sorted(p for p in set(wrong_files) & set(files) if SRC.search(p) and not TEST.search(p))
    if task == "task04_pcl_gpu_cuda12":
        area = [p for p in area if p.startswith("gpu/people/src/cuda/nvidia/")]
    if not area:
        return {"task": task, "skipped": "the frontier patch and the oracle share no source file"}
    inside = lambda p: p in area
    start_patch = join(files, lambda p: not inside(p))
    res = await trap_from_run(task, "sol_v3" if task == "task04_pcl_gpu_cuda12" else run, inside, base, start_patch, files, sig, gate=gate)
    return {"task": task, "signal": sig, "gate": gate, "area": area, "start_patch": start_patch, "oracle_patch": join(files, inside), **res}


# ------------------------------------------------------------------------------------------------ CPU acceptance
async def cpu_accept_span(d: dict[str, Any], known: set[str]) -> dict[str, Any]:
    roots = SCOPES[d["scope"]]["roots"]
    base = base_reader("task04_pcl_gpu_cuda12")
    rows = {}

    def tree_static(fs: FileState) -> bool:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for p in d["span_files"]:
                txt = fs.read(p)
                if txt is not None:
                    (root / p).parent.mkdir(parents=True, exist_ok=True)
                    (root / p).write_text(txt)
            return scan_scoped(root, roots)["pass"]

    rows["A1_static_oracle"] = tree_static(d["oracle"])
    rows["A2_static_start_fails"] = not tree_static(d["start"])
    neg_rows = {}
    for n in d["negatives"]:
        try:
            fs = state(base, d["start_patch"], n.patch or "")
        except Exception as ex:  # noqa: BLE001 -- a negative that does not apply is recorded, never counted as killed
            neg_rows[n.id] = {"applies": False, "error": str(ex)[:200]}
            continue
        entry = {"applies": True, "kind": n.kind, "expected_gate": n.expected_gate}
        if n.expected_gate == "static":
            entry["static_fails"] = not tree_static(fs)
        trap_checks = [c for c in d["checks"] if c.kind == "milestone"]
        if trap_checks:
            from crucible.core.schema.milestone import Milestone

            ms = [Milestone.model_validate(c.spec["milestone"]) for c in trap_checks]
            ph = await evaluate(ms, FileStateView(fs))
            entry["c2_checks_pass"] = all(v == 1.0 for v in ph.values())
        neg_rows[n.id] = entry
    if any(c.kind == "milestone" for c in d["checks"]):
        from crucible.core.schema.milestone import Milestone

        ms = [Milestone.model_validate(c.spec["milestone"]) for c in d["checks"] if c.kind == "milestone"]
        rows["A1_c2_oracle"] = all(v == 1.0 for v in (await evaluate(ms, FileStateView(d["oracle"]))).values())
    from crucible.world.leaks import leak_tokens

    props = {p.id for p in d["properties"]}
    graded = {s for c in d["checks"] for s in c.states}
    rows["A5_instruction_clean"] = not leak_tokens(d["instruction"])
    base_lines = {ln.strip() for p in d["span_files"] for ln in (base(p) or "").splitlines()}
    novel = {ln[1:].strip() for ln in d["oracle_patch"].splitlines() if ln.startswith("+") and not ln.startswith("+++") and len(ln.strip()) > 40} - base_lines
    span_start = {ln.strip() for p in d["span_files"] for ln in (d["start"].read(p) or "").splitlines()}
    rows["A5_no_oracle_lines_in_span_start"] = not (novel & span_start)
    rows["A5_no_hidden_paths_in_start"] = not any(re.search(r"(^|/)(tests/hidden|solution)/|grade\.py$|fingerprints\.json$", p) for p in split_patch(d["start_patch"]))
    others = {ln[1:].strip() for v in split_patch(d["start_patch"]).values() for ln in v.splitlines() if ln.startswith("+")}
    sibling = len(novel & others)
    rows["A5_start_does_not_reveal_via_siblings"] = sibling <= MAX_SIBLING_OVERLAP * max(1, len(novel))
    rows["closure_consumers"] = not d.get("closure_violations")
    rows["A7_all_properties_graded"] = props <= graded
    rows["A7_no_unstated_gate"] = all(c.states for c in d["checks"])  # no milestone exemption (M8 audit V8)
    rows["A8_provenance"] = all(c.provenance for c in d["checks"])
    rows["A8_trap_signal_resolves"] = (not d["trap"]) or all(s in known for s in d["provenance"])
    ok_negs = all(r.get("applies") and r.get("static_fails", True) and not (r.get("c2_checks_pass") and r["kind"] == "trap_reversion")
                  for r in neg_rows.values() if r.get("kind") != "valid_alternative")
    rows["A3_cpu_negatives"] = ok_negs
    return {"cpu": rows, "negatives": neg_rows, "cpu_ok": all(rows.values()), "closure_violations": d.get("closure_violations"),
            "excluded_alternatives": d.get("excluded_alternatives"),
            "sibling_overlap": {"novel_oracle_lines": len(novel), "also_in_sibling_modules_of_the_start": sibling,
                                "note": "the start state holds the expert's port of sibling modules; similar code there weakens asymmetry 1 (a difficulty metric, not hidden material)"},
            "gpu_pending": ["A1 behaviour (3 grades in the verifier image)", "A2 start/naive behaviour", "A3 negatives at their behavioural gates",
                            "valid alternative passes behaviourally"]}


def write_task(d: dict[str, Any], out: Path, *, family: str, seed_id: str, derived_from: str, generator: str) -> Path:
    t = out / d["scope"] if "scope" in d else out
    (t / "crucible" / "negatives").mkdir(parents=True, exist_ok=True)
    (t / "solution").mkdir(exist_ok=True)
    (t / "instruction.md").write_text(d["instruction"])
    (t / "start.patch").write_text(d["start_patch"])
    (t / "solution" / "solution.patch").write_text(d["oracle_patch"])
    (t / "crucible" / "checks.json").write_text(json.dumps([c.model_dump(mode="json") for c in d["checks"]], indent=1) + "\n")
    if d.get("milestones"):
        (t / "crucible" / "milestones.yaml").write_text(yaml.safe_dump(MilestoneSet(family=family, milestones=d["milestones"]).model_dump(mode="json"), sort_keys=False))
    negs = []
    for n in d["negatives"]:
        (t / "crucible" / "negatives" / f"{n.id}.patch").write_text(n.patch or "")
        negs.append(Negative(id=n.id, path=f"crucible/negatives/{n.id}.patch", expected_gate=n.expected_gate, kind=n.kind, expected=n.expected,
                             provenance=n.provenance, description=n.description))
    task = CrucibleTask(id=f"coding/{family}/{seed_id}@1", family=family, horizon=d["horizon"], world="../../../../packs/coding/families/legacy_gpu_migration/seeds/task04_pcl_gpu_cuda12/crucible/world.yaml",
                        milestones="crucible/milestones.yaml" if d.get("milestones") else None, negatives=negs, derived_from=derived_from)
    card = TaskCard(id=task.id, family=family, horizon=d["horizon"], title=d["title"],
                    grounding=Grounding(traces=list(d.get("provenance") or []), oracle_source="expert_end_state",
                                        checks_with_provenance=f"{sum(1 for c in d['checks'] if c.provenance)}/{len(d['checks'])}"),
                    splits=Splits(showcase=True), forge=ForgeRecord(generator=generator, knobs={"scope": d.get("scope")}, seed_ref=derived_from, state="constructed"),
                    difficulty=Difficulty(traps=1 if d.get("trap") else 0))
    (t / "crucible" / "task.yaml").write_text(yaml.safe_dump(task.model_dump(mode="json", exclude_none=True), sort_keys=False))
    (t / "crucible" / "card.yaml").write_text(yaml.safe_dump(card.model_dump(mode="json", exclude_none=True), sort_keys=False))
    return t


async def main_async() -> dict[str, Any]:
    signals, known = run_signals()
    base = base_reader("task04_pcl_gpu_cuda12")
    files = split_patch(oracle_patch("task04_pcl_gpu_cuda12"))
    t4sigs = signals.get(("task04_pcl_gpu_cuda12", "gpt-5.6-sol-v3-run1"), [])
    spans = {}
    for scope in SPAN_SCOPES:
        d = await derive_span(scope, files, base, t4sigs)
        write_task(d, OUT / "task04_spans", family="legacy_gpu_migration", seed_id=f"task04_span_{scope}", derived_from="task04_pcl_gpu_cuda12@v3",
                   generator="span" if not d["trap"] else "span+trap")
        acc = await cpu_accept_span(d, known)
        spans[scope] = {"title": d["title"], "horizon": d["horizon"], "start_files": len(split_patch(d["start_patch"])), "span_files": len(d["span_files"]),
                        "checks": [c.id for c in d["checks"]], "milestones_for_phi": len(d["milestones"]), "negatives": {n.id: n.kind for n in d["negatives"]},
                        "trap": d["trap"], "trap_signals": d["provenance"], "c2": d.get("c2"), **acc}
    traps = {}
    for task in ("task01_prims_omp", "task01v2_tessera_omp", "task02_pcops_triton", "task03_mdlite_pallas", "task04_pcl_gpu_cuda12"):
        r = await derive_seed_trap(task, signals)
        if "skipped" in r or not r.get("negative"):
            traps[task] = {k: v for k, v in r.items() if k in ("skipped", "area", "gate", "c2")} or {"skipped": "C2 found no separating check"}
            continue
        base_t = base_reader(task)
        start = state(base_t, r["start_patch"])
        oracle = state(base_t, r["start_patch"], r["oracle_patch"])
        wrong = state(base_t, r["start_patch"], r["negative"].patch)
        ms = list(r.get("milestones") or [])
        ph_o, ph_w, ph_s = [await evaluate(ms, FileStateView(x)) for x in (oracle, wrong, start)]
        traps[task] = {"signal": r["signal"]["id"], "gate": r["gate"], "model": r["signal"]["model"], "area": r["area"], "c2": r["c2"],
                       "milestones_for_phi": [m.id for m in ms], "oracle_passes": all(v == 1.0 for v in ph_o.values()) and bool(ms),
                       "frontier_version_fails": any(v == 0.0 for v in ph_w.values()), "start_fails": any(v == 0.0 for v in ph_s.values()),
                       "signal_resolves": r["signal"]["id"] in known}
        traps[task]["cpu_ok"] = traps[task]["oracle_passes"] and traps[task]["frontier_version_fails"] and traps[task]["signal_resolves"]
        # the trap task states the seed's requirements (M8 audit V12: the instruction was empty and A7 failed), and its gate is
        # the seed grader's behavioural gate the frontier failed; the C2 milestones are dense credit only
        instruction = (TASKS / task / "instruction.md").read_text()
        prop = trap_property(instruction, r["gate"])
        gate_check = CheckSpec(id=f"trap/{r['gate']}", gate=r["gate"], gate_kind=seed_gate_kind(task, r["gate"]), kind="suite", states=[prop.id], provenance=r["signal"]["id"],
                               spec={"seed_gate": r["gate"], "grader": task})
        traps[task]["instruction_states_gate"] = " ".join(prop.text.split()) in " ".join(instruction.split())
        d = {"scope": "trap", "title": f"{task}: the {r['gate']} trap", "horizon": "H2", "instruction": instruction, "properties": [prop],
             "checks": [gate_check], "milestones": ms, "negatives": [r["negative"]], "start_patch": r["start_patch"], "oracle_patch": r["oracle_patch"],
             "provenance": [r["signal"]["id"]], "trap": r["trap"]}
        write_task(d, OUT / f"{task}_trap", family="legacy_gpu_migration", seed_id=f"{task}_trap", derived_from=f"{task}", generator="trap")
    rep = {"gate": "M8 coding: span + trap tasks (CPU acceptance; behavioural acceptance queued for the verifier image)", "spans": spans, "traps": traps}
    rep["span_cpu_ok"] = all(v["cpu_ok"] for v in spans.values())
    rep["traps_cpu_ok"] = {k: v.get("cpu_ok", False) for k, v in traps.items()}
    return rep


def main() -> None:
    rep = asyncio.run(main_async())
    (CRUCIBLE / "docs" / "modules" / "reports" / "m8_coding_derive.json").write_text(json.dumps(rep, indent=1, default=str) + "\n")
    print(json.dumps({"span_cpu_ok": rep["span_cpu_ok"], "spans": {k: (v["cpu_ok"], v["cpu"], v["trap"] is not None) for k, v in rep["spans"].items()},
                      "traps": rep["traps"]}, indent=1, default=str)[:6000])


if __name__ == "__main__":
    main()
