"""The kernel debugging lane (swarm-pipeline/10): long-horizon CUDA kernel debugging tasks as ``LongTask`` environments.

Three families, one task shape (a start tree that is a real fixed release with real changes taken out, hidden checks
on the end state, a GPU grader that owns the build):

- ``kernel-bug`` (intent KD1): 2-4 real CUDA kernel bug fixes of one backend (races, divergent barriers, out-of-bounds
  access, index overflow, grid limits, strides, launch configurations, stream misuse) reverted together at B, the
  newest of them; faults connected through shared ops or files. The agent gets a user's symptom report;
- ``kernel-perf`` (KD2): 2-3 real kernel speed-ups reverted at B: correctness still passes, a per-GPU perf bar fails;
- ``kernel-migration`` (KD3): several real toolkit-migration commits reverted at B (``kernel_debug_migrate``: CUDA 12 ->
  13 API removals, CUDA 11 -> 12 texture/warp changes regrouped into multi-program tasks).

**The record** (``store/<split>/<id>/record.json``) is built deterministically (``synthesize``): A' = B with each fault's
backend change reversed (newest first, 3-way through a private index, ``generators.base.apply_chain``) and the cases the
fixes added to the repository's op test removed from the agent's copy; the synthetic history A' -> fault 1 -> ... -> B'
is written under ``refs/crucible/kernel-debug/<id>`` of a derived clone (the source clone is never written) and B' == B
is checked. One milestone per fault; its hidden checks are the fault's ops' cases of B's harness (pooled on the CPU by
executing the harness, ``kernel_debug_world``) under the fault class's detectors (``kernel_debug_mine.CLASSES``).
Which cases a fault breaks, and the executed ``requires`` edges between faults, are measured on the GPU (the bundle's
calibration: B passes, B with only fault k reverted fails milestone k's cases, A' + fault k alone passes or fails the
others); until then edges are *claimed* and S6 is ``pending_gpu``.

**The task** (``KernelDebugTask``) grades through a GPU backend running ``kernel_debug_plan`` in the verifier image;
with no GPU backend every grade is masked ``no_gpu`` (V8), never a 0. Prompts are written by the builder
(``BUILDER``, recorded), symptom-only, and linted (``lint_prompt``): no faulty function, kernel or file names, no hidden
case strings, no hashes, links, PR numbers or versions.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from crucible.factory.lanes import kernel as K
from crucible.factory.lanes import kernel_debug_ops as ko
from crucible.factory.lanes.kernel_debug_mine import CLASSES, SINCE, KernelChange
from crucible.factory.lanes.kernel_harness import HARNESS
from crucible.factory.langs import cuda
from crucible.factory.longtask import GroundCase, LongTaskRef, new_file_patch, status_claim_patch
from crucible.factory.pathcheck import PathRules

LANE = "kernel_debug"
BUILDER = "claude-code:opus-5.5"
STUDENT = "Qwen/Qwen3.6-35B-A3B"
FAMILIES = {"bug": "kernel-bug", "perf": "kernel-perf", "migration": "kernel-migration"}
INTENTS = {"bug": "KD1", "perf": "KD2", "migration": "KD3"}
SEALED_REPOS = ("ikawrakow/ik_llama.cpp",)  # the held-out benchmark's kernel source: never a training source
TRAIN_REPOS = ("ggml-org/llama.cpp",)
PRUNE = ("models", "docs", "media")  # large non-code trees left out of the World and the grader's trees
REF = "refs/crucible/kernel-debug"
ARCH_TOKEN = "{ARCH}"
PERF_TOKEN = "{PERF_BAR}"
HEADING = "What I am seeing:"
LANE_SENTENCES = (
    f"Only CUDA 12.8 is installed; the GPU is `sm_{ARCH_TOKEN}`.",
    "The work has to run on the GPU: an op the CUDA backend stops supporting, or computes on the host, counts as broken.",
    "Only your changes under `ggml/src/ggml-cuda/` are kept; changes anywhere else (tests, build files, other backends) are discarded.",
)
EPILOGUE = ("How this works: you are working in a sandboxed copy of the repository with no network access. After you finish, the CUDA "
            "backend is rebuilt from your tree with the grader's own build command for this GPU, and the repository's op tests (as they are "
            "in the fixed release, which can include cases you do not have) run on inputs drawn at grading time. Each case passes when the "
            "GPU result matches the CPU reference within that case's own tolerance; ops whose problem is a race, a barrier or a memory "
            "access are also run under compute-sanitizer, and a speed requirement is timed against a reference build on the same GPU. "
            "Trying to download code or reading version-control history beyond the current commit scores the run 0. Before you finish, "
            "write STATUS.json at the repository root mapping each requirement id above to your probability (0 to 1) that it is met. If you "
            "decide you cannot finish, end your final message with a line 'GIVE_UP: <reason>'; your work is graded as it stands.")


# ---- synthesis: A'..B' ---------------------------------------------------------------------------------------------------------
def backend_paths(ch: KernelChange | Mapping[str, Any], layout: ko.Layout, *, with_harness: bool) -> list[str]:
    files = ch.files if isinstance(ch, KernelChange) else ch["files"]
    return [f for f in files if layout.is_backend(f) or (with_harness and f == HARNESS)]


@dataclass
class Synth:
    ok: bool
    faults: list[KernelChange]
    dropped: list[dict[str, str]] = field(default_factory=list)
    a: str = ""
    b: str = ""
    b_prime: str = ""
    chain: list[str] = field(default_factory=list)
    alone: dict[str, bool] = field(default_factory=dict)  # fault sha -> its fix applies alone on A'
    revert_alone: dict[str, bool] = field(default_factory=dict)  # fault sha -> its revert applies alone at B (the revert mutant)
    reason: str = ""


def synthesize(derived: Path, faults: Sequence[Any], layout: ko.Layout | None, *, rid: str, paths_of: Any = None) -> Synth:
    """A'..B' for ``faults`` (module doc), in ``derived`` (a ``generators.base.derived_clone``). A fault whose revert
    conflicts with the later ones is dropped and the chain retried while two faults remain. ``paths_of(fault,
    with_harness)``: the paths each change reverts (default: its CUDA backend files under ``layout``, plus the op test);
    the migration family passes the commit's migrated sources. A fault is anything with ``sha``, ``parent``, ``date``,
    ``subject`` (and optionally ``body``)."""
    from crucible.factory.generators import base as gb

    if paths_of is None:
        def paths_of(f: Any, with_harness: bool) -> list[str]:
            assert layout is not None
            return backend_paths(f, layout, with_harness=with_harness)

    faults = sorted(faults, key=lambda c: c.date)
    dropped: list[dict[str, str]] = []
    while len(faults) >= 2:
        b = faults[-1].sha
        steps = [(gb.full_patch(derived, f.parent, f.sha, paths_of(f, True)), True) for f in reversed(faults)]
        trees, bad = gb.apply_chain(derived, b, steps)
        if bad is None:
            break
        drop = list(reversed(faults))[bad]
        dropped.append({"sha": drop.sha, "why": "its revert conflicts with the fixes after it at B"})
        faults = [f for f in faults if f.sha != drop.sha]
    else:
        return Synth(False, list(faults), dropped, reason="fewer than two faults revert together")
    b = faults[-1].sha
    tb = gb.tree_of(derived, b)
    forward = [*reversed(trees), tb]  # A', A'+f1, ..., B
    msgs = [f"kernel-debug start: {len(faults)} changes reverted"] + [f"{f.subject}\n\n{getattr(f, 'body', '')}" for f in faults]
    dates = [faults[-1].date] + [f.date for f in faults]
    shas = gb.write_chain(derived, forward[:1] + forward[1:], messages=msgs, dates=dates, parent=b, ref=f"{REF}/{rid}")
    a, b_prime = shas[0], shas[-1]
    alone, revert_alone = {}, {}
    for f in faults:
        fwd = gb.full_patch(derived, f.parent, f.sha, paths_of(f, False))
        _t, bad = gb.apply_chain(derived, a, [(fwd, False)])
        alone[f.sha] = bad is None
        _t, bad = gb.apply_chain(derived, b, [(fwd, True)])
        revert_alone[f.sha] = bad is None
    ok = gb.tree_of(derived, b_prime) == tb
    return Synth(ok, list(faults), dropped, a=a, b=b, b_prime=b_prime, chain=shas, alone=alone, revert_alone=revert_alone,
                 reason="" if ok else "B' differs from B")


def export(derived: Path, rev: str, dest: Path) -> Path:
    """``rev`` exported to ``dest`` without history and without ``PRUNE`` (the grader's and the World's trees)."""
    from crucible.factory.lanes import kernel_world as kw

    kw.export_tree(derived, rev, dest)
    for p in PRUNE:
        shutil.rmtree(Path(dest) / p, ignore_errors=True)
    return Path(dest)


def diff_patch(derived: Path, a: str, b: str, paths: Sequence[str]) -> str:
    """``a -> b`` restricted to ``paths``, as a patch on the start tree (the oracle, a mutant)."""
    from crucible.factory.generators import base as gb

    return gb.git(derived, "diff", "--full-index", "--binary", a, b, "--", *paths).stdout.decode("utf-8", "replace")


def tree_patch(derived: Path, a: str, steps: Sequence[tuple[bytes, bool]], paths: Sequence[str]) -> str | None:
    """The patch from ``a`` to ``a`` with ``steps`` applied (None when a step does not apply)."""
    from crucible.factory.generators import base as gb

    trees, bad = gb.apply_chain(derived, a, steps)
    if bad is not None:
        return None
    return gb.git(derived, "diff", "--full-index", "--binary", gb.tree_of(derived, a), trees[-1], "--", *paths).stdout.decode("utf-8", "replace")


# ---- units the CPU gates compile ---------------------------------------------------------------------------------------------------
INCLUDE = re.compile(r'^\s*#\s*include\s*"([^"]+)"', re.MULTILINE)


def includers(tree: Path, layout: ko.Layout) -> dict[str, set[str]]:
    """backend file -> the ``.cu`` units that include it (directly or through other backend headers)."""
    files = [p.relative_to(tree).as_posix() for p in (Path(tree) / layout.cuda_dir).rglob("*") if p.is_file() and p.suffix in (".cu", ".cuh", ".h")]
    if (Path(tree) / layout.dispatch_file).is_file():
        files.append(layout.dispatch_file)
    inc: dict[str, set[str]] = {}
    names = {f: f for f in files}
    for f in files:
        base = f.rsplit("/", 1)[0]
        for h in INCLUDE.findall((Path(tree) / f).read_text(errors="replace")):
            cand = [f"{base}/{h}", f"{layout.cuda_dir}{h}", f"{layout.cuda_dir}{h.split('/')[-1]}"]
            for c in cand:
                c = "/".join(p for p in c.split("/") if p not in ("", "."))
                while "/../" in c:
                    c = re.sub(r"[^/]+/\.\./", "", c, count=1)
                if c in names:
                    inc.setdefault(f, set()).add(c)
                    break
    units = [f for f in files if f.endswith(".cu")]
    out: dict[str, set[str]] = {}
    for u in units:
        seen, frontier = {u}, [u]
        while frontier:
            nxt = []
            for x in frontier:
                for y in inc.get(x, ()):
                    if y not in seen:
                        seen.add(y)
                        nxt.append(y)
            frontier = nxt
        for h in seen:
            out.setdefault(h, set()).add(u)
    return out


def units_for(tree: Path, files: Sequence[str], layout: ko.Layout, *, per_file: int = 1) -> list[str]:
    """The cheapest units (smallest source, template instances last) that compile each changed file: a ``.cu`` changed
    file is its own unit; a header is compiled through ``per_file`` of its includers."""
    inc = includers(tree, layout)
    out: list[str] = []
    for f in files:
        if f.endswith(".cu"):
            out.append(f)
            continue
        cands = sorted(inc.get(f, ()), key=lambda u: ("template-instances" in u, (Path(tree) / u).stat().st_size))
        out += cands[:per_file]
    return list(dict.fromkeys(out))


# ---- milestones and the record ---------------------------------------------------------------------------------------------------
def milestones_of(faults: Sequence[KernelChange], pools: Mapping[str, Sequence[str]], added: Mapping[str, Sequence[str]]) -> list[dict[str, Any]]:
    """One milestone per fault (date order): its ops, the CPU-executed case pool of those ops, the cases the fix added,
    its checks (golden always; the class's sanitizer tools; the perf bar for a speed-up), claimed interactions."""
    ms = []
    for i, f in enumerate(sorted(faults, key=lambda c: c.date), 1):
        dets = list(f.detectors)
        checks = ["golden"] + [d for d in dets if d in ("racecheck", "synccheck", "memcheck")] + (["perf"] if "perf_bar" in dets else [])
        mid = f"m{i}"
        ms.append({"id": mid, "kind": f.kind, "fault": f.sha, "class": f.fault_class, "ops": list(f.ops), "files": [x for x in f.files if x != HARNESS],
                   "functions": list(f.functions), "kernels": list(f.kernels), "case_pool": list(pools.get(f.sha) or []),
                   "cases_added": list(added.get(f.sha) or []), "repeat": 5 if "golden_repeat" in dets else 1, "checks": checks,
                   "tests": [f"{mid}:{c}" for c in checks], "requires": [], "requires_evidence": {}, "claimed_requires": []})
    by = {m["fault"]: m for m in ms}
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            if set(a["ops"]) & set(b["ops"]) or set(a["files"]) & set(b["files"]):
                b["claimed_requires"].append(a["id"])  # the later fix's cases may need the earlier one (direction measured on the GPU)
    return [by[f.sha] for f in sorted(faults, key=lambda c: c.date)]


def record_id(kind: str, repo: str, faults: Sequence[KernelChange]) -> str:
    short = re.sub(r"[^a-z0-9]+", "", repo.split("/")[-1].lower())[:14]
    h = hashlib.sha256("|".join(sorted(f.sha for f in faults)).encode()).hexdigest()[:8]
    return f"kd-{kind}-{short}-{h}"


def contamination(faults: Sequence[KernelChange], since: str = SINCE) -> str:
    return "post-release" if all(f.date[:10] >= since for f in faults) else "fallback"


# ---- the grader spec -------------------------------------------------------------------------------------------------------------
def kdspec(rec: Mapping[str, Any]) -> dict[str, Any]:
    """What ``kernel_debug_plan`` reads (no oracle, no fault identities beyond the milestone ids)."""
    return {"task_id": rec["id"], "family": rec["family"], "layout": rec["layout"], "harness": {"file": HARNESS, "dialect": rec["harness"]["dialect"]},
            "arch_default": "90", "keep_paths": [rec["layout"]["cuda_dir"]] + ([rec["layout"]["dispatch_file"]] if not rec["layout"]["dispatch_file"].startswith(rec["layout"]["cuda_dir"]) else []),
            "milestones": [{"id": m["id"], "kind": m["kind"], "checks": m["checks"], "cases": list(m.get("cases") or m.get("cases_added") or []) ,
                            "case_pool": list(m.get("case_pool") or []), "ops": m["ops"], "repeat": m.get("repeat", 1),
                            "perf_cases": list(m.get("perf_cases") or [])} for m in rec["milestones"]],
            "seeds": {"fixed": [7, 123456], "fresh": 1}, "perf_bars": rec.get("perf_bars") or {}, "perf": {"r": 3, "rounds": 3}}


# ---- the task ------------------------------------------------------------------------------------------------------------------
class GpuBackend(Protocol):
    async def grade(self, rec: Mapping[str, Any], patch: str) -> dict[str, Any]: ...


@dataclass
class KernelDebugTask:
    """A stored kernel-debugging record as a ``LongTask`` (``longtask.check_longtask`` conformance)."""

    rec: dict[str, Any]
    backend: GpuBackend | None = None
    verifier_version: str = "kernel-debug-v1"
    generator = LANE

    @property
    def window(self) -> dict[str, Any]:  # the name the factory's handlers read
        return self.rec

    @property
    def task_id(self) -> str:
        return str(self.rec["id"])

    @property
    def world(self) -> dict[str, Any]:
        w = self.rec.get("world") or {}
        return {"kind": "kernel_workspace", "repos": [self.rec["repo"]], "image": w.get("image", "crucible-lh-k:base-12.8.1"), "digest": w.get("digest", ""),
                "base_dir": w.get("base_dir") or self.rec.get("tree_a") or "", "workdir": f"/workspace/{self.rec['repo'].split('/')[-1]}", "network": "none",
                "history": "cut", "gpu": True, "arch_env": "CRUCIBLE_CUDA_ARCH"}

    @property
    def prompt(self) -> str:
        return str(self.rec.get("prompt") or "")

    @property
    def requirements(self) -> list[dict[str, Any]]:
        return list(self.rec.get("requirements") or [])

    @property
    def milestones(self) -> list[dict[str, Any]]:
        return [{"id": m["id"], "kind": m["kind"], "tests": list(m["tests"]), "requires": list(m.get("requires") or []),
                 "requires_evidence": dict(m.get("requires_evidence") or {}), "files": list(m["files"]), "ops": list(m["ops"])} for m in self.rec["milestones"]]

    @property
    def path_rules(self) -> PathRules:
        return PathRules(grader_paths=(HARNESS,), step_budget=self.rec.get("step_budget"))

    @property
    def oracle(self) -> str:
        return self.oracle_patch()

    @property
    def provenance(self) -> list[str]:
        return [f"obs-{self.task_id}"]

    def oracle_patch(self) -> str:
        return str(self.rec.get("oracle_patch") or "")

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None) -> K.GradeReport:
        ids = [m["id"] for m in self.rec["milestones"]]
        if self.backend is None:
            return K.GradeReport(0.0, {"status": "masked", "failure_kind": "no_gpu", "reason": "no GPU backend configured"}, {"phi": None, "milestones": None})
        raw = await self.backend.grade(self.rec, str(patch or ""))
        infra = raw.get("infra") or {"status": "ok"}
        if infra.get("status") == "masked":
            return K.GradeReport(0.0, infra, {"phi": None, "milestones": None}, raw)
        # milestone gates are outcome gates: the kit reports them in report.json's gate statuses (``gate_status``) and as
        # ``check/<gate>`` components of the reward document, never in its hard ``gates``
        gates = {g: ("pass" if float(v) == 1.0 else "fail") for g, v in (raw.get("components") or {}).items() if g.startswith("check/")}
        gates = {g.removeprefix("check/"): v for g, v in gates.items()}
        gates.update({g: ("pass" if s == "pass" else "fail") for g, s in (raw.get("gate_status") or {}).items()})
        reward, phi, status = K.lane_reward(gates, ids)
        return K.GradeReport(reward, {"status": "ok", "failure_kind": None, "reason": None},
                             {"phi": phi, "milestones": status, "full_pass": all(status.values()) if status else False, "path": {"violations": []}}, raw)

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        rep = await self.grade_patch(patch)
        if rep.infra.get("status") != "ok":
            raise RuntimeError(f"milestones unmeasured: {rep.infra}")
        return dict(rep.meta["milestones"])

    def floors(self) -> list[GroundCase]:
        lay = self.rec["layout"]
        test_edit = new_file_patch("tests/crucible_floor_test.cpp", "int main() { return 0; }\n")
        build_edit = new_file_patch("cmake/crucible-floor.cmake", "set(CMAKE_CUDA_ARCHITECTURES 52)\n")
        cases = [GroundCase("no_op", "floor", "", note="the start state"), GroundCase("claims_only", "floor", status_claim_patch(self), note="STATUS.json only"),
                 GroundCase("tests_only", "floor", test_edit, note="an edit outside the CUDA backend (discarded by the grader)"),
                 GroundCase("build_only", "floor", build_edit, note="a build-file edit lowering the arch (discarded)")]
        for f in self.rec.get("floor_patches") or []:
            cases.append(GroundCase(f["id"], "floor", f["patch"], note=f.get("note", "")))
        del lay
        return cases

    def mutants(self) -> list[K.Mutant]:
        return [K.Mutant(m["id"], m["patch"], m["milestone"], m.get("kind", "revert"), m.get("note", "")) for m in self.rec.get("mutants") or []
                if m.get("patch") is not None]

    def alternatives(self) -> list[GroundCase]:
        return [GroundCase(a["id"], "alternative", a["patch"]) for a in self.rec.get("alternatives") or []]

    def simplify(self, kind: str, **kw: Any) -> KernelDebugTask:
        """``locate``: the prompt names each milestone's changed files and functions (the faulty-function counterfactual,
        ``00`` §7.4). ``prerequisites``: the faults whose milestones others require (executed edges, or ``pre``) are
        pre-applied: their milestones leave the task."""
        r = json.loads(json.dumps(self.rec))
        if kind == "locate":
            lines = "\n".join(f"- [{m['id']}] look at {', '.join('`' + f + '`' for f in m['files'])}: {', '.join(m['functions'][:6])}" for m in r["milestones"])
            r["prompt"] = f"{r.get('prompt', '')}\n\nWhere the problems are:\n{lines}\n"
        elif kind == "prerequisites":
            pre = set(kw.get("pre") or {q for m in r["milestones"] for q in m.get("requires") or [] if (m.get("requires_evidence") or {}).get(q) == "executed"})
            if not pre:
                raise ValueError("no prerequisite milestone (no executed edge yet: pass pre=)")
            r["start_patch_milestones"] = sorted(pre)
            r["milestones"] = [{**m, "requires": [q for q in m.get("requires") or [] if q not in pre]} for m in r["milestones"] if m["id"] not in pre]
            r["requirements"] = [q for q in r.get("requirements") or [] if q.get("milestone") not in pre]
        else:
            raise ValueError(f"kernel debugging tasks simplify by locate or prerequisites, not {kind!r}")
        r["simplified_from"] = {"task": self.task_id, "kind": kind}
        r["id"] = f"{self.task_id}~{kind}"
        return KernelDebugTask(r, self.backend, self.verifier_version)

    def card(self, **kw: Any) -> Any:
        from crucible.factory.cards import RequirementLedger, TaskCard, VerifierCard

        led = ledger_of(self.rec)[0] if self.requirements else RequirementLedger()
        fam = self.rec["family"]
        return TaskCard(id=self.task_id.replace("~", "-"), family=fam, intent=kw.get("intent", self.rec.get("intent_id") or "KD1"),
                        surface={"domain": "gpu kernels", "world": self.rec["repo"], "tools": "shell workspace with a GPU", "data_shape": "repository",
                                 "prompt_style": "bug report" if fam != FAMILIES["migration"] else "maintainer request"},
                        prompt=self.prompt or "(unauthored)", ledger=led, world_ref=f"image:{self.world['image']}", checkset_ref="sha256:" + K.kp_hash(kdspec(self.rec)),
                        oracle_ref=f"git:{self.rec['repo']}@{self.rec['b']}", verifier=VerifierCard(verifier_version=self.verifier_version),
                        stats={"faults": len(self.rec["milestones"]), "split": self.rec.get("split")},
                        lineage={"generator": "kernel_rebreak", "b": self.rec["b"], "faults": [m["fault"] for m in self.rec["milestones"]],
                                 "prompt_builder": (self.rec.get("prompt_provenance") or {}).get("builder"), "simplified_from": self.rec.get("simplified_from")},
                        recipe_version=kw.get("recipe_version", "kd1"), status=kw.get("status", "draft"))

    def ref(self) -> LongTaskRef:
        return LongTaskRef(lane=LANE, task_id=self.task_id, source_id=self.task_id, store=str(self.rec.get("store") or ""), split=str(self.rec.get("split") or "train"),
                           prompt_sha256=K.kp_hash(self.prompt), checkset_sha256=K.kp_hash(kdspec(self.rec)), verifier_version=self.verifier_version,
                           simplified=dict(self.rec.get("simplified_from") or {}))


# ---- prompts: the builder's symptom report, composed, linted, ledgered -------------------------------------------------------------
LEAKY = re.compile(r"\b(?=[0-9a-f]*[a-f])(?=[0-9a-f]*\d)[0-9a-f]{7,40}\b|https?://\S+|#\d{2,6}\b|\bPR\s*\d+|Co-Authored-By:.*|Signed-off-by:.*|\brevert\w*", re.IGNORECASE)
VERSION = re.compile(r"\bb\d{4,5}\b|\bv?\d+\.\d+\.\d+\b|\brelease\s+\d+")


def compose(authored: Mapping[str, Any], rec: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]], dict[str, list[int]]]:
    """Prompt, requirements (one per milestone, ``R<i>``) and spans from the builder's ``{opening, requirements:
    {mid: text}, closing}``; the lane sentences and the harness note are fixed text (V5)."""
    ids = [m["id"] for m in rec["milestones"]]
    reqs = [{"id": f"R{i}", "milestone": mid, "text": " ".join(str((authored.get("requirements") or {}).get(mid) or "").split()), "kind": "behaviour"}
            for i, mid in enumerate(ids, 1)]
    text = str(authored.get("opening") or "").strip() + f"\n\n{HEADING}\n"
    spans: dict[str, list[int]] = {}
    for r in reqs:
        line = f"- [{r['id']}] "
        spans[r["id"]] = [len(text) + len(line), len(text) + len(line) + len(r["text"])]
        text += line + r["text"] + "\n"
    closing = str(authored.get("closing") or "").strip()
    if closing:
        text += f"\n{closing}\n"
    text += "\n" + " ".join(LANE_SENTENCES) + "\n\n" + EPILOGUE + "\n"
    return text, reqs, spans


def hidden_strings(rec: Mapping[str, Any]) -> set[str]:
    """Strings the prompt must not contain: the faulty functions and kernels, the changed files (path and stem), the
    hidden cases' full names and their parameter strings."""
    out: set[str] = set()
    # an op's own name is what a user sees (``ARGSORT``, ``SOFT_MAX``): a file named after its op is not a hint by name
    public = {o.lower() for m in rec["milestones"] for o in m.get("ops") or []} | {o.lower().replace("_", "") for m in rec["milestones"] for o in m.get("ops") or []}
    for m in rec["milestones"]:
        out |= {n for n in [*(m.get("functions") or []), *(m.get("kernels") or [])] if len(n) >= 4 and n.lower() not in public}
        for f in m.get("files") or []:
            out |= {f, f.rsplit("/", 1)[-1]}
            stem = f.rsplit("/", 1)[-1].rsplit(".", 1)[0]
            if len(stem) >= 5 and stem not in ("ggml-cuda", "common") and stem.lower() not in public and stem.lower().replace("-", "") not in public:
                out.add(stem)
        for c in m.get("cases_added") or []:
            out.add(c)
            inner = c[c.find("(") + 1:-1] if "(" in c else ""
            out |= {p for p in inner.split(",") if len(p) >= 12}
    return out


def lint_prompt(text: str, rec: Mapping[str, Any]) -> list[str]:
    out = [f"lint leak: {m.group(0)!r} (hash, link, PR number, trailer or a word that names the construction)" for m in LEAKY.finditer(text)]
    out += [f"lint version: {m.group(0)!r}" for m in VERSION.finditer(text)]
    low = text.lower()
    for s in sorted(hidden_strings(rec)):
        if re.search(rf"(?<![\w.-]){re.escape(s.lower())}(?![\w-])", low):
            out.append(f"lint hidden: {s!r} (a faulty function, kernel, changed file or hidden case)")
    return out


def ledger_of(rec: Mapping[str, Any]) -> tuple[Any, list[str]]:
    """The two-way requirement ledger (``kernel.ledger_of`` on this record's fields)."""
    view = {"id": rec["id"], "repo": rec["repo"], "a": rec["a"], "b": rec["b"], "b_date": rec.get("b_date"), "oracle_patch": rec.get("oracle_patch") or "",
            "requirements": rec.get("requirements") or [], "spans": rec.get("spans") or {}, "prompt": rec.get("prompt") or "", "milestones": rec["milestones"]}
    return K.ledger_of(view)


def author(rec: dict[str, Any], authored: Mapping[str, Any]) -> dict[str, Any]:
    """Compose, lint and ledger the builder's report into ``rec`` (``prompt``, ``requirements``, ``spans``, ``lint``,
    ``prompt_provenance``); returns the findings (empty = publishable)."""
    problems: list[str] = []
    if not str(authored.get("builder") or ""):
        problems.append("provenance: no builder recorded")
    ids = [m["id"] for m in rec["milestones"]]
    got = authored.get("requirements") or {}
    problems += [f"unknown_requirement_id: {k}" for k in got if k not in ids]
    problems += [f"unmapped_milestone: {k}" for k in ids if not str(got.get(k) or "").strip()]
    prompt, reqs, spans = compose(authored, rec)
    authored_text = "\n".join([str(authored.get("opening") or ""), str(authored.get("closing") or ""), *[str(v) for v in got.values()]])
    problems += lint_prompt(authored_text, rec)
    rec.update(prompt=prompt, requirements=reqs, spans=spans)
    _led, led_problems = ledger_of(rec)
    problems += led_problems
    rec["lint"] = {"findings": problems}
    rec["prompt_provenance"] = {"builder": str(authored.get("builder") or ""), "kind": "symptom report", "sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                                "states_only": "symptoms and goals; no fix, faulty names, files or hidden case names (lint_prompt)"}
    return {"findings": problems}


def instruction(prompt: str, arch: str, perf_bar: str | None = None) -> str:
    out = prompt.replace(ARCH_TOKEN, str(arch))
    return out.replace(PERF_TOKEN, perf_bar) if perf_bar else out


# ---- intent (00 §7: the task fails for the intended reason) -------------------------------------------------------------------
SKILLS: dict[str, dict[str, str]] = {
    "kernel_fault_localization": {
        "behaviour": "finds the faulty kernels behind op-level symptoms in a large CUDA backend, across files, without a failing test named",
        "failure_mode": "edits the op's dispatch or the CPU side, or patches near the symptom instead of the faulty kernel",
        "evidence": "files the agent edited vs the milestone's fault files (lineage, never shown); milestone vector per checkpoint"},
    "gpu_concurrency": {
        "behaviour": "reasons about shared memory, barriers and warp synchronisation; uses compute-sanitizer to confirm a race is gone",
        "failure_mode": "adds a barrier in the wrong place (or inside divergent code), or 'fixes' a race by serialising work",
        "evidence": "compute-sanitizer racecheck/synccheck runs in the trail before and after the edit; the sanitizer check's verdict"},
    "index_arithmetic": {
        "behaviour": "finds 32-bit overflow, grid-dimension limits and wrong strides that only show on large or non-contiguous tensors",
        "failure_mode": "tests only the small default shapes, or clamps sizes instead of fixing the index math",
        "evidence": "harness runs with large or permuted shapes in the trail; golden failures only on the large cases"},
    "interacting_faults": {
        "behaviour": "keeps going after a correct fix whose op still fails because a second fault in the same op masks it",
        "failure_mode": "reverts a correct fix because the op still fails, or stops after the first fault",
        "evidence": "a milestone that passed and was lost in the checkpoint series; the dependent milestone never passes; STATUS.json vs the grade"},
    "reproduce_with_op_tests": {
        "behaviour": "turns the report into op-test or sanitizer reproductions (test-backend-ops -o OP) and re-runs them after each change",
        "failure_mode": "edits without ever running the op test on the GPU, or claims fixes it never exercised",
        "evidence": "test-backend-ops / compute-sanitizer commands before the first edit and after the last; Brier score of STATUS.json"},
    "perf_regression_diagnosis": {
        "behaviour": "finds why a kernel got slower (launch configuration, memory access pattern, a lost fast path) and restores the speed "
                     "without breaking results",
        "failure_mode": "micro-optimises unrelated code, or trades correctness for speed",
        "evidence": "timing runs (test-backend-ops perf, nsys/ncu) in the trail; the perf check vs the golden check of the same milestone"},
    "toolkit_migration": {
        "behaviour": "maps removed or changed toolkit APIs (CUDA 12 -> 13, CCCL 3, texture/warp intrinsics) to their replacements "
                     "across several programs while keeping exact results",
        "failure_mode": "silences the compiler (lower arch, casts, stubs) or changes semantics (memory location, rounding, warp masks)",
        "evidence": "compile errors read in the trail; per-program golden results; static scan of removed constructs"},
}


def task_intent(rec: Mapping[str, Any]) -> dict[str, Any]:
    """The task's intent card (``00`` §7), derived from the record (faults, classes, ops, claimed and executed edges),
    never from the prompt."""
    ms = list(rec["milestones"])
    fam = rec["family"]
    classes = sorted({m["class"] for m in ms})
    skills = ["kernel_fault_localization", "reproduce_with_op_tests"]
    if fam == FAMILIES["perf"]:
        skills = ["perf_regression_diagnosis", "kernel_fault_localization", "reproduce_with_op_tests"]
    if fam == FAMILIES["migration"]:
        skills = ["toolkit_migration", "kernel_fault_localization"]
    if set(classes) & {"race", "divergent_barrier", "stream_async"}:
        skills.append("gpu_concurrency")
    if set(classes) & {"overflow", "grid_limit", "indexing", "bounds", "launch_config"}:
        skills.append("index_arithmetic")
    claimed = [(m["id"], q) for m in ms for q in m.get("claimed_requires") or []]
    executed = [(m["id"], q) for m in ms for q in m.get("requires") or [] if (m.get("requires_evidence") or {}).get(q) == "executed"]
    if claimed or executed:
        skills.append("interacting_faults")
    files = sorted({f for m in ms for f in m["files"]})
    return {"intent": INTENTS[next(k for k, v in FAMILIES.items() if v == fam)], "family": fam, "skills": skills, "faults": len(ms), "classes": classes,
            "ops": sorted({o for m in ms for o in m["ops"]}), "source_files": files,
            "why_long": (f"{len(ms)} real faults in {len(files)} files of one CUDA backend, connected through shared ops "
                         f"({', '.join(f'{a} with {b}' for a, b in claimed) or 'none claimed'}); executed edges: "
                         f"{', '.join(f'{a} needs {b}' for a, b in executed) or 'pending the GPU prober'}; each fault needs its own reproduction, "
                         "localisation in a ~100k-line backend, a fix and a re-check"),
            "milestones": [{"id": m["id"], "class": m["class"], "ops": m["ops"], "checks": m["checks"], "files": m["files"]} for m in ms],
            "expected_failures": {k: SKILLS[k]["failure_mode"] for k in skills}, "trace_evidence": {k: SKILLS[k]["evidence"] for k in skills},
            "attribution_rule": ("a failure counts toward a skill gap only when its trace evidence is present (S14/S15); >= 70% of failures must be so "
                                 "attributed, and the locate counterfactual (faulty files named) must succeed significantly more often"),
            "targets": {"student": STUDENT, "train_band": [0.10, 0.60], "benchmark_max": 0.30, "fairness": "a strong solver passes (S13)"}}


def intent_cards() -> list[Any]:
    from crucible.factory.cards import IntentCard

    beh = {"KD1": "debugs several interacting real CUDA kernel faults (races, barriers, bounds, index overflow, strides, launch configuration) "
                  "from op-level symptoms, on the GPU, without breaking results",
           "KD2": "finds and undoes real kernel performance regressions while results stay exact, judged on the grading GPU",
           "KD3": "migrates several programs across a toolkit generation (CUDA 12 -> 13, CUDA 11 -> 12) where APIs were removed, keeping exact results"}
    fail = {"KD1": "fixes one fault and stops, edits the wrong kernel, or masks a race by serialising work",
            "KD2": "changes unrelated code, or makes it fast by computing something else",
            "KD3": "silences the compiler (stubs, lower arch) or changes semantics of the replaced API"}
    return [IntentCard(id=f"intent-kernel-debug-{k.lower()}", behaviour=beh[k], failure_mode=fail[k], difficulty_axes=["faults", "files", "edge depth", "op breadth"],
                       coverage={"behaviour": k, "surface": "CUDA kernels"}) for k in ("KD1", "KD2", "KD3")]


def save(store: Path, rec: Mapping[str, Any]) -> Path:
    d = Path(store) / rec["split"] / rec["id"]
    d.mkdir(parents=True, exist_ok=True)
    p = d / "record.json"
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str) + "\n")
    tmp.replace(p)
    return p


def load(store: Path, split: str, rid: str) -> dict[str, Any]:
    return json.loads((Path(store) / split / rid / "record.json").read_text())


def records(store: Path, split: str | None = None) -> list[dict[str, Any]]:
    out = []
    for p in sorted(Path(store).glob(f"{split or '*'}/*/record.json")):
        out.append(json.loads(p.read_text()))
    return out


def run_sync(coro: Any) -> Any:
    return asyncio.run(coro)


def class_text(cls: str) -> str:
    return str((CLASSES.get(cls) or {}).get("text") or cls)


def changed_names_of(rec: Mapping[str, Any]) -> list[str]:
    return sorted({n for m in rec["milestones"] for n in m.get("functions") or []})


def kernels_named(texts: Sequence[str]) -> list[str]:
    return cuda.kernels_of(texts)
