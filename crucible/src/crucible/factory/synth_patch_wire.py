"""The debug lane bound to ``crucible.synth.patch`` (part2-paper-core §11, M5; the evidence doc §12 "the debug lane as
one plug-in domain"). Synth never imports factory; this module is the one place the two meet.

- ``open_pool(root)``: the lane over a debug root's **training** store (``WindowStore`` refuses a held-out store; any
  path under ``data/sealed`` is refused before anything is read). Tasks are the lane's own ``DebugTask``s with their
  authored bug reports, built with no clone (the oracle is the window's ``oracle_files``).
- ``lane_roles(task)``: the path classifier from the task's language profile and the lane's protected-path rule
  (``executable.is_protected`` under the profile's scope) plus the hidden test files: every path the lane protects is
  ``test``, ``golden``, ``plumbing`` or ``config``; vendored dependencies and ``.git`` are forbidden too.
- ``patch_spec(task)``: the window record as a ``PatchSpec`` (milestones, the lane's checkset, gating, weights, Φ share,
  B's files, the commit-revert mutants with their executed vectors, the floors as the lane's own patches, the fix units).
- ``lane_runner(task)``: a ``ThreadedRunner`` whose grade is the lane's: ``WindowTask.grade`` on the end state's files
  (``apply_files``, protected paths dropped as the lane drops them) in the task's own World image, with the per-test
  statuses read from the lane's sandbox run; mode ``visible`` runs A′'s own suite (``run_tests`` with no hidden test
  written, S3's existing-suite run) over the files. Containers are ``crucible-lh-d-synth-*``, at most ``parallel`` at
  once, each removed by Docker on exit (``--rm``); ``remove_containers`` removes any left by name, never by prune.
- ``stored_gates`` / ``prior_of``: the lane's gate record and its executed reruns (S10, S3) as REP's prior trials.
"""

from __future__ import annotations

import contextvars
import json
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from crucible.factory.generators.base import GenLane
from crucible.factory.generators.debug import NAME, DebugTask, debug_sandboxes, scope
from crucible.factory.lanes.windows import WindowStore
from crucible.factory.langs.runtime import PREFIX
from crucible.synth.patch import (
    DOC_RX,
    VCS_RX,
    VENDORED_RX,
    ExecResult,
    FixUnit,
    Milestone,
    PatchEnv,
    PatchSpec,
    Requirement,
    RevertMutant,
    RoleFn,
    ThreadedRunner,
)

SYNTH_PREFIX = f"{PREFIX}synth-"
COMMENTS = {"python": "#", "go": "//", "node": "//", "rust": "//", "java": "//"}
FLOORS = ("claims_only", "test_edit", "pass_stub")
_SINK: contextvars.ContextVar[list[Any] | None] = contextvars.ContextVar("synth_patch_sink", default=None)


class SealedPathError(PermissionError):
    pass


def _guard(path: Path) -> Path:
    p = Path(path).resolve()
    if "sealed" in p.parts and "data" in p.parts:
        raise SealedPathError(f"{p} is under data/sealed: never read by synth")
    return p


class _Recording:
    """The lane's sandbox, unchanged, with each run's ``TestRun`` handed to the grade that asked for it."""

    def __init__(self, inner: Any):
        self.inner = inner

    def __getattr__(self, k: str) -> Any:
        return getattr(self.inner, k)

    async def run(self, tree: Path, targets: Any) -> Any:
        tr = await self.inner.run(tree, targets)
        sink = _SINK.get()
        if sink is not None:
            sink.append(tr)  # ``run_tests`` normalises ``tr.results`` in place afterwards: the sink sees the final ids
        return tr


class Pool:
    """A debug root's training store as lane tasks (module doc)."""

    def __init__(self, root: Path, *, parallel: int = 4):
        self.root = _guard(root)
        self.store = WindowStore(self.root / "gen" / "store" / "train", split="train")
        self.lane = GenLane(self.store, generator=NAME, task_cls=DebugTask, repo_dirs={}, sandboxes=debug_sandboxes(parallel=parallel, prefix=SYNTH_PREFIX))
        self.lane.world_prefix = f"{SYNTH_PREFIX}world-"
        self.parallel = parallel

    def ids(self) -> list[str]:
        return self.store.window_ids()

    def task(self, wid: str) -> Any:
        rec = self.store.window(wid)
        if rec.get("split", "train") != "train":
            raise PermissionError(f"{wid} is not a training window")
        af = self.root / "authored" / f"{wid}.json"
        authored = json.loads(af.read_text()) if af.is_file() else None
        t = self.lane.task(wid, authored=authored)
        if not isinstance(t.sandbox, _Recording):
            t.sandbox = _Recording(t.sandbox)
        return t

    def gates(self, wid: str) -> dict[str, Any]:
        f = self.root / "gates" / f"{wid}.json"
        return json.loads(f.read_text()) if f.is_file() else {}


def open_pool(root: Path, *, parallel: int = 4) -> Pool:
    return Pool(root, parallel=parallel)


def lane_roles(task: Any) -> RoleFn:
    """The lane's path rules as synth roles (module doc)."""
    from crucible.factory.lanes.executable import is_protected, is_test_path
    from crucible.factory.langs import get

    lang = str(task.window.get("lang") or "python")
    prof = get(lang)
    hidden = set(task.window.get("tests_b") or {}) | set(task.window.get("tests_deleted") or [])
    cache: dict[str, str] = {}

    def roles(p: str) -> str:
        if p in cache:
            return cache[p]
        with scope(lang):
            if VCS_RX.search(p):
                r = "vcs"
            elif p in hidden:
                r = "test"
            elif prof.is_plumbing(p):
                r = "plumbing"
            elif prof.is_test_file(p):
                r = "test"
            elif is_test_path(p):
                r = "golden"
            elif is_protected(p):
                r = "config"
            elif VENDORED_RX.search(p):
                r = "vendored"
            elif DOC_RX.search(p):
                r = "doc"
            else:
                r = "source"
        cache[p] = r
        return r

    return roles


def patch_spec(task: Any) -> PatchSpec:
    """The window record of a lane task as a ``PatchSpec`` (module doc)."""
    w = task.window
    lang = str(w.get("lang") or "python")
    observed = {m["id"]: m.get("observed") or {} for m in w.get("mutants") or []}
    mutants = [RevertMutant(id=str(m.id), milestone=str(m.milestone), files=dict((m.solution or {}).get("files") or {}), observed=dict(observed.get(str(m.id)) or {}))
               for m in task.mutants()]
    floors = {c.id: c.solution for c in task.floors() if c.id in FLOORS and isinstance(c.solution, str)}
    guards = [c for c in task.check_tests if c == "c_regress"]
    excluded = w.get("excluded") or {}
    ignored = sorted(set(excluded.get("timing") or []) | set(w.get("s3_flaky_excluded") or []) | set(w.get("s8_flaky_excluded") or []))
    return PatchSpec(
        task_id=task.task_id, domain="debug", lang=lang, base_dir=str(task.base_dir), instruction=task.instruction,
        requirements=[Requirement(id=str(r["id"]), text=str(r.get("text") or ""), milestone=r.get("milestone")) for r in task.requirements],
        milestones=[Milestone(id=str(m["id"]), tests=list(m["tests"]), requires=list(m.get("requires") or [])) for m in w["milestones"]],
        checks={c: list(ts) for c, ts in task.check_tests.items()}, guards=guards, gating=[c for c in task.check_tests if task.gating.get(c, True)],
        weights=dict(task.weights), phi_weight=float(task.phi_weight), oracle_files=dict(w["oracle_files"]), mutants=mutants,
        hidden_tests=dict(task.grader_tests), units=[FixUnit(sha=str(u.get("sha") or ""), subject=str(u.get("subject") or "")) for u in w.get("units") or []],
        floors=floors, comment=COMMENTS.get(lang, "#"), meta={"ignored_tests": ignored, "window_a": w.get("a"), "image": (w.get("world") or {}).get("image")})


def patch_env(task: Any) -> PatchEnv:
    return PatchEnv(patch_spec(task), roles=lane_roles(task))


def lane_runner(task: Any, *, parallel: int = 4) -> ThreadedRunner:
    """The lane's grade as a synth exec runner (module doc)."""
    from crucible.factory.lanes.executable import is_protected
    from crucible.factory.lanes.windows import run_tests, suite_targets

    lang = str(task.window.get("lang") or "python")

    async def grade(files: dict[str, str | None], mode: str) -> ExecResult:
        sink: list[Any] = []
        tok = _SINK.set(sink)
        try:
            if mode == "visible":
                with scope(lang):
                    eff = {p: t for p, t in files.items() if not is_protected(p)}
                    targets = suite_targets(task.base_dir, {})
                    tr = await run_tests(task.sandbox, task.base_dir, eff, {}, targets)
                if tr.infra != "ok" or not tr.results:
                    return ExecResult(infra="masked" if tr.infra != "ok" else "empty", why=str(tr.why)[:300])
                return ExecResult(statuses=dict(tr.results))
            rep, _ = await task.grade([{"function_name": "apply_files", "arguments": {"files": dict(files)}}])
        finally:
            _SINK.reset(tok)
        if rep.infra.get("status") != "ok":
            return ExecResult(infra="masked", why=str(rep.infra.get("why") or "")[:300])
        meta = rep.meta or {}
        if meta.get("apply_errors"):
            return ExecResult(infra="unapplied", why="; ".join(meta["apply_errors"])[:300])
        statuses = dict(sink[-1].results) if sink else {}
        return ExecResult(checks={c: rep.gates.get(c) == "pass" for c in task.check_tests}, statuses=statuses)

    return ThreadedRunner(grade, parallel=parallel, name=f"synth-{task.task_id}")


def prior_of(record: Mapping[str, Any]) -> dict[str, dict[str, int]]:
    """The lane's executed reruns from a gate record: S10's oracle grades (false negatives) and S3's existing-suite
    reruns at A′ (unclean runs)."""
    g = record.get("gates") or {}
    s10 = (g.get("S10") or {}).get("evidence") or {}
    s3 = ((g.get("S3") or {}).get("evidence") or {}).get("reruns") or {}
    out: dict[str, dict[str, int]] = {}
    if s10.get("n"):
        out["S10"] = {"n": int(s10["n"]), "bad": int(s10.get("false_negatives") or 0)}
    if s3.get("n"):
        out["S3"] = {"n": int(s3["n"]), "bad": int(s3["n"]) - int(s3.get("clean") or 0)}
    return out


def remove_containers(prefix: str = SYNTH_PREFIX) -> list[str]:
    """Remove this module's own containers by name (never a prune). Returns the names removed."""
    if not prefix.startswith(SYNTH_PREFIX):
        raise ValueError(f"only {SYNTH_PREFIX}* containers are this module's")
    r = subprocess.run(["docker", "ps", "-a", "--filter", f"name=^{prefix}", "--format", "{{.Names}}"], capture_output=True, text=True, check=False)
    names = [n for n in r.stdout.split() if n.startswith(prefix)]
    for n in names:
        subprocess.run(["docker", "rm", "-f", n], capture_output=True, check=False)
    return names


# ---- the real-defect set: debug-lane defect versions still on disk -------------------------------------------------------
class _NoRunner:
    """A runner for property probes only: any exec call is an error (nothing may run Docker)."""

    def submit(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no runner: property probes only")

    def result(self, *a: Any, **k: Any) -> Any:
        raise RuntimeError("no runner: property probes only")

    def prefetch(self, *a: Any, **k: Any) -> None:
        return None


async def detect_debug(defect: Any, *, root: Path, parallel: int = 4) -> Any:
    """Run the detector of one debug-lane real defect (``synth.realdefects.DETECTOR``) on its defective task version in
    the training store under ``root`` (``data/debug``): that class's probes of that op only, executed and labelled."""
    from crucible.synth.falsify.patch import PatchMUT, PatchTAINT
    from crucible.synth.realdefects import Detection

    cls, op = str(defect.executable.detector).split(":", 1)
    pool = Pool(root, parallel=parallel)
    task = pool.task(str(defect.task_version))
    env = patch_env(task)
    runner = lane_runner(task, parallel=parallel) if op == "visible_contradicts" else None
    try:
        f = PatchMUT(env) if cls == "MUT" else PatchTAINT(env, runner, visible=runner is not None)
        st = env.task()
        ctx = env.context(runner if runner is not None else _NoRunner())
        hits: list[str] = []
        details: list[Any] = []
        n = invalid = 0
        for p in [p for p in await f.propose(st, ctx) if p.op == op]:
            o = await f.execute(p, st, ctx)
            if not o.valid:
                invalid += 1
                details.append(o.error)
                continue
            n += 1
            lab, _d = f.label(p, o)
            if lab == "confirmed":
                hits.append(f"{cls}:{op}:{p.site}")
                details.append({k: v for k, v in o.data.items() if k in ("shared", "lines", "stale", "n_stale")})
        if n == 0:
            return Detection(detected=None, detail={"why": "no valid probe", "invalid": invalid, "errors": details[:3]})
        # a hit shows the artifact still has the defect; a miss alone cannot tell a recall gap from a version rebuilt clean
        return Detection(detected=bool(hits), exhibits=True if hits else None, by=hits[:10], detail={"probes": n, "invalid": invalid, "evidence": details[:5]})
    finally:
        if runner is not None:
            runner.close()
            remove_containers()
