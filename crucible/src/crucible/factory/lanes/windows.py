"""The window lane (study 08 §0-§5, §9 step 2; report 09): long-horizon coding tasks from real release windows.

A window runs from release A to release B of a real repository (or across a run of units that build on each other).
The repository at A is the start state, B is an executed oracle, the tests the window added are the checks, and each
unit's own tests are a milestone in a known real order. ``windows_git`` mines the windows and builds trees,
``windows_world`` builds the per-(repo, A) image, runs the leak probes and opens the policy's World.

**Validation** (``validate_window``, the SWE-smith pattern of ``executable.validate_candidate`` at window scale): the
repository's suite runs at A (with B's test files) and at B, ``reruns`` times each in the World image. Fail-to-pass tests
are candidate checks, pass-to-pass tests regression checks; anything flaky is dropped and a test that passed at A and
fails at B is dropped (the window removed that behaviour). Fail-to-pass tests are grouped by their owner unit
(``windows_git.owners_of_tests``): one milestone per unit, in history order; tests no unit changed are collateral regression
checks.

**Executed dependency prober** (``probe_dependencies``): milestone k is A plus only unit k's diff, run on k's own tests
(the test files as unit k wrote them). If they fail, the minimal set of earlier units that makes them pass is found by
removal from the full prefix (ddmin's complement step: halves first, then one at a time). Each edge is labelled
``patch-conflict`` (without that unit, unit k's diff no longer applies) or ``executed``. ``requires`` holds the milestone
ids among them; a window with no milestone edge is ``independent`` (long, not long-horizon) and is excluded from Type A.

**Commit-revert mutants** (``revert_mutants``): B with only unit k's diff reverted (3-way; skipped on conflict), graded
by the window's verifier. Expected: milestone k fails, every milestone that does not depend on unit k still passes.

**The task** (``WindowTask``, an ``executable.ExecTask``): the accept contract and ``driver.LANE_TASK`` (grade, noop,
oracle, instruction, negatives, mutants, floor_kwargs, agent_visible, hidden_strings, pinned_images, property_audit),
plus the coding seam of ``runtime.harness.workspace.WorkspaceHarness`` (``open``, ``workspace``, ``broken``,
``grade_patch``) and ``milestone_status(patch)`` for the checkpoint sidecar. Grading applies the solution to a pristine
A tree (protected paths are never written), overwrites every test file from B, runs the hidden fail-to-pass and
pass-to-pass tests in a separate ``--network none`` container and reads the guard's JUnit report. Reward is 1 iff every
gating check passes; ``meta`` carries the per-milestone pass vector and Phi (weighted milestones whose ``requires`` all
pass; uniform weights unless injected).

**Prompt and ledger** (``author_window``): a maintainer's release request written from the units' messages, the
changelog and a diff summary (public names only; never test bodies, never file paths), one requirement id per
milestone, then a fixed epilogue that states truthfully how the episode works (sandbox, offline, hidden tests, existing
tests, ``GIVE_UP:``, ``STATUS.json``). The ledger maps each requirement to its milestone's tests both ways; ``lint_prompt``
refuses process instructions no check enforces, file paths, test names and commit ids.
"""

from __future__ import annotations

import asyncio
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from crucible.factory.lanes.executable import (
    REGRESSION_REQ,
    REGRESSION_TEXT,
    DockerSandbox,
    ExecLane,
    ExecTask,
    LaneStore,
    MinedTask,
    TestRun,
    _safe,
    apply_solution,
    build_tree,
    call_supported,
    fingerprint,
    is_protected,
    is_test_path,
    oracle_solution,
    scrub,
    sha256_text,
)
from crucible.factory.lanes.windows_git import RepoGit, Unit, Window, _defs, mine_windows, owners_of_tests
from crucible.factory.lanes.windows_world import BASE_IMAGE, ImageBook, leak_probes, open_world, project_name

LANE = "window"
WINDOW_MINER = "miner:m17-window-lane"
SMALL_REPOS = ("python-semver", "parse", "boltons", "cachetools", "humanize", "more-itertools")
# medium pure-Python repositories with real releases and work that builds on itself (09 §2 gives the reasons)
POOL_REPOS = ("click", "marshmallow", "pyparsing", "lark", "markdown", "arrow", "glom", "toolz", "funcy", "cerberus", "typer", "dateutil")
# chosen before any mining: never written into a training store (H1: same repo class; H2: a larger app, other styles)
HELD_OUT_REPOS: dict[str, str] = {"sqlparse": "H1", "tomlkit": "H1", "python-dotenv": "H1", "voluptuous": "H1", "jinja": "H2", "mistune": "H2"}
REPO_URLS = {
    "python-semver": "python-semver/python-semver", "parse": "r1chardj0n3s/parse", "boltons": "mahmoud/boltons", "cachetools": "tkem/cachetools",
    "humanize": "python-humanize/humanize", "more-itertools": "more-itertools/more-itertools", "click": "pallets/click",
    "marshmallow": "marshmallow-code/marshmallow", "pyparsing": "pyparsing/pyparsing", "lark": "lark-parser/lark", "markdown": "Python-Markdown/markdown",
    "arrow": "arrow-py/arrow", "glom": "mahmoud/glom", "toolz": "pytoolz/toolz", "funcy": "Suor/funcy", "cerberus": "pyeve/cerberus",
    "typer": "fastapi/typer", "dateutil": "dateutil/dateutil", "sqlparse": "andialbrecht/sqlparse", "tomlkit": "python-poetry/tomlkit",
    "python-dotenv": "theskumar/python-dotenv", "voluptuous": "alecthomas/voluptuous", "jinja": "pallets/jinja", "mistune": "lepture/mistune",
}
# one module that fails to import must not stop the others from running: a milestone's tests are judged on their own
COLLECT_ALL = ("--continue-on-collection-errors",)
STAGES = ("windows", "world_built", "leak_ok", "f2p_valid", "dependent", "mutants_valid")
SUITE_FILE = re.compile(r"(^|/)(test_[^/]*|[^/]*_test|tests?)\.py$")  # tests.py: voluptuous keeps its suite there


class HeldOutError(RuntimeError):
    """A held-out window offered to a training store (or the reverse)."""


def split_of(repo: str) -> str:
    return "heldout" if repo in HELD_OUT_REPOS else "train"


# ---- the store -------------------------------------------------------------------------------------------------------
class WindowStore(LaneStore):
    """``<root>/tasks/<id>/{record.json (MinedTask), window.json, base/}``. A store holds one split (``split.json``):
    a training store refuses held-out repositories and a held-out store refuses the rest."""

    def __init__(self, root: Path, *, split: str = "train"):
        if split not in ("train", "heldout"):
            raise ValueError(f"split must be train or heldout, got {split!r}")
        root = Path(root)
        marker = root / "split.json"
        if marker.exists():
            have = json.loads(marker.read_text()).get("split")
            if have != split:
                raise HeldOutError(f"{root} is a {have} store, not {split}")
        super().__init__(root)
        if not marker.exists():
            marker.write_text(json.dumps({"split": split, "held_out_repos": HELD_OUT_REPOS}, indent=1, sort_keys=True))
        self.split = split

    def guard(self, repo: str) -> None:
        if split_of(repo) != self.split:
            raise HeldOutError(f"{repo} is {split_of(repo)}: it never goes into the {self.split} store at {self.root}")

    def save(self, t: MinedTask) -> None:
        self.guard(t.repo)
        super().save(t)

    def save_window(self, rec: Mapping[str, Any]) -> None:
        self.guard(str(rec["repo"]))
        d = self.dir(str(rec["id"]))
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "window.json.tmp"
        tmp.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
        tmp.replace(d / "window.json")

    def window(self, wid: str) -> dict[str, Any]:
        return json.loads((self.dir(wid) / "window.json").read_text())

    def window_ids(self) -> list[str]:
        return sorted(p.parent.name for p in (self.root / "tasks").glob("*/window.json"))


# ---- trees and solutions ---------------------------------------------------------------------------------------------
_DIFF_HEAD = re.compile(r"^diff --git a/(\S+) b/(\S+)\s*$", re.MULTILINE)


def split_patch(patch: str) -> list[tuple[str, str]]:
    """``(path, chunk)`` per file of a unified git diff (the path after the change; the old path for a deletion)."""
    heads = list(_DIFF_HEAD.finditer(patch))
    out = []
    for i, m in enumerate(heads):
        chunk = patch[m.start(): heads[i + 1].start() if i + 1 < len(heads) else len(patch)]
        deleted = re.search(r"^\+\+\+ /dev/null$", chunk, re.MULTILINE) is not None
        out.append((m.group(1) if deleted else m.group(2), chunk))
    return out


def _apply_patch(tree: Path, patch: str) -> tuple[list[str], list[str], list[str]]:
    """Apply a git diff in ``tree`` (protected paths dropped). Returns (touched paths, protected, errors)."""
    keep, protected, touched = [], [], []
    for path, chunk in split_patch(patch):
        names = {path, *(_DIFF_HEAD.match(chunk).groups())} if _DIFF_HEAD.match(chunk) else {path}
        if any(_safe(p) is None for p in names):
            return [], [], [f"unsafe path in the patch: {sorted(names)}"]
        if any(is_protected(p) for p in names):
            protected.append(path)
            continue
        keep.append(chunk)
        touched += sorted(names)
    if not keep:
        return [], sorted(set(protected)), [] if protected or not patch.strip() else ["the patch has no file diff"]
    with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False) as fh:
        fh.write("".join(c if c.endswith("\n") else c + "\n" for c in keep))
    try:
        r = subprocess.run(["git", "apply", "--whitespace=nowarn", "--unsafe-paths", fh.name], cwd=tree, capture_output=True, text=True, check=False,
                           env={"GIT_CEILING_DIRECTORIES": str(Path(tree).parent), "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin",
                                "HOME": str(tree)})
    finally:
        Path(fh.name).unlink(missing_ok=True)
    errs = [f"the patch does not apply: {r.stderr.strip()[-400:]}"] if r.returncode != 0 else []
    return sorted(set(touched)), sorted(set(protected)), errs


def solution_files(base_dir: Path, solution: Any) -> tuple[dict[str, str | None], list[str], list[str]]:
    """Any solution form -> ``({path: text | None}, protected paths it tried to write, errors)``: a unified diff (what
    ``WorkspaceHarness`` collects), a list of steps (the oracle: ``{"function_name": "apply_unit", "arguments":
    {"patch": ...}}``), or the executable lane's ``{edits, files}``."""
    if isinstance(solution, Mapping) or solution is None:
        with tempfile.TemporaryDirectory(prefix="lhs-") as td:
            tree = Path(td) / "t"
            shutil.copytree(base_dir, tree, symlinks=True)
            ap = apply_solution(tree, solution or {})
            return dict(ap.written), list(ap.protected), list(ap.errors)
    patches: list[str] = []
    if isinstance(solution, str):
        patches = [solution]
    elif isinstance(solution, list):
        for s in solution:
            p = s if isinstance(s, str) else ((s.get("arguments") or {}).get("patch") if isinstance(s, Mapping) else None)
            if isinstance(p, str):
                patches.append(p)
    else:
        return {}, [], [f"a solution is a patch, a list of steps or {{edits, files}}, got {type(solution).__name__}"]
    with tempfile.TemporaryDirectory(prefix="lhs-") as td:
        tree = Path(td) / "t"
        shutil.copytree(base_dir, tree, symlinks=True)
        touched: set[str] = set()
        protected: set[str] = set()
        for p in patches:
            t, pr, errs = _apply_patch(tree, p)
            if errs:
                return {}, sorted(protected | set(pr)), errs
            touched |= set(t)
            protected |= set(pr)
        files: dict[str, str | None] = {}
        for rel in sorted(touched):
            f = tree / rel
            if not f.exists():
                files[rel] = None
                continue
            try:
                files[rel] = f.read_text()
            except UnicodeDecodeError:
                continue  # a binary file: never source the tests import
        return files, sorted(protected), []


def suite_targets(tree: Path, tests: Mapping[str, str], deleted: Sequence[str] = ()) -> list[str]:
    files = {f.relative_to(tree).as_posix() for f in Path(tree).rglob("*.py")} - set(deleted)
    files |= set(tests)
    return sorted(p for p in files if is_test_path(p) and SUITE_FILE.search(p) and not p.startswith((".", "docs/", "doc/")))


async def run_tests(sandbox: DockerSandbox, base_dir: Path, files: Mapping[str, str | None], tests: Mapping[str, str], targets: Sequence[str],
                    deleted: Sequence[str] = ()) -> TestRun:
    """A fresh copy of A, ``files`` written (protected paths never), ``tests`` written over it, the test files B deleted
    (a moved conftest) removed, ``targets`` run."""
    work = Path(tempfile.mkdtemp(prefix="lht-"))
    try:
        tree = work / "t"
        ap = build_tree(base_dir, tree, solution={"files": dict(files)}, tests=tests)
        if ap.errors:
            return TestRun({}, "ok", None, 0.0, "", why="; ".join(ap.errors[:3]))
        for p in deleted:
            if _safe(p) and is_test_path(p):
                (tree / p).unlink(missing_ok=True)
        return await sandbox.run(tree, list(targets))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def classify_runs(pre: Sequence[TestRun], post: Sequence[TestRun]) -> dict[str, list[str]]:
    """SWE-smith's classes over reruns (``executable.validate_candidate``): flaky = status differs between reruns."""
    ids = sorted({t for r in [*pre, *post] for t in r.results})

    def st(runs: Sequence[TestRun], t: str) -> set[str]:
        return {r.results.get(t, "missing") for r in runs}

    flaky = [t for t in ids if len(st(pre, t)) > 1 or len(st(post, t)) > 1]
    stable = [t for t in ids if t not in flaky]
    post_pass = {t for t in stable if st(post, t) == {"passed"}}
    pre_pass = {t for t in stable if st(pre, t) == {"passed"}}
    return {"f2p": sorted(post_pass - pre_pass), "p2p": sorted(post_pass & pre_pass), "p2f": sorted(pre_pass - post_pass), "flaky": flaky,
            "post_fail": sorted(t for t in stable if t not in post_pass and t not in pre_pass), "post_pass": sorted(post_pass)}


def passed_group(results: Mapping[str, str], tests: Sequence[str], *, require_all: bool = True) -> bool:
    present = [t for t in tests if t in results] if not require_all else list(tests)
    return bool(present) and all(results.get(t) == "passed" for t in present)


# ---- the dependency prober -------------------------------------------------------------------------------------------
async def ddmin(items: Sequence[int], ok: Callable[[list[int]], Any]) -> list[int]:
    """A 1-minimal subset S of ``items`` with ``ok(S)`` (``ok(items)`` holds): complements removed halves first."""
    s = list(items)
    n = 2
    while s:
        chunk = max(1, -(-len(s) // n))
        removed = False
        for i in range(0, len(s), chunk):
            cand = s[:i] + s[i + chunk:]
            if await ok(cand):
                s, n, removed = cand, max(n - 1, 2), True
                break
        if not removed:
            if chunk == 1:
                break
            n = min(len(s), n * 2)
    return s


async def probe_dependencies(rec: dict[str, Any], repo: RepoGit, sandbox: DockerSandbox) -> dict[str, Any]:
    """Fill ``requires`` / ``requires_units`` / ``edges`` / ``probe`` of every milestone; returns the prober's counts."""
    units = [Unit(**u) for u in rec["units"]]
    index = {u.sha: i for i, u in enumerate(units)}
    ms_by_unit = {m["unit"]: m for m in rec["milestones"]}
    runs = 0

    async def probe(m: dict[str, Any]) -> None:
        nonlocal runs
        i = index[m["unit"]]
        u = units[i]
        test_files = sorted({t.split("::", 1)[0] for t in m["tests"]})
        tests_k = {p: repo.show(u.sha, p) or "" for p in test_files}
        for p in rec["tests_b"]:  # the unit's conftest files, as the unit saw them
            if p.endswith("conftest.py") and (txt := repo.show(u.sha, p)) is not None:
                tests_k[p] = txt
        gone_k = [p for p in rec.get("tests_deleted") or [] if repo.show(u.sha, p) is None]
        cache: dict[tuple[int, ...], tuple[bool, str]] = {}

        async def run(s: Sequence[int]) -> tuple[bool, str]:
            nonlocal runs
            key = tuple(sorted(s))
            if key not in cache:
                work = Path(tempfile.mkdtemp(prefix="lhp-"))
                try:
                    built, bad = await asyncio.to_thread(lambda: repo.build(rec["a"], [units[j] for j in key] + [u], work / "t", source_only=True))
                    if not built:
                        cache[key] = (False, f"conflict:{bad}")
                    else:
                        for p, txt in tests_k.items():
                            (work / "t" / p).parent.mkdir(parents=True, exist_ok=True)
                            (work / "t" / p).write_text(txt)
                        for p in gone_k:
                            (work / "t" / p).unlink(missing_ok=True)
                        tr = await sandbox.run(work / "t", test_files)
                        runs += 1
                        cache[key] = (tr.infra == "ok" and passed_group(tr.results, m["tests"], require_all=False), "run" if tr.infra == "ok" else "masked")
                finally:
                    shutil.rmtree(work, ignore_errors=True)
            return cache[key]

        async def ok(s: Sequence[int]) -> bool:
            return (await run(s))[0]

        earlier = list(range(i))
        if await ok([]):
            m.update(requires=[], requires_units=[], edges={}, probe="alone")
            return
        if not earlier or not await ok(earlier):
            m.update(requires=[x["id"] for x in rec["milestones"] if index[x["unit"]] < i], requires_units=[units[j].sha for j in earlier], edges={},
                     probe="prefix_fails")
            return
        s = await ddmin(earlier, ok)
        edges = {}
        for j in s:
            without = [x for x in s if x != j]
            ok_build = (await asyncio.to_thread(_applies, repo, rec["a"], [units[x] for x in without] + [u]))
            edges[units[j].sha] = "executed" if ok_build else "patch-conflict"
        m.update(requires=[ms_by_unit[units[j].sha]["id"] for j in s if units[j].sha in ms_by_unit], requires_units=[units[j].sha for j in s], edges=edges,
                 probe="minimized")

    await asyncio.gather(*[probe(m) for m in rec["milestones"]])
    edges = sum(len(m["requires"]) for m in rec["milestones"] if m.get("probe") == "minimized")
    rec["dependency_edges"], rec["independent"] = edges, edges == 0
    return {"runs": runs, "milestone_edges": edges, "prefix_fails": sum(1 for m in rec["milestones"] if m.get("probe") == "prefix_fails"),
            "alone": sum(1 for m in rec["milestones"] if m.get("probe") == "alone")}


def _applies(repo: RepoGit, base: str, units: Sequence[Unit]) -> bool:
    with tempfile.TemporaryDirectory(prefix="lha-") as td:
        return repo.build(base, units, Path(td) / "t", source_only=True)[0]


def closure(rec: Mapping[str, Any], mid: str) -> set[str]:
    """Every unit sha milestone ``mid`` depends on, transitively."""
    by_id = {m["id"]: m for m in rec["milestones"]}
    by_unit = {m["unit"]: m for m in rec["milestones"]}
    out: set[str] = set()
    todo = list(by_id[mid].get("requires_units") or [])
    while todo:
        s = todo.pop()
        if s in out:
            continue
        out.add(s)
        if s in by_unit:
            todo += list(by_unit[s].get("requires_units") or [])
    return out


# ---- commit-revert mutants --------------------------------------------------------------------------------------------
def mutant_files(repo: RepoGit, rec: Mapping[str, Any], base_dir: Path, unit: Unit) -> dict[str, str | None] | None:
    """B with only ``unit`` reverted, as the files that differ from A; None when the revert does not apply."""
    with tempfile.TemporaryDirectory(prefix="lhm-") as td:
        tree = Path(td) / "t"
        ok, _ = repo.build(rec["b"], [], tree, reverse=[unit], source_only=True)
        if not ok:
            return None
        out: dict[str, str | None] = {}
        for p in sorted(set(rec["oracle_files"]) | set(unit.writable_files)):
            f, b = tree / p, base_dir / p
            now = f.read_text(errors="replace") if f.is_file() else None
            was = b.read_text(errors="replace") if b.is_file() else None
            if now != was and not is_protected(p) and not is_test_path(p):
                out[p] = now
        return out


async def revert_mutants(rec: dict[str, Any], repo: RepoGit, base_dir: Path, sandbox: DockerSandbox) -> dict[str, Any]:
    units = {u["sha"]: Unit(**u) for u in rec["units"]}
    targets = rec["targets"]
    out = []

    async def one(m: dict[str, Any]) -> dict[str, Any]:
        files = await asyncio.to_thread(mutant_files, repo, rec, base_dir, units[m["unit"]])
        if files is None:
            return {"id": f"revert_{m['id']}", "milestone": m["id"], "status": "revert_conflict"}
        if not files:
            return {"id": f"revert_{m['id']}", "milestone": m["id"], "status": "no_source_change"}
        tr = await run_tests(sandbox, base_dir, files, rec["tests_b"], targets, rec.get("tests_deleted") or ())
        if tr.infra != "ok" or not tr.results:
            return {"id": f"revert_{m['id']}", "milestone": m["id"], "status": "masked", "why": tr.why}
        observed = {x["id"]: passed_group(tr.results, x["tests"]) for x in rec["milestones"]}
        expected = {x["id"]: (False if x["id"] == m["id"] else (None if m["unit"] in closure(rec, x["id"]) else True)) for x in rec["milestones"]}
        # valid: exactly as expected; broad: milestone k fails but so does a milestone that does not depend on it (the
        # revert broke more than its unit); uncaught: milestone k still passes (a later unit re-did the work)
        status = "uncaught" if observed[m["id"]] else ("valid" if all(v is None or observed[k] == v for k, v in expected.items()) else "broad")
        return {"id": f"revert_{m['id']}", "milestone": m["id"], "status": status, "expected": expected, "observed": observed,
                "solution": {"files": files}}

    out = list(await asyncio.gather(*[one(m) for m in rec["milestones"]]))
    counts: dict[str, int] = {}
    for x in out:
        counts[x["status"]] = counts.get(x["status"], 0) + 1
    return {"mutants": out, "counts": counts}


# ---- validation: one window, end to end -------------------------------------------------------------------------------
def observation(rec: Mapping[str, Any]) -> dict[str, Any]:
    from crucible.factory.cards import ObservationRecord

    tests = rec.get("tests_b") or {}
    sha = sha256_text(rec["a"], rec["b"], *[f"{p}\n{tests[p]}" for p in sorted(tests)])
    created = rec.get("b_date") or "1970-01-01T00:00:00+00:00"
    return ObservationRecord(id=f"obs-{rec['id']}", type="real_pr_tests", pointer=f"git:{rec['repo']}@{rec['a']}..{rec['b']}", sha256=sha,
                             created=created, by=WINDOW_MINER).model_dump(mode="json")


def mined_of(rec: Mapping[str, Any], base_dir: Path) -> MinedTask:
    """The window as the executable lane's record (``ExecTask`` grades, floors and attacks it unchanged)."""
    src = sorted(p for p in rec["oracle_files"] if p.endswith(".py"))
    base_src = {p: (base_dir / p).read_text(errors="replace") for p in src if (base_dir / p).is_file()}
    fix_src = {p: rec["oracle_files"][p] for p in src}
    ms = rec["milestones"]
    return MinedTask(id=rec["id"], repo=rec["repo"], sha=rec["b"], parent=rec["a"], date=rec.get("b_date") or "",
                     subject=f"{len(ms)} milestones from {rec['a_ref']} to {rec['b_ref']}", body="\n".join(m["subject"] for m in ms),
                     src_files=src, test_files=sorted(rec["tests_b"]), tests=dict(rec["tests_b"]), base_src=base_src, fix_src=fix_src,
                     oracle=oracle_solution(base_src, fix_src), f2p=sorted({t for m in ms for t in m["tests"]}),
                     p2p=sorted(set(rec["p2p"]) | set(rec["collateral"])), excluded=dict(rec["excluded"]), observation=observation(rec),
                     validation=dict(rec.get("validation") or {}), base_dir=str(base_dir))


def window_files(rec: dict[str, Any], repo: RepoGit) -> None:
    """``tests_b`` (every test file the window adds or changes, at B), ``tests_deleted`` and ``oracle_files`` (every
    agent-writable path that differs between A and B, at B; None = deleted)."""
    changed = repo.changed(rec["a"], rec["b"])
    rec["tests_b"] = {p: txt for s, p in changed if s != "D" and is_test_path(p) and p.endswith(".py") and (txt := repo.show(rec["b"], p)) is not None}
    rec["tests_deleted"] = sorted(p for s, p in changed if s == "D" and is_test_path(p))
    oracle_files: dict[str, str | None] = {}
    for s, p in changed:
        if is_protected(p) or is_test_path(p):
            continue
        txt = None if s == "D" else repo.show(rec["b"], p)
        if s != "D" and txt is None:
            continue  # binary: no test imports it as source
        oracle_files[p] = txt
    rec["oracle_files"] = oracle_files


def new_test_names(rec: Mapping[str, Any], repo: RepoGit, base: Path) -> set[str]:
    """Test functions B defines that A's tree never mentions (what the leak probe looks for in the image)."""
    names: set[str] = set()
    for p, txt in rec["tests_b"].items():
        before = set(_defs(repo.show(rec["a"], p)))
        names |= {q.rsplit("::", 1)[-1] for q in set(_defs(txt)) - before}
    a_text = "\n".join(f.read_text(errors="replace") for f in Path(base).rglob("*.py") if f.is_file())
    return {n for n in names if n.startswith("test") and n not in a_text}


async def f2p_stage(rec: dict[str, Any], repo: RepoGit, base: Path, sandbox: DockerSandbox, *, reruns: int = 3, min_milestones: int = 2) -> str | None:
    """Fail-to-pass / pass-to-pass with reruns, then milestones by owner unit. Returns why the window stops, or None."""
    tests_b, gone = rec["tests_b"], rec["tests_deleted"]
    targets = suite_targets(base, tests_b, gone)
    t1 = time.monotonic()
    pre = await asyncio.gather(*[run_tests(sandbox, base, {}, tests_b, targets, gone) for _ in range(reruns)])
    post = await asyncio.gather(*[run_tests(sandbox, base, rec["oracle_files"], tests_b, targets, gone) for _ in range(reruns)])
    rec.setdefault("timings", {})["f2p"] = round(time.monotonic() - t1, 1)
    ev = {"reruns": reruns, "targets": len(targets), "pre_counts": [len(r.results) for r in pre], "post_counts": [len(r.results) for r in post],
          "post_seconds": [round(r.seconds, 1) for r in post]}
    rec["validation"] = ev
    if any(r.infra != "ok" for r in [*pre, *post]):
        return "infra: the sandbox did not run"
    if any(not r.results for r in post):
        ev["post_tail"] = post[0].tail[-1200:] or post[0].why
        return "no test result at B (collection failed: environment mismatch)"
    cls = classify_runs(pre, post)
    ev.update({k: len(v) for k, v in cls.items()})
    if not cls["post_pass"] or len(cls["post_fail"]) > max(20, len(cls["post_pass"]) // 5):
        ev["post_fail_sample"] = cls["post_fail"][:10]
        return f"{len(cls['post_fail'])} tests fail at B (environment mismatch)"
    units = [Unit(**u) for u in rec["units"]]
    src_units = {u.sha for u in units if u.src_files}
    # a unit that changes no source (tests or docs only) is not a milestone: its new tests pin behaviour another unit
    # wrote, so they gate as regression checks (collateral)
    owners = {t: (o if o in src_units else None) for t, o in owners_of_tests(repo, units, cls["f2p"]).items()}
    groups: dict[str, list[str]] = {}
    for t, o in owners.items():
        if o is not None:
            groups.setdefault(o, []).append(t)
    order = {u.sha: i for i, u in enumerate(units)}
    rec["milestones"] = [{"id": f"m{order[s] + 1:02d}", "unit": s, "index": order[s], "subject": units[order[s]].subject, "tests": sorted(ts)}
                         for s, ts in sorted(groups.items(), key=lambda kv: order[kv[0]])]
    rec.update(targets=targets, p2p=cls["p2p"], collateral=sorted(t for t, o in owners.items() if o is None),
               excluded={"flaky": cls["flaky"], "p2f": cls["p2f"], "post_fail": cls["post_fail"]})
    if len(rec["milestones"]) < min_milestones:
        return f"{len(rec['milestones'])} milestone(s) with fail-to-pass tests (need {min_milestones})"
    return None


async def validate_window(win: Window, store: WindowStore, images: ImageBook, sandboxes: Callable[[str], DockerSandbox], *, reruns: int = 3,
                          min_milestones: int = 2, mutants: bool = True, log: Callable[[dict], None] | None = None) -> dict[str, Any]:
    """Every stage for one window; the record (saved whatever stage it reached) says how far it got and why it stopped."""
    store.guard(win.repo)
    t0 = time.monotonic()
    repo = RepoGit(Path(win.repo_dir))
    rec: dict[str, Any] = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows", "reason": "",
                           "lane": LANE, "timings": {}}
    d = store.dir(win.id)
    base = d / "base"

    def stop(reason: str) -> dict[str, Any]:
        rec["reason"] = reason
        rec["timings"]["total"] = round(time.monotonic() - t0, 1)
        store.save_window(rec)
        if log:
            log({"event": "window", "id": win.id, "repo": win.repo, "stage": rec["stage"], "reason": reason, "seconds": rec["timings"]["total"]})
        return rec

    if not base.exists():
        ok, bad = await asyncio.to_thread(repo.build, win.a, [], base)
        if not ok:
            shutil.rmtree(base, ignore_errors=True)
            return stop(f"A does not check out ({bad})")
    # the World at A
    card = await images.ensure(win.repo, base)
    rec["world"] = {k: v for k, v in card.items() if k != "build_log"}
    if not card["ok"]:
        rec["world"]["build_log_tail"] = str(card.get("build_log", ""))[-600:]
        return stop("world image did not build")
    rec["stage"] = "world_built"
    window_files(rec, repo)
    t1 = time.monotonic()
    leak = await leak_probes(card["image"], base, package=card.get("project") or project_name(base), version=win.b_version,
                             new_test_names=new_test_names(rec, repo, base))
    rec["world"]["leak"] = leak
    rec["timings"]["leak"] = round(time.monotonic() - t1, 1)
    if not leak["ok"]:
        return stop(f"leak probe failed: {leak.get('probes') or leak.get('evidence')}")
    rec["stage"] = "leak_ok"
    sandbox = sandboxes(card["image"])
    why = await f2p_stage(rec, repo, base, sandbox, reruns=reruns, min_milestones=min_milestones)
    if why is not None:
        return stop(why)
    rec["stage"] = "f2p_valid"
    t1 = time.monotonic()
    rec["prober"] = await probe_dependencies(rec, repo, sandbox)
    rec["timings"]["prober"] = round(time.monotonic() - t1, 1)
    rec["dependency_edges"] = rec["prober"]["milestone_edges"]
    rec["independent"] = rec["dependency_edges"] == 0
    store.save(mined_of(rec, base))
    if rec["independent"]:
        return stop("independent: no milestone needs an earlier one (long, not long-horizon)")
    rec["stage"] = "dependent"
    if not mutants:
        return stop("mutants not run")
    t1 = time.monotonic()
    mut = await revert_mutants(rec, repo, base, sandbox)
    rec["mutants"] = mut["mutants"]
    rec["mutant_counts"] = mut["counts"]
    rec["timings"]["mutants"] = round(time.monotonic() - t1, 1)
    if mut["counts"].get("valid", 0) >= 1:
        rec["stage"] = "mutants_valid"
        return stop("")
    return stop(f"mutants: {mut['counts']}")


def sandbox_factory(*, parallel: int = 8, timeout_s: float = 600.0, cpus: str = "2", memory: str = "2g") -> Callable[[str], DockerSandbox]:
    """One grading sandbox per World image, all sharing one concurrency limit."""
    sem = asyncio.Semaphore(parallel)
    made: dict[str, DockerSandbox] = {}

    def get(image: str) -> DockerSandbox:
        if image not in made:
            sb = DockerSandbox(image, guard=True, timeout_s=timeout_s, per_test_s=60, cpus=cpus, memory=memory, parallel=parallel, extra_args=COLLECT_ALL)
            sb._sem = sem
            made[image] = sb
        return made[image]

    return get


async def mine(repos: Mapping[str, Path], store: WindowStore, *, images: ImageBook | None = None, sandboxes: Callable[[str], DockerSandbox] | None = None,
               cap: int = 8, since: str = "2019-01-01", min_units: int = 5, max_units: int = 25, reruns: int = 3, parallel_windows: int = 4,
               deadline: float | None = None, log: Callable[[dict], None] | None = None, window_sem: asyncio.Semaphore | None = None,
               redo: Sequence[str] = ("deadline", "error")) -> dict[str, Any]:
    """Windows per repository, each validated end to end; a repository of the other split is refused before mining."""
    images = images or ImageBook()
    sandboxes = sandboxes or sandbox_factory()
    sem = window_sem or asyncio.Semaphore(parallel_windows)
    out: dict[str, Any] = {}
    for name, path in repos.items():
        store.guard(name)
        wins = [w for w in await asyncio.to_thread(mine_windows, Path(path), name, cap=cap, since=since, min_units=min_units, max_units=max_units)
                if not (store.dir(w.id) / "window.json").exists() or str(store.window(w.id).get("reason", "")).startswith(tuple(redo))]

        async def go(w: Window) -> None:
            async with sem:
                if deadline is not None and time.time() > deadline:  # recorded, so the funnel counts what was never reached
                    store.save_window({**{k: v for k, v in w.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                       "reason": "deadline: not validated", "lane": LANE})
                    return
                try:
                    await validate_window(w, store, images, sandboxes, reruns=reruns, log=log)
                except Exception as e:  # noqa: BLE001 - one window's failure is recorded, never the run's
                    store.save_window({**{k: v for k, v in w.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                       "reason": f"error: {type(e).__name__}: {e}"[:500], "lane": LANE})

        await asyncio.gather(*[go(w) for w in wins])
        out[name] = len(wins)
    return out


def funnel(store: WindowStore) -> dict[str, Any]:
    """Windows reaching each stage, per repository, with the stop reasons."""
    per: dict[str, dict[str, Any]] = {}
    for wid in store.window_ids():
        r = store.window(wid)
        row = per.setdefault(r["repo"], {**dict.fromkeys(STAGES, 0), "reasons": {}, "milestones": 0, "edges": 0, "mutants": {}})
        reached = STAGES.index(r.get("stage", "windows")) if r.get("stage") in STAGES else 0
        for s in STAGES[: reached + 1]:
            row[s] += 1
        key = re.split(r"[:(]", r.get("reason") or "accepted")[0].strip()[:60] or "accepted"
        row["reasons"][key] = row["reasons"].get(key, 0) + 1
        if r.get("stage") in ("dependent", "mutants_valid"):
            row["milestones"] += len(r.get("milestones") or [])
            row["edges"] += int(r.get("dependency_edges") or 0)
            for k, v in (r.get("mutant_counts") or {}).items():
                row["mutants"][k] = row["mutants"].get(k, 0) + v
    total = {s: sum(v[s] for v in per.values()) for s in STAGES}
    return {"split": store.split, "per_repo": per, "total": total}


# ---- prompt, ledger, lint -----------------------------------------------------------------------------------------------
EPILOGUE = ("How this works: you are working in a sandboxed copy of the repository with no network access. Hidden tests check the "
            "behaviour described above, and the repository's existing tests must keep passing; you can run the existing tests yourself. "
            "Before you finish, write STATUS.json at the repository root mapping each requirement id above to your probability (0 to 1) "
            "that it is met; it is scored against the hidden tests with the Brier score, so say what you believe. If you decide you "
            "cannot finish, end your final message with a line 'GIVE_UP: <reason>'; your work is graded as it stands, and an honest "
            "'not done' costs nothing extra.")
AUTHOR_BRIEF = ("You write the release request a maintainer of this Python library would send to a developer: what the next release "
                "must deliver, in the maintainer's voice. You get the commit messages, the changelog and a summary of each change (the "
                "public names it touched). Write one requirement per change id given, each one or two plain sentences that state the "
                "observable behaviour of the public API (inputs, outputs, errors, formats) the release must have, never how to "
                "implement it. Never write file paths, module file names, test names, commit ids, issue or pull request numbers, "
                "links, or process instructions (what tools to use or avoid). Return {opening, requirements: [{id, text}], closing}: "
                "opening is two or three sentences of context, closing is optional, requirements has exactly one entry per change id.")
REALISM_BRIEF = ("You review a maintainer's release request for a coding task. Reject it (ok false) when it gives hints a real "
                 "maintainer would not: it names files or tests, says how to implement something, quotes code from a patch, or "
                 "states rules the developer could not know from the request and the repository. Requirements that state "
                 "observable public behaviour are fine. Return {ok: bool, findings: [one sentence each]}.")
LINT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("process", re.compile(r"\b(do not|don't|never|avoid|without)\s+(using|use|importing|import|calling|call|reading|read|looking|look|consulting|consult|"
                           r"searching|search|installing|install|copying|copy|running|run|checking|check|editing|edit|touching|touch|modifying|modify)\b", re.IGNORECASE)),
    ("process", re.compile(r"\b(by hand|manually|from scratch|from memory)\b", re.IGNORECASE)),
    ("process", re.compile(r"\b(git (log|history|blame)|commit history|version history)\b", re.IGNORECASE)),
    ("file_path", re.compile(r"\b[\w./-]+\.(py|cfg|toml|ini|txt|rst|md|json|ya?ml)\b|\btests?/", re.IGNORECASE)),
    ("test_name", re.compile(r"\btest_\w+")),
    ("commit_id", re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,40}\b")),
    ("link", re.compile(r"https?://|\bgh-\d+|(?<![\w&])#\d+\b|\b(PR|pull request|issue) ?#?\d+", re.IGNORECASE)),
)


def lint_prompt(text: str, *, enforced: Sequence[str] = ()) -> list[str]:
    """Findings for the author's text (the fixed epilogue is ours and is not linted). A process instruction passes only
    when it matches one of ``enforced`` (patterns an outcome check enforces)."""
    out = []
    allowed = [re.compile(p, re.IGNORECASE) for p in enforced]
    for kind, rx in LINT_RULES:
        for m in rx.finditer(text):
            frag = text[max(0, m.start() - 30): m.end() + 30].replace("\n", " ")
            if kind == "process" and any(a.search(frag) for a in allowed):
                continue
            out.append(f"lint {kind}: {m.group(0)!r} in ...{frag}...")
    return out


def maintainer_text(prompt: str) -> str:
    """The author's part of a window prompt (everything before the fixed epilogue)."""
    return prompt.split("\n\nHow this works:", 1)[0]


def compose_window_prompt(opening: str, requirements: Sequence[Mapping[str, str]], closing: str = "", *, epilogue: str = EPILOGUE) -> str:
    body = "\n".join(f"- [{r['id']}] {r['text']}" for r in requirements)
    return (f"{opening.strip()}\n\nWhat the release needs:\n{body}" + (f"\n\n{closing.strip()}" if closing.strip() else "")
            + f"\n\n{epilogue}")


_DEF_LINE = re.compile(r"^[+-]\s*(?:async\s+)?(?:def|class)\s+([A-Za-z]\w*)", re.MULTILINE)


def public_names(patch: str) -> list[str]:
    """Public functions and classes a unit adds or changes (from its diff), never paths."""
    names = set(_DEF_LINE.findall(patch))
    names |= set(re.findall(r"^@@[^@]*@@\s*(?:async\s+)?(?:def|class)\s+([A-Za-z]\w*)", patch, re.MULTILINE))
    return sorted(n for n in names if not n.startswith("_") and not n.startswith("test"))[:20]


def changelog_excerpt(repo: RepoGit, a: str, b: str, *, limit: int = 3000) -> str:
    """Lines the window added to the changelog (CHANGES/CHANGELOG/HISTORY/NEWS), scrubbed."""
    files = [p for s, p in repo.changed(a, b) if re.search(r"(^|/)(changes|changelog|history|news|release[-_]?notes)[^/]*$", p, re.IGNORECASE)]
    out = []
    for p in files[:2]:
        diff = subprocess.run(["git", "-C", str(repo.dir), "diff", a, b, "--", p], capture_output=True, text=True, check=False).stdout
        out += [ln[1:] for ln in diff.splitlines() if ln.startswith("+") and not ln.startswith("+++")]
    return scrub("\n".join(out))[:limit]


def author_payload(rec: Mapping[str, Any], repo: RepoGit, base_dir: Path) -> dict[str, Any]:
    units = {u["sha"]: Unit(**u) for u in rec["units"]}
    readme = next(iter(sorted(Path(base_dir).glob("README*"))), None)
    changes = []
    for m in rec["milestones"]:
        u = units[m["unit"]]
        changes.append({"id": m["id"], "message": scrub(f"{u.subject}\n{u.body}")[:900],
                        "public_names": public_names(repo.patch(u).decode("utf-8", "replace"))})
    return {"library": rec["repo"], "readme_head": scrub(readme.read_text(errors="replace"))[:1500] if readme else "",
            "changelog": changelog_excerpt(repo, rec["a"], rec["b"]), "changes": changes}


def locate(prompt: str, text: str) -> tuple[int, int] | None:
    i = prompt.find(text)
    return (i, i + len(text)) if i >= 0 else None


def window_checkset(rec: Mapping[str, Any], requirements: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """One gating check per milestone (its tests, its requirement) plus the regression check."""
    obs = f"obs-{rec['id']}"
    by_ms = {r["milestone"]: r["id"] for r in requirements if r.get("milestone")}
    checks = [{"id": f"c_{m['id']}", "requirement_ids": [by_ms[m["id"]]] if m["id"] in by_ms else [], "observation_ids": [obs], "gating": True,
               "spec": {"tests": list(m["tests"])}} for m in rec["milestones"]]
    reg = sorted(set(rec["p2p"]) | set(rec["collateral"]))
    if reg and any(r["id"] == REGRESSION_REQ for r in requirements):
        checks.append({"id": "c_regress", "requirement_ids": [REGRESSION_REQ], "observation_ids": [obs], "gating": True, "spec": {"tests": reg}})
    return {"checks": checks}


def window_ledger(rec: Mapping[str, Any], requirements: Sequence[Mapping[str, Any]], spans: Mapping[str, Sequence[int]], prompt: str) -> tuple[Any, list[str]]:
    """The two-way ledger (``cards.RequirementLedger``) and its problems, plus the window's own: a requirement with no
    milestone, a milestone with no requirement."""
    from crucible.factory.cards import CheckRef, ObservationRecord, Requirement, RequirementLedger

    checks = window_checkset(rec, requirements)["checks"]
    led = RequirementLedger(requirements=[Requirement(id=r["id"], text=r["text"], kind=r.get("kind") or "behaviour",
                                                      prompt_span=tuple(spans.get(r["id"]) or (0, 1))) for r in requirements],
                            checks=[CheckRef(id=c["id"], requirement_ids=c["requirement_ids"], observation_ids=c["observation_ids"], gating=c["gating"])
                                    for c in checks])
    obs = ObservationRecord.model_validate(observation(rec))
    problems = led.problems({obs.id: obs}, prompt=prompt)
    mapped = {r.get("milestone") for r in requirements}
    problems += [f"unmapped_milestone: milestone {m['id']} has no requirement" for m in rec["milestones"] if m["id"] not in mapped]
    return led, problems


async def author_window(client: Any, rec: Mapping[str, Any], repo: RepoGit, base_dir: Path, *, findings: Sequence[str] = (), max_tokens: int = 16000,
                        sample_key: str | None = None) -> dict[str, Any]:
    """The release request: ``{prompt, requirements, spans, checkset, payload, findings}``. ``findings`` is non-empty when
    the draft must go back to the author (unmapped either way, lint, ledger)."""
    payload = author_payload(rec, repo, base_dir)
    user = {**payload, **({"findings_to_address": list(findings)} if findings else {})}
    msgs = [{"role": "system", "content": f"ROLE: author\n{AUTHOR_BRIEF}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(user, sort_keys=True)}]
    out = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0, **({"sample_key": sample_key} if sample_key else {}))
    problems: list[str] = []
    ids = [m["id"] for m in rec["milestones"]]
    reqs_in = out.get("requirements") if isinstance(out, dict) else None
    got = {str(r.get("id")): " ".join(str(r.get("text") or "").split()) for r in reqs_in or [] if isinstance(r, dict)}
    problems += [f"unknown_requirement_id: {k} is not a change id" for k in got if k not in ids]
    problems += [f"unmapped_milestone: change {k} has no requirement" for k in ids if not got.get(k)]
    requirements = [{"id": f"R{i}", "milestone": mid, "text": got[mid], "kind": "behaviour"} for i, mid in enumerate(ids, 1) if got.get(mid)]
    opening = str((out or {}).get("opening") or "").strip()
    closing = str((out or {}).get("closing") or "").strip()
    if not opening:
        problems.append("author.opening is empty")
    authored = "\n".join([opening, closing, *[r["text"] for r in requirements]])
    problems += lint_prompt(authored)
    requirements.append({"id": REGRESSION_REQ, "milestone": None, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
    prompt = compose_window_prompt(opening, [r for r in requirements if r["id"] != REGRESSION_REQ], closing + ("\n\n" if closing else "") + REGRESSION_TEXT)
    spans = {r["id"]: list(loc) for r in requirements if (loc := locate(prompt, r["text"]))}
    _, ledger_problems = window_ledger(rec, requirements, spans, prompt)
    problems += ledger_problems
    return {"prompt": prompt, "requirements": requirements, "spans": spans, "checkset": window_checkset(rec, requirements), "payload": payload,
            "findings": problems, "opening": opening, "closing": closing}


async def review_realism(client: Any, prompt: str, *, max_tokens: int = 8000) -> dict[str, Any]:
    """The maintainer's text only: the fixed epilogue is the lane's truthful description of the episode (study 08 §3: the
    model may know it is in an environment), not the author's, and a reviewer judging it as a hint was a false finding
    on every smoke-test window."""
    msgs = [{"role": "system", "content": f"ROLE: realism\n{REALISM_BRIEF}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps({"request": maintainer_text(prompt)})}]
    out = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0)
    return {"ok": bool(isinstance(out, dict) and out.get("ok") is True), "findings": [str(x) for x in (out or {}).get("findings") or []]}


# ---- the task ----------------------------------------------------------------------------------------------------------
class WindowTask(ExecTask):
    """One window under one prompt: ``ExecTask`` with the window's grader, milestones, Phi, mutants and World."""

    horizon = "H2"

    def __init__(self, lane: WindowLane, mined: MinedTask, window: Mapping[str, Any], *, task_id: str, prompt: str, requirements: Sequence[Mapping[str, Any]],
                 spans: Mapping[str, Sequence[int]], checkset: Mapping[str, Any], weights: Mapping[str, float] | None = None):
        self.window = dict(window)
        self.sandbox = lane.sandbox_for(self.window["world"]["image"])
        super().__init__(lane, mined, task_id=task_id, prompt=prompt, requirements=requirements, spans=spans, checkset=checkset)
        self.milestones = [dict(m) for m in self.window["milestones"]]
        w = dict(weights or {})
        raw = {m["id"]: float(w.get(m["id"], 1.0)) for m in self.milestones}
        total = sum(raw.values()) or 1.0
        self.weights = {k: v / total for k, v in raw.items()}
        self.oracle = self.oracle_steps()
        self.verifier_timeout_s = self.sandbox.timeout_s
        self.id = self.task_id  # the runner's key (runtime.run)
        self.world_id = f"factory/window/{self.window['repo']}"
        # the checkpoint sidecar's contract (factory.checkpoint.MilestoneTask): the prompt's requirement ids, each graded
        # through the milestones the ledger maps it to (STATUS.json is Brier-scored on these)
        self.requirement_milestones = {str(r["id"]): [str(r["milestone"])] for r in self.requirements if r.get("milestone")}
        self.requirement_ids = list(self.requirement_milestones)
        self.checkpoint: Any = None  # a factory.checkpoint.CheckpointConfig turns the sidecar on (off by default)
        self._checkpoint_cache: dict[str, Any] = {}

    # -- the oracle as steps (horizon replay, shortcut probe) -------------------------------------------------------
    def oracle_steps(self) -> list[dict[str, Any]]:
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]])) if self.window["repo"] in self.lane.repo_dirs else None
        if repo is None:  # no clone at hand: the whole window as one step
            return [{"function_name": "apply_files", "arguments": {"files": dict(self.window["oracle_files"])}}]
        return [{"function_name": "apply_unit", "arguments": {"unit": u["sha"], "patch": repo.patch(Unit(**u)).decode("utf-8", "replace")}}
                for u in self.window["units"] if repo.patch(Unit(**u)).strip()]

    def _files(self, solution: Any) -> tuple[dict[str, str | None], list[str], list[str]]:
        if isinstance(solution, list) and solution and all(isinstance(s, Mapping) and s.get("function_name") == "apply_files" for s in solution):
            files: dict[str, str | None] = {}
            for s in solution:
                files.update((s.get("arguments") or {}).get("files") or {})
            return {p: t for p, t in files.items() if not is_protected(p)}, sorted(p for p in files if is_protected(p)), []
        return solution_files(self.base_dir, solution)

    def end_fingerprint(self, solution: Any) -> str:
        files, _, errs = self._files(solution)
        return fingerprint(self._before(files), files) if not errs else sha256_text("unapplied", json.dumps(errs))

    # -- grading ---------------------------------------------------------------------------------------------------
    def phi(self, status: Mapping[str, bool]) -> float:
        """Phi(s) = sum of w_m over milestones that pass with every milestone in ``requires`` passing."""
        return round(sum(self.weights[m["id"]] for m in self.milestones if status.get(m["id"]) and all(status.get(r) for r in m.get("requires") or [])), 6)

    async def grade(self, solution: Any, *, run: int = 0, checks: Mapping[str, Sequence[str]] | None = None) -> tuple[Any, str]:
        from crucible.verify.runtime.plan import CheckResult, GradeReport

        check_tests = dict(checks) if checks is not None else self.check_tests
        t0 = time.monotonic()
        files, protected, errors = await asyncio.to_thread(self._files, solution)
        fp = fingerprint(self._before(files), files) if not errors else sha256_text("unapplied", json.dumps(errors))
        statuses: dict[str, str] = {}
        why, tail = "", ""
        if errors:
            why = "the solution did not apply: " + "; ".join(errors[:3])
        else:
            targets = sorted({x.split("::", 1)[0] for ts in check_tests.values() for x in ts})
            tr = await run_tests(self.sandbox, self.base_dir, files, self.mined.tests, targets, self.window.get("tests_deleted") or ())
            if tr.infra != "ok":
                rep = GradeReport(task=self.task_id, tier="full", checks=[], gates={c: "infra" for c in check_tests}, reward=0.0, outcome=0.0,
                                  infra={"status": "masked", "why": tr.why}, trusted=False, elapsed_s=time.monotonic() - t0, meta={"why": tr.why})
                return rep, fp
            statuses, why, tail = tr.results, tr.why, tr.tail
        gates, results, failing = {}, [], []
        for cid, tests in check_tests.items():
            bad = [x for x in tests if statuses.get(x) != "passed"]
            failing += bad
            gates[cid] = "pass" if tests and not bad else "fail"
            results.append(CheckResult(id=cid, gate=cid, status=gates[cid], detail={"failing": bad[:20], "tests": len(tests)}))
        gating = [c for c in check_tests if self.gating.get(c, True)]
        reward = 1.0 if gating and all(gates[c] == "pass" for c in gating) else 0.0
        status = {m["id"]: passed_group(statuses, m["tests"]) for m in self.milestones}
        meta = {"failing_tests": sorted(set(failing))[:60], "statuses": {s: sum(1 for v in statuses.values() if v == s) for s in set(statuses.values())},
                "tail": tail[-2500:], "apply_errors": errors, "protected_writes": protected, "tests_intact": True, "why": why,
                "kinds": {c: "outcome" for c in check_tests}, "milestones": status, "phi": self.phi(status)}
        self.lane.note_grade(self.task_id, solution, meta)
        return GradeReport(task=self.task_id, tier="full", checks=results, gates=gates, reward=reward, outcome=reward, infra={"status": "ok"},
                           trusted=True, elapsed_s=time.monotonic() - t0, meta=meta), fp

    async def grade_patch(self, patch: str, workdir: Path | None = None) -> Any:
        """``WorkspaceHarness``'s verifier call: the collected patch graded from a pristine A. ``meta['milestones']`` is the
        pass vector, ``meta['phi']`` the potential."""
        rep, _ = await self.grade(patch)
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            (Path(workdir) / "window_grade.json").write_text(json.dumps({"reward": rep.reward, "milestones": rep.meta.get("milestones"),
                                                                         "phi": rep.meta.get("phi"), "infra": rep.infra}, indent=1))
        return rep

    async def milestone_status(self, patch: str) -> dict[str, bool]:
        """For the checkpoint sidecar: milestone id -> its tests all pass on A + ``patch`` (``requires`` not applied;
        ``phi(status)`` applies it). An unmeasured grade raises ``RuntimeError`` (never read as all-fail)."""
        rep, _ = await self.grade(patch, checks={f"c_{m['id']}": m["tests"] for m in self.milestones})
        if rep.infra.get("status") != "ok":
            raise RuntimeError(f"milestone status unmeasured: {rep.infra.get('why')}")
        return dict(rep.meta["milestones"])

    async def milestones_of(self, prefix: Any) -> dict[str, bool]:
        """``horizon.series_by_replay``: the milestones holding after an oracle prefix."""
        rep, _ = await self.grade(list(prefix), checks={f"c_{m['id']}": m["tests"] for m in self.milestones})
        return dict(rep.meta.get("milestones") or {})

    # -- mutants, audit, environment -----------------------------------------------------------------------------------
    def mutants(self) -> list[Any]:
        """The commit-revert mutants the verifier must reject: ``valid`` (wrong on exactly milestone k) and ``broad``
        (wrong on k and more)."""
        from crucible.accept.mutation.base import Mutant

        return [Mutant(id=m["id"], family="commit_revert", solution=m["solution"], description=f"B with the unit of {m['milestone']} reverted")
                for m in self.window.get("mutants") or [] if m.get("status") in ("valid", "broad")]

    async def property_audit(self, actions: Any) -> bool:
        files, protected, errors = await asyncio.to_thread(self._files, actions)
        if protected or errors:
            return False
        return await ExecTask.property_audit(self, {"files": files})

    async def isolation_audit(self) -> dict[str, str]:
        lane_sb = self.lane.sandbox
        self.lane.sandbox = self.sandbox
        try:
            return await ExecTask.isolation_audit(self)
        finally:
            self.lane.sandbox = lane_sb

    def key_inputs(self) -> dict[str, Any]:
        return {"overlay_hash": self.task_hash, "images": {"grader": self.sandbox.image, "world": self.sandbox.image}}

    def pinned_images(self) -> dict[str, str]:
        d = str(self.window["world"].get("digest") or "")
        return {"grader": d, "world": d}

    def declared_env(self) -> dict[str, Any]:
        return {"kind": "container", **self.sandbox.describe()}

    def facts(self) -> dict[str, Any]:
        return {"lane": LANE, "repo": self.window["repo"], "a": self.window["a"], "b": self.window["b"], "milestones": len(self.milestones),
                "dependency_edges": self.window.get("dependency_edges")}

    def grade_env(self) -> str:
        return sha256_text(str(self.window["world"].get("digest")), self.mined.id, json.dumps(self.check_tests, sort_keys=True))[:24]

    # -- the coding seam (WorkspaceHarness) -------------------------------------------------------------------------------
    async def open(self, workdir: Path) -> Any:
        return await open_world(image=self.window["world"]["image"], base_dir=self.base_dir, workdir=Path(workdir), instruction=self.instruction)

    def workspace(self, world: Any) -> Path:
        return world._crucible_ws

    async def broken(self, world: Any) -> list[str]:
        return [] if (self.workspace(world) / ".git").exists() else ["the workspace repository (.git) was deleted"]

    async def snapshot_patch(self, world: Any) -> str:
        """The workspace as a patch against ``base`` (new files included), read inside the World's container through a
        private index: the agent's own index and history are never touched."""
        from crucible.world.exec import DockerExec

        ws = str(self.workspace(world))
        cmd = ("set -e; I=$(mktemp); cp .git/index \"$I\" 2>/dev/null || true; GIT_INDEX_FILE=\"$I\" git add -A >/dev/null 2>&1; "
               "GIT_INDEX_FILE=\"$I\" git -c core.quotepath=off diff --cached --binary base; rm -f \"$I\"")
        out = await DockerExec(world._crucible_container).run(cmd, cwd=ws, timeout_s=300)
        if not out.ok:
            raise RuntimeError(f"workspace snapshot failed: {out.stderr[-300:]}")
        return out.stdout

    def episode_sidecar(self, attempt: Any) -> Any:
        """``WorkspaceHarness`` calls this per attempt: the checkpoint-Phi sidecar when ``self.checkpoint`` is set (a
        ``CheckpointConfig``), else None (off: the episode runs unchanged)."""
        if self.checkpoint is None:
            return None
        from crucible.factory.checkpoint import CheckpointSidecar

        return CheckpointSidecar(self, attempt.dir, self.checkpoint, baseline=self._checkpoint_cache)


class WindowLane(ExecLane):
    """The window lane for one factory run: the window store, a grading sandbox per World image, the tasks
    materialized from drafts, and the lane's machine handlers."""

    def __init__(self, store: WindowStore, *, repo_dirs: Mapping[str, str] | None = None, reviewer: Any = None, weights: Mapping[str, Mapping[str, float]] | None = None,
                 sandboxes: Callable[[str], DockerSandbox] | None = None, **kw: Any):
        self._sandboxes = sandboxes or sandbox_factory()
        super().__init__(store, DockerSandbox(BASE_IMAGE, guard=True), repo_dirs=repo_dirs, reviewer=reviewer, **kw)
        self.image_digest = ""
        self.weights = {k: dict(v) for k, v in (weights or {}).items()}
        self.window_tasks: dict[str, WindowTask] = {}

    def sandbox_for(self, image: str) -> DockerSandbox:
        return self._sandboxes(image)

    def world_of(self, t: MinedTask) -> dict[str, Any]:
        w = self.store.window(t.id)
        return {"lane": LANE, "task": t.id, "repo": t.repo, "start": t.parent, "end": t.sha, "image": w["world"]["image"], "milestones": len(w["milestones"])}

    def materialize(self, draft: Mapping[str, Any]) -> WindowTask:
        mined = self.mined_of(draft)
        win = self.store.window(mined.id)
        reqs = list((draft.get("spec") or {}).get("requirements") or [])
        spans = dict(draft.get("spans") or {})
        checkset = dict(draft.get("checkset") or window_checkset(win, reqs))
        prompt = str(draft.get("prompt") or "")
        key = sha256_text(mined.id, prompt, json.dumps([reqs, spans, checkset], sort_keys=True, default=str))[:24]
        if key not in self.window_tasks:
            tid = str(draft.get("task_id") or mined.id)
            self.window_tasks[key] = WindowTask(self, mined, win, task_id=f"{tid}~{key[:6]}", prompt=prompt, requirements=reqs, spans=spans,
                                                checkset=checkset, weights=self.weights.get(mined.id))
        return self.window_tasks[key]

    async def full_suite(self, task: ExecTask, solution: Any) -> dict[str, str] | None:
        wt = task if isinstance(task, WindowTask) else None
        if wt is None:
            return await super().full_suite(task, solution)
        key = (task.mined.id, task.end_fingerprint(solution))
        if key not in self._suites:
            files, _protected, errs = await asyncio.to_thread(wt._files, solution)
            if errs:
                self._suites[key] = None
            else:
                tr = await run_tests(wt.sandbox, wt.base_dir, files, wt.mined.tests, wt.window["targets"], wt.window.get("tests_deleted") or ())
                self._suites[key] = tr.results if tr.infra == "ok" else None
        return self._suites[key]

    def handlers(self) -> dict[str, Any]:
        """spec + author (the release request, ledger-checked), realism (lint + reviewer), oracle (B as unit steps),
        verifier and second verifier (the deterministic milestone map: the checks are the repository's tests)."""
        from crucible.actors.llm import LLMOutputError
        from crucible.factory.driver import verdict
        from crucible.factory.machine import Settle

        lane = self

        def window_of(ctx: Any) -> tuple[MinedTask, dict[str, Any]]:
            t = lane.mined_from_cell(ctx.inputs.get("cell") or ctx.params.get("cell") or {})
            return t, lane.store.window(t.id)

        async def s_spec(ctx: Any) -> Any:
            t, w = window_of(ctx)
            repo = RepoGit(Path(lane.repo_dirs[t.repo]))
            kw = {"sample_key": f"{ctx.state_id}#{ctx.entry}.{ctx.attempt}"} if ctx.attempt > 1 else {}
            out = await author_window(ctx.client, w, repo, Path(t.base_dir), findings=ctx.findings, **kw)
            if out["findings"]:
                raise LLMOutputError("; ".join(out["findings"])[:800])
            spec = {"requirements": out["requirements"], "milestones": [{"id": m["id"], "requires": m.get("requires") or []} for m in w["milestones"]],
                    "long": True, "verifier_plan": "the window's fail-to-pass tests per milestone plus pass-to-pass regression tests",
                    "expected_causes": [], "surface": {"domain": "python library", "world": t.repo, "tools": "shell workspace", "data_shape": "repository",
                                                       "prompt_style": "release request"}, "draft_prompt": out["prompt"], "draft_spans": out["spans"]}
            return Settle({"spec": spec})

        async def s_author(ctx: Any) -> Any:
            t, _ = window_of(ctx)
            spec = ctx.inputs["spec"]
            return Settle({"world": lane.world_of(t), "prompt": spec["draft_prompt"], "spans": spec["draft_spans"]})

        async def s_realism(ctx: Any) -> Any:
            prompt = ctx.inputs["prompt"]
            det = lint_prompt(maintainer_text(prompt))
            rev = await review_realism(ctx.client, prompt)
            v = verdict(rev["ok"] and not det, owner="author", findings=det + [f"realism: {f}" for f in rev["findings"]])
            return Settle({"verdict": v}, v["findings"])

        async def s_oracle(ctx: Any) -> Any:
            t, w = window_of(ctx)
            task = lane.materialize({"world": {"task": t.id}, "prompt": "", "spec": {"requirements": []}, "checkset": {"checks": []}})
            return Settle({"oracle": {"solution": task.oracle, "notes": f"the real window {t.repo} {w['a_ref']}..{w['b_ref']}"}})

        def make_verifier(second: bool) -> Any:
            async def s_verify(ctx: Any) -> Any:
                _, w = window_of(ctx)
                return Settle({"checkset": window_checkset(w, ctx.inputs["spec"]["requirements"])})

            return s_verify

        return {"spec": s_spec, "author": s_author, "realism": s_realism, "oracle": s_oracle, "verifier": make_verifier(False),
                "second_verifier": make_verifier(True)}


def open_store(root: Path, repo: str) -> WindowStore:
    """The store a repository's windows belong in: ``<root>/train`` or ``<root>/heldout``."""
    split = split_of(repo)
    return WindowStore(Path(root) / split, split=split)

