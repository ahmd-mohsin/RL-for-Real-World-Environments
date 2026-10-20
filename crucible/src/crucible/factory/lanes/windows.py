"""The window lane (study 08 §0-§5, §9 step 2; report 09): long-horizon coding tasks from real release windows.

A window runs from release A to release B of a real repository (or across a run of units that build on each other).
The repository at A is the start state, B is an executed oracle, the tests the window added are the checks, and each
unit's own tests are a milestone in a known real order. ``windows_git`` mines the windows and builds trees,
``windows_world`` builds the per-(repo, A) image, runs the leak probes and opens the policy's World.

**Validation** (``validate_window``, the SWE-smith pattern of ``executable.validate_candidate`` at window scale): the
repository's suite runs at A (with B's test files) and at B, ``reruns`` times each in the World image. Fail-to-pass tests
are candidate checks, pass-to-pass tests regression checks; anything flaky is dropped and a test that passed at A and
fails at B is dropped (the window removed that behaviour). Fail-to-pass tests are grouped by their owner unit
(``prefix_owners``: the first unit after whose prefix A + u1..uk the test passes under B's tests, <= 25 runs by bisection;
report 11 F3): one milestone per owner unit, in history order; tests no prefix makes pass are collateral regression
checks.

**Executed dependency prober** (``probe_dependencies``): milestone k is A plus only unit k's diff, run on k's tests in
B's test files (as the grader runs them). If they fail, the minimal set of earlier units that makes them pass is found by
removal from the full prefix (ddmin's complement step: halves first, then one at a time). Each edge is labelled
``patch-conflict`` (without that unit, unit k's diff no longer applies) or ``executed``. Only ``executed`` edges are
behavioural and gate Phi (``requires``); ``patch-conflict`` edges are ``order_hints``, metadata only (report 11 F2). A
window with no behavioural edge is ``independent`` (long, not long-horizon) and is excluded from Type A.

**Commit-revert mutants** (``revert_mutants``): B with only unit k's diff reverted (3-way; skipped on conflict), graded
by the window's verifier. Expected: milestone k fails, every milestone that does not depend on unit k still passes.

**The task** (``WindowTask``, an ``executable.ExecTask``): the accept contract and ``driver.LANE_TASK`` (grade, noop,
oracle, instruction, negatives, mutants, floor_kwargs, agent_visible, hidden_strings, pinned_images, property_audit),
plus the coding seam of ``runtime.harness.workspace.WorkspaceHarness`` (``open``, ``workspace``, ``broken``,
``grade_patch``) and ``milestone_status(patch)`` for the checkpoint sidecar. Grading applies the solution to a pristine
A tree (protected paths are never written), overwrites every test file from B, runs the hidden fail-to-pass and
pass-to-pass tests in a separate ``--network none`` container and reads the guard's JUnit report. The outcome reward is
``phi_weight * Phi + (1 - phi_weight) * [every gating check passes]`` (default 0.5; report 11 F1), so it is 1 only for a
full pass and 0 for every floor; ``meta`` carries the per-milestone pass vector, Phi (weighted milestones whose behavioural
``requires`` all pass; uniform weights unless injected) and ``full_pass``.

**Prompt and ledger** (``author_window``): a maintainer's release request written from the units' messages, the
changelog and a diff summary (public names only; never test bodies, never file paths), one requirement id per
milestone, then a fixed epilogue that states truthfully how the episode works (sandbox, offline, hidden tests, existing
tests, ``GIVE_UP:``, ``STATUS.json``). The ledger maps each requirement to its milestone's tests both ways; ``lint_prompt``
refuses process instructions no check enforces, file paths, test names and commit ids.

**P0 (swarm-pipeline/00 §15):** ``WindowTask`` is a ``factory.longtask.LongTask`` (``world``, ``path_rules``,
``oracle_patch``, ``floors``, ``alternatives``, ``simplify('prerequisites')``, ``card``, ``ref``); ``grade_patch`` adds the
path gates of ``factory.pathcheck`` (the epilogue states them); windows mined from P0 on are import-tolerant
(``grader_tests``, ``windows_imports``; report 11 F8); findings sent back to the author are test-derived
(``findings_from_tests``; D8); the pass-stub floor (``stub_files``) moved here from ``scripts/lh_run.py``.
"""

from __future__ import annotations

import ast
import asyncio
import contextvars
import json
import re
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from crucible.accept.mutation.base import Mutant
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
from crucible.factory.lanes.windows_git import RepoGit, Unit, Window, _defs, mine_windows
from crucible.factory.lanes.windows_imports import project_roots, tolerant_tests
from crucible.factory.lanes.windows_timing import timing_tests
from crucible.factory.lanes.windows_world import BASE_IMAGE, ImageBook, leak_probes, open_world, project_name
from crucible.factory.longtask import GroundCase, LongTaskRef, status_claim_patch
from crucible.factory.pathcheck import PathRules, RequiredAction, ToolCall, check_paths, find_trail

LANE = "window"
WINDOW_MINER = "miner:m17-window-lane"
SMALL_REPOS = ("python-semver", "parse", "boltons", "cachetools", "humanize", "more-itertools")
# medium pure-Python repositories with real releases and work that builds on itself (09 §2 gives the reasons)
POOL_REPOS = ("click", "marshmallow", "pyparsing", "lark", "markdown", "arrow", "glom", "toolz", "funcy", "cerberus", "typer", "dateutil")
# chosen before any mining: never written into a training store (H1: same repo class; H2: a larger app, other styles)
HELD_OUT_REPOS: dict[str, str] = {"sqlparse": "H1", "tomlkit": "H1", "python-dotenv": "H1", "voluptuous": "H1", "jinja": "H2", "mistune": "H2"}
# benchmark v2 (2026-10-18), chosen before mining (factory/benchmark.py BENCH_V2 records why): organisations disjoint from
# every training source, permissive licence, pure Python, pytest, windows that end after the training cutoff. jinja stays
# held out (its windows were seen as held-out) but leaves the benchmark: pallets also owns click (training)
HELD_OUT_REPOS.update({"fsspec": "H2", "filelock": "H1", "sphinx-autodoc-typehints": "H2", "w3lib": "H1", "starlette": "H2", "pyjwt": "H1",
                       "jsonpickle": "H1", "peewee": "H2", "streamlink": "H2", "pydash": "H1", "pint": "H2", "dateparser": "H2", "graphql-core": "H2",
                       "alembic": "H2",
                       # reserves, mined only if the selection above seals fewer than 20 tasks
                       "parsel": "H1", "validators": "H1", "huey": "H1", "tenacity": "H1", "cattrs": "H1", "loguru": "H1", "nox": "H2",
                       "litestar": "H2", "tablib": "H1", "jsonpath-ng": "H1"})
REPO_URLS = {
    "python-semver": "python-semver/python-semver", "parse": "r1chardj0n3s/parse", "boltons": "mahmoud/boltons", "cachetools": "tkem/cachetools",
    "humanize": "python-humanize/humanize", "more-itertools": "more-itertools/more-itertools", "click": "pallets/click",
    "marshmallow": "marshmallow-code/marshmallow", "pyparsing": "pyparsing/pyparsing", "lark": "lark-parser/lark", "markdown": "Python-Markdown/markdown",
    "arrow": "arrow-py/arrow", "glom": "mahmoud/glom", "toolz": "pytoolz/toolz", "funcy": "Suor/funcy", "cerberus": "pyeve/cerberus",
    "typer": "fastapi/typer", "dateutil": "dateutil/dateutil", "sqlparse": "andialbrecht/sqlparse", "tomlkit": "python-poetry/tomlkit",
    "python-dotenv": "theskumar/python-dotenv", "voluptuous": "alecthomas/voluptuous", "jinja": "pallets/jinja", "mistune": "lepture/mistune",
    "fsspec": "fsspec/filesystem_spec", "filelock": "tox-dev/filelock", "sphinx-autodoc-typehints": "tox-dev/sphinx-autodoc-typehints",
    "w3lib": "scrapy/w3lib", "starlette": "Kludex/starlette", "pyjwt": "jpadilla/pyjwt", "jsonpickle": "jsonpickle/jsonpickle",
    "peewee": "coleifer/peewee", "streamlink": "streamlink/streamlink", "pydash": "dgilland/pydash", "pint": "hgrecco/pint",
    "dateparser": "scrapinghub/dateparser", "graphql-core": "graphql-python/graphql-core", "alembic": "sqlalchemy/alembic",
    "parsel": "scrapy/parsel", "validators": "python-validators/validators", "huey": "coleifer/huey", "tenacity": "jd/tenacity",
    "cattrs": "python-attrs/cattrs", "loguru": "Delgan/loguru", "nox": "wntrblm/nox", "litestar": "litestar-org/litestar",
    "tablib": "jazzband/tablib", "jsonpath-ng": "h2non/jsonpath-ng",
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


def normalize_ids(results: Mapping[str, str], targets: Sequence[str]) -> dict[str, str]:
    """Test ids keyed by the file pytest collected them from (P2; the P0 open item on cachetools and markdown). pytest's
    xunit1 ``file`` attribute names the file that *defines* a test method, so a test a class inherits from a mixin in
    ``tests/__init__.py`` or ``markdown/test_tools.py`` came back as ``tests/__init__.py::tests::test_cache::CacheTest::t``:
    an id whose file runs nothing (or the wrong file: ``markdown/test_tools.py`` is itself a suite target), so a regression
    check made of such ids failed on the oracle. The parser then put the whole dotted class path after the file; when the
    longest dotted prefix of that path is one of the run's ``targets``, it becomes the file
    (``tests/test_cache.py::CacheTest::t``). A well-formed id (``file::Class::test``) never has a target file inside its
    class path and is kept, as is an id no target explains."""
    files = {t for t in targets if t.endswith(".py")}
    out: dict[str, str] = {}
    for tid, st in results.items():
        head, bracket, tail = tid.partition("[")
        parts = head.split("::")
        if len(parts) < 3:
            out[tid] = st
            continue
        mid, name = parts[1:-1], parts[-1]
        new = tid
        for i in range(len(mid), 0, -1):
            f = "/".join(mid[:i]) + ".py"
            if f in files:
                new = "::".join([f, *mid[i:], name]) + bracket + tail
                break
        if new not in out or out[new] == "passed":  # a collision keeps the worse status
            out[new] = st
    return out


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
        tr = await sandbox.run(tree, list(targets))
        tr.results = normalize_ids(tr.results, targets)
        return tr
    finally:
        shutil.rmtree(work, ignore_errors=True)


def grader_tests(rec: Mapping[str, Any], base_dir: Path | None = None) -> tuple[dict[str, str], dict[str, list[str]]]:
    """B's test files as the grader writes them over a tree, and the imports guarded per test file. A window mined
    import-tolerant (``rec["import_tolerant"]``, report 11 F8) gets ``windows_imports.tolerant_tests``: a module-level
    import of a name the tree lacks fails only the tests that use it. Every run of the window (fail-to-pass, ownership,
    prober, mutants, grading) uses the same files, so the window is classified and graded under one mode."""
    tests = dict(rec.get("tests_b") or {})
    if not rec.get("import_tolerant"):
        return tests, {}
    roots = set(rec.get("import_roots") or []) or project_roots(Path(base_dir) if base_dir else Path("/nonexistent"), list(rec.get("oracle_files") or {}))
    return tolerant_tests(tests, roots)


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


# ---- test ownership by prefix (report 11 F3) -----------------------------------------------------------------------------
def candidate_units(units: Sequence[Unit]) -> list[int]:
    """Positions of units that can change a test's outcome: they change a writable file other than prose (a unit that
    only edits docs or the changelog leaves every prefix's test results as they were)."""
    prose = re.compile(r"\.(md|rst|txt)$|(^|/)(docs?|changes|changelog|history|news)(/|$)", re.IGNORECASE)
    return [i for i, u in enumerate(units) if any(not prose.search(p) for p in u.writable_files)]


async def bisect_owners(n: int, tests: Sequence[str], run: Callable[[int], Any], *, budget: int = 25) -> dict[str, Any]:
    """For each test, the first of ``n`` ordered prefixes (0..n-1) at which it passes, with at most ``budget`` calls of
    ``run(i) -> set of passing tests``. Prefix -1 (A) is taken as failing every test; prefix n-1 is run first. Each next
    run splits the most unresolved intervals (lo, hi]; a test left unresolved when the budget ends is owned by its ``hi``
    (the first prefix seen passing). Assumes a test, once passing, keeps passing (counted when violated)."""
    seen: dict[int, set[str]] = {}

    async def at(i: int) -> set[str]:
        if i not in seen:
            seen[i] = set(await run(i))
        return seen[i]

    last = await at(n - 1)
    lo = {t: -1 for t in tests if t in last}
    hi = {t: n - 1 for t in lo}
    while len(seen) < budget:
        open_ = [t for t in lo if hi[t] - lo[t] > 1]
        if not open_:
            break
        counts: dict[int, int] = {}
        for t in open_:
            mid = (lo[t] + hi[t]) // 2
            counts[mid] = counts.get(mid, 0) + 1
        m = max(counts, key=lambda k: (counts[k], -abs(k - (n - 1) / 2)))
        ok = await at(m)
        for t in open_:
            if lo[t] < m < hi[t]:
                if t in ok:
                    hi[t] = m
                else:
                    lo[t] = m
    violations = sum(1 for t in hi for i, ok in seen.items() if i > hi[t] and t not in ok)
    return {"owner_prefix": dict(hi), "resolved": {t: hi[t] - lo[t] == 1 for t in hi}, "never": sorted(set(tests) - set(hi)), "runs": len(seen),
            "monotone_violations": violations}


async def prefix_owners(rec: Mapping[str, Any], repo: RepoGit, base: Path, sandbox: DockerSandbox, tests: Sequence[str], *, budget: int = 25
                        ) -> dict[str, Any]:
    """Owner of each fail-to-pass test = the first unit of the window after whose prefix (A + u1..uk, every writable file)
    the test passes under B's test files, as the grader runs it. A unit that never changed the test can own it; a
    unit that only rewrote it cannot (report 11 F3: parse's "convert tests to pytest" owned tests an earlier unit had made
    pass). ``{"owners": {test: sha | None}, ...stats}``."""
    units = [Unit(**u) for u in rec["units"]]
    cand = candidate_units(units)
    files_b = sorted({t.split("::", 1)[0] for t in tests})
    tests_g = grader_tests(rec, base)[0]

    def prefix_files(sha: str) -> dict[str, str | None]:
        """The agent-writable files that differ between A and unit k's commit (units are first-parent steps, so A + u1..uk
        is that commit: read, never re-applied, so a unit whose diff does not apply in isolation cannot break it)."""
        out: dict[str, str | None] = {}
        for st, p in repo.changed(rec["a"], sha):
            if is_protected(p) or is_test_path(p):
                continue
            if st == "D":
                out[p] = None
            elif (txt := repo.show(sha, p)) is not None:
                out[p] = txt
        return out

    async def run(i: int) -> set[str]:
        files = await asyncio.to_thread(prefix_files, units[cand[i]].sha)
        tr = await run_tests(sandbox, base, files, tests_g, files_b, rec.get("tests_deleted") or ())
        return {t for t, st in tr.results.items() if st == "passed"} if tr.infra == "ok" else set()

    if not cand:
        return {"owners": dict.fromkeys(tests), "runs": 0, "candidates": 0}
    res = await bisect_owners(len(cand), list(tests), run, budget=budget)
    owners: dict[str, str | None] = {t: units[cand[i]].sha for t, i in res["owner_prefix"].items()}
    owners.update(dict.fromkeys(res["never"]))
    return {"owners": owners, "runs": res["runs"], "candidates": len(cand), "unresolved": sorted(t for t, r in res["resolved"].items() if not r),
            "monotone_violations": res["monotone_violations"]}


def milestones_from_owners(rec: Mapping[str, Any], owners: Mapping[str, str | None]) -> list[dict[str, Any]]:
    """One milestone per owner unit, in history order (ids ``m<unit index+1>``)."""
    units = [Unit(**u) for u in rec["units"]]
    order = {u.sha: i for i, u in enumerate(units)}
    groups: dict[str, list[str]] = {}
    for t, o in owners.items():
        if o is not None:
            groups.setdefault(o, []).append(t)
    return [{"id": f"m{order[s] + 1:02d}", "unit": s, "index": order[s], "subject": units[order[s]].subject, "tests": sorted(ts)}
            for s, ts in sorted(groups.items(), key=lambda kv: order[kv[0]])]


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
        # B's test files, as the grader runs them (report 11 F3: ownership is decided the same way); an import of a name an
        # earlier unit adds is then a real prerequisite of the milestone as the grader sees it
        tests_k = grader_tests(rec)[0]  # import-tolerant when the window is (F8): roots recorded at f2p
        gone_k = list(rec.get("tests_deleted") or [])
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
                        tr.results = normalize_ids(tr.results, test_files)
                        runs += 1
                        cache[key] = (tr.infra == "ok" and passed_group(tr.results, m["tests"]), "run" if tr.infra == "ok" else "masked")
                finally:
                    shutil.rmtree(work, ignore_errors=True)
            return cache[key]

        async def ok(s: Sequence[int]) -> bool:
            return (await run(s))[0]

        earlier = list(range(i))
        if await ok([]):
            m.update(requires=[], requires_units=[], order_hints=[], order_units=[], edges={}, probe="alone")
            return
        if not earlier or not await ok(earlier):
            # no edge is measured: nothing gates Phi; every earlier milestone is an order hint only (report 11 F2)
            m.update(requires=[], requires_units=[], order_hints=[x["id"] for x in rec["milestones"] if index[x["unit"]] < i],
                     order_units=[units[j].sha for j in earlier], edges={}, probe="prefix_fails")
            return
        s = await ddmin(earlier, ok)
        edges = {}
        for j in s:
            without = [x for x in s if x != j]
            ok_build = (await asyncio.to_thread(_applies, repo, rec["a"], [units[x] for x in without] + [u]))
            edges[units[j].sha] = "executed" if ok_build else "patch-conflict"
        # only behavioural edges gate Phi: without unit j, unit k's change applies and its tests fail. A patch-conflict edge
        # (unit k's diff edits code unit j wrote) is textual: an order hint, metadata only (report 11 F2: toolz m14 passed
        # 4/4 in context while its patch-conflict "prerequisites" never did, and C1 credited nothing)
        beh = [j for j in s if edges[units[j].sha] == "executed"]
        txt = [j for j in s if edges[units[j].sha] != "executed"]
        m.update(requires=[ms_by_unit[units[j].sha]["id"] for j in beh if units[j].sha in ms_by_unit], requires_units=[units[j].sha for j in beh],
                 order_hints=[ms_by_unit[units[j].sha]["id"] for j in txt if units[j].sha in ms_by_unit], order_units=[units[j].sha for j in txt],
                 edges=edges, probe="minimized")

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
        tr = await run_tests(sandbox, base_dir, files, grader_tests(rec, base_dir)[0], targets, rec.get("tests_deleted") or ())
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


def volatile_ids(late: Mapping[str, str] | None, ids: Sequence[str]) -> list[str]:
    """Ids a later run of the same tree does not report (an id built from the clock, a random value or a memory
    address); none when the late run was unmeasured or empty."""
    if not late:
        return []
    return sorted(t for t in set(ids) if t not in late)


async def f2p_stage(rec: dict[str, Any], repo: RepoGit, base: Path, sandbox: DockerSandbox, *, reruns: int = 3, min_milestones: int = 2,
                    exclude: Mapping[str, str] | None = None) -> str | None:
    """Fail-to-pass / pass-to-pass with reruns, then milestones by owner unit. Returns why the window stops, or None.
    Benchmark v2: timing tests (``windows_timing.timing_tests``: name, marker, fixture or a body that measures elapsed
    time) and ``exclude`` (test id -> reason; S10's load-dependent tests) are never checks: they leave fail-to-pass and
    pass-to-pass before ownership, recorded in ``excluded['timing']`` and ``timing_rule``."""
    if rec.get("import_tolerant"):
        rec["import_roots"] = sorted(project_roots(base, list(rec.get("oracle_files") or {})))
    tests_b, gone = grader_tests(rec, base)[0], rec["tests_deleted"]
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
    timing = {**timing_tests([*cls["f2p"], *cls["p2p"]], tests_b), **{t: w for t, w in (exclude or {}).items() if t in {*cls["f2p"], *cls["p2p"]}}}
    cls["f2p"] = [t for t in cls["f2p"] if t not in timing]
    cls["p2p"] = [t for t in cls["p2p"] if t not in timing]
    ev.update({k: len(v) for k, v in cls.items()})
    ev["timing"] = len(timing)
    # a World mismatch is tests that FAIL or ERROR at B; skipped and xfailed tests are not (benchmark v2: w3lib's 345
    # xfails read as 354 "failures" and stopped every window)
    hard = [t for t in cls["post_fail"] if all(r.results.get(t) in ("failed", "error") for r in post)]
    ev["post_hard_fail"] = len(hard)
    if not cls["post_pass"] or len(hard) > max(20, len(cls["post_pass"]) // 5):
        ev["post_fail_sample"] = hard[:10]
        return f"{len(hard)} tests fail at B (environment mismatch)"
    # owner = the first unit after whose prefix the test passes under B's tests (report 11 F3); a test that never passes
    # on a prefix is a collateral regression check
    po = await prefix_owners(rec, repo, base, sandbox, cls["f2p"])
    owners = po.pop("owners")
    # P2: a test id that changes from run to run (marshmallow parametrizes on datetime.now(): concurrent reruns share the
    # second, so the id looked stable) never reproduces at grading. One more oracle run, minutes after the first ones:
    # an id it lacks is volatile and is never a check
    late = await run_tests(sandbox, base, rec["oracle_files"], tests_b, targets, gone)
    volatile = volatile_ids(late.results if late.infra == "ok" else None, [*cls["f2p"], *cls["p2p"]])
    owners = {t: o for t, o in owners.items() if t not in volatile}
    rec["ownership"] = po
    rec["milestones"] = milestones_from_owners(rec, owners)
    rec.update(targets=targets, p2p=[t for t in cls["p2p"] if t not in volatile], collateral=sorted(t for t, o in owners.items() if o is None),
               excluded={"flaky": cls["flaky"], "p2f": cls["p2f"], "post_fail": cls["post_fail"], "volatile": volatile, "timing": sorted(timing)},
               timing_rule=dict(sorted(timing.items())))
    if len(rec["milestones"]) < min_milestones:
        return f"{len(rec['milestones'])} milestone(s) with fail-to-pass tests (need {min_milestones})"
    return None


async def validate_window(win: Window, store: WindowStore, images: ImageBook, sandboxes: Callable[[str], DockerSandbox], *, reruns: int = 3,
                          min_milestones: int = 2, mutants: bool = True, log: Callable[[dict], None] | None = None,
                          import_tolerant: bool = True, leak_prefix: str | None = None, exclude: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Every stage for one window; the record (saved whatever stage it reached) says how far it got and why it stopped.
    ``import_tolerant`` (report 11 F8, the default for new windows): the window is classified, probed and graded with
    import-tolerant test files (``grader_tests``); a window mined before P0 has no flag and stays strict."""
    store.guard(win.repo)
    t0 = time.monotonic()
    repo = RepoGit(Path(win.repo_dir))
    rec: dict[str, Any] = {**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows", "reason": "",
                           "lane": LANE, "timings": {}, "import_tolerant": bool(import_tolerant)}
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
                             new_test_names=new_test_names(rec, repo, base), installed=card.get("projects"),
                             **({"name_prefix": leak_prefix} if leak_prefix else {}))
    rec["world"]["leak"] = leak
    rec["timings"]["leak"] = round(time.monotonic() - t1, 1)
    if not leak["ok"]:
        return stop(f"leak probe failed: {leak.get('probes') or leak.get('evidence')}")
    rec["stage"] = "leak_ok"
    sandbox = sandboxes(card["image"])
    why = await f2p_stage(rec, repo, base, sandbox, reruns=reruns, min_milestones=min_milestones, exclude=exclude)
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


_RUN_NONCE: contextvars.ContextVar[str] = contextvars.ContextVar("window_sandbox_nonce", default="")


class NamedSandbox(DockerSandbox):
    """``DockerSandbox`` whose grading containers are named ``<name_prefix><nonce>`` (a caller finds and removes exactly
    its own; the base class names them ``crucible-m17-*``). A run the base class times out is killed under this name."""

    def __init__(self, *a: Any, name_prefix: str = "crucible-m17-", **kw: Any):
        super().__init__(*a, **kw)
        self.name_prefix = name_prefix

    def command(self, tree: Path, name: str, nonce: str, targets: Sequence[str], out: Path | None = None) -> list[str]:
        cmd = super().command(tree, name, nonce, targets, out)
        _RUN_NONCE.set(nonce)
        return [f"{self.name_prefix}{nonce}" if c == name else c for c in cmd]

    async def run(self, tree: Path, targets: Sequence[str]) -> TestRun:
        tr = await super().run(tree, targets)
        if tr.why.startswith("timeout after") and (nonce := _RUN_NONCE.get()):
            await asyncio.to_thread(subprocess.run, ["docker", "kill", f"{self.name_prefix}{nonce}"], capture_output=True, check=False)
        return tr


def sandbox_factory(*, parallel: int = 8, timeout_s: float = 600.0, cpus: str = "2", memory: str = "2g", name_prefix: str = "crucible-m17-"
                    ) -> Callable[[str], DockerSandbox]:
    """One grading sandbox per World image, all sharing one concurrency limit. ``name_prefix`` names the containers."""
    sem = asyncio.Semaphore(parallel)
    made: dict[str, DockerSandbox] = {}

    def get(image: str) -> DockerSandbox:
        if image not in made:
            sb = NamedSandbox(image, guard=True, timeout_s=timeout_s, per_test_s=60, cpus=cpus, memory=memory, parallel=parallel, extra_args=COLLECT_ALL,
                              name_prefix=name_prefix)
            sb._sem = sem
            made[image] = sb
        return made[image]

    return get


async def mine(repos: Mapping[str, Path], store: WindowStore, *, images: ImageBook | None = None, sandboxes: Callable[[str], DockerSandbox] | None = None,
               cap: int = 8, since: str = "2019-01-01", min_units: int = 5, max_units: int = 25, reruns: int = 3, parallel_windows: int = 4,
               deadline: float | None = None, log: Callable[[dict], None] | None = None, window_sem: asyncio.Semaphore | None = None,
               redo: Sequence[str] = ("deadline", "error"), import_tolerant: bool = True, leak_prefix: str | None = None) -> dict[str, Any]:
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
                    await validate_window(w, store, images, sandboxes, reruns=reruns, log=log, import_tolerant=import_tolerant, leak_prefix=leak_prefix)
                except Exception as e:  # noqa: BLE001 - one window's failure is recorded, never the run's
                    store.save_window({**{k: v for k, v in w.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                       "reason": f"error: {type(e).__name__}: {e}"[:500], "lane": LANE})

        await asyncio.gather(*[go(w) for w in wins])
        out[name] = len(wins)
    return out


async def remine(source: WindowStore, store: WindowStore, repo_dirs: Mapping[str, Path], *, images: ImageBook | None = None,
                 sandboxes: Callable[[str], DockerSandbox] | None = None, reruns: int = 3, parallel_windows: int = 4, deadline: float | None = None,
                 log: Callable[[dict], None] | None = None, import_tolerant: bool = True, leak_prefix: str | None = None,
                 redo: Sequence[str] = ("deadline", "error")) -> dict[str, int]:
    """Validate again, into ``store``, exactly the windows ``source`` recorded (P2: the P0 re-mine, import-tolerant, with
    the test-id fix): the same A, B and units, so the two funnels compare window for window (``mine`` would draw a fresh
    sample: a store built by several mining runs holds windows one run's cap never reaches). A window already in
    ``store`` is skipped unless its stop reason starts with one of ``redo``."""
    if source.split != store.split:
        raise HeldOutError(f"{source.root} is {source.split}, {store.root} is {store.split}")
    images = images or ImageBook()
    sandboxes = sandboxes or sandbox_factory()
    sem = asyncio.Semaphore(parallel_windows)
    done: dict[str, int] = {}

    async def go(wid: str) -> None:
        rec = source.window(wid)
        if rec["repo"] not in repo_dirs or ((store.dir(wid) / "window.json").exists()
                                            and not str(store.window(wid).get("reason", "")).startswith(tuple(redo))):
            return
        win = Window.from_json({**rec, "repo_dir": str(repo_dirs[rec["repo"]])})
        async with sem:
            if deadline is not None and time.time() > deadline:
                store.save_window({**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                   "reason": "deadline: not validated", "lane": LANE})
                return
            try:
                await validate_window(win, store, images, sandboxes, reruns=reruns, log=log, import_tolerant=import_tolerant, leak_prefix=leak_prefix)
            except Exception as e:  # noqa: BLE001 - one window's failure is recorded, never the run's
                store.save_window({**{k: v for k, v in win.to_json().items() if k != "repo_dir"}, "split": store.split, "stage": "windows",
                                   "reason": f"error: {type(e).__name__}: {e}"[:500], "lane": LANE})
        done[rec["repo"]] = done.get(rec["repo"], 0) + 1

    await asyncio.gather(*[go(w) for w in source.window_ids()])
    return done


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
            "Changes to test files and to packaging or test configuration are discarded before grading, and a run that makes them, "
            "or that tries to install packages, does not earn the full-completion credit; adding test plumbing (conftest.py, "
            "sitecustomize.py, usercustomize.py or .pth files), trying to download code (this library's releases or repository, or "
            "anything over the network), or reading version-control history beyond the current commit scores the run 0. "
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
                "opening is two or three sentences of context, closing is optional, requirements has exactly one entry per change id. "
                "Names listed in a change's must_name (or in must_name_any) are public names the release must expose under exactly "
                "that name: write each one, verbatim, in the requirement of its change (must_name_any: in the one it fits).")
REALISM_BRIEF = ("You review a maintainer's release request for a coding task. Reject it (ok false) when it gives hints a real "
                 "maintainer would not: it names files or tests, says how to implement something, quotes code from a patch, or "
                 "states rules the developer could not know from the request and the repository. Requirements that state "
                 "observable public behaviour are fine. Return {ok: bool, findings: [one sentence each]}.")
LINT_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("process", re.compile(r"\b(do not|don't|never|avoid|without)\s+(using|use|importing|import|calling|call|reading|read|looking|look|consulting|consult|"
                           r"searching|search|installing|install|copying|copy|running|run|checking|check)\b", re.IGNORECASE)),
    # editing verbs are process only when their object is the work environment ("do not modify the tests"); "returns a
    # copy without modifying the input" or "without modifying the decorated function's signature" is behaviour (report 11
    # D1: the old rule refused cachetools' request three times)
    ("process", re.compile(r"\b(do not|don't|never|avoid|without)\s+(editing|edit|touching|touch|modifying|modify|changing|change)\s+(the\s+|any\s+)?"
                           r"(existing\s+)?(tests?|test\s+(suite|files?)|conftest|ci\b|continuous integration|build|packaging|git|history|repository|setup)\b",
                           re.IGNORECASE)),
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


def _module_file(base_dir: Path, module: str) -> Path | None:
    """``a.b`` -> ``a/b.py`` or ``a/b/__init__.py`` under A's tree (or its ``src/``); None when it is not the project's."""
    rel = Path(*module.split("."))
    for root in (Path(base_dir), Path(base_dir) / "src"):
        for cand in (root / rel.with_suffix(".py"), root / rel / "__init__.py"):
            if cand.is_file():
                return cand
    return None


def _bound_names(mod_file: Path) -> set[str] | None:
    """Names a module binds at top level (defs, classes, assignments, imports, its package's submodules); None when it
    re-exports with ``*``, makes attributes dynamically (a module ``__getattr__``, an import hook) or does not parse
    (unknown: never required)."""
    text = mod_file.read_text(errors="replace")
    pkg = [f.read_text(errors="replace") for f in mod_file.parent.glob("*.py")] if mod_file.name == "__init__.py" else []
    if any(re.search(r"sys\.meta_path|sys\.modules\[|^def __getattr__\(", t, re.MULTILINE) for t in [text, *pkg]):
        return None  # the package (or a module of it) makes attributes at import time
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return None
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(n.name)
        elif isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.For, ast.With)):
            for t in ast.walk(n):
                if isinstance(t, ast.Name) and isinstance(t.ctx, ast.Store):
                    out.add(t.id)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                if a.name == "*":
                    return None
                out.add(a.asname or a.name.split(".")[0])
    if mod_file.name == "__init__.py":
        out |= {p.stem for p in mod_file.parent.glob("*.py")} | {p.name for p in mod_file.parent.iterdir() if (p / "__init__.py").is_file()}
    return out


def _dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")  # every module has its dunders


def required_names(tests: Mapping[str, str], base_dir: Path) -> dict[str, list[str]]:
    """Public names B's test files import from the project (``from <project module> import name``) that A does not
    have: the hidden tests need exactly these names, so the prompt must state them. A module-level import of a missing
    name fails collection of the whole test file (every milestone in it and its regression tests read as failing), an
    in-function one (or ``import pkg as p`` then ``p.name``) fails its tests. ``{"module_level", "in_function", "attribute",
    "private"}`` lists; private names
    (``_x``) are never required in a prompt: a hidden test that imports one is testing an implementation detail."""
    found: dict[str, set[str]] = {"module_level": set(), "in_function": set(), "attribute": set(), "private": set()}
    bound: dict[str, set[str] | None] = {}
    for _path, text in sorted(tests.items()):
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        top = {id(n) for n in tree.body}
        aliases = {a.asname or a.name: a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names
                   if _module_file(base_dir, a.name) is not None}
        for n in ast.walk(tree):
            if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id in aliases:  # ``mi.new_name``
                mod = aliases[n.value.id]
                if mod not in bound:
                    bound[mod] = _bound_names(_module_file(base_dir, mod))  # type: ignore[arg-type]
                if (bound[mod] is not None and n.attr not in bound[mod] and not _dunder(n.attr)  # type: ignore[operator]
                        and _module_file(base_dir, f"{mod}.{n.attr}") is None):
                    found["private" if n.attr.startswith("_") else "attribute"].add(n.attr)
                continue
            if not (isinstance(n, ast.ImportFrom) and n.module and n.level == 0):
                continue
            f = _module_file(base_dir, n.module)
            if f is None:
                continue
            if n.module not in bound:
                bound[n.module] = _bound_names(f)
            have = bound[n.module]
            if have is None:
                continue
            for a in n.names:
                if a.name == "*" or a.name in have or _dunder(a.name) or _module_file(base_dir, f"{n.module}.{a.name}") is not None:
                    continue
                kind = "private" if a.name.startswith("_") else ("module_level" if id(n) in top else "in_function")
                found[kind].add(a.name)
    found["in_function"] -= found["module_level"]
    found["attribute"] -= found["module_level"] | found["in_function"]
    return {k: sorted(v) for k, v in found.items()}


def author_payload(rec: Mapping[str, Any], repo: RepoGit, base_dir: Path) -> dict[str, Any]:
    units = {u["sha"]: Unit(**u) for u in rec["units"]}
    readme = next(iter(sorted(Path(base_dir).glob("README*"))), None)
    req = required_names(rec.get("tests_b") or {}, base_dir)
    must = set(req["module_level"]) | set(req["in_function"]) | set(req["attribute"])
    changes, placed = [], set()
    for m in rec["milestones"]:
        u = units[m["unit"]]
        names = public_names(repo.patch(u).decode("utf-8", "replace"))
        mine = sorted(must & set(names) - placed)
        placed |= set(mine)
        changes.append({"id": m["id"], "message": scrub(f"{u.subject}\n{u.body}")[:900], "public_names": names,
                        **({"must_name": mine} if mine else {})})
    for n in sorted(must - placed):  # an export or a constant: the change whose diff adds a line naming it
        for c, m in zip(changes, rec["milestones"], strict=True):
            if re.search(rf"^\+(?!\+\+).*(?<!\w){re.escape(n)}(?!\w)", repo.patch(units[m["unit"]]).decode("utf-8", "replace"), re.MULTILINE):
                c["must_name"] = sorted({*c.get("must_name", []), n})
                placed.add(n)
                break
    rest = sorted(must - placed)
    return {"library": rec["repo"], "readme_head": scrub(readme.read_text(errors="replace"))[:1500] if readme else "",
            "changelog": changelog_excerpt(repo, rec["a"], rec["b"]), "changes": changes, **({"must_name_any": rest} if rest else {})}


def unstated_names(payload: Mapping[str, Any], text: str) -> list[str]:
    """Findings for required public names (``author_payload``'s ``must_name`` / ``must_name_any``) the authored text never
    writes verbatim."""
    want = sorted({n for c in payload.get("changes") or [] for n in c.get("must_name") or []} | set(payload.get("must_name_any") or []))
    return [f"unstated_name: the hidden tests import {n!r}; the request must name it" for n in want if not re.search(rf"(?<!\w){re.escape(n)}(?!\w)", text)]


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
    # a public name the hidden tests import is a fact the tests enforce: the author is asked to write it in its change's
    # requirement; any it leaves out are stated once, verbatim, by the lane (report 11 run 2: the author ignored names whose
    # unit owns no milestone in 6 of 6 tries), never left unstated
    missing = [f.split("'")[1] for f in unstated_names(payload, authored)]
    names_line = f"The release must also expose these public names: {', '.join(missing)}." if missing else ""
    if names_line:
        closing = (closing + "\n\n" if closing else "") + names_line
    regress = bool(set(rec.get("p2p") or []) | set(rec.get("collateral") or []))  # no regression tests: no regression requirement
    if regress:
        requirements.append({"id": REGRESSION_REQ, "milestone": None, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
    prompt = compose_window_prompt(opening, [r for r in requirements if r["id"] != REGRESSION_REQ],
                                   closing + (("\n\n" if closing else "") + REGRESSION_TEXT if regress else ""))
    spans = {r["id"]: list(loc) for r in requirements if (loc := locate(prompt, r["text"]))}
    _, ledger_problems = window_ledger(rec, requirements, spans, prompt)
    problems += ledger_problems
    return {"prompt": prompt, "requirements": requirements, "spans": spans, "checkset": window_checkset(rec, requirements), "payload": payload,
            "findings": problems, "opening": opening, "closing": closing, "names_stated_by_lane": missing}


async def review_realism(client: Any, prompt: str, *, max_tokens: int = 8000) -> dict[str, Any]:
    """The maintainer's text only: the fixed epilogue is the lane's truthful description of the episode (study 08 §3: the
    model may know it is in an environment), not the author's, and a reviewer judging it as a hint was a false finding
    on every smoke-test window."""
    msgs = [{"role": "system", "content": f"ROLE: realism\n{REALISM_BRIEF}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps({"request": maintainer_text(prompt)})}]
    out = await call_supported(client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0)
    return {"ok": bool(isinstance(out, dict) and out.get("ok") is True), "findings": [str(x) for x in (out or {}).get("findings") or []]}


# ---- findings sent back to the author are test-derived (report 11 D8) ---------------------------------------------------
_NOT_NAMES = frozenset({"assert", "len", "list", "dict", "set", "tuple", "str", "int", "float", "bool", "repr", "isinstance", "type", "sorted",
                        "print", "range", "getattr", "hasattr", "next", "iter", "callable", "issubclass", "min", "max", "sum", "any", "all"})


def _test_node(text: str, names: Sequence[str]) -> Any:
    import warnings

    try:
        with warnings.catch_warnings():  # old test files carry invalid escape sequences
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return None
    body: list[ast.stmt] = tree.body
    node: Any = None
    for n in names:
        node = next((x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == n), None)
        if node is None:
            return None
        body = getattr(node, "body", [])
    return node


def hidden_test_facts(tests: Mapping[str, str], test_id: str, *, max_items: int = 8, max_chars: int = 160) -> dict[str, Any] | None:
    """What one hidden test checks, read from its own source (never a model's account): the public names it calls, its
    assertions, the exceptions it expects and its parameter cases. None when the test's source is not found."""
    parts = test_id.split("::")
    names = [re.sub(r"\[.*$", "", n) for n in parts[1:]]
    text = tests.get(parts[0])
    node = _test_node(text, names) if text is not None and names else None
    if node is None:
        return None

    def short(n: ast.AST) -> str:
        u = ast.unparse(n)
        return u if len(u) <= max_chars else u[: max_chars - 3] + "..."

    calls: list[str] = []
    asserts: list[str] = []
    raises: list[str] = []
    params: list[str] = []
    for d in getattr(node, "decorator_list", []):
        if isinstance(d, ast.Call) and ast.unparse(d.func).endswith("parametrize") and len(d.args) >= 2:
            vals = d.args[1].elts if isinstance(d.args[1], (ast.List, ast.Tuple)) else [d.args[1]]
            params += [f"{short(d.args[0])} = {short(v)}" for v in vals]
    for n in (x for stmt in getattr(node, "body", []) for x in ast.walk(stmt)):  # the body: decorators are the parameters above
        if isinstance(n, ast.Assert):
            asserts.append(short(n.test))
        elif isinstance(n, ast.With):
            for it in n.items:
                c = it.context_expr
                if isinstance(c, ast.Call) and ast.unparse(c.func).endswith("raises") and c.args:
                    body = "; ".join(short(b) for b in n.body[:2])
                    raises.append(f"{short(c.args[0])} from {body}")
        elif isinstance(n, ast.Call):
            f = n.func
            name = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else "")
            if name and not name.startswith(("_", "test", "assert")) and name not in _NOT_NAMES and name not in calls:
                calls.append(name)
    return {"calls": calls[:max_items], "asserts": list(dict.fromkeys(asserts))[:max_items], "raises": list(dict.fromkeys(raises))[:max_items],
            "params": params[:max_items]}


def findings_from_tests(rec: Mapping[str, Any], items: Sequence[Any]) -> tuple[list[str], list[str]]:
    """Findings for the prompt author from failing hidden tests only (report 11 D8: the auditor's guess "capitalised True"
    contradicted the test, which expects None or ValueError). Each item must be a hidden test id of the window (a string
    or ``{"test": id}``); the finding states that test's own facts (``hidden_test_facts``) under its milestone's change id, with no
    test name or file path. Anything else (free text, an unknown test) is rejected. Returns (findings, rejected)."""
    owner = {t: m["id"] for m in rec.get("milestones") or [] for t in m["tests"]}
    regress = set(rec.get("p2p") or []) | set(rec.get("collateral") or [])
    out: list[str] = []
    rejected: list[str] = []
    for it in items:
        tid = it.get("test") if isinstance(it, Mapping) else it
        if not isinstance(tid, str) or (tid not in owner and tid not in regress):
            rejected.append(f"not a hidden test id of this window (free text is never sent to the author): {str(it)[:200]}")
            continue
        facts = hidden_test_facts(rec.get("tests_b") or {}, tid)
        if facts is None:
            rejected.append(f"no source for {tid}")
            continue
        where = f"change {owner[tid]}" if tid in owner else "existing behaviour"
        bits = [f"it calls {', '.join(facts['calls'])}" if facts["calls"] else "",
                f"it expects {'; '.join(facts['asserts'])}" if facts["asserts"] else "",
                f"it expects these exceptions: {'; '.join(facts['raises'])}" if facts["raises"] else "",
                f"over these cases: {'; '.join(facts['params'])}" if facts["params"] else ""]
        text = re.sub(r"\btest_\w+", "a check", "; ".join(b for b in bits if b))
        out.append(f"spec_fact ({where}): a hidden check the request must make achievable: {text}"[:900])
    return list(dict.fromkeys(out)), rejected


# ---- the pass-stub floor (study 08 §5 step 6; moved from scripts/lh_run.py so every WindowTask offers it) -------------------
def _top_defs(tree: ast.Module) -> dict[str, ast.AST]:
    return {n.name: n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def _stub(node: ast.AST, indent: str) -> str:
    keep = [d.id for d in getattr(node, "decorator_list", []) if isinstance(d, ast.Name) and d.id in ("staticmethod", "classmethod", "property")]
    deco = "".join(f"{indent}@{d}\n" for d in keep)
    if isinstance(node, ast.ClassDef):
        return f"{indent}class {node.name}:\n{indent}    pass\n"
    head = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    return f"{deco}{indent}{head} {node.name}(*args, **kwargs):\n{indent}    pass\n"


def stub_files(base_dir: Path, oracle_files: Mapping[str, str | None]) -> dict[str, str]:
    """The ``pass``-stub floor: A's source with every top-level function or class B adds, and every method B adds to a
    class A already has, present as a stub that does nothing (``def f(*args, **kwargs): pass``, ``class C: pass``). A module
    B adds becomes a module of stubs. Only ``.py`` files outside tests; files that do not parse are left out."""
    out: dict[str, str] = {}
    for path, b_text in sorted(oracle_files.items()):
        if b_text is None or not path.endswith(".py") or re.search(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|conftest\.py$", path):
            continue
        try:
            b_tree = ast.parse(b_text)
        except SyntaxError:
            continue
        a_file = Path(base_dir) / path
        a_text = a_file.read_text(errors="replace") if a_file.is_file() else None
        if a_text is None:
            stubs = [_stub(n, "") for n in _top_defs(b_tree).values()]
            if stubs:
                out[path] = "\n\n".join(stubs)
            continue
        try:
            a_tree = ast.parse(a_text)
        except SyntaxError:
            continue
        a_defs, b_defs = _top_defs(a_tree), _top_defs(b_tree)
        lines = a_text.splitlines(keepends=True)
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        inserts: list[tuple[int, str]] = []  # (after line number, text)
        for name, bnode in b_defs.items():
            anode = a_defs.get(name)
            if not (isinstance(bnode, ast.ClassDef) and isinstance(anode, ast.ClassDef)):
                continue
            have = set(_top_defs(ast.Module(body=anode.body, type_ignores=[])))
            new = [n for n in _top_defs(ast.Module(body=bnode.body, type_ignores=[])).values()
                   if n.name not in have and not isinstance(n, ast.ClassDef)]
            if new:
                indent = " " * (anode.body[0].col_offset if anode.body else anode.col_offset + 4)
                inserts.append((anode.end_lineno or len(lines), "".join(_stub(n, indent) for n in new)))
        for line_no, text in sorted(inserts, reverse=True):
            lines.insert(line_no, text)
        tail = [_stub(n, "") for name, n in b_defs.items() if name not in a_defs]
        new_text = "".join(lines) + ("\n\n" + "\n\n".join(tail) if tail else "")
        if new_text != a_text:
            out[path] = new_text
    return out


def stub_test_files(tests: Mapping[str, str]) -> dict[str, str]:
    """The test-edit floor: every hidden test file replaced by tests that pass without checking anything (the grader
    restores them, so this must score 0 and fail ``path:protected``)."""
    out = {}
    for p, text in tests.items():
        try:
            names = [n.name for n in ast.parse(text).body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name.startswith("test")]
        except SyntaxError:
            names = []
        out[p] = "\n\n".join(f"def {n}():\n    pass\n" for n in names or ["test_ok"])
    return out


# ---- the task ----------------------------------------------------------------------------------------------------------
SIMPLIFY_KINDS = ("prerequisites", "locate")


def changed_defs(before: str | None, after: str | None) -> list[str]:
    """Qualified names of the functions and classes ``after`` defines differently from ``before`` (or newly): what a
    unit changed, for ``simplify('locate')``. A file that does not parse names nothing."""

    def defs(text: str | None) -> dict[str, str]:
        if not text:
            return {}
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError):
            return {}
        out: dict[str, str] = {}

        def walk(node: ast.AST, prefix: str) -> None:
            for ch in ast.iter_child_nodes(node):
                if isinstance(ch, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
                    q = f"{prefix}{ch.name}"
                    body = ch.body if isinstance(ch, ast.ClassDef) else None
                    out[q] = ast.dump(ch if body is None else ast.ClassDef(name=ch.name, bases=ch.bases, keywords=ch.keywords, body=[],
                                                                           decorator_list=ch.decorator_list, type_params=[]))
                    walk(ch, q + ".")

        walk(tree, "")
        return out

    a, b = defs(before), defs(after)
    return [q for q in b if a.get(q) != b[q] and not (q.rsplit(".", 1)[-1].startswith("test"))]


@dataclass
class MilestoneMutant(Mutant):
    """A commit-revert mutant with the milestone it must fail (``longtask.LongTask.mutants``)."""

    milestone: str = ""


VERIFIER_RECIPE = "window-grader/p0"  # bump when the grading rule changes (path gates, import tolerance)


class WindowTask(ExecTask):
    """One window under one prompt: ``ExecTask`` with the window's grader, milestones, Phi, mutants and World. It is a
    ``factory.longtask.LongTask`` (``check_longtask`` finds nothing missing): ``world``, ``prompt``, ``path_rules``,
    ``oracle_patch``, ``floors``, ``alternatives``, ``simplify('prerequisites')``, ``card``, ``ref``."""

    horizon = "H2"

    def __init__(self, lane: WindowLane, mined: MinedTask, window: Mapping[str, Any], *, task_id: str, prompt: str, requirements: Sequence[Mapping[str, Any]],
                 spans: Mapping[str, Sequence[int]], checkset: Mapping[str, Any], weights: Mapping[str, float] | None = None,
                 phi_weight: float | None = None, import_tolerant: bool | None = None, path_rules: PathRules | None = None):
        self.window = dict(window)
        # Type A outcome reward (report 11 F1): phi_weight * Phi(end state) + (1 - phi_weight) * [every gating check passes].
        # Phi is itself an end-state check (hidden milestone tests, prerequisite-gated), so the reward still grades only the
        # outcome; an all-fail group keeps a spread when its end states differ. 0 = the binary pass reward of run 1.
        self.phi_weight = float(lane.phi_weight if phi_weight is None else phi_weight)
        if not 0.0 <= self.phi_weight <= 1.0:
            raise ValueError("phi_weight is in [0, 1]")
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
        self.spans = {k: list(v) for k, v in spans.items()}
        # the grader's test files (report 11 F8): the window's own mode unless overridden (a strict-mined window graded
        # tolerantly would let the start state pass tests that were fail-to-pass only through the import)
        self.import_tolerant = bool(self.window.get("import_tolerant")) if import_tolerant is None else bool(import_tolerant)
        self.grader_tests, self.guarded_imports = grader_tests({**self.window, "import_tolerant": self.import_tolerant}, self.base_dir)
        hidden = set(self.window.get("tests_b") or {}) | set(self.window.get("tests_deleted") or [])
        own = {self.window["repo"], str((self.window.get("world") or {}).get("project") or ""), *(self.window.get("import_roots") or []),
               *project_roots(self.base_dir, list(self.window.get("oracle_files") or {}))} - {"", "tests", "test", "testing", "docs", "doc"}
        self.path_rules = path_rules or PathRules(grader_paths=tuple(sorted(hidden)),
                                                  required=tuple(RequiredAction(**r) for r in self.window.get("required_actions") or []),
                                                  step_budget=self.window.get("step_budget"), project_names=tuple(sorted(own)))
        self.verifier_version = sha256_text(VERIFIER_RECIPE, str(self.import_tolerant), json.dumps(self.path_rules.to_json(), sort_keys=True),
                                            json.dumps(self.check_tests, sort_keys=True))[:16]
        self._oracle_patch: str | None = None
        # the policy's World only (the grader always runs ``window.world.image``): a seeded tooling fault (P4,
        # ``targeting.seeded_tooling``) swaps the image the agent works in, or mounts its workspace read-only
        self.world_image: str | None = None
        self.workspace_mode: str = "rw"

    # -- the oracle as steps (horizon replay, shortcut probe) -------------------------------------------------------
    def oracle_steps(self) -> list[dict[str, Any]]:
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]])) if self.window["repo"] in self.lane.repo_dirs else None
        if repo is None or self.window.get("simplified_from"):  # no clone at hand, or a simplified start: the window as one step
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
            tr = await run_tests(self.sandbox, self.base_dir, files, self.grader_tests, targets, self.window.get("tests_deleted") or ())
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
        full = bool(gating and all(gates[c] == "pass" for c in gating))
        status = {m["id"]: passed_group(statuses, m["tests"]) for m in self.milestones}
        phi = self.phi(status)
        reward = 1.0 if full else 0.0  # the accept contract's verdict (floors, mutants killed at 0, flake) stays binary
        shaped = round(self.phi_weight * phi + (1.0 - self.phi_weight) * reward, 6) if not errors else 0.0
        meta = {"failing_tests": sorted(set(failing))[:60], "statuses": {s: sum(1 for v in statuses.values() if v == s) for s in set(statuses.values())},
                "tail": tail[-2500:], "apply_errors": errors, "protected_writes": protected, "tests_intact": True, "why": why,
                "kinds": {c: "outcome" for c in check_tests}, "milestones": status, "phi": phi, "full_pass": full,
                "outcome_reward": shaped, "reward_rule": f"{self.phi_weight:g}*phi + {1 - self.phi_weight:g}*full_pass",
                "import_tolerant": self.import_tolerant, "guarded_imports": sum(len(v) for v in self.guarded_imports.values())}
        self.lane.note_grade(self.task_id, solution, meta)
        return GradeReport(task=self.task_id, tier="full", checks=results, gates=gates, reward=reward, outcome=reward, infra={"status": "ok"},
                           trusted=True, elapsed_s=time.monotonic() - t0, meta=meta), fp

    async def grade_patch(self, patch: Any, workdir: Path | None = None, *, trail: Sequence[Any] | None = None,
                          network_events: Sequence[str] | None = None) -> Any:
        """``WorkspaceHarness``'s verifier call: the collected patch graded from a pristine A, then the path (V3: state first,
        path second). ``meta['milestones']`` is the pass vector, ``meta['phi']`` the potential, ``meta['path']`` the path
        report (``factory.pathcheck``) on the diff before restore and the trail.

        The episode's reward (report 11 F1 plus P0's path gates): ``phi_weight * Phi + (1 - phi_weight) * [every end-state
        check and every measured path gate passes]``; a hack-severity path violation (network, history or grader reads,
        test plumbing, downloading code) makes it 0 and sets ``meta['hack_suspect']``. A gate-severity violation (a test or
        config edit, restored before the hidden tests ran; installing other packages) removes only the full-pass bonus. The trail is ``trail`` when given, else the
        attempt's ``trace.json`` next to ``workdir`` (``pathcheck.find_trail``); without one the trail checks are
        unmeasured (listed, never gating). ``grade`` keeps the binary end-state verdict the accept contract reads."""
        rep, _ = await self.grade(patch)
        if rep.infra.get("status") == "ok" and "outcome_reward" in (rep.meta or {}):
            self._apply_path(rep, patch, workdir, trail, network_events)
        if workdir is not None:
            Path(workdir).mkdir(parents=True, exist_ok=True)
            m = rep.meta or {}
            (Path(workdir) / "window_grade.json").write_text(json.dumps({"reward": rep.reward, "milestones": m.get("milestones"), "phi": m.get("phi"),
                                                                         "infra": rep.infra, "path": m.get("path"), "hack_suspect": m.get("hack_suspect")},
                                                                        indent=1))
        return rep

    def _apply_path(self, rep: Any, patch: Any, workdir: Path | None, trail: Sequence[Any] | None, network_events: Sequence[str] | None) -> None:
        from crucible.verify.runtime.plan import CheckResult

        if trail is None:
            calls, source = find_trail(workdir)
        else:
            calls = [t if isinstance(t, ToolCall) else ToolCall(str(t.get("function") or t.get("function_name") or ""), t.get("arguments"))
                     for t in trail]
            source = "given"
        entries = None
        if not isinstance(patch, str):  # the lane's other solution forms (oracle steps, {files}): what they would write
            files, protected, _ = self._files(patch)
            entries = [("D" if t is None else ("M" if (self.base_dir / p).exists() else "A"), p) for p, t in files.items()]
            entries += [("M", p) for p in protected]
        pr = check_paths(self.path_rules, patch=patch if isinstance(patch, str) else "", entries=entries, trail=calls, trail_source=source,
                         network_events=network_events)
        m = rep.meta
        end_full = bool(m["full_pass"])
        full = end_full and pr.ok
        hack = pr.hack_suspect
        reward = 0.0 if (hack or m.get("apply_errors")) else round(self.phi_weight * m["phi"] + (1.0 - self.phi_weight) * float(full), 6)
        for gid, st in pr.gates().items():
            rep.gates[gid] = st
            m["kinds"][gid] = "gate"
            chk = next(c for c in pr.checks if c.id == gid)
            rep.checks.append(CheckResult(id=gid, gate=gid, status=st, detail={"severity": chk.severity, "evidence": chk.evidence[:10]}))
        m.update(path=pr.to_json(), end_state_full_pass=end_full, full_pass=full, hack_suspect=hack, outcome_reward=reward,
                 reward_rule=f"0 if hack_suspect else {self.phi_weight:g}*phi + {1 - self.phi_weight:g}*[full_pass and no path violation]")
        rep.reward = rep.outcome = reward

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
        (wrong on k and more), each with the milestone it must fail."""
        return [MilestoneMutant(id=m["id"], family="commit_revert", solution=m["solution"], description=f"B with the unit of {m['milestone']} reverted",
                                milestone=m["milestone"])
                for m in self.window.get("mutants") or [] if m.get("status") in ("valid", "broad")]

    # -- the LongTask contract (factory.longtask) ---------------------------------------------------------------------------
    @property
    def world(self) -> dict[str, Any]:
        w = self.window.get("world") or {}
        return {"kind": "repo_window", "repos": [self.window["repo"]], "image": w.get("image"), "digest": w.get("digest"), "base_dir": str(self.base_dir),
                "start": self.window.get("a"), "network": "none", "history": "cut", "leak_ok": (w.get("leak") or {}).get("ok"),
                "simplified_from": self.window.get("simplified_from")}

    @property
    def prompt(self) -> str:
        return self.instruction

    def patch_of(self, files: Mapping[str, str | None]) -> str:
        """``files`` written over A (None = deleted) as the git patch the workspace harness would collect (``--binary``)."""
        env = {"GIT_CONFIG_NOSYSTEM": "1", "GIT_AUTHOR_NAME": "c", "GIT_AUTHOR_EMAIL": "c@c", "GIT_COMMITTER_NAME": "c", "GIT_COMMITTER_EMAIL": "c@c",
               "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}
        with tempfile.TemporaryDirectory(prefix="lhd-") as td:
            t = Path(td) / "t"
            shutil.copytree(self.base_dir, t, symlinks=True)
            env.update(HOME=td, GIT_CEILING_DIRECTORIES=td)

            def git(*a: str) -> str:
                return subprocess.run(["git", "-c", "core.autocrlf=false", "-c", "commit.gpgsign=false", *a], cwd=t, env=env, capture_output=True,
                                      check=True).stdout.decode("utf-8", "replace")

            git("init", "-q")
            git("add", "-A", "-f")
            git("commit", "-q", "--allow-empty", "-m", "base")
            for p, text in files.items():
                if _safe(p) is None:
                    continue
                f = t / p
                if text is None:
                    f.unlink(missing_ok=True)
                else:
                    f.parent.mkdir(parents=True, exist_ok=True)
                    f.write_text(text)
            git("add", "-A", "-f")
            return git("-c", "core.quotepath=off", "diff", "--cached", "--binary", "HEAD")

    def oracle_patch(self) -> str:
        """B's agent-writable change as one patch against A (what a solver that does the whole window submits)."""
        if self._oracle_patch is None:
            self._oracle_patch = self.patch_of(self.window["oracle_files"])
        return self._oracle_patch

    def floors(self) -> list[GroundCase]:
        """Solutions that do no work, as patches: each must score reward 0 with Phi 0 on ``grade_patch``."""
        out = [GroundCase("no_op", "floor", "", note="the start state (the empty patch)"),
               GroundCase("claims_only", "floor", status_claim_patch(self), note="STATUS.json claims every requirement, nothing else"),
               GroundCase("test_edit", "floor", self.patch_of(stub_test_files(self.window.get("tests_b") or {})),
                          note="every hidden test file replaced by tests that check nothing: restored, path:protected fails")]
        stub = stub_files(self.base_dir, self.window["oracle_files"])
        if stub:
            out.append(GroundCase("pass_stub", "floor", self.patch_of(stub), note="every name B adds present as a stub that does nothing"))
        return out

    def alternatives(self) -> list[GroundCase]:
        """Other solutions that must score 1: the oracle as the window's own unit steps (solver passes confirmed by the
        second verifier join in P2)."""
        return [GroundCase("oracle_steps", "alternative", self.oracle, note="B as its units, in order")]

    def simplify(self, kind: str, *, milestones: Sequence[str] | None = None, root: Path | None = None) -> WindowTask:
        """The counterfactual control's task (swarm-pipeline/00 §7): ``kind="prerequisites"`` pre-applies the executed
        prerequisite units of ``milestones`` (default: every milestone with an executed ``requires`` edge) to A. The new start is A plus
        those units (source only; when they do not apply alone, the whole prefix up to the last of them); the returned task
        keeps only the milestones not pre-applied, their requirements and their mutants, with the same tests, World image and
        grading mode. Needs the repository clone; the new start is written under ``root`` (default: the store's
        ``simplified/``). It is a changed task: S8 (``longtask.null_agent_score``, ``ground_truth``) runs on it again."""
        if kind == "locate":
            return self._simplify_locate(milestones=milestones)
        if kind != "prerequisites":
            raise ValueError(f"a window task simplifies by {SIMPLIFY_KINDS}, not {kind!r}")
        by_id = {m["id"]: m for m in self.milestones}
        by_unit = {m["unit"]: m for m in self.milestones}

        def executed(m: Mapping[str, Any]) -> list[str]:  # prerequisite units behind executed edges only (a record mined before
            edges = m.get("edges")  # report 11 F2 also lists patch-conflict units in requires_units)
            return [u for u in m.get("requires_units") or [] if not isinstance(edges, Mapping) or edges.get(u) == "executed"]

        def exec_closure(mid: str) -> set[str]:
            out: set[str] = set()
            todo = executed(by_id[mid])
            while todo:
                u = todo.pop()
                if u not in out:
                    out.add(u)
                    todo += executed(by_unit[u]) if u in by_unit else []
            return out

        targets = list(milestones) if milestones else [m["id"] for m in self.milestones if executed(m)]
        if not targets or any(t not in by_id for t in targets):
            raise ValueError(f"simplify needs milestones with executed prerequisites; got {targets or 'none'}")
        pre_units = set().union(*(exec_closure(t) for t in targets))
        if not pre_units:
            raise ValueError(f"{targets} have no prerequisite to pre-apply")
        if self.window["repo"] not in self.lane.repo_dirs:
            raise RuntimeError(f"simplify needs the clone of {self.window['repo']} (WindowLane(repo_dirs=...))")
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]]))
        units = [Unit(**u) for u in self.window["units"]]
        key = sha256_text(self.mined.id, *sorted(pre_units))[:10]
        dest = (Path(root) if root is not None else Path(self.lane.store.root) / "simplified").resolve()  # git runs inside it
        new_id = f"{self.mined.id}~pre-{key}"
        base = dest / new_id / "base"
        mode = "exact"
        ok, _ = repo.build(self.window["a"], [u for u in units if u.sha in pre_units], base, source_only=True)
        if not ok:  # the prerequisites do not apply on their own: the prefix up to the last of them (its milestones are pre-applied too)
            last = max(i for i, u in enumerate(units) if u.sha in pre_units)
            pre_units = {u.sha for u in units[: last + 1]}
            ok, bad = repo.build(self.window["a"], units[: last + 1], base, source_only=True)
            mode = "prefix"
            if not ok:
                raise RuntimeError(f"the prerequisite prefix does not apply at {bad}")
        pre_ms = sorted(m["id"] for m in self.milestones if m["unit"] in pre_units)
        keep = []
        for m in self.milestones:
            if m["id"] in pre_ms:
                continue
            m2 = json.loads(json.dumps(m))
            m2["requires"] = [r for r in m2.get("requires") or [] if r not in pre_ms]
            m2["requires_units"] = [u for u in m2.get("requires_units") or [] if u not in pre_units]
            keep.append(m2)
        if not keep:
            raise ValueError("every milestone would be pre-applied: nothing is left to solve")
        keep_ids = {m["id"] for m in keep}

        def now(p: str) -> str | None:
            f = base / p
            return f.read_text(errors="replace") if f.is_file() else None

        rec = json.loads(json.dumps(self.window, default=str))
        rec.update(id=new_id, milestones=keep, oracle_files={p: t for p, t in self.window["oracle_files"].items() if now(p) != t},
                   mutants=[x for x in self.window.get("mutants") or [] if x.get("milestone") in keep_ids],
                   dependency_edges=sum(len(m["requires"]) for m in keep),
                   simplified_from={"task": self.task_id, "window": self.mined.id, "kind": kind, "targets": targets, "pre_applied": pre_ms,
                                    "units": sorted(pre_units), "mode": mode})
        reqs = [dict(r) for r in self.requirements if r.get("milestone") is None or r.get("milestone") in keep_ids]
        dropped = [r["id"] for r in self.requirements if r.get("milestone") and r.get("milestone") not in keep_ids]
        prompt = "".join(ln for ln in self.instruction.splitlines(keepends=True) if not any(ln.startswith(f"- [{d}] ") for d in dropped))
        spans = {r["id"]: list(loc) for r in reqs if (loc := locate(prompt, r["text"]))}
        mined = mined_of(rec, base)
        t = WindowTask(self.lane, mined, rec, task_id=f"{self.task_id}~pre-{key[:6]}", prompt=prompt, requirements=reqs, spans=spans,
                       checkset=window_checkset(rec, reqs), phi_weight=self.phi_weight, import_tolerant=self.import_tolerant)
        t.world_image, t.workspace_mode = self.world_image, self.workspace_mode
        return t

    def target_milestones(self) -> list[str]:
        """The default target of the counterfactual control: the milestone with the deepest executed prerequisite chain (the
        last such one in history order), else the last milestone."""
        by_unit = {m["unit"]: m for m in self.milestones}

        def depth(m: Mapping[str, Any], seen: frozenset[str] = frozenset()) -> int:
            edges = m.get("edges")
            pre = [u for u in m.get("requires_units") or [] if (not isinstance(edges, Mapping) or edges.get(u) == "executed") and u in by_unit and u not in seen]
            return 1 + max((depth(by_unit[u], seen | {u}) for u in pre), default=0) if pre else 0

        scored = [(depth(m), i, m["id"]) for i, m in enumerate(self.milestones)]
        best = max(scored)
        return [best[2]] if best[0] > 0 else [self.milestones[-1]["id"]]

    def locate_hints(self, milestones: Sequence[str]) -> list[dict[str, Any]]:
        """Per target milestone: the source files its unit changed and, with the clone at hand, the functions and classes
        whose definition the unit changed or added (``ast`` dumps compared at the unit's parent and the unit)."""
        by_id = {m["id"]: m for m in self.milestones}
        units = {u["sha"]: Unit(**u) for u in self.window["units"]}
        repo = RepoGit(Path(self.lane.repo_dirs[self.window["repo"]])) if self.window["repo"] in self.lane.repo_dirs else None
        out = []
        for mid in milestones:
            u = units.get(by_id[mid]["unit"])
            if u is None:
                continue
            files = [p for p in u.src_files if not is_test_path(p)]
            names: list[str] = []
            for p in files if repo is not None else []:
                names += changed_defs(repo.show(u.parent, p), repo.show(u.sha, p))
            req = next((str(r["id"]) for r in self.requirements if r.get("milestone") == mid), None)
            out.append({"milestone": mid, "requirement": req, "unit": u.sha, "files": files, "functions": names[:12]})
        return out

    def _simplify_locate(self, *, milestones: Sequence[str] | None = None) -> WindowTask:
        """``simplify('locate')``: the same start, World, milestones, checks and requirements; the prompt also names where
        each target milestone's change lives (its unit's source files and changed functions). Difficulty removed: finding
        the code. The hint names file paths, which ``lint_prompt`` refuses in a published prompt: a simplified variant is
        a control, never published (``card().status`` stays ``draft``; ``simplified_from.kind == 'locate'``)."""
        targets = list(milestones) if milestones else self.target_milestones()
        if any(t not in {m["id"] for m in self.milestones} for t in targets):
            raise ValueError(f"unknown target milestones {targets}")
        hints = [h for h in self.locate_hints(targets) if h["files"]]
        if not hints:
            raise ValueError(f"{targets}: no source file to name (the units change tests or docs only)")
        lines = []
        for h in hints:
            fn = f" (look at {', '.join(f'`{n}`' for n in h['functions'])})" if h["functions"] else ""
            who = f"[{h['requirement']}]" if h["requirement"] else "one of the changes"
            lines.append(f"For {who}, the code to change is in {', '.join(f'`{f}`' for f in h['files'])}{fn}.")
        note = "Where to look: " + " ".join(lines) + "\n\n"
        i = self.instruction.find(EPILOGUE[:40])
        prompt = self.instruction[:i] + note + self.instruction[i:] if i >= 0 else self.instruction.rstrip() + "\n\n" + note
        key = sha256_text(self.mined.id, "locate", *targets)[:10]
        rec = json.loads(json.dumps(self.window, default=str))
        rec.update(id=f"{self.mined.id}~loc-{key}", simplified_from={"task": self.task_id, "window": self.mined.id, "kind": "locate", "targets": targets,
                                                                    "pre_applied": [], "hints": hints, "mode": "hint"})
        spans = {str(r["id"]): list(loc) for r in self.requirements if (loc := locate(prompt, r["text"]))}
        mined = mined_of(rec, self.base_dir)
        t = WindowTask(self.lane, mined, rec, task_id=f"{self.task_id}~loc-{key[:6]}", prompt=prompt, requirements=[dict(r) for r in self.requirements],
                       spans=spans, checkset=window_checkset(rec, self.requirements), phi_weight=self.phi_weight, import_tolerant=self.import_tolerant)
        t.world_image, t.workspace_mode = self.world_image, self.workspace_mode
        return t

    def card(self, *, intent: str = "unassigned", recipe_version: str = "p0", verifier: Mapping[str, Any] | None = None, status: str = "draft") -> Any:
        """The ``cards.TaskCard`` of this task (an authored one: the card needs the prompt and its ledger)."""
        from crucible.factory.cards import RequirementLedger, TaskCard, VerifierCard

        if not self.instruction.strip():
            raise ValueError("an unauthored window task has no card (no prompt)")
        led = window_ledger(self.window, self.requirements, self.spans, self.instruction)[0] if self.requirements else RequirementLedger()
        w = self.window.get("world") or {}
        edges = sum(len(m.get("requires") or []) for m in self.milestones)
        return TaskCard(id=self.task_id, family=f"window-{self.window['repo']}", intent=intent,
                        surface={"domain": "python library", "world": self.window["repo"], "tools": "shell workspace", "data_shape": "repository",
                                 "prompt_style": "release request"},
                        prompt=self.instruction, ledger=led, world_ref=f"image:{w.get('image')}@{w.get('digest')}",
                        checkset_ref="sha256:" + sha256_text(json.dumps(self.checkset, sort_keys=True)), oracle_ref=f"git:{self.window['repo']}@{self.window['b']}",
                        verifier=VerifierCard(verifier_version=self.verifier_version, **dict(verifier or {})),
                        stats={"split": self.window.get("split") or split_of(self.window["repo"]), "milestones": len(self.milestones), "executed_edges": edges},
                        lineage={"generator": "mined_window", "window": self.mined.id, "a": self.window.get("a"), "b": self.window.get("b"),
                                 "simplified_from": self.window.get("simplified_from")},
                        recipe_version=recipe_version, status=status)

    def ref(self) -> LongTaskRef:
        sf = self.window.get("simplified_from") or {}
        return LongTaskRef(lane=LANE, task_id=self.task_id, source_id=str(sf.get("window") or self.mined.id), store=str(self.lane.store.root),
                           split=str(self.window.get("split") or split_of(self.window["repo"])), prompt_sha256=sha256_text(self.instruction),
                           checkset_sha256=sha256_text(json.dumps(self.checkset, sort_keys=True)), verifier_version=self.verifier_version,
                           simplified={k: sf[k] for k in ("kind", "pre_applied", "mode") if k in sf})

    async def apply_patch_in_world(self, world: Any, patch: str) -> None:
        """Apply ``patch`` inside an open World's workspace (``longtask.clean_world_solvable(in_world=True)``)."""
        from crucible.world.exec import DockerExec

        name = ".crucible-oracle.patch"
        (self.workspace(world) / name).write_text(patch)
        out = await DockerExec(world._crucible_container).run(f"git apply --whitespace=nowarn {name}; rc=$?; rm -f {name}; exit $rc",
                                                              cwd=str(self.workspace(world)), timeout_s=300)
        if not out.ok:
            raise RuntimeError(f"the patch does not apply in the World: {out.stderr[-300:]}")

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
        return await open_world(image=self.world_image or self.window["world"]["image"], base_dir=self.base_dir, workdir=Path(workdir),
                                instruction=self.instruction, name_prefix=self.lane.world_prefix, workspace_mode=self.workspace_mode)

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
                 sandboxes: Callable[[str], DockerSandbox] | None = None, phi_weight: float = 0.5, **kw: Any):
        self._sandboxes = sandboxes or sandbox_factory()
        super().__init__(store, DockerSandbox(BASE_IMAGE, guard=True), repo_dirs=repo_dirs, reviewer=reviewer, **kw)
        self.image_digest = ""
        self.weights = {k: dict(v) for k, v in (weights or {}).items()}
        self.phi_weight = float(phi_weight)  # WindowTask's outcome reward mix (report 11 F1); 0.0 = binary pass reward
        self.world_prefix = "crucible-lh-world-"  # the policy's World containers (open_world name_prefix)
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

