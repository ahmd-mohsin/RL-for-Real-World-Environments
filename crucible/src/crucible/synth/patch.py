"""``PatchTask``: the file-tree substrate of ``crucible.synth`` (part2-paper-core §11 row 2, M5).

A code task is a repository at a start tree A′, an instruction (a user's bug report), an oracle (the fix commit B as
data) and hidden tests. This module makes it a ``SynthWorld`` task, so every falsifier, the certificate and the archive
run on it unchanged, and it keeps the debug lane's verdicts: the grade of an end state is the lane's own grade, reached
through an injected exec runner (``factory/synth_patch_wire.py`` binds the lane; synth never imports factory).

**The World** (``PatchWorld``). An overlay of writes over the read-only start tree ``base_dir``. Tools:
``repo.read_file``, ``repo.list_files`` (read-only), ``repo.write_file``, ``repo.delete_file``, ``repo.apply_patch`` (a
unified git diff, applied in pure Python with context checks; a hunk that does not match is a tool error and nothing is
written). ``canon()`` is ``canon_fs`` of A′ (path → ``sha256``, ``mode``, ``text`` when small) with the overlay applied,
plus a ``role`` per path from the injected path classifier: ``source``, ``doc``, ``test``, ``golden`` (test data,
snapshots, case files), ``config`` (build, CI, lockfiles), ``vendored``, ``plumbing`` (what can steer a test run) and
``vcs`` (``.git``). Paths are relative and may not escape the tree. Every written text is kept in a content store by
``sha256`` (``PatchEnv.blobs``), so an end state can be turned back into files without trusting the agent.

**The oracle program.** One ``repo.write_file`` (or ``repo.delete_file``) step per file B changes, sorted by path: the
fix commit as data. Mutants and null agents are programs of the same tools.

**The contract** (``PatchEnv.contract``). The lane's grader as data:
- R: one ``exec`` check per lane check (``c_<milestone>``: that milestone's hidden fail-to-pass tests, kind
  ``require``; ``c_regress``: the existing pass-to-pass suite, kind ``guard``);
- A: any change to a path whose role is ``source`` or ``doc`` (``why``: the agent may change any source; the hidden
  tests decide);
- every other change is a ``FORBIDDEN_EFFECT``: tests, golden data, build and CI config, vendored dependencies, test
  plumbing and VCS metadata. The lane restores protected paths before its tests run and fails the full pass on a
  protected write (``path:protected``); the frame states the same rule over the end state.

**Exec in custody** (``PatchEnv.exec_fn``). The exec predicate reads the run's end ``CanonState`` only, maps every path
whose content differs from A′ back to its text through the content store (an unknown hash is a ``CustodyError``: fail
closed), and asks the runner for the verdict of those files. A runner (``ExecRunner``) returns an ``ExecResult`` (per
check pass/fail, per test status, infra); ``ThreadedRunner`` runs an async grade function on its own event loop thread
with at most ``parallel`` runs at once and caches one result per distinct file set (two probes with one end state are
one Docker run); ``FuncRunner`` wraps a plain function (tests). Infra failures are never verdicts: the grader raises,
the contract fails closed with ``GRADER_ERROR:exec``, and the patch falsifiers mark that probe invalid.

**The lane's reward** (``PatchEnv.phi``, ``lane_reward``): Φ = Σ w_m over milestones that pass with every ``requires``
passing, reward = ``phi_weight``·Φ + (1 − ``phi_weight``)·[every gating check passes], rounded as the lane rounds them,
so a stored reward can be recomputed to the bit (``migrate``).

**Migration** (``migrate``): the oracle, the start state, the lane's floors (as their own patches) and its commit-revert
mutants graded through the adapter and compared with the lane's stored gate record (S8 rows, S3 grading) and the
window's mutant vectors. Every row must be equal.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import difflib
import hashlib
import re
import threading
import time
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import CanonState, canon_fs, cjson, content_hash
from crucible.synth.contract import Contract, EffectPattern, ExecCheck, ExecFn, PIn
from crucible.synth.falsify.base import FalsifyContext
from crucible.synth.iface import Obs, ToolInfo
from crucible.synth.program import Lit, Program, Run, Step
from crucible.synth.task import IntentCard, Task, WorldRef

APP, COLL = "repo", "files"
ALLOWED_ROLES = ("source", "doc")
FORBIDDEN_ROLES = ("test", "golden", "config", "vendored", "plumbing", "vcs")
ROLES = ALLOWED_ROLES + FORBIDDEN_ROLES
RoleFn = Callable[[str], str]
MAX_READ = 1_000_000
MAX_LIST = 5000


class PatchError(ValueError):
    """A tool refusing a call (bad path, missing file, a hunk that does not apply): an observation, never a crash."""


class CustodyError(LookupError):
    """An end state names content the World never wrote (a hash outside the content store): the grade fails closed."""


class InfraError(RuntimeError):
    """The runner could not measure an end state (a sandbox that did not start, a timeout): never a verdict."""


# ---------------------------------------------------------------------------------------------------------- roles

VCS_RX = re.compile(r"(^|/)\.git(/|$)|(^|/)\.gitmodules$")
VENDORED_RX = re.compile(r"(^|/)(vendor|node_modules|third_party|\.venv|site-packages)(/|$)")
PLUMBING_RX = re.compile(r"(^|/)(conftest\.py|sitecustomize\.py|usercustomize\.py|go\.work(\.sum)?|\.npmrc|\.yarnrc(\.yml)?|\.pnpmfile\.cjs)$|\.pth$")
GOLDEN_RX = re.compile(r"(^|/)(testdata|tests?/data|__snapshots__|fixtures?|snapshots)(/|$)|\.(snap|golden)$")
TEST_RX = re.compile(r"(^|/)(tests?|__tests__|spec)/|(^|/)test_[^/]*\.py$|_test\.(py|go|rs)$|\.(test|spec)\.[cm]?[jt]sx?$")
CONFIG_RX = re.compile(r"(^|/)(\.github|\.gitlab|\.circleci)/|(^|/)(\.gitlab-ci\.yml|\.travis\.yml|azure-pipelines\.yml|Jenkinsfile|Makefile|"
                     r"pyproject\.toml|setup\.cfg|setup\.py|tox\.ini|pytest\.ini|noxfile\.py|go\.mod|go\.sum|Cargo\.toml|Cargo\.lock|"
                     r"package\.json|package-lock\.json|yarn\.lock|pnpm-lock\.yaml|tsconfig[^/]*\.json|(jest|vitest|vite)\.config\.[^/]+|"
                     r"\.gitattributes|\.golangci\.ya?ml|rust-toolchain(\.toml)?)$")
DOC_RX = re.compile(r"\.(md|rst|txt|adoc)$|(^|/)(docs?)/|(^|/)(CHANGELOG|CHANGES|NEWS|HISTORY|AUTHORS|LICENSE|README)[^/]*$", re.IGNORECASE)


def default_roles(path: str) -> str:
    """A generic path classifier (the wire injects the lane's own: ``langs`` profiles and the hidden test paths)."""
    if VCS_RX.search(path):
        return "vcs"
    if VENDORED_RX.search(path):
        return "vendored"
    if PLUMBING_RX.search(path):
        return "plumbing"
    if GOLDEN_RX.search(path):
        return "golden"
    if TEST_RX.search(path):
        return "test"
    if CONFIG_RX.search(path):
        return "config"
    if DOC_RX.search(path):
        return "doc"
    return "source"


def safe_path(path: Any) -> str:
    if not isinstance(path, str):
        raise PatchError("path must be a string")
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or "\\" in p or "\0" in p or any(s in ("..", "", ".") for s in p.split("/")):
        raise PatchError(f"unsafe path {path!r}")
    return p


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------------------------------------- the spec

class Milestone(CrucibleModel):
    id: str
    tests: list[str] = Field(default_factory=list)
    requires: list[str] = Field(default_factory=list)


class Requirement(CrucibleModel):
    id: str
    text: str = ""
    milestone: str | None = None


class FixUnit(CrucibleModel):
    sha: str = ""
    subject: str = ""


class RevertMutant(CrucibleModel):
    """A commit-revert mutant of the lane: B with one fix unit reverted; ``observed`` is the lane's executed vector."""

    id: str
    milestone: str
    files: dict[str, str | None] = Field(default_factory=dict)
    observed: dict[str, bool] = Field(default_factory=dict)
    status: str = "valid"


class PatchSpec(CrucibleModel):
    """Everything the adapter needs about one code task, as data (the wire fills it from the lane's window record)."""

    task_id: str
    domain: str = "debug"
    lang: str = "python"
    base_dir: str
    instruction: str = ""
    requirements: list[Requirement] = Field(default_factory=list)
    milestones: list[Milestone] = Field(default_factory=list)
    checks: dict[str, list[str]] = Field(default_factory=dict)  # check id -> its tests (the lane's checkset)
    guards: list[str] = Field(default_factory=list)  # check ids that are pass-to-pass guards (c_regress)
    gating: list[str] | None = None  # check ids that gate the full pass (None: all)
    milestone_check: dict[str, str] = Field(default_factory=dict)  # milestone -> its check id (default c_<id>)
    weights: dict[str, float] = Field(default_factory=dict)  # the lane's normalised milestone weights (empty: equal)
    phi_weight: float = 0.5
    oracle_files: dict[str, str | None] = Field(default_factory=dict)
    mutants: list[RevertMutant] = Field(default_factory=list)
    hidden_tests: dict[str, str] = Field(default_factory=dict)  # B's test files the grader writes over the tree
    units: list[FixUnit] = Field(default_factory=list)
    floors: dict[str, str] = Field(default_factory=dict)  # the lane's floor solutions as patches (name -> git diff)
    start_files: dict[str, str | None] = Field(default_factory=dict)  # a start overlay over base_dir (a defect version)
    comment: str = "//"  # the language's line comment (inert edits)
    text_max: int = 4096
    meta: dict[str, Any] = Field(default_factory=dict)

    def check_of(self, mid: str) -> str:
        return self.milestone_check.get(mid, f"c_{mid}")

    def gating_checks(self) -> list[str]:
        return list(self.checks) if self.gating is None else [c for c in self.checks if c in self.gating]


# ---------------------------------------------------------------------------------------------------------- the runner

class ExecResult(CrucibleModel):
    """One measured end state: per check pass/fail, per test status, and whether it was measured at all."""

    infra: str = "ok"
    why: str = ""
    checks: dict[str, bool] = Field(default_factory=dict)
    statuses: dict[str, str] = Field(default_factory=dict)
    seconds: float = 0.0
    key: str = ""

    @property
    def ok(self) -> bool:
        return self.infra == "ok"


def passed_all(statuses: Mapping[str, str], tests: Sequence[str]) -> bool:
    """Every test passed (the lane's ``passed_group``)."""
    return bool(tests) and all(statuses.get(t) == "passed" for t in tests)


def present_passed(statuses: Mapping[str, str], tests: Sequence[str]) -> bool:
    """Every *collected* test passed (the lane's ``passed_group(require_all=False)``, S3's start-state probe)."""
    present = [t for t in tests if t in statuses]
    return bool(present) and all(statuses[t] == "passed" for t in present)


def files_key(files: Mapping[str, str | None], mode: str = "grade") -> str:
    return content_hash({"mode": mode, "files": {p: (None if t is None else sha_text(t)) for p, t in sorted(files.items())}}, 32)


GradeFn = Callable[[dict[str, str | None], str], Awaitable[ExecResult]]


class ExecRunner(Protocol):
    def submit(self, files: Mapping[str, str | None], *, mode: str = "grade", fresh: bool = False) -> concurrent.futures.Future[ExecResult]: ...

    def result(self, files: Mapping[str, str | None], *, mode: str = "grade", fresh: bool = False, timeout: float | None = None) -> ExecResult: ...

    def prefetch(self, many: Iterable[Mapping[str, str | None]], *, mode: str = "grade") -> None: ...


class _CachingRunner:
    def __init__(self) -> None:
        self._cache: dict[str, concurrent.futures.Future[ExecResult]] = {}
        self._lock = threading.Lock()
        self.log: list[dict[str, Any]] = []  # one row per executed measurement (cache hits are not runs)

    def _start(self, files: dict[str, str | None], mode: str, key: str, fresh: bool) -> concurrent.futures.Future[ExecResult]:
        raise NotImplementedError

    def submit(self, files: Mapping[str, str | None], *, mode: str = "grade", fresh: bool = False) -> concurrent.futures.Future[ExecResult]:
        key = files_key(files, mode)
        with self._lock:
            if not fresh and key in self._cache:
                return self._cache[key]
            fut = self._start(dict(files), mode, key, fresh)
            if not fresh:
                self._cache[key] = fut
        return fut

    def result(self, files: Mapping[str, str | None], *, mode: str = "grade", fresh: bool = False, timeout: float | None = None) -> ExecResult:
        res = self.submit(files, mode=mode, fresh=fresh).result(timeout)
        if not res.ok and not fresh:  # an unmeasured state is retried by the next caller, never cached as a verdict
            with self._lock:
                self._cache.pop(res.key, None)
        return res

    def prefetch(self, many: Iterable[Mapping[str, str | None]], *, mode: str = "grade") -> None:
        for f in many:
            self.submit(f, mode=mode)

    def _note(self, res: ExecResult, mode: str, fresh: bool, n_files: int) -> None:
        self.log.append({"key": res.key, "mode": mode, "fresh": fresh, "files": n_files, "infra": res.infra, "seconds": res.seconds,
                         "checks": dict(res.checks)})


class FuncRunner(_CachingRunner):
    """A synchronous grade function ``fn(files, mode) -> ExecResult`` (tests, fixtures), cached like the threaded one."""

    def __init__(self, fn: Callable[[dict[str, str | None], str], ExecResult]):
        super().__init__()
        self.fn = fn

    def _start(self, files: dict[str, str | None], mode: str, key: str, fresh: bool) -> concurrent.futures.Future[ExecResult]:
        fut: concurrent.futures.Future[ExecResult] = concurrent.futures.Future()
        t0 = time.monotonic()
        try:
            res = self.fn(files, mode)
        except Exception as e:  # noqa: BLE001 - a crashed measurement is infra, never a verdict
            res = ExecResult(infra="error", why=f"{type(e).__name__}: {e}"[:500])
        res = res.model_copy(update={"key": key, "seconds": round(time.monotonic() - t0, 3)})
        self._note(res, mode, fresh, len(files))
        fut.set_result(res)
        return fut


class ThreadedRunner(_CachingRunner):
    """An async grade function on a dedicated event loop thread, at most ``parallel`` measurements at once. The grader
    calls exec predicates synchronously (in a worker thread), so the lane's async sandboxes live on this one loop."""

    def __init__(self, grade: GradeFn, *, parallel: int = 4, name: str = "synth-patch-exec"):
        super().__init__()
        if parallel < 1:
            raise ValueError("parallel >= 1")
        self.grade, self.parallel = grade, parallel
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, name=name, daemon=True)
        self._thread.start()
        self._sem: asyncio.Semaphore | None = None

    async def _run(self, files: dict[str, str | None], mode: str, key: str, fresh: bool) -> ExecResult:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.parallel)
        async with self._sem:
            t0 = time.monotonic()
            try:
                res = await self.grade(files, mode)
            except Exception as e:  # noqa: BLE001
                res = ExecResult(infra="error", why=f"{type(e).__name__}: {e}"[:500])
            res = res.model_copy(update={"key": key, "seconds": round(time.monotonic() - t0, 3)})
        self._note(res, mode, fresh, len(files))
        return res

    def _start(self, files: dict[str, str | None], mode: str, key: str, fresh: bool) -> concurrent.futures.Future[ExecResult]:
        return asyncio.run_coroutine_threadsafe(self._run(files, mode, key, fresh), self._loop)

    def close(self) -> None:
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            self._thread.join(timeout=30)


# ---------------------------------------------------------------------------------------------------------- unified diffs

_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


class FilePatch(CrucibleModel):
    old: str | None = None  # None: /dev/null (a new file)
    new: str | None = None  # None: /dev/null (a deletion)
    hunks: list[list[str]] = Field(default_factory=list)  # each: the header line, then ' ', '-', '+', '\\' lines


def _strip_ab(p: str) -> str | None:
    p = p.strip().split("\t", 1)[0].strip('"')
    if p == "/dev/null":
        return None
    return p[2:] if p.startswith(("a/", "b/")) else p


def parse_patch(patch: str) -> list[FilePatch]:
    """The file patches of a unified (git) diff. Binary patches are refused."""
    if "GIT binary patch" in patch:
        raise PatchError("binary patches are not supported")
    out: list[FilePatch] = []
    cur: FilePatch | None = None
    lines = patch.splitlines()
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("--- ") and i + 1 < len(lines) and lines[i + 1].startswith("+++ "):
            cur = FilePatch(old=_strip_ab(ln[4:]), new=_strip_ab(lines[i + 1][4:]))
            out.append(cur)
            i += 2
            continue
        if ln.startswith("@@ ") and cur is not None:
            cur.hunks.append([ln])
        elif cur is not None and cur.hunks and ln[:1] in (" ", "-", "+", "\\"):
            cur.hunks[-1].append(ln)
        elif cur is not None and cur.hunks and ln == "":  # an empty context line whose leading space was stripped
            cur.hunks[-1].append(" ")
        i += 1
    if not out:
        raise PatchError("the patch has no file diff")
    return out


def apply_hunks(old: str | None, hunks: Sequence[Sequence[str]]) -> str:
    """``old`` with ``hunks`` applied; every context and removed line must match (an offset of up to 200 lines is
    searched, nearest first). A ``\\ No newline at end of file`` marker drops the newline of the line before it."""
    src = (old or "").splitlines(keepends=True)
    out: list[str] = []
    pos = 0
    for h in hunks:
        m = _HUNK.match(h[0])
        if m is None:
            raise PatchError(f"bad hunk header {h[0][:60]!r}")
        want: list[str] = []
        repl: list[str] = []
        prev: list[str] | None = None
        for ln in h[1:]:
            tag, body = ln[:1], ln[1:] + "\n"
            if tag == "\\":
                for lst in (want, repl) if prev is None else (prev,):
                    if lst and lst[-1].endswith("\n"):
                        lst[-1] = lst[-1][:-1]
                continue
            if tag == " ":
                want.append(body)
                repl.append(body)
                prev = None
            elif tag == "-":
                want.append(body)
                prev = want
            elif tag == "+":
                repl.append(body)
                prev = repl
        start = max(int(m.group(1)) - 1, 0) if want else int(m.group(1))
        at = _find(src, want, start, pos)
        if at is None:
            raise PatchError(f"hunk {h[0]} does not apply")
        out += src[pos:at]
        out += repl
        pos = at + len(want)
    out += src[pos:]
    return "".join(out)


def _same(a: str, b: str) -> bool:
    return a.rstrip("\n") == b.rstrip("\n")


def _find(src: Sequence[str], want: Sequence[str], start: int, lo: int) -> int | None:
    if not want:
        return max(lo, min(start, len(src)))
    for d in range(201):
        for at in (start + d, start - d) if d else (start,):
            if at < lo or at + len(want) > len(src):
                continue
            if all(_same(src[at + k], want[k]) for k in range(len(want))):
                return at
    return None


def diff_hunks(old: str, new: str) -> list[tuple[int, int, int, int]]:
    """The changed regions between two texts as line opcodes ``(i1, i2, j1, j2)`` (``difflib``, no context merged)."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    return [(i1, i2, j1, j2) for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]


def revert_hunk(old: str, new: str, k: int) -> str:
    """``new`` with its ``k``-th changed region put back as it is in ``old`` (every other change kept)."""
    a, b = old.splitlines(keepends=True), new.splitlines(keepends=True)
    out: list[str] = []
    n = -1
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            out += b[j1:j2]
            continue
        n += 1
        out += a[i1:i2] if n == k else b[j1:j2]
    return "".join(out)


def cosmetic(old_lines: Sequence[str], new_lines: Sequence[str], comment: str) -> bool:
    """A change of blank or comment lines only (it cannot change behaviour, except where tests read comments)."""
    marks = tuple({comment, "#", "//", "/*", "*", "*/", "--"})
    return all(not ln.strip() or ln.strip().startswith(marks) for ln in [*old_lines, *new_lines])


# ---------------------------------------------------------------------------------------------------------- the World

TOOLS = [
    ToolInfo(name="repo.read_file", description="Read one file of the repository as UTF-8 text.",
             input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, read_only=True),
    ToolInfo(name="repo.list_files", description="List the repository's files whose path starts with a prefix.",
             input_schema={"type": "object", "properties": {"prefix": {"type": "string", "default": ""}}, "required": []}, read_only=True),
    ToolInfo(name="repo.write_file", description="Write a whole file (create or replace).",
             input_schema={"type": "object", "properties": {"path": {"type": "string"}, "text": {"type": "string"}}, "required": ["path", "text"]},
             writes=[f"{APP}.{COLL}"]),
    ToolInfo(name="repo.delete_file", description="Delete one file.",
             input_schema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}, writes=[f"{APP}.{COLL}"]),
    ToolInfo(name="repo.apply_patch", description="Apply a unified diff to the repository.",
             input_schema={"type": "object", "properties": {"patch": {"type": "string"}}, "required": ["patch"]}, writes=[f"{APP}.{COLL}"]),
]


class PatchWorld:
    """A ``SynthWorld`` over one start tree (see the module docstring)."""

    def __init__(self, env: PatchEnv):
        self.env = env
        self.overlay: dict[str, str | None] = dict(env.spec.start_files)
        self._snaps: dict[str, dict[str, str | None]] = {}

    async def reset(self, seed: int) -> str:
        self.overlay = dict(self.env.spec.start_files)
        return files_key(self.overlay)

    async def tools(self) -> list[ToolInfo]:
        return [t.model_copy(deep=True) for t in TOOLS]

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        if not isinstance(args, dict):
            return Obs(error="arguments must be an object")
        try:
            return Obs(value=self._call(tool, args))
        except PatchError as e:
            return Obs(error=str(e))

    def text(self, path: str) -> str | None:
        if path in self.overlay:
            return self.overlay[path]
        return self.env.base_text(path)

    def _call(self, tool: str, args: dict[str, Any]) -> Any:
        if tool == "repo.read_file":
            p = safe_path(args.get("path"))
            t = self.text(p)
            if t is None:
                raise PatchError(f"no file {p}")
            if len(t) > MAX_READ:
                raise PatchError(f"{p} is larger than {MAX_READ} characters")
            return t
        if tool == "repo.list_files":
            prefix = args.get("prefix") or ""
            if not isinstance(prefix, str):
                raise PatchError("prefix must be a string")
            paths = sorted(p for p in self.paths() if p.startswith(prefix))
            return paths[:MAX_LIST]
        if tool == "repo.write_file":
            p, t = safe_path(args.get("path")), args.get("text")
            if not isinstance(t, str):
                raise PatchError("text must be a string")
            self.env.put(t)
            self.overlay[p] = t
            return {"written": p, "bytes": len(t.encode())}
        if tool == "repo.delete_file":
            p = safe_path(args.get("path"))
            if self.text(p) is None:
                raise PatchError(f"no file {p}")
            self.overlay[p] = None
            return {"deleted": p}
        if tool == "repo.apply_patch":
            patch = args.get("patch")
            if not isinstance(patch, str):
                raise PatchError("patch must be a string")
            staged: dict[str, str | None] = {}
            for fp in parse_patch(patch):
                src = safe_path(fp.old) if fp.old is not None else None
                dst = safe_path(fp.new) if fp.new is not None else None
                cur = staged[src] if src is not None and src in staged else (self.text(src) if src is not None else None)
                if src is not None and cur is None:
                    raise PatchError(f"no file {src}")
                if dst is None:
                    staged[src] = None  # type: ignore[index]
                    continue
                new = apply_hunks(cur, fp.hunks)
                if src is not None and src != dst:
                    staged[src] = None
                staged[dst] = new
            for p, t in staged.items():
                if t is not None:
                    self.env.put(t)
                self.overlay[p] = t
            return {"changed": sorted(staged)}
        raise PatchError(f"unknown tool {tool!r}")

    def paths(self) -> set[str]:
        out = set(self.env.base_rows())
        for p, t in self.overlay.items():
            if t is None:
                out.discard(p)
            else:
                out.add(p)
        return out

    async def snapshot(self) -> str:
        ref = f"snap-{len(self._snaps)}-{files_key(self.overlay)[:12]}"
        self._snaps[ref] = dict(self.overlay)
        return ref

    async def restore(self, ref: Any) -> None:
        self.overlay = dict(self._snaps[ref])

    async def canon(self) -> CanonState:
        return self.env.canon(self.overlay)


# ---------------------------------------------------------------------------------------------------------- the task

class PatchEnv:
    """One code task: the World, the oracle, the contract, the exec predicate and the lane's reward (module doc)."""

    def __init__(self, spec: PatchSpec, *, roles: RoleFn | None = None):
        self.spec = spec
        self.roles = roles or default_roles
        self.base_dir = Path(spec.base_dir)
        self.blobs: dict[str, str] = {}
        self._base: dict[str, dict[str, Any]] | None = None
        self.exec_name = f"patch:{spec.task_id}"
        for t in spec.start_files.values():
            if t is not None:
                self.put(t)

    # -- state -----------------------------------------------------------------------------------------------------
    def put(self, text: str) -> str:
        h = sha_text(text)
        self.blobs[h] = text
        return h

    def base_rows(self) -> dict[str, dict[str, Any]]:
        if self._base is None:
            rows = canon_fs(APP, self.base_dir, coll=COLL, text_max=self.spec.text_max).collection(APP, COLL)
            self._base = {p: {**r, "role": self.roles(p)} for p, r in rows.items()}
        return self._base

    def base_text(self, path: str) -> str | None:
        if path not in self.base_rows() or "link" in self.base_rows()[path]:
            return None
        try:
            return (self.base_dir / path).read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            return None

    def row(self, path: str, text: str) -> dict[str, Any]:
        base = self.base_rows().get(path)
        r: dict[str, Any] = {"sha256": sha_text(text), "mode": (base or {}).get("mode", "644"), "role": self.roles(path)}
        if len(text.encode("utf-8")) <= self.spec.text_max:
            r["text"] = text
        return r

    def canon(self, overlay: Mapping[str, str | None]) -> CanonState:
        rows = dict(self.base_rows())
        for p, t in overlay.items():
            if t is None:
                rows.pop(p, None)
            else:
                rows[p] = self.row(p, t)
        return CanonState.build({APP: {COLL: rows}})

    def overlay_of(self, state: CanonState) -> dict[str, str | None]:
        """The files an end state changed relative to A′, with their texts from the content store (custody)."""
        base = self.base_rows()
        rows = state.collection(APP, COLL)
        out: dict[str, str | None] = {}
        for p, r in rows.items():
            b = base.get(p)
            if b is not None and b.get("sha256") == r.get("sha256") and b.get("link") == r.get("link"):
                continue
            if "link" in r:
                raise CustodyError(f"{p}: a symlink the World cannot write")
            t = r.get("text")
            if not isinstance(t, str) or sha_text(t) != r.get("sha256"):
                t = self.blobs.get(str(r.get("sha256")))
            if t is None:
                raise CustodyError(f"{p}: content {str(r.get('sha256'))[:12]} is not in custody")
            out[p] = t
        for p in base:
            if p not in rows:
                out[p] = None
        return out

    def world(self) -> PatchWorld:
        return PatchWorld(self)

    # -- programs --------------------------------------------------------------------------------------------------
    def program(self, files: Mapping[str, str | None], *, prefix: str = "w", tail: Sequence[Step] = ()) -> Program:
        steps = []
        for i, (p, t) in enumerate(sorted(files.items())):
            if t is None:
                steps.append(Step(id=f"{prefix}{i}", tool="repo.delete_file", args={"path": Lit(v=p)}))
            else:
                steps.append(Step(id=f"{prefix}{i}", tool="repo.write_file", args={"path": Lit(v=p), "text": Lit(v=t)}))
        return Program(steps=[*steps, *tail])

    def oracle(self) -> Program:
        return self.program(self.spec.oracle_files)

    def patch_program(self, patch: str, sid: str = "p0") -> Program:
        return Program(steps=[Step(id=sid, tool="repo.apply_patch", args={"patch": Lit(v=patch)})])

    # -- the contract ------------------------------------------------------------------------------------------------
    def contract(self) -> Contract:
        allowed = [EffectPattern(id="a_source", app=APP, coll=COLL, key="*", field="*", kind="any", where={"role": PIn(vs=list(ALLOWED_ROLES))},
                                 why="the agent may change any source or doc file; the hidden tests decide")]
        ex = [ExecCheck(id=cid, name=self.exec_name, args={"check": cid}, kind="guard" if cid in self.spec.guards else "require")
              for cid in self.spec.gating_checks()]
        return Contract(allowed=allowed, exec=ex)

    def task(self) -> Task:
        intent = IntentCard(skill="debug interacting faults from a symptom report", notes=f"lang={self.spec.lang}")
        return Task(id=self.spec.task_id, domain=self.spec.domain, world=WorldRef(factory=f"patch:{self.spec.task_id}"), instruction=self.spec.instruction,
                    contract=self.contract(), oracle=self.oracle(), intent=intent)

    def exec_fn(self, runner: ExecRunner, *, timeout: float | None = None) -> ExecFn:
        def f(run: Run, args: dict[str, Any]) -> bool:
            if run.end is None:
                raise CustodyError("no end state")
            res = runner.result(self.overlay_of(run.end), mode=str(args.get("mode") or "grade"), timeout=timeout)
            if not res.ok:
                raise InfraError(res.why or res.infra)
            cid = str(args["check"])
            if cid not in res.checks:
                raise InfraError(f"check {cid} was not measured")
            return bool(res.checks[cid])

        return f

    def context(self, runner: ExecRunner, *, deadline_s: float = 7200.0, registry: Mapping[str, ExecFn] | None = None) -> FalsifyContext:
        reg = {**dict(registry or {}), self.exec_name: self.exec_fn(runner)}
        return FalsifyContext(lambda _task: self.world(), exec_registry=reg, deadline_s=deadline_s)

    # -- the lane's reward -------------------------------------------------------------------------------------------
    def weights(self) -> dict[str, float]:
        if self.spec.weights:
            return {m.id: float(self.spec.weights.get(m.id, 0.0)) for m in self.spec.milestones}
        total = float(len(self.spec.milestones)) or 1.0
        return {m.id: 1.0 / total for m in self.spec.milestones}

    def milestones_of(self, res: ExecResult) -> dict[str, bool]:
        return {m.id: passed_all(res.statuses, m.tests) if res.statuses else bool(res.checks.get(self.spec.check_of(m.id))) for m in self.spec.milestones}

    def phi(self, status: Mapping[str, bool]) -> float:
        w = self.weights()
        return round(sum(w[m.id] for m in self.spec.milestones if status.get(m.id) and all(status.get(r) for r in m.requires)), 6)

    def full_pass(self, res: ExecResult) -> bool:
        g = self.spec.gating_checks()
        return bool(g) and all(res.checks.get(c) for c in g)

    def lane_reward(self, res: ExecResult, *, path_ok: bool = True) -> float:
        full = self.full_pass(res) and path_ok
        return round(self.spec.phi_weight * float(self.phi(self.milestones_of(res))) + (1.0 - self.spec.phi_weight) * float(full), 6)

    def forbidden_writes(self, overlay: Mapping[str, str | None]) -> list[str]:
        return sorted(p for p in overlay if self.roles(p) in FORBIDDEN_ROLES)


# ---------------------------------------------------------------------------------------------------------- migration

class Row(CrucibleModel):
    case: str
    field: str
    lane: Any = None
    synth: Any = None
    equal: bool = False
    note: str = ""


class MigrationReport(CrucibleModel):
    task_id: str
    rows: list[Row] = Field(default_factory=list)
    runs: int = 0
    seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)

    @property
    def equal(self) -> bool:
        return bool(self.rows) and not self.errors and all(r.equal for r in self.rows)

    def differences(self) -> list[Row]:
        return [r for r in self.rows if not r.equal]


class CaseVerdict(CrucibleModel):
    passed: bool
    reasons: list[str] = Field(default_factory=list)
    milestones: dict[str, bool] = Field(default_factory=dict)
    present: dict[str, bool] = Field(default_factory=dict)
    reward: float = 0.0
    full_pass: bool = False
    infra: str = "ok"


async def case_verdict(env: PatchEnv, ctx: FalsifyContext, task: Task, program: Program, runner: ExecRunner) -> CaseVerdict:
    """The adapter's verdict on one end state (the contract's), with the lane-shaped numbers recomputed from its run."""
    r = await ctx.run(task, program)
    v = ctx.grade(r, task.contract)
    overlay = env.overlay_of(r.end) if r.end is not None else {}
    res = await asyncio.to_thread(runner.result, overlay)
    if not res.ok:
        return CaseVerdict(passed=False, reasons=v.reasons, infra=res.infra)
    path_ok = not env.forbidden_writes(overlay)
    return CaseVerdict(passed=v.passed, reasons=v.reasons, milestones=env.milestones_of(res),
                       present={m.id: present_passed(res.statuses, m.tests) for m in env.spec.milestones}, reward=env.lane_reward(res, path_ok=path_ok),
                       full_pass=env.full_pass(res) and path_ok, infra=res.infra)


def _rows(case: str, pairs: Iterable[tuple[str, Any, Any]]) -> list[Row]:
    out = []
    for f, lane, synth in pairs:
        eq = cjson(lane) == cjson(synth) if not isinstance(lane, float) else isinstance(synth, int | float) and abs(float(synth) - lane) <= 1e-9
        out.append(Row(case=case, field=f, lane=lane, synth=synth, equal=eq))
    return out


async def migrate(env: PatchEnv, runner: ExecRunner, gate_record: Mapping[str, Any]) -> MigrationReport:
    """Recompute the lane's stored verdicts through the adapter (module doc). ``gate_record`` is the lane's
    ``gates/<id>.json``; the mutant vectors are the spec's (the window record's ``observed``)."""
    t0 = time.monotonic()
    rep = MigrationReport(task_id=env.spec.task_id)
    task = env.task()
    ctx = env.context(runner)
    gates = gate_record.get("gates") or {}
    rows8 = ((gates.get("S8") or {}).get("evidence") or {}).get("rows") or {}
    grading = ((gates.get("S3") or {}).get("evidence") or {}).get("grading") or {}
    cases: list[tuple[str, Program]] = [("oracle_patch", env.oracle()), ("no_op", Program())]
    cases += [(name, env.patch_program(p)) for name, p in sorted(env.spec.floors.items()) if name in rows8]
    cases += [(m.id, env.program(m.files)) for m in env.spec.mutants]
    # every end state measured at once (the runner bounds the concurrency), then graded case by case from the cache
    worlds = []
    for _name, prog in cases:
        r = await ctx.run(task, prog)
        if r.error is None and r.end is not None:
            worlds.append(env.overlay_of(r.end))
    runner.prefetch(worlds)
    for name, prog in cases:
        try:
            cv = await case_verdict(env, ctx, task, prog, runner)
        except Exception as e:  # noqa: BLE001 - a case that cannot be graded is reported, never a silent pass
            rep.errors.append(f"{name}: {type(e).__name__}: {e}"[:300])
            continue
        if cv.infra != "ok":
            rep.errors.append(f"{name}: unmeasured ({cv.infra})")
            continue
        row = rows8.get(name)
        if row is not None:
            rep.rows += _rows(name, [("reward", float(row.get("reward") or 0.0), cv.reward)])
            if name in ("oracle_patch", "no_op") or str(row.get("kind")) == "floor":
                rep.rows += _rows(name, [("verdict", float(row.get("reward") or 0.0) >= 1.0 - 1e-9, cv.passed)])
        if name == "oracle_patch" and grading.get("oracle_milestones") is not None:
            rep.rows += _rows(name, [("milestones", dict(grading["oracle_milestones"]), cv.milestones)])
        if name == "no_op" and grading.get("start_any_milestone_test") is not None:
            rep.rows += _rows(name, [("start_any_milestone_test", dict(grading["start_any_milestone_test"]), cv.present)])
        mut = next((m for m in env.spec.mutants if m.id == name), None)
        if mut is not None:
            if mut.observed:
                own = {m.id for m in env.spec.milestones}
                # a simplified window keeps its parent's mutant vectors: milestones it no longer has are not compared
                extra = sorted(set(mut.observed) - own)
                rows = _rows(name, [("milestones", {k: v for k, v in mut.observed.items() if k in own}, cv.milestones)])
                if extra:
                    rows[0].note = f"stored vector also names {extra}, milestones this task does not have (its parent's); compared on its own"
                rep.rows += rows
            rep.rows += _rows(name, [("verdict", False, cv.passed)])
    rep.runs = len(getattr(runner, "log", []))
    rep.seconds = round(time.monotonic() - t0, 1)
    return rep

