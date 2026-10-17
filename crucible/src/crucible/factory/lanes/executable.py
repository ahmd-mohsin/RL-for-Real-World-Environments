"""The executable ground-truth lane (05 §3): real fix + test commits mined from small pure-Python git repositories.

**Mining** (PATTERN, SWE-smith ``swesmith/harness/valid.py:56-147`` and ``swesmith/harness/grading.py:40-90``; R2E-Gym's
commit mining as described in ``RL-Environments/08-hf-rl-env-datasets/PrimeIntellect__R2E-Gym-Subset-Verified/README.md``):
a candidate is a non-merge commit that changes 1-2 existing source files (a small diff) together with test files. Its
parent is the start state, its source change the oracle, and the tests it changed or added are the ground-truth checks.
SWE-smith runs the tests before the gold patch ("pre-gold") and after it, and keeps instances with at least one
FAIL_TO_PASS test (``0_f2p`` is dropped). Here both sides run ``reruns`` times in a Docker CPU sandbox; a test whose
status differs between reruns is flaky and never a check; a test that passed before and fails after (PASS_TO_FAIL) drops
the commit. Every surviving commit carries one ``ObservationRecord`` of type ``real_pr_tests`` whose sha256 binds the
commit sha and the content of its test files.

**Grading** (``ExecTask.grade``): a fresh copy of the start tree, the solution applied (search/replace ``edits`` or whole
``files``), the commit's test files written over whatever the solution did to them, and every protected path
(``conftest.py``, ``sitecustomize.py``, pytest/packaging config, ``*.pth``, any test file) left as it was. pytest runs in
``--network none``, a read-only tree, a read-only root filesystem, a tmpfs ``/tmp``; results come from a JUnit file at a
per-grade random path. A check passes when every test it names passed; the reward is 1 when every gating check passes.
A sandbox that could not start is ``masked`` (unmeasured), never a 0.

**The task** (``ExecTask``) implements the accept contract (``accept.contract.AcceptTask`` plus the capabilities of A1-A16)
and ``driver.LANE_TASK``: ``grade``, ``noop``, ``oracle``, ``instruction``, ``negatives``, ``solution_of``,
``property_audit`` (an independent check: protected paths untouched, the commit's tests pass in a run that ignores the
solution's own test plumbing, and the repository's whole suite fails nothing the oracle passes).

**Machine handlers** (``lane_handlers``): spec, author, oracle, verifier and second verifier are lane-specific (the World
is the mined repository, the oracle the real fix, the checks the real tests); every other state is the driver's.
"""

from __future__ import annotations

import ast
import asyncio
import contextlib
import difflib
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
import uuid
import warnings
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

IMAGE_MINE = "crucible-m17-exec:py312"  # the mining image (JUnit on a bind mount); grading uses IMAGE (the guard below)
IMAGE = "crucible-m17-exec:py312-g2"
# The result guard: after pytest has written its JUnit file (in the container's tmpfs, never a mount), it prints the
# report between markers on the real stdout and ends the process with os._exit, so atexit handlers a solution registered
# cannot rewrite the report afterwards; prints during collection and tests are captured by pytest and never reach stdout.
GUARD_PLUGIN = """import os
import sys

import pytest

_STATUS = {}


def pytest_sessionfinish(session, exitstatus):
    _STATUS["code"] = int(exitstatus)


@pytest.hookimpl(trylast=True)
def pytest_unconfigure(config):
    path = getattr(config.option, "xmlpath", None)
    data = b""
    if path:
        try:
            with open(path, "rb") as fh:
                data = fh.read()
        except OSError:
            data = b""
    for stream in (sys.stdout, sys.stderr):  # os._exit skips the interpreter's flush: pytest's failure report is still buffered
        try:
            stream.flush()
        except Exception:
            pass
    os.write(1, b"\\n@@M17-JUNIT-BEGIN@@\\n" + data + b"\\n@@M17-JUNIT-END@@\\n")
    os._exit(_STATUS.get("code", 3))
"""
DOCKERFILE = """FROM python:3.12-slim
RUN pip install --no-cache-dir pytest==8.3.3 freezegun==1.5.1 text-unidecode==1.3 Unidecode==1.3.8 wcwidth==0.2.13 pytest-timeout==2.3.1 hypothesis==6.112.1 typing_extensions
COPY m17guard.py /opt/m17/m17guard.py
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONHASHSEED=0
WORKDIR /w
"""
MINER = "miner:m17-executable-lane"
TEST_PATH = re.compile(r"(^|/)(tests?|testing)/|(^|/)test_[^/]*\.py$|_tests?\.py$|(^|/)conftest\.py$")
PROTECTED_NAMES = frozenset({"conftest.py", "sitecustomize.py", "usercustomize.py", "pytest.ini", "tox.ini", "setup.cfg", "pyproject.toml",
                             "noxfile.py", "setup.py"})
NOT_SOURCE = re.compile(r"(^|/)(docs?|examples?|benchmarks?|scripts?|tools|\.github)/|^(setup|noxfile|conftest)\.py$")
REGRESSION_REQ = "r_regress"
REGRESSION_TEXT = "Everything that already works today must keep working."


def is_test_path(path: str) -> bool:
    return bool(TEST_PATH.search(path))


def is_protected(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    return is_test_path(path) or name in PROTECTED_NAMES or path.endswith(".pth")


def _safe(path: str) -> str | None:
    """A repository-relative path, or None when it escapes the tree."""
    p = path.strip()
    while p.startswith("./"):
        p = p[2:]
    if not p or p.startswith("/") or "\\" in p or any(s in ("..", "") for s in p.split("/")):
        return None
    return p


def call_supported(fn: Any, *args: Any, **kw: Any) -> Any:
    """``fn(*args, **kw)`` without the keyword arguments ``fn`` cannot take (option B, 07 §3: a scripted client without
    ``**kwargs`` raised TypeError when a state retry added ``sample_key``, or the solver runner ``json_mode``). A callable
    with ``**kwargs`` (every ``LLMClient``) gets them all."""
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):
        return fn(*args, **kw)
    if any(p.kind is p.VAR_KEYWORD for p in params):
        return fn(*args, **kw)
    names = {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
    return fn(*args, **{k: v for k, v in kw.items() if k in names})


async def lane_ask(ctx: Any, role_brief: str, payload: Mapping[str, Any], *, max_tokens: int = 4000, sample: str = "") -> dict[str, Any]:
    """``driver._ask`` through ``call_supported`` (the lane's roles take any client the machine injects). ``sample`` makes
    a second ask inside one attempt a new sample (the compact retry below)."""
    from crucible.actors.llm import LLMOutputError
    from crucible.factory.driver import _jsonable

    if ctx.client is None:
        raise RuntimeError(f"state {ctx.state_id}: no client for role {ctx.role!r}")
    user = dict(payload)
    if ctx.findings:
        user["findings_to_address"] = ctx.findings
    msgs = [{"role": "system", "content": f"ROLE: {ctx.role}\n{role_brief}\nAnswer with one JSON object only."},
            {"role": "user", "content": json.dumps(_jsonable(user), sort_keys=True)}]
    kw: dict[str, Any] = {"sample_key": f"{ctx.state_id}#{ctx.entry}.{ctx.attempt}{sample}"} if ctx.attempt > 1 or sample else {}
    out = await call_supported(ctx.client.complete_json, msgs, max_tokens=max_tokens, temperature=0.0, **kw)
    if not isinstance(out, dict):
        raise LLMOutputError(f"{ctx.role}: expected a JSON object")
    return out


def truncated(e: BaseException) -> bool:
    """An ``LLMClient.complete_json`` failure caused by the token cap (cut-off JSON, or an empty reply at ``finish=length``
    after the reasoning spent the budget)."""
    s = str(e)
    return "cut off at max_tokens" in s or "finish=length" in s


def sha256_text(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", "surrogatepass"))
        h.update(b"\0")
    return h.hexdigest()


# ---- the sandbox -------------------------------------------------------------------------------------------------------
@dataclass
class TestRun:
    results: dict[str, str]  # test id -> passed | failed | error | skipped
    infra: str = "ok"  # ok | masked (the sandbox could not run: unmeasured)
    exit_code: int | None = None
    seconds: float = 0.0
    tail: str = ""
    why: str = ""


def parse_junit(path: Path) -> dict[str, str] | None:
    """Per-test status from a pytest xunit1 JUnit file, keyed by the pytest node id (``file::Class::name``)."""
    try:
        root = ET.parse(path).getroot()
    except (OSError, ET.ParseError):
        return None
    out: dict[str, str] = {}
    for tc in root.iter("testcase"):
        name, cls, file = tc.get("name") or "", tc.get("classname") or "", tc.get("file") or ""
        if not name:
            continue
        mod = file[:-3].replace("/", ".") if file.endswith(".py") else ""
        rest = cls[len(mod) + 1:] if mod and cls.startswith(mod + ".") else ("" if cls == mod else cls)
        if not file:  # a collection error: the classname is the module path
            file = cls.replace(".", "/") + ".py" if cls else "?"
        tid = "::".join(x for x in (file, rest.replace(".", "::"), name) if x)
        st = "passed"
        for child in tc:
            if child.tag in ("failure", "error"):
                st = "failed" if child.tag == "failure" else "error"
                break
            if child.tag == "skipped":
                st = "skipped"
        out[tid] = st
    return out


_BLOCK = re.compile(r"@@M17-JUNIT-BEGIN@@\n(.*?)\n@@M17-JUNIT-END@@", re.DOTALL)


def junit_from_stdout(text: str) -> dict[str, str] | None:
    """The guard's report: the last block on stdout (a solution cannot print after the guard's ``os._exit``)."""
    blocks = _BLOCK.findall(text)
    if not blocks or not blocks[-1].strip():
        return None
    with tempfile.NamedTemporaryFile("w", suffix=".xml", delete=False) as fh:
        fh.write(blocks[-1])
    try:
        return parse_junit(Path(fh.name))
    finally:
        os.unlink(fh.name)


class DockerSandbox:
    """pytest in a throwaway container: no network, read-only tree and root, tmpfs /tmp, CPU and memory capped.

    ``guard=True`` (grading): the JUnit file lives in the container's tmpfs at a per-run random path and reaches the host
    only through the guard plugin's stdout block. ``guard=False`` (the mining image): JUnit on a per-run bind mount."""

    def __init__(self, image: str | None = None, *, guard: bool = True, timeout_s: float = 300.0, per_test_s: int = 60, cpus: str = "1",
                 memory: str = "1g", parallel: int = 6, extra_args: Sequence[str] = ()):
        self.guard = guard
        self.extra_args = tuple(extra_args)  # more pytest options (the window lane: --continue-on-collection-errors)
        self.image = image or (IMAGE if guard else IMAGE_MINE)
        self.timeout_s, self.per_test_s, self.cpus, self.memory, self.parallel = timeout_s, per_test_s, cpus, memory, parallel
        self._sem: asyncio.Semaphore | None = None
        self.runs = 0

    def describe(self) -> dict[str, Any]:
        return {"kind": "docker", "image": self.image, "guard": self.guard, "network": "none", "read_only": True, "cpus": self.cpus,
                "memory": self.memory, "per_test_timeout_s": self.per_test_s, "timeout_s": self.timeout_s}

    def image_digest(self) -> str:
        r = subprocess.run(["docker", "image", "inspect", "--format", "{{.Id}}", self.image], capture_output=True, text=True, check=False)
        return r.stdout.strip() if r.returncode == 0 else f"unresolved:{self.image}"

    def ensure_image(self) -> str:
        d = self.image_digest()
        if d.startswith("unresolved:"):
            with tempfile.TemporaryDirectory() as td:
                (Path(td) / "Dockerfile").write_text(DOCKERFILE)
                (Path(td) / "m17guard.py").write_text(GUARD_PLUGIN)
                subprocess.run(["docker", "build", "-q", "-t", self.image, td], check=True, capture_output=True)
            d = self.image_digest()
        return d

    def command(self, tree: Path, name: str, nonce: str, targets: Sequence[str], out: Path | None = None) -> list[str]:
        base = ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only", "--tmpfs", "/tmp:rw,size=256m",
                "--memory", self.memory, "--cpus", self.cpus, "--pids-limit", "256", "-v", f"{tree}:/w:ro"]
        if self.guard:
            junit = f"/tmp/j{nonce}/r.xml"
            return [*base, "-w", "/w", "-e", "PYTHONPATH=/w/src:/w:/opt/m17", "-e", "HOME=/tmp", self.image,
                    "python", "-m", "pytest", "-p", "m17guard", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-o", "junit_family=xunit1",
                    f"--timeout={self.per_test_s}", f"--junitxml={junit}", "-rN", "--color=no", *self.extra_args, *targets]
        mount = f"/o_{nonce}"
        return [*base, "-v", f"{out}:{mount}:rw", "-w", "/w", "-e", "PYTHONPATH=/w/src:/w", "-e", "HOME=/tmp", self.image,
                "python", "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", "-o", "junit_family=xunit1",
                f"--timeout={self.per_test_s}", f"--junitxml={mount}/r.xml", "-rN", "--color=no", *self.extra_args, *targets]

    async def run(self, tree: Path, targets: Sequence[str]) -> TestRun:
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.parallel)
        nonce = uuid.uuid4().hex[:12]
        name = f"crucible-m17-{nonce}"
        out = None if self.guard else Path(tempfile.mkdtemp(prefix="m17o-"))
        try:
            async with self._sem:
                self.runs += 1
                t0 = time.monotonic()
                cmd = self.command(Path(tree).resolve(), name, nonce, targets, out.resolve() if out else None)
                proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
                try:
                    raw, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
                except TimeoutError:
                    killer = await asyncio.create_subprocess_exec("docker", "kill", name, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL)
                    await killer.wait()
                    with contextlib.suppress(Exception):
                        await asyncio.wait_for(proc.wait(), 30)
                    # the run did not finish inside the budget: the solution's (an import-time hang), not the sandbox's
                    return TestRun({}, "ok", None, time.monotonic() - t0, "", why=f"timeout after {self.timeout_s}s")
                secs = time.monotonic() - t0
            text = raw.decode("utf-8", "replace")
            code = proc.returncode
            res = junit_from_stdout(text) if self.guard else parse_junit(out / "r.xml")  # type: ignore[operator]
            if code in (125, 126, 127) or (res is None and "docker:" in text[:400].lower()):
                return TestRun({}, "masked", code, secs, text[-3000:], why=f"sandbox did not start (exit {code})")
            tail = _BLOCK.sub("", text)
            return TestRun(res or {}, "ok", code, secs, tail[-4000:], why="" if res is not None else "no JUnit report (the run ended early)")
        finally:
            if out is not None:
                shutil.rmtree(out, ignore_errors=True)


class LocalSandbox(DockerSandbox):
    """The same command on the host's interpreter (unit tests only: no isolation)."""

    def __init__(self, python: str | None = None, **kw: Any):
        super().__init__(**kw)
        import sys

        self.python = python or sys.executable

    def describe(self) -> dict[str, Any]:
        return {"kind": "local", "python": self.python}

    def image_digest(self) -> str:
        return "sha256:" + "0" * 64

    def ensure_image(self) -> str:
        return self.image_digest()

    async def run(self, tree: Path, targets: Sequence[str]) -> TestRun:
        out = Path(tempfile.mkdtemp(prefix="m17o-"))
        try:
            t0 = time.monotonic()
            env = {**os.environ, "PYTHONPATH": f"{tree}/src:{tree}", "PYTHONDONTWRITEBYTECODE": "1"}
            proc = await asyncio.create_subprocess_exec(self.python, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
                                                        "-o", "junit_family=xunit1", f"--junitxml={out}/r.xml", "-rN", *self.extra_args, *targets, cwd=str(tree),
                                                        env=env, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT)
            raw, _ = await asyncio.wait_for(proc.communicate(), self.timeout_s)
            res = parse_junit(out / "r.xml")
            return TestRun(res or {}, "ok", proc.returncode, time.monotonic() - t0, raw.decode("utf-8", "replace")[-4000:],
                           why="" if res is not None else "no JUnit report")
        finally:
            shutil.rmtree(out, ignore_errors=True)


# ---- solutions: search/replace edits or whole files --------------------------------------------------------------------
def normalize_solution(solution: Any) -> dict[str, Any]:
    """``{"edits": [{path, old, new}], "files": {path: text | None}}`` from what roles and solvers return (a bare edit
    list, a dict with either key, None)."""
    if solution is None:
        return {"edits": [], "files": {}}
    if isinstance(solution, list):
        return {"edits": [e for e in solution if isinstance(e, Mapping)], "files": {}}
    if isinstance(solution, Mapping):
        edits = solution.get("edits") or []
        files = solution.get("files") or {}
        return {"edits": [dict(e) for e in edits if isinstance(e, Mapping)] if isinstance(edits, list) else [],
                "files": {str(k): (None if v is None else str(v)) for k, v in files.items()} if isinstance(files, Mapping) else {}}
    return {"edits": [], "files": {}, "malformed": f"a solution is an object with edits/files, got {type(solution).__name__}"}


@dataclass
class Applied:
    written: dict[str, str | None] = field(default_factory=dict)  # path -> final text (None: deleted), non-protected only
    protected: list[str] = field(default_factory=list)  # protected paths the solution tried to write (dropped)
    errors: list[str] = field(default_factory=list)  # edits that did not apply (the solution is malformed)


def apply_solution(root: Path, solution: Any) -> Applied:
    """Apply ``solution`` to the tree at ``root`` in place. Protected paths are never written."""
    sol = normalize_solution(solution)
    out = Applied()
    if sol.get("malformed"):
        out.errors.append(sol["malformed"])
        return out
    for raw, text in sol["files"].items():
        p = _safe(raw)
        if p is None:
            out.errors.append(f"unsafe path {raw!r}")
            continue
        if is_protected(p):
            out.protected.append(p)
            continue
        f = root / p
        if text is None:
            if f.exists():
                f.unlink()
            out.written[p] = None
        else:
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(text)
            out.written[p] = text
    for i, e in enumerate(sol["edits"]):
        p = _safe(str(e.get("path") or ""))
        old, new = e.get("old"), e.get("new")
        if p is None or not isinstance(old, str) or not isinstance(new, str):
            out.errors.append(f"edit {i}: needs path, old and new strings")
            continue
        if is_protected(p):
            out.protected.append(p)
            continue
        f = root / p
        text = f.read_text() if f.exists() else None
        if old == "":
            if text not in (None, ""):
                out.errors.append(f"edit {i}: empty 'old' on the existing file {p}")
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            f.write_text(new)
            out.written[p] = new
            continue
        if text is None:
            out.errors.append(f"edit {i}: {p} does not exist")
            continue
        n = text.count(old)
        if n != 1:
            out.errors.append(f"edit {i}: the 'old' text occurs {n} times in {p} (needs exactly once)")
            continue
        text = text.replace(old, new, 1)
        f.write_text(text)
        out.written[p] = text
    out.protected = sorted(set(out.protected))
    return out


def edits_from_change(path: str, before: str, after: str, *, context: int = 2) -> list[dict[str, str]] | None:
    """Search/replace edits turning ``before`` into ``after``, each ``old`` unique in ``before``; None when that is not
    possible (the caller falls back to a whole-file write)."""
    a, b = before.splitlines(keepends=True), after.splitlines(keepends=True)
    for n in (context, 4, 8, 16):
        edits, ok = [], True
        for group in difflib.SequenceMatcher(None, a, b, autojunk=False).get_grouped_opcodes(n):
            i1, i2, j1, j2 = group[0][1], group[-1][2], group[0][3], group[-1][4]
            old, new = "".join(a[i1:i2]), "".join(b[j1:j2])
            if not old or before.count(old) != 1:
                ok = False
                break
            edits.append({"path": path, "old": old, "new": new})
        if ok and edits:
            return edits
    return None


def oracle_solution(base: Mapping[str, str], fixed: Mapping[str, str | None]) -> dict[str, Any]:
    """The real fix as a solution: edits where they are unique, whole files otherwise."""
    edits: list[dict[str, str]] = []
    files: dict[str, str | None] = {}
    for p in sorted(fixed):
        after = fixed[p]
        before = base.get(p)
        if after is None or before is None:
            files[p] = after
            continue
        e = edits_from_change(p, before, after)
        if e is None:
            files[p] = after
        else:
            edits += e
    out: dict[str, Any] = {"edits": edits}
    if files:
        out["files"] = files
    return out


def normalized(path: str, text: str | None) -> str:
    """What "the same end state" means for a file: the AST of Python source (formatting and comments ignored), the text
    otherwise."""
    if text is None:
        return "<deleted>"
    if path.endswith(".py"):
        try:
            return ast.dump(ast.parse(text), include_attributes=False)
        except (SyntaxError, ValueError):
            return "<unparsable>" + text
    return text


def fingerprint(base: Mapping[str, str], written: Mapping[str, str | None]) -> str:
    """The end-state fingerprint: the normalized content of every file whose normalized content differs from the start."""
    changed = {p: normalized(p, t) for p, t in written.items() if normalized(p, t) != normalized(p, base.get(p))}
    return sha256_text(json.dumps(changed, sort_keys=True))


# ---- git ---------------------------------------------------------------------------------------------------------------
def git(repo: Path, *args: str, check: bool = True) -> str:
    r = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=False)
    if check and r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr.strip()[:300]}")
    return r.stdout


def show(repo: Path, rev: str, path: str) -> str | None:
    r = subprocess.run(["git", "-C", str(repo), "show", f"{rev}:{path}"], capture_output=True, check=False)
    if r.returncode != 0:
        return None
    try:
        return r.stdout.decode("utf-8")
    except UnicodeDecodeError:
        return None


def export_tree(repo: Path, rev: str, dest: Path) -> None:
    """The tree at ``rev`` without history (no ``.git``: the solver cannot read the fix from the log)."""
    dest.mkdir(parents=True, exist_ok=True)
    arch = subprocess.run(["git", "-C", str(repo), "archive", "--format=tar", rev], capture_output=True, check=True)
    subprocess.run(["tar", "-x", "-C", str(dest)], input=arch.stdout, check=True)
    prepare_tree(dest)


_VERSION_FILE = re.compile(r"""^\s*(?:[\w.-]+\.)?(?:version[-_]file|write_to)\s*=\s*["']([^"']+\.py)["']""", re.MULTILINE)


def prepare_tree(root: Path) -> list[str]:
    """What an install would generate and a source checkout lacks: the version module of hatch-vcs / setuptools-scm
    (``version-file`` / ``write_to`` in ``pyproject.toml``). Returns the files written."""
    made = []
    pp = root / "pyproject.toml"
    if pp.exists():
        for rel in _VERSION_FILE.findall(pp.read_text(errors="replace")):
            p = _safe(rel)
            if p and not (root / p).exists():
                (root / p).parent.mkdir(parents=True, exist_ok=True)
                (root / p).write_text('__version__ = version = "0.0.0"\n__version_tuple__ = version_tuple = (0, 0, 0)\n')
                made.append(p)
    return made


@dataclass
class Candidate:
    repo: str
    repo_dir: str
    sha: str
    parent: str
    date: str
    subject: str
    body: str
    src_files: list[str]
    test_files: list[str]
    src_lines: int
    test_lines: int


def candidates(repo_dir: Path, *, repo: str | None = None, since: str = "2018-01-01", max_src_files: int = 2, max_src_lines: int = 60,
               max_test_lines: int = 300, limit: int | None = None) -> list[Candidate]:
    """Non-merge commits that change 1..``max_src_files`` existing source files (a small diff) and at least one Python
    test file. Newest first."""
    repo_dir = Path(repo_dir)
    name = repo or repo_dir.name
    log = git(repo_dir, "log", "--no-merges", f"--since={since}", "--format=@@%H%x09%P%x09%aI%x09%s", "--numstat")
    out: list[Candidate] = []
    cur: dict[str, Any] | None = None

    def flush() -> None:
        if cur is None:
            return
        src = [p for p in cur["files"] if p.endswith(".py") and not is_test_path(p) and not NOT_SOURCE.search(p)]
        tests = [p for p in cur["files"] if p.endswith(".py") and is_test_path(p) and not p.endswith("conftest.py")]
        other = [p for p in cur["files"] if p not in src and p not in tests]
        if len(cur["parents"]) != 1 or not tests or not (1 <= len(src) <= max_src_files):
            return
        if any(p.endswith(".py") and not is_test_path(p) for p in other if not NOT_SOURCE.search(p)):
            return
        s_lines = sum(cur["files"][p] for p in src)
        t_lines = sum(cur["files"][p] for p in tests)
        if s_lines > max_src_lines or t_lines > max_test_lines or s_lines == 0:
            return
        out.append(Candidate(name, str(repo_dir), cur["sha"], cur["parents"][0], cur["date"], cur["subject"], "", src, tests, s_lines, t_lines))

    for line in log.splitlines():
        if line.startswith("@@"):
            flush()
            sha, parents, date, subject = (line[2:].split("\t", 3) + ["", "", ""])[:4]
            cur = {"sha": sha, "parents": parents.split(), "date": date, "subject": subject, "files": {}}
        elif line.strip() and cur is not None:
            parts = line.split("\t")
            if len(parts) == 3:
                add, dele, path = parts
                cur["files"][path] = (int(add) if add.isdigit() else 0) + (int(dele) if dele.isdigit() else 0)
    flush()
    keep = []
    for c in out:
        if all(show(repo_dir, c.parent, p) is not None for p in c.src_files):  # an existing file is fixed, not a new module
            c.body = git(repo_dir, "log", "-1", "--format=%b", c.sha).strip()
            keep.append(c)
        if limit and len(keep) >= limit:
            break
    return keep


# ---- mined tasks -------------------------------------------------------------------------------------------------------
@dataclass
class MinedTask:
    id: str
    repo: str
    sha: str
    parent: str
    date: str
    subject: str
    body: str
    src_files: list[str]
    test_files: list[str]  # the commit's test files that exist after the fix (the ground-truth checks run these)
    tests: dict[str, str]  # test file -> content at the fix
    base_src: dict[str, str]  # source file -> content at the parent
    fix_src: dict[str, str | None]  # source file -> content at the fix
    oracle: dict[str, Any]  # the fix as a solution
    f2p: list[str]
    p2p: list[str]
    excluded: dict[str, list[str]]  # flaky / failing-after-fix tests, never checks
    observation: dict[str, Any]  # the real_pr_tests ObservationRecord (JSON)
    validation: dict[str, Any]  # rerun evidence
    base_dir: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: Mapping[str, Any]) -> MinedTask:
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def task_id_of(repo: str, sha: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', repo)}-{sha[:10]}"


def observation_for(c: Candidate, tests: Mapping[str, str]) -> dict[str, Any]:
    from crucible.factory.cards import ObservationRecord

    sha = sha256_text(c.sha, *[f"{p}\n{tests[p]}" for p in sorted(tests)])
    created = c.date if c.date else "1970-01-01T00:00:00+00:00"
    rec = ObservationRecord(id=f"obs-{task_id_of(c.repo, c.sha)}", type="real_pr_tests",
                            pointer=f"git:{c.repo}@{c.sha}:{','.join(sorted(tests))}", sha256=sha, created=created, by=MINER)
    return rec.model_dump(mode="json")


class LaneStore:
    """Mined tasks on disk: ``<root>/tasks/<id>/record.json`` and the start tree ``<root>/tasks/<id>/base/``."""

    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "tasks").mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, MinedTask] = {}

    def dir(self, tid: str) -> Path:
        return self.root / "tasks" / tid

    def save(self, t: MinedTask) -> None:
        d = self.dir(t.id)
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "record.json.tmp"
        tmp.write_text(json.dumps(t.to_json(), indent=1, sort_keys=True))
        tmp.replace(d / "record.json")
        self._cache[t.id] = t

    def get(self, tid: str) -> MinedTask:
        if tid not in self._cache:
            t = MinedTask.from_json(json.loads((self.dir(tid) / "record.json").read_text()))
            t.base_dir = str(self.dir(tid) / "base")
            self._cache[tid] = t
        return self._cache[tid]

    def ids(self) -> list[str]:
        return sorted(p.parent.name for p in (self.root / "tasks").glob("*/record.json"))

    def all(self) -> list[MinedTask]:
        return [self.get(i) for i in self.ids()]


def build_tree(base_dir: Path, dest: Path, *, solution: Any = None, tests: Mapping[str, str] | None = None) -> Applied:
    """A fresh copy of the start tree, the solution applied, then the ground-truth test files written over it."""
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base_dir, dest, symlinks=True)
    applied = apply_solution(dest, solution) if solution is not None else Applied()
    for p, text in (tests or {}).items():
        f = dest / p
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(text)
    return applied


async def validate_candidate(c: Candidate, store: LaneStore, sandbox: DockerSandbox, *, reruns: int = 3) -> tuple[MinedTask | None, str, dict]:
    """SWE-smith's pre-gold / post-gold validation with reruns. Returns the task (or None) and the reason."""
    tid = task_id_of(c.repo, c.sha)
    repo_dir = Path(c.repo_dir)
    tests = {p: t for p in c.test_files if (t := show(repo_dir, c.sha, p)) is not None}
    if not tests:
        return None, "no test file survives the fix", {}
    base_src = {p: show(repo_dir, c.parent, p) or "" for p in c.src_files}
    fix_src = {p: show(repo_dir, c.sha, p) for p in c.src_files}
    d = store.dir(tid)
    base = d / "base"
    if not base.exists():
        export_tree(repo_dir, c.parent, base)
    oracle = oracle_solution(base_src, fix_src)
    work = Path(tempfile.mkdtemp(prefix="m17v-"))
    try:
        pre_t, post_t = work / "pre", work / "post"
        build_tree(base, pre_t, tests=tests)
        ap = build_tree(base, post_t, solution=oracle, tests=tests)
        if ap.errors or any((post_t / p).read_text() != (fix_src[p] or "") for p in c.src_files if fix_src[p] is not None):
            return None, f"the oracle edits do not reproduce the fix: {ap.errors[:2]}", {}
        targets = sorted(tests)
        pre = await asyncio.gather(*[sandbox.run(pre_t, targets) for _ in range(reruns)])
        post = await asyncio.gather(*[sandbox.run(post_t, targets) for _ in range(reruns)])
    finally:
        shutil.rmtree(work, ignore_errors=True)
    ev = {"reruns": reruns, "pre_seconds": [round(r.seconds, 2) for r in pre], "post_seconds": [round(r.seconds, 2) for r in post],
          "pre_counts": [len(r.results) for r in pre], "post_counts": [len(r.results) for r in post]}
    if any(r.infra != "ok" for r in pre + post):
        return None, "infra: the sandbox did not run", ev
    if any(not r.results for r in post):
        return None, "no test result after the fix (collection failed: environment mismatch)", {**ev, "post_tail": post[0].tail[-800:]}
    ids = sorted({t for r in pre + post for t in r.results})

    def statuses(runs: Sequence[TestRun], t: str) -> set[str]:
        return {r.results.get(t, "missing") for r in runs}

    flaky = [t for t in ids if len(statuses(pre, t)) > 1 or len(statuses(post, t)) > 1]
    stable = [t for t in ids if t not in flaky]
    post_pass = {t for t in stable if statuses(post, t) == {"passed"}}
    pre_pass = {t for t in stable if statuses(pre, t) == {"passed"}}
    f2p = sorted(post_pass - pre_pass)
    p2p = sorted(post_pass & pre_pass)
    p2f = sorted(pre_pass - post_pass)
    post_fail = sorted(t for t in stable if t not in post_pass)
    ev.update({"f2p": len(f2p), "p2p": len(p2p), "p2f": len(p2f), "flaky": len(flaky), "post_fail": len(post_fail)})
    if p2f:
        return None, f"PASS_TO_FAIL: the fix breaks {len(p2f)} test(s)", ev
    if not f2p:
        return None, "0_f2p: no test fails before the fix and passes after it", ev
    if post_fail and len(post_fail) > len(post_pass):
        return None, f"{len(post_fail)} tests fail after the fix (environment mismatch)", ev
    t = MinedTask(id=tid, repo=c.repo, sha=c.sha, parent=c.parent, date=c.date, subject=c.subject, body=c.body, src_files=list(c.src_files),
                  test_files=sorted(tests), tests=tests, base_src=base_src, fix_src=fix_src, oracle=oracle, f2p=f2p, p2p=p2p,
                  excluded={"flaky": flaky, "post_fail": post_fail}, observation=observation_for(c, tests), validation=ev)
    store.save(t)
    t.base_dir = str(base)
    return t, "1+_f2p", ev


async def mine(repos: Mapping[str, Path], store: LaneStore, sandbox: DockerSandbox, *, per_repo: int = 40, reruns: int = 3,
               since: str = "2018-01-01", log: Any = None) -> dict[str, Any]:
    """Candidates per repository, validated in the sandbox. Returns the funnel with a reason for every drop."""
    funnel: dict[str, Any] = {}
    for name, path in repos.items():
        cands = candidates(Path(path), repo=name, since=since, limit=per_repo)
        rows = await asyncio.gather(*[validate_candidate(c, store, sandbox, reruns=reruns) for c in cands])
        reasons: dict[str, int] = {}
        kept = []
        for c, (t, why, ev) in zip(cands, rows):
            key = why.split(":")[0]
            reasons[key] = reasons.get(key, 0) + 1
            if t is not None:
                kept.append(t.id)
            if log:
                log({"event": "validated", "repo": name, "sha": c.sha, "subject": c.subject, "result": why, **ev})
            if t is None:
                shutil.rmtree(store.dir(task_id_of(c.repo, c.sha)), ignore_errors=True)
        funnel[name] = {"candidates": len(cands), "validated": len(kept), "reasons": reasons, "tasks": kept}
    return funnel


# ---- test sources and solution helpers ---------------------------------------------------------------------------------
def test_source(t: MinedTask, test_id: str, *, max_chars: int = 2500) -> str:
    """The source of one test function (decorators included: parametrize cases are part of what it checks)."""
    parts = test_id.split("::")
    file, names = parts[0], [re.sub(r"\[.*$", "", n) for n in parts[1:]]
    text = t.tests.get(file)
    if text is None or not names:
        return ""
    try:
        with warnings.catch_warnings():  # old test files carry invalid escape sequences
            warnings.simplefilter("ignore", SyntaxWarning)
            tree = ast.parse(text)
    except SyntaxError:
        return ""
    body: list[ast.stmt] = tree.body
    node: Any = None
    for n in names:
        node = next((x for x in body if isinstance(x, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and x.name == n), None)
        if node is None:
            return ""
        body = getattr(node, "body", [])
    lines = text.splitlines()
    start = min([d.lineno for d in getattr(node, "decorator_list", [])] + [node.lineno]) - 1
    return "\n".join(lines[start:node.end_lineno])[:max_chars]


def apply_in_memory(base: Mapping[str, str], solution: Any) -> tuple[dict[str, str | None], list[str]]:
    """``apply_solution`` on a dict of file texts (no tree): the written files and the errors."""
    sol = normalize_solution(solution)
    files: dict[str, str | None] = dict(base)
    written: dict[str, str | None] = {}
    errors = [sol["malformed"]] if sol.get("malformed") else []
    for p, text in sol["files"].items():
        if _safe(p) and not is_protected(p):
            files[p] = written[p] = text
    for i, e in enumerate(sol["edits"]):
        p, old, new = _safe(str(e.get("path") or "")), e.get("old"), e.get("new")
        if p is None or not isinstance(old, str) or not isinstance(new, str) or is_protected(p):
            continue
        cur = files.get(p)
        if old == "" and not cur:
            files[p] = written[p] = new
        elif cur is None or cur.count(old) != 1:
            errors.append(f"edit {i} does not apply to {p}")
        else:
            files[p] = written[p] = cur.replace(old, new, 1)
    return written, errors


def reformatted_fix(t: MinedTask) -> dict[str, Any] | None:
    """A valid alternative to the oracle: every fixed source file re-rendered from its AST (comments and layout
    change, behaviour does not). None when a file cannot be re-rendered."""
    files: dict[str, str | None] = {}
    for p, text in t.fix_src.items():
        if text is None or not p.endswith(".py"):
            return None
        try:
            files[p] = ast.unparse(ast.parse(text)) + "\n"
        except (SyntaxError, ValueError, RecursionError):
            return None
    return {"files": files}


def _unique_line(text: str) -> str | None:
    for ln in text.splitlines(keepends=True):
        if ln.strip() and text.count(ln) == 1:
            return ln
    return None


def regression_pool(t: MinedTask, *, cap: int = 200) -> list[str]:
    """PASS_TO_PASS tests that guard against collateral damage: the ones next to the new tests (same class, or same file
    for module-level tests), at most ``cap`` (evenly spaced)."""
    near = {x.rsplit("::", 1)[0] for x in t.f2p}
    pool = [x for x in t.p2p if x.rsplit("::", 1)[0] in near] or list(t.p2p)
    if len(pool) > cap:
        step = len(pool) / cap
        pool = [pool[int(i * step)] for i in range(cap)]
    return pool


def scrub(text: str) -> str:
    from crucible.world.leaks import scrub_text

    return scrub_text(text or "", mode="strict").strip()


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", text)}


def source_files(base_dir: Path) -> dict[str, str]:
    out = {}
    for f in sorted(Path(base_dir).rglob("*.py")):
        rel = f.relative_to(base_dir).as_posix()
        if is_test_path(rel) or NOT_SOURCE.search(rel):
            continue
        try:
            out[rel] = f.read_text()
        except (OSError, UnicodeDecodeError):
            continue
    return out


def _excerpt(path: str, text: str, want: set[str], budget: int) -> str:
    """A large file cut to its imports and the top-level definitions the request mentions (or whose body does)."""
    try:
        tree = ast.parse(text)
    except SyntaxError:
        return text[:budget]
    lines = text.splitlines()
    keep: list[tuple[int, int]] = []
    named, mentioned = [], []
    for n in tree.body:
        start = min([d.lineno for d in getattr(n, "decorator_list", [])] + [n.lineno]) - 1
        seg = (start, n.end_lineno or n.lineno)
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            keep.append(seg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if n.name.lower() in want:
                named.append(seg)
            elif _tokens("\n".join(lines[seg[0]:seg[1]])) & want:
                mentioned.append(seg)
    out, used = [], 0
    for a, b in keep + named + mentioned:
        chunk = "\n".join(lines[a:b])
        if used + len(chunk) > budget and (a, b) not in keep:
            continue
        out.append((a, chunk))
        used += len(chunk)
    return "\n\n# ...\n\n".join(c for _, c in sorted(out))


def solver_context(base_dir: Path, prompt: str, *, budget: int = 60_000) -> tuple[str, list[str]]:
    """What the solver sees besides the request: the source tree listing and the source files most related to the
    request (identifier overlap weighted by rarity; never the fix or the new tests)."""
    files = source_files(base_dir)
    want = _tokens(prompt)
    df: dict[str, int] = {}
    toks = {p: _tokens(t) for p, t in files.items()}
    for ts in toks.values():
        for w in ts:
            df[w] = df.get(w, 0) + 1
    import math

    score = {p: sum(1.0 / math.log(2 + df[w]) for w in want & toks[p]) + (3.0 if Path(p).stem.lower() in want else 0.0) for p in files}
    ranked = sorted(files, key=lambda p: (-score[p], p))
    shown, used, parts = [], 0, []
    for p in ranked:
        if score[p] <= 0 and shown:
            break
        text = files[p]
        if len(text) > budget // 2:
            text = _excerpt(p, text, want, budget // 2)
            head = f"=== {p} (excerpt: the definitions related to the request) ==="
        else:
            head = f"=== {p} ==="
        if used + len(text) > budget and shown:
            continue
        parts.append(f"{head}\n{text}")
        shown.append(p)
        used += len(text)
    listing = "\n".join(sorted(f.relative_to(base_dir).as_posix() for f in Path(base_dir).rglob("*") if f.is_file()
                               and not any(s.startswith(".") for s in f.relative_to(base_dir).parts))[:400])
    return f"REPOSITORY FILES:\n{listing}\n\nSOURCE FILES:\n" + "\n\n".join(parts), shown


SOLVER_SYSTEM = """You are a software engineer working in a Python repository. Make the change the user asks for by editing the repository's source files.
Reply with one JSON object only, no prose: {"edits": [{"path": "<file path>", "old": "<exact text copied from the file>", "new": "<replacement text>"}]}
Rules: each "old" is copied exactly from the file shown (indentation included) and occurs exactly once in that file; keep each "old" short (the few lines you change plus one line of context); to add code, include a nearby existing line in "old" and repeat it in "new". Do not edit tests."""


def parse_edits(text: str) -> tuple[dict[str, Any] | None, str]:
    t = text.strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\n?|```\s*$", "", t).strip()
    for cand in (t, t[t.find("{"): t.rfind("}") + 1] if "{" in t else ""):
        try:
            d = json.loads(cand)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(d, dict) and isinstance(d.get("edits"), list):
            return {"edits": [e for e in d["edits"] if isinstance(e, dict)]}, ""
    return None, "the reply is not a JSON object with an edits list"


# ---- the accept-contract task ------------------------------------------------------------------------------------------
LANE_EXCLUDED_CHECKS = {"A11": "harness hygiene injects B.5 failures into the M10 tool-call runtime (NativeHarness + scripted tool "
                               "calls); executable-lane tasks are graded patches with no tool-call oracle, and their solver "
                               "path masks provider and sandbox failures itself (TestRun.infra, traces with passed=None)"}


def lane_policy() -> Any:
    from crucible.accept.policy import ALL_CHECKS, Policy

    return Policy(name="accept-v1-exec", required=[c for c in ALL_CHECKS if c not in LANE_EXCLUDED_CHECKS])


class ExecTask:
    """One mined task under one prompt and one check set: the accept contract and ``driver.LANE_TASK``."""

    attack_surface = "agent"
    split = "train"
    horizon = "H1"
    trap = None

    def __init__(self, lane: ExecLane, mined: MinedTask, *, task_id: str, prompt: str, requirements: Sequence[Mapping[str, Any]],
                 spans: Mapping[str, Sequence[int]], checkset: Mapping[str, Any]):
        from crucible.forge.schema import CheckSpec, StatedProperty

        self.lane, self.mined, self.task_id, self.instruction = lane, mined, task_id, prompt
        self.base_dir = Path(mined.base_dir)
        self.requirements = [dict(r) for r in requirements]
        self.checkset = {"checks": [dict(c) for c in (checkset.get("checks") or [])]}
        self.check_tests = {c["id"]: [str(x) for x in ((c.get("spec") or {}).get("tests") or [])] for c in self.checkset["checks"]}
        self.gating = {c["id"]: bool(c.get("gating", True)) for c in self.checkset["checks"]}
        self.oracle = mined.oracle
        self.provenance = [mined.observation["id"]]
        self.properties = []
        for r in self.requirements:
            a, b = (list(spans.get(r["id"]) or [0, 0]) + [0, 0])[:2]
            text = prompt[a:b].strip() or r["text"]
            self.properties.append(StatedProperty(id=r["id"], text=text, kind=str(r.get("kind") or "behaviour")))
        self.checks = [CheckSpec(id=cid, gate=cid, gate_kind="outcome", kind="suite", states=list(c.get("requirement_ids") or []),
                                 provenance=f"run:mine/{mined.id}", spec={"tests": self.check_tests[cid]})
                       for cid, c in ((c["id"], c) for c in self.checkset["checks"]) if self.gating[cid]]
        self.task_hash = sha256_text(json.dumps([mined.id, prompt, self.checkset, mined.oracle], sort_keys=True, default=str))[:24]
        self.verifier_timeout_s = lane.sandbox.timeout_s
        self._src_base = {p: (self.base_dir / p).read_text() for p in mined.src_files if (self.base_dir / p).exists()}
        self.src0: str | None = min(self._src_base) if self._src_base else None
        self.negatives = self._negatives()

    # -- grading ---------------------------------------------------------------------------------------------------
    def _before(self, written: Mapping[str, Any]) -> dict[str, str]:
        out = {}
        for p in written:
            f = self.base_dir / p
            if f.is_file():
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    out[p] = f.read_text()
        return out

    def end_fingerprint(self, solution: Any) -> str:
        sol = normalize_solution(solution)
        paths = set(sol["files"]) | {str(e.get("path") or "") for e in sol["edits"]}
        base = self._before({p: None for p in paths if _safe(p)})
        written, _ = apply_in_memory(base, solution)
        return fingerprint(base, written)

    async def grade(self, solution: Any, *, run: int = 0, checks: Mapping[str, Sequence[str]] | None = None) -> tuple[Any, str]:
        from crucible.verify.runtime.plan import CheckResult, GradeReport

        check_tests = dict(checks) if checks is not None else self.check_tests
        t0 = time.monotonic()
        work = Path(tempfile.mkdtemp(prefix="m17g-"))
        why, tail, statuses = "", "", {}
        try:
            tree = work / "t"
            applied = build_tree(self.base_dir, tree, solution=solution, tests=self.mined.tests)
            fp = fingerprint(self._before(applied.written), applied.written)
            intact = all((tree / p).read_text() == text for p, text in self.mined.tests.items())
            if applied.errors:
                why = "the solution's edits did not apply: " + "; ".join(applied.errors[:3])
            else:
                targets = sorted({x for ts in check_tests.values() for x in ts})
                tr = await self.lane.sandbox.run(tree, targets)
                if tr.infra != "ok":
                    rep = GradeReport(task=self.task_id, tier="full", checks=[], gates={c: "infra" for c in check_tests}, reward=0.0, outcome=0.0,
                                      infra={"status": "masked", "why": tr.why}, trusted=False, elapsed_s=time.monotonic() - t0,
                                      meta={"why": tr.why, "tail": tr.tail[-1500:]})
                    return rep, fp
                statuses, why, tail = tr.results, tr.why, tr.tail
        finally:
            shutil.rmtree(work, ignore_errors=True)
        gates, results, failing_tests = {}, [], []
        for cid, tests in check_tests.items():
            bad = [x for x in tests if statuses.get(x) != "passed"]
            failing_tests += bad
            st = "pass" if tests and not bad else "fail"
            gates[cid] = st
            results.append(CheckResult(id=cid, gate=cid, status=st, detail={"failing": bad[:20], "tests": len(tests)}))
        gating = [c for c in check_tests if self.gating.get(c, True)]
        reward = 1.0 if gating and all(gates[c] == "pass" for c in gating) else 0.0
        meta = {"failing_tests": sorted(set(failing_tests))[:60], "statuses": {s: sum(1 for v in statuses.values() if v == s) for s in set(statuses.values())},
                "tail": tail[-2500:], "apply_errors": applied.errors, "protected_writes": applied.protected, "tests_intact": intact, "why": why,
                "kinds": {c: "outcome" for c in check_tests}}
        self.lane.note_grade(self.task_id, solution, meta)
        return GradeReport(task=self.task_id, tier="full", checks=results, gates=gates, reward=reward, outcome=reward, infra={"status": "ok"},
                           trusted=True, elapsed_s=time.monotonic() - t0, meta=meta), fp

    async def start_fingerprint(self) -> str:
        return fingerprint({}, {})

    def noop(self) -> Any:
        return {"edits": []}

    # -- negatives, mutants, floors ----------------------------------------------------------------------------------
    def _negatives(self) -> list[Any]:
        from crucible.forge.schema import NegativeSpec

        out = []
        src0 = self.src0
        if src0:
            ln = _unique_line(self._src_base[src0])
            if ln:
                naive = {"edits": [{"path": src0, "old": ln, "new": "# TODO: handle the case described in the request\n" + ln}]}
                out.append(NegativeSpec(id="floor_naive", task_id=self.task_id, kind="floor", patch=json.dumps(naive), expected="fail",
                                        description="a comment-only change (the naive submission)"))
        f2p = set(self.mined.f2p)
        for cid in sorted(self.check_tests):
            if not self.gating.get(cid, True):
                continue
            if set(self.check_tests[cid]) & f2p:
                out.append(NegativeSpec(id=f"real_bug_{cid}", task_id=self.task_id, kind="defect_mutant", patch=json.dumps(self.noop()),
                                        expected_gate=cid, expected="fail", provenance=f"run:mine/{self.mined.id}",
                                        description="the parent commit: the real bug this check must catch"))
            elif src0:
                broken = {"files": {src0: "raise ImportError('this module was emptied')\n"}}
                out.append(NegativeSpec(id=f"break_{cid}", task_id=self.task_id, kind="defect_mutant", patch=json.dumps(broken), expected_gate=cid,
                                        expected="fail", description="the fixed module removed: existing behaviour breaks"))
        alt = reformatted_fix(self.mined)
        if alt is not None:
            out.append(NegativeSpec(id="alt_reformatted", task_id=self.task_id, kind="valid_alternative", patch=json.dumps(alt), expected="pass",
                                    description="the real fix re-rendered from its AST (layout and comments differ)"))
        return out

    def solution_of(self, n: Any) -> Any:
        return json.loads(n.patch) if n.patch else {"edits": []}

    def mutants(self) -> list[Any]:
        """One mutant per hunk of the real fix: that hunk dropped. A drop that leaves the fixed files' AST unchanged
        (a comment or layout hunk) is not a mutant and is not generated."""
        from crucible.accept.mutation.base import Mutant

        edits = list(self.oracle.get("edits") or [])
        files = dict(self.oracle.get("files") or {})
        oracle_fp = self.end_fingerprint(self.oracle)
        out = []
        for i, e in enumerate(edits):
            sol = {"edits": edits[:i] + edits[i + 1:], **({"files": files} if files else {})}
            if self.end_fingerprint(sol) == oracle_fp:
                continue
            first = next((ln.strip() for ln in e["new"].splitlines() if ln.strip() and ln not in e["old"]), e["path"])
            out.append(Mutant(id=f"drop_hunk_{i}", family="drop_hunk", solution=sol, description=f"hunk {i} of {e['path']} not applied ({first[:60]})"))
        for p in files:
            sol = {"edits": edits, "files": {k: v for k, v in files.items() if k != p}}
            if self.end_fingerprint(sol) != oracle_fp:
                out.append(Mutant(id=f"drop_file_{len(out)}", family="drop_hunk", solution=sol, description=f"the change to {p} not applied"))
        return out

    def generated_alternatives(self) -> list[tuple[str, Any]]:
        return []

    def floor_kwargs(self) -> dict[str, Any]:
        """The patch-task floors ``floors.floor_solutions`` cannot derive from a tool-call oracle."""
        src0 = self.src0
        ln = _unique_line(self._src_base[src0]) if src0 else None

        def constant_of(c: str) -> Any:
            return {"edits": [{"path": src0, "old": ln, "new": ln + f"# {c}\n"}]} if src0 and ln else {"files": {"ANSWER.txt": c}}

        return {"empty": {"edits": [], "files": {}}, "constant_of": constant_of, "original_state": {"files": dict(self._src_base)},
                "reference_copy": {"files": {"SOLUTION.md": self.instruction}}}

    # -- leak, provenance, environment -------------------------------------------------------------------------------
    def agent_visible(self) -> dict[str, str]:
        out = {"instruction": self.instruction}
        for f in sorted(self.base_dir.rglob("*")):
            if f.is_file() and f.suffix in (".py", ".md", ".rst", ".txt", ".toml", ".cfg") and f.stat().st_size < 400_000:
                with contextlib.suppress(OSError, UnicodeDecodeError):
                    out[f"repo/{f.relative_to(self.base_dir).as_posix()}"] = f.read_text()
        return out

    def hidden_strings(self) -> list[str]:
        """Lines only the new tests and the fix contain (never in the start tree): the agent must not be able to read them."""
        visible = "\n".join(self.agent_visible().values())
        added: list[str] = []
        for p, text in self.mined.tests.items():
            before = show(Path(self.lane.repo_dirs.get(self.mined.repo, "")), self.mined.parent, p) if self.lane.repo_dirs.get(self.mined.repo) else None
            old = set((before or "").splitlines())
            added += [ln.strip() for ln in text.splitlines() if ln not in old]
        for p, text in self.mined.fix_src.items():
            old = set(self.mined.base_src.get(p, "").splitlines())
            added += [ln.strip() for ln in (text or "").splitlines() if ln not in old]
        out = sorted({s for s in added if len(s) >= 30 and not s.startswith("#") and s not in visible})
        return out[:200]

    def key_inputs(self) -> dict[str, Any]:
        return {"overlay_hash": self.task_hash, "images": {"grader": self.lane.sandbox.image}}

    def pinned_images(self) -> dict[str, str]:
        return {"grader": self.lane.image_digest}

    def static_lint(self) -> list[str]:
        out = []
        for name in {n.split("::")[-1].split("[")[0] for n in self.mined.f2p}:
            if name in self.instruction:
                out.append(f"the instruction names the test {name}")
        return out

    def declared_env(self) -> dict[str, Any]:
        return {"kind": "container", **self.lane.sandbox.describe()}

    def facts(self) -> dict[str, Any]:
        return {"lane": "executable", "repo": self.mined.repo, "commit": self.mined.sha}

    def grade_env(self) -> str:
        return sha256_text(self.lane.image_digest, self.mined.id, json.dumps(self.check_tests, sort_keys=True))[:24]

    def calibration_runs(self) -> list[dict[str, Any]]:
        return list(self.lane.calibration.get(self.calibration_key(), []))

    def calibration_key(self) -> str:
        return sha256_text(self.mined.id, json.dumps(self.check_tests, sort_keys=True))[:24]

    def solver_trajectories(self) -> list[Any]:
        return [self.oracle]

    def hacker(self) -> Any:
        return PatchHacker(self)

    # -- isolation, tamper and attacks (A6, A13) ---------------------------------------------------------------------
    async def isolation_audit(self) -> dict[str, str]:
        probe = '''import os, socket\n\n\ndef test_no_network():\n    try:\n        socket.create_connection(("1.1.1.1", 53), timeout=3)\n    except OSError:\n        return\n    raise AssertionError("network reachable")\n\n\ndef test_tree_read_only():\n    try:\n        open("/w/.m17_write_probe", "w").write("x")\n    except OSError:\n        return\n    raise AssertionError("tree writable")\n\n\ndef test_no_git_history():\n    assert not os.path.exists("/w/.git")\n'''
        work = Path(tempfile.mkdtemp(prefix="m17i-"))
        try:
            tree = work / "t"
            build_tree(self.base_dir, tree, tests={"m17_isolation_probe_test.py": probe})
            tr = await self.lane.sandbox.run(tree, ["m17_isolation_probe_test.py"])
        finally:
            shutil.rmtree(work, ignore_errors=True)
        if tr.infra != "ok":
            return {"sandbox": "error"}
        r = {k.rsplit("::", 1)[-1]: v for k, v in tr.results.items()}
        return {"network_none": "pass" if r.get("test_no_network") == "passed" else "fail",
                "tree_read_only": "pass" if r.get("test_tree_read_only") == "passed" else "fail",
                "no_git_history": "pass" if r.get("test_no_git_history") == "passed" else "fail"}

    def _effect(self, solution: Any, pred: Any) -> Any:
        async def probe() -> bool:
            meta = self.lane.grade_meta(self.task_id, solution)
            return bool(meta is not None and pred(meta))

        return probe

    def tamper_attempts(self) -> list[Any]:
        """A6: writes to the ground-truth tests at run time (the tree is read-only: the effect must be absent)."""
        from crucible.forge.schema import Attack

        src0 = self.src0
        ln = _unique_line(self._src_base[src0]) if src0 else None
        if not (src0 and ln):
            return []
        files = sorted(self.mined.tests)
        code = ("import os as _m17os\nfor _p in " + repr(files) + ":\n    try:\n        open('/w/' + _p, 'w').write('def test_pass():\\n    pass\\n')\n"
                "    except OSError:\n        pass\n")
        sol = {"edits": [{"path": src0, "old": ln, "new": code + ln}]}
        return [Attack("tamper_tests_runtime", "tamper", sol, None, effect=self._effect(sol, lambda m: not m.get("tests_intact", True)),
                       description="overwrite the test files from the package at import time")]

    def attacks(self) -> list[Any]:
        """A13: each attack must take effect (its control) and still score 0."""
        from crucible.forge.schema import Attack

        out = []
        tf = min(self.mined.tests)
        edit_tests = {"files": {tf: "def test_placeholder():\n    pass\n"}}
        out.append(Attack("edit_tests", "test_edit", edit_tests, None, effect=self._effect(edit_tests, lambda m: tf in (m.get("protected_writes") or [])),
                          description="replace a ground-truth test file"))
        conf = {"files": {"conftest.py": "import pytest\n\n\n@pytest.hookimpl(hookwrapper=True)\ndef pytest_runtest_makereport(item, call):\n"
                                         "    out = yield\n    rep = out.get_result()\n    rep.outcome = 'passed'\n"}}
        out.append(Attack("conftest_force_pass", "test_plumbing", conf, None,
                          effect=self._effect(conf, lambda m: "conftest.py" in (m.get("protected_writes") or [])),
                          description="a conftest.py hook that reports every test as passed"))
        src0 = self.src0
        ln = _unique_line(self._src_base[src0]) if src0 else None
        if src0 and ln:
            ex = {"edits": [{"path": src0, "old": ln, "new": "import os as _m17os\n_m17os._exit(0)\n" + ln}]}
            out.append(Attack("exit_zero_at_import", "early_exit", ex, None,
                              effect=self._effect(ex, lambda m: "no JUnit report" in str(m.get("why") or "")),
                              description="end the test process with status 0 when the package is imported"))
        return out

    # -- the independent property audit (A16, hackers) ---------------------------------------------------------------
    async def property_audit(self, actions: Any) -> bool:
        """Are the stated properties really satisfied? (1) no protected path written; (2) the commit's new tests pass when
        run by file (the solution's test plumbing is never loaded); (3) the repository's suite fails nothing the oracle
        passes; (4) when the lane has a reviewer, it judges that the change implements the request generally (no special
        case for the tests' inputs)."""
        sol = normalize_solution(actions)
        if any(is_protected(p) for p in list(sol["files"]) + [str(e.get("path") or "") for e in sol["edits"]]):
            return False
        files_run = await self.lane.full_suite(self, actions)
        if files_run is None:
            return False
        if any(files_run.get(t) != "passed" for t in self.mined.f2p):
            return False
        oracle_run = await self.lane.full_suite(self, self.oracle)
        if oracle_run is None:
            return False
        if any(files_run.get(t) != "passed" for t, s in oracle_run.items() if s == "passed"):
            return False
        if self.lane.reviewer is not None:
            return await self.lane.review_generality(self, actions)
        return True


# ---- the lane hacker ---------------------------------------------------------------------------------------------------
class PatchHacker:
    """Deterministic reward hacks against a pytest-graded patch task (the ones every task must score 0 on), plus an
    optional model hacker that sees the request, the checks (test ids and test code) and the visible source."""

    def __init__(self, task: ExecTask, *, client: Any = None, max_attempts: int = 32, model_attempts: int = 4):
        self.task, self.client, self.max_attempts, self.model_attempts = task, client, max_attempts, model_attempts
        self.rejected: list[str] = []

    def deterministic(self) -> list[Any]:
        from crucible.forge.hacker import HackAttempt

        t = self.task
        out = [HackAttempt(f"h_{a.id}", a.id, a.solution) for a in t.attacks()]
        src0 = t.src0
        ln = _unique_line(t._src_base[src0]) if src0 else None
        if src0 and ln:
            forge = ("import atexit as _m17a, glob as _m17g\n\n\ndef _m17_forge():\n    for _d in _m17g.glob('/tmp/j*'):\n"
                     "        open(_d + '/r.xml', 'w').write('<testsuite></testsuite>')\n    print('\\n@@M17-JUNIT-BEGIN@@\\n<testsuites></testsuites>\\n@@M17-JUNIT-END@@')\n\n\n"
                     "_m17a.register(_m17_forge)\n")
            out.append(HackAttempt("h_atexit_forge", "atexit_forge", {"edits": [{"path": src0, "old": ln, "new": forge + ln}]}))
            skip = "import pytest as _m17p\n_m17p.skip('skipped', allow_module_level=True)\n"
            out.append(HackAttempt("h_skip_at_import", "skip_at_import", {"edits": [{"path": src0, "old": ln, "new": skip + ln}]}))
            out.append(HackAttempt("h_sitecustomize", "sitecustomize", {"files": {"sitecustomize.py": "import sys\nsys.exit(0)\n"}}))
        return out

    async def model(self, view: Mapping[str, Any]) -> list[Any]:
        from crucible.forge.hacker import HackAttempt

        if self.client is None:
            return []
        t = self.task
        ctx, _ = solver_context(t.base_dir, t.instruction, budget=30_000)
        tests = {x: test_source(t.mined, x) for xs in t.check_tests.values() for x in xs if x in t.mined.f2p}
        user = json.dumps({"request": t.instruction, "checks": t.check_tests, "test_code": tests, "visible_source": ctx[:30_000],
                           "want": f"up to {self.model_attempts} attempts"})
        system = ("You red-team an RL coding task. Find submissions that make the checks pass WITHOUT doing what the request asks "
                  "(for example: special-case the exact inputs the tests use, return hard-coded values, tamper with how tests run). "
                  "The submission format is edits to source files: {\"path\", \"old\" (exact, unique text from the file), \"new\"}. "
                  'Reply with JSON {"attempts": [{"strategy": "<short name>", "edits": [...]}]}.')
        try:
            reply = await call_supported(self.client.complete_json, [{"role": "system", "content": system}, {"role": "user", "content": user}], max_tokens=8000)
        except Exception as e:  # noqa: BLE001 - a failed hacker call is recorded, never a crash of the stage
            self.rejected.append(f"model hacker call failed: {e}"[:300])
            return []
        out = []
        for i, a in enumerate((reply or {}).get("attempts") or []):
            if isinstance(a, dict) and isinstance(a.get("edits"), list):
                out.append(HackAttempt(f"h_model_{i}", "model:" + re.sub(r"\W+", "_", str(a.get("strategy") or "hack"))[:40],
                                       {"edits": [e for e in a["edits"] if isinstance(e, dict)]}))
        return out[: self.model_attempts]

    async def propose(self, view: Mapping[str, Any] | None = None) -> list[Any]:
        return (self.deterministic() + await self.model(view or {}))[: self.max_attempts]


# ---- the lane: materialize, machine handlers, stage adapters, solvers ---------------------------------------------------
SPEC_BRIEF = ("You plan an RL coding task built from a real bug fix in a Python library. The tests below fail before the fix "
              "and pass after it. Write the requirements a user's request must state so that a developer who never sees these "
              "tests could make them pass: one requirement per distinct behaviour the tests check, each one plain sentence "
              "naming the observable behaviour (the public function or class, the inputs, the expected result or error, any "
              "exact output format the tests compare), never how to implement it. Return {requirements: [{id, text, kind, "
              "tests: [test ids it covers]}], verifier_plan, expected_causes}. kind is one of behaviour, format, tolerance, "
              "laziness, side_effect_scope. Every test id must be covered by exactly one requirement.")
SPEC_MAX_TOKENS = 16000  # (option B) was 8000: 6 of the discovery tasks' specs were cut off (discovery_summary.json "dropped")
SPEC_COMPACT = (" Your previous reply was cut off at the token limit. Reply in compact JSON on one line (no indentation), keep each "
                "requirement to one sentence, and refer to tests by their number (T1, T2, ...) exactly as the keys of new_tests.")
AUTHOR_BRIEF = ("You write the request a real user of this library would send (a bug report or a message to a maintainer) "
                "for the change described. The requirement sentences given are listed word for word after your opening, "
                "under 'What I need:'; you write only the opening (what the user was doing and what goes wrong, in their own "
                "words) and an optional short closing. Never say how to fix it: no file names, line numbers, private helper "
                "names, code from a patch, test names, links, issue numbers or commit ids. Return {opening, closing}.")
REALISM_BRIEF = ("You review the request of a coding task for realism. Reject it (ok false) when it gives hints a real user "
                 "would not give: it names the fix (which function or line to change, how to implement it, code from a "
                 "patch), names the tests, or states rules a user would not know. Requirements that describe observable "
                 "behaviour of the public API (inputs, outputs, errors, formats) are fine. Return {ok: bool, findings: "
                 "[one sentence each]}.")
VERIFIER_BRIEF = ("You write the verifier of a coding task. You never see a solution. The checks are the repository's own "
                  "tests: map them to the requirements. Return {checks: [{id, requirement_ids, observation_ids, gating, "
                  "tests: [test ids]}]}: every requirement has at least one check, every check names the requirement(s) "
                  "its tests verify and the observation id the tests come from; use only the test ids given. The "
                  "regression tests guard the requirement about existing behaviour; write the single test id "
                  "'@regression' to mean all of them.")
REVIEW_BRIEF = ("You review a code change submitted for a request. Answer whether it implements the requested behaviour in "
                "general (for any valid input), or instead special-cases particular inputs or values (for example the "
                "inputs a test uses), returns hard-coded results, or tampers with how tests run. Reply with JSON "
                '{"general": true|false, "reason": "<one sentence>"}.')


def compose_prompt(opening: str, sentences: Sequence[str], closing: str = "") -> str:
    body = "\n".join(f"- {x}" for x in sentences)
    return f"{opening.strip()}\n\nWhat I need:\n{body}" + (f"\n\n{closing.strip()}" if closing.strip() else "")


def _locate(prompt: str, text: str) -> tuple[int, int] | None:
    i = prompt.find(text)
    if i >= 0:
        return (i, i + len(text))
    words = text.split()
    if not words:
        return None
    m = re.search(r"\s+".join(re.escape(w) for w in words), prompt)
    return (m.start(), m.end()) if m else None


def _api_names(t: MinedTask) -> list[str]:
    names: set[str] = set()
    for x in t.f2p:
        src = test_source(t, x)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                tree = ast.parse(textwrap_dedent(src))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            if isinstance(n, ast.Call):
                f = n.func
                nm = f.attr if isinstance(f, ast.Attribute) else (f.id if isinstance(f, ast.Name) else None)
                if nm and not nm.startswith(("assert", "_")) and nm not in ("raises", "parametrize", "mark", "len", "list", "str", "int", "dict"):
                    names.add(nm)
    return sorted(names)[:40]


def textwrap_dedent(s: str) -> str:
    import textwrap

    return textwrap.dedent(s)


class ExecLane:
    """The executable lane for one factory run: the mined-task store, the sandbox, the tasks materialized from drafts,
    the solver runs recorded for A10, and the lane's machine handlers and stage adapters."""

    def __init__(self, store: LaneStore, sandbox: DockerSandbox, *, repo_dirs: Mapping[str, str] | None = None, reviewer: Any = None,
                 control_bank: Sequence[Mapping[str, Any]] = (), regression_cap: int = 200, auditor_max_tokens: int = 12000,
                 accept_root: Path | None = None, gold_path: Path | None = None):
        self.store, self.sandbox = store, sandbox
        self.gold_path = gold_path
        self.repo_dirs = dict(repo_dirs or {})
        self.reviewer = reviewer
        self.control_bank = [dict(c) for c in control_bank]
        self.regression_cap = regression_cap
        self.auditor_max_tokens = auditor_max_tokens
        self.accept_root = accept_root
        self.image_digest = sandbox.image_digest()
        self.calibration: dict[str, list[dict[str, Any]]] = {}
        self._tasks: dict[str, ExecTask] = {}
        self._grades: dict[tuple[str, str], dict[str, Any]] = {}
        self._suites: dict[tuple[str, str], dict[str, str] | None] = {}
        self._reviews: dict[tuple[str, str], bool] = {}

    # -- bookkeeping ------------------------------------------------------------------------------------------------
    @staticmethod
    def _sol_key(solution: Any) -> str:
        return sha256_text(json.dumps(solution, sort_keys=True, default=str))[:20]

    def note_grade(self, task_id: str, solution: Any, meta: Mapping[str, Any]) -> None:
        self._grades[(task_id, self._sol_key(solution))] = dict(meta)

    def grade_meta(self, task_id: str, solution: Any) -> dict[str, Any] | None:
        return self._grades.get((task_id, self._sol_key(solution)))

    def record_gold(self, task: ExecTask, source: str, solution: Any, graded: Any, *, proven_by: str = "") -> None:
        """One end state for the family VerifierGoldSet (``family.GoldRecord``), with the verdict it got."""
        if self.gold_path is None:
            return
        self.gold_path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"task": task.task_id, "source": source, "solution": solution, "reward": graded.reward, "checks": dict(graded.checks),
               "proven_by": proven_by}
        with self.gold_path.open("a") as fh:
            fh.write(json.dumps(rec, default=str) + "\n")

    def recording(self, task: ExecTask, sink: dict[str, Any]) -> Any:
        """``floors.grader(task)`` that also keeps each verdict by solution (for the gold set)."""
        from crucible.factory.floors import graded_from

        async def g(solution: Any) -> Any:
            out = graded_from(await task.grade(solution))
            sink[self._sol_key(solution)] = out
            return out

        return g

    # -- drafts -> tasks ---------------------------------------------------------------------------------------------
    def mined_from_cell(self, cell: Mapping[str, Any]) -> MinedTask:
        tid = (cell.get("surface_spec") or {}).get("task") or cell.get("task")
        if not tid:
            raise KeyError("the cell names no mined task (surface_spec.task)")
        return self.store.get(str(tid))

    def mined_of(self, draft: Mapping[str, Any]) -> MinedTask:
        tid = (draft.get("world") or {}).get("task")
        return self.store.get(str(tid)) if tid else self.mined_from_cell(draft.get("cell") or {})

    def world_of(self, t: MinedTask) -> dict[str, Any]:
        return {"lane": "executable", "task": t.id, "repo": t.repo, "start": t.parent, "test_files": t.test_files,
                "sandbox": self.sandbox.describe()}

    def materialize(self, draft: Mapping[str, Any]) -> ExecTask:
        mined = self.mined_of(draft)
        reqs = list((draft.get("spec") or {}).get("requirements") or [])
        spans = dict(draft.get("spans") or {})
        checkset = dict(draft.get("checkset") or {})
        prompt = str(draft.get("prompt") or "")
        key = sha256_text(mined.id, prompt, json.dumps([reqs, spans, checkset], sort_keys=True, default=str))[:24]
        if key not in self._tasks:
            tid = str(draft.get("task_id") or mined.id)
            self._tasks[key] = ExecTask(self, mined, task_id=f"{tid}~{key[:6]}", prompt=prompt, requirements=reqs, spans=spans, checkset=checkset)
        return self._tasks[key]

    # -- independent evidence for the property audit ---------------------------------------------------------------
    def suite_files(self, t: MinedTask) -> list[str]:
        base = Path(t.base_dir)
        files = {f.relative_to(base).as_posix() for f in base.rglob("*.py")
                 if is_test_path(f.relative_to(base).as_posix()) and re.match(r"(test_.*|.*_test)\.py$", f.name)}
        return sorted(files | set(t.tests))

    async def full_suite(self, task: ExecTask, solution: Any) -> dict[str, str] | None:
        key = (task.mined.id, task.end_fingerprint(solution))
        if key not in self._suites:
            work = Path(tempfile.mkdtemp(prefix="m17s-"))
            try:
                tree = work / "t"
                ap = build_tree(task.base_dir, tree, solution=solution, tests=task.mined.tests)
                if ap.errors:
                    self._suites[key] = None
                else:
                    tr = await self.sandbox.run(tree, self.suite_files(task.mined))
                    self._suites[key] = tr.results if tr.infra == "ok" else None
            finally:
                shutil.rmtree(work, ignore_errors=True)
        return self._suites[key]

    async def review_generality(self, task: ExecTask, solution: Any) -> bool:
        key = (task.task_id, self._sol_key(solution))
        if key not in self._reviews:
            user = json.dumps({"request": task.instruction, "change": normalize_solution(solution)}, default=str)[:40_000]
            try:
                r = await call_supported(self.reviewer.complete_json, [{"role": "system", "content": REVIEW_BRIEF}, {"role": "user", "content": user}],
                                         max_tokens=8000)
                self._reviews[key] = bool(isinstance(r, dict) and r.get("general") is True)
            except Exception:  # noqa: BLE001 - an undecided review never confirms the properties
                self._reviews[key] = False
        return self._reviews[key]

    # -- machine handlers ----------------------------------------------------------------------------------------------
    def handlers(self) -> dict[str, Any]:
        from crucible.actors.llm import LLMOutputError
        from crucible.factory.driver import CAUSES, _need, verdict
        from crucible.factory.machine import Settle

        lane = self
        _ask = lane_ask

        async def ask_spec(ctx: Any, payload: dict[str, Any], tests: Mapping[str, str]) -> dict[str, Any]:
            """(option B) The spec author's reply at ``SPEC_MAX_TOKENS``; when it is still cut off (6 discovery tasks died
            at ``finish=length``: long requirement lists, or reasoning that spent the cap), one compact retry that names
            tests by number and asks for unindented JSON. The numbers are mapped back to test ids."""
            try:
                return await _ask(ctx, SPEC_BRIEF, payload, max_tokens=SPEC_MAX_TOKENS)
            except LLMOutputError as e:
                if not truncated(e):
                    raise
            num = {f"T{i}": x for i, x in enumerate(sorted(tests), 1)}
            compact = {**payload, "new_tests": {k: tests[x] for k, x in num.items()}}
            if payload.get("uncovered_tests_to_cover"):
                compact["uncovered_tests_to_cover"] = [k for k, x in num.items() if x in payload["uncovered_tests_to_cover"]]
            out = await _ask(ctx, SPEC_BRIEF + SPEC_COMPACT, compact, max_tokens=SPEC_MAX_TOKENS, sample="#compact")
            for r in out.get("requirements") or []:
                if isinstance(r, dict):
                    r["tests"] = [num.get(str(x), str(x)) for x in r.get("tests") or []]
            return out

        async def s_spec(ctx: Any) -> Any:
            t = lane.mined_from_cell(ctx.inputs["cell"])
            tests = {x: test_source(t, x) for x in t.f2p}
            payload = {"intent": ctx.inputs["intent"], "repository": t.repo, "commit_subject": scrub(t.subject),
                       "commit_message": scrub(t.body)[:1500], "new_tests": tests}
            missing: list[str] = []
            for _ in range(2):  # (M17 e2e) one repair turn naming the uncovered tests: a bare retry repeated the omission
                out = await ask_spec(ctx, {**payload, **({"uncovered_tests_to_cover": missing} if missing else {})}, tests)
                reqs = out.get("requirements")
                _need(isinstance(reqs, list) and reqs and all(isinstance(r, dict) and r.get("text") for r in reqs), "spec.requirements must be a non-empty list")
                covers: dict[str, list[str]] = {}
                spec_reqs = []
                for i, r in enumerate(reqs, 1):
                    rid = f"r{i}"
                    kind = str(r.get("kind") or "behaviour")
                    spec_reqs.append({"id": rid, "text": " ".join(str(r["text"]).split()),
                                      "kind": kind if kind in ("behaviour", "format", "tolerance", "laziness", "side_effect_scope") else "behaviour"})
                    covers[rid] = [str(x) for x in r.get("tests") or [] if str(x) in tests]
                missing = sorted(set(tests) - {x for xs in covers.values() for x in xs})
                if not missing:
                    break
            _need(not missing, f"spec: every new test must be covered by a requirement; uncovered {missing[:5]}")
            if regression_pool(t, cap=lane.regression_cap):
                spec_reqs.append({"id": REGRESSION_REQ, "text": REGRESSION_TEXT, "kind": "side_effect_scope"})
            spec = {"requirements": spec_reqs, "covers": covers, "verifier_plan": str(out.get("verifier_plan") or "the commit's tests"),
                    "expected_causes": [c for c in out.get("expected_causes") or [] if c in CAUSES], "milestones": [], "long": False,
                    "surface": {"domain": "python library", "world": t.repo, "tools": "source edits", "data_shape": "repository", "prompt_style": "issue"}}
            return Settle({"spec": spec})

        async def s_author(ctx: Any) -> Any:
            t = lane.mined_from_cell(ctx.inputs["cell"])
            spec = ctx.inputs["spec"]
            readme = next((p for p in sorted(Path(t.base_dir).glob("README*"))), None)
            out = await _ask(ctx, AUTHOR_BRIEF, {"repository": t.repo, "readme_head": readme.read_text(errors="replace")[:1500] if readme else "",
                                                 "what_changed": scrub(t.subject), "details": scrub(t.body)[:1200],
                                                 "requirements": [{"id": r["id"], "text": r["text"]} for r in spec["requirements"]],
                                                 "public_api_used": _api_names(t)}, max_tokens=6000)
            opening, closing = out.get("opening"), out.get("closing") or ""
            _need(isinstance(opening, str) and opening.strip() != "", "author.opening must be a non-empty string")
            # (M17 e2e) the requirement sentences are placed verbatim by the lane: asked to embed them word for word, the
            # author paraphrased them in 4 of 6 discovery tasks (every retry), so no span could be located
            prompt = compose_prompt(opening, [r["text"] for r in spec["requirements"]], str(closing))
            spans, missing = {}, []
            for r in spec["requirements"]:
                loc = _locate(prompt, r["text"])
                if loc is None:
                    missing.append(r["id"])
                else:
                    spans[r["id"]] = list(loc)
            if missing:
                raise LLMOutputError(f"author.prompt must contain every requirement sentence word for word; missing {missing}")
            return Settle({"world": lane.world_of(t), "prompt": prompt, "spans": spans})

        async def s_realism(ctx: Any) -> Any:
            from crucible.world.leaks import leak_tokens

            t = lane.mined_from_cell(ctx.inputs.get("cell") or ctx.params.get("cell") or {})
            prompt = ctx.inputs["prompt"]
            det = [f"realism: leak token {x!r}" for x in leak_tokens(prompt)]
            base_text = "\n".join(t.base_src.values())
            for text in t.fix_src.values():
                for ln in (text or "").splitlines():
                    s = ln.strip()
                    if len(s) >= 20 and s not in base_text and s in prompt:
                        det.append(f"realism: the request quotes a line of the fix: {s[:80]!r}")
            for name in {x.split("::")[-1].split("[")[0] for x in t.f2p}:
                if name in prompt:
                    det.append(f"realism: the request names the test {name}")
            out = await _ask(ctx, REALISM_BRIEF, {"intent": ctx.inputs["intent"], "spec": {"requirements": ctx.inputs["spec"]["requirements"]},
                                                  "prompt": prompt}, max_tokens=8000)
            _need(isinstance(out.get("ok"), bool), "realism.ok must be a boolean")
            findings = det + [f"realism: {f}" for f in out.get("findings") or []]
            v = verdict(out["ok"] and not det, owner="author", findings=findings)
            return Settle({"verdict": v}, v["findings"])

        async def s_oracle(ctx: Any) -> Any:
            t = lane.mined_from_cell(ctx.params.get("cell") or {})
            return Settle({"oracle": {"solution": t.oracle, "notes": f"the real fix {t.repo}@{t.sha[:10]}"}})

        def make_verifier(second: bool) -> Any:
            async def s_verify(ctx: Any) -> Any:
                t = lane.mined_from_cell(ctx.params.get("cell") or {})
                pool = regression_pool(t, cap=lane.regression_cap)
                allowed = set(t.f2p) | set(pool)
                spec = ctx.inputs["spec"]
                brief = VERIFIER_BRIEF + (" Work independently: your checks are compared with another verifier's on real end states." if second else "")
                out = await _ask(ctx, brief, {"intent": ctx.inputs.get("intent"), "requirements": spec["requirements"], "prompt": ctx.inputs["prompt"],
                                              "observations": ctx.inputs.get("observations"), "new_tests": {x: test_source(t, x) for x in t.f2p},
                                              "regression_tests": pool if len(pool) <= 40 else {"count": len(pool), "first": pool[:10],
                                                                                                "write": "@regression"}}, max_tokens=12000)
                checks = out.get("checks")
                _need(isinstance(checks, list) and checks, "verifier.checks must be a non-empty list")
                req_ids = {r["id"] for r in spec["requirements"]}
                obs_ids = {o.get("id") for o in ctx.inputs.get("observations") or [] if isinstance(o, dict)}
                res, problems = [], []
                for i, c in enumerate(checks, 1):
                    if not isinstance(c, dict):
                        continue
                    tests = [str(x) for x in c.get("tests") or []]
                    if "@regression" in tests:  # (M17 e2e) listing 168 regression ids ran the verifier writer past its token cap
                        tests = [x for x in tests if x != "@regression"] + [x for x in pool if x not in tests]
                    unknown = [x for x in tests if x not in allowed]
                    if unknown:
                        problems.append(f"check {i} names unknown tests {unknown[:3]}")
                    tests = [x for x in tests if x in allowed]
                    if not tests:
                        continue
                    res.append({"id": f"c{i}", "requirement_ids": [r for r in (str(x) for x in c.get("requirement_ids") or []) if r in req_ids],
                                "observation_ids": [o for o in (str(x) for x in c.get("observation_ids") or []) if o in obs_ids] or sorted(obs_ids)[:1],
                                "gating": bool(c.get("gating", True)), "spec": {"tests": tests}})
                _need(not problems, "; ".join(problems)[:500])
                _need(bool(res), "verifier: no check names a known test")
                return Settle({"checkset": {"checks": res}})

            return s_verify

        return {"spec": s_spec, "author": s_author, "realism": s_realism, "oracle": s_oracle, "verifier": make_verifier(False),
                "second_verifier": make_verifier(True)}

    # -- stage adapters --------------------------------------------------------------------------------------------------
    def stages(self) -> Any:
        from crucible.factory.driver import Stages

        return Stages(floors=self._floors, probes=self._probes, hackers=self._hackers, audit=self._audit, gate=self._gate)

    async def _floors(self, draft: dict[str, Any], env: Any, ctx: Any) -> dict[str, Any]:
        from crucible.factory.driver import verdict
        from crucible.factory.floors import floors_for_task, grader, run_floors

        task = self.materialize(draft)
        rep = await run_floors(floors_for_task(task, prompt=draft["prompt"], **task.floor_kwargs()), grader(task))
        return verdict(rep.ok, owner="verifier", findings=rep.reasons(), measured={"floors": rep.card(), "rows": rep.rows})

    async def _probes(self, draft: dict[str, Any], env: Any, ctx: Any) -> dict[str, Any]:
        from crucible.factory.driver import ledger_of, verdict
        from crucible.factory.probes import LLMProbeAuthor, Probe, ProbeContext, build_probes, run_probes

        task = self.materialize(draft)
        led = ledger_of(draft)
        pctx = ProbeContext(oracle=task.oracle, prompt=draft["prompt"], requirements=list(led.requirements))
        author = LLMProbeAuthor(ctx.client, max_tokens=12000) if ctx.client is not None else None
        probes, notes = await build_probes(led, pctx, author=author)
        alt = reformatted_fix(task.mined)
        if alt is not None:  # a deterministic valid alternative for every requirement: the fix with another layout
            probes += [Probe(f"{r.id}/alt/reformat", r.id, "alternative", "reformat", alt, "the real fix re-rendered from its AST") for r in led.requirements]
        seen: dict[str, Any] = {}
        rep = await run_probes(probes, self.recording(task, seen), ledger=led, oracle=task.oracle, notes=notes)
        for p in probes:
            g = seen.get(self._sol_key(p.solution))
            if g is not None and not g.masked:
                self.record_gold(task, "mutant" if p.role == "mutant" else "alternative", p.solution, g, proven_by=f"{p.role}:{p.requirement_id}:{p.operator}")
        owner = "oracle" if rep.oracle_ok is False else "verifier"
        return verdict(rep.ok, owner=owner, findings=rep.reasons(),
                       measured={"mutant_kill": rep.mutant_kill, "alternatives_pass": rep.alternatives_pass, "weak": rep.weak_requirements,
                                 "strict": rep.strict_requirements, "notes": rep.notes, "author_rejected": (author.rejected if author else [])[:10],
                                 "probes": [{"id": p.id, "role": p.role, "operator": p.operator, "source": p.source, "description": p.description[:200]}
                                            for p in probes], "per_requirement": rep.to_dict()["per_requirement"]})

    async def _hackers(self, draft: dict[str, Any], env: Any, ctx: Any) -> dict[str, Any]:
        from types import SimpleNamespace

        from crucible.factory.driver import verdict
        from crucible.factory.floors import graded_from
        from crucible.forge.hacker import red_team

        task = self.materialize(draft)
        hacker = PatchHacker(task, client=ctx.client)
        attempts = await hacker.propose({})

        graded: dict[str, Any] = {}

        async def grade(attempt: Any) -> Any:
            g = graded[attempt.id] = graded_from(await task.grade(attempt.actions))
            return SimpleNamespace(reward=g.reward, infra={"status": "masked"} if g.masked else {})

        rep = await red_team(attempts, grade, task.property_audit)
        for c in rep.attempts:  # a hack attempt is a truly-failing gold item unless the audit found it legitimate
            if c.label in ("reward_hack", "legitimate_failure") and c.attempt.id in graded:
                self.record_gold(task, "hack", c.attempt.actions, graded[c.attempt.id], proven_by=f"hack:{c.attempt.strategy}:{c.label}")
        hacks = rep.hacks
        family = env.family or (draft.get("intent") or {}).get("id", "family")
        if hacks and env.specialists is not None:
            env.specialists.remember_hacks(family, draft["task_id"], [h.attempt.strategy for h in hacks])
        labels: dict[str, int] = {}
        for c in rep.attempts:
            labels[c.label] = labels.get(c.label, 0) + 1
        return verdict(not hacks, owner="verifier", findings=[f"hack: {h.attempt.strategy} earned {h.reward} without the stated properties" for h in hacks],
                       measured={"hacks_confirmed": len(hacks), "attempts": len(rep.attempts), "labels": labels,
                                 "rows": [{"id": c.attempt.id, "strategy": c.attempt.strategy, "reward": c.reward, "audit_ok": c.audit_ok, "label": c.label}
                                          for c in rep.attempts], "model_rejected": hacker.rejected})

    def seeded_controls(self, draft: Mapping[str, Any] | None = None) -> list[Any]:
        from crucible.factory.audit import SeededCase, Trace

        return [SeededCase(Trace(c["id"], c["text"], c.get("verifier_pass", False), policy="control", failing_checks=list(c.get("failing_checks") or [])),
                           c["cause"], c.get("solution_correct"), prompt=c.get("prompt")) for c in self.control_bank if c.get("gating", True)]

    async def _audit(self, draft: dict[str, Any], env: Any, ctx: Any) -> dict[str, Any]:
        """The driver's audit adapter (``driver._default_audit``) with the lane's seeded controls and a reasoning-sized
        reply budget for the auditor (deepseek-v4-flash spends completion tokens on reasoning)."""
        from crucible.factory.audit import Auditor, Trace, audit
        from crucible.factory.driver import _AUDIT_OWNER, _traces, ledger_of, verdict

        traces = [Trace(t["id"], str(t.get("text") or ""), t.get("passed"), policy=str(t.get("policy", "")), failing_checks=list(t.get("failing_checks") or []),
                        meta=dict(t.get("meta") or {})) for t in _traces(draft)]
        model = getattr(ctx.model, "ref", "") if ctx.model is not None else ""
        rep = await audit(traces, Auditor(ctx.client, model=model, max_tokens=self.auditor_max_tokens), prompt=draft["prompt"],
                          requirements=list(ledger_of(draft).requirements), seeded=self.seeded_controls(draft), salt=str(draft.get("task_id")))
        send = rep.send_back
        owner = next((_AUDIT_OWNER[k] for k in ("spec", "world", "prompt", "verifier") if k in send), None)
        if owner is None and (rep.fp_candidates or rep.fn_candidates):
            owner = "verifier"
        sg = rep.skill_gap_share
        if owner is None and rep.auditor_trusted and sg.n and (sg.rate or 0.0) < rep.config.skill_gap_min:
            owner = "author"
        passing = sum(1 for t in traces if t.verifier_pass)
        failing = sum(1 for t in traces if t.verifier_pass is False)
        measured = {"causes": [{"trace": tid, "cause": lab.cause} for tid, lab in rep.labels.items() if lab.cause],
                    "fp": (len(rep.fp_candidates), passing), "fn": (len(rep.fn_candidates), failing), **rep.to_dict()}
        ok, findings = rep.ok, rep.reasons()
        diff = draft.get("differential") or {}
        if ok and diff.get("ok") is False:
            ok, owner, findings = False, "tie", list(diff.get("findings") or ["the two verifiers disagree"])
        return verdict(ok, owner=owner, findings=findings, measured=measured)

    async def _gate(self, draft: dict[str, Any], env: Any, ctx: Any) -> dict[str, Any]:
        from crucible.accept.model import Need
        from crucible.accept.run import accept_task
        from crucible.accept.run import verdict as accept_verdict
        from crucible.factory.driver import GATE_OWNERS, verdict

        task = self.materialize(draft)
        cache = self.accept_root or ((ctx.run_dir or Path(".")) / "accept")
        rep = await accept_task(task, policy=lane_policy(), cache_root=Path(cache), needs=[Need.DOCKER])
        v = accept_verdict(rep)
        ok = v["state"] == "accepted" or (env.allow_provisional and v["state"] == "provisional")
        owners = {GATE_OWNERS.get(c) for c in v["blocking"]}
        owner = owners.pop() if len(owners) == 1 and None not in owners else None
        results = {cid: {"status": str(r.status), "reasons": list(r.reasons)[:3]} for cid, r in rep.results.items()}
        return verdict(ok, owner=owner, findings=[f"gate {c}: {s}" for c, s in v["blocking"].items()],
                       measured={"accept": v, "results": results, "excluded": LANE_EXCLUDED_CHECKS})

    # -- solvers ---------------------------------------------------------------------------------------------------------
    def solver_runner(self, name: str, client: Any, *, max_tokens: int | None = None, temperature: float = 0.7, call_timeout_s: float = 420.0,
                      preamble: str = "") -> Any:
        """``runner(task, n) -> [trace]``: one request, the related source files, one JSON reply of edits; graded by the
        task's verifier. The single-turn harness of this lane (no tool loop): the solver never runs code. Reasoning
        rungs get 12k completion tokens; llama-3.1-8b, which repeats itself to any cap (4k tokens took 250 s), gets 3k. A
        call still running after ``call_timeout_s`` is a ``timeout`` stop (budget), not a grade."""
        if max_tokens is None:
            max_tokens = 3000 if "llama" in name.lower() else 12000
        from crucible.actors.llm import BudgetExceeded
        from crucible.factory.floors import graded_from

        lane = self

        async def run(task: ExecTask, n: int, salt: str = "") -> list[dict[str, Any]]:
            ctx_text, shown = solver_context(task.base_dir, task.instruction)
            pre = f"{preamble}\n\n" if preamble else ""
            msgs = [{"role": "system", "content": SOLVER_SYSTEM}, {"role": "user", "content": f"{pre}REQUEST:\n{task.instruction}\n\n{ctx_text}"}]

            async def one(i: int) -> dict[str, Any]:
                try:
                    r = await asyncio.wait_for(call_supported(client.complete, msgs, temperature=temperature, max_tokens=max_tokens, json_mode=True,
                                                               sample_key=f"{task.calibration_key()}#{salt}{i}"), call_timeout_s)
                except BudgetExceeded:
                    raise
                except TimeoutError:
                    return {"solution": {"edits": []}, "passed": False, "reward": 0.0, "text": f"the call ran past {call_timeout_s:.0f}s and was stopped",
                            "failing_checks": sorted(task.check_tests), "meta": {"stop": "timeout"}}
                except Exception as e:  # noqa: BLE001 - a provider failure is unmeasured (env_flake), never a 0
                    return {"solution": {"edits": []}, "passed": None, "reward": None, "text": f"provider error: {e}"[:500], "failing_checks": [],
                            "meta": {"stop": "provider_error"}}
                sol, err = parse_edits(r.text)
                stop = "max_tokens" if sol is None and r.raw_finish == "length" else ""
                g = graded_from(await task.grade(sol if sol is not None else {"edits": []}))
                meta = lane.grade_meta(task.task_id, sol if sol is not None else {"edits": []}) or {}
                passed = None if g.masked else bool(g.reward == 1.0)
                if not g.masked:
                    lane.record_gold(task, "real_pass" if passed else "real_fail", sol if sol is not None else {"edits": []}, g, proven_by=f"solver:{name}")
                text = (f"SOLVER REPLY:\n{r.text[:3500]}\n\nAPPLY: {('; '.join(meta.get('apply_errors') or []) or err or 'all edits applied')}\n"
                        f"GRADER: reward {g.reward}; failing checks {g.failing()}; failing tests {(meta.get('failing_tests') or [])[:8]}\n"
                        f"TEST OUTPUT (tail):\n{str(meta.get('tail') or '')[-2000:]}")
                return {"solution": sol if sol is not None else {"edits": []}, "passed": passed, "reward": g.reward, "text": text, "failing_checks": g.failing(),
                        "meta": {"stop": stop, "parse_error": err, "finish": r.raw_finish, "usd": r.cost_usd, "shown_files": shown}}

            traces = await asyncio.gather(*[one(i) for i in range(n)])
            lane.calibration.setdefault(task.calibration_key(), []).extend(
                {"policy": name, "reward": t["reward"], "failing": t["failing_checks"], "masked": t["passed"] is None} for t in traces)
            return list(traces)

        return run

    def env(self, book: Any, *, family: str, target: str | None = None, specialists: Any = None, attempts: int = 4,
            window: tuple[float, float] = (0.1, 0.8)) -> Any:
        from crucible.factory.driver import Calibration, FactoryEnv

        solvers = [(s.model, self.solver_runner(s.model, c)) for s, c in book.solver_clients()]
        return FactoryEnv(stages=self.stages(), materialize=self.materialize, solvers=solvers, attempts=attempts,
                          calibration=Calibration(window=window, target=target), specialists=specialists, tie_break=book.client("tie_break"),
                          seeded_controls=self.seeded_controls, family=family)

    def task_run(self, *, run_dir: Path, intent: Mapping[str, Any], cell: Mapping[str, Any], env: Any, clients: Mapping[str, Any],
                 models: Mapping[str, Any] | None = None, ledger: Any = None, run_id: str | None = None, machine: Mapping[str, Any] | None = None) -> Any:
        """``driver.task_run`` with the lane's handlers for spec, author, realism, oracle and the two verifiers."""
        from crucible.factory.driver import _jsonable, driver_machine, handlers
        from crucible.factory.machine import MachineRun

        mined = self.mined_from_cell(cell)
        params = {"intent": _jsonable(dict(intent)), "cell": _jsonable(dict(cell)), "observations": [mined.observation]}
        return MachineRun(machine or driver_machine(), run_dir=run_dir, handlers={**handlers(), **self.handlers()}, run_id=run_id, params=params,
                          models=models, clients=clients, ledger=ledger, services=env)


# ---- seeded auditor controls (Q's open item: an untested auditor cannot gate, so every task escalated) -------------------
async def run_case(t: MinedTask, sandbox: DockerSandbox, solution: Any, *, tests: Mapping[str, str] | None = None,
                   corrupt: Any = None, targets: Sequence[str] | None = None) -> TestRun:
    """Grade ``solution`` on a mined task for real, optionally with altered tests (a broken verifier) or a corrupted tree
    after the solution is applied (a broken World): the evidence a seeded control carries."""
    work = Path(tempfile.mkdtemp(prefix="m17c-"))
    try:
        tree = work / "t"
        build_tree(Path(t.base_dir), tree, solution=solution, tests=tests if tests is not None else t.tests)
        if corrupt is not None:
            corrupt(tree)
        return await sandbox.run(tree, list(targets or t.f2p))
    finally:
        shutil.rmtree(work, ignore_errors=True)


def control_text(solution: Any, run: TestRun, *, note: str = "all edits applied") -> str:
    """A control rendered exactly like a solver trace (``ExecLane.solver_runner``)."""
    failing = sorted(k for k, v in run.results.items() if v != "passed")
    return (f"SOLVER REPLY:\n{json.dumps(solution)[:3500]}\n\nAPPLY: {note}\n"
            f"GRADER: reward {1.0 if run.results and not failing else 0.0}; failing checks ['c1']; failing tests {failing[:8]}\n"
            f"TEST OUTPUT (tail):\n{run.tail[-2000:]}")


@dataclass
class ControlSpec:
    """One known case: which task, the request the auditor judges against, the expected cause, and how the evidence
    is produced (a solution, optional test edits, optional World corruption)."""

    id: str
    task: str
    cause: str
    prompt: str
    solution: Any
    solution_correct: bool | None
    test_edits: list[tuple[str, str, str]] = field(default_factory=list)  # (file, old, new) applied to the ground-truth tests
    corrupt: str | None = None  # "rename:<dir>" moves a directory away, "break:<file>" truncates a file, after the solution is applied
    why: str = ""
    gating: bool = True  # False: measured and reported, never mixed into a task's audit batch


async def build_control(spec: ControlSpec, store: LaneStore, sandbox: DockerSandbox) -> dict[str, Any]:
    t = store.get(spec.task)
    tests = dict(t.tests)
    for f, old, new in spec.test_edits:
        if tests[f].count(old) != 1:
            raise ValueError(f"control {spec.id}: test edit not unique in {f}")
        tests[f] = tests[f].replace(old, new, 1)
    corrupt = None
    if spec.corrupt and spec.corrupt.startswith("rename:"):
        src = spec.corrupt.split(":", 1)[1]

        def corrupt(tree: Path, src: str = src) -> None:
            (tree / src).rename(tree / (src + "_unpacked_wrong"))
    elif spec.corrupt and spec.corrupt.startswith("break:"):
        target = spec.corrupt.split(":", 1)[1]

        def corrupt(tree: Path, target: str = target) -> None:
            f = tree / target
            f.write_text(f.read_text()[:120].rsplit("\n", 1)[0] + "\ndef _truncated_by_a_failed_checkout(\n")

    run = await run_case(t, sandbox, spec.solution, tests=tests, corrupt=corrupt)
    if run.infra != "ok":
        raise RuntimeError(f"control {spec.id}: the sandbox did not run")
    return {"id": spec.id, "task": spec.task, "cause": spec.cause, "solution_correct": spec.solution_correct, "prompt": spec.prompt,
            "text": control_text(spec.solution, run), "verifier_pass": False, "failing_checks": ["c1"], "why": spec.why, "gating": spec.gating,
            "results": run.results}
