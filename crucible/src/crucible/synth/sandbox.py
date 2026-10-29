"""Untrusted App code in a restricted subprocess (part2-paper-core §11 "synthesized Apps"; evidence doc §3.2 F1).

A synthesized App is model-written Python. It is **never executed in this process**. ``SandboxWorld`` is a
``SynthWorld`` (and ``Perturbable``) whose every operation is one JSON request to a child process
(``_sandbox_child.py``) that holds the ``PyWorld``; ``run_tool_tests`` runs blind tool tests the same way.

The child is restricted in four layers:
1. **Static check** (``check_code``, here and again in the child): an AST allowlist. Imports only from ``ALLOWED_IMPORTS``;
   no ``eval``/``exec``/``compile``/``open``/``__import__``/``getattr``-family names, no ``global``/``nonlocal``, no
   dunder attribute access, no ``async``. A module that fails is refused before any process starts.
2. **Restricted namespace** in the child: the module runs with a builtins allowlist and an import hook that admits only
   the allowed modules; ``build(app)`` is the one entry point.
3. **The process**: a scrubbed environment (no API keys: only ``PATH``, ``LANG`` and ``PYTHONHASHSEED``), ``python -I -B``
   (no user site, no environment variables, no bytecode writes), the working directory a fresh ``work/`` under the
   caller's run directory, CPU-time and file-size limits (``resource``), sockets disabled before the module loads.
4. **The OS sandbox** where one exists: on macOS ``sandbox-exec`` with a profile that denies all network access, denies
   writes outside the working directory and denies reading the repository's ``.env``; on Linux ``unshare -rn`` (a new
   network namespace with no interfaces) when it is permitted. ``SANDBOX_KIND`` records which layer 4 applied.

Every request has a wall-clock timeout; a timeout or a dead child kills the process and the next ``reset`` starts a new
one (a World exception for the call that hit it, which ``program.run`` turns into an ended episode). An undeclared write
in the child is raised here as ``iface.UndeclaredWrite`` (a World defect, never an observation).
"""

from __future__ import annotations

import ast
import asyncio
import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from crucible.synth.canon import CanonState
from crucible.synth.iface import Obs, Perturbation, ToolInfo, UndeclaredWrite

ALLOWED_IMPORTS = frozenset({"math", "re", "datetime", "json", "copy", "itertools", "collections", "functools", "statistics", "string", "decimal",
                             "typing", "__future__", "dataclasses", "enum"})
FORBIDDEN_NAMES = frozenset({"eval", "exec", "compile", "open", "__import__", "globals", "locals", "vars", "getattr", "setattr", "delattr",
                             "input", "breakpoint", "help", "exit", "quit", "memoryview", "__builtins__", "__loader__", "__spec__"})
CHILD = Path(__file__).with_name("_sandbox_child.py")
SRC = Path(__file__).resolve().parents[2]  # the directory holding the crucible package
# every .env on the way up from the sources (a worktree's and its main checkout's): the child may not read them
def _secret_files() -> tuple[str, ...]:
    out = []
    for p in SRC.parents:
        try:
            if (p / ".env").exists():  # a stat, never a read
                out.append(str(p / ".env"))
        except OSError:  # inside the sandbox the stat itself is denied
            continue
    return tuple(out)


class SandboxError(RuntimeError):
    """The child failed (startup, a protocol error, a timeout): the World is unusable until the next reset."""


class CodeRejected(ValueError):
    """The App code failed the static check."""


def check_code(code: str) -> list[str]:
    """Static problems of an App module (empty = it may be loaded in the sandbox)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error line {e.lineno}: {e.msg}"]
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in ALLOWED_IMPORTS:
                    out.append(f"line {node.lineno}: import {a.name} is not allowed")
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS or node.level:
                out.append(f"line {node.lineno}: from {node.module} import is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            out.append(f"line {node.lineno}: name {node.id} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__") and node.attr.endswith("__"):
            out.append(f"line {node.lineno}: dunder attribute {node.attr} is not allowed")
        elif isinstance(node, ast.Attribute) and node.attr.startswith("_") and node.attr in ("_getframe", "_current_frames"):
            out.append(f"line {node.lineno}: attribute {node.attr} is not allowed")
        elif isinstance(node, ast.Global | ast.Nonlocal):
            out.append(f"line {node.lineno}: global/nonlocal is not allowed")
        elif isinstance(node, ast.AsyncFunctionDef | ast.Await | ast.AsyncFor | ast.AsyncWith):
            out.append(f"line {node.lineno}: async code is not allowed")
    if not any(isinstance(n, ast.FunctionDef) and n.name == "build" for n in tree.body):
        out.append("the module defines no top-level build(app)")
    return out


def sandbox_kind() -> str:
    if platform.system() == "Darwin" and Path("/usr/bin/sandbox-exec").exists():
        return "sandbox-exec"
    if platform.system() == "Linux" and shutil.which("unshare"):
        return "unshare"
    return "none"


SANDBOX_KIND = sandbox_kind()


def _profile(work: Path) -> str:
    w = str(work.resolve()).replace('"', "")
    deny_read = "".join(f'(deny file-read* (literal "{s.replace(chr(34), "")}"))' for s in _secret_files())
    return ("(version 1)(allow default)(deny network*)(deny file-write*)"
            f'(allow file-write* (subpath "{w}") (subpath "/dev"))' + deny_read)


def _command(app_dir: Path, work: Path, mode: str, cpu_s: int) -> list[str]:
    py = [sys.executable, "-I", "-B", str(CHILD), "--src", str(SRC), "--app", str(app_dir), "--work", str(work), "--mode", mode, "--cpu", str(cpu_s)]
    kind = SANDBOX_KIND
    if kind == "sandbox-exec":
        return ["/usr/bin/sandbox-exec", "-p", _profile(work), *py]
    if kind == "unshare":
        return ["unshare", "-rn", *py]
    return py


def _env() -> dict[str, str]:
    return {"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "PYTHONHASHSEED": "0"}


class _Child:
    def __init__(self, app_dir: Path, work: Path, mode: str, *, timeout_s: float, cpu_s: int):
        self.app_dir, self.work, self.mode, self.timeout_s, self.cpu_s = app_dir, work, mode, timeout_s, cpu_s
        self.proc: asyncio.subprocess.Process | None = None
        self.lock = asyncio.Lock()
        self.stderr_tail = ""

    async def start(self) -> None:
        self.work.mkdir(parents=True, exist_ok=True)
        cmd = _command(self.app_dir, self.work, self.mode, self.cpu_s)
        try:
            self.proc = await asyncio.create_subprocess_exec(*cmd, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                             stderr=asyncio.subprocess.PIPE, cwd=str(self.work), env=_env(), limit=64 * 1024 * 1024)
        except OSError as e:
            if SANDBOX_KIND == "unshare":  # unprivileged namespaces refused: fall back to the in-process restrictions only
                py = _command(self.app_dir, self.work, self.mode, self.cpu_s)[2:]
                self.proc = await asyncio.create_subprocess_exec(*py, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                                                                 stderr=asyncio.subprocess.PIPE, cwd=str(self.work), env=_env(), limit=64 * 1024 * 1024)
            else:
                raise SandboxError(f"cannot start the sandbox: {e}") from None
        hello = await self._read(self.timeout_s * 3)
        if not hello.get("ok"):
            await self.kill()
            raise SandboxError(f"the sandbox refused the App: {str(hello.get('error'))[:2000]}")

    async def _read(self, timeout: float) -> dict[str, Any]:
        assert self.proc is not None and self.proc.stdout is not None
        try:
            line = await asyncio.wait_for(self.proc.stdout.readline(), timeout)
        except TimeoutError:
            await self.kill()
            raise SandboxError("the sandbox timed out") from None
        if not line:
            err = b""
            if self.proc.stderr is not None:
                try:
                    err = await asyncio.wait_for(self.proc.stderr.read(4000), 2)
                except TimeoutError:
                    pass
            self.stderr_tail = err.decode(errors="replace")[-2000:]
            await self.kill()
            raise SandboxError(f"the sandbox exited: {self.stderr_tail[-600:]}")
        return json.loads(line)

    async def request(self, req: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        async with self.lock:
            if self.proc is None or self.proc.returncode is not None:
                await self.start()
            assert self.proc is not None and self.proc.stdin is not None
            self.proc.stdin.write((json.dumps(req, default=str) + "\n").encode())
            await self.proc.stdin.drain()
            return await self._read(timeout or self.timeout_s)

    async def kill(self) -> None:
        p, self.proc = self.proc, None
        if p is not None and p.returncode is None:
            try:
                p.kill()
            except ProcessLookupError:
                pass
            try:
                await asyncio.wait_for(p.wait(), 5)
            except TimeoutError:
                pass


class SandboxWorld:
    """A ``SynthWorld`` over an App directory (``app.py``, ``design.json``, ``schema.json``, ``seed.json``) served by a
    sandboxed child (module docstring). ``work`` is the child's working directory (a run directory's ``work/<n>``)."""

    def __init__(self, app_dir: str | os.PathLike[str], work: str | os.PathLike[str], *, timeout_s: float = 20.0, cpu_s: int = 3600):
        self.app_dir = Path(app_dir).resolve()
        problems = check_code((self.app_dir / "app.py").read_text())
        if problems:
            raise CodeRejected("; ".join(problems[:8]))
        self.child = _Child(self.app_dir, Path(work).resolve(), "world", timeout_s=timeout_s, cpu_s=cpu_s)
        self._tools: list[ToolInfo] | None = None

    async def _req(self, op: str, **kw: Any) -> Any:
        r = await self.child.request({"op": op, **kw})
        if r.get("undeclared"):
            raise UndeclaredWrite(str(r["undeclared"]))
        if not r.get("ok"):
            raise SandboxError(f"{op}: {str(r.get('error'))[:1000]}")
        return r.get("value")

    async def reset(self, seed: int) -> Any:
        return await self._req("reset", seed=int(seed))

    async def tools(self) -> list[ToolInfo]:
        if self._tools is None:
            self._tools = [ToolInfo.model_validate(t) for t in await self._req("tools")]
        return list(self._tools)

    async def call(self, tool: str, args: dict[str, Any]) -> Obs:
        r = await self._req("call", tool=tool, args=args)
        return Obs.model_validate(r)

    async def snapshot(self) -> Any:
        return await self._req("snapshot")

    async def restore(self, ref: Any) -> None:
        await self._req("restore", ref=ref)

    async def canon(self) -> CanonState:
        v = await self._req("canon")
        return CanonState.build(v["apps"], v.get("unobserved") or [])

    async def perturb(self, p: Perturbation) -> None:
        await self._req("perturb", p=p.model_dump(mode="json"))

    async def tables(self) -> dict[str, Any]:
        """The raw tables (worldgen's checks; never shown to a policy)."""
        return await self._req("tables")

    async def close(self) -> None:
        await self.child.kill()


async def run_tool_tests(app_dir: str | os.PathLike[str], work: str | os.PathLike[str], tests: list[dict[str, Any]], *, timeout_s: float = 120.0
                         ) -> list[dict[str, Any]]:
    """Run blind tool tests against an App in the sandbox: one result ``{id, passed, detail}`` per test. A module that
    fails the static check fails every test with the problems as the detail (the debugger sees them)."""
    app = Path(app_dir).resolve()
    problems = check_code((app / "app.py").read_text())
    if problems:
        return [{"id": t.get("id"), "passed": False, "detail": "code rejected: " + "; ".join(problems[:8])} for t in tests]
    child = _Child(app, Path(work).resolve(), "test", timeout_s=timeout_s, cpu_s=600)
    try:
        r = await child.request({"op": "run_tests", "tests": tests}, timeout=timeout_s)
    except SandboxError as e:
        return [{"id": t.get("id"), "passed": False, "detail": f"sandbox: {e}"[:1500]} for t in tests]
    finally:
        await child.kill()
    if not r.get("ok"):
        return [{"id": t.get("id"), "passed": False, "detail": f"load: {str(r.get('error'))[:1500]}"} for t in tests]
    return list(r["value"])


async def static_info(app_dir: str | os.PathLike[str], work: str | os.PathLike[str]) -> dict[str, Any]:
    """The tools the module registers (name, read_only, writes) without serving a World: the conformance check."""
    child = _Child(Path(app_dir).resolve(), Path(work).resolve(), "test", timeout_s=60.0, cpu_s=120)
    try:
        r = await child.request({"op": "info"})
    finally:
        await child.kill()
    if not r.get("ok"):
        raise SandboxError(str(r.get("error"))[:1500])
    return dict(r["value"])
