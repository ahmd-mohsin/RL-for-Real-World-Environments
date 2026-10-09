"""Host-side client for the in-sandbox guard (``guard.py``): install, then call ops as root.

``Guard(exec)`` wraps a root Exec into a container world. ``PolicyExec`` is the Exec every *policy* tool uses
in an isolated world: each command runs through ``guard run`` as the unprivileged policy user (setsid,
``oom_score_adj=1000``, no supplementary groups, clean environment, process-group kill on timeout).
Verifier operations keep using the root Exec, with the interpreter pinned at setup (``python -I``).
"""

from __future__ import annotations

import importlib.resources
import json
import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

from crucible.world.exec import Exec, ExecError, ExecOut

GUARD_PATH = "/var/lib/crucible/guard.py"


class GuardError(RuntimeError):
    pass


class Guard:
    def __init__(self, root_exec: Exec, *, python: str = "python3"):
        self.exec = root_exec
        self.python = python
        self.info: dict[str, Any] = {}

    async def install(self) -> None:
        src = importlib.resources.files("crucible.world.guard").joinpath("guard.py").read_text()
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
        try:
            await self.exec.run("mkdir -p /var/lib/crucible && chmod 700 /var/lib/crucible")
            await self.exec.push_file(Path(f.name), GUARD_PATH)
        finally:
            os.unlink(f.name)
        out = await self.exec.run(f"chown root:root {GUARD_PATH} && chmod 600 {GUARD_PATH} && readlink -f \"$(command -v python3)\"")
        if not out.ok:
            raise GuardError(f"cannot install guard: {out.stderr[-300:]}")
        self.python = out.stdout.strip().splitlines()[-1] if out.stdout.strip() else "python3"  # resolved in the sandbox, not on the host

    async def call(self, op: str, args: dict[str, Any], *, timeout_s: float = 600) -> Any:
        out = await self.exec.run(f"{shlex.quote(self.python)} -I {GUARD_PATH} {op}", stdin=json.dumps(args).encode(),
                                  timeout_s=timeout_s, env={"PATH": "/usr/sbin:/usr/bin:/sbin:/bin"})
        line = next((ln for ln in out.stdout.splitlines() if ln.startswith("__GUARD__")), None)
        if line is None:
            raise GuardError(f"guard {op} produced no result (rc={out.rc}): {(out.stderr or out.stdout)[-600:]}")
        res = json.loads(line[len("__GUARD__"):])
        if not res["ok"]:
            raise GuardError(f"guard {op}: {res['error']}")
        return res["result"]

    async def setup(self, **kw: Any) -> dict[str, Any]:
        self.info = await self.call("setup", kw)
        self.python = self.info.get("python", self.python)
        return self.info


class PolicyExec(Exec):
    """Runs commands as the policy user through the guard. File pushes/pulls stay root-side (staging)."""

    name = "policy"

    def __init__(self, guard: Guard, uid: int):
        self.guard = guard
        self.uid = uid
        self.python = "python3"

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None) -> ExecOut:
        t0 = time.monotonic()
        try:
            r = await self.guard.call("run", {"uid": self.uid, "cmd": cmd, "cwd": cwd, "env": env or {}, "timeout": timeout_s or 600,
                                              "stdin": stdin.decode() if stdin is not None else None}, timeout_s=(timeout_s or 600) + 30)
        except GuardError as e:
            raise ExecError(str(e)) from e
        return ExecOut(rc=r["rc"], stdout=r["stdout"], stderr=r["stderr"], duration_s=time.monotonic() - t0, timed_out=r["timed_out"])

    async def push_file(self, src: Path, dest: str) -> None:
        await self.guard.exec.push_file(src, dest)
        await self.guard.exec.run(f"chown {self.uid}:{self.uid} {shlex.quote(dest)} && chmod 644 {shlex.quote(dest)}")

    async def pull_file(self, src: str, dest: Path) -> None:
        await self.guard.exec.pull_file(src, dest)
