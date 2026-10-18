"""The sandbox protocol of a coding World (swarm-pipeline/00 §11 "port from slime"; 03 §2.1, §4.1, §5 item 2).

**PORT** (Apache-2.0, ``LICENSES/Apache-2.0.txt``) of THUDM slime ``slime/agent/sandbox.py`` at 859938d:
``SandboxTerminatedError`` (:28-29), the ``Sandbox`` Protocol (:32-73), ``EXIT_TIME_BUDGET_EXCEEDED`` (:76),
``_await_done_marker`` (:79-96) and ``exec_and_wait`` (:99-162). The four-method contract (``exec``, ``write_file``,
``read_file``, async context) and the detached-spawn + exit-marker discipline are kept as written, so a slime harness
(Claude Code, Codex) drives our World through it unchanged. Changed (Apache §4(b)): ``_await_done_marker`` takes the poll
interval as an argument (slime hard-codes 5 s), and the E2B / Sunabako providers are not ported.

**Ours.** ``ExecSandbox`` is the provider over a Crucible World: any ``world.exec.Exec`` (``DockerExec`` into the World's
``--network none`` container, ``LocalExec`` for a host workspace) rooted at the World's workspace.
- A dead container (gone, killed, the daemon refusing it) raises ``SandboxTerminatedError``: infrastructure, which every
  caller masks (V8), never a 0.
- **Egress.** The World runs with no network; the only route out is the adapter. ``egress_probe`` executes the check
  inside the sandbox (a TCP connect to a public address must fail; the adapter, when one is given as ``host:port``, must
  answer), so the claim "egress only to the adapter" is evidence, not configuration.
"""

from __future__ import annotations

import asyncio
import shlex
import time
import uuid
from pathlib import Path
from typing import Any, Protocol, Self, runtime_checkable

ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


class SandboxTerminatedError(RuntimeError):
    """The sandbox stopped while executing an agent trajectory."""


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts (slime ``sandbox.py:32-73``).

    ``write_file`` accepts in-memory content (``str``/``bytes``) or a host ``Path``. ``idempotent`` hints a backend's
    transport-retry policy; backends without retries ignore it. ``read_file`` returns ``""`` for a missing file;
    lifecycle, permission and transport failures propagate."""

    sandbox_id: str

    async def __aenter__(self) -> Self: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None, timeout: int = 120, check: bool = False,
                   idempotent: bool = True) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


EXIT_TIME_BUDGET_EXCEEDED = -1


async def _await_done_marker(sb: Sandbox, done_file: str, *, user: str, time_budget_sec: float, poll_s: float = 5.0) -> int:
    """Poll a detached command's exit-code marker until it appears; ``EXIT_TIME_BUDGET_EXCEEDED`` when the budget runs out
    first (slime ``sandbox.py:79-96``; the poll interval is a parameter here)."""
    deadline = time.time() + time_budget_sec
    while time.time() < deadline:
        await asyncio.sleep(poll_s)
        try:
            out = await sb.read_file(done_file, user=user)
        except FileNotFoundError:
            continue
        if (out or "").strip():
            return int(out.strip())
    return EXIT_TIME_BUDGET_EXCEEDED


async def exec_and_wait(sb: Sandbox, *, cmd: str, time_budget_sec: float, tag: str, user: str = "root", env: dict[str, str] | None = None,
                        workdir: str | None = None, out_file: str | None = None, want_output: bool = False, poll_s: float = 5.0) -> tuple[int, str]:
    """Run ``cmd`` to completion detached and return ``(exit_code, output)`` (slime ``sandbox.py:99-162``): a launcher
    script, a ``mkdir`` lock that dedupes a replayed spawn, ``setsid`` with output to a file and an exit-code marker, then
    short idempotent polls. Every RPC is short, so a severed stream never loses an exit code."""
    out_file = out_file or f"/tmp/.{tag}.out"
    done_file = f"/tmp/.{tag}.done"
    launcher = f"/tmp/.{tag}.sh"
    lock_dir = f"/tmp/.{tag}.spawned"
    prefix = f"cd {workdir}\nexport HOME=/home/{user}\n" if workdir else ""
    await sb.write_file(launcher, f"#!/bin/bash\n{prefix}{cmd}\necho $? > {done_file}\n", user=user)
    await sb.exec(f"rm -rf {lock_dir}; rm -f {out_file} {done_file}", user=user, timeout=30, check=True, idempotent=True)
    await sb.exec(f"chmod +x {launcher}; mkdir {lock_dir} 2>/dev/null || exit 0; setsid bash {launcher} < /dev/null > {out_file} 2>&1 &",
                  user=user, env=env, timeout=30, check=True, idempotent=True)
    code = await _await_done_marker(sb, done_file, user=user, time_budget_sec=time_budget_sec, poll_s=poll_s)
    if code == 0 and not want_output:
        return code, ""
    if want_output:
        return code, await sb.read_file(out_file, user=user)
    _, tail, _ = await sb.exec(f"tail -c 512 {out_file} 2>/dev/null", user=user, timeout=15, check=False)
    return code, tail or ""


# ---- the provider over a Crucible World ----------------------------------------------------------------------------------
# a failed command is checked for a dead sandbox when docker's own wording says so, or on 125 (the daemon refused) and
# 137 / -9 (killed); only the liveness probe decides
_DEAD_WORDS = ("No such container", "is not running", "Cannot connect to the Docker daemon", "container not found", "OCI runtime exec failed")


class ExecSandbox:
    """slime's ``Sandbox`` over a Crucible ``Exec`` rooted at a World's workspace (relative paths resolve there).

    ``alive``: an optional async probe (``() -> bool``) consulted when a command fails in a way only a dead sandbox
    explains; a ``DockerExec`` gets one from ``docker inspect`` by default. ``user`` is honoured only by executors that can
    switch users (``DockerExec``); ``LocalExec`` runs as the host user."""

    def __init__(self, exec_: Any, *, root: str | Path, sandbox_id: str | None = None, adapter_url: str | None = None, alive: Any = None,
                 network: str = "none"):
        self.exec_, self.root = exec_, str(root)
        self.sandbox_id = sandbox_id or f"crucible-sbx-{uuid.uuid4().hex[:10]}"
        self.adapter_url = adapter_url
        self.network = network
        self._alive = alive
        self.closed = False
        self.terminated: str | None = None
        self.calls = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.closed = True

    def _path(self, p: str) -> str:
        return p if p.startswith("/") else f"{self.root.rstrip('/')}/{p}"

    async def _is_alive(self) -> bool:
        if self._alive is not None:
            return bool(await self._alive())
        container = getattr(self.exec_, "container", None)
        if container is None:
            return True
        p = await asyncio.create_subprocess_exec("docker", "inspect", "-f", "{{.State.Running}}", container, stdout=asyncio.subprocess.PIPE,
                                                 stderr=asyncio.subprocess.PIPE)
        out, _ = await p.communicate()
        return p.returncode == 0 and out.decode().strip() == "true"

    async def _dead(self, why: str) -> SandboxTerminatedError:
        self.terminated = why[-300:]
        return SandboxTerminatedError(f"sandbox {self.sandbox_id} terminated: {self.terminated}")

    async def exec(self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None, timeout: int = 120, check: bool = False,
                   idempotent: bool = True) -> ExecResult:
        return await self._run(cmd, user=user, env=env, timeout=timeout, check=check)

    async def _run(self, cmd: str, *, user: str = "root", env: dict[str, str] | None = None, timeout: int = 120, check: bool = False,
                   stdin: bytes | None = None) -> ExecResult:
        if self.closed or self.terminated:
            raise SandboxTerminatedError(f"sandbox {self.sandbox_id} is {'closed' if self.closed else 'terminated'}")
        self.calls += 1
        kw: dict[str, Any] = {"cwd": self.root, "env": env, "timeout_s": timeout}
        if stdin is not None:
            kw["stdin"] = stdin
        if user != "root" and getattr(self.exec_, "name", "") == "docker":
            kw["user"] = user
        try:
            out = await self.exec_.run(cmd, **kw)
        except Exception as e:  # a transport failure: dead or not, decided by the probe
            if not await self._is_alive():
                raise await self._dead(f"{type(e).__name__}: {e}") from e
            raise
        said = (out.stderr or "") + (out.stdout or "")
        dead_like = out.rc != 0 and (any(w in said for w in _DEAD_WORDS) or out.rc in (125, 137, -9))
        if dead_like and not await self._is_alive():
            raise await self._dead(f"exit {out.rc}: {(out.stderr or out.stdout or '')[-200:]}")
        if check and out.rc != 0:
            raise RuntimeError(f"command failed ({out.rc}): {cmd[:200]}: {(out.stderr or '')[-300:]}")
        return out.rc, out.stdout, out.stderr

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        """The bytes travel on stdin (``docker exec -i``): a command argument is capped at 128 KiB on Linux, and a 52 KB
        source file base64-encoded inside a deadline wrapper exceeded it."""
        if isinstance(content, Path):
            content = content.read_bytes()
        data = content.encode() if isinstance(content, str) else bytes(content)
        q = shlex.quote(self._path(sandbox_path))
        rc, out, err = await self._run(f"mkdir -p \"$(dirname {q})\" && cat > {q}", user=user, timeout=120, stdin=data)
        if rc != 0:
            raise OSError(f"write_file {sandbox_path}: {(err or out)[-300:]}")

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        q = shlex.quote(self._path(sandbox_path))
        rc, out, err = await self.exec(f"if [ -e {q} ]; then cat {q}; else exit 3; fi", user=user, timeout=120)
        if rc == 3:
            return ""
        if rc != 0:
            raise OSError(f"read_file {sandbox_path}: {err[-300:]}")
        return out

    async def egress_probe(self, *, public: str = "1.1.1.1:53", timeout_s: float = 3.0) -> dict[str, Any]:
        """Executed inside the sandbox: a connect to ``public`` must fail; the adapter (``host:port`` of ``adapter_url``),
        when set, must connect. ``ok`` = no egress except the adapter."""
        def target(hostport: str) -> tuple[str, int]:
            h, _, p = hostport.rpartition(":")
            return h, int(p)

        code = ("import socket,sys\nh,p=sys.argv[1],int(sys.argv[2])\ns=socket.socket();s.settimeout(float(sys.argv[3]))\n"
                "try:\n s.connect((h,p));print('open')\nexcept Exception as e:\n print('closed',type(e).__name__)\n")
        q = shlex.quote(code)

        async def probe(hostport: str) -> str:
            h, p = target(hostport)
            rc, out, err = await self.exec(f"python3 -c {q} {shlex.quote(h)} {p} {timeout_s}", timeout=int(timeout_s) + 20)
            return out.strip() or f"error {rc}: {err[-120:]}"

        pub = await probe(public)
        res: dict[str, Any] = {"network": self.network, "public": public, "public_result": pub, "public_blocked": not pub.startswith("open")}
        if self.adapter_url:
            from urllib.parse import urlparse

            u = urlparse(self.adapter_url)
            ad = await probe(f"{u.hostname}:{u.port or (443 if u.scheme == 'https' else 80)}")
            res.update(adapter=self.adapter_url, adapter_result=ad, adapter_reachable=ad.startswith("open"))
        res["ok"] = res["public_blocked"] and res.get("adapter_reachable", True)
        return res
