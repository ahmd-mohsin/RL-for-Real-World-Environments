"""GPU hosts and sandbox preflight (research m9-m10/02 §3.7; Appendix B.5, B.7).

``GpuHost`` comes from ``crucible.toml [gpu_hosts]`` (kind ``docker_ssh`` for the lab RTX 5090: ``--runtime=nvidia``, never
``--gpus``; ``shared = true`` means polite mode). ``preflight(run)`` runs inside a freshly started sandbox before any agent
or grade sees it, through any ``run(cmd) -> (rc, stdout)`` callable:

- **PID 1 must be an init** (``docker-init``, ``tini``, ``init``, ``dumb-init``): with ``sleep infinity`` as PID 1 zombies are
  never reaped and wait loops hang (B.5) -> ``crucible:zombie_wait``;
- **the GPU must be the declared one** (``nvidia-smi --query-gpu=compute_cap``): a GPU task silently running on CPU or on
  the wrong architecture (B.7) -> ``crucible:backend_error``.
A preflight failure is infrastructure: it never reaches the agent and is never scored.

M10 lab audit R-5 item 4: every host call an attempt makes (``aexec``: ssh or bash; ``aput`` / ``aget``: rsync; ``apath``)
is a coroutine on ``world.exec._run_subprocess``: off the event loop, bounded, its session killed past the bound or on a
cancel. They used to be synchronous ``subprocess.run`` calls (rsync with no timeout at all), so a stalled transfer froze every
attempt's agent loop, every ``wait_for`` and the wall clock. A call past its timeout raises ``HostTimeout``, which is
infrastructure (``transport_unreachable``), never the answer's ``verifier_timeout``. The sync ``exec`` stays for
``PerfGuard``'s threaded probe only.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from crucible.core.schema.run import Failure
from crucible.world.exec import _run_subprocess

INITS = {"docker-init", "tini", "init", "dumb-init", "catatonit", "systemd"}
TRANSFER_TIMEOUT_S = 1800.0  # M10 lab audit R-5 item 4: the longest one rsync (a task's tests, a grade's artifacts) may take


class HostTimeout(ConnectionError):
    """M10 lab audit R-5 item 4: a host call (ssh, rsync, bash, docker) that ran past its timeout. It is infrastructure:
    ``infra.classify_exception`` reads a ``ConnectionError`` as ``transport_unreachable`` (regradable, retryable), and it is
    deliberately *not* a ``TimeoutError``, so ``grading._raised_timeout`` never reads it as the answer's ``verifier_timeout``
    (a stalled host is not a patch that hung)."""


@dataclass
class GpuHost:
    name: str
    kind: str = "docker_ssh"  # docker_ssh | ssh_exec | harbor_cloud | agentenv_cloud
    docker_host: str | None = None
    gpu_arch: str | None = None  # e.g. "12.0" (sm_120)
    shared: bool = False
    ssh: str | None = None  # user@host for file staging and host probes; None = this machine
    workroot: str = "crucible_runtime"  # staging directory on the host (relative to its home)

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> GpuHost:
        return cls(name=name, kind=cfg.get("kind", "docker_ssh"), docker_host=cfg.get("docker_host") or (f"ssh://{cfg['ssh']}" if cfg.get("ssh") else None),
                   gpu_arch=cfg.get("gpu_arch"), shared=bool(cfg.get("shared", False)), ssh=cfg.get("ssh"), workroot=cfg.get("workroot", "crucible_runtime"))

    def _argv(self, cmd: str) -> list[str]:
        return ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", self.ssh, cmd] if self.ssh else ["bash", "-c", cmd]

    def exec(self, cmd: str, *, timeout: float = 120) -> tuple[int, str]:
        """Run a shell command on the host, **blocking**. M10 lab audit R-5 item 4: only for ``PerfGuard``'s probe, which runs
        in a worker thread (``perf_guard`` calls it through ``to_thread``); anything on the event loop uses ``aexec``."""
        r = subprocess.run(self._argv(cmd), capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, r.stdout

    async def aexec(self, cmd: str, *, timeout: float = 120) -> tuple[int, str]:
        """Run a shell command on the host off the event loop (staging, cleanup); ``HostTimeout`` past ``timeout``."""
        out = await _run_subprocess(self._argv(cmd), cwd=None, env=None, timeout_s=timeout, stdin=None)
        if out.timed_out:
            raise HostTimeout(f"{cmd.split()[0] if cmd.split() else 'command'} on {self.ssh or self.name} ran past {timeout:g}s")
        return out.rc, out.stdout

    async def apath(self, rel: str) -> str:
        """The absolute host path of a staging-relative path (bind mounts name host paths)."""
        if not self.ssh:
            return str((Path.home() / self.workroot / rel).resolve())
        rc, home = await self.aexec("echo $HOME")
        if rc != 0 or not home.strip().startswith("/"):
            raise RuntimeError(f"cannot resolve $HOME on {self.ssh} (rc {rc}): refusing to stage into /{self.workroot}")
        return f"{home.strip()}/{self.workroot}/{rel}"

    async def aput(self, local: Path, rel: str, *, timeout: float = TRANSFER_TIMEOUT_S) -> str:
        """Copy a local file or directory to ``<workroot>/<rel>`` on the host; returns the host path. ``HostTimeout`` when the
        rsync runs past ``timeout`` (its session is killed)."""
        dest = await self.apath(rel)
        await self.aexec(f"mkdir -p {shlex.quote(str(Path(dest).parent))}")
        src = f"{Path(local).resolve()}/" if Path(local).is_dir() else str(Path(local).resolve())
        target = f"{self.ssh}:{dest}/" if self.ssh and Path(local).is_dir() else (f"{self.ssh}:{dest}" if self.ssh else (dest + "/" if Path(local).is_dir() else dest))
        argv = ["rsync", "-a", "--delete", src, target] if Path(local).is_dir() else ["rsync", "-a", src, target]
        r = await _run_subprocess(argv, cwd=None, env=None, timeout_s=timeout, stdin=None)
        if r.timed_out:
            raise HostTimeout(f"staging {local} -> {target} ran past {timeout:g}s")
        if r.rc:
            raise RuntimeError(f"staging {local} -> {target} failed: {r.stderr[-500:]}")
        return dest

    async def aget(self, rel: str, local: Path, *, timeout: float = TRANSFER_TIMEOUT_S) -> None:
        src = await self.apath(rel) + "/"
        Path(local).mkdir(parents=True, exist_ok=True)
        r = await _run_subprocess(["rsync", "-a", f"{self.ssh}:{src}" if self.ssh else src, f"{Path(local).resolve()}/"], cwd=None, env=None, timeout_s=timeout,
                                  stdin=None)
        if r.timed_out:
            raise HostTimeout(f"fetching {src} ran past {timeout:g}s")
        if r.rc:  # a silent failure left stale or missing results to be graded (M10 audit V17)
            raise RuntimeError(f"fetching {src} failed: {r.stderr[-500:]}")

    def docker_gpu_flags(self) -> list[str]:
        return ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"]  # `--gpus all` fails on the lab host (missing MPS control mount)


def preflight(run: Callable[[str], tuple[int, str]], *, gpu_arch: str | None = None) -> Failure | None:
    rc, comm = run("cat /proc/1/comm")
    if rc != 0 or comm.strip() not in INITS:
        return Failure(kind="crucible:zombie_wait", stage="seed", reason=f"PID 1 is {comm.strip()!r}, not an init: zombies would never be reaped (start with --init)")
    if gpu_arch:
        rc, cap = run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
        caps = {c.strip() for c in cap.split()} if rc == 0 else set()
        if not caps or caps != {gpu_arch}:  # every visible GPU must match, not any one of them (M10 audit V17)
            return Failure(kind="crucible:backend_error", stage="seed", reason=f"declared GPU arch {gpu_arch}, sandbox reports {sorted(caps) or 'no GPU'}")
    return None
