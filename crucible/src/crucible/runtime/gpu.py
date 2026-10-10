"""GPU hosts and sandbox preflight (research m9-m10/02 §3.7; Appendix B.5, B.7).

``GpuHost`` comes from ``crucible.toml [gpu_hosts]`` (kind ``docker_ssh`` for the lab RTX 5090: ``--runtime=nvidia``, never
``--gpus``; ``shared = true`` means polite mode). ``preflight(run)`` runs inside a freshly started sandbox before any agent
or grade sees it, through any ``run(cmd) -> (rc, stdout)`` callable:

- **PID 1 must be an init** (``docker-init``, ``tini``, ``init``, ``dumb-init``): with ``sleep infinity`` as PID 1 zombies are
  never reaped and wait loops hang (B.5) -> ``crucible:zombie_wait``;
- **the GPU must be the declared one** (``nvidia-smi --query-gpu=compute_cap``): a GPU task silently running on CPU or on
  the wrong architecture (B.7) -> ``crucible:backend_error``.
A preflight failure is infrastructure: it never reaches the agent and is never scored.
"""

from __future__ import annotations

import shlex
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from crucible.core.schema.run import Failure

INITS = {"docker-init", "tini", "init", "dumb-init", "catatonit", "systemd"}


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

    def exec(self, cmd: str, *, timeout: float = 120) -> tuple[int, str]:
        """Run a shell command on the host (``PerfGuard`` probe, staging)."""
        argv = ["ssh", "-o", "BatchMode=yes", "-o", "ServerAliveInterval=30", self.ssh, cmd] if self.ssh else ["bash", "-c", cmd]
        r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, check=False)
        return r.returncode, r.stdout

    def path(self, rel: str) -> str:
        """The absolute host path of a staging-relative path (bind mounts name host paths)."""
        if not self.ssh:
            return str((Path.home() / self.workroot / rel).resolve())
        _, home = self.exec("echo $HOME")
        return f"{home.strip()}/{self.workroot}/{rel}"

    def put(self, local: Path, rel: str) -> str:
        """Copy a local file or directory to ``<workroot>/<rel>`` on the host; returns the host path."""
        dest = self.path(rel)
        self.exec(f"mkdir -p {shlex.quote(str(Path(dest).parent))}")
        src = f"{Path(local).resolve()}/" if Path(local).is_dir() else str(Path(local).resolve())
        target = f"{self.ssh}:{dest}/" if self.ssh and Path(local).is_dir() else (f"{self.ssh}:{dest}" if self.ssh else (dest + "/" if Path(local).is_dir() else dest))
        r = subprocess.run(["rsync", "-a", "--delete", src, target] if Path(local).is_dir() else ["rsync", "-a", src, target], capture_output=True, text=True, check=False)
        if r.returncode:
            raise RuntimeError(f"staging {local} -> {target} failed: {r.stderr[-500:]}")
        return dest

    def get(self, rel: str, local: Path) -> None:
        src = self.path(rel) + "/"
        Path(local).mkdir(parents=True, exist_ok=True)
        subprocess.run(["rsync", "-a", f"{self.ssh}:{src}" if self.ssh else src, f"{Path(local).resolve()}/"], capture_output=True, check=False)

    def docker_gpu_flags(self) -> list[str]:
        return ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all"]  # `--gpus all` fails on the lab host (missing MPS control mount)


def preflight(run: Callable[[str], tuple[int, str]], *, gpu_arch: str | None = None) -> Failure | None:
    rc, comm = run("cat /proc/1/comm")
    if rc != 0 or comm.strip() not in INITS:
        return Failure(kind="crucible:zombie_wait", stage="seed", reason=f"PID 1 is {comm.strip()!r}, not an init: zombies would never be reaped (start with --init)")
    if gpu_arch:
        rc, cap = run("nvidia-smi --query-gpu=compute_cap --format=csv,noheader")
        caps = {c.strip() for c in cap.split()} if rc == 0 else set()
        if gpu_arch not in caps:
            return Failure(kind="crucible:backend_error", stage="seed", reason=f"declared GPU arch {gpu_arch}, sandbox reports {sorted(caps) or 'no GPU'}")
    return None
