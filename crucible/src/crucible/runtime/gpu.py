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

from collections.abc import Callable
from dataclasses import dataclass

from crucible.core.schema.run import Failure

INITS = {"docker-init", "tini", "init", "dumb-init", "catatonit", "systemd"}


@dataclass
class GpuHost:
    name: str
    kind: str = "docker_ssh"  # docker_ssh | ssh_exec | harbor_cloud | agentenv_cloud
    docker_host: str | None = None
    gpu_arch: str | None = None  # e.g. "12.0" (sm_120)
    shared: bool = False

    @classmethod
    def from_config(cls, name: str, cfg: dict) -> GpuHost:
        return cls(name=name, kind=cfg.get("kind", "docker_ssh"), docker_host=cfg.get("docker_host") or (f"ssh://{cfg['ssh']}" if cfg.get("ssh") else None),
                   gpu_arch=cfg.get("gpu_arch"), shared=bool(cfg.get("shared", False)))

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
