"""Harbor ``DockerEnvironment`` with NVIDIA GPUs, no-network isolation and an init process.

Harbor's local Docker provider has no GPU path (``capabilities.gpus`` stays False, so any task with
``gpus >= 1`` is rejected). Following Harbor's own ``KataEnvironment`` pattern, this subclass changes
nothing but the compose overlay appended last to ``_docker_compose_paths``:

- ``gpu_mode="runtime"`` (default): ``runtime: nvidia`` + ``NVIDIA_VISIBLE_DEVICES`` — for daemons where
  the nvidia runtime is registered but not default (our lab RTX 5090 host);
- ``gpu_mode="devices"``: compose device reservations (``--gpus``), for nvidia-container-toolkit hosts;
- ``network_none``: ``network_mode: none`` on ``main`` (what our migration runs used: ``--network none``);
- ``init: true`` (zombie reaping — DESIGN Appendix B.5) and ``shm_size``.

Loaded through Harbor's factory: ``create_environment_from_import_path("crucible.interop.harbor.gpu_docker:GpuDockerEnvironment", ...)``.
Remote daemons work unchanged: Harbor's compose subprocess inherits ``DOCKER_HOST`` (e.g. ``ssh://user@gpu-host``).
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any

from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.environments.docker.docker import DockerEnvironment


class GpuDockerEnvironment(DockerEnvironment):
    def __init__(self, *args: Any, gpu_mode: str = "runtime", gpu_devices: str = "all", network_none: bool = False,
                 init: bool = True, shm_size: str | None = "8g", **kwargs: Any) -> None:
        if gpu_mode not in ("runtime", "devices", "none"):
            raise ValueError(f"gpu_mode must be runtime|devices|none, got {gpu_mode!r}")
        self._gpu_mode = gpu_mode
        self._gpu_devices = gpu_devices
        self._network_none = network_none
        self._init = init
        self._shm_size = shm_size
        self._overlay_dir: tempfile.TemporaryDirectory[str] | None = None
        self._overlay_path: Path | None = None
        super().__init__(*args, **kwargs)  # capability validation runs in here: state above must exist first

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        update: dict[str, Any] = {"gpus": self._gpu_mode != "none"}
        if self._network_none:
            update["disable_internet"] = True
        return super().capabilities.model_copy(update=update)

    @property
    def _docker_compose_paths(self) -> list[Path]:
        return [*super()._docker_compose_paths, self._ensure_overlay()]

    def _ensure_overlay(self) -> Path:
        if self._overlay_path is None:
            self._overlay_dir = tempfile.TemporaryDirectory(prefix="crucible-gpu-overlay-")
            svc: dict[str, Any] = {}
            if self._gpu_mode == "runtime":
                svc["runtime"] = "nvidia"
                svc["environment"] = {"NVIDIA_VISIBLE_DEVICES": self._gpu_devices, "NVIDIA_DRIVER_CAPABILITIES": "compute,utility"}
            elif self._gpu_mode == "devices":
                dev: dict[str, Any] = {"driver": "nvidia", "capabilities": ["gpu"]}
                if self._gpu_devices == "all":
                    dev["count"] = "all"
                else:
                    dev["device_ids"] = self._gpu_devices.split(",")
                svc["deploy"] = {"resources": {"reservations": {"devices": [dev]}}}
            if self._network_none:
                svc["network_mode"] = "none"
            if self._init:
                svc["init"] = True
            if self._shm_size:
                svc["shm_size"] = self._shm_size
            path = Path(self._overlay_dir.name) / "docker-compose-crucible-gpu.json"
            path.write_text(json.dumps({"services": {"main": svc}}, indent=2))
            self._overlay_path = path
        return self._overlay_path

    async def stop(self, delete: bool) -> None:
        try:
            await super().stop(delete)
        finally:
            if self._overlay_dir is not None:
                self._overlay_dir.cleanup()
                self._overlay_dir = None
                self._overlay_path = None
