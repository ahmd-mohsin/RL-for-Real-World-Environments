"""Container-exec backend: every App of a World runs inside one container that is already running, through ``docker exec``,
on this machine's daemon or a remote one (``docker_host``, e.g. the tunnel ``world.docker_tunnel.local_docker_host`` gives).

The caller starts and removes the container (it owns the flags: ``--network none``, the GPU runtime, the limits); this
backend only routes every App's commands, state snapshots and seed copies into it (``exec_for``). The App state lives at
``base`` inside the container, so a World on a remote GPU host keeps nothing on this machine: file tools, the shell, git
and the ``fs`` state adapter all run there (the kernel lanes' GPU World, swarm-pipeline/10 §9).

Params: ``container`` (required), ``docker_host`` (None = the local daemon), ``base`` (the App base path inside the
container, default ``/``), ``user`` (the default ``docker exec`` user, None = the image's), ``gpus`` and ``network_enforced``
(what the caller's container really has: reported in ``capabilities``, never assumed).
"""

from __future__ import annotations

from crucible.core.schema.world import AppSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import DockerExec, Exec


class ContainerExecBackend(Backend):
    name = "container_exec"

    def __init__(self, spec, params, *, workdir):
        super().__init__(spec, params, workdir=workdir)
        self._exec = DockerExec(str(params.get("container") or ""), docker_host=params.get("docker_host"), default_user=params.get("user"))

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolation="container", gpus=bool(self.params.get("gpus")), network_enforced=bool(self.params.get("network_enforced")),
                                   notes=[f"apps run in container {self.params.get('container')!r}" + (f" on {self.params['docker_host']}" if self.params.get("docker_host") else "")])

    def validate(self) -> list[str]:
        return [] if self.params.get("container") else ["container_exec needs backend_params.container (a running container)"]

    async def provision(self) -> None:
        return None

    def exec_for(self, app: AppSpec) -> Exec:
        return self._exec

    def base_path(self, app: AppSpec) -> str:
        return str(self.params.get("base") or "/")

    async def teardown(self) -> None:
        return None  # the caller removes the container it started
