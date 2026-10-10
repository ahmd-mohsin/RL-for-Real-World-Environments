"""``gpu_docker``: agentenv's local Docker sandbox provider with GPU flags (research m11-m12/05 §4.1).

agentenv's local provider passes no GPU flags. The lab host needs ``--runtime=nvidia`` (``--gpus all`` fails there:
missing MPS control mount), ``--init`` against zombie wait loops (Appendix B.5) and a large ``/dev/shm``. Flags go
through ``SandboxProvider.EXTRA_CONTAINER_RUN_ARGS``, which ``_container_args`` splices into the agent / env containers'
``docker run``. ``DOCKER_HOST=ssh://`` in the environment reaches a remote daemon (``LocalSandbox.exec`` inherits it).
Caveat (documented upstream): ``run_docker_container`` steps build their own ``docker run`` and bypass this provider.
``CONTAINER_ENV`` stays empty: it is injected into verifier environments, and secrets never go there.
"""

from __future__ import annotations

from typing import ClassVar

from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider

GPU_FLAGS = "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --init --shm-size 8g"


class GpuDockerSandboxProvider(LocalSandboxProvider):
    EXTRA_CONTAINER_RUN_ARGS = f"{LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS} {GPU_FLAGS}".strip()
    CONTAINER_ENV: ClassVar[dict] = {}
