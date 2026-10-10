"""GPU flags for agentenv's local Docker sandboxes (research m11-m12/05 §4.1; m13 audit 02, C1).

agentenv's local provider passes no GPU flags. The lab host needs ``--runtime=nvidia`` (``--gpus all`` fails there:
missing MPS control mount), ``--init`` against zombie wait loops (Appendix B.5) and a large ``/dev/shm``.

**Why not a separate provider.** The first version registered a ``gpu_docker`` provider (a ``LocalSandboxProvider``
subclass). agentenv's deploy-time type guard destroys any plugin sandbox whose ``.type`` differs from its config name
(``sandbox_provider._install_sandbox_type_guard``), and the subclass produced ``LocalSandbox`` (``type="local"``), so it
could never create a sandbox. A distinct ``gpu_docker`` type is not the fix either: agentenv decides host-URL
externalization (``host_url_for``, ``deploy_agent``, ``install_agent``, ``context``) and local-CA trust
(``store/object_store/local``) by ``sandbox_type == "local"``. So GPU support is an explicit opt-in on the builtin
local provider: ``local_gpu()`` adds the flags to ``LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS`` (which
``_container_args`` splices into every agent / env container's ``docker run``) and restores them afterwards; sandboxes
stay ``type="local"`` and keep agentenv's networking. ``DOCKER_HOST=ssh://`` reaches a remote daemon
(``LocalSandbox.exec`` inherits it). ``run_docker_container`` steps build their own ``docker run`` and bypass providers
(upstream caveat). No env vars are added: provider ``CONTAINER_ENV`` reaches verifier environments, and secrets never go
there.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

from agent_env.providers.sandbox_providers.local_sandbox import LocalSandboxProvider

GPU_FLAGS = "--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all --init --shm-size 8g"


def gpu_enabled() -> bool:
    return GPU_FLAGS in LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS


def enable_local_gpu(flags: str = GPU_FLAGS) -> str:
    """Add ``flags`` to agentenv's builtin local provider (idempotent); returns the previous value for ``disable``."""
    prev = LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS
    if flags not in prev:
        LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS = f"{prev} {flags}".strip()
    return prev


def disable_local_gpu(prev: str) -> None:
    LocalSandboxProvider.EXTRA_CONTAINER_RUN_ARGS = prev


@contextlib.contextmanager
def local_gpu(flags: str = GPU_FLAGS) -> Iterator[None]:
    prev = enable_local_gpu(flags)
    try:
        yield
    finally:
        disable_local_gpu(prev)
