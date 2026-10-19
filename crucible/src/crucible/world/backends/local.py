"""Local backend: every App runs on this machine under the world's work directory.

For CI, toy worlds, imported MiMo/AWM worlds driven through tools, and trusted scripted policies.
There is no isolation and no network enforcement; it refuses GPU Apps and policy shells unless the
App explicitly opts in (``allow_unsafe_local_shell``).
"""

from __future__ import annotations

import shutil

from crucible.core.schema.world import AppSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import Exec, LocalExec


class LocalBackend(Backend):
    name = "local"

    def __init__(self, spec, params, *, workdir):
        super().__init__(spec, params, workdir=workdir)
        self._exec = LocalExec()
        self.apps_dir = (workdir / "apps").resolve()

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolation="none", gpus=False, network_enforced=False,
                                   notes=["no sandbox: do not give untrusted policies a shell"])

    def validate(self) -> list[str]:
        problems = []
        for app in self.spec.apps:
            if app.resources.gpus:
                problems.append(f"app {app.id!r} needs {app.resources.gpus} GPU(s); use the harbor backend")
        # a host allowlist is a promise this backend cannot keep: its tools reach any host (M1 audit R-13). ``no-network``,
        # every spec's default, stays allowed and is reported unenforced by audit_world (network.enforced)
        if self.spec.network == "allowlist" and not self.params.get("accept_unenforced"):
            problems.append("network 'allowlist' cannot be enforced on the local backend (its tools reach any host): use a container "
                            "backend, or set backend_params.accept_unenforced to run it unenforced")
        return problems

    async def provision(self) -> None:
        self.apps_dir.mkdir(parents=True, exist_ok=True)

    def exec_for(self, app: AppSpec) -> Exec:
        return self._exec

    def base_path(self, app: AppSpec) -> str:
        return str(self.apps_dir)

    async def teardown(self) -> None:
        if not self.params.get("keep_workdir"):
            shutil.rmtree(self.apps_dir, ignore_errors=True)
