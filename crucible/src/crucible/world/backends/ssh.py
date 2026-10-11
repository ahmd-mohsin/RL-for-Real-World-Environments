"""ssh_exec backend (m13 audit 04, S4): a World's Apps run as host processes on a remote machine over SSH.

For GPU hosts without Docker (a Runpod pod is itself a container: no user namespaces, no CAP_SYS_ADMIN, so neither rootful
nor rootless containers run there). Everything goes through ``SSHExec``: one multiplexed connection, files by ``scp``,
resource limits by ``ulimit`` (``backend_params.rlimits``: nproc, as_mb, fsize_mb, cpu_s, nofile).

- **Isolation: none.** The capability says so; the backend refuses a World that gives a policy a shell unless
  ``backend_params.allow_untrusted_shell`` is set (a trusted scripted policy, or a grader-only World).
- **GPUs: yes**, the host's own (``runtime.gpu.preflight`` checks the arch).
- Each World works in its own directory under ``workroot`` (default ``~/crucible_worlds``), removed at teardown.

``backend_params``: ``host`` (a ``crucible.toml`` [gpu_hosts] name with ``ssh``) or ``ssh`` (user@host), ``port``, ``key``,
``workroot``, ``rlimits``, ``known_hosts``, ``allow_untrusted_shell``, ``keep_workdir``.
"""

from __future__ import annotations

import shlex
import uuid

from crucible.core.schema.world import AppSpec
from crucible.world.backends.base import Backend, BackendCapabilities
from crucible.world.exec import Exec, SSHExec


class SSHExecBackend(Backend):
    name = "ssh_exec"

    def __init__(self, spec, params, *, workdir):
        super().__init__(spec, params, workdir=workdir)
        target, port, key = params.get("ssh"), params.get("port"), params.get("key")
        if params.get("host"):
            from crucible.core.config import get_config

            h = get_config().gpu_hosts.get(params["host"])
            if h is None or not getattr(h, "ssh", None):
                raise RuntimeError(f"ssh_exec host {params['host']!r} is not configured with ssh in crucible.toml [gpu_hosts]")
            target = target or h.ssh
        if not target:
            raise RuntimeError("ssh_exec needs backend_params.ssh (user@host) or backend_params.host")
        self._exec = SSHExec(target, port=port, key=key, rlimits=params.get("rlimits"), known_hosts=params.get("known_hosts"))
        self.world_dir = f"{params.get('workroot', 'crucible_worlds')}/{spec.world.replace('/', '-')}-{uuid.uuid4().hex[:8]}"
        self._abs: str | None = None

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(isolation="none", gpus=True, network_enforced=False,
                                   notes=[f"host processes on {self._exec.target} over ssh (rlimits only): do not give untrusted policies a shell"])

    def validate(self) -> list[str]:
        problems = []
        if not self.params.get("allow_untrusted_shell"):
            policy_tools = [t for a in self.spec.policy_actors for t in a.tools]
            for app in self.spec.apps:
                if app.params.get("shell") and any(t == f"{app.id}.*" or t.startswith(f"{app.id}.shell") for t in policy_tools):
                    problems.append(f"app {app.id!r} gives the policy a shell on an unisolated ssh_exec host; set allow_untrusted_shell for trusted policies")
        if self.spec.network == "no-network":
            problems.append("ssh_exec cannot enforce no-network (host processes); use network: public with a trusted policy, or a sandboxed backend")
        return problems

    async def provision(self) -> None:
        out = await self._exec.run(f"mkdir -p {shlex.quote(self.world_dir)} && cd {shlex.quote(self.world_dir)} && pwd && command -v python3 >/dev/null && "
                                   "command -v git >/dev/null && echo ok", timeout_s=60)
        if "ok" not in out.stdout:
            raise RuntimeError(f"ssh_exec host needs python3 and git for Crucible's state adapters: {(out.stderr or out.stdout)[-300:]}")
        self._abs = out.stdout.split()[0]

    def exec_for(self, app: AppSpec) -> Exec:
        return self._exec

    def base_path(self, app: AppSpec) -> str:
        return self._abs or self.world_dir

    async def teardown(self) -> None:
        if not self.params.get("keep_workdir") and self._abs:
            await self._exec.run(f"rm -rf {shlex.quote(self._abs)}", timeout_s=120)
        await self._exec.close()
