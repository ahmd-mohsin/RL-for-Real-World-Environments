"""Container sandboxes a backend can put a World's state and commands into (no framework imports here).

``DockerSandbox`` runs one long-lived container from an image on this machine's Docker daemon or a remote one
(``gpu_host`` from ``crucible.toml``, e.g. the lab RTX 5090 over SSH), with the same hardening as the Harbor
overlay: ``--init``, no network for no-network worlds, a pids limit, ``SYS_RESOURCE`` (so the guard can pin the
policy's OOM score) and GPUs through the nvidia runtime or ``--gpus``. Cloud sandboxes (Modal and the other
agentenv providers) live in ``crucible.interop.agentenv.sandbox`` behind the same small interface.
"""

from __future__ import annotations

import os
import shlex
import socket
import time
import uuid
from typing import Protocol

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world.exec import DockerExec, Exec, _run_subprocess, deadline_hit, with_deadline

# A sandbox container ends by itself (M1 audit R-6): its main process is ``sleep <timeout_s + LIFETIME_MARGIN_S>``, not
# ``sleep infinity``, so one left by a crashed or killed process is gone (``--rm``) within its lifetime; it is labelled
# with its owner (host:pid), and the perf guard ignores containers whose owner is a dead process on this host.
OWNER_LABEL = "crucible.owner"
LIFETIME_MARGIN_S = 900  # restores and grading after the episode's own budget


def owner_id() -> str:
    return f"{socket.gethostname()}:{os.getpid()}"


def owner_dead(owner: str) -> bool:
    """True when ``owner`` (``host:pid``) is a process on this host that no longer runs; unknown or other hosts: False."""
    host, _, pid = owner.rpartition(":")
    if host != socket.gethostname() or not pid.isdigit() or int(pid) == os.getpid():
        return False
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return True
    except OSError:  # another user's live process
        return False
    return False


class Sandbox(Protocol):
    exec: Exec
    gpus: int
    description: str

    async def start(self) -> None: ...

    async def close(self) -> None: ...


def gpu_request(spec: WorldSpec) -> tuple[int, str | None]:
    """(count, type) the World's Apps ask for: the largest count, and the type if all GPU apps agree on one."""
    gpu_apps = [a.resources for a in spec.apps if a.resources.gpus]
    types = {r.gpu_type for r in gpu_apps if r.gpu_type}
    return max((r.gpus for r in gpu_apps), default=0), (types.pop() if len(types) == 1 else None)


def resolve_docker_host(sb: SandboxSpec) -> str | None:
    if sb.docker_host:
        return sb.docker_host
    if sb.gpu_host:
        from crucible.core.config import get_config

        host = get_config().gpu_hosts.get(sb.gpu_host)
        if host is None:
            raise RuntimeError(f"gpu_host {sb.gpu_host!r} is not configured in crucible.toml [gpu_hosts]")
        return host.docker_host or f"ssh://{host.ssh}"
    return None


async def image_id(ref: str, docker_host: str | None = None, *, container: bool = False) -> str | None:
    """Digest of the filesystem layers of an image (or of a running container's image): what a WorldRef was taken in.
    Layers, not the image id: compose stamps project labels into images it builds, so two identical builds differ in id."""
    import hashlib

    env = {"DOCKER_HOST": docker_host} if docker_host else None
    if container:
        out = await _run_subprocess(["docker", "inspect", "-f", "{{.Image}}", ref], cwd=None, env=env, timeout_s=60, stdin=None)
        if not out.ok or not out.stdout.strip():
            return None
        ref = out.stdout.strip()
    out = await _run_subprocess(["docker", "image", "inspect", "-f", "{{json .RootFS.Layers}}", ref], cwd=None, env=env, timeout_s=60, stdin=None)
    if not out.ok or not out.stdout.strip():
        return None
    return "layers:" + hashlib.sha256(out.stdout.strip().encode()).hexdigest()


def check_image(state: dict, current: str | None) -> None:
    """A WorldRef restores only into the image it was taken in (DESIGN B.5: stale images silently change behavior)."""
    taken = state.get("image")
    if taken and current and taken != current:
        from crucible.world.state.base import SnapshotMismatch

        raise SnapshotMismatch(f"WorldRef was taken in image {taken[:19]}, this world runs {current[:19]}")


class DockerSandbox:
    def __init__(self, sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None):
        self.sb, self.spec = sb, spec
        self.gpus, self.gpu_type = gpu_request(spec)
        self.docker_host = resolve_docker_host(sb)
        self.name = f"crucible-sbx-{spec.world.replace('/', '-')[:40]}-{uuid.uuid4().hex[:8]}"
        self.pids_limit, self.memory = pids_limit, memory
        self.exec: Exec = DockerExec(self.name, docker_host=self.docker_host)
        self.description = f"docker {sb.image} on {self.docker_host or 'local daemon'}" + (f", {self.gpus} GPU(s)" if self.gpus else "")

    def argv(self) -> list[str]:
        a = ["docker", "run", "-d", "--rm", "--init", "--name", self.name, "--label", f"crucible.world={self.spec.world}", "--label", "crucible.sandbox=1",
             "--label", f"{OWNER_LABEL}={owner_id()}"]
        if self.spec.network == "no-network":
            a += ["--network", "none"]
        if self.pids_limit:
            a += ["--pids-limit", str(self.pids_limit)]
        if self.spec.isolation.policy_user:
            a += ["--cap-add", "SYS_RESOURCE"]
            if (self.spec.isolation.firewall or "auto") != "off":  # root sets per-uid iptables rules; the demoted policy has no caps
                a += ["--cap-add", "NET_ADMIN"]
        if self.sb.runtime == "runsc":
            if self.gpus:
                raise RuntimeError("runtime runsc with GPUs needs gVisor's nvproxy, which this sandbox does not configure")
            a += ["--runtime=runsc"]
        if self.gpus:
            if self.sb.gpu_mode == "runtime":  # hosts where --gpus fails (the lab 5090: missing MPS mount)
                a += ["--runtime=nvidia", "-e", "NVIDIA_VISIBLE_DEVICES=all", "-e", "NVIDIA_DRIVER_CAPABILITIES=compute,utility"]
            else:
                a += ["--gpus", str(self.gpus) if self.gpus else "all"]
        if self.sb.cpus:
            a += ["--cpus", str(self.sb.cpus)]
        mem = self.memory or (f"{self.sb.memory_mb}m" if self.sb.memory_mb else None)
        if mem:
            a += ["--memory", mem]
        return [*a, "--entrypoint", "", self.sb.image, "sleep", str(self.lifetime_s)]

    @property
    def lifetime_s(self) -> int:
        """How long the container lives at most: the sandbox's ``timeout_s`` plus a margin, then it ends and is removed."""
        return self.sb.timeout_s + LIFETIME_MARGIN_S

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    image_id: str | None = None

    async def start(self) -> None:
        if isinstance(self.exec, DockerExec):  # taken before the run: the container cannot outlive it (its end is infrastructure)
            self.exec.expires_at = time.time() + self.lifetime_s
        out = await _run_subprocess(self.argv(), cwd=None, env=self._env(), timeout_s=1800, stdin=None)
        if not out.ok:
            raise RuntimeError(f"sandbox container failed to start ({self.description}): {(out.stderr or out.stdout)[-800:]}")
        self.image_id = await image_id(self.name, self.docker_host, container=True)

    async def pause(self) -> None:
        """Freeze every process in the container (cgroup freezer): no CPU, memory and files kept as they are (m13 S6)."""
        out = await _run_subprocess(["docker", "pause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker pause {self.name}: {(out.stderr or out.stdout)[-300:]}")

    async def resume(self) -> None:
        out = await _run_subprocess(["docker", "unpause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker unpause {self.name}: {(out.stderr or out.stdout)[-300:]}")

    async def close(self) -> None:
        await _run_subprocess(["docker", "rm", "-f", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)


GVISOR_VERSION = "20261005.0"
GVISOR_IMAGE = f"crucible-gvisor:{GVISOR_VERSION}"
GVISOR_TOOLBOX = f"crucible-gvisor-toolbox-{GVISOR_VERSION}"


class GVisorExec(DockerExec):
    """``runsc exec`` into the gVisor sandbox running inside the outer container. Files go through the outer container
    (``docker cp``): the sandbox's root is the outer filesystem, served by the gofer with ``--file-access=shared``."""

    def __init__(self, container: str, *, cid: str, state: str, docker_host: str | None = None):
        super().__init__(container, docker_host=docker_host)
        self.cid, self.state = cid, state
        self._uids: dict[str, str] = {}

    async def _uid(self, user: str) -> str:
        if user.replace(":", "").isdigit():
            return user
        if user not in self._uids:
            out = await self._raw(f"echo $(id -u {shlex.quote(user)}):$(id -g {shlex.quote(user)})", None, None, 30, None, None)
            if not out.ok or ":" not in out.stdout.strip() or out.stdout.strip().startswith(":"):
                raise RuntimeError(f"unknown user {user!r} in the gVisor sandbox: {out.stderr[-200:]}")
            self._uids[user] = out.stdout.strip()
        return self._uids[user]

    async def _raw(self, cmd, cwd, env, timeout_s, user, stdin):
        argv = ["docker", "exec"] + (["-i"] if stdin is not None else []) + [self.container, "/opt/gvisor/runsc", "--root", self.state, "exec"]
        if cwd:
            argv += ["--cwd", cwd]
        for k, v in (env or {}).items():
            argv += ["--env", f"{k}={v}"]
        if user:
            argv += ["--user", user]
        argv += [self.cid, "bash", "-c", with_deadline(cmd, timeout_s)]
        out = await _run_subprocess(argv, cwd=None, env=self._env(), timeout_s=timeout_s + 15 if timeout_s else None, stdin=stdin)
        if deadline_hit(out.rc, out.duration_s, timeout_s):
            out.timed_out, out.rc = True, -9
        return await self._unless_lost(out)  # the outer container's loss, confirmed by its daemon (DockerExec.lost)

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None):
        u = user or self.default_user
        return await self._raw(cmd, cwd, env, timeout_s, await self._uid(u) if u else None, stdin)


_GVISOR_SPEC = r"""
import json, sys
spec = json.load(open(sys.argv[1]))
spec["root"] = {"path": "/", "readonly": False}
spec["process"]["terminal"] = False
spec["process"]["args"] = ["sleep", "infinity"]
spec["process"]["cwd"] = "/"
spec["hostname"] = "crucible-gvisor"
# Docker's default capability set (+ SYS_RESOURCE, as the Docker tier adds for the guard): root inside behaves as in runc
caps = ["CAP_CHOWN", "CAP_DAC_OVERRIDE", "CAP_FSETID", "CAP_FOWNER", "CAP_MKNOD", "CAP_NET_RAW", "CAP_SETGID", "CAP_SETUID", "CAP_SETFCAP",
        "CAP_SETPCAP", "CAP_NET_BIND_SERVICE", "CAP_SYS_CHROOT", "CAP_KILL", "CAP_AUDIT_WRITE", "CAP_SYS_RESOURCE"]
spec["process"]["capabilities"] = {k: list(caps) for k in ("bounding", "effective", "permitted", "inheritable")}
# gVisor puts its own tmpfs on /tmp; bind the outer /tmp so files pushed through the outer container are visible
spec["mounts"] = [m for m in spec["mounts"] if m["destination"] != "/tmp"]
spec["mounts"].append({"destination": "/tmp", "type": "bind", "source": "/tmp", "options": ["rbind", "rw"]})
# the runtime's own state, bundle and binaries are masked inside the sandbox (the policy must not reach them)
spec["mounts"] += [{"destination": d, "type": "tmpfs", "source": "tmpfs", "options": ["nosuid", "nodev", "mode=700"]}
                   for d in ("/run/crucible-runsc", "/run/crucible-gvisor", "/opt/gvisor")]
json.dump(spec, open(sys.argv[1], "w"))
"""


class GVisorSandbox(DockerSandbox):
    """gVisor without a daemon change (m13 audit 04, S5): the World's image runs as a privileged *outer* container with
    the pinned runsc toolbox mounted read-only (``docker/gvisor``); inside it, one long-lived gVisor sandbox
    (``--platform=systrap``, ``--network=none`` for no-network worlds) uses the outer filesystem as its root, and every
    command runs through ``runsc exec``. The outer container keeps Docker's limits (pids, memory, cpus); the World's
    processes see gVisor's kernel."""

    CID, STATE, BUNDLE = "crucible", "/run/crucible-runsc", "/run/crucible-gvisor"

    def __init__(self, sb: SandboxSpec, spec: WorldSpec, **kw):
        if sb.runtime != "runsc-in-container":
            raise ValueError("GVisorSandbox is the runsc-in-container runtime")
        super().__init__(sb, spec, **kw)
        if self.gpus:
            raise RuntimeError("the gVisor tier does not pass GPUs through (nvproxy is not configured)")
        self.exec = GVisorExec(self.name, cid=self.CID, state=self.STATE, docker_host=self.docker_host)
        self.description = f"gVisor (runsc {GVISOR_VERSION}, in a privileged container) {sb.image} on {self.docker_host or 'local daemon'}"
        self.kernel: str | None = None

    def argv(self) -> list[str]:
        a = super().argv()
        i = a.index("--entrypoint")
        return [*a[:i], "--privileged", "--volumes-from", f"{GVISOR_TOOLBOX}:ro", *a[i:]]

    async def _docker(self, *args: str, timeout_s: float = 600):
        return await _run_subprocess(["docker", *args], cwd=None, env=self._env(), timeout_s=timeout_s, stdin=None)

    async def ensure_toolbox(self) -> None:
        from pathlib import Path

        if not (await self._docker("image", "inspect", GVISOR_IMAGE, timeout_s=60)).ok:
            ctx = Path(__file__).resolve().parents[3] / "docker" / "gvisor"
            out = await self._docker("build", "-q", "-t", GVISOR_IMAGE, "--build-arg", f"GVISOR={GVISOR_VERSION}", str(ctx), timeout_s=1800)
            if not out.ok:
                raise RuntimeError(f"building the gVisor toolbox failed: {(out.stderr or out.stdout)[-600:]}")
        if not (await self._docker("container", "inspect", GVISOR_TOOLBOX, timeout_s=60)).ok:
            out = await self._docker("create", "--name", GVISOR_TOOLBOX, "--label", "crucible.sandbox=1", GVISOR_IMAGE, "true", timeout_s=120)
            if not out.ok and "already in use" not in (out.stderr or ""):
                raise RuntimeError(f"creating the gVisor toolbox container failed: {(out.stderr or out.stdout)[-400:]}")

    async def start(self) -> None:
        await self.ensure_toolbox()
        await super().start()
        outer = DockerExec(self.name, docker_host=self.docker_host)
        net = "none" if self.spec.network == "no-network" else "sandbox"
        script = (f"mkdir -p {self.BUNDLE} {self.STATE} && cd {self.BUNDLE} && /opt/gvisor/runsc spec && python3 - config.json <<'PY'\n{_GVISOR_SPEC}\nPY\n"
                  f"/opt/gvisor/runsc --root {self.STATE} --platform=systrap --network={net} --ignore-cgroups --overlay2=none --file-access=shared "
                  f"run --bundle {self.BUNDLE} --detach {self.CID}")
        out = await outer.run(script, timeout_s=180)
        if not out.ok:
            await self.close()
            raise RuntimeError(f"gVisor sandbox failed to start: {(out.stderr or out.stdout)[-800:]}")
        probe = await self.exec.run("uname -r", timeout_s=60)
        self.kernel = probe.stdout.strip()
        if "gvisor" not in self.kernel:
            await self.close()
            raise RuntimeError(f"commands do not run under gVisor (kernel {self.kernel!r})")
