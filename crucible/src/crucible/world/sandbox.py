"""Container sandboxes a backend can put a World's state and commands into (no framework imports here).

``DockerSandbox`` runs one long-lived container from an image on this machine's Docker daemon or a remote one
(``gpu_host`` from ``crucible.toml``, e.g. the lab RTX 5090 over SSH), with the same hardening as the Harbor
overlay: ``--init``, no network for no-network worlds, a pids limit, ``SYS_RESOURCE`` (so the guard can pin the
policy's OOM score) and GPUs through the nvidia runtime or ``--gpus``. Cloud sandboxes (Modal and the other
agentenv providers) live in ``crucible.interop.agentenv.sandbox`` behind the same small interface.

**Lifecycle (M10 lab audit R-6, ``crucible.world.lifecycle``).** The keeper is never ``sleep infinity``: the container names its
owner and expiry (``crucible.owner``, ``crucible.expires``, and inside a Runner attempt ``crucible.run``, ``crucible.attempt``,
``crucible.lease``) and ends by itself, as ``sleep <ttl>`` or, where a holder renews it (a Runner attempt, the warm pools), the RLM
kernel's life loop. ``start`` shields its ``docker run`` from cancellation and removes what it made on a cancel or a failure;
``close`` is shielded, idempotent and rc-checked, records its outcome in the scope the sandbox started in, and never raises (a
removal that failed is retried by the attempt's reclaim and reported in ``teardown.json``, never turned into a lost grade).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import shlex
import time
import uuid
from typing import Protocol

from crucible.core.schema.world import SandboxSpec, WorldSpec
from crucible.world import lifecycle
from crucible.world.exec import DockerExec, Exec, _run_subprocess, deadline_hit, with_deadline

logger = logging.getLogger(__name__)


class Sandbox(Protocol):
    exec: Exec
    gpus: int
    description: str

    async def start(self) -> None: ...

    async def close(self) -> object: ...  # DockerSandbox returns (removed, why); other sandboxes return None


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


class StaleImage(RuntimeError):
    """The image a World started in is not the one its spec pins (M1 audit R-14): ``crucible:stale_image``, never retried."""

    failure_kind = "crucible:stale_image"


def check_pinned(pinned: str | None, current: str | None, what: str) -> None:
    """Refuse an image other than the pinned ``image_digest`` (``crucible world pin`` prints it): a tag re-pushed or rebuilt
    under the same name otherwise changes what every task runs, unseen (ARCHITECTURE §7 rule 4). No pin: nothing to check."""
    if pinned and current != pinned:
        raise StaleImage(f"{what} is image {current or '(identity unreadable)'}, but the spec pins {pinned}: the tag was re-pushed or "
                         "rebuilt; check the image, then pin it again with `crucible world pin`")


def check_image(state: dict, current: str | None) -> None:
    """A WorldRef restores only into the image it was taken in (DESIGN B.5: stale images silently change behavior)."""
    taken = state.get("image")
    if taken and current and taken != current:
        from crucible.world.state.base import SnapshotMismatch

        raise SnapshotMismatch(f"WorldRef was taken in image {taken[:19]}, this world runs {current[:19]}")


class DockerSandbox:
    """One container. ``scope`` (default: the one the calling task runs in, ``lifecycle.current()``) says who owns it and records
    what became of it; ``renewable`` (default: the scope's) picks the kernel's life loop over ``sleep <ttl>``; ``lifetime_s``
    (default: the scope's ``ttl_s``) is its life from ``start`` (a pool keeps containers longer than one attempt needs)."""

    def __init__(self, sb: SandboxSpec, spec: WorldSpec, *, pids_limit: int | None = None, memory: str | None = None,
                 scope: lifecycle.Scope | None = None, renewable: bool | None = None, lifetime_s: int | None = None):
        self.sb, self.spec = sb, spec
        self.gpus, self.gpu_type = gpu_request(spec)
        self.docker_host = resolve_docker_host(sb)
        self.name = f"crucible-sbx-{spec.world.replace('/', '-')[:40]}-{uuid.uuid4().hex[:8]}"
        self.pids_limit, self.memory = pids_limit, memory
        self.scope = scope if scope is not None else lifecycle.current()
        self.renewable = self.scope.renewable if renewable is None else renewable
        self.lifetime_s = lifetime_s
        self.deadline: float | None = None  # epoch seconds at which the keeper ends by itself (set by ``argv``, pushed out by ``renew``)
        self.expires: int | None = None  # the ``crucible.expires`` label
        self._removed = False
        self._closing: asyncio.Future | None = None
        self._renewer: asyncio.Future | None = None  # the keep-alive of a sandbox outside a Runner (``start``)
        self.exec: Exec = DockerExec(self.name, docker_host=self.docker_host)
        self.description = f"docker {sb.image} on {self.docker_host or 'local daemon'}" + (f", {self.gpus} GPU(s)" if self.gpus else "")

    def argv(self) -> list[str]:
        run_args, cmd, deadline, expires = self.scope.keeper(renewable=self.renewable, lifetime_s=self.lifetime_s)
        self.deadline, self.expires = float(deadline), expires
        a = ["docker", "run", "-d", "--rm", "--init", "--name", self.name, "--label", f"crucible.world={self.spec.world}", "--label", "crucible.sandbox=1",
             *self.scope.labels(expires=expires, renewable=self.renewable)]
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
        if mem:  # swap capped at the memory limit: without it the container may use twice its limit, swapping (M1 audit R-26)
            a += ["--memory", mem, "--memory-swap", mem]
        return [*a, *run_args, "--entrypoint", "", self.sb.image, *cmd]

    def _env(self) -> dict[str, str] | None:
        return {"DOCKER_HOST": self.docker_host} if self.docker_host else None

    image_id: str | None = None

    async def start(self) -> None:
        """M10 lab audit R-6 (the RLM kernel's B4 pattern): the container is recorded in the scope *before* ``docker run`` (every
        exit path knows its name), the run is shielded, and a cancel or a failure removes whatever it made. A cancelled run is given
        ``START_GRACE_S`` to finish (an image pull); one that finishes later is removed by a callback."""
        argv = self.argv()
        if isinstance(self.exec, DockerExec):  # its end by itself is infrastructure, not the policy's (M1 audit R-1)
            self.exec.expires_at = self.deadline
        self.scope.started(self.name, self.docker_host, renewable=self.renewable, deadline=self.deadline or 0.0, renew=self.renew, remove=self._rm)
        run = asyncio.ensure_future(_run_subprocess(argv, cwd=None, env=self._env(), timeout_s=1800, stdin=None))
        try:
            out = await asyncio.shield(run)
        except asyncio.CancelledError:  # cancelled mid-``docker run``: let it finish, then remove what it made
            self.scope.raced = True
            await lifecycle.finish(run, lifecycle.START_GRACE_S)
            if not run.done():
                run.add_done_callback(self._remove_late)
            with contextlib.suppress(Exception):
                await self.close()
            raise
        except BaseException:
            with contextlib.suppress(Exception):
                await self.close()
            raise
        if not out.ok:
            with contextlib.suppress(Exception):
                await self.close()  # a failed run may still have left a container behind
            raise RuntimeError(f"sandbox container failed to start ({self.description}): {(out.stderr or out.stdout)[-800:]}")
        try:
            self.image_id = await image_id(self.name, self.docker_host, container=True)
            check_pinned(self.sb.image_digest, self.image_id, self.sb.image)  # StaleImage (M1 audit R-14)
        except BaseException:  # review round 1: the container is up, and a cancel (or an OSError) in the inspect used to leave it so
            with contextlib.suppress(Exception):
                await self.close()
            raise
        if self.renewable and not self.scope.record:
            # nobody records or renews this container (the module default scope: a Runner attempt's scope renews its own, and a pool's
            # containers are renewed by their borrower): the sandbox keeps its own life while it is open (review round 1)
            self._renewer = lifecycle.hold(self._renew_loop())

    async def _renew_loop(self) -> None:
        """Outside a Runner (``Scope.record`` False) a renewable sandbox renews itself, as ``Scope._renew_loop`` renews an attempt's:
        once its remaining life is shorter than the renew window its deadline moves out to ``now + ttl``. Ends with ``close`` (it is
        cancelled there), or when a renewal fails (the container is gone). A crashed process stops renewing: the keeper then ends at the
        last renewed deadline, and the sweep reaps it past its label. Never raises."""
        ttl = float(self.lifetime_s or self.scope.ttl_s)
        try:
            while True:
                await asyncio.sleep(min(lifecycle.RENEW_EVERY_S, ttl / 4))
                if (self.deadline or 0.0) - time.time() >= min(lifecycle.RENEW_BEFORE_S, ttl / 2):
                    continue
                if not await self.renew(ttl):
                    return
        except asyncio.CancelledError:
            raise
        except Exception:  # noqa: BLE001 -- a failed renewal ends the keep-alive, never the episode
            return

    def _remove_late(self, run: asyncio.Future) -> None:
        """Done-callback of a ``docker run`` that outlived its cancelled start: the container exists now, remove it (by name, whatever
        ``close`` found earlier). One that finishes after the loop has stopped is left to its TTL and the next sweep."""
        if not run.cancelled():
            run.exception()  # retrieved: a failed run left nothing to remove, and is nobody's error any more
        with contextlib.suppress(RuntimeError):
            lifecycle.hold(self._rm())

    async def renew(self, lifetime_s: float | None = None) -> bool:
        """Push the keeper's deadline out to ``now + lifetime_s`` (default: the scope's ttl), through the deadline file only root can write
        (``docker exec --user 0``); never shortens it. A fixed ``sleep <ttl>`` keeper cannot be renewed: ``True`` (nothing to do).
        ``False`` when the exec failed (the container is gone)."""
        if not self.renewable:
            return True
        d = int(time.time() + (lifetime_s or self.scope.ttl_s))
        try:
            out = await _run_subprocess(["docker", "exec", "--user", "0", self.name, "sh", "-c", f"echo {d} > {lifecycle.CTL_DIR}/deadline"],
                                        cwd=None, env=self._env(), timeout_s=120, stdin=None)
        except OSError:
            return False
        if not out.ok:
            return False
        self.deadline = max(self.deadline or 0.0, float(d))
        if isinstance(self.exec, DockerExec):
            self.exec.expires_at = self.deadline
        return True

    async def pause(self) -> None:
        """Freeze every process in the container (cgroup freezer): no CPU, memory and files kept as they are (m13 S6). A renewable
        keeper is renewed first (best effort), so a pause shorter than the ttl cannot end the container on thaw."""
        if self.renewable:
            with contextlib.suppress(Exception):
                await self.renew()
        out = await _run_subprocess(["docker", "pause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker pause {self.name}: {(out.stderr or out.stdout)[-300:]}")

    async def resume(self) -> None:
        out = await _run_subprocess(["docker", "unpause", self.name], cwd=None, env=self._env(), timeout_s=120, stdin=None)
        if not out.ok:
            raise RuntimeError(f"docker unpause {self.name}: {(out.stderr or out.stdout)[-300:]}")
        if self.renewable:
            with contextlib.suppress(Exception):
                await self.renew()

    async def _rm(self) -> tuple[bool, str]:
        return await lifecycle.remove(self.name, self.docker_host)

    async def _close_once(self) -> tuple[bool, str]:
        r, self._renewer = self._renewer, None
        if r is not None:
            r.cancel()  # not awaited: a renewal ``docker exec`` that is cancelled is killed with its session, and the removal must not wait on it
        try:
            ok, err = await self._rm()
        except Exception as e:  # noqa: BLE001 -- close never raises
            ok, err = False, f"{type(e).__name__}: {e}"
        self.scope.closed(self.name, ok=ok, error=err)
        if ok:
            self._removed = True
        else:
            logger.warning("sandbox %s: docker rm -f failed (%s); the attempt's reclaim retries it and teardown.json reports it", self.name, err[-300:])
        return ok, err

    async def close(self) -> tuple[bool, str]:
        """Remove the container: shielded from cancellation, idempotent (a second caller awaits the first removal), bounded, and
        rc-checked (``lifecycle.remove``). The outcome is recorded in the scope the sandbox started in. It never raises: a failed
        removal returns ``(False, why)``, stays recorded (the attempt's reclaim retries it, ``teardown.json`` reports it), and must
        not turn a grade that was already computed into a lost one (the agentenv backend closes it in a ``finally``)."""
        if self._removed:
            return True, ""
        if self._closing is None or (self._closing.done() and not self._removed):
            self._closing = lifecycle.hold(self._close_once())
        return await lifecycle.shielded(self._closing, lifecycle.CLOSE_TIMEOUT_S + 15)


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
    processes see gVisor's kernel.

    Lifecycle (M10 lab audit R-6): the *outer* container is the one that is labelled and bounded (``DockerSandbox.argv``: owner,
    expiry, ``sleep <ttl>`` or the life loop); the inner sandbox's ``sleep infinity`` lives and dies with it. A failed removal in
    ``start`` is recorded by ``close`` and never replaces the gVisor error. The toolbox container (``GVISOR_TOOLBOX``) carries no
    owner label, so no sweep ever touches it."""

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
