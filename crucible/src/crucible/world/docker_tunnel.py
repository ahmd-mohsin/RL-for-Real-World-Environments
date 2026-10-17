"""One SSH tunnel per remote Docker host, shared by every Docker CLI call of this process.

``DOCKER_HOST=ssh://user@host`` makes the Docker CLI open a new SSH connection (``docker system dial-stdio``) for every
command. Under concurrency (dozens of sandboxes starting, a VM pool resetting, many ``docker exec`` probes) sshd's
``MaxStartups`` throttling drops new unauthenticated connections, and the CLI hangs on them (found live on the lab: 5 of
24 concurrent ``docker run`` calls stuck for 11+ minutes with no error).

``local_docker_host(host)`` starts, once per host, ``ssh -N -L <local.sock>:/var/run/docker.sock host`` (keepalives,
``ExitOnForwardFailure``) and returns ``unix://<local.sock>``. Every CLI call then talks to the remote daemon's API over
that one authenticated connection: no handshake per command and no throttling. A tunnel that dies is restarted on the
next call; the tunnel processes end with this process. ``CRUCIBLE_DOCKER_SSH_TUNNEL=0`` turns it off (plain ``ssh://``).
The remote user must be able to use the remote socket (the ``docker`` group), which ``ssh://`` needs anyway.
"""

from __future__ import annotations

import atexit
import contextlib
import hashlib
import os
import subprocess
import tempfile
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

_LOCK = threading.Lock()
_TUNNELS: dict[str, tuple[subprocess.Popen, str]] = {}


def enabled() -> bool:
    from crucible.core import env as knobs

    return bool(knobs.get("CRUCIBLE_DOCKER_SSH_TUNNEL"))


def _stop_all() -> None:
    for p, sock in list(_TUNNELS.values()):
        with contextlib.suppress(Exception):
            p.terminate()
            p.wait(timeout=5)
        with contextlib.suppress(OSError):
            os.unlink(sock)
    _TUNNELS.clear()


atexit.register(_stop_all)


def local_docker_host(host: str, *, remote_socket: str = "/var/run/docker.sock", timeout_s: float = 20.0) -> str:
    """``unix://`` path of a live tunnel to ``host`` (an ``ssh://user@host[:port]`` DOCKER_HOST)."""
    with _LOCK:
        cur = _TUNNELS.get(host)
        if cur is not None and cur[0].poll() is None and Path(cur[1]).exists():
            return f"unix://{cur[1]}"
        if cur is not None:  # dead: restart
            with contextlib.suppress(OSError):
                os.unlink(cur[1])
        u = urlparse(host)
        if u.scheme != "ssh" or not u.hostname:
            raise ValueError(f"not an ssh:// docker host: {host!r}")
        tag = hashlib.sha256(host.encode()).hexdigest()[:10]
        sock = str(Path(tempfile.gettempdir()) / f"crucible-docker-{tag}-{os.getpid()}.sock")
        if len(sock) > 100:  # sun_path holds 104 bytes on macOS (108 on Linux): a long TMPDIR falls back to /tmp
            sock = f"/tmp/crucible-docker-{tag}-{os.getpid()}.sock"
        with contextlib.suppress(OSError):
            os.unlink(sock)
        target = f"{u.username}@{u.hostname}" if u.username else u.hostname
        argv = ["ssh", "-N", "-T", "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "StreamLocalBindUnlink=yes", "-o", "ConnectTimeout=20", "-L", f"{sock}:{remote_socket}"]
        if u.port:
            argv += ["-p", str(u.port)]
        p = subprocess.Popen([*argv, target], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, start_new_session=True)
        t0 = time.monotonic()
        while not Path(sock).exists():
            if p.poll() is not None:
                err = p.stderr.read().decode(errors="replace") if p.stderr else ""
                raise RuntimeError(f"SSH tunnel to {host} failed: {err[-400:]}")
            if time.monotonic() - t0 > timeout_s:
                p.terminate()
                raise RuntimeError(f"SSH tunnel to {host} did not come up in {timeout_s}s")
            time.sleep(0.05)
        _TUNNELS[host] = (p, sock)
        return f"unix://{sock}"


def rewrite_env(argv: list[str], env: dict[str, str]) -> dict[str, str]:
    """For a Docker CLI call with an ``ssh://`` DOCKER_HOST: the same environment pointed at the shared tunnel."""
    host = env.get("DOCKER_HOST", "")
    if not argv or Path(argv[0]).name != "docker" or not host.startswith("ssh://") or not enabled():
        return env
    return {**env, "DOCKER_HOST": local_docker_host(host)}
