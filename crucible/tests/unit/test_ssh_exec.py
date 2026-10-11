"""m13 S4: the ssh_exec backend (host processes over SSH, for Docker-less GPU hosts). Against a throwaway sshd
container: a World's Apps work over SSH (tool calls, snapshot/restore round trip), rlimits bound memory and processes,
the capability says isolation none, and a policy shell or a no-network World is refused unless opted in. The Runpod
GPU acceptance (sm_89 preflight, verifier score = manual run) needs a running pod."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.exec import SSHExec
from crucible.world.world import World

pytestmark = [pytest.mark.docker]

DOCKERFILE = """FROM debian:bookworm-slim
RUN apt-get update -qq && apt-get install -y -qq openssh-server python3 git procps >/dev/null && rm -rf /var/lib/apt/lists/* \\
 && mkdir -p /run/sshd /root/.ssh && chmod 700 /root/.ssh && useradd -m -s /bin/bash worker && mkdir -p /home/worker/.ssh
CMD ["/usr/sbin/sshd", "-D", "-e"]
"""


@pytest.fixture(scope="module")
def sshd(tmp_path_factory):
    import hashlib

    d = tmp_path_factory.mktemp("sshd")
    tag = "crucible-test-sshd:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        (d / "Dockerfile").write_text(DOCKERFILE)
        out = subprocess.run(["docker", "build", "-q", "-t", tag, str(d)], capture_output=True, text=True, timeout=900, check=False)
        assert out.returncode == 0, out.stderr[-1500:]
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / "key")], check=True)
    name = f"crucible-test-sshd-{int(time.time() * 1000)}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, "-p", "127.0.0.1::22", tag], check=True, capture_output=True)
    try:
        pub = (d / "key.pub").read_text()
        install = f"echo '{pub.strip()}' > /home/worker/.ssh/authorized_keys && chown -R worker /home/worker/.ssh && chmod 600 /home/worker/.ssh/authorized_keys"
        subprocess.run(["docker", "exec", name, "bash", "-c", install], check=True)
        port = int(subprocess.run(["docker", "port", name, "22"], capture_output=True, text=True, check=True).stdout.strip().rsplit(":", 1)[1])
        for _ in range(50):
            if subprocess.run(["ssh", "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new", "-o", f"UserKnownHostsFile={d / 'kh'}", "-i", str(d / "key"),
                               "-p", str(port), "worker@127.0.0.1", "true"], capture_output=True, check=False).returncode == 0:
                break
            time.sleep(0.2)
        yield {"ssh": "worker@127.0.0.1", "port": port, "key": str(d / "key"), "known_hosts": str(d / "kh")}
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def _spec(sshd, **params) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/ssh-exec", "backend": "ssh_exec", "network": "public",
        "backend_params": {**sshd, **params},
        "apps": [{"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["mail.*"]}],
    })


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=f"c{time.monotonic_ns()}", function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_world_over_ssh(sshd, tmp_path: Path):
    w = World(_spec(sshd), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        assert w.backend.capabilities.isolation == "none" and w.backend.capabilities.gpus
        assert (await call(w, "mail.send_email", to=["bob@corp.example"], subject="over ssh", body="x")).ok
        ref = await w.snapshot()
        assert (await call(w, "mail.send_email", to=["bob@corp.example"], subject="after snapshot", body="y")).ok
        await w.restore(ref)
        sent = (await call(w, "mail.list_sent")).content
        assert "over ssh" in sent and "after snapshot" not in sent
        wd = w.backend.base_path(w.spec.apps[0])
        assert wd.startswith("/home/worker/crucible_worlds/")
    finally:
        await w.close()
    assert "No such file" in (await SSHExec(sshd["ssh"], port=sshd["port"], key=sshd["key"], known_hosts=sshd["known_hosts"]).run(f"ls {wd} 2>&1")).stdout


async def test_rlimits_bound_memory_and_processes(sshd):
    ex = SSHExec(sshd["ssh"], port=sshd["port"], key=sshd["key"], known_hosts=sshd["known_hosts"], rlimits={"as_mb": 200, "nproc": 64})
    out = await ex.run("python3 -c 'b = bytearray(400 * 1024 * 1024); print(\"allocated\")' 2>&1 | tail -1; ulimit -u; ulimit -v", timeout_s=60)
    assert "MemoryError" in out.stdout and "allocated" not in out.stdout and out.stdout.split()[-2:] == ["64", str(200 * 1024)], out.stdout
    with pytest.raises(Exception, match="cannot switch users"):
        await ex.run("id", user="root")
    await ex.close()


def test_refuses_policy_shell_and_no_network(sshd):
    from crucible.world.backends.ssh import SSHExecBackend

    shell = WorldSpec.model_validate({**_spec(sshd).model_dump(), "apps": [{"id": "repo", "template": "common/git_repo", "params": {"shell": True},
                                                                            "state": {"adapter": "fs_git", "params": {"root": "proj"}}}],
                                      "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})
    b = SSHExecBackend(shell, shell.backend_params, workdir=Path("/tmp"))
    assert any("policy a shell" in p for p in b.validate())
    ok = SSHExecBackend(shell, {**shell.backend_params, "allow_untrusted_shell": True}, workdir=Path("/tmp"))
    assert not ok.validate()
    nonet = WorldSpec.model_validate({**_spec(sshd).model_dump(), "network": "no-network"})
    assert any("cannot enforce no-network" in p for p in SSHExecBackend(nonet, nonet.backend_params, workdir=Path("/tmp")).validate())
