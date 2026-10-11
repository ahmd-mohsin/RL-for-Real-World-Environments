"""M1 item 5: the per-uid egress firewall (guard ops ``firewall`` / ``firewall_lift``, ported from Karotte confinement).

In a real container with CAP_NET_ADMIN: after the rules, the policy uid reaches localhost and the allowlisted address
but not the internet, the cloud metadata address, or an App server's blocked localhost port; root is unaffected; the
guard's egress probe agrees; lifting the rules restores the uid's network. Needs a local Docker daemon.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import shlex
import shutil
import subprocess
import uuid

import pytest

from crucible.world.exec import DockerExec
from crucible.world.guard import Guard

pytestmark = pytest.mark.docker

DOCKERFILE = "FROM python:3.12-slim\nRUN apt-get update && apt-get install -y --no-install-recommends iptables && rm -rf /var/lib/apt/lists/*\n"
PROBE = """
import json, os, socket, sys
uid, targets = int(sys.argv[1]), json.loads(sys.argv[2])
if uid:
    os.setgroups([]); os.setgid(uid); os.setuid(uid)
out = {}
for host, port in targets:
    try:
        socket.create_connection((host, port), timeout=3).close(); out[f"{host}:{port}"] = True
    except OSError:
        out[f"{host}:{port}"] = False
print(json.dumps(out))
"""


def _docker_ok() -> bool:
    if not shutil.which("docker"):
        return False
    return subprocess.run(["docker", "info"], capture_output=True, timeout=30, check=False).returncode == 0


@pytest.fixture(scope="module")
def image():
    if not _docker_ok():
        pytest.skip("no local Docker daemon")
    tag = "crucible-test-fw:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        r = subprocess.run(["docker", "build", "-t", tag, "-"], input=DOCKERFILE.encode(), capture_output=True, timeout=900, check=False)
        if r.returncode != 0:
            pytest.skip(f"could not build the test image: {r.stderr.decode()[-300:]}")
    return tag


@pytest.fixture
def container(image):
    name = f"crucible-test-fw-{uuid.uuid4().hex[:8]}"
    r = subprocess.run(["docker", "run", "-d", "--rm", "--init", "--cap-add", "NET_ADMIN", "--name", name, "--entrypoint", "", image, "sleep", "infinity"],
                       capture_output=True, timeout=120, check=False)
    if r.returncode != 0:
        pytest.skip(f"container did not start: {r.stderr.decode()[-300:]}")
    yield name
    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=60, check=False)


async def _probe(ex: DockerExec, uid: int, targets: list[list]) -> dict[str, bool]:
    out = await ex.run(f"python3 -c {shlex.quote(PROBE)} {uid} {shlex.quote(json.dumps(targets))}", timeout_s=120)
    assert out.ok, out.stderr
    return json.loads(out.stdout.strip().splitlines()[-1])


async def test_policy_uid_firewall(container):
    ex = DockerExec(container)
    g = Guard(ex)
    await g.install()
    uid = int((await g.setup(user="agent"))["uid"])
    await ex.run("nohup python3 -m http.server 8123 --bind 127.0.0.1 >/dev/null 2>&1 & nohup python3 -m http.server 8124 --bind 127.0.0.1 >/dev/null 2>&1 &")
    await asyncio.sleep(1.5)
    targets = [["127.0.0.1", 8123], ["127.0.0.1", 8124], ["1.1.1.1", 80], ["1.0.0.1", 80]]
    before = await _probe(ex, uid, targets)
    assert before["127.0.0.1:8123"] and before["127.0.0.1:8124"]
    online = before["1.1.1.1:80"] and before["1.0.0.1:80"]

    rep = await g.call("firewall", {"uid": uid, "mode": "allowlist", "allowed": ["1.0.0.1"], "blocked_ports": [8123]})
    assert rep["took"], rep
    assert "1.0.0.1" in rep["allowed_ips"] and rep["blocked_ports"] == [8123] and rep["own"]

    after = await _probe(ex, uid, targets)
    assert after["127.0.0.1:8124"]  # localhost: the policy's own servers
    assert not after["127.0.0.1:8123"]  # an App server's port: tools only
    assert not after["1.1.1.1:80"]  # the internet
    if online:
        assert after["1.0.0.1:80"]  # the allowlisted address still works
    root = await _probe(ex, 0, targets)
    assert root["127.0.0.1:8123"] and (root["1.1.1.1:80"] or not online)  # root (the Apps, the verifier) is unaffected

    probe = await g.call("egress", {"uid": uid, "allowed": ["1.0.0.1"], "timeout": 2})
    assert probe["isolated"], probe

    lifted = await g.call("firewall_lift", {"uid": uid})
    assert lifted["clean"], lifted
    again = await _probe(ex, uid, targets)
    assert again["127.0.0.1:8123"] and (again["1.1.1.1:80"] or not online)


async def test_ports_mode_and_deny(container):
    ex = DockerExec(container)
    g = Guard(ex)
    await g.install()
    uid = int((await g.setup(user="agent"))["uid"])
    await ex.run("nohup python3 -m http.server 8125 --bind 127.0.0.1 >/dev/null 2>&1 &")
    await asyncio.sleep(1.5)
    rep = await g.call("firewall", {"uid": uid, "mode": "ports", "blocked_ports": [8125]})
    assert rep["took"] and rep["mode"] == "ports"
    got = await _probe(ex, uid, [["127.0.0.1", 8125], ["1.1.1.1", 80]])
    assert not got["127.0.0.1:8125"]  # only the port is closed; a public world keeps its internet
    await g.call("firewall_lift", {"uid": uid})
    rep = await g.call("firewall", {"uid": uid, "mode": "deny"})
    got = await _probe(ex, uid, [["127.0.0.1", 8125], ["1.1.1.1", 80]])
    assert rep["took"] and got["127.0.0.1:8125"] and not got["1.1.1.1:80"]
