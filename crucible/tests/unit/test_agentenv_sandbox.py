"""The agentenv backend with a container sandbox: the real agentenv gateway (roles, ACL) routes every tool call,
while state and commands live in a hardened Docker container (the same guard as the Harbor backend).

What must hold: the policy works through the gateway as the demoted ``agent`` uid; App-owned state (the
mailbox's SQLite file) is unreadable to its shell yet fully usable through the App's tools; the gateway's host
port is unreachable from the sandbox; an MCP client cannot impersonate another actor; reset restores and cleans."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.exec import LocalExec
from crucible.world.world import World

pytestmark = [pytest.mark.docker]
pytest.importorskip("agent_env")

DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq git >/dev/null && rm -rf /var/lib/apt/lists/*
RUN mkdir -p /workspace/proj && cd /workspace/proj && printf 'def add(a, b):\\n    return a - b\\n' > calc.py \\
 && git init -q -b main && git add -A && git -c user.email=t@t -c user.name=t commit -qm base && git tag base
"""


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    import subprocess

    tag = "crucible-test-sbx:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        d = tmp_path_factory.mktemp("sbx-img")
        (d / "Dockerfile").write_text(DOCKERFILE)
        out = subprocess.run(["docker", "build", "-q", "-t", tag, str(d)], capture_output=True, text=True, timeout=900, check=False)
        assert out.returncode == 0, out.stderr[-2000:]
    return tag


def spec(image: str, sandbox: dict | None = None, apps_extra: list | None = None) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/agentenv-sandbox",
        "backend": "agentenv",
        "backend_params": {"sandbox": {"provider": "docker", "image": image, **(sandbox or {})}},
        "network": "no-network",
        "isolation": {"policy_user": "agent"},
        "apps": [
            {"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs_git", "params": {"root": "/workspace/proj"}}, "params": {"shell": True}},
            {"id": "mail", "template": "common/mailbox", "state": {"adapter": "sql", "params": {"path": "mail.db"}}},
            *(apps_extra or []),
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*", "mail.*", *[f"{a['id']}.*" for a in apps_extra or []]]}],
    })


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=f"c{time.monotonic_ns()}", function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_hardened_agentenv_world_in_a_docker_sandbox(image: str, tmp_path: Path):
    w = World(spec(image), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        b = w.backend
        assert b.capabilities.isolation == "container" and b.policy_uid
        r = await call(w, "repo.shell", command="id -un; cat /crucible/mail.db 2>&1 | head -c 200; ls /crucible 2>&1")
        assert r.ok and "agent" in r.content and "Permission denied" in r.content and "SQLite" not in r.content, r.content
        r = await call(w, "mail.send_email", to=["bob@corp.example"], subject="hello", body="sent through the gateway")
        assert r.ok, r.content
        assert "hello" in (await call(w, "mail.list_sent")).content  # App-owned state works through the App's tools
        assert b.egress_report["isolated"] and any(str(b.gateway_port) in t for t in b.egress_report["tested"])
        # an MCP client that names an actor without the World's signature acts as the default actor, not as "boss"
        assert b.verify_actor("boss") is None and b.verify_actor("boss:" + "0" * 32) is None
        assert b.verify_actor(b.sign_actor("boss")) == "boss"
        await call(w, "repo.shell", command="echo '# x' >> calc.py; touch /tmp/stash")
        s1 = await w.snapshot()
        await w.reset()
        assert w.initial is not None and (await w.snapshot()).fingerprint == w.initial.fingerprint != s1.fingerprint
        r = await call(w, "repo.shell", command="ls /tmp; grep -c x calc.py; echo '# ok' >> calc.py && echo writable")
        assert "stash" not in r.content and "writable" in r.content
        name = b.sandbox.name
    finally:
        await w.close()
    assert not (await LocalExec().run(f"docker ps -q --filter name={name}")).stdout.strip()


async def test_sandboxed_agentenv_refuses_host_side_servers(image: str, tmp_path: Path):
    s = spec(image, apps_extra=[{"id": "srv", "template": "common/mcp_server", "state": {"adapter": "json_doc", "params": {"path": "srv.json"}},
                                 "params": {"command": ["true"]}}])
    w = World(s, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    with pytest.raises(Exception, match="outside the sandbox"):
        await w.start()


async def test_pause_resume_keeps_state_and_virtual_time(image: str, tmp_path: Path):
    """m13 S6: a sandboxed World paused while the trainer holds the GPU keeps its state fingerprints and its virtual time."""
    import asyncio
    from datetime import timedelta

    from crucible.world.clock import parse_time

    sp = WorldSpec.model_validate({**spec(image).model_dump(), "clock": {"mode": "wall", "t0": "2026-03-02T09:00:00Z", "rate": 600}})
    w = World(sp, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        assert w.backend.capabilities.pausable
        assert (await call(w, "mail.send_email", to=["bob@corp.example"], subject="before pause", body="x")).ok
        fp0 = await w.verifier_view().fingerprints()
        await w.pause()
        t_paused = w.clock.now()
        await asyncio.sleep(3)  # 30 virtual minutes at this rate, if the pause leaked
        assert w.clock.now() == t_paused
        await w.resume()
        assert await w.verifier_view().fingerprints() == fp0
        gw = await w.backend.gateway_time()
        assert gw is not None and abs(parse_time(gw) - t_paused) < timedelta(minutes=10), (gw, t_paused)  # 1 real s of slack; a leak is 30 min
        assert (await call(w, "mail.list_sent")).ok  # the World works after resume
    finally:
        await w.close()
