"""m13 S5: the gVisor tier without a daemon change (runsc in a privileged container). A hardened agentenv World runs its
commands under gVisor's kernel (the dmesg banner, uname), keeps the guard's guarantees (demoted policy uid, App state
unreadable, egress probe isolated), and a snapshot / restore round trip works; the runtime's own state is masked."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = [pytest.mark.docker]
pytest.importorskip("agent_env")


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    import hashlib
    import subprocess

    from tests.unit.test_agentenv_sandbox import DOCKERFILE

    tag = "crucible-test-sbx:" + hashlib.sha256(DOCKERFILE.encode()).hexdigest()[:12]
    if subprocess.run(["docker", "image", "inspect", tag], capture_output=True, check=False).returncode != 0:
        d = tmp_path_factory.mktemp("sbx-img")
        (d / "Dockerfile").write_text(DOCKERFILE)
        out = subprocess.run(["docker", "build", "-q", "-t", tag, str(d)], capture_output=True, text=True, timeout=900, check=False)
        assert out.returncode == 0, out.stderr[-2000:]
    return tag


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=f"c{time.monotonic_ns()}", function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_agentenv_world_under_gvisor(image: str, tmp_path: Path):
    from tests.unit.test_agentenv_sandbox import spec

    sp = WorldSpec.model_validate({**spec(image).model_dump(), "backend_params": {"sandbox": {"provider": "docker", "image": image, "runtime": "runsc-in-container"}}})
    w = World(sp, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        b = w.backend
        assert b.capabilities.isolation == "gvisor" and "gvisor" in b.sandbox.kernel
        r = await call(w, "repo.shell", command="dmesg 2>/dev/null | head -1; uname -r; id -un; cat /crucible/mail.db 2>&1 | head -c 80; ls /run/crucible-runsc /opt/gvisor 2>&1 | head -3")
        assert r.ok and "Starting gVisor" in r.content and "gvisor" in r.content and "agent" in r.content, r.content
        assert "Permission denied" in r.content and "SQLite" not in r.content  # App state stays unreadable to the policy
        assert "runsc" not in r.content.split("Permission denied")[-1]  # the runtime's state and binaries are masked (empty tmpfs)
        assert b.egress_report["isolated"]
        assert (await call(w, "mail.send_email", to=["bob@corp.example"], subject="under gvisor", body="x")).ok
        ref = await w.snapshot()
        assert (await call(w, "repo.shell", command="echo changed >> calc.py && git -C /workspace/proj status --short")).ok
        await w.restore(ref)
        r = await call(w, "repo.shell", command="git -C /workspace/proj status --short | wc -l; tail -1 /workspace/proj/calc.py")
        assert r.ok and r.content.split()[2] == "0" and "changed" not in r.content, r.content
        assert "under gvisor" in (await call(w, "mail.list_sent")).content
    finally:
        await w.close()
