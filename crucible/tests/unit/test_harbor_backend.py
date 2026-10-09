"""M1 harbor backend on local Docker: Harbor builds and drives the sandbox; World snapshot/restore
across two fresh containers; the policy's shell runs inside the sandbox with no network."""

from __future__ import annotations

from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = pytest.mark.docker
pytest.importorskip("harbor")

DOCKERFILE = """FROM python:3.12-slim
RUN apt-get update -qq && apt-get install -y -qq git curl >/dev/null && rm -rf /var/lib/apt/lists/*
RUN git config --global user.email t@t && git config --global user.name t && git config --global --add safe.directory '*'
RUN mkdir -p /workspace/proj && cd /workspace/proj && printf 'def add(a, b):\\n    return a - b\\n' > calc.py && \\
    printf 'build/\\n' > .gitignore && git init -q -b main && git add -A && git commit -qm base && git tag base
WORKDIR /workspace
"""


@pytest.fixture(scope="module")
def env_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("harbor-env")
    (d / "Dockerfile").write_text(DOCKERFILE)
    return d


def spec(env_dir: Path) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/harbor-repo",
        "backend": "harbor",
        "backend_params": {"environment_dir": str(env_dir)},
        "network": "no-network",
        "apps": [
            {"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs_git", "params": {"root": "/workspace/proj", "exclude": ["**/__pycache__/**"]}}, "params": {"shell": True}},
            {"id": "tool", "template": "common/toolchain", "state": {"adapter": "process", "params": {"cwd": "/workspace/proj"}}},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}],
    })


async def test_harbor_world_snapshot_restore_across_containers(env_dir: Path, tmp_path: Path):
    s = spec(env_dir)
    store = Store(tmp_path / "store")
    w1 = World(s, store=store, workdir=tmp_path / "w1")
    await w1.start()
    try:
        assert w1.backend.capabilities.isolation == "container" and w1.backend.capabilities.network_enforced
        await w1.reset()
        r = await w1.step("agent", ToolCall(tool_call_id="1", function_name="repo.edit_file",
                                            arguments={"path": "calc.py", "old_string": "a - b", "new_string": "a + b"}))
        assert r.observation.tool_results[0].ok, r.observation.tool_results[0].content
        r = await w1.step("agent", ToolCall(tool_call_id="2", function_name="repo.shell",
                                            arguments={"command": "mkdir -p build && echo x > build/o && python3 -c 'import calc; print(calc.add(2,3))'"}))
        assert "5" in r.observation.tool_results[0].content
        r = await w1.step("agent", ToolCall(tool_call_id="3", function_name="repo.shell", arguments={"command": "curl -sS -m 5 https://example.com >/dev/null; echo rc=$?"}))
        assert "rc=0" not in r.observation.tool_results[0].content  # no network inside the sandbox
        r = await w1.step("agent", ToolCall(tool_call_id="4", function_name="repo.git_diff", arguments={}))
        assert "+    return a + b" in r.observation.tool_results[0].content
        ref = await w1.snapshot()
        assert ref.apps["repo"].meta["files"] == 2  # build/ (gitignored) and __pycache__ (excluded) are derived state
    finally:
        await w1.close()
    w2 = World(s, store=store, workdir=tmp_path / "w2")
    await w2.start()
    try:
        fresh = await w2.snapshot()
        assert fresh.apps["repo"].fingerprint != ref.apps["repo"].fingerprint
        await w2.restore(ref)
        assert (await w2.snapshot()).fingerprint == ref.fingerprint
        out = await w2.verifier_view().query("tool", {"op": "run", "args": {"cmd": "python3 -c 'import calc; print(calc.add(2,3))'"}})
        assert out.stdout.strip() == "5"
    finally:
        await w2.close()
