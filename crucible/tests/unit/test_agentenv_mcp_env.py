"""m13 C9: agentenv-protocol env images as Apps. agentenv's own v1 test env (``tst/data/agentenv_mcp``) loads as App
``agentenv/mcp_env``: tools from its MCP endpoint, cross-checked against its card; the seed is loaded over the data
plane; reset -> mutate -> snapshot -> mutate -> restore gives equal fingerprints (data/get); agentenv's core-protocol
probe sees data/add and data/get supported."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = [pytest.mark.docker]
pytest.importorskip("agentenv_protocol")
AE = Path(__file__).resolve().parents[3] / "agentenv-framework"
TAG = "crucible-test-aemcp:1"


@pytest.fixture(scope="module")
def image(tmp_path_factory) -> str:
    src = AE / "tst" / "data" / "agentenv_mcp"
    if not src.exists():
        pytest.skip("agentenv-framework test envs not present")
    if subprocess.run(["docker", "image", "inspect", TAG], capture_output=True, check=False).returncode != 0:
        d = tmp_path_factory.mktemp("aemcp")
        shutil.copytree(AE / "packages" / "agentenv-protocol" / "src" / "agentenv_protocol", d / "agentenv_protocol", ignore=shutil.ignore_patterns("__pycache__"))
        for f in ("server.py", "seed.json", "Dockerfile"):
            shutil.copy(src / f, d / f)
        out = subprocess.run(["docker", "build", "-q", "-t", TAG, str(d)], capture_output=True, text=True, timeout=900, check=False)
        assert out.returncode == 0, out.stderr[-1500:]
    return TAG


def spec(image: str, **params) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/agentenv-mcp-env", "network": "public",
        "apps": [{"id": "items", "template": "agentenv/mcp_env", "state": {"adapter": "agentenv_data"},
                  "params": {"image": image, "seed": [{"kind": "data", "data": {"items": ["seeded-a", "seeded-b"]}}], **params}}],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["items.*"]}],
    })


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=f"c{time.monotonic_ns()}", function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_env_image_as_app(image, tmp_path):
    w = World(spec(image), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        app = w.apps["items"]
        names = {t.name for t in await app.list_tools()}
        assert "items_add_item" in names and "list_items" in app.uncarded  # card tools served; imperative tools reported
        assert json.loads((await call(w, "items.list_items")).content)["items"] == ["seeded-a", "seeded-b"]  # seeded over data/add
        assert (await call(w, "items.items_add_item", item="x", times=2)).ok
        ref = await w.snapshot()
        assert (await call(w, "items.items_add_item", item="y")).ok
        assert len(json.loads((await call(w, "items.list_items")).content)["items"]) == 5
        await w.restore(ref)
        assert json.loads((await call(w, "items.list_items")).content)["items"] == ["seeded-a", "seeded-b", "x", "x"]
        assert (await w.snapshot()).fingerprint == ref.fingerprint
        await w.reset()  # a fresh episode: back to the seed
        assert json.loads((await call(w, "items.list_items")).content)["items"] == ["seeded-a", "seeded-b"]
        from agent_env.task_step.task_steps.env_card_validator.verify_env_core_protocol import VerifyCoreEnvironmentProtocolStep

        step = VerifyCoreEnvironmentProtocolStep.__new__(VerifyCoreEnvironmentProtocolStep)
        step.env_id = "items"
        assert await step._probe_operation(app.base_url, "data/get", {}) is True
        assert await step._probe_operation(app.base_url, "data/add", {"parts": []}) is True
    finally:
        await w.close()
    gone = await w.apps["items"]._docker("inspect", w.apps["items"].container)
    assert not gone.ok  # container removed


async def test_card_tools_only_hides_uncarded(image, tmp_path):
    w = World(spec(image, card_tools_only=True), store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    try:
        assert {t.name for t in await w.apps["items"].list_tools()} == {"items_add_item"}
    finally:
        await w.close()
