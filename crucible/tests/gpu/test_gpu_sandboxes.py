"""GPU sandboxes on the lab RTX 5090 for the agentenv backend (the Harbor backend's GPU path is covered by
``test_m1_seed_worlds.py`` and ``test_ci_lab_gpu.py``): agentenv's gateway routes the tools, the World's state and
commands live in a hardened container on the GPU host (nvidia runtime), and the demoted policy uses the GPU.

Cloud GPUs (Modal H100/A100/L4 through agentenv's ``ModalSandboxProvider`` or Harbor's ``environment_type: modal``)
are wired and unit-tested in ``tests/unit/test_sandboxes.py``; a live run needs Modal credentials
(``modal token new``) and is gated by ``CRUCIBLE_MODAL=1`` below.

Run: ``CRUCIBLE_GPU=1 pytest tests/gpu/test_gpu_sandboxes.py -m gpu``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.state import Query
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world.world import World

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
pytest.importorskip("agent_env")

REPORT = Path(__file__).resolve().parents[2] / "docs" / "modules" / "reports" / "m1_gpu_sandboxes.json"


def gpu_world(sandbox: dict, gpu_type: str | None = None) -> WorldSpec:
    return WorldSpec.model_validate({
        "world": "test/gpu-sandbox",
        "backend": "agentenv",
        "backend_params": {"sandbox": sandbox},
        "network": "no-network",
        "isolation": {"policy_user": "agent"},
        "apps": [
            {"id": "pcl", "template": "common/git_repo", "state": {"adapter": "fs_git", "params": {"root": "/workspace/pcl", "exclude": ["**/build*/**"]}},
             "params": {"shell": True}},
            {"id": "gpu", "template": "common/toolchain", "state": {"adapter": "process", "params": {"cwd": "/workspace/pcl"}},
             "resources": {"gpus": 1, "gpu_arch": "sm_120", "gpu_type": gpu_type}},
        ],
        "actors": [{"id": "agent", "kind": "policy", "tools": ["pcl.*"]}],
    })


async def _exercise(w: World) -> dict:
    rec: dict = {}
    t0 = time.monotonic()
    await w.start()
    try:
        await w.reset()
        rec["start_s"] = round(time.monotonic() - t0, 1)
        r = await w.step("agent", ToolCall(tool_call_id="1", function_name="pcl.shell",
                                            arguments={"command": "id -un; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; nvcc --version | tail -1"}))
        rec["policy_shell"] = r.observation.tool_results[0].content
        q = await w.verifier_view().query("gpu", Query(op="gpu"))
        rec["verifier_gpu"] = q.value
        rec["egress"] = w.backend.egress_report
        rec["sandbox"] = w.backend.sandbox.description
    finally:
        await w.close()
    return rec


@pytest.mark.skipif(not os.environ.get("CRUCIBLE_GPU"), reason="set CRUCIBLE_GPU=1 to use the lab RTX 5090")
async def test_agentenv_gpu_sandbox_on_the_lab_5090(tmp_path: Path):
    w = World(gpu_world({"provider": "docker", "image": "task04-pcl-gpu-cuda12:v1", "gpu_host": "lab5090"}),
              store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    rec = await _exercise(w)
    assert "agent" in rec["policy_shell"] and "RTX 5090" in rec["policy_shell"] and "12.8" in rec["policy_shell"], rec["policy_shell"]
    assert rec["verifier_gpu"]["gpus"][0]["name"].endswith("RTX 5090") and rec["egress"]["isolated"]
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data["agentenv_docker_lab5090"] = rec
    REPORT.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))


@pytest.mark.skipif(not os.environ.get("CRUCIBLE_MODAL"), reason="set CRUCIBLE_MODAL=1 (and Modal credentials) to rent a cloud GPU")
async def test_agentenv_gpu_sandbox_on_modal(tmp_path: Path):
    image = os.environ.get("CRUCIBLE_MODAL_IMAGE", "nvidia/cuda:12.8.1-devel-ubuntu24.04")
    w = World(gpu_world({"provider": "modal", "image": image}, gpu_type=os.environ.get("CRUCIBLE_MODAL_GPU", "L4")),
              store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    rec = await _exercise(w)
    assert "agent" in rec["policy_shell"] and rec["verifier_gpu"]["gpus"]
    data = json.loads(REPORT.read_text()) if REPORT.exists() else {}
    data["agentenv_modal"] = rec
    REPORT.write_text(json.dumps(data, indent=1, sort_keys=True, default=str))
