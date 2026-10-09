"""Hidden-check CI on the lab RTX 5090: the task04 seed world, hardened (``isolation.policy_user``), with a CI App
whose runner is task04's own verifier image on the same GPU host.

The policy (scripted here) works only through its tools as the demoted ``agent`` user: it can use the GPU, it
cannot see the grader, and it learns from CI only ``name: PASS/FAIL``. The checks are task04's real gates, run
from the hidden grader: ``static`` (``grade.static_checks``: no texture/surface references), ``gpu`` (the runner
sees the RTX 5090) and ``configure`` (PCL configures for CUDA 12.8 / sm_120 as an unprivileged user). The full
build/perf gates are left to the verifier (they take an hour on a shared machine).

Run: ``CRUCIBLE_GPU=1 pytest tests/gpu/test_ci_lab_gpu.py -m gpu``.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from crucible.core.config import get_config
from crucible.core.schema.actions import ToolCall
from crucible.core.store import Store
from crucible.world.exec import LocalExec
from crucible.world.world import World

pytestmark = [pytest.mark.gpu, pytest.mark.slow]
if not os.environ.get("CRUCIBLE_GPU"):
    pytest.skip("set CRUCIBLE_GPU=1 to run GPU gates on the lab host", allow_module_level=True)

from crucible_pack_coding import seed_dirs

from crucible.interop.harbor.task_io import load_crucible_task

REPORT = Path(__file__).resolve().parents[2] / "docs" / "modules" / "reports" / "m1_ci_lab_gpu.json"
VERIFIER_IMAGE = "task04-pcl-gpu-cuda12-verifier:v1"
CMAKE = ("cmake -S {root} -B /tmp/ci-build -G Ninja -DCMAKE_BUILD_TYPE=Release -DWITH_CUDA=ON -DBUILD_GPU=ON -DCUDA_ARCH_BIN=120 "
         "-DBUILD_apps=OFF -DBUILD_examples=OFF -DBUILD_tools=OFF -DBUILD_global_tests=OFF -DWITH_OPENNI=OFF -DWITH_OPENNI2=OFF "
         "-DWITH_QT=OFF -DWITH_PCAP=OFF -DWITH_LIBUSB=OFF -DWITH_ENSENSO=OFF -DWITH_DAVIDSDK=OFF -DWITH_DSSDK=OFF -DWITH_RSSDK=OFF "
         "-DWITH_RSSDK2=OFF -DBUILD_gpu_kinfu_tools=OFF -DBUILD_gpu_kinfu_large_scale_tools=OFF -DBUILD_gpu_people_tools=OFF")


def _world(seed: Path):
    t = load_crucible_task(seed)
    host = get_config().gpu_hosts["lab5090"]
    spec = t.world.model_dump()
    spec["isolation"] = {"policy_user": "agent", "pids_limit": 8192}
    for a in spec["apps"]:
        if a["id"] == "pcl":
            a["params"].update({"scrub_history": True, "history_keep": "base"})
    spec["apps"].append({
        "id": "ci", "template": "common/ci", "state": {"adapter": "json_doc", "params": {"path": "/var/lib/crucible/ci.json"}},
        "params": {
            "source": "pcl", "target_root": "/src/pcl", "max_runs": 4, "latency": "30m",
            "runner": {"kind": "docker", "image": VERIFIER_IMAGE, "docker_host": host.docker_host, "gpus": True, "pids_limit": 8192},
            "hidden": {"src": str(t.harbor.path / "tests"), "dest": "/opt/crucible-hidden"},
            "checks": [
                {"name": "static", "show_output": True, "timeout_s": 300,
                 "cmd": "python3 -c \"import sys; sys.path.insert(0, '{hidden}'); import grade; from pathlib import Path; "
                        "h = grade.static_checks(Path('{root}')); print(len(h), 'texture/surface references remain'); sys.exit(1 if h else 0)\""},
                {"name": "gpu", "show_output": True, "timeout_s": 120, "cmd": "nvidia-smi --query-gpu=name --format=csv,noheader | grep -q 'RTX 5090'"},
                {"name": "configure", "user": "65534", "timeout_s": 1800,
                 "cmd": "export PATH=/usr/local/cuda/bin:$PATH HOME=/tmp; rm -rf /tmp/ci-build; " + CMAKE},
            ],
        },
    })
    spec["actors"][0]["tools"].append("ci.*")
    return t, type(t.world).model_validate(spec)


async def call(w: World, tool: str, **args):
    r = await w.step("agent", ToolCall(tool_call_id=f"{tool}-{time.monotonic_ns()}", function_name=tool, arguments=args))
    return r.observation.tool_results[0]


async def test_hidden_ci_on_the_lab_gpu(tmp_path: Path):
    seed = next(d for d in seed_dirs() if d.name == "task04_pcl_gpu_cuda12")
    t, spec = _world(seed)
    rec: dict = {"world": spec.world, "runner_image": VERIFIER_IMAGE}
    t0 = time.monotonic()
    w = World(spec, store=Store(tmp_path / "s"), pack_root=seed, workdir=tmp_path / "w")
    await w.start()
    try:
        await w.reset()
        rec["start_s"] = round(time.monotonic() - t0, 1)
        rec["egress"] = w.backend.egress_report
        # the policy is the demoted user, and still has the GPU (device nodes are world-usable under the nvidia runtime)
        r = await call(w, "pcl.shell", command="id -un; nvidia-smi --query-gpu=name --format=csv,noheader; ls /opt/crucible-hidden /var/lib/crucible 2>&1 | head -3")
        rec["policy_shell"] = r.content
        assert "agent" in r.content and "RTX 5090" in r.content and "Permission denied" in r.content
        # base: the static gate fails, the GPU is there; the grader itself is never shown
        t1 = time.monotonic()
        r = await call(w, "ci.run_ci", checks=["static", "gpu"])
        rec["ci_base"] = {"content": r.content, "seconds": round(time.monotonic() - t1, 1)}
        assert r.ok and "static: FAIL" in r.content and "gpu: PASS" in r.content
        assert "TEXREF" not in r.content and "grade.py" not in r.content and "def static_checks" not in r.content
        # the oracle, applied by the policy through its own shell (as the agent user)
        await w.apps["pcl"].policy_exec.push_file(t.harbor.path / "solution" / "solution.patch", "/home/agent/oracle.patch")
        r = await call(w, "pcl.shell", command="git apply --binary ~/oracle.patch && git status --short | wc -l", timeout_s=600)
        assert r.ok and "[exit 0]" in r.content, r.content
        t2 = time.monotonic()
        r = await call(w, "ci.run_ci")
        rec["ci_oracle"] = {"content": r.content, "seconds": round(time.monotonic() - t2, 1)}
        assert "static: PASS" in r.content and "gpu: PASS" in r.content and "configure: PASS" in r.content, r.content
        hist = await call(w, "ci.ci_history")
        rec["ci_history"] = json.loads(hist.content) if hist.content.startswith("{") else hist.content
        rec["virtual_time"] = w.clock.now_iso() if w.clock else None
    finally:
        await w.close()
    host = get_config().gpu_hosts["lab5090"]
    left = (await LocalExec().run("docker ps -q --filter label=crucible.app=ci", env={"DOCKER_HOST": host.docker_host})).stdout.split()
    assert not left, f"CI runner containers left on the lab host: {left}"
    rec["total_s"] = round(time.monotonic() - t0, 1)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(rec, indent=1, sort_keys=True, default=str))
