"""M10 exit-gate injections that need a real Linux container (Appendix B.5): zombie PID 1, stale verifier image,
verifier timeout, and the self-matching pkill over a control channel. Each must be classified as infrastructure and
masked, never scored.

    pytest tests/redteam/test_runtime_docker.py -m docker
"""

from __future__ import annotations

import subprocess
import uuid
from pathlib import Path

import pytest

from crucible.runtime.gpu import preflight
from crucible.runtime.infra import classify_verifier
from crucible.runtime.remote import control
from crucible.verify.stage import bundle, docker_grade, image_id

pytestmark = pytest.mark.docker
IMAGE = "crucible-verify-test:py312"
DOCKERFILE = "FROM python:3.12-slim\nRUN pip install --no-cache-dir pytest==8.3.3 && apt-get update && apt-get install -y --no-install-recommends git procps util-linux && rm -rf /var/lib/apt/lists/*\n"


@pytest.fixture(scope="module")
def image() -> str:
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True, check=False).returncode != 0:
        p = subprocess.run(["docker", "build", "-t", IMAGE, "-"], input=DOCKERFILE, text=True, capture_output=True, check=False)
        assert p.returncode == 0, p.stderr[-2000:]
    return IMAGE


def container(image: str, *, init: bool) -> str:
    name = f"crucible-rt-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--name", name, *(["--init"] if init else []), image, "sleep", "600"], check=True, capture_output=True)
    return name


def runner(name: str):
    def run(cmd: str) -> tuple[int, str]:
        r = subprocess.run(["docker", "exec", name, "sh", "-c", cmd], capture_output=True, text=True, check=False)
        return r.returncode, r.stdout

    return run


def test_zombie_pid1_preflight(image):
    bad, good = container(image, init=False), container(image, init=True)
    try:
        f = preflight(runner(bad))
        assert f is not None and f.kind == "crucible:zombie_wait" and f.masks  # `sleep` as PID 1: refuse before the agent starts
        assert preflight(runner(good)) is None
    finally:
        subprocess.run(["docker", "rm", "-f", bad, good], capture_output=True, check=False)


def test_stale_verifier_image_masked_before_grading(image, tmp_path):
    kit = bundle(tmp_path / "b", {"plan_sleepy": Path(__file__).parent / "runtime_task" / "plan_sleepy.py"})
    g = docker_grade(image=image, bundle_dir=kit, plan="plan_sleepy", out_dir=tmp_path / "o", expected_image="sha256:" + "0" * 64)
    assert g.reward["infra"]["status"] == "masked" and g.rc == -1  # nothing ran
    assert classify_verifier(g.reward["infra"]).kind == "crucible:stale_image"
    ok = docker_grade(image=image, bundle_dir=kit, plan="plan_sleepy", out_dir=tmp_path / "o2", expected_image=image_id(image), params={"sleep": 0})
    assert ok.reward["reward"] == 1.0  # the pinned image grades normally


def test_verifier_timeout_masked_and_container_removed(image, tmp_path):
    kit = bundle(tmp_path / "b", {"plan_sleepy": Path(__file__).parent / "runtime_task" / "plan_sleepy.py"})
    name = f"crucible-grade-to-{uuid.uuid4().hex[:6]}"
    g = docker_grade(image=image, bundle_dir=kit, plan="plan_sleepy", out_dir=tmp_path / "o", name=name, timeout=8, params={"sleep": 120})
    f = classify_verifier(g.reward["infra"])
    assert f.kind == "verifier_error" and f.detail == "verifier_timeout" and f.masks
    ps = subprocess.run(["docker", "ps", "-a", "--filter", f"name={name}", "-q"], capture_output=True, text=True, check=False).stdout.strip()
    assert ps == ""  # the named container was killed and removed


def test_self_matching_pkill_over_linux_channel(image):
    name = container(image, init=True)
    try:
        _, f = control(f"pkill -f 'crucible-self-kill-{name}' ; true", ssh=["docker", "exec", name, "bash", "-c"])
        assert f is not None and f.kind == "crucible:harness_self_kill" and f.masks  # Linux pkill matches its own channel (B.5)
        _, ok = control(f"pkill -f '[c]rucible-self-kill-{name}' ; true", ssh=["docker", "exec", name, "bash", "-c"])
        assert ok is None  # the [x]yz form cannot match itself
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
