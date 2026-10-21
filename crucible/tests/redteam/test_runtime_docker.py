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


TEST_SH = """#!/bin/bash
set -u
mkdir -p /logs/verifier; W=$(mktemp -d); cd $W && git init -q
if [ -s /logs/artifacts/model.patch ] && ! git apply /logs/artifacts/model.patch 2>/logs/verifier/apply.log; then echo 0 > /logs/verifier/reward.txt; exit 0; fi
if [ "${MODE:-}" = silent ]; then exit 3; fi
ok=false; [ "$(python3 fib.py 2>/dev/null | tr '\\n' ' ')" = "0 1 1 2 3 5 8 13 21 34 " ] && ok=true
echo "{\\"gates\\": {\\"output\\": {\\"pass\\": $ok}}}" > /logs/verifier/report.json
$ok && echo 1 > /logs/verifier/reward.txt || echo 0 > /logs/verifier/reward.txt
"""


def harbor_dir(tmp_path, command="bash /tests/test.sh") -> Path:
    d = tmp_path / "htask"
    (d / "tests").mkdir(parents=True)
    (d / "tests" / "test.sh").write_text(TEST_SH)
    (d / "instruction.md").write_text("write fib.py")
    (d / "task.toml").write_text(f'schema_version = "1.3"\n[task]\nname = "t/fib"\n[verifier]\nnetwork_mode = "no-network"\ncommand = "{command}"\ntimeout_sec = 60\n')
    return d


FIB_PATCH = """diff --git a/fib.py b/fib.py
new file mode 100644
--- /dev/null
+++ b/fib.py
@@ -0,0 +1,4 @@
+a, b = 0, 1
+for _ in range(10):
+    print(a)
+    a, b = b, a + b
"""


async def test_harbor_dir_task_grades_with_its_own_verifier(image, tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask

    t = HarborDirTask(harbor_dir(tmp_path), image=image, expected_image=image_id(image))
    g = await t.grade_patch(FIB_PATCH, tmp_path / "g1")
    assert g.reward == 1.0 and g.gates == {"output": "pass"} and g.infra["status"] == "ok"
    g0 = await t.grade_patch("", tmp_path / "g0")
    assert g0.reward == 0.0 and g0.gates == {"output": "fail"} and g0.infra["status"] == "ok"
    silent = HarborDirTask(harbor_dir(tmp_path / "s", command="MODE=silent bash /tests/test.sh"), image=image)
    gs = await silent.grade_patch(FIB_PATCH, tmp_path / "gs")
    # no reward file is not a 0; M7 R5: in the separate verifier the code under test can delete or plant it, so it is
    # answer-sensitive (graded again, then decided by the control), Harbor's name in the reason
    assert gs.infra["status"] == "masked" and classify_verifier(gs.infra).detail == "reward_file_invalid"
    assert "RewardFileNotFoundError" in gs.infra["reason"]
    stale = HarborDirTask(harbor_dir(tmp_path / "st"), image=image, expected_image="sha256:" + "1" * 64)
    assert classify_verifier((await stale.grade_patch(FIB_PATCH, tmp_path / "gst")).infra).kind == "crucible:stale_image"


async def test_imported_run_through_the_runner(image, tmp_path):
    from crucible.interop.harbor.verify_dir import HarborDirTask
    from crucible.runtime.harness.imported import ImportedHarness
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.run import Runner, RunSpec

    (tmp_path / "p.patch").write_text(FIB_PATCH)
    pol = PolicyRef(id="import:fib-run1", make=lambda a: {}, config={"recorded": "fib-run1"})  # M10 lab audit R-4: declared configuration
    spec = RunSpec(run_id="imp", tasks=[HarborDirTask(harbor_dir(tmp_path), image=image)], policies=[pol],
                   harnesses=[ImportedHarness({"import:fib-run1": tmp_path / "p.patch"})])
    s = await Runner(tmp_path / "runs").run(spec)
    assert (s.attempts, s.scored, s.mean_reward) == (1, 1, 1.0)


# ---- M10 lab audit R-1: the installed-agent harness's process stop, in a real container


def _stop_harness(tmp_path):
    from crucible.connectors.capture_proxy import CaptureProxy
    from crucible.interop.harbor.agent_harness import HarborAgentHarness

    return HarborAgentHarness("mini-swe-agent", "2.4.6", model="openai/m", proxy=CaptureProxy(tmp_path / "cap"), proxy_url="http://10.1.2.3:8788",
                              agent_factory=lambda **k: None, harbor_version="0.24.0")


def _running(name: str) -> bool:
    r = subprocess.run(["docker", "inspect", "-f", "{{.State.Running}}", name], capture_output=True, text=True, check=False)
    return r.stdout.strip() == "true"


def _alive(name: str, pattern: str) -> bool:
    return subprocess.run(["docker", "exec", name, "pgrep", "-f", pattern], capture_output=True, check=False).returncode == 0


def test_user_stop_spares_a_keep_alive_running_as_the_agent_user(image, tmp_path):
    """A non-root agent user's stop kills a detached (setsid) agent process and spares the keep-alive the image's USER
    runs under docker-init: the container keeps running (a bare ``pkill -u`` stopped it, graded the policy's 0)."""
    from types import SimpleNamespace

    name = f"crucible-rt-{uuid.uuid4().hex[:8]}"
    subprocess.run(["docker", "run", "-d", "--rm", "--init", "--user", "nobody", "--name", name, image, "sleep", "600"], check=True, capture_output=True)
    try:
        subprocess.run(["docker", "exec", "-d", "-u", "nobody", name, "sh", "-c", "setsid sleep 900"], check=True, capture_output=True)
        assert _alive(name, "sleep 900")
        cmd = _stop_harness(tmp_path)._kill_command(SimpleNamespace(agent_user="nobody"))
        r = subprocess.run(["docker", "exec", "-u", "root", name, "sh", "-c", cmd], capture_output=True, text=True, check=False)
        assert r.returncode == 0, r.stderr
        assert _running(name) and _alive(name, "sleep 600") and not _alive(name, "sleep 900"), r.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)


def test_cohort_stop_kills_only_what_started_after_the_mark(image, tmp_path):
    """A root agent's stop kills every process started since the phase mark (a detached one included) and spares PID 1
    and everything older (the keep-alive): the container keeps running."""
    from types import SimpleNamespace

    from crucible.interop.harbor.agent_harness import _PHASE_MARK

    name = container(image, init=True)
    try:
        mark = int(subprocess.run(["docker", "exec", name, "sh", "-c", _PHASE_MARK], capture_output=True, text=True, check=True).stdout.strip())
        subprocess.run(["docker", "exec", "-d", name, "sh", "-c", "setsid sleep 900"], check=True, capture_output=True)
        assert _alive(name, "sleep 900")
        cmd = _stop_harness(tmp_path)._kill_command(SimpleNamespace(agent_user=None), mark)
        r = subprocess.run(["docker", "exec", "-u", "root", name, "sh", "-c", cmd], capture_output=True, text=True, check=False)
        assert r.returncode == 0 and "killed:" in r.stdout, r.stderr
        assert _running(name) and _alive(name, "sleep 600") and not _alive(name, "sleep 900"), r.stdout
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True, check=False)
