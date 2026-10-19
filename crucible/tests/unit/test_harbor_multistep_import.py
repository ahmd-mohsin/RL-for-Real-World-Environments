"""M11: Harbor multi-step task directories as Crucible multi-step tasks (``interop/harbor/multi_step.py``).

A synthetic task in Harbor's exact layout (``task.toml`` with ``[[steps]]``, ``steps/<name>/{instruction.md, tests,
solution, workdir/setup.sh}``, a shared top-level ``tests/``) and Harbor's own examples. Without Docker: steps map in
order with their instructions, tests (step script, else the shared one), ``min_reward`` (judged by Harbor's own rule)
and reward strategy; separate-verifier steps and single-step tasks are refused. On Docker, end to end through the
Runner: the oracle replay scores 1 on every step; the shared verifier's ``/tests`` is gone when the next step opens; a
no-op stops at the first gate; a per-key ``min_reward`` stops the task where Harbor would; Harbor's
``hello-multi-step-simple`` and ``hello-multi-step-advanced`` (workdir upload, setup.sh, step healthcheck, step verifier
env, shared helpers) score 1.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from crucible.interop.harbor.multi_step import MIN_REWARD_GATE, HarborMultiStepTask, min_reward_failure

EXAMPLES = Path(__file__).resolve().parents[3] / "RL-Environments/06-frameworks-and-env-code/harbor-framework-harbor/examples/tasks"

TASK_TOML = """schema_version = "1.4"

[task]
name = "crucible-fixture/notes-multi-step"
description = "Three steps over one container: write, configure (workdir + setup.sh + step env), finish (shared test)."

[metadata]
tags = ["multi-step"]

[verifier]
timeout_sec = 60.0

[agent]
timeout_sec = 60.0

[environment]
workdir = "/app"
cpus = 1
memory_mb = 512
network_mode = "no-network"

[[steps]]
name = "write"
min_reward = 1.0

[steps.verifier]
timeout_sec = 30.0

[[steps]]
name = "configure"
min_reward = { reward = 1.0, style = 0.5 }

[steps.agent]
timeout_sec = 90.0

[steps.verifier]
timeout_sec = 30.0

[steps.verifier.env]
EXPECTED_LIMIT = "42"

[[steps]]
name = "finish"
"""
FILES = {
    "environment/Dockerfile": "FROM python:3.12-slim\nRUN mkdir -p /app\nWORKDIR /app\n",
    "tests/helpers.sh": "pass() { echo 1 > /logs/verifier/reward.txt; exit 0; }\nfail() { echo 0 > /logs/verifier/reward.txt; exit 0; }\n",
    "tests/test.sh": "#!/bin/bash\nsource /tests/helpers.sh\n[ -f /app/DONE ] && grep -qx alpha /app/notes.txt && pass\nfail\n",
    "steps/write/instruction.md": "Create /app/notes.txt containing the single line `alpha`.\n",
    "steps/write/tests/test.sh": "#!/bin/bash\nsource /tests/helpers.sh\ngrep -qx alpha /app/notes.txt && pass\nfail\n",
    "steps/write/solution/solve.sh": "#!/bin/bash\necho alpha > /app/notes.txt\n",
    "steps/configure/instruction.md": "A file limits.txt is in /app. Write its number to /app/limit.conf as `limit=<n>`, with a `# limit` comment line first.\n",
    "steps/configure/workdir/limits.txt": "  42  \n",
    "steps/configure/workdir/setup.sh": "#!/bin/bash\nset -e\nprintf '%s\\n' \"$(tr -d '[:space:]' < limits.txt)\" > limits.txt\nrm -- \"$0\"\n",
    "steps/configure/tests/expected.txt": "limit=42\n",
    "steps/configure/tests/test.sh": (
        "#!/bin/bash\n[ \"$EXPECTED_LIMIT\" = 42 ] || { echo 0 > /logs/verifier/reward.txt; exit 0; }\n"
        "r=0; grep -qxF \"$(cat /tests/expected.txt)\" /app/limit.conf && r=1\n"
        "s=0.3; head -1 /app/limit.conf | grep -q '^# limit' && s=1\n"
        "echo \"{\\\"reward\\\": $r, \\\"style\\\": $s}\" > /logs/verifier/reward.json\n"),
    "steps/configure/solution/solve.sh": "#!/bin/bash\nprintf '# limit\\nlimit=%s\\n' \"$(cat /app/limits.txt)\" > /app/limit.conf\n",
    "steps/finish/instruction.md": "Create an empty file /app/DONE. Keep notes.txt as it is.\n",
    "steps/finish/solution/solve.sh": "#!/bin/bash\ntouch /app/DONE\n",
}


@pytest.fixture
def task_dir(tmp_path) -> Path:
    root = tmp_path / "notes-multi-step"
    (root / "task.toml").parent.mkdir(parents=True)
    (root / "task.toml").write_text(TASK_TOML)
    for rel, text in FILES.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return root


def test_steps_map_in_order(task_dir):
    t = HarborMultiStepTask(task_dir)
    assert [s.name for s in t.steps] == ["write", "configure", "finish"] and t.step_reward == "mean_executed"  # Harbor's default "mean"
    assert t.steps[1].instruction.startswith("A file limits.txt") and t.instruction == t.steps[0].instruction
    assert [s.min_reward for s in t.steps] == [{f"check/{MIN_REWARD_GATE}": 1.0}] * 2 + [None]
    assert [m["tests"] for m in t.step_meta.values()] == ["steps/write/tests/test.sh", "steps/configure/tests/test.sh", "tests/test.sh"]  # shared fallback
    assert t.step_meta["configure"]["agent_timeout_sec"] == 90.0 and t.step_meta["write"]["agent_timeout_sec"] == 60.0 and t.step_meta["configure"]["has_workdir"]
    acts = t.oracle_actions()
    assert sum(a.kind == "done" for a in acts) == 3 and not any("/solution" in json.dumps(getattr(a, "arguments", {})) for a in acts)
    (task_dir / "task.toml").write_text(TASK_TOML.replace('[task]', 'multi_step_reward_strategy = "final"\n\n[task]'))
    assert HarborMultiStepTask(task_dir).step_reward == "final"


def test_min_reward_is_harbors_rule():
    assert min_reward_failure({"reward": 1.0}, 1.0) is None
    assert min_reward_failure({"reward": 0.0}, 1.0) == "reward=0.0 below min_reward 1.0"
    assert min_reward_failure({"reward": 1, "style": 0.3}, {"reward": 1.0, "style": 0.5}) == "style=0.3 below min_reward 0.5"
    assert min_reward_failure({"reward": 1}, {"style": 0.0}) == "style=-inf below min_reward 0.0"  # a missing key is -inf
    assert min_reward_failure(None, 0.0) is not None


@pytest.mark.skipif(not EXAMPLES.is_dir(), reason="Harbor checkout not present")
def test_real_examples_map_or_refuse():
    adv = HarborMultiStepTask(EXAMPLES / "hello-multi-step-advanced")
    assert [s.name for s in adv.steps] == ["scaffold", "implement", "document"]
    assert [m["min_reward"] for m in adv.step_meta.values()] == [1.0, 0.5, None] and adv.step_meta["document"]["artifacts"] == ["/app/README.md"]
    assert [s.name for s in HarborMultiStepTask(EXAMPLES / "hello-multi-step-simple").steps] == ["create-file", "append-content"]
    with pytest.raises(ValueError, match="separate environment"):
        HarborMultiStepTask(EXAMPLES / "separate-verifier-multi-step")
    with pytest.raises(ValueError, match="not a multi-step task"):
        HarborMultiStepTask(EXAMPLES / "hello-world")
    with pytest.raises(ValueError, match="POSIX"):
        HarborMultiStepTask(EXAMPLES / "hello-multi-step-bat").oracle_actions()


# ----------------------------------------------------------------------------------------------- Docker, end to end
def _docker() -> bool:
    return bool(shutil.which("docker")) and subprocess.run(["docker", "info"], capture_output=True, check=False).returncode == 0


async def _run(tmp_path, task, actions, rid):
    from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec

    ref = PolicyRef(id=f"scripted:{rid}", make=lambda attempt: {"agent": ScriptedPolicy("agent", list(actions))}, config={"script": rid})  # M10 lab audit R-4: declared configuration
    await Runner(tmp_path / "runs").run(RunSpec(run_id=rid, tasks=[task], policies=[ref], harnesses=[NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=20))], k=1))
    adir = next((tmp_path / "runs" / rid / "samples").glob("*/a1"))
    return json.loads((adir / "record.json").read_text()), adir


def _results(adir: Path) -> list[str]:
    return [e["text"] or "" for e in json.loads((adir / "trace.json").read_text())["events"] if e.get("type") == "tool_result"]


@pytest.mark.docker
@pytest.mark.skipif(not _docker(), reason="no Docker daemon")
async def test_synthetic_task_end_to_end(task_dir, tmp_path):
    from crucible.core.schema.actions import Done, ToolCall

    t = HarborMultiStepTask(task_dir)
    oracle = t.oracle_actions()
    first_done = next(i for i, a in enumerate(oracle) if a.kind == "done")
    probe = ToolCall(tool_call_id="p1", function_name="term.exec", arguments={"command": "echo setup=$(ls setup.sh 2>/dev/null | wc -l) limits=$(cat limits.txt)"})
    rec, adir = await _run(tmp_path, t, [*oracle[: first_done + 1], probe, *oracle[first_done + 1:]], "oracle")
    assert rec["failure"] is None and rec["reward"] == 1.0, rec
    steps = rec["meta"]["task_steps"]
    assert [(s["name"], s["reward"], s["passed_gate"]) for s in steps] == [("write", 1.0, True), ("configure", 1.0, True), ("finish", 1.0, True)]
    marks = [e["data"]["name"] for e in json.loads((adir / "trace.json").read_text())["events"] if e.get("type") == "task_step"]
    assert marks == ["write", "configure", "finish"]
    assert any('"output": "setup=0 limits=42' in r for r in _results(adir))  # workdir uploaded, setup.sh ran (normalized, removed itself)
    rep = json.loads((adir / "steps" / "01-configure" / "verifier" / "report.json").read_text())
    assert rep["meta"]["rewards"] == {"reward": 1, "style": 1} and rep["gates"] == {MIN_REWARD_GATE: "pass"}

    rec, adir = await _run(tmp_path, t, [Done(reason="no")] * 3, "noop")  # stops at the first gate (Harbor: min_reward 1.0)
    assert rec["failure"] is None and rec["reward"] == 0.0 and [s["name"] for s in rec["meta"]["task_steps"]] == ["write"]
    assert rec["meta"]["task_steps"][0]["abort"] == f"check/{MIN_REWARD_GATE}=0.0 below min_reward 1.0" and not (adir / "steps" / "01-configure").exists()

    no_comment = ToolCall(tool_call_id="n1", function_name="term.exec", arguments={"command": "echo limit=$(cat limits.txt) > /app/limit.conf"})
    rec, adir = await _run(tmp_path, t, [*oracle[: first_done + 1], no_comment, Done(reason="configured"), Done(reason="x")], "style")
    steps = rec["meta"]["task_steps"]
    assert [s["name"] for s in steps] == ["write", "configure"] and steps[1]["reward"] == 1.0 and not steps[1]["passed_gate"]  # per-key gate
    rep = json.loads((adir / "steps" / "01-configure" / "verifier" / "report.json").read_text())
    assert rep["checks"][0]["detail"]["harbor"] == "style=0.3 below min_reward 0.5" and rec["reward"] == 1.0  # Harbor's mean over executed steps


@pytest.mark.docker
@pytest.mark.skipif(not _docker(), reason="no Docker daemon")
async def test_step_boundaries_with_harbors_oracle_agent(task_dir, tmp_path):
    """Driven by hand with Harbor's own ``OracleAgent`` per step: the shared verifier's ``/tests`` and ``/logs/verifier``
    are in the container after a step's grade and gone when the next step opens (the policy never sees them)."""
    t = HarborMultiStepTask(task_dir)
    w = await t.open(tmp_path / "w")
    try:
        env = w.backend.env
        await t.steps[0].setup(w)
        await t.oracle_step(w, 0)
        assert (await t.steps[0].grade(w, tmp_path / "g0")).reward == 1.0
        assert (await env.exec("ls -A /tests | wc -l")).stdout.strip() != "0"
        await t.steps[1].setup(w)
        assert (await env.exec("echo $(ls -A /tests | wc -l) $(ls -A /logs/verifier | wc -l)")).stdout.strip() == "0 0"
        await t.oracle_step(w, 1)
        rep = await t.steps[1].grade(w, tmp_path / "g1")
        assert rep.reward == 1.0 and rep.meta["components"] == {"harbor/reward": 1.0, "harbor/style": 1.0}
    finally:
        await w.close()


@pytest.mark.docker
@pytest.mark.skipif(not _docker() or not EXAMPLES.is_dir(), reason="no Docker daemon or Harbor checkout")
@pytest.mark.parametrize("name", ["hello-multi-step-simple", "hello-multi-step-advanced"])
async def test_real_examples_oracle_end_to_end(tmp_path, name):
    t = HarborMultiStepTask(EXAMPLES / name)
    rec, _ = await _run(tmp_path, t, t.oracle_actions(), name)
    assert rec["failure"] is None and rec["reward"] == 1.0, rec
    assert [s["reward"] for s in rec["meta"]["task_steps"]] == [1.0] * len(t.steps)
