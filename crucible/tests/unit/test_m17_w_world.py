"""Window lane (study 08 §1, §3; 09): the World at A and its leak probes.

Offline: the probe verdicts, the packaging readers and the seeded workspace (one commit, tagged ``base``). With Docker:
the World image of the synthetic repository, the four leak probes executed in it, and a whole episode through the M10
runner and ``WorkspaceHarness`` in a ``--network none`` container (shell runs inside, the patch is graded from a pristine
A, the checkpoint sidecar opts in and writes its evidence)."""

from __future__ import annotations

import asyncio
import json
import subprocess
from pathlib import Path

import pytest
from test_m17_w_miner import SB
from test_m17_w_task import REQS, make_task

from crucible.factory.lanes.windows_world import (
    ImageBook,
    judge_leak,
    leak_probes,
    own_containers,
    parse_leak,
    project_name,
    requirement_names,
    requirement_sources,
    seed_workspace,
    world_dockerfile,
)


def test_leak_verdicts() -> None:
    clean = parse_leak("PIP_RC=1\nDIST_FILES=\nNAME_HITS=\nGIT_COMMITS=1\nGIT_TAGS=base \n")
    assert judge_leak(clean, version="1.2.0") == {"probes": {"pip_download_b_fails": "pass", "no_b_distribution_file": "pass",
                                                             "no_new_test_names_in_image": "pass", "git_log_only_base": "pass"},
                                                  "ok": True, "evidence": clean}
    assert judge_leak(clean, version=None)["probes"]["pip_download_b_fails"] == "n/a" and judge_leak(clean, version=None)["ok"]
    for bad in ({"PIP_RC": "0"}, {"DIST_FILES": "/root/semver-3.0.4.whl"}, {"NAME_HITS": "/opt/x/test_new.py"}, {"GIT_COMMITS": "7"},
                {"GIT_TAGS": "base v3.0.4"}):
        assert not judge_leak({**clean, **bad}, version="3.0.4")["ok"], bad


def test_seeded_workspace_has_one_commit_tagged_base(tmp_path: Path) -> None:
    src = tmp_path / "a"
    (src / "pkg").mkdir(parents=True)
    (src / "pkg" / "x.py").write_text("X = 1\n")
    seed_workspace(src, tmp_path / "ws")
    log = subprocess.run(["git", "log", "--all", "--format=%H"], cwd=tmp_path / "ws", capture_output=True, text=True, check=True).stdout.split()
    tags = subprocess.run(["git", "tag"], cwd=tmp_path / "ws", capture_output=True, text=True, check=True).stdout.split()
    assert len(log) == 1 and tags == ["base"]


def test_packaging_readers(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text('[project]\nname = "toylib"\nversion = "1.0"\n[dependency-groups]\ntests = ["pytest-mock>=3"]\n')
    (tmp_path / "requirements").mkdir()
    (tmp_path / "requirements" / "tests.txt").write_text("pytest==6.0  # pinned old\nhypothesis>=5\n-e .\nfreezegun[extra]==1.0\nhttps://x/y.whl\n")
    assert project_name(tmp_path) == "toylib"
    files, groups = requirement_sources(tmp_path)
    assert files == ["requirements/tests.txt"] and groups == ["pytest-mock>=3"]
    assert requirement_names((tmp_path / "requirements" / "tests.txt").read_text()) == ["hypothesis", "freezegun[extra]"]  # pins dropped, pytest kept at ours
    df = world_dockerfile("toylib", ["hypothesis"], ["pytest-mock>=3"])
    assert "uninstall" not in df and ".[test,tests,testing]" in df and df.index("pytest==8.3.3") > df.index("hypothesis")
    assert "pip install --no-cache-dir ." not in world_dockerfile(None, [], [], packaged=False)


# ---- Docker ----------------------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def world(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("wworld")
    task, rec, repo = make_task(tmp)
    card = asyncio.run(ImageBook().ensure("toyrepo", task.base_dir))
    assert card["ok"], card.get("build_log")
    task.window["world"] = {"image": card["image"], "digest": card["digest"]}
    return task, rec, repo, card


@pytest.mark.docker
def test_leak_probes_pass_in_the_world_image(world) -> None:
    task, _rec, _, card = world
    assert card["image"].startswith("crucible-lh-w:toyrepo-")
    out = asyncio.run(leak_probes(card["image"], task.base_dir, package="toyrepo", version="0.2", new_test_names={"test_gamma", "test_beta"}))
    assert out["ok"], out
    assert out["probes"] == {"pip_download_b_fails": "pass", "no_b_distribution_file": "pass", "no_new_test_names_in_image": "pass",
                             "git_log_only_base": "pass"}


@pytest.mark.docker
def test_a_planted_test_definition_is_found(world, tmp_path: Path) -> None:
    task, _, _, card = world
    base = tmp_path / "a"
    subprocess.run(["cp", "-R", str(task.base_dir), str(base)], check=True)
    (base / "pkg" / "leak.py").write_text("def test_gamma():\n    pass\n")  # B's test, smuggled into the tree at A
    out = asyncio.run(leak_probes(card["image"], base, package="toyrepo", version=None, new_test_names={"test_gamma"}))
    assert out["probes"]["no_new_test_names_in_image"] == "fail" and not out["ok"]


@pytest.mark.docker
async def test_an_episode_in_the_window_world(world, tmp_path: Path) -> None:
    import uuid

    from crucible.actors.policy import ScriptedPolicy
    from crucible.core.schema.actions import Done, ToolCall
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.workspace import WorkspaceHarness
    from crucible.runtime.run import Runner, RunSpec

    task, rec, _repo, _ = world
    mine = f"crucible-lh-world-t{uuid.uuid4().hex[:6]}-"  # this run's containers only (another run's Worlds may be up)
    task.lane.world_prefix = mine
    task.sandbox = SB()  # grading on the host interpreter here; the World itself is the Docker container
    task.checkpoint = CheckpointConfig(every=2, cap=6)
    files = {p: t for p, t in rec["oracle_files"].items() if p.startswith("pkg/")}
    probe = "python -c \"import socket\ntry:\n    socket.create_connection(('1.1.1.1', 53), timeout=3); print('NET-OPEN')\nexcept OSError: print('NET-NONE')\"; ls /opt/m17"
    acts = [ToolCall(tool_call_id="s0", function_name="repo.shell", arguments={"command": probe}),
            *[ToolCall(tool_call_id=f"w{i}", function_name="repo.write_file", arguments={"path": p, "content": t}) for i, (p, t) in enumerate(sorted(files.items()))],
            ToolCall(tool_call_id="s1", function_name="repo.shell", arguments={"command": "python -m pytest -q -p no:cacheprovider tests 2>&1 | tail -1"}),
            ToolCall(tool_call_id="st", function_name="repo.write_file", arguments={"path": "STATUS.json", "content": json.dumps({r["id"]: 0.9 for r in REQS})})]
    seen: list[str] = []

    def step(obs):
        seen.append(obs.model_dump_json())
        return acts[len(seen) - 1] if len(seen) <= len(acts) else Done(reason="done")

    ref = PolicyRef(id="scripted:window", make=lambda a: {"agent": ScriptedPolicy("agent", step)})
    await Runner(tmp_path).run(RunSpec(run_id="win", tasks=[task], policies=[ref], harnesses=[WorkspaceHarness()], k=1))
    adir = min((tmp_path / "win" / "samples").glob("*/a*"))
    out = json.loads((adir / "record.json").read_text())
    blob = "\n".join(seen)
    assert "NET-NONE" in blob and "m17guard.py" in blob  # the shell ran in the World's container, offline
    assert "3 passed" in blob or "passed" in blob
    assert out["reward"] == 1.0, out
    assert out["meta"]["progress"]["checkpoints"] >= 2 and (adir / "checkpoints.jsonl").exists()
    patch = (adir / "artifacts" / "model.patch").read_text()
    assert "pkg/gamma.py" in patch and "STATUS.json" in patch
    assert not [c for c in own_containers() if c.startswith(mine)]  # the World's container is gone
