"""Open risks of the harness and reward audit (docs/research/harness-reward-audit.md §8), reward-integrity half:

- R6: what a grade or a credit trusts inside the agent's tree. The windows lane's mid-episode snapshot diffed against the
  agent-movable ``base`` tag with the repository's own config; host git in the agent's repository ran its filter drivers
  (host code execution, content substituted into the patch) and followed a redirected ``.git``; the grader loads test-runner
  plumbing a patch adds (a distribution's ``[pytest11]`` entry point, a module shadowing the runner).
- R3: a policy that kills its own World container turned its 0 into a masked ``env_container_lost``.

Host git in tmp dirs only; no Docker (the World's exec is replaced by a local one), no model."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.run import Failure
from crucible.factory import envs
from crucible.factory.pathcheck import PathRules, check_paths
from crucible.runtime.collect import repo_broken
from crucible.verify import collect as vcollect
from crucible.world.exec import LocalExec

ENV = {"GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@a", "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@a", "GIT_CONFIG_NOSYSTEM": "1",
       "PATH": "/usr/bin:/bin:/usr/local/bin:/opt/homebrew/bin"}


def git(ws: Path, *a: str) -> str:
    return subprocess.run(["git", "-C", str(ws), *a], check=True, capture_output=True, text=True, env={**ENV, "HOME": str(ws.parent)}).stdout


def repo(tmp_path: Path, name: str = "ws") -> tuple[Path, str]:
    ws = tmp_path / name
    ws.mkdir()
    git(ws, "init", "-q")
    (ws / "a.py").write_text("x = 1\n")
    git(ws, "add", "-A")
    git(ws, "commit", "-qm", "base")
    git(ws, "tag", "base")
    return ws, git(ws, "rev-parse", "HEAD").strip()


def plant_filter(ws: Path, sentinel: Path, name: str = "evil") -> None:
    """A filter driver whose clean command runs on whoever indexes the file: it leaves a sentinel and substitutes content."""
    git(ws, "config", f"filter.{name}.clean", f"sh -c 'touch {sentinel}; echo hacked'")
    (ws / ".gitattributes").write_text("*.py filter=evil\n")


@pytest.fixture
def local_docker(monkeypatch):
    """``DockerExec.run`` as a local shell in the workspace (the command a World's container would run)."""
    from crucible.world import exec as wexec

    async def run(self, cmd, *, cwd=None, env=None, timeout_s=None, user=None, stdin=None):
        return await LocalExec().run(cmd, cwd=cwd, env={**ENV, "HOME": str(Path(cwd).parent)}, timeout_s=timeout_s)

    monkeypatch.setattr(wexec.DockerExec, "run", run)


async def window_snapshot(ws: Path, base: str | None) -> str:
    from crucible.factory.lanes.windows import WindowTask

    world = SimpleNamespace(_crucible_ws=ws, _crucible_container="c1", _crucible_base_sha=base)
    return await WindowTask.snapshot_patch(SimpleNamespace(workspace=lambda w: w._crucible_ws), world)


# ---- R6: the mid-episode snapshot -----------------------------------------------------------------------------------------------
async def test_R6_snapshot_diffs_against_the_recorded_base_not_the_tag(tmp_path, local_docker):
    ws, sha = repo(tmp_path)
    (ws / "a.py").write_text("x = 2\n")
    git(ws, "commit", "-qam", "fix")
    git(ws, "tag", "-f", "base")  # the agent moves the tag onto its own commit: a tag diff shows nothing
    patch = await window_snapshot(ws, sha)
    assert "+x = 2" in patch and "-x = 1" in patch


async def test_R6_snapshot_without_a_recorded_base_is_unmeasured(tmp_path, local_docker):
    ws, _ = repo(tmp_path)
    with pytest.raises(RuntimeError, match="no base commit"):
        await window_snapshot(ws, None)


async def test_R6_snapshot_runs_no_filter_or_external_diff_of_the_agent(tmp_path, local_docker):
    ws, sha = repo(tmp_path)
    sentinel = tmp_path / "pwned"
    plant_filter(ws, sentinel)
    git(ws, "config", "diff.external", f"sh -c 'touch {sentinel}x'")
    git(ws, "config", "diff.noprefix", "true")
    (ws / "a.py").write_text("x = 3\n")
    patch = await window_snapshot(ws, sha)
    assert not sentinel.exists() and not Path(f"{sentinel}x").exists()
    assert "+x = 3" in patch and "hacked" not in patch and "--- a/a.py" in patch


async def test_R6_snapshot_drops_derived_files_like_the_collect(tmp_path, local_docker):
    ws, sha = repo(tmp_path)
    (ws / "__pycache__").mkdir()
    (ws / "__pycache__" / "a.cpython-312.pyc").write_bytes(b"\x00\x01")
    (ws / "b.py").write_text("y = 1\n")
    patch = await window_snapshot(ws, sha)
    assert "b/b.py" in patch and "__pycache__" not in patch


async def test_R6_slime_snapshot_template_takes_the_recorded_base(tmp_path):
    from crucible.train.bridges.slime.episode import SNAPSHOT_CMD

    ws, sha = repo(tmp_path)
    (ws / "a.py").write_text("x = 4\n")
    git(ws, "commit", "-qam", "fix")
    git(ws, "tag", "-f", "base")
    out = await LocalExec().run(SNAPSHOT_CMD.format(base=sha), cwd=str(ws), env={**ENV, "HOME": str(tmp_path)}, timeout_s=60)
    assert out.ok and "+x = 4" in out.stdout
    with pytest.raises(ValueError):
        vcollect.snapshot_command("base")  # the tag is never a snapshot's base


async def test_R6_open_world_records_the_base_commit(tmp_path):
    from crucible.factory.lanes.windows_world import base_commit

    ws, sha = repo(tmp_path)
    assert await base_commit(ws) == sha


# ---- R6: host git in the agent's repository ------------------------------------------------------------------------------------
async def test_R6_final_collect_neutralises_the_agents_filter_driver(tmp_path):
    ws, sha = repo(tmp_path)
    sentinel = tmp_path / "pwned"
    plant_filter(ws, sentinel)
    (ws / "a.py").write_text("x = 5\n")
    got = await vcollect.collect(ws, base=sha)
    assert not sentinel.exists(), "the policy's clean command ran on the host"
    assert "+x = 5" in got["patch"] and "hacked" not in got["patch"]


async def test_R6_repo_broken_refuses_a_redirected_repository(tmp_path):
    ws, sha = repo(tmp_path)
    other, _ = repo(tmp_path, "elsewhere")  # a host repository holding the same base commit
    shutil.rmtree(ws / ".git")
    (ws / ".git").write_text(f"gitdir: {other / '.git'}\n")
    got = await repo_broken(ws, sha)
    assert got and "gitdir" in got[0]


async def test_R6_repo_broken_refuses_core_worktree_and_unnameable_filters(tmp_path):
    ws, sha = repo(tmp_path)
    git(ws, "config", "core.worktree", str(tmp_path))
    assert any("core.worktree" in x for x in await repo_broken(ws, sha))
    ws2, sha2 = repo(tmp_path, "ws2")
    git(ws2, "config", "filter.a=b.clean", "true")  # ``-c filter.a=b.clean=`` would name filter.a, not this driver
    assert any("filter driver" in x for x in await repo_broken(ws2, sha2))


# ---- R6: test-runner plumbing the grader would load -----------------------------------------------------------------------------
@pytest.mark.parametrize("entry,bad", [
    (("A", "src/evil-1.0.dist-info/entry_points.txt"), True),  # a [pytest11] entry point: pytest autoloads it
    (("A", "helper.egg-info/entry_points.txt"), True),
    (("A", "pluggy/__init__.py"), True),  # shadows pytest's hook caller
    (("A", "src/iniconfig.py"), True),
    (("A", "py.py"), True),
    (("M", "py.py"), False),  # the repository's own module, edited
    (("A", "src/_pytest/extra.py"), False),  # pytest's own repository (its project names)
    (("A", "src/mylib/pytest_helpers.py"), False),
])
def test_R6_runner_plumbing_is_a_hack(entry, bad):
    rep = check_paths(PathRules(project_names=("pytest",)), patch="", trail=None, entries=[entry])
    plumb = next(c for c in rep.checks if c.id == "path:plumbing")
    assert (plumb.status == "fail") is bad and (rep.hack_suspect is bad)


# ---- R3: a World the policy killed -----------------------------------------------------------------------------------------------
class FakeWorld:
    def __init__(self):
        self.failures: list[Failure] = []
        self.logged: list[tuple] = []

    def _infra(self, kind, reason, detail=None, *, failure_detail=None):
        self.failures.append(Failure(kind=kind, stage="agent", reason=reason[:500], detail=failure_detail))

    def _log(self, *a, **k):
        self.logged.append(a)


async def dead(_w):
    return False


def watch(ev):
    async def evidence(_w):
        return ev

    return envs.InfraWatch(None, alive=dead, evidence=evidence)


def shell(cmd: str) -> ToolCall:
    return ToolCall(tool_call_id="c1", function_name="repo.shell", arguments={"command": cmd})


async def test_R3_oom_in_its_own_container_is_the_policys_zero():
    w = FakeWorld()
    await watch({"container": "c1", "lost": None, "state": {"Status": "exited", "ExitCode": 137, "OOMKilled": True}}).on_step(
        w, 3, "agent", shell("python -c 'x = bytearray(10**11)'"), None)
    [f] = w.failures
    assert f.kind == "crucible:agent_broke_env" and f.mask is False and f.blame == "policy" and "OOMKilled" in f.reason
    assert f.evidence["last_command"].startswith("python -c")


async def test_R3_kill_command_then_exit_is_the_policys_zero():
    w = FakeWorld()
    await watch({"container": "c1", "lost": None, "state": {"Status": "exited", "ExitCode": 0, "OOMKilled": False}}).on_step(
        w, 5, "agent", shell("cd /w && kill -9 1"), None)
    assert [f.kind for f in w.failures] == ["crucible:agent_broke_env"] and w.failures[0].mask is False


@pytest.mark.parametrize("ev,cmd", [
    ({"container": "c1", "lost": "the task container c1 was stopped from outside (kill; status exited)", "state": {"Status": "exited", "OOMKilled": False}},
     "kill -9 1"),  # stopped through the daemon: never the policy, whatever it ran
    ({"container": "c1", "lost": None, "state": {"Status": "exited", "ExitCode": 137, "OOMKilled": False}}, "ls"),  # no evidence names it
    ({"container": "c1", "lost": "the task container c1 was removed"}, "shutdown -h now"),
    ({}, "kill 1"),  # no container to ask
])
async def test_R3_anything_else_stays_infrastructure(ev, cmd):
    w = FakeWorld()
    await watch(ev).on_step(w, 1, "agent", shell(cmd), None)
    [f] = w.failures
    assert f.kind == "crucible:env_container_lost" and f.mask is None and "evidence" in f.reason


def test_R3_a_policy_killed_world_ends_the_episode_and_scores_zero():
    from crucible.actors.policy import _masking_failure
    from crucible.runtime.harness.native import broke_world

    w = SimpleNamespace(responders={}, failures=[Failure(kind="crucible:agent_broke_env", stage="agent", reason="OOMKilled", mask=False, blame="policy")])
    assert _masking_failure(w) == "crucible:agent_broke_env"  # the loop stops: the container is dead
    ep = SimpleNamespace(failures=[f.model_dump() for f in w.failures])
    sf = broke_world(ep)
    assert sf is not None and sf.failure.kind == "crucible:agent_broke_env" and sf.failure.mask is False
    assert broke_world(SimpleNamespace(failures=[{"kind": "crucible:env_container_lost", "mask": None}])) is None


def test_R3_kill_pattern():
    for c in ("kill -9 1", "kill -1", "kill -s KILL 1", "sudo pkill -f python", "killall5", "shutdown -h now", "rm -rf /", "rm -rf --no-preserve-root /"):
        assert envs.KILL_COMMAND.search(c), c
    for c in ("kill 12345", "pytest -q tests/test_kill.py", "rm -rf /tmp/build", "grep -rn shutdowns ."):
        assert not envs.KILL_COMMAND.search(c), c
