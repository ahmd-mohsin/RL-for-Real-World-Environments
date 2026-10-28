"""A task container lost mid-episode is infrastructure, never the policy's failure (M1 audit 2026-10-14 R-1): the docker
CLI's "No such container" counts only once the daemon confirms the loss, and the World records env_container_lost."""

from __future__ import annotations

import asyncio
import json
import subprocess
import uuid
from pathlib import Path

import pytest

from crucible.core.schema.actions import ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.world import exec as exec_mod
from crucible.world.apps.base import ToolError
from crucible.world.exec import DockerExec, EnvContainerLost, ExecError, ExecOut
from crucible.world.world import World

GONE = ExecOut(1, "", "Error response from daemon: No such container: c1")


def fake_docker(monkeypatch, *, out: ExecOut, inspect: ExecOut | None = None, events: str = "") -> list[str]:
    """``_run_subprocess`` answering ``docker exec`` / ``cp`` with ``out``, ``inspect`` and ``events`` as given; returns the verbs run."""
    verbs: list[str] = []

    async def run(argv, *, cwd, env, timeout_s, stdin, sessions=None):
        verbs.append(argv[1])
        return {"inspect": inspect, "events": ExecOut(0, events)}.get(argv[1], out)

    monkeypatch.setattr(exec_mod, "_run_subprocess", run)
    return verbs


def state(**kw) -> ExecOut:
    return ExecOut(0, json.dumps({"Running": False, "Paused": False, "Status": "exited", "StartedAt": "2026-10-19T08:00:00Z", **kw}))


async def test_a_removed_container_raises_once_the_daemon_confirms(monkeypatch):
    verbs = fake_docker(monkeypatch, out=GONE, inspect=ExecOut(1, "", "Error: No such object: c1"))
    with pytest.raises(EnvContainerLost, match="c1 was removed") as e:
        await DockerExec("c1").run("ls")
    assert verbs == ["exec", "inspect", "events"] and e.value.failure_kind == "crucible:env_container_lost"  # no die seen: not the policy's


async def test_the_words_alone_never_decide(monkeypatch):
    # the policy prints the daemon's message and exits 1; the container runs, so this is its own failed command
    fake_docker(monkeypatch, out=ExecOut(1, "No such container: c1"), inspect=state(Running=True, Status="running"))
    assert (await DockerExec("c1").run("echo 'No such container: c1'; exit 1")).rc == 1
    verbs = fake_docker(monkeypatch, out=ExecOut(2, "", "ls: cannot access 'x'"))
    assert (await DockerExec("c1").run("ls x")).rc == 2 and verbs == ["exec"]  # an ordinary failure asks the daemon nothing


async def test_stopped_from_outside_is_lost_but_a_container_that_exited_by_itself_is_not(monkeypatch):
    stopped = ExecOut(1, "", "Error response from daemon: container 3f2a is not running")
    fake_docker(monkeypatch, out=stopped, inspect=state(), events="start\nkill\ndie\n")
    with pytest.raises(EnvContainerLost, match=r"stopped from outside \(kill; status exited\)"):
        await DockerExec("c1").run("true")
    fake_docker(monkeypatch, out=stopped, inspect=state(OOMKilled=True), events="start\noom\ndie\n")  # its own process: the episode's doing
    assert (await DockerExec("c1").run("true")).rc == 1


async def test_a_removed_container_is_lost_only_on_evidence(monkeypatch):
    # --rm deletes a container whose main process exited: the events tell who ended it (M1 audit R-6)
    gone = ExecOut(1, "", "Error: No such object: c1")
    fake_docker(monkeypatch, out=GONE, inspect=gone, events="start\ndie\ndestroy\n")  # died from inside: the policy's doing
    assert (await DockerExec("c1").run("true")).rc == 1
    fake_docker(monkeypatch, out=GONE, inspect=gone, events="start\nkill\ndie\ndestroy\n")  # docker rm -f
    with pytest.raises(EnvContainerLost, match="was removed"):
        await DockerExec("c1").run("true")
    fake_docker(monkeypatch, out=GONE, inspect=gone, events="")  # no evidence either way: not the policy's
    with pytest.raises(EnvContainerLost, match="was removed"):
        await DockerExec("c1").run("true")
    ex = DockerExec("c1")
    ex.expires_at = 0.0  # its lifetime ran out: the sandbox's end, not the policy's
    fake_docker(monkeypatch, out=GONE, inspect=gone, events="start\ndie\ndestroy\n")
    with pytest.raises(EnvContainerLost, match="end of its lifetime"):
        await ex.run("true")
    fake_docker(monkeypatch, out=ExecOut(1, "", "Error response from daemon: container 3f2a is not running"), inspect=state(), events="die\n")
    with pytest.raises(EnvContainerLost, match="end of its lifetime"):
        await ex.run("true")


async def test_paused_and_unreachable_daemons_are_lost(monkeypatch):
    fake_docker(monkeypatch, out=ExecOut(1, "", "Error response from daemon: container 3f2a is paused, unpause the container before exec"),
                inspect=state(Running=True, Paused=True, Status="paused"))
    with pytest.raises(EnvContainerLost, match="paused from outside"):
        await DockerExec("c1").run("true")
    down = ExecOut(1, "", "error during connect: Get \"http://docker.example/v1.47/containers/c1/json\": EOF")
    fake_docker(monkeypatch, out=down, inspect=down)
    from crucible.world import hostreach
    from crucible.world.exec import HostUnreachable

    async def answers() -> bool:
        return True

    hostreach.register("docker:ssh://lab", hostreach.HostReach("lab", answers))  # the host answers: the daemon's own failure, as before
    with pytest.raises(EnvContainerLost, match="daemon cannot be reached"):
        await DockerExec("c1", docker_host="ssh://lab").run("true")
    hostreach.reset()  # 2026-10-26: a host that does not answer its probe either is an outage (world.hostreach), paused for, then typed
    with pytest.raises(HostUnreachable, match="host_unreachable"):
        await DockerExec("c1", docker_host="ssh://lab").run("true")


async def test_file_copies_check_too(monkeypatch, tmp_path: Path):
    fake_docker(monkeypatch, out=GONE, inspect=ExecOut(1, "", "Error: No such object: c1"))
    with pytest.raises(EnvContainerLost):
        await DockerExec("c1").pull_file("/a", tmp_path / "a")
    fake_docker(monkeypatch, out=ExecOut(1, "", "Error: Could not find the file /a in container c1"))
    with pytest.raises(ExecError):  # a missing file is still the caller's
        await DockerExec("c1").pull_file("/a", tmp_path / "a")


async def test_the_world_records_a_lost_container_as_infrastructure(tmp_path: Path):
    spec = WorldSpec.model_validate({"world": "test/lost", "actors": [{"id": "agent", "kind": "policy", "tools": ["ws.*"]}],
                                     "apps": [{"id": "ws", "template": "common/workspace", "state": {"adapter": "fs", "params": {"root": "ws"}}}]})
    w = World(spec, store=Store(tmp_path / "s"), workdir=tmp_path / "w")
    await w.start()
    await w.reset()
    try:
        async def wrapped(**kw):  # a file tool turns exec errors into ToolError (WorkspaceApp._ws)
            try:
                raise EnvContainerLost("the task container c1 was removed")
            except EnvContainerLost as e:
                raise ToolError("No such container") from e

        async def direct(**kw):  # the shell lets it through
            raise EnvContainerLost("the Docker daemon cannot be reached")

        w.apps["ws"].tool_list_dir, w.apps["ws"].tool_read_file = wrapped, direct
        for name, args in (("ws.list_dir", {}), ("ws.read_file", {"path": "a"})):
            r = (await w.step("agent", ToolCall(tool_call_id="c", function_name=name, arguments=args))).observation.tool_results[0]
            assert not r.ok and r.fault == "infra" and "container was lost" in r.content
        assert [f.kind for f in w.failures] == ["crucible:env_container_lost"] * 2 and {f.who for f in w.failures} == {"infra"}
        assert "c1 was removed" in w.failures[0].reason
    finally:
        await w.close()


def _docker(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=120, check=False)


@pytest.mark.docker
async def test_a_real_container_lost_mid_episode():
    name = f"crucible-r1-{uuid.uuid4().hex[:8]}"
    assert _docker("run", "-d", "--name", name, "python:3.12-slim", "sleep", "300").returncode == 0
    try:
        ex = DockerExec(name)
        assert (await ex.run("echo 'No such container: x' >&2; exit 1")).rc == 1  # forged inside a live container: scored
        _docker("kill", name)
        with pytest.raises(EnvContainerLost, match="stopped from outside"):
            await ex.run("true")
        _docker("rm", "-f", name)
        with pytest.raises(EnvContainerLost, match="was removed"):
            await ex.run("true")
        assert _docker("run", "-d", "--name", name, "python:3.12-slim", "sleep", "1").returncode == 0
        await asyncio.sleep(3)  # its own main process exited: the episode's doing, not an outage
        assert (await ex.run("true")).rc != 0
    finally:
        _docker("rm", "-f", name)
