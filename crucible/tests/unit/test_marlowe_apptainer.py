"""The Apptainer sandbox and World backend (world/backends/apptainer.py) against a fake ``apptainer`` runner: the argv of
every lifecycle call, the isolation flags, the slime four-method contract through ``ExecSandbox``, a dead instance as
``SandboxTerminatedError``, and the Marlowe stage driver's guards (scripts/marlowe/stage.sh, plan ``selftest``)."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest

from crucible.core.registry import get_registry
from crucible.core.schema.world import WorldSpec
from crucible.runtime.harness.sandbox import ExecSandbox, SandboxTerminatedError
from crucible.world.backends.apptainer import ApptainerBackend, ApptainerSandbox, run_once
from crucible.world.exec import ExecError, ExecOut

ROOT = Path(__file__).resolve().parents[2]


class FakeApptainer:
    """Records every argv; ``exec`` into a live instance runs the command's bash script on this host (so files and exit
    codes behave), ``instance list --json`` reports the live set, ``instance start`` refuses cgroup flags when told to."""

    def __init__(self, *, refuse_cgroups: bool = False):
        self.calls: list[list[str]] = []
        self.envs: list[dict | None] = []
        self.live: set[str] = set()
        self.refuse_cgroups = refuse_cgroups

    async def __call__(self, argv, *, cwd, env, timeout_s, stdin, **_):
        self.calls.append(list(argv))
        self.envs.append(env)
        a = argv[1:]
        if a[:2] == ["instance", "start"]:
            if self.refuse_cgroups and any(x in a for x in ("--memory", "--cpus", "--pids-limit")):
                return ExecOut(rc=255, stdout="", stderr="FATAL: while executing starter: failed to start instance")
            self.live.add(a[-1])
            return ExecOut(rc=0, stdout="INFO: instance started successfully")
        if a[:2] == ["instance", "stop"]:
            self.live.discard(a[-1])
            return ExecOut(rc=0, stdout="")
        if a[:2] == ["instance", "list"]:
            return ExecOut(rc=0, stdout=json.dumps({"instances": [{"instance": n, "pid": 1, "img": "x.sif"} for n in sorted(self.live)]}))
        if a[0] == "exec":
            target = next(x for x in a if x.startswith("instance://") or x.endswith(".sif"))
            if target.startswith("instance://") and target.removeprefix("instance://") not in self.live:
                return ExecOut(rc=255, stdout="", stderr=f"FATAL:   While setting image/instance: no instance found with name {target[11:]}")
            script = a[a.index("-c") + 1]
            pwd = a[a.index("--pwd") + 1] if "--pwd" in a else None
            p = subprocess.run(["bash", "-c", script], input=stdin, capture_output=True, timeout=60, check=False,  # noqa: ASYNC221 - a fake, on purpose
                               cwd=pwd if pwd and Path(pwd).is_dir() else None)
            return ExecOut(rc=p.returncode, stdout=p.stdout.decode(errors="replace"), stderr=p.stderr.decode(errors="replace"))
        raise AssertionError(f"unexpected apptainer call {argv}")


def spec(network: str = "no-network", gpus: int = 0) -> WorldSpec:
    app = {"id": "repo", "template": "common/git_repo", "state": {"adapter": "fs", "params": {"root": "repo"}}, "params": {"shell": True}}
    if gpus:
        app["resources"] = {"gpus": gpus}
    return WorldSpec.model_validate({"world": "test/apptainer", "network": network, "backend": "apptainer", "apps": [app],
                                     "actors": [{"id": "agent", "kind": "policy", "tools": ["repo.*"]}]})


async def test_instance_start_flags_are_the_isolation_contract(tmp_path):
    fake = FakeApptainer()
    sb = ApptainerSandbox("/s/world.sif", binds=[(str(tmp_path), str(tmp_path), "rw")], configdir=str(tmp_path / "cfg"), runner=fake)
    await sb.start()
    start = fake.calls[0]
    assert start[:3] == ["apptainer", "instance", "start"]
    for flag in ("--containall", "--cleanenv", "--no-home", "--writable-tmpfs"):
        assert flag in start
    assert start[start.index("--net"):start.index("--net") + 3] == ["--net", "--network", "none"]
    assert f"{tmp_path}:{tmp_path}:rw" in start and "--nv" not in start
    assert start[-2:] == ["/s/world.sif", sb.name] and sb.name.startswith("crucible-apt-")
    assert fake.envs[0] == {"APPTAINER_CONFIGDIR": str(tmp_path / "cfg")}  # instance state off the (full) home directory
    assert await sb.alive()
    await sb.close()
    assert fake.calls[-1] == ["apptainer", "instance", "stop", sb.name] and not await sb.alive()


async def test_gpu_overlay_and_host_network():
    sb = ApptainerSandbox("docker://nvidia/cuda:12.8.1-devel-ubuntu24.04", gpus=1, network="host", overlay="/s/ov.img")
    a = sb.argv()
    assert "--nv" in a and "--net" not in a and a[a.index("--overlay") + 1] == "/s/ov.img" and "--writable-tmpfs" not in a
    assert "1 GPU(s) via --nv" in sb.description
    with pytest.raises(ValueError):
        ApptainerSandbox("x.sif", network="bridge")
    with pytest.raises(ValueError):
        ApptainerSandbox("x.sif", binds=[("/a", "/a", "rwx")])


async def test_cgroup_limits_fall_back_when_the_host_refuses(tmp_path):
    fake = FakeApptainer(refuse_cgroups=True)
    sb = ApptainerSandbox("w.sif", limits={"memory": "4G", "cpus": 2, "pids": 512}, runner=fake)
    await sb.start()
    assert "--memory" in fake.calls[0] and "--pids-limit" in fake.calls[0] and "--memory" not in fake.calls[1]
    assert sb.cgroups_applied is False and "no cgroup limits" in sb.description
    strict = ApptainerSandbox("w.sif", limits={"memory": "4G"}, cgroups="on", runner=FakeApptainer(refuse_cgroups=True))
    with pytest.raises(RuntimeError, match="failed to start"):
        await strict.start()
    ok = ApptainerSandbox("w.sif", limits={"pids": 64}, runner=FakeApptainer())
    await ok.start()
    assert ok.cgroups_applied is True


async def test_exec_cleans_env_exports_values_and_refuses_user_switch(tmp_path):
    fake = FakeApptainer()
    sb = ApptainerSandbox("w.sif", env={"PYTHONPATH": "/w/src:/w"}, runner=fake)
    await sb.start()
    out = await sb.exec.run("echo $HOME $PYTHONPATH $A", cwd=str(tmp_path), env={"A": "x,y z"}, timeout_s=30)
    argv = fake.calls[-1]
    assert argv[:3] == ["apptainer", "exec", "--cleanenv"] and argv[argv.index("--pwd") + 1] == str(tmp_path)
    assert f"instance://{sb.name}" in argv
    assert out.stdout.split() == ["/tmp", "/w/src:/w", "x,y", "z"]  # commas survive (apptainer --env would split them)
    assert "timeout -s KILL 30" in argv[-1]
    with pytest.raises(ExecError, match="invoking user"):
        await sb.exec.run("id", user="agent")


async def test_files_round_trip_binary_safe(tmp_path):
    fake = FakeApptainer()
    sb = ApptainerSandbox("w.sif", runner=fake)
    await sb.start()
    blob = bytes(range(256)) * 3
    src = tmp_path / "in.bin"
    src.write_bytes(blob)
    dest = tmp_path / "inside" / "deep" / "x.bin"
    await sb.exec.push_file(src, str(dest))
    assert dest.read_bytes() == blob
    back = tmp_path / "back.bin"
    await sb.exec.pull_file(str(dest), back)
    assert back.read_bytes() == blob
    assert await sb.exec.run_python("def main(a):\n    return {'n': a['n'] + 1}\n", {"n": 41}) == {"n": 42}


async def test_slime_contract_through_exec_sandbox_and_dead_instance(tmp_path):
    fake = FakeApptainer()
    sb = ApptainerSandbox("w.sif", binds=[(str(tmp_path), str(tmp_path), "rw")], runner=fake)
    await sb.start()
    async with ExecSandbox(sb.exec, root=str(tmp_path), sandbox_id=sb.name, alive=sb.alive) as box:
        await box.write_file("pkg/mod.py", "X = 1\n")
        assert (tmp_path / "pkg" / "mod.py").read_text() == "X = 1\n"
        assert await box.read_file("pkg/mod.py") == "X = 1\n"
        assert await box.read_file("missing.txt") == ""
        rc, out, _ = await box.exec("cat pkg/mod.py && exit 3")
        assert rc == 3 and out == "X = 1\n"
        await sb.close()  # the node lost the instance: infrastructure, never a 0
        with pytest.raises(SandboxTerminatedError):
            await box.exec("true")
        assert box.terminated


async def test_run_once_is_a_fresh_isolated_container(tmp_path):
    fake = FakeApptainer()
    out = await run_once("g.sif", "echo graded", binds=[(str(tmp_path), "/g", "ro")], cwd="/g", gpus=1, runner=fake)
    a = fake.calls[0]
    assert out.stdout.strip() == "graded"
    assert a[:2] == ["apptainer", "exec"] and "--containall" in a and "--nv" in a and "--network" in a and f"{tmp_path}:/g:ro" in a
    assert not any(x.startswith("instance://") for x in a)


async def test_backend_registered_validated_and_binds_the_apps_dir(tmp_path):
    assert get_registry().get("world_backend", "apptainer") is ApptainerBackend
    fake = FakeApptainer()
    be = ApptainerBackend(spec(gpus=1), {"image": "w.sif", "configdir": str(tmp_path / "c")}, workdir=tmp_path, runner=fake)
    assert be.validate() == [] and be.capabilities.network_enforced and be.capabilities.isolation == "container"
    await be.provision()
    start = fake.calls[0]
    apps = str((tmp_path / "apps").resolve())
    assert f"{apps}:{apps}:rw" in start and "--nv" in start and "--network" in start
    assert be.base_path(spec().apps[0]) == apps and be.exec_for(spec().apps[0]) is be.sandbox.exec
    await be.teardown()
    assert not (tmp_path / "apps").exists() and fake.calls[-1][:3] == ["apptainer", "instance", "stop"]
    assert ApptainerBackend(spec(network="public"), {}, workdir=tmp_path, runner=fake).validate()[0].startswith("apptainer backend needs")


# ---- the stage driver (scripts/marlowe/stage.sh) ---------------------------------------------------------------------------
def _driver(tmp_path: Path, **env: str) -> subprocess.Popen:
    root = tmp_path / "root"
    (root / ".setup").mkdir(parents=True, exist_ok=True)
    e = {**os.environ, "CRU_ROOT": str(root), "CRU_CODE": str(ROOT), "CRU_ENV": str(Path(os.environ.get("VIRTUAL_ENV", "/nonexistent"))),
         "PY": shutil.which("python3") or "python3", "CRU_RUN": str(tmp_path / "run"), "CRU_STAGE_TRIES": "1", **env}
    e.pop("SLURM_JOB_ID", None)
    return subprocess.Popen(["bash", str(ROOT / "scripts" / "marlowe" / "stage.sh"), "selftest"], env=e, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
def test_stage_driver_guards_and_forwards_term(tmp_path):
    run = tmp_path / "run"
    # 1. a TERM while the long stage runs: forwarded, recorded, the stage not marked done
    p = _driver(tmp_path, CRU_SELFTEST_SLEEP="60", CRU_SELFTEST_ONLY="sleeper")
    t0 = time.time()
    while not (run / "logs" / "sleeper.log").exists() and time.time() - t0 < 60:
        time.sleep(0.2)
    time.sleep(1.0)
    os.killpg(p.pid, signal.SIGTERM)
    out = p.communicate(timeout=60)[0].decode()
    assert p.returncode == 143, out
    assert not (run / "markers" / "sleeper.done").exists()
    intr = [json.loads(x) for x in (run / "interruptions.jsonl").read_text().splitlines()]
    assert intr[-1]["stage"] == "sleeper" and intr[-1]["signal"] == "TERM"
    # 2. the restart (a requeue) finishes it; a finished stage never runs again
    p = _driver(tmp_path, CRU_SELFTEST_SLEEP="0", CRU_SELFTEST_ONLY="sleeper")
    out = p.communicate(timeout=60)[0].decode()
    assert p.returncode == 0, out
    done = json.loads((run / "markers" / "sleeper.done").read_text())
    assert done["stage"] == "sleeper"
    p = _driver(tmp_path, CRU_SELFTEST_SLEEP="0", CRU_SELFTEST_ONLY="sleeper")
    out = p.communicate(timeout=60)[0].decode()
    assert "sleeper: done before, skipped" in out
    assert json.loads((run / "summary.json").read_text())["stages"]["sleeper"]["status"] == "done"
