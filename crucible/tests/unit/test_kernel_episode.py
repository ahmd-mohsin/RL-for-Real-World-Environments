"""Kernel episodes on a GPU host (swarm-pipeline/10 §9): the container-exec World backend, the host's locks, the World's
template and container flags, the episode task's contract, the grader client, the workspace pull and the targeting glue."""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from crucible.factory.lanes import kernel_episode as ke

HOST = ke.host_of("rtx5090")
SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def bug_rec(**kw):
    rec = {"id": "kd-bug-x-1", "family": "kernel-bug", "repo": "ggml-org/llama.cpp", "split": "train", "b": "b" * 40,
           "layout": {"cuda_dir": "ggml/src/ggml-cuda/", "dispatch_file": "ggml/src/ggml-cuda/ggml-cuda.cu"}, "harness": {"dialect": "ggml-2025"},
           "prompt": "Ops fail on the GPU `sm_{ARCH}`.\n- [R1] SOFT_MAX races.\n- [R2] NORM fails on big tensors.",
           "requirements": [{"id": "R1", "milestone": "m1", "text": "x"}, {"id": "R2", "milestone": "m2", "text": "y"}],
           "milestones": [{"id": "m1", "kind": "bug", "class": "race", "tests": ["SOFT_MAX(a)"], "files": ["f.cu"], "ops": ["SOFT_MAX"], "requires": [], "checks": ["golden"]},
                          {"id": "m2", "kind": "bug", "class": "indexing", "tests": ["NORM(b)"], "files": ["g.cu"], "ops": ["NORM"], "requires": ["m1"], "checks": ["golden"]}]}
    rec.update(kw)
    return rec


def mig_rec():
    return {"id": "km-cuda13-hecbench-abc-2", "family": "kernel-migration", "repo": "zjin-lcf/HeCBench", "split": "train", "prompt": "Build with CUDA 13.",
            "requirements": [{"id": "R1", "milestone": "c01", "text": "compiles"}],
            "milestones": [{"id": "c01", "kind": "compile", "files": ["src/a/main.cu"], "programs": ["src_a_main"], "requires": []},
                           {"id": "r01", "kind": "run", "files": ["src/a/main.cu"], "programs": ["src_a_main"], "requires": ["c01"]}]}


def test_container_exec_backend_routes_every_app_into_the_running_container():
    from crucible.core.registry import get_registry
    from crucible.world.exec import DockerExec

    cls = get_registry().get("world_backend", "container_exec")
    b = cls(ke.world_spec(), {"container": "crucible-lh-k-ep-x", "docker_host": "ssh://u@h", "base": "/ws", "gpus": True, "network_enforced": True}, workdir=Path("/tmp"))
    ex = b.exec_for(ke.world_spec().apps[0])
    assert isinstance(ex, DockerExec) and ex.container == "crucible-lh-k-ep-x" and ex.docker_host == "ssh://u@h"
    assert b.base_path(ke.world_spec().apps[0]) == "/ws" and b.capabilities.gpus and b.capabilities.network_enforced and b.validate() == []
    assert cls(ke.world_spec(), {}, workdir=Path("/tmp")).validate()  # no container: refused


def test_the_gpu_lock_is_shared_for_correctness_and_exclusive_for_perf_and_builds_take_the_build_lock():
    lock = ke.GpuLock(HOST)
    assert "flock -s" in lock.wrap("grade") and "build.lock" in lock.wrap("grade")
    assert "flock -x" in lock.wrap("grade", exclusive=True)
    assert "build.lock" not in lock.wrap("run", build=False)


def test_last_json_reads_the_runner_s_result_line_after_its_notes():
    out = 'building image\n{"not": "this"}\nmore notes\n{"reward": 1.0, "gate_status": {"m1": "pass"}}\n'
    assert ke.last_json(out) == {"reward": 1.0, "gate_status": {"m1": "pass"}}
    assert ke.last_json("no json here") is None


def test_the_template_is_a_prebuilt_start_tree_built_as_the_host_user_under_the_build_lock():
    t, script = ke.template_script(HOST, bug_rec())
    assert t.startswith(HOST.work + "/tmpl/kd-bug-x-1-120-")
    for needle in ("--runtime=nvidia", "--network none", "--user $(id -u):$(id -g)", "a.tar.gz", "build.lock", ".ready", "test-backend-ops",
                   "120a-real", "-DGGML_CUDA=ON"):
        assert needle in script, needle
    _, mscript = ke.template_script(HOST, mig_rec())
    assert "base.tar.gz" in mscript and "cmake" not in mscript  # a migration's programs are built by the agent
    assert ke.world_image(HOST, mig_rec()) == "crucible-lh-k:base-13.0.2" and ke.world_image(HOST, bug_rec()) == HOST.image
    assert ke.template_key(HOST, bug_rec()) != ke.template_key(HOST, bug_rec(id="kd-bug-x-2"))


def test_the_world_container_has_the_gpu_no_network_limits_and_a_read_only_template():
    argv = ke.world_run_argv(HOST, "crucible-lh-k-ep-abc", "/t", "img:1")
    s = " ".join(argv)
    for needle in ("--runtime=nvidia", "NVIDIA_VISIBLE_DEVICES=all", "--network none", f"--memory {HOST.world_memory}", f"--cpus {HOST.world_cpus}",
                   "--pids-limit", "/t/ws:/tmpl:ro", "img:1"):
        assert needle in s, needle
    assert argv[argv.index("--name") + 1].startswith(ke.PREFIX)


def test_the_episode_task_offers_the_checkpoint_and_honesty_contract_for_both_record_shapes():
    from crucible.factory.checkpoint import CheckpointConfig, MilestoneTask

    t = ke.KernelEpisodeTask(bug_rec(), HOST)
    assert isinstance(t, MilestoneTask)
    assert t.requirement_ids == ["R1", "R2"] and t.requirement_milestones == {"R1": ["m1"], "R2": ["m2"]}
    assert [m["requires"] for m in t.milestones] == [[], ["m1"]]
    assert "sm_120" in t.instruction and "/ws/build" in t.instruction and "{ARCH}" not in t.instruction
    assert t.verifier_version.endswith("+rtx5090") and t.language == "cuda" and t.split == "train"
    assert not t.checkpoint_defer and t.episode_checkpoint(CheckpointConfig()).defer is False
    perf = ke.KernelEpisodeTask(bug_rec(milestones=[{**bug_rec()["milestones"][0], "kind": "perf"}]), HOST)
    assert perf.checkpoint_defer and perf.episode_checkpoint(CheckpointConfig()).defer is True  # a perf task's checkpoints wait for the end
    m = ke.KernelEpisodeTask(mig_rec(), HOST)
    assert [x["id"] for x in m.milestones] == ["c01", "r01"] and m.intent == "toolkit_migration" and "/ws/build" not in m.instruction


def test_a_grade_maps_the_runner_s_gate_status_onto_milestones_and_masks_what_cannot_run(monkeypatch):
    calls = []

    async def fake_bundle(host, *, migration=False):
        return f"{host.work}/{'mbundle' if migration else 'bundle'}"

    async def fake_remote(host, script, *, stdin=None, timeout=0, retries=0):
        calls.append((script, stdin))
        return ke.Remote(0, 'note\n{"reward": 0.5, "infra": {"status": "ok"}, "gate_status": {"m1": "pass", "m2": "fail"}, "grade_s": 90}\n', "", 1.0)

    monkeypatch.setattr(ke, "ensure_episode_bundle", fake_bundle)
    monkeypatch.setattr(ke, "remote", fake_remote)
    t = ke.KernelEpisodeTask(bug_rec(), HOST)
    rep = asyncio.run(t.grade_patch("diff --git a/x b/x\n"))
    assert rep.meta["milestones"] == {"m1": True, "m2": False} and rep.infra["status"] == "ok"
    assert "--grade-one kd-bug-x-1" in calls[0][0] and "flock -s" in calls[0][0] and calls[0][1].startswith("diff --git")
    assert asyncio.run(t.milestone_status("p")) == {"m1": True, "m2": False}
    assert t.backend.grades and t.backend.grades[0]["grade_s"] == 90

    async def dead_remote(host, script, *, stdin=None, timeout=0, retries=0):
        return ke.Remote(255, "", "ssh: connect timed out", 1.0)

    monkeypatch.setattr(ke, "remote", dead_remote)
    rep = asyncio.run(t.grade_patch("p"))
    assert rep.infra["status"] == "masked"  # an unreachable host is infrastructure, never a 0 (V8)
    with pytest.raises(RuntimeError):
        asyncio.run(t.milestone_status("p"))


def test_the_workspace_pull_streams_tracked_and_untracked_files_and_git_but_not_ignored_builds(tmp_path, monkeypatch):
    repo = tmp_path / "ws"
    repo.mkdir()
    sh = lambda *c: subprocess.run(list(c), cwd=repo, check=True, capture_output=True)  # noqa: E731
    sh("git", "init", "-q", "-b", "main")
    (repo / ".gitignore").write_text("/build*\n")
    (repo / "src.cu").write_text("a\n")
    sh("git", "add", "-A")
    sh("git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    (repo / "new.cu").write_text("b\n")
    (repo / "build").mkdir()
    (repo / "build" / "big.o").write_text("x" * 100)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text("#!/bin/sh\nshift 2\nexec \"$@\"\n")  # docker exec <container> bash -c CMD  ->  bash -c CMD, here
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{bin_dir}:{os.environ['PATH']}"}
    dest = tmp_path / "mirror"
    dest.mkdir()
    ke.pull_tree("c", str(repo), dest, env=env)
    assert (dest / "src.cu").read_text() == "a\n" and (dest / "new.cu").exists() and (dest / ".git" / "HEAD").exists()
    assert not (dest / "build").exists()  # gitignored build outputs stay in the World


def load_script(name: str):
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_runner_s_episode_result_carries_the_gate_status_and_the_gpu():
    g = load_script("lh_kernel_debug_gpu")
    r = g.episode_result({"reward": 1.0, "infra": {"status": "ok"}, "gates": {"m1": "pass"}}, gpu="RTX", seconds=12.34)
    assert r == {"reward": 1.0, "infra": {"status": "ok"}, "gate_status": {"m1": "pass"}, "components": {}, "gpu": "RTX", "grade_s": 12.3}
    assert g.episode_result({}, gpu="RTX", seconds=1)["infra"]["status"] == "masked"


def test_kernel_targeting_picks_the_tasks_passing_on_the_host_s_gpu_and_labels_skills_coarsely(tmp_path):
    kt = load_script("lh_kernel_target")
    funnel = tmp_path / "f.json"
    funnel.write_text(json.dumps({"tasks": [{"id": "a", "gpu_pass": True, "gpu": "NVIDIA GeForce RTX 5090"}, {"id": "b", "gpu_pass": False},
                                            {"id": "c", "gpu_pass": False, "gpu_by_hardware": {"NVIDIA GeForce RTX 5090": {"pass": True}}},
                                            {"id": "d", "gpu_pass": True, "gpu": "NVIDIA GH200 120GB"}]}))
    assert kt.gpu_passing(funnel, "NVIDIA GeForce RTX 5090") == ["a", "c"]
    sk = kt.kernel_failure_skill({"cause": "skill_gap"}, {"skills": ["kernel_fault_localization"]})
    assert sk["on_intent"] and sk["skill"] == "kernel_fault_localization"
    assert not kt.kernel_failure_skill({"cause": "budget"}, {"skills": ["x"]})["on_intent"]
    a = kt.parse_args(["student", "--host", "rtx5090", "--tasks", "kd-bug-x-1", "--k", "1"])
    assert a.host == "rtx5090" and a.pilot_task == kt.PILOT_TASK and a.out.endswith("kernel-rtx5090") and a.kernel.endswith("kernel_debug")


def test_a_task_may_defer_its_own_checkpoints_through_the_env():
    from crucible.factory.checkpoint import CheckpointConfig
    from crucible.factory.envs import make_env

    t = ke.KernelEpisodeTask(bug_rec(milestones=[{**bug_rec()["milestones"][0], "kind": "perf"}]), HOST)
    env = make_env(t, checkpoint=CheckpointConfig(every=20, cap=15), tool_limit=40)
    side = env.sidecar(SimpleNamespace(task=t, dir=Path("/tmp/crucible-ke-test")))
    assert side.inner.cfg.defer is True and side.inner.cfg.every == 20
