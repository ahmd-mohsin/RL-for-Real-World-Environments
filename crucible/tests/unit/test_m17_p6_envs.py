"""P6 native environments (``factory.envs``; swarm-pipeline/00 §11, §11.3): ``make_env`` (harness, sidecar, honesty, tool
limit), the two timeouts on the native path, ``InfraWatch``, whole-group drop, the portable rows and the objective id,
slime rows refusing held-out tasks, and the Harbor export (validated by Harbor's loader, its grader executed)."""

from __future__ import annotations

import asyncio
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_p6_equiv import HostWindow
from test_m17_w_task import make_task

from crucible.factory.checkpoint import CheckpointConfig
from crucible.factory.envs import (
    HARBOR_GRADER,
    InfraWatch,
    Timeouts,
    drop_infra_groups,
    export_harbor,
    make_env,
    objective_of,
    read_task_rows,
    scripted_policy,
    slime_rows,
    task_row,
    write_task_rows,
)
from crucible.factory.versioning import objective_id


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p6env")
    task, rec, repo = make_task(tmp)
    task.__class__ = HostWindow
    return SimpleNamespace(task=task, rec=rec, repo=repo, tmp=tmp)


def test_make_env_wires_harness_sidecar_honesty_and_limits(toy) -> None:
    env = make_env(toy.task, checkpoint=CheckpointConfig(every=4), tool_limit=20, timeouts=Timeouts(agent_s=60, rollout_s=120))
    assert env.harness.limits.max_steps == 22 and env.harness.limits.timeout_s == 60  # limit + final STATUS write + Done
    assert env.policy_kwargs() == {"tool_limit": 20, "limit_warning": 5}
    spec = env.run_spec(run_id="r", policies=[scripted_policy([("final", "x")])])
    assert spec.attempt_timeout_s == 120 and spec.harnesses == [env.harness]
    side = env.sidecar(SimpleNamespace(task=toy.task, dir=toy.tmp / "a1"))
    assert isinstance(side, InfraWatch) and type(side.inner).__name__ == "CheckpointSidecar" and side.inner.cfg.every == 4
    assert env.objective_id == objective_id({toy.task.task_id: toy.task.verifier_version}) and env.ref.task_id == toy.task.task_id
    assert callable(env.honesty_group_fn)
    with pytest.raises(ValueError):
        Timeouts(agent_s=100, rollout_s=100)


def test_infra_watch_masks_a_dead_world_once() -> None:
    calls = []
    world = SimpleNamespace(_crucible_alive=lambda: False, _infra=lambda kind, why: calls.append((kind, why)))

    async def go() -> dict:
        w = InfraWatch(None)
        await w.on_start(world)
        await w.on_step(world, 1, "agent", None, None)
        await w.on_step(world, 2, "agent", None, None)
        return await w.on_end(world)

    out = asyncio.run(go())
    assert calls == [("crucible:env_container_lost", "the World's sandbox died (after step 1)")] and out["infra_dead"] == "after step 1"


def test_native_two_timeouts(toy, monkeypatch) -> None:
    from crucible.runtime.run import Runner

    o = toy.rec["oracle_files"]
    steps = [("write", "pkg/core.py", o["pkg/core.py"]), ("shell", "true"), ("shell", "true"), ("shell", "true"), ("final", "done")]

    async def run(rid: str, timeouts: Timeouts) -> dict:
        env = make_env(toy.task, checkpoint=CheckpointConfig(every=0, on_commit=False), timeouts=timeouts)
        await Runner(toy.tmp / "runs").run(env.run_spec(run_id=rid, policies=[scripted_policy(steps, delay_s=0.5)]))
        adir = min((toy.tmp / "runs" / rid / "samples").glob("*/a*"))
        return json.loads((adir / "record.json").read_text())

    HostWindow._dead = False
    agent = asyncio.run(run("agent-timeout", Timeouts(agent_s=1.2, rollout_s=600)))
    assert not agent["mask_sample"] and agent["meta"]["stop"] == "timeout" and agent["reward"] == pytest.approx(1 / 6, abs=1e-5)  # m01 only: graded
    real_snapshot = HostWindow.snapshot_patch

    async def slow_snapshot(self, world) -> str:  # grading outlasts rollout_s on any host; setup speed must not decide the outcome
        await asyncio.sleep(5)
        return await real_snapshot(self, world)

    monkeypatch.setattr(HostWindow, "snapshot_patch", slow_snapshot)
    roll = asyncio.run(run("rollout-timeout", Timeouts(agent_s=0.5, rollout_s=0.8)))
    assert roll["mask_sample"] and roll["reward"] is None


def test_drop_infra_groups() -> None:
    r = [SimpleNamespace(mask_sample=False, reward=1.0), SimpleNamespace(mask_sample=True, reward=None), SimpleNamespace(mask_sample=False, reward=0.0),
         SimpleNamespace(mask_sample=False, reward=0.5)]
    kept, groups, refill = drop_infra_groups(r, ["a", "a", "b", "b"])
    assert kept == r[2:] and groups == ["b", "b"] and refill == {"a": 2}


def test_portable_rows_round_trip_and_refuse_held_out(toy, tmp_path: Path) -> None:
    row = task_row(toy.task, runtime_config={"tool_limit": 150})
    assert row["env"] == "crucible/longtask" and row["template"] == "window" and row["split"] == "train" and row["slug"] == toy.task.task_id
    assert row["objective_id"] == objective_of([toy.task]) and row["args"]["ref"]["verifier_version"] == toy.task.verifier_version
    info = write_task_rows([row], tmp_path / "rows.jsonl")
    assert info["rows"] == 1 and read_task_rows(tmp_path / "rows.jsonl") == [row]
    held = {**row, "split": "heldout"}
    with pytest.raises(ValueError, match="sealed benchmark"):
        write_task_rows([held], tmp_path / "h.jsonl")
    (srow,) = slime_rows([toy.task], agent="claude_code", workdir="/w")
    md = srow["metadata"]
    assert md["split"] == "train" and md["crucible_task_ref"]["task_id"] == toy.task.task_id and md["objective_id"] == row["objective_id"]
    toy.task.window["split"] = "heldout"
    try:
        with pytest.raises(ValueError, match="may not train"):
            slime_rows([toy.task])
    finally:
        toy.task.window["split"] = "train"
    with pytest.raises(ValueError, match="may not train"):
        slime_rows([toy.task], sealed={toy.task.task_id})


def test_harbor_export_validates_and_its_grader_scores_like_grade_patch(toy, tmp_path: Path) -> None:
    pytest.importorskip("harbor")
    info = export_harbor(toy.task, tmp_path / "harbor", timeouts=Timeouts(agent_s=600, rollout_s=1200))
    assert info["verifier_mode"] == "separate" and info["network_mode"] == "no-network" and not info["ignored_keys"]
    toml = (tmp_path / "harbor" / "task.toml").read_text()
    assert "[[verifier.collect]]" in toml and "crucible_ref" in toml and (tmp_path / "harbor" / "environment" / "repo" / "pkg" / "core.py").exists()
    assert "gamma" not in (tmp_path / "harbor" / "environment" / "repo" / "pkg" / "core.py").read_text()  # the agent's World is A
    spec = json.loads((tmp_path / "harbor" / "tests" / "milestones.json").read_text())
    # run the exported grader on the host against A + a patch, logs redirected: it must give grade_patch's reward
    for name, patch in (("oracle", toy.task.oracle_patch()), ("empty", "")):
        app = tmp_path / f"app-{name}"
        subprocess.run(["cp", "-R", str(toy.task.base_dir), str(app)], check=True)
        logs = tmp_path / f"logs-{name}"
        tests = tmp_path / f"tests-{name}"
        tests.mkdir()
        (tests / "milestones.json").write_text(json.dumps({**spec, "workdir": str(app)}))
        (tests / "crucible_grade.py").write_text(HARBOR_GRADER.replace("/logs/verifier", str(logs)).replace("/tmp/crucible-junit.xml", str(tmp_path / f"j-{name}.xml")))
        pf = tmp_path / f"{name}.patch"
        pf.write_text(patch)
        subprocess.run([sys.executable, str(tests / "crucible_grade.py"), str(pf)], cwd=app, capture_output=True, check=False)
        got = json.loads((logs / "reward.json").read_text())
        want = asyncio.run(toy.task.grade_patch(patch, trail=[]))
        assert got["reward"] == pytest.approx(want.reward) and got["milestones"] == want.meta["milestones"], (name, got)
