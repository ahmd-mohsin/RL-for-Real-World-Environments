"""Quarantine of training tasks that teach nothing (``train.quarantine``) and its use in the Tinker trainer's scheduling.

A fake sampler drives the backend-neutral rule step by step; then the real ``TinkerTrainer`` loop runs on fake
environments whose groups come from a scripted sampler (``_group`` overridden) and whose update is a no-op (``_step``
overridden), on the fake Tinker service."""

from __future__ import annotations

import asyncio
import json
import statistics
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from crucible.core.schema.rollout import RolloutRecord
from crucible.train.bridges.tinker.fake import FakeTinker
from crucible.train.bridges.tinker.trainer import Group, RunRefused, TinkerTrainer, TrainConfig
from crucible.train.chat import ByteChatCodec
from crucible.train.quarantine import QuarantineConfig, TaskQuarantine, flat

MODEL = "Qwen/Qwen3.6-35B-A3B"

# a fake sampler: task -> the (rewards, final phis) of its groups, in order
FLAT0 = ([0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0])
PHI_SPREAD = ([0.0, 0.0, 0.0, 0.0], [0.0, 0.25, 0.5, 0.25])  # no full pass, but Φ moves: still teaches
SPREAD = ([0.0, 1.0, 0.5, 0.0], [0.0, 1.0, 0.5, 0.0])


def test_flat_needs_equal_rewards_and_no_phi_spread() -> None:
    assert flat(*FLAT0) and not flat(*SPREAD) and not flat(*PHI_SPREAD)
    assert flat([1.0, 1.0], [None, None]) and not flat([0.0], [0.0])  # one live member is not a group verdict


def _drive(q: TaskQuarantine, script: dict[str, list[tuple]], steps: int, per_step: int) -> list[str]:
    """Run ``steps`` optimizer steps of ``per_step`` groups each through the fake sampler; returns the sampled tasks."""
    cursor, seen = 0, []
    pos = {t: 0 for t in script}
    for step in range(steps):
        for _ in range(per_step):
            i = q.pick(cursor, step)
            cursor = i + 1
            t = q.tasks[i]
            seen.append(t)
            rs, ps = script[t][min(pos[t], len(script[t]) - 1)]
            pos[t] += 1
            q.observe(t, rs, ps, step)
    return seen


def test_a_flat_task_is_quarantined_for_r_steps_then_retried_once() -> None:
    q = TaskQuarantine(["dead", "live", "phi"], QuarantineConfig(q=2, r=3))
    seen = _drive(q, {"dead": [FLAT0], "live": [SPREAD], "phi": [PHI_SPREAD]}, steps=9, per_step=3)
    kinds = [(e["event"], e["task"], e["step"]) for e in q.events]
    assert ("quarantined", "dead", 1) in kinds  # flat at step 0 and step 1
    assert ("probation", "dead", 4) in kinds and ("requarantined", "dead", 4) in kinds  # one retry at step 4, still flat
    assert not any(e["task"] in ("live", "phi") for e in q.events)  # Φ spread without a pass is never quarantined
    per_step = [seen[i * 3:(i + 1) * 3] for i in range(9)]
    assert all("dead" not in s for s in per_step[2:4]) and per_step[4].count("dead") == 1  # skipped for 3 steps, retried once
    assert all("dead" not in s for s in per_step[5:7]) and q.active(5) == ["dead"]
    assert q.events[0]["reason"].startswith("2 consecutive flat groups") and q.events[0]["until"] == 4


def test_two_flat_groups_of_one_step_are_one_strike_and_spread_resets() -> None:
    q = TaskQuarantine(["t"], QuarantineConfig(q=2, r=3))
    assert q.observe("t", *FLAT0, step=0) is None and q.observe("t", *FLAT0, step=0) is None
    assert q.observe("t", *SPREAD, step=1) is None  # spread resets the strikes
    assert q.observe("t", *FLAT0, step=2) is None
    assert q.observe("t", *FLAT0, step=3)["event"] == "quarantined"


def test_a_released_task_comes_back_and_a_lost_retry_reopens_the_probation() -> None:
    q = TaskQuarantine(["a", "b"], QuarantineConfig(q=1, r=2))
    assert q.observe("a", *FLAT0, step=0)["event"] == "quarantined"
    assert q.blocked("a", 1) and q.pick(0, 1) == 1
    assert q.pick(0, 2) == 0 and q.blocked("a", 2)  # probation: its one retry is in flight
    q.lost("a")  # the retry was masked (infrastructure): not an observation
    assert not q.blocked("a", 2) and q.pick(0, 2) == 0
    assert q.observe("a", *SPREAD, step=2)["event"] == "released"
    assert not q.blocked("a", 3) and q.state["a"].strikes == 0


def test_every_task_quarantined_never_stalls_the_run() -> None:
    q = TaskQuarantine(["a", "b"], QuarantineConfig(q=1, r=5))
    q.observe("a", *FLAT0, step=0)
    q.observe("b", *FLAT0, step=1)
    assert q.pick(0, 2) == 0 and q.events[-1]["event"] == "all_quarantined"  # the one released soonest


def test_disabled_and_resume_state() -> None:
    off = TaskQuarantine(["a"], QuarantineConfig(q=0))
    assert off.observe("a", *FLAT0, step=0) is None and off.observe("a", *FLAT0, step=1) is None and not off.blocked("a", 2)
    q = TaskQuarantine(["a", "b"], QuarantineConfig(q=1, r=3))
    q.observe("a", *FLAT0, step=0)
    doc = json.loads(json.dumps(q.to_json()))
    q2 = TaskQuarantine(["a", "b"], QuarantineConfig(q=1, r=3)).load(doc)
    assert q2.blocked("a", 2) and not q2.blocked("a", 3) and q2.state["a"].times == 1
    with pytest.raises(ValueError):
        QuarantineConfig(q=-1)


def test_held_out_tasks_are_refused() -> None:
    with pytest.raises(ValueError, match="held-out"):
        TaskQuarantine(["a", "h"], splits={"a": "train", "h": "heldout"})


# ---- the Tinker trainer's scheduling, on a fake sampler -----------------------------------------------------------------------------
def _env(tid: str, split: str = "train") -> SimpleNamespace:
    return SimpleNamespace(task=SimpleNamespace(task_id=tid, split=split, verifier_version="v1"), run_spec=None, policy_kwargs=dict)


class ScriptedTrainer(TinkerTrainer):
    """Groups from a scripted sampler (task -> its groups' rewards); the update only advances the step."""

    script: ClassVar[dict[str, tuple[list[float], list[float]]]] = {}

    async def _group(self, env, gi):
        tid = str(env.task.task_id)
        rs, ps = self.script[tid]
        recs = [RolloutRecord.model_validate({"rollout_id": f"{tid}-{gi}-{j}", "sample": f"s{j}", "task": tid, "harness": "h", "policy": "p", "reward": r,
                                              "progress": {"phi_final": p}}) for j, (r, p) in enumerate(zip(rs, ps, strict=True))]
        return Group(gi, env, tid, "v1", f"g{gi}", recs, self.state["version"], 0.0)

    async def _step(self, batch):
        step = self.state["step"] + 1
        self.state.update(step=step, version=step)
        self.step_rows.append({"step": step, "tasks": [g.task_id for g in batch], "groups": [g.run_id for g in batch]})
        self.tracking.log({"reward/mean": statistics.fmean(x for g in batch for x in g.rewards()), "quarantine/active": float(len(self.quarantine.active(step)))}, step)


class Recorder:
    def __init__(self):
        self.records, self.logs = [], []

    def log(self, metrics, step=None):
        self.logs.append((step, dict(metrics)))

    def record(self, kind, data, step):
        self.records.append((kind, dict(data), step))

    def summary(self, data):
        pass

    def finish(self, status="ok"):
        pass


def test_the_tinker_trainer_skips_a_quarantined_task_and_logs_it(tmp_path: Path) -> None:
    sink = Recorder()
    envs = [_env("dead"), _env("live"), _env("phi")]
    t = ScriptedTrainer(envs, config=TrainConfig(model=MODEL, group_size=4, groups_per_step=2, steps=8, require_band=False, drop_zero_spread=False,
                                                 quarantine_q=2, quarantine_r=3),
                        backend=FakeTinker(), run_dir=tmp_path / "run", codec=ByteChatCodec(), objective_id="obj", sinks=[sink])
    t.script = {"dead": FLAT0, "live": SPREAD, "phi": PHI_SPREAD}
    out = asyncio.run(t.run())
    assert out["stop"] == "done" and out["quarantine"]["quarantined"] == ["dead"]
    rows = [json.loads(x) for x in (tmp_path / "run" / "quarantine.jsonl").read_text().splitlines()]
    assert rows[0]["event"] == "quarantined" and rows[0]["task"] == "dead" and "consecutive flat groups" in rows[0]["reason"]
    assert any(k == "quarantine" and d["task"] == "dead" for k, d, _ in sink.records)
    trained = [s["tasks"] for s in out["steps_log"]]
    q_step = rows[0]["step"]
    window = trained[q_step + 1:rows[0]["until"]]  # batches sampled at the quarantined steps (a batch sampled at step s trains as step s + 1)
    assert len(window) == rows[0]["until"] - q_step - 1 and all("dead" not in ts for ts in window)
    assert any("dead" in ts for ts in trained[rows[0]["until"]:])  # retried after r steps
    assert any(m.get("quarantine/active") == 1.0 for _, m in sink.logs)
    state = json.loads((tmp_path / "run" / "state.json").read_text())
    assert state["quarantine"]["state"]["dead"]["times"] >= 1 and "cursor" in state


def test_the_trainer_refuses_held_out_tasks_before_any_quarantine(tmp_path: Path) -> None:
    with pytest.raises(RunRefused):
        ScriptedTrainer([_env("a"), _env("h", "heldout")], config=TrainConfig(model=MODEL, require_band=False), backend=FakeTinker(),
                        run_dir=tmp_path / "r", codec=ByteChatCodec(), objective_id="o")


def test_quarantine_settings_may_change_on_resume(tmp_path: Path) -> None:
    kw = {"backend": FakeTinker(), "run_dir": tmp_path / "r", "codec": ByteChatCodec(), "objective_id": "o"}
    a = ScriptedTrainer([_env("a")], config=TrainConfig(model=MODEL, require_band=False, quarantine_q=2), **kw)
    a._open()
    b = ScriptedTrainer([_env("a")], config=TrainConfig(model=MODEL, require_band=False, quarantine_q=0, quarantine_r=5), **kw)
    b._open()  # not refused: the quarantine changes scheduling, never the objective
