"""M12: durable records of control runs (``train/control_record.py``, ``crucible.control_rollout/v1``).

A short PPO run on CartPole (discrete) and Pendulum (box, clipped actions) writes one record per episode with the
behaviour log-prob and value of every sampled action; each record round-trips through JSON and the JSONL sink, and
replaying it through a fresh ``ControlApp`` reproduces every observation digest. A language model acting through the
App's ``step`` / ``wait`` tools gets the same record from the journal, each step linked to the model call that issued it.
"""

from __future__ import annotations

import json

import pytest

from crucible.runtime.control_task import ControlTask
from crucible.train.control import PPOConfig, ppo
from crucible.train.control_record import CONTROL_SCHEMA, RECORD_KIND, ControlRecord, ControlRecorder, record_from_world, replay
from crucible.train.sinks import JsonlSink, Tracking
from crucible.world.gym import VectorWorldEnv, WorldEnv

pytest.importorskip("gymnasium")


def _task(env_id: str, rr, **kw) -> ControlTask:
    return ControlTask(f"t/{env_id}@1", env=f"gymnasium:{env_id}", return_range=rr, journal="lazy", clock=False, **kw)


async def test_ppo_writes_replayable_records_to_sinks(tmp_path):
    task = _task("CartPole-v1", (0, 500))
    venv = VectorWorldEnv(lambda i: WorldEnv(task, workdir=tmp_path / f"e{i}"), 2, seed=3)
    sink = JsonlSink(tmp_path / "sink")
    rec = ControlRecorder(Tracking([sink]), values=True, run_id="cp")
    try:
        res = await ppo(venv, PPOConfig(total_steps=128, n_envs=2, rollout=64, epochs=1, minibatches=2, seed=0), recorder=rec)
    finally:
        await venv.aclose()
        sink.finish("ok")
    finished = [r for r in rec.records if r.stop != "run_end"]
    assert len(finished) == len(res.episodes) >= 3 and len(rec.records) == len(finished) + 2  # two episodes cut by the run's end
    for r, ret in zip(finished, res.episodes, strict=True):  # ordered as the trainer saw the episodes end
        assert r.schema_ == CONTROL_SCHEMA and r.env == "gymnasium:CartPole-v1" and r.task == task.id
        assert abs(r.episode_return - ret) < 1e-9 and (r.terminated or r.truncated)
        assert all(s.logprob is not None and s.logprob <= 0 and s.value is not None and s.policy_version == 0 for s in r.steps)
        assert all(isinstance(s.action, int) and s.raw_action is None and len(s.obs) == 4 for s in r.steps) and len(r.reset_obs) == 4
        assert r.valid_for_training() == (True, "ok") and r.ticks == len(r.steps)
        assert ControlRecord.model_validate_json(r.model_dump_json(by_alias=True)) == r  # JSON round trip
    lines = [json.loads(x) for x in (tmp_path / "sink" / "records.jsonl").read_text().splitlines()]
    assert [x["kind"] for x in lines] == [RECORD_KIND] * len(rec.records)
    assert ControlRecord.model_validate(lines[0]["data"]) == rec.records[0]
    for r in [finished[0], rec.records[-1]]:  # a finished episode and one cut mid-way both replay exactly
        rep = await replay(r, task, tmp_path / f"replay-{r.rollout_id}")
        assert rep.ok, rep.mismatches
        assert rep.digests == r.digests() and rep.steps == len(r.steps)


async def test_box_actions_keep_the_sampled_action_and_replay_the_executed_one(tmp_path):
    task = _task("Pendulum-v1", (-1600, 0), max_episode_ticks=20)
    venv = VectorWorldEnv(lambda i: WorldEnv(task, workdir=tmp_path / f"e{i}"), 2, seed=0)
    rec = ControlRecorder()
    try:
        await ppo(venv, PPOConfig(total_steps=80, n_envs=2, rollout=40, epochs=1, minibatches=2, seed=0), recorder=rec)
    finally:
        await venv.aclose()
    r = rec.records[0]
    assert r.truncated and not r.terminated and len(r.steps) == 20 and r.stop == "truncated"
    assert all(-2.0 <= s.action[0] <= 2.0 for s in r.steps)
    assert all(s.raw_action is None or s.raw_action != s.action for s in r.steps)  # the sampled action is kept only when clipping changed it
    rep = await replay(r, task, tmp_path / "replay")
    assert rep.ok, rep.mismatches
    tampered = r.model_copy(deep=True)
    tampered.steps[3].action = [0.5 if r.steps[3].action[0] != 0.5 else -0.5]
    bad = await replay(tampered, task, tmp_path / "replay-bad")
    assert not bad.ok and bad.mismatches[0].startswith("step 3: observation digest")  # a record that does not reproduce is caught
    other = _task("Pendulum-v1", (-1600, 0), max_episode_ticks=30)
    assert "config" in (await replay(r, other, tmp_path / "replay-cfg")).mismatches[0]


async def test_llm_policy_through_control_app_gets_a_linked_record(tmp_path):
    from crucible.actors.policy import EpisodeLimits, LLMToolPolicy, run_episode
    from crucible.runtime.inject import ReplayLLM

    task = ControlTask("t/pendulum/hold@1", env="gymnasium:Pendulum-v1", return_range=(-2000.0, 0.0), idle_action=[0.0], frame_skip=5, max_episode_ticks=60)
    acts = [{"function_name": "sim.step", "arguments": {"action": [1.0]}}, {"function_name": "sim.wait", "arguments": {"ticks": 30}},
            {"function_name": "sim.step", "arguments": {"action": [-1.0]}}]
    world = await task.open(tmp_path / "w")
    try:
        pol = LLMToolPolicy("agent", ReplayLLM(acts))
        ep = await run_episode(world, {"agent": pol}, reset=True, limits=EpisodeLimits(max_steps=10))
        r = record_from_world(world, rollout_id="llm-1", task=task.id, model_calls=ep.model_calls, policy="llm:replay")
    finally:
        await world.close()
    assert [s.tool for s in r.steps] == ["step", "wait", "step"] and [s.ticks for s in r.steps] == [5, 30, 5] and r.ticks == 40
    assert [s.call_id for s in r.steps] == [c["call_id"] for c in ep.model_calls["agent"][:3]]  # each step joined to its model call
    assert all(s.logprob is None for s in r.steps) and not r.valid_for_training()[0]  # the model's tokens train through rollout/v1
    assert ControlRecord.model_validate(json.loads(r.model_dump_json(by_alias=True))) == r
    rep = await replay(r, task, tmp_path / "replay")
    assert rep.ok and rep.digests == r.digests(), rep.mismatches
