"""m13 E (audit 03 G5, pick 7): NeMo-Gym resources servers imported as Crucible Worlds. Real servers (workplace_assistant,
mrcr, ruler) run in the NeMo-Gym Python; recorded NeMo-Gym rollouts replayed through Crucible Worlds reproduce NeMo's own
rewards; oracle -> 1, no-op -> 0; snapshot/restore re-seeds and replays; mask_sample masks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy, run_episode
from crucible.core.schema.actions import Done, ToolCall
from crucible.interop.nemo_gym.import_server import NEMO_PY, NEMO_ROOT, NemoGymServer, NemoGymTask, load_rows

pytestmark = pytest.mark.skipif(not NEMO_PY.exists() or not NEMO_ROOT.exists(), reason="needs crucible/.envs/nemo and the NeMo-Gym checkout")

SERVERS = {"workplace_assistant": "resources_servers.workplace_assistant.app:WorkbenchResourcesServer",
           "mrcr": "resources_servers.mrcr.app:MRCRResourcesServer",
           "ruler": "resources_servers.ruler.app:RulerResourcesServer"}


@pytest.fixture(scope="module")
def urls():
    srvs = {k: NemoGymServer(v) for k, v in SERVERS.items()}
    try:
        yield {k: s.start() for k, s in srvs.items()}
    finally:
        for s in srvs.values():
            s.stop()


def _data(server, name):
    return load_rows(NEMO_ROOT / "resources_servers" / server / "data" / name)


async def _episode(task, actions):
    d = Path(tempfile.mkdtemp())
    w = await task.open(d / "w")
    try:
        await run_episode(w, {"agent": ScriptedPolicy("agent", list(actions))}, limits=EpisodeLimits(max_steps=60), reset=False)
        return await task.grade(w, d / "g"), w
    finally:
        await w.close()


def _replay_actions(response, app="gym"):
    acts, final = [], None
    for o in response["output"]:
        if o.get("type") == "function_call":
            args = o.get("arguments") or "{}"
            acts.append(ToolCall(tool_call_id=o.get("call_id") or f"c{len(acts)}", function_name=f"{app}.{o['name']}",
                                 arguments=json.loads(args) if isinstance(args, str) else args))
        elif o.get("type") == "message":
            final = "".join(c.get("text", "") for c in o.get("content", []) if c.get("type") == "output_text")
    if final:
        acts.append(Done(reason=final))
    return acts


async def test_workplace_assistant_oracle_and_noop(urls):
    for row in _data("workplace_assistant", "example.jsonl"):
        t = NemoGymTask(f"nemo/wa{row['id']}", urls["workplace_assistant"], row)
        assert (await _episode(t, t.oracle_actions()))[0].reward == 1.0
        assert (await _episode(t, []))[0].reward == 0.0
    # (workplace_assistant's example_rollouts are other tasks, without ground truth: the identity check runs on mrcr / ruler)


@pytest.mark.parametrize("server", ["mrcr", "ruler"])
async def test_single_turn_answers_reproduce_nemo_rewards(urls, server):
    recs = _data(server, "example_rollouts.jsonl")
    assert recs
    for rec in recs:
        row = {k: v for k, v in rec.items() if k not in ("response", "reward")}
        rep, _w = await _episode(NemoGymTask(f"nemo/{server}", urls[server], row), _replay_actions(rec["response"]))
        assert rep.reward == pytest.approx(float(rec["reward"])), (server, rec.get("_ng_task_index"))


async def test_snapshot_restore_replays_and_mask_sample_masks(urls, monkeypatch):
    row = _data("workplace_assistant", "example.jsonl")[0]
    t = NemoGymTask("nemo/wa0", urls["workplace_assistant"], row)
    d = Path(tempfile.mkdtemp())
    w = await t.open(d / "w")
    try:
        gt = t.oracle_actions()
        ref = await w.snapshot()
        for a in gt:
            await w.step("agent", a)
        await w.restore(ref)  # back to the start: a fresh session, nothing re-issued
        for a in gt:
            await w.step("agent", a)
        assert (await t.grade(w, d / "g")).reward == 1.0
        from crucible.interop.nemo_gym import import_server

        async def masked(self, response):
            return {"reward": 0.0, "mask_sample": True, "failure_kind": "judge_unavailable"}

        monkeypatch.setattr(import_server.NemoGymApp, "verify", masked)
        rep = await t.grade(w, d / "g2")
        assert rep.infra.get("status") == "masked"  # consumers read the status: a masked grade is never trained as 0
    finally:
        await w.close()
