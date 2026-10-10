"""Long horizons (DESIGN §10.4): a backlog world whose episodes run thousands of steps; dense progress that telescopes
exactly to the final reward; policy-written compaction with recorded segments; mid-episode checkpoint and resume after
a crash; linear token storage. API-free: ``ReplayLLM`` drives the real ``LLMToolPolicy``."""

from __future__ import annotations

import json

import pytest
from crucible_pack_toy.longhorizon.task import BacklogTask, oracle_policy
from crucible_pack_toy.longhorizon.world import diligent

from crucible.actors.policy import ContextConfig, EpisodeLimits, LLMToolPolicy, ScriptedPolicy
from crucible.runtime.harness.native import NativeHarness, PolicyRef
from crucible.runtime.inject import ReplayLLM
from crucible.runtime.run import Runner, RunSpec
from crucible.train.rollout import RolloutRecord
from crucible.verify.progress import read_rows

LIMITS = EpisodeLimits(max_steps=100_000, max_tool_errors=None)


def replay_ref(task: BacklogTask, *, budget: int = 6_000, crash: tuple[int, int] | None = None, name: str = "replay") -> PolicyRef:
    acts = diligent(task.b)

    def make(attempt):
        crash_at = (crash[1], ConnectionError("worker lost")) if crash and attempt.n == crash[0] else None
        return {"agent": LLMToolPolicy("agent", ReplayLLM(acts, crash_at=crash_at), context=ContextConfig(budget_tokens=budget, keep_recent_turns=2))}

    return PolicyRef(id=f"replay:{name}", make=make)


def one(tmp_path, task, pol, harness, rid="lh"):
    return Runner(tmp_path).run(RunSpec(run_id=rid, tasks=[task], policies=[pol], harnesses=[harness], k=1))


async def test_oracle_and_dense_progress_telescope(tmp_path):
    t = BacklogTask(1, 40)
    s = await one(tmp_path, t, oracle_policy(t), NativeHarness("native:scripted", limits=LIMITS))
    assert s.mean_reward == 1.0
    adir = next((tmp_path / "lh" / "samples").glob("*/a1"))
    rec = json.loads((adir / "record.json").read_text())
    p = rec["meta"]["progress"]
    assert p["phi0"] == 0.0 and p["phi_final"] == 1.0 and p["drift"] == [] and abs(p["sum_delta"] - 1.0) < 1e-6  # dense rewards sum to R
    rows = read_rows(adir / "progress.jsonl")
    assert len(rows) == rec["steps"] and all(r["delta_phi"] >= 0 for r in rows)  # the oracle never regresses
    assert sum(1 for r in rows if r["delta_phi"] > 0) == 40  # one positive reward per ticket, at the step that completed it


async def test_noop_and_premature_action_score_zero_per_ticket(tmp_path):
    t = BacklogTask(2, 12)
    s = await one(tmp_path, t, PolicyRef(id="scripted:noop", make=lambda a: {"agent": ScriptedPolicy("agent", [])}), NativeHarness("native:scripted", limits=LIMITS))
    assert s.mean_reward == 0.0


async def test_compaction_segments_and_linear_accounting(tmp_path):
    t = BacklogTask(3, 60)
    s = await one(tmp_path, t, replay_ref(t, budget=6_000), NativeHarness(limits=LIMITS))
    assert s.mean_reward == 1.0  # the policy finished a backlog far longer than its context
    adir = next((tmp_path / "lh" / "samples").glob("*/a1"))
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    comps = [x for x in ro.turns if x.kind == "compaction"]
    assert len(comps) >= 10 and len(ro.segments) == len(comps)
    assert max(sg.context_before for sg in ro.segments) < 6_000 and all(sg.context_after < sg.context_before for sg in ro.segments)
    assert ro.trajectory_tokens > 8 * 6_000  # the trajectory is many contexts long
    assert {x.segment for x in ro.turns} == set(range(len(ro.segments) + 1))
    credit = sum(x.advantage or 0 for x in ro.turns)
    assert abs(credit + ro.progress["unattributed_delta"] - 1.0) < 1e-6  # per-call credit from the dense stream sums to R


async def test_crash_mid_episode_resumes_from_checkpoint(tmp_path):
    t = BacklogTask(4, 40)
    s = await one(tmp_path, t, replay_ref(t, budget=5_000, crash=(1, 150)), NativeHarness(limits=LIMITS, checkpoint_every=50))
    assert s.mean_reward == 1.0 and s.attempts == 2
    a1, a2 = sorted((tmp_path / "lh" / "samples").glob("*/a*"))[:2]
    r1, r2 = (json.loads((a / "record.json").read_text()) for a in (a1, a2))
    assert r1["mask_sample"] and r1["failure"]["kind"] == "transport_unreachable" and r1["failure"]["stage"] == "agent"
    assert r2["meta"]["artifacts"]["resumed_from"] == "a1@step100"  # last checkpoint before the crash at call 150
    assert r2["steps"] > 100 and r2["meta"]["progress"]["drift"] == []
    ev = [json.loads(x) for x in (a2 / "trace.jsonl").read_text().splitlines()]
    assert ev[0]["seq"] == 0 and [e["seq"] for e in ev] == list(range(len(ev)))  # one continuous trace across the resume
    rows = read_rows(a2 / "progress.jsonl")
    assert abs(sum(r["delta_phi"] for r in rows) - 1.0) < 1e-6  # the dense stream still telescopes across the resume


def test_xl_preset_exceeds_a_million_tokens():
    from crucible_pack_toy.longhorizon.world import PRESETS, draw

    b = draw(0, PRESETS["xl"])
    acts = diligent(b)
    assert len(acts) > 7000 and len(b.tickets) == 1200


@pytest.mark.parametrize("seed", [0, 5])
def test_ground_truth_is_consistent(seed):
    from crucible_pack_toy.longhorizon.world import draw

    b = draw(seed, 200)
    for t in b.tickets:
        if t.kind == "repeat":
            assert t.expected["refund"] is None
        if t.kind == "awaiting":
            assert t.follow_up and t.expected["wait_for"] == t.follow_up["id"]
