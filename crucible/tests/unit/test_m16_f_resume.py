"""M16 agent F: checkpoints, resume and records of the integrated RLM loop (design §3.5, §3.6; stress B3, B5, B8).

- B3: a worker lost mid-episode is masked and the next attempt resumes from the checkpoint; the resumed rollout is
  train-valid: a ``resume`` segment opens on the root, its first call stores its whole prompt, nothing is a prefix break.
- B5: a checkpoint holds one context window, the journal since the last snapshot and an offset into the streamed model
  calls; its size does not grow with the episode.
- The harness: the evidence file (context health, ledger stats, tree usage), truncating stops, the snapshot restore probe.
"""

from __future__ import annotations

import json

from crucible_pack_toy.longhorizon.task import BacklogTask
from crucible_pack_toy.runtime import HelpdeskTask

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.actors.rlm.agent import RLMConfig
from crucible.actors.rlm.kernel import LocalLauncher
from crucible.actors.rlm.policy import RLMPolicy
from crucible.runtime.scripted_rlm import backlog_model, crash_then_resume_ref
from crucible.train.chat import ByteChatCodec
from crucible.train.tito_scripted import TitoLLM

LOCAL = {"kernel": "local", "allow_host": True, "temperature": 0.0}


def rows(adir):
    from crucible.core.schema.rollout import RolloutRecord

    return RolloutRecord.model_validate_json((adir / "rollout.json").read_text())


async def test_b3_crash_and_resume_is_train_valid(tmp_path):
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    model = backlog_model("faithful")
    ref, fired = crash_then_resume_ref(lambda attempt: TitoLLM(model, ByteChatCodec()), crash_at_call=10)
    # the byte codec: one token per byte, so a 24k window compacts on the 15-ticket backlog
    h = RLMHarness(RLMConfig(window_tokens=24_000, max_tokens=4_000, max_turns=10_000, **LOCAL),
                   limits=EpisodeLimits(max_steps=100_000, max_tool_errors=None), checkpoint_every=20)
    task = BacklogTask(0, 15)
    try:
        s = await Runner(tmp_path).run(RunSpec(run_id="c", tasks=[task], policies=[ref], harnesses=[h], k=1, rollout_mode="train"))
    finally:
        task.cleanup()
    assert fired and fired[0].fired[0]["fault"] == "crash"
    adirs = sorted((tmp_path / "c" / "samples").glob("*/a*"))
    first = json.loads((adirs[0] / "record.json").read_text())
    assert first["mask_sample"] and first["failure"]["kind"] == "agent_run_error"
    last = json.loads((adirs[-1] / "record.json").read_text())
    assert (last.get("meta") or {}).get("artifacts", {}).get("resumed_from"), "the retry did not resume from the checkpoint"
    ro = rows(adirs[-1])
    # the runner's record is runtime.record.build's; build it again from the attempt's calls and segments (train mode)
    from crucible.core.schema.run import RunRecord
    from crucible.runtime.record import build

    calls = json.loads((adirs[-1] / "calls.json").read_text())
    rr = RunRecord(run_id="c", attempt="c-a2", task="t", world="w", policy="p", outcome="completed", reward=1.0, started_at="2026-10-13T00:00:00Z")
    again = build(rr, calls=calls, segments={"agent": json.loads((adirs[-1] / "rlm.agent.json").read_text())["segments"]}, mode="train")
    assert again.valid_for_training() == (True, "ok") and again.segments[0].boundary_reason == "resume"
    assert s.mean_reward == 1.0 and not ro.mask_sample and ro.valid_for_training() == (True, "ok"), (ro.stop, ro.failure_kind)
    resume = [g for g in ro.segments if g.boundary_reason == "resume"]
    assert len(resume) == 1 and resume[0].agent == ""
    turns = [t for t in ro.turns if not t.node()]
    first_after = next(t for t in turns if t.segment >= resume[0].index)  # (a compaction may open the next segment first)
    assert not first_after.prompt_is_delta and first_after.prompt_token_ids  # the whole prompt (B3: it was a prefix break)
    assert not [t.call_id for t in ro.turns if t.prefix_break]
    # the checkpoint streamed the settled model calls (an offset, not a copy) and the resumed attempt kept the ledger
    ck = json.loads((adirs[0] / "checkpoint.json").read_text())["policies"]["agent"]
    assert ck["calls"]["streamed"] > 0 and (adirs[0] / "calls.agent.jsonl").exists() and ck["ledger"]["lengths"][""] > 0
    doc = json.loads((adirs[-1] / "rlm.agent.json").read_text())
    assert doc["ledger"]["windows"] >= 2 and doc["context_health"]["restores_by_mode"]


async def test_b5_checkpoint_size_does_not_grow_with_the_episode(tmp_path):
    sizes: dict[int, int] = {}
    holder: dict = {}

    class Counting:
        model = "count"

        async def complete(self, messages, *, tools=None, **kw):
            if not tools:
                return LLMResponse(text="SUMMARY: counting; n is in the kernel.", usage={"prompt_tokens": 100, "completion_tokens": 10})
            p = holder["p"]
            n = p.counters["cells"]
            if n in (300, 600):
                sizes[n] = len(json.dumps(p.state_dict(), default=str))
            code = "rlm.finish('ok')" if n >= 600 else "n = globals().get('n', 0) + 1\nprint('row', n, 'x' * 400)"
            return LLMResponse(text="", tool_calls=[{"id": f"c{n}", "name": "python", "arguments": {"code": code}}], usage={"completion_tokens": 20})

    task = HelpdeskTask()
    w = await task.open(tmp_path / "work")
    p = holder["p"] = RLMPolicy("agent", Counting(), launcher=LocalLauncher(allow_host=True), ledger_dir=tmp_path / "ledger", stream_dir=tmp_path,
                                config=RLMConfig(window_tokens=16_000, snapshot_every=100, **LOCAL))
    try:
        ep = await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=100))
    finally:
        await w.close()
        await p.aclose()
    assert ep.stop_condition == "agent_done" and p.counters["cells"] == 601 and p.counters["compactions"] >= 10
    assert sizes[600] < 1.3 * sizes[300] and sizes[600] < 400_000, sizes  # O(window + journal suffix), not O(episode) (B5)
    assert (tmp_path / "calls.agent.jsonl").exists()


async def test_harness_evidence_truncation_and_restore_probe(tmp_path):
    from crucible.runtime.harness.native import PolicyRef
    from crucible.runtime.harness.rlm import RLMHarness
    from crucible.runtime.run import Runner, RunSpec

    model = backlog_model("faithful")
    h = RLMHarness(RLMConfig(window_tokens=32_000, max_tokens=4_000, max_turns=12, snapshot_every=5, **LOCAL),
                   limits=EpisodeLimits(max_steps=100_000, max_tool_errors=None))
    made = []

    class Keep(RLMHarness):
        def wrap(self, attempt, actor, x):
            made.append(super().wrap(attempt, actor, x))
            return made[-1]

    h.__class__ = Keep
    task = BacklogTask(0, 15)
    try:
        await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[task], policies=[PolicyRef(id="s", make=lambda a: {"agent": TitoLLM(model, ByteChatCodec())}, config={})],
                                           harnesses=[h], k=1, rollout_mode="train"))
    finally:
        task.cleanup()
    adir = next((tmp_path / "r" / "samples").glob("*/a1"))
    rec = json.loads((adir / "record.json").read_text())
    ro = rows(adir)
    assert ro.stop == "budget:turns" and ro.truncated and not ro.mask_sample and rec["meta"]["stop"] == "budget:turns"
    doc = json.loads((adir / "rlm.agent.json").read_text())
    for k in ("config", "counters", "tree_counters", "segments", "context_health", "ledger", "tree", "trajectory_tokens", "kernel", "harness_notes"):
        assert k in doc, k
    assert doc["truncated"] and doc["stop"] == "budget:turns" and doc["ledger"]["events"] > 0 and doc["tree"]["turns"] == 12
    assert doc["trajectory_tokens"]["source"] == "usage" and doc["context_health"]["threshold"] == h.config.context_spec().threshold()
    probe = await made[0].restore_probe()
    assert probe["restore"] == "snapshot" and probe["error"] is None and probe["mode"] == "snapshot" and probe["snapshot_bytes"] > 0
    calls = json.loads((adir / "calls.json").read_text())["agent"]
    turn = next(c for c in calls if c["kind"] == "turn")
    for k in ("agent", "depth", "edge", "segment", "window", "ledger_index", "trainable"):
        assert k in turn, k


async def test_an_m14_checkpoint_is_refused():
    p = RLMPolicy("a", None, launcher=LocalLauncher(allow_host=True), config=RLMConfig(**LOCAL))
    import pytest

    with pytest.raises(ValueError, match="version 2"):
        p.load_state_dict({"messages": [], "model_calls": [], "kernel": None})
