"""m14: RLMPolicy, the Recursive-Language-Model policy. On the toy helpdesk with scripted models: code cells solve the task
through ``world.call`` (one World step per call, reward 1.0); ``rlm.batch`` sub-calls are recorded with their parent;
printed output is capped while ``OBS`` keeps every full result; a premature stop gets a goal continuation; stalls are
detected and stop the episode; a wedged cell is killed and the kernel restored by replay; a crash in the middle of a cell
resumes from a checkpoint to the same final World state and reward."""

from __future__ import annotations

import json

import pytest
from crucible_pack_toy.runtime import HelpdeskTask
from crucible_pack_toy.verify import GOLD

from crucible.actors.llm import LLMResponse
from crucible.actors.policy import EpisodeLimits, run_episode
from crucible.actors.rlm.policy import RLMPolicy
from crucible.runtime.inject import FakeLLM


def code(src: str, i: int = 0) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": src}}], usage={"completion_tokens": 10})


GOLD_CODE = "\n".join(f"await world.call({a.function_name!r}, **{a.arguments!r})" for a in GOLD)


async def episode(tmp_path, script, *, sub=None, limits=None, **kw):
    task = HelpdeskTask()
    w = await task.open(tmp_path / "work")
    p = RLMPolicy("agent", FakeLLM(script), sub_llm=sub, **kw)
    try:
        ep = await run_episode(w, {"agent": p}, limits=limits or EpisodeLimits(max_steps=60))
        rep = await task.grade(w, tmp_path / "grade")
        return ep, rep, p
    finally:
        await w.close()
        await p._close_kernel()


async def test_rlm_solves_the_task_through_code(tmp_path):
    sub = FakeLLM(["ticket 1 is a damaged blender", "ticket 2 is a duplicate charge"])
    script = [code("t = await world.tickets.list_tickets(status='open')\nlen(OBS), t.ok", 1),
              code("notes = await rlm.batch(['summarize ticket 1', 'summarize ticket 2'])\n" + GOLD_CODE + "\nrlm.finish('all tickets handled')", 2)]
    ep, rep, p = await episode(tmp_path, script, sub=sub)
    assert rep.reward == 1.0, rep.failing_gates()
    assert ep.steps == 1 + len(GOLD) + 1 and ep.stop_condition == "agent_done"  # world calls + the final Done
    calls = ep.model_calls["agent"]
    turns, subs = [c for c in calls if c["kind"] == "turn"], [c for c in calls if c["kind"] == "sub"]
    assert len(turns) == 2 and len(subs) == 2 and all(s["parent"] == turns[1]["call_id"] for s in subs)
    assert turns[1]["actions"][-1] == "done" and len([a for a in turns[1]["actions"] if a.startswith("tool:")]) == len(GOLD)
    tool_msgs = [m for m in p.messages if m["role"] == "tool"]
    assert "(1, True)" in tool_msgs[0]["content"]


async def test_output_cap_keeps_full_data_in_variables(tmp_path):
    script = [code("big = 'x' * 50000\nprint(big)", 1), code("print(len(big), len(OBS))\nrlm.finish('ok')", 2)]
    _ep, _, p = await episode(tmp_path, script, max_output_chars=2000)
    first = next(m for m in p.messages if m["role"] == "tool")["content"]
    assert len(first) < 2200 and "characters truncated" in first
    assert "50000 0" in [m for m in p.messages if m["role"] == "tool"][1]["content"]


async def test_goal_continuation_then_finish(tmp_path):
    script = [LLMResponse(text="I think I am done."), code("rlm.finish('really done')", 1)]
    ep, _, p = await episode(tmp_path, script)
    assert any("[goal: continuation]" in str(m.get("content")) for m in p.messages) and ep.stop_condition == "agent_done"
    assert p.counters["continuations"] == 1


async def test_stall_detected_and_stops(tmp_path):
    bad = "await world.call('tickets.get_ticket', ticket_id=999)"
    script = [code(bad, i) for i in range(12)]
    ep, _, p = await episode(tmp_path, script, max_stalls=2, stall_repeat=3)
    assert ep.stop_condition == "too_many_errors" and p.counters["stalls"] == 2
    assert any("[stall-detected:" in str(m.get("content")) for m in p.messages)


async def test_wedged_cell_killed_and_kernel_restored(tmp_path):
    script = [code("keep = 41", 1), code("import time\nkeep = 0\nwhile True: time.sleep(0.05)", 2), code("print(keep + 1)\nrlm.finish('ok')", 3)]
    _ep, _, p = await episode(tmp_path, script, cell_timeout_s=1)
    outs = [m["content"] for m in p.messages if m["role"] == "tool"]
    assert "kernel state was restored" in outs[1] and outs[2].startswith("42")  # the cell's partial effect (keep = 0) was rolled back


async def test_crash_mid_cell_resumes_to_the_same_end_state(tmp_path):
    from crucible.runtime import checkpoint

    script = [code(GOLD_CODE + "\nrlm.finish('done')", 1)]
    # reference: an uninterrupted run
    ref_ep, ref_rep, _ = await episode(tmp_path / "ref", list(script))
    # interrupted: checkpoint after step 3 (inside the cell), then crash
    task = HelpdeskTask()
    w = await task.open(tmp_path / "run" / "work")
    w.stream_trace(tmp_path / "run" / "trace.jsonl")
    p = RLMPolicy("agent", FakeLLM(list(script)))
    ck = checkpoint.Checkpointer(tmp_path / "run", w, {"agent": p}, [], every=3)

    class Crash(Exception):
        pass

    async def after(step):
        await ck(step)
        if step == 3:
            raise Crash()

    with pytest.raises(Crash):
        await run_episode(w, {"agent": p}, limits=EpisodeLimits(max_steps=60), after_step=after)
    await w.close()
    await p._close_kernel()
    saved = checkpoint.load(tmp_path / "run")
    assert saved["step"] == 3 and saved["policies"]["agent"]["in_cell"]
    # resume in a fresh process-equivalent: new World object on the same store, new policy, no further model calls needed
    w2 = await task.open(tmp_path / "run" / "work")
    p2 = RLMPolicy("agent", FakeLLM([]))
    try:
        start = await checkpoint.resume(w2, {"agent": p2}, [], saved, prev_dir=tmp_path / "run", new_dir=tmp_path / "run2")
        ep2 = await run_episode(w2, {"agent": p2}, limits=EpisodeLimits(max_steps=60), reset=False, start_steps=start)
        rep2 = await task.grade(w2, tmp_path / "grade2")
        assert rep2.reward == ref_rep.reward == 1.0
        assert ep2.final.fingerprint == ref_ep.final.fingerprint and ep2.steps == ref_ep.steps
        tool_events = [e for e in w2.log if e.kind == "action" and e.type == "tool_call"]
        assert len(tool_events) == len(GOLD)  # nothing was stepped twice
    finally:
        await w2.close()
        await p2._close_kernel()


async def test_state_dict_is_json(tmp_path):
    script = [code("x = 1", 1), code("rlm.finish('ok')", 2)]
    _, _, p = await episode(tmp_path, script)
    json.dumps(p.state_dict())


async def test_train_rollout_complete_with_token_records(tmp_path):
    """Through the runtime (NativeHarness, rollout_mode="train"): root turns and sub-LLM calls carry token records, so the
    rollout is trainable; sub-calls appear as their own turns."""
    from crucible.runtime.harness.native import NativeHarness, PolicyRef
    from crucible.runtime.run import Runner, RunSpec
    from crucible.train.rollout import RolloutRecord

    tok = {"prompt_token_ids": [1, 2, 3], "completion_token_ids": [7, 8], "logprobs": [-0.1, -0.2], "policy_version": 1}

    def with_tok(r):
        r.tokens = dict(tok)
        return r

    def make(attempt):
        script = [with_tok(code("s = await rlm.query('summarize the open tickets')\n" + GOLD_CODE + "\nrlm.finish('done')", 1))]
        sub = FakeLLM([with_tok(LLMResponse(text="three open tickets"))])
        return {"agent": RLMPolicy("agent", FakeLLM(script), sub_llm=sub)}

    await Runner(tmp_path).run(RunSpec(run_id="r", tasks=[HelpdeskTask()], policies=[PolicyRef(id="rlm:fake", make=make)], harnesses=[NativeHarness()],
                                       k=1, rollout_mode="train"))
    adir = min((tmp_path / "r" / "samples").glob("*/a*"))
    ro = RolloutRecord.model_validate_json((adir / "rollout.json").read_text())
    assert not ro.mask_sample and ro.reward == 1.0, (ro.failure_kind, ro.stop)
    assert [t.kind for t in ro.turns] == ["turn", "sub"] and all(t.completion_token_ids == [7, 8] for t in ro.turns)
