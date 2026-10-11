"""m13 C-8: reward shaping (A2-19 verbosity budget, A2-21 speed term, A2-24 incomplete-trajectory penalty, A2-25
tool-format penalty on the offending call's tokens, A2-26 grader compute accounting)."""

from __future__ import annotations

import itertools
import json

import pytest
from crucible_pack_toy.runtime import HelpdeskTask, scripted

from crucible.actors.policy import EpisodeLimits
from crucible.core.schema.actions import ToolCall
from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.credit.shaping import format_errors_by_call, speed_term, verbosity_group_fn, with_speed
from crucible.runtime.harness.native import NativeHarness
from crucible.runtime.infra import LimitPolicy
from crucible.runtime.run import Runner, RunSpec
from crucible.train.engine import TinyPolicy
from crucible.train.trainer import TorchTrainer, TrainerConfig


async def test_incomplete_trajectory_penalty_through_the_runner(tmp_path):
    loop = scripted("loop", [ToolCall(tool_call_id=f"l{i}", function_name="tickets.list_tickets", arguments={"status": "open"}) for i in range(10)])
    h = NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=3), limit_policy=LimitPolicy(on_steps="penalty:-0.5"))
    await Runner(tmp_path).run(RunSpec(run_id="pen", tasks=[HelpdeskTask()], policies=[loop], harnesses=[h], k=1))
    res = [json.loads(x) for x in (tmp_path / "pen" / "results.jsonl").read_text().splitlines()]
    assert res[0]["reward"] == -0.5 and res[0]["meta"]["truncated"] and not res[0]["mask_sample"]
    assert LimitPolicy().penalty("max_steps") is None and LimitPolicy(on_timeout="mask").penalty("timeout") is None


async def test_format_errors_are_attributed_to_the_issuing_call(tmp_path):
    bad = [ToolCall(tool_call_id="b1", function_name="tickets.list_tickets", arguments={"status": 7}),  # schema: string
           ToolCall(tool_call_id="b2", function_name="tickets.nope", arguments={}),
           ToolCall(tool_call_id="g1", function_name="tickets.list_tickets", arguments={"status": "open"})]
    h = NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=5))
    await Runner(tmp_path).run(RunSpec(run_id="fmt", tasks=[HelpdeskTask()], policies=[scripted("bad", bad)], harnesses=[h], k=1))
    w = await HelpdeskTask().open(tmp_path / "w2")  # the attribution function on a hand-made call log
    try:
        for a in bad:
            await w.step("agent", a)
        calls = {"agent": [{"call_id": "agent:c1", "actions": ["tool:b1", "tool:b2"]}, {"call_id": "agent:c2", "actions": ["tool:g1"]}]}
        assert format_errors_by_call(w.to_trace(), calls) == {"agent:c1": 2}
    finally:
        await w.close()


def _rec(rid, reward, n_tokens=3, format_errors=0):
    return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward,
                         turns=[Turn(call_id="agent:c1", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[3] * n_tokens,
                                     logprobs=[-0.1] * n_tokens, policy_version=0, format_errors=format_errors),
                                Turn(call_id="agent:c2", actor="agent", prompt_token_ids=[4], prompt_is_delta=True, completion_token_ids=[5, 6],
                                     logprobs=[-0.1, -0.1], policy_version=0)])


@pytest.mark.parametrize("unit", ["call", "segment"])
def test_tool_format_penalty_hits_only_the_offending_call(unit):
    tr = TorchTrainer(TinyPolicy(16, 8, seed=0), TrainerConfig(item_unit=unit, tool_format_penalty=0.2))
    recs = [_rec("a", 1.0, format_errors=2), _rec("b", 0.0)]
    items = tr._items(recs, ["g", "g"])
    if unit == "call":
        assert items[0].adv == pytest.approx([0.5 - 0.4] * 3) and items[1].adv == pytest.approx([0.5, 0.5])  # c1 penalized, c2 not
        assert items[2].adv == pytest.approx([-0.5] * 3)
    else:
        a = items[0]
        trained = [x for x, m in zip(a.adv, a.mask, strict=True) if m]
        assert trained == pytest.approx([0.1, 0.1, 0.1, 0.5, 0.5])
    plain = TorchTrainer(TinyPolicy(16, 8, seed=0), TrainerConfig(item_unit="call"))._items(recs, ["g", "g"])
    assert plain[0].adv == 0.5  # penalty 0: untouched


async def test_speed_term_is_monotone_and_gated_on_success():
    xs = [speed_term(t, 10.0) for t in (0, 5, 10, 40)]
    assert xs[0] == 1.0 and xs[2] == 0.5 and all(a > b for a, b in itertools.pairwise(xs))
    fn = with_speed(None, beta=0.1, t_base=10.0)
    fast, slow, fail = _rec("f", 1.0), _rec("s", 1.0), _rec("x", 0.0)
    fast.timings["active_s"], slow.timings["active_s"], fail.timings["active_s"] = 2.0, 40.0, 1.0
    rf, rs, rx = await fn(fast), await fn(slow), await fn(fail)
    assert rf > rs > 1.0 and rx == 0.0 and "speed" not in fail.reward_components  # a fast failure earns nothing


def test_terminal_reward_reaches_every_step():
    items = TorchTrainer(TinyPolicy(16, 8, seed=0), TrainerConfig(item_unit="call"))._items([_rec("a", 1.0), _rec("b", 0.0)], ["g", "g"])
    assert items[0].adv == items[1].adv == 0.5  # reward-to-go of a terminal outcome: the same advantage on every call


def test_verbosity_budget_loses_automatically():
    stage = verbosity_group_fn(1.5)
    members = [_rec("a", 1.0, n_tokens=10), _rec("b", 1.0, n_tokens=10), _rec("c", 1.0, n_tokens=30), _rec("d", 0.0, n_tokens=5)]
    out = stage("g", members)  # l0 = median passing length = 12 (10 + 2)
    assert [m.reward for m in out] == [1.0, 1.0, 0.0, 0.0] and "verbosity/over_budget" in out[2].reward_components
    out = verbosity_group_fn(2.0, l0_of=lambda task: 5.0)("g", [_rec("a", 1.0, n_tokens=8), _rec("b", 1.0, n_tokens=9)])  # 10 = 2·5 stays, 11 loses
    assert [m.reward for m in out] == [1.0, 0.0]


async def test_grader_compute_is_accounted_per_step():
    import asyncio

    from crucible.train.engine import EnginePool, TorchEngine
    from crucible.train.envs import CopyTask
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.weights import InprocPublisher

    async def grader(rec):
        await asyncio.sleep(0.01)
        rec.reward_components["grader/usd"] = 0.002
        return rec.reward

    model = TinyPolicy(32, 48, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, numerics="bf16", decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-2, correction=Correction("dis")))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=3, max_turn_tokens=3), SchedulerConfig(n=2, B=2, max_staleness=2), eng, tr,
                 InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8, latency_s=0.001), {"copy": [f"p{i}" for i in range(500)]}, reward_fn=grader)
    rep = await rt.run()
    for s in rep.steps:
        m = s.metrics
        assert m["grader/s"] >= 0.01 * 4 and m["grader/usd"] == pytest.approx(0.002 * 4) and 0 < m["grader/share"] < 1
