"""Typed token usage, finish reasons and stage timings (M0 audit 2026-10-14 R-19)."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace as NS

import pytest
from openai.types.chat import ChatCompletion

from crucible.actors.llm import LLMClient
from crucible.bench.metrics import _attempt_stats
from crucible.core.config import Config
from crucible.core.schema.rollout import RolloutRecord, Turn
from crucible.core.schema.run import RunRecord
from crucible.core.schema.usage import Timing, Usage, UsageTotals, finish_reason
from crucible.runtime.harness.base import HarnessResult, settle_usage
from crucible.runtime.keys import SampleKey
from crucible.runtime.run import Runner

# one call: a 1000-token prompt, 900 of it read from the provider's prompt cache, 50 output tokens (20 of them reasoning)
SHAPES = {
    "openai_chat": {"prompt_tokens": 1000, "completion_tokens": 50, "total_tokens": 1050, "prompt_tokens_details": {"cached_tokens": 900},
                    "completion_tokens_details": {"reasoning_tokens": 20}},
    "anthropic": {"input_tokens": 100, "cache_read_input_tokens": 900, "cache_creation_input_tokens": 0, "output_tokens": 50},
    "responses": {"input_tokens": 1000, "input_tokens_details": {"cached_tokens": 900}, "output_tokens": 50,
                  "output_tokens_details": {"reasoning_tokens": 20}, "total_tokens": 1050},
    "crucible_pre_r19": {"prompt_tokens": 1000, "completion_tokens": 50, "cached_tokens": 900, "reasoning_tokens": 20},
}


@pytest.mark.parametrize("shape", SHAPES)
def test_provider_shapes_agree(shape):
    u = Usage.model_validate(SHAPES[shape])
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens, u.output_tokens, u.source) == (100, 900, 0, 50, "provider")
    assert (u.prompt_tokens, u.completion_tokens) == (1000, 50)  # the totals readers of a call's usage dict use
    assert Usage.model_validate(u.model_dump()) == u
    # cache writes: Anthropic reports them apart from input_tokens, OpenRouter inside prompt_tokens
    anth = Usage.model_validate({"input_tokens": 100, "cache_read_input_tokens": 700, "cache_creation_input_tokens": 200, "output_tokens": 50})
    orr = Usage.model_validate({"prompt_tokens": 1000, "completion_tokens": 50, "prompt_tokens_details": {"cached_tokens": 700, "cache_write_tokens": 200}})
    assert anth == orr and (orr.input_tokens, orr.cache_write_tokens, orr.prompt_tokens) == (100, 200, 1000)


def _client(tmp_path, prices):
    cfg = Config.model_validate({"llm_providers": {"api": {"base_url": "https://api.example.com/v1", "api_key": "EMPTY"}},
                                 "models": {"policy-default": "api:m"}, "data_dir": str(tmp_path), "prices": {"m": prices}})
    llm = LLMClient("policy-default", config=cfg, cache_mode="record", cache_dir=tmp_path / "cache")

    class Completions:
        async def create(self, **kw):
            return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                                  "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                                                  "usage": SHAPES["openai_chat"]})

    llm._openai = lambda: NS(chat=NS(completions=Completions()))
    return llm


def test_cached_prompt_tokens_are_priced_at_the_cache_rate(tmp_path):
    msgs = [{"role": "user", "content": "q"}]
    llm = _client(tmp_path / "a", [2.5, 10.0, 0.625])
    r = asyncio.run(llm.complete(msgs))
    assert r.cost_usd == pytest.approx((100 * 2.5 + 900 * 0.625 + 50 * 10.0) / 1e6)  # was 0.003: 2.29x (audit probe v1_usage)
    assert (r.usage["cache_read_tokens"], r.usage["prompt_tokens"], r.usage["reasoning_tokens"]) == (900, 1000, 20)
    again = asyncio.run(llm.complete(msgs))  # a replay keeps the usage it was produced with, and costs nothing
    assert again.cached and again.cost_usd == 0.0 and again.usage == r.usage
    # no cache price configured: the input rate, as before (the budget never counts less than was spent)
    assert asyncio.run(_client(tmp_path / "b", [2.5, 10.0]).complete(msgs)).cost_usd == pytest.approx(0.003)
    with pytest.raises(ValueError, match="cache read and cache write"):
        Config.model_validate({"prices": {"m": [1.0]}})


def test_replays_and_record_tokens_are_not_live_usage(tmp_path):
    calls = {"agent": [{"call_id": "agent:c1", "usage": SHAPES["crucible_pre_r19"]}, {"call_id": "agent:c2", "usage": SHAPES["openai_chat"], "cached": True},
                       {"call_id": "agent:c3", "usage": {}}]}
    t = UsageTotals.of(calls)
    assert (t.calls, t.reported, t.blank, t.replayed, t.replayed_tokens, t.prompt_tokens, t.cache_read_tokens) == (2, 1, 1, 1, 1050, 1000, 900)
    assert t.coverage == 0.5
    # bench on a v1 record written before R-19: its calls.json summed, the replay apart; its tokens are never guessed at
    (tmp_path / "calls.json").write_text(json.dumps(calls))
    rec = RunRecord.model_validate({"schema_version": "crucible.run/v1", "run_id": "r", "world": "w", "policy": "p", "started_at": "2026-01-01T00:00:00Z",
                                    "outcome": "completed", "reward": 1.0, "tokens": {"cache_read_input_tokens": 900},
                                    "meta": {"timings": {"agent": 2.0, "verification": 1.0}}})
    s = _attempt_stats(tmp_path, rec)
    assert (s["tokens_in"], s["tokens_out"], s["tokens_cached"], s["tokens_replayed"], s["model_calls"], s["calls_reported"], s["wall_s"]) == (
        1000, 50, 900, 1050, 2, 1, 3.0)
    # a record with its own usage and timing: read as written (calls.json not needed)
    (tmp_path / "calls.json").unlink()
    rec = rec.model_copy(update={"usage": t, "timing": Timing(agent=2.0, model=1.5, verification=1.0)})
    s = _attempt_stats(tmp_path, rec)
    assert (s["tokens_in"], s["tokens_replayed"], s["wall_s"]) == (1000, 1050, 3.0)


def test_an_attempt_records_its_own_usage_and_model_time():
    res = HarnessResult(timings={"seed": 1.0, "agent": 4.0, "verification": 0.5})
    calls = {"agent": [{"call_id": "agent:c0", "usage": SHAPES["openai_chat"], "latency_s": 9.0},
                       {"call_id": "agent:c1", "usage": SHAPES["openai_chat"], "latency_s": 1.25},
                       {"call_id": "agent:c2", "usage": SHAPES["anthropic"], "latency_s": 0.75}]}
    settle_usage(res, calls, exclude={"agent:c0"})  # c0 was restored from a checkpoint: an earlier attempt's call
    assert (res.usage.calls, res.usage.prompt_tokens, res.model_s) == (2, 2000, 2.0)
    sk = SampleKey(task="t", policy="p", harness="h@1", sample_idx=0, seed=0)
    rec = Runner._record(NS(run_id="r"), sk, "s-a1", NS(id="t"), NS(id="p"), NS(id="h@1"), res.model_copy(update={"reward": 1.0}))
    timing = rec.timing
    assert not {"timings", "timing", "usage"} & set(rec.meta) and rec.usage == res.usage
    assert (timing.agent, timing.model, timing.wall_s, timing.stages()) == (4.0, 2.0, 5.5, res.timings)


def test_finish_reasons_are_one_vocabulary(tmp_path):
    words = ("end_turn", "max_tokens", "tool_use", "completed", "max_output_tokens", "MAX_TOKENS", "abort", "incomplete", "weird", None)
    assert [finish_reason(w) for w in words] == ["stop", "length", "tool_calls", "stop", "length", "length", "abort", None, None, None]
    # a v1 rollout holding a provider's words reads normalized: the trainer's "length" check now sees Anthropic's cut-offs
    ro = RolloutRecord.model_validate({"schema": "crucible.rollout/v1", "rollout_id": "s-a1", "sample": "s", "task": "t", "harness": "h@1", "policy": "p",
                                       "turns": [{"call_id": "c1", "actor": "agent", "finish": "max_tokens"},
                                                 {"call_id": "c2", "actor": "agent", "finish": "end_turn"}]})
    assert [t.finish for t in ro.turns] == ["length", "stop"] and Turn(call_id="c", actor="a", finish="weird").finish is None
    # a captured Anthropic call (agentenv harness): one usage shape, the finish normalized, the provider's word kept
    from crucible.interop.agentenv.harness import calls_from_capture

    log = tmp_path / "s.jsonl"
    log.write_text(json.dumps({"request": {"model": "claude"}, "response": {"stop_reason": "max_tokens", "usage": SHAPES["anthropic"]}}) + "\n")
    [call] = calls_from_capture(log, "agent", None)
    assert (call["finish"], call["finish_raw"], call["usage"]["prompt_tokens"], call["usage"]["cache_read_tokens"]) == ("length", "max_tokens", 1000, 900)
