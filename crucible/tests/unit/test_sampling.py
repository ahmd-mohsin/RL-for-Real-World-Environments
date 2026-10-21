"""The sampling parameters actually sent, per call (M0 audit 2026-10-14 R-3)."""

from __future__ import annotations

import asyncio

import pytest
from openai.types.chat import ChatCompletion

from crucible.actors.llm import LLMClient
from crucible.core.config import Config
from crucible.core.schema.rollout import RolloutRecord, SamplingParams, Turn
from crucible.core.schema.run import RunRecord
from crucible.runtime.record import build


def test_sampling_params_from_a_request_and_from_a_v1_dict():
    s = SamplingParams.from_request({"model": "m", "messages": [], "temperature": 0.7, "max_tokens": 800, "seed": None, "logprobs": True,
                                     "extra_body": {"top_k": 20, "return_token_ids": True}, "stop": "END"})
    assert s.model_dump() == {"temperature": 0.7, "top_k": 20, "max_tokens": 800, "stop": ["END"]} and s.truncated
    legacy = SamplingParams.model_validate({"temperature": 1.0, "max_completion_tokens": 64, "requested": {"temperature": 0.5}})
    assert (legacy.temperature, legacy.max_tokens, legacy.extra) == (1.0, 64, {"requested": {"temperature": 0.5}})
    assert SamplingParams().model_dump() == {} and not SamplingParams(top_p=1.0, top_k=-1).truncated


def _client(tmp_path, *, cache_mode="off"):
    cfg = Config.model_validate({"llm_providers": {"lab": {"base_url": "http://127.0.0.1:8000/v1", "api_key": "EMPTY",
                                                           "request_defaults": {"top_p": 0.9, "extra_body": {"return_token_ids": True}}}},
                                 "models": {"policy-default": "lab:m"}, "data_dir": str(tmp_path)})
    llm = LLMClient("policy-default", config=cfg, cache_mode=cache_mode, cache_dir=tmp_path / "cache")
    sent = []

    class Completions:
        async def create(self, **kw):
            sent.append(kw)
            return ChatCompletion.model_validate({"id": "x", "object": "chat.completion", "created": 0, "model": "m",
                                                  "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "hi"}}],
                                                  "usage": {"prompt_tokens": 3, "completion_tokens": 1, "total_tokens": 4}})

    class Client:
        chat = type("Chat", (), {"completions": Completions()})()

    llm._openai = lambda: Client()
    return llm, sent


def test_the_client_records_what_it_sent_including_provider_defaults(tmp_path):
    llm, sent = _client(tmp_path)
    r = asyncio.run(llm.complete([{"role": "user", "content": "q"}], temperature=0.7, max_tokens=64, seed=None, extra_body={"top_k": 20}))
    assert sent[0]["top_p"] == 0.9  # the provider's default went out
    assert r.sampling == {"temperature": 0.7, "top_p": 0.9, "top_k": 20, "max_tokens": 64}


def test_a_cached_reply_keeps_the_sampling_it_was_produced_with(tmp_path):
    llm, sent = _client(tmp_path, cache_mode="record")
    msgs = [{"role": "user", "content": "q"}]
    first = asyncio.run(llm.complete(msgs, temperature=0.7, max_tokens=64))
    again = asyncio.run(llm.complete(msgs, temperature=0.7, max_tokens=64))
    assert again.cached and len(sent) == 1 and again.sampling == first.sampling == {"temperature": 0.7, "top_p": 0.9, "max_tokens": 64, "seed": 0}


def test_a_turn_records_its_calls_sampling_not_the_runs_declaration():
    rec = RunRecord(run_id="r", task="t", world="w", policy="p", started_at="2026-01-01T00:00:00Z", steps=1, outcome="completed", reward=1.0)
    calls = {"agent": [{"call_id": "agent:c1", "kind": "turn", "segment": 0, "actions": [],
                        "tokens": {"prompt_token_ids": [1, 2], "completion_token_ids": [3], "logprobs": [-0.1], "policy_version": 0},
                        "sampling": {"temperature": 0.7, "top_p": 0.9}},
                       {"call_id": "agent:c2", "kind": "turn", "segment": 0, "actions": [],
                        "tokens": {"prompt_token_ids": [3], "completion_token_ids": [4], "logprobs": [-0.2], "policy_version": 0}}]}
    ro = build(rec, calls=calls)
    first, second = ro.turns
    assert (first.sampling.temperature, first.sampling.top_p) == (0.7, 0.9) and second.sampling == SamplingParams()
    assert first.sampling.truncated and not first.candidates_valid()  # top_p 0.9 needs candidate sets
    assert ro.schema_ == RolloutRecord.SCHEMA  # the current rollout version (v1.1 typed sampling; v1.2 added gate_kinds)


def test_the_trainer_refuses_an_unrecorded_or_different_temperature():
    pytest.importorskip("torch")
    from crucible.train.engine import TinyPolicy
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    def rec(rid, reward, sampling):
        t = Turn(call_id="c0", actor="agent", prompt_token_ids=[1, 2], completion_token_ids=[3, 4], logprobs=[-1.0, -1.0], policy_version=0,
                 sampling=sampling)
        return RolloutRecord(rollout_id=rid, sample=rid, task="t", harness="h", policy="p", reward=reward, turns=[t])

    with pytest.raises(ValueError, match="recorded no sampling temperature"):
        TorchTrainer(TinyPolicy(32, 8), TrainerConfig()).step([rec("a", 1.0, {}), rec("b", 0.0, {})], ["g", "g"])
    with pytest.raises(ValueError, match="temperature 0.7"):
        TorchTrainer(TinyPolicy(32, 8), TrainerConfig()).step([rec("a", 1.0, {"temperature": 0.7}), rec("b", 0.0, {"temperature": 0.7})], ["g", "g"])
    out = TorchTrainer(TinyPolicy(32, 8), TrainerConfig(allow_unrecorded_sampling=True)).step([rec("a", 1.0, {}), rec("b", 0.0, {})], ["g", "g"])
    assert "train/loss" in out
