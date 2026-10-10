"""Found by the M12 GPU gate (2026-10-10): with transformers 5 the HF chat codec took the *keys* of a BatchEncoding as
prompt token ids, every World rollout failed in record building, and the runtime masked 228k rollouts in 600 s without
saying why. Now: the codec returns ids (tested with the real Qwen3 tokenizer), every escaping exception is recorded by
blame and kind, and a circuit breaker (the M10 stop rules) stops a broken run with its reason."""

from __future__ import annotations

import asyncio

import pytest

from crucible.train.engine import EnginePool, GenResult, TinyPolicy, TorchEngine
from crucible.train.envs import CopyTask, EnvInfraError
from crucible.train.runtime import Runtime, RuntimeConfig
from crucible.train.scheduler import SchedulerConfig
from crucible.train.trainer import TorchTrainer, TrainerConfig
from crucible.train.weights import InprocPublisher


def _qwen_tokenizer():
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    except Exception as e:  # noqa: BLE001 -- offline / no cache
        pytest.skip(f"Qwen3 tokenizer unavailable: {type(e).__name__}")


def test_hf_chat_codec_returns_token_ids_and_world_rollout_builds_a_valid_record():
    from crucible_pack_toy.longhorizon.task import BacklogTask

    from crucible.actors.policy import ContextConfig
    from crucible.train.chat import HFChatCodec
    from crucible.train.rollout import full_prompts
    from crucible.train.world_env import world_rollout

    tok = _qwen_tokenizer()
    codec = HFChatCodec(tok, template_kwargs={"enable_thinking": False})
    ids = codec.render([{"role": "user", "content": "hi"}], None)
    assert ids and all(isinstance(i, int) for i in ids)
    fn = world_rollout(lambda p: BacklogTask(seed=1, n=4), codec, max_steps=3, max_tokens=64, context=ContextConfig(budget_tokens=6000, keep_recent_turns=2))

    async def gen(prompt, max_tokens, temperature):
        out = tok.encode("Let me look at the queue.", add_special_tokens=False) + [codec.eos_ids[0]]
        return GenResult(out, [-0.5] * len(out), 0, "stop", None, len(prompt))

    rec = asyncio.run(fn("r1", "s1", 0, gen, temperature=1.0, harness="h"))
    assert rec is not None and rec.turns and rec.valid_for_training()[0]
    prompts = full_prompts(rec)
    assert all(isinstance(i, int) for t in rec.turns for i in prompts[t.call_id])


def _rt(rollout_fn=None, env=None, **cfg):
    model = TinyPolicy(32, 16, seed=1)
    eng = EnginePool([TorchEngine(model, capacity=64, decode_delay_s=0.001)])
    tr = TorchTrainer(model, TrainerConfig(lr=1e-3))
    return Runtime(RuntimeConfig(topology="colocated_partial", steps=5, max_turn_tokens=2, warmup_batches=0, max_wall_s=30, **cfg), SchedulerConfig(n=2, B=2, max_staleness=2),
                   eng, tr, InprocPublisher(tr, eng), env, {"copy": [f"p{i}" for i in range(5000)]}, rollout_fn=rollout_fn)


async def test_a_harness_bug_stops_the_run_with_its_reason():
    async def broken(rid, prompt, seed, gen, **kw):
        raise TypeError("prompt ids are strings")

    rt = _rt(rollout_fn=broken)
    rep = await rt.run()
    assert rep.stop_reason == "harness_share" and not rep.steps
    assert rep.failures == {"harness:TypeError": rep.masked_rollouts} and rep.masked_rollouts < 200  # stopped early, not 228k
    assert "prompt ids are strings" in rep.failure_examples[0]["message"] and "Traceback" in rep.failure_examples[0]["traceback"]


async def test_an_infra_outage_stops_on_the_masked_share():
    class Down(CopyTask):
        async def step(self, completion):
            raise EnvInfraError("sandbox pool unreachable")

    rep = await _rt(env=lambda: Down(turns=2)).run()
    assert rep.stop_reason == "masked_share" and rep.failures.get("infra:EnvInfraError", 0) >= 20


async def test_occasional_infra_failures_do_not_trip_the_breaker():
    rep = await _rt(env=lambda: CopyTask(turns=2, infra_fail=0.03)).run()
    assert rep.stop_reason == "steps" and len(rep.steps) == 5
