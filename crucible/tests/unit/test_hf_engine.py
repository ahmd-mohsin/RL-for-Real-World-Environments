"""m13 F-1 / F-2: a real HF causal LM as a generation engine (CPU), and the engine capability protocol."""

from __future__ import annotations

import asyncio
import copy

import pytest
import torch

from crucible.train.engine import EnginePool, GenRequest, ScriptedEngine
from crucible.train.hf import HFPolicy
from crucible.train.hf_engine import HFGenerateEngine, capabilities, probe_logprobs_mode


def _qwen(seed=0):
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                                        num_key_value_heads=2, head_dim=8, max_position_embeddings=512, tie_word_embeddings=True))


async def test_behaviour_logprobs_equal_the_trainers_logprobs():
    base = _qwen()
    eng = HFGenerateEngine(copy.deepcopy(base), device="cpu")
    pol = HFPolicy(copy.deepcopy(base), lm_head_fp32=True)
    for temp, top_p in ((1.0, 1.0), (0.7, 1.0), (1.3, 1.0)):
        r = await eng.generate(GenRequest(rid="r", call_id=f"c{temp}", prompt_ids=[1, 5, 9, 2], max_tokens=12, temperature=temp, top_p=top_p, seed=3))
        assert r.finish == "length" and len(r.completion_ids) == 12 and r.version == 0
        with torch.no_grad():
            ref = pol.token_logprobs([1, 5, 9, 2], r.completion_ids, temp)
        assert torch.allclose(torch.tensor(r.logprobs), ref.float(), atol=1e-4), (temp, r.logprobs[:3], ref[:3])  # KV-cache decode = full forward
    assert await probe_logprobs_mode(eng, pol, [1, 2, 3]) == "processed"


async def test_probe_detects_raw_logprobs():
    base = _qwen()
    pol = HFPolicy(copy.deepcopy(base), lm_head_fp32=True)

    class RawEngine(HFGenerateEngine):  # reports temperature-1 log-probs, as vLLM does by default
        async def generate(self, req):
            res = await super().generate(req)
            with torch.no_grad():
                res.logprobs = pol.token_logprobs(list(req.prompt_ids), res.completion_ids, 1.0).float().tolist()
            return res

    assert await probe_logprobs_mode(RawEngine(copy.deepcopy(base), device="cpu"), pol, [1, 2, 3]) == "raw"


async def test_per_call_abort_and_pinned_weights_on_publish():
    eng = HFGenerateEngine(_qwen(), device="cpu", decode_delay_s=0.01)
    a = asyncio.create_task(eng.generate(GenRequest(rid="a", call_id="a:c1", prompt_ids=[1, 2], max_tokens=40, seed=1)))
    b = asyncio.create_task(eng.generate(GenRequest(rid="b", call_id="b:c1", prompt_ids=[1, 2], max_tokens=6, seed=1)))
    await asyncio.sleep(0.05)
    await eng.abort("a:c1")
    new = _qwen(seed=9).state_dict()
    await eng.load_weights(new, 1)  # calls are running: loaded into a copy, they keep version 0
    ra, rb = await a, await b
    assert ra.finish == "abort" and len(ra.completion_ids) < 40 and rb.finish == "length" and rb.version == 0
    rc = await eng.generate(GenRequest(rid="c", call_id="c:c1", prompt_ids=[1, 2], max_tokens=6, seed=1))
    assert rc.version == 1 and rc.logprobs != rb.logprobs  # the new weights serve new calls
    with pytest.raises(ValueError, match="do not match"):
        await eng.load_weights({"bogus.weight": torch.zeros(1)}, 2)


def test_capabilities_protocol():
    eng = HFGenerateEngine(_qwen(), device="cpu")
    assert capabilities(eng)["abort"] == "per_call" and capabilities(eng)["logprobs"] == "processed"
    pool = EnginePool([eng, ScriptedEngine(lambda rid, k, p: [1])])
    caps = pool.capabilities()
    assert caps["token_ids"] is True and caps["top_p_candidates"] is False and caps["logprobs"] == "mixed"  # only what every engine offers


async def test_m12_runtime_on_a_real_hf_engine():
    from crucible.train.envs import CopyTask
    from crucible.train.mismatch import Correction
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig
    from crucible.train.weights import InprocPublisher

    base = _qwen()
    eng = EnginePool([HFGenerateEngine(copy.deepcopy(base), device="cpu", capacity=32)])
    tr = TorchTrainer(HFPolicy(copy.deepcopy(base), lm_head_fp32=True), TrainerConfig(lr=1e-3, correction=Correction("dis"), item_unit="call"))
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=4, max_turn_tokens=3, warmup_batches=0), SchedulerConfig(n=2, B=2, max_staleness=2),
                 eng, tr, InprocPublisher(tr, eng), lambda: CopyTask(turns=2, symbols=8), {"copy": [f"p{i}" for i in range(500)]})
    rep = await rt.run()
    assert len(rep.steps) == 4 and rep.publishes == 4 and rep.masked_rollouts == 0
    assert max(abs(s.metrics["mismatch/rollout_probs_diff"]) for s in rep.steps) < 1e-3  # same fp32 weights: no train-inference gap
