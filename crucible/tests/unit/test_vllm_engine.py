"""VLLMServerEngine against a mock vLLM server (m13 audit 04: the engine had no tests). Token-in-token-out completions
with version stamping; the colocated lifecycle sequence (pause → sleep; publish wakes only the weights, loads, resets the
prefix cache; resume wakes the KV cache and reopens) — the KV cache must not wake inside the publish, because the
runtime offloads the trainer between publish and resume; aborts; split placement never sleeps."""

from __future__ import annotations

import asyncio
import json

import httpx

from crucible.train.engine import GenRequest
from crucible.train.vllm_http import VLLMServerEngine


def _engine(placement: str = "colocated", delay: float = 0.0):
    calls: list[str] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        q = "&".join(f"{k}={v}" for k, v in req.url.params.multi_items())
        calls.append(req.url.path + (f"?{q}" if q else ""))
        if req.url.path == "/v1/completions":
            body = json.loads(req.content)
            await asyncio.sleep(delay)
            n = body["max_tokens"]
            return httpx.Response(200, json={"choices": [{"token_ids": list(range(10, 10 + n)), "finish_reason": "length",
                                                          "logprobs": {"token_logprobs": [-0.5] * n}}]})
        return httpx.Response(200, json={})

    eng = VLLMServerEngine("http://vllm", "m", placement=placement)
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return eng, calls


async def test_token_in_token_out_and_version_stamp():
    eng, calls = _engine()
    eng.version = 3
    r = await eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1, 2, 3], max_tokens=4, stop_ids=[0]))
    assert r.completion_ids == [10, 11, 12, 13] and r.logprobs == [-0.5] * 4 and r.version == 3 and r.finish == "length" and r.prompt_tokens == 3
    assert calls == ["/v1/completions"]


async def test_colocated_lifecycle_wakes_kv_only_at_resume():
    eng, calls = _engine()
    await eng.pause()
    assert calls == ["/pause?mode=abort", "/sleep?level=1"] and eng.asleep == {"weights", "kv_cache"}
    calls.clear()
    await eng.load_weights("/ckpt/v1", 1)
    assert calls == ["/wake_up?tags=weights", "/collective_rpc", "/reset_prefix_cache"]  # no kv_cache wake inside the publish
    assert eng.version == 1 and eng.asleep == {"kv_cache"}
    calls.clear()
    await eng.resume()
    assert calls == ["/wake_up?tags=kv_cache", "/resume"] and not eng.asleep


async def test_generation_waits_while_paused_and_abort_returns_abort():
    eng, calls = _engine(delay=0.2)
    await eng.pause()
    t = asyncio.create_task(eng.generate(GenRequest(rid="r", call_id="c", prompt_ids=[1], max_tokens=2)))
    await asyncio.sleep(0.05)
    assert not t.done() and "/v1/completions" not in calls  # paused: the request is not even sent
    await eng.resume()
    await asyncio.sleep(0.05)
    n = await eng.abort_all()
    r = await t
    assert n == 1 and r.finish == "abort" and "/abort_requests" in calls


async def test_split_placement_never_sleeps():
    eng, calls = _engine(placement="split")
    await eng.pause()
    await eng.load_weights("/ckpt/v1", 1)
    await eng.resume()
    assert not any(c.startswith(("/sleep", "/wake_up")) for c in calls)


async def test_runtime_offloads_the_trainer_before_the_kv_cache_wakes():
    """The ordering contract with the runtime: publish (weights only) → trainer offload → engine resume (KV)."""
    from crucible.train.engine import EnginePool, TinyPolicy
    from crucible.train.envs import CopyTask
    from crucible.train.runtime import Runtime, RuntimeConfig
    from crucible.train.scheduler import SchedulerConfig
    from crucible.train.trainer import TorchTrainer, TrainerConfig

    eng, _calls = _engine()
    order: list[str] = []
    tr = TorchTrainer(TinyPolicy(32, 8), TrainerConfig())
    off = tr.offload

    def spy_off(*a, **k):
        order.append("offload")
        return off(*a, **k)

    tr.offload = spy_off

    class Pub:
        version = 0

        async def publish(self):
            self.version += 1
            await eng.load_weights(f"/ckpt/v{self.version}", self.version)
            order.append("publish")
            return self.version

        async def restore(self, v):
            pass

    real_resume = eng.resume

    async def spy_resume():
        order.append("resume")
        await real_resume()

    eng.resume = spy_resume
    rt = Runtime(RuntimeConfig(topology="colocated_partial", steps=2, max_turn_tokens=2, warmup_batches=0, train_state_on_rollout="cpu",
                               stop_ids=()), SchedulerConfig(n=2, B=1, max_staleness=2), EnginePool([eng]), tr, Pub(), lambda: CopyTask(turns=1),
                 {"c": [f"p{i}" for i in range(50)]})
    await rt.run()
    steps = [o for o in order if o in ("publish", "offload", "resume")]
    first = steps.index("publish")
    assert steps[first:first + 3] == ["publish", "offload", "resume"], steps
