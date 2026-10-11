"""m13 F-3: SGLangServerEngine against a mock SGLang server (SGLang's native API, as slime drives it)."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from crucible.train.engine import GenRequest
from crucible.train.sglang_http import SGLangServerEngine


def _engine(*, routed=False, delay=0.0, refuse=False):
    calls: list[tuple[str, dict]] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content) if req.content else {}
        calls.append((req.url.path, body))
        if req.url.path == "/generate":
            await asyncio.sleep(delay)
            n = body["sampling_params"]["max_new_tokens"]
            meta = {"output_token_logprobs": [[-0.25, 100 + i, None] for i in range(n)], "finish_reason": {"type": "length"}}
            if routed:
                meta["routed_experts"] = [[[i, i + 1], [i + 2, i + 3]] for i in range(n)]  # per token, per layer, top-2
            return httpx.Response(200, json={"text": "x", "meta_info": meta})
        if req.url.path == "/update_weights_from_disk":
            return httpx.Response(200, json={"success": not refuse, "message": "bad shapes" if refuse else ""})
        return httpx.Response(200, json={})

    eng = SGLangServerEngine("http://sglang", routed_experts=routed)
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return eng, calls


async def test_token_in_token_out_routing_and_resume():
    eng, calls = _engine(routed=True)
    eng.version = 2
    r = await eng.generate(GenRequest(rid="r", call_id="c1", prompt_ids=[1, 2], max_tokens=3, stop_ids=[0], seed=5))
    path, body = calls[0]
    assert path == "/generate" and body["input_ids"] == [1, 2] and body["return_logprob"] and body["return_routed_experts"]
    assert body["sampling_params"]["stop_token_ids"] == [0] and body["sampling_params"]["seed"] == 5 and not body["sampling_params"]["skip_special_tokens"]
    assert r.completion_ids == [100, 101, 102] and r.logprobs == [-0.25] * 3 and r.version == 2 and r.finish == "length"
    assert r.routing == [[0, 1, 2, 3], [1, 2, 3, 4], [2, 3, 4, 5]]  # flattened per token for R3 replay
    calls.clear()
    r2 = await eng.generate(GenRequest(rid="r", call_id="c2", prompt_ids=[1, 2], max_tokens=5, resume_completion=[7, 8], resume_logprobs=[-1.0, -1.0],
                                       resume_spans=[(0, 1)], resume_routing=[[9], [9]]))
    assert calls[0][1]["input_ids"] == [1, 2, 7, 8] and calls[0][1]["sampling_params"]["max_new_tokens"] == 3
    assert r2.completion_ids[:2] == [7, 8] and len(r2.completion_ids) == 5 and r2.version_spans == [(0, 1), (2, 2)]


async def test_per_call_abort_targets_the_calls_rid():
    eng, calls = _engine(delay=0.3)
    t = asyncio.create_task(eng.generate(GenRequest(rid="r", call_id="slow", prompt_ids=[1], max_tokens=2)))
    await asyncio.sleep(0.05)
    await eng.abort("slow")
    await t
    rid = next(b["rid"] for p, b in calls if p == "/generate")
    assert ("/abort_request", {"rid": rid}) in calls


async def test_colocated_lifecycle_and_refused_weights():
    eng, calls = _engine()
    await eng.pause()
    assert [p for p, _ in calls] == ["/pause_generation", "/release_memory_occupation"] and eng.asleep == {"weights", "kv_cache"}
    calls.clear()
    await eng.load_weights("/ckpt/v1", 1)
    assert calls == [("/resume_memory_occupation", {"tags": ["weights"]}), ("/update_weights_from_disk", {"model_path": "/ckpt/v1"}), ("/flush_cache", {})]
    assert eng.version == 1 and eng.asleep == {"kv_cache"}  # the KV cache does not wake inside the publish
    calls.clear()
    await eng.resume()
    assert calls == [("/resume_memory_occupation", {"tags": ["kv_cache"]}), ("/continue_generation", {})] and not eng.asleep
    bad, _ = _engine(refuse=True)
    with pytest.raises(RuntimeError, match="refused"):
        await bad.load_weights("/ckpt/v2", 2)
    assert bad.version == 0


async def test_resume_without_publish_reloads_released_weights():
    """Live finding (Modal, SGLang 0.5.21): release_memory_occupation frees the weights without their contents, so a resume
    with no publish in between generated from garbage. The engine reloads the last weights; after a publish it does not."""
    eng, calls = _engine()
    eng.weights_path = "/models/base"
    await eng.pause()
    await eng.resume()
    loads = [b for p, b in calls if p == "/update_weights_from_disk"]
    assert loads == [{"model_path": "/models/base"}] and ("/continue_generation", {}) in calls
    calls.clear()
    await eng.pause()
    await eng.load_weights("/ckpt/v1", 1)
    await eng.resume()
    assert [b for p, b in calls if p == "/update_weights_from_disk"] == [{"model_path": "/ckpt/v1"}]  # the publish only, no second reload
    blind, _ = _engine()
    await blind.pause()
    with pytest.raises(RuntimeError, match="no weights path"):
        await blind.resume()
