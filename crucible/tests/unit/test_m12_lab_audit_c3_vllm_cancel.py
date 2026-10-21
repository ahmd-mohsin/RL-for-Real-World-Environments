"""M12 lab audit C-3(a): ``VLLMServerEngine.generate`` turned the caller's own cancellation into an ``abort`` result.

Cancelling the task that awaits ``generate`` cancels the request task it awaits, so ``task.cancelled()`` was True on both
paths. An evicted or redundant rollout, a shutdown or a timeout therefore came back as an abort, which ``Runtime._call``
re-issues: the rollout kept running and holding an engine slot. Only ``abort_all`` (a barrier) is an abort now."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from crucible.train.engine import GenRequest
from crucible.train.vllm_http import VLLMServerEngine


def _engine(delay: float):
    sent: list[int] = []

    async def handler(req: httpx.Request) -> httpx.Response:
        if req.url.path == "/v1/completions":
            body = json.loads(req.content)
            sent.append(body["max_tokens"])
            await asyncio.sleep(delay)
            n = body["max_tokens"]
            return httpx.Response(200, json={"choices": [{"token_ids": list(range(10, 10 + n)), "finish_reason": "length",
                                                          "logprobs": {"token_logprobs": [-0.5] * n}}]})
        return httpx.Response(200, json={})

    eng = VLLMServerEngine("http://vllm", "m", placement="split")
    eng.client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return eng, sent


def _req(**kw) -> GenRequest:
    return GenRequest(rid="r", call_id="c", prompt_ids=[1], max_tokens=2, **kw)


async def test_cancelling_the_caller_propagates():
    eng, sent = _engine(delay=5.0)
    t = asyncio.create_task(eng.generate(_req()))
    await asyncio.sleep(0.05)
    assert sent == [2] and len(eng._tasks) == 1
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t
    assert t.cancelled() and not eng._tasks  # the request was dropped with its caller


async def test_cancelling_a_resumed_call_propagates_too():
    eng, _ = _engine(delay=5.0)
    t = asyncio.create_task(eng.generate(_req(resume_completion=[7], resume_logprobs=[-0.1], resume_spans=[(0, 0)])))
    await asyncio.sleep(0.05)
    t.cancel()
    with pytest.raises(asyncio.CancelledError):
        await t


async def test_a_timeout_is_a_timeout_not_an_abort():
    eng, _ = _engine(delay=5.0)
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(eng.generate(_req()), timeout=0.05)


async def test_abort_all_is_still_an_abort():
    eng, _ = _engine(delay=5.0)
    t = asyncio.create_task(eng.generate(_req(resume_completion=[7], resume_logprobs=[-0.1], resume_spans=[(0, 0)])))
    await asyncio.sleep(0.05)
    assert await eng.abort_all() == 1
    r = await t
    assert r.finish == "abort" and r.completion_ids == [7]  # the kept output survives the barrier's abort
