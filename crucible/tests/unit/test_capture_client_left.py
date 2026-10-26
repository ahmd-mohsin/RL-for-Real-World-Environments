"""M3 audit (2026-10-20) R-1: a client that abandons a stream is recorded as "client disconnected" when the capture proxy
is served by uvicorn, as Crucible serves it (``coding/runtime/r3.py``). Starlette cancels the stream task on a disconnect,
so the row read ``CancelledError: ...`` and M10's ``verdict_of`` blamed the upstream (``infra``), masking an evidenced
deadline stop as unknown blame (``docs/research/improvements/2026-10-20/M03-connectors.md`` R-1)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from crucible.connectors.capture_proxy import CaptureProxy, Upstream
from crucible.runtime.agent_evidence import verdict_of

uvicorn = pytest.importorskip("uvicorn")


def _slow_stream() -> httpx.MockTransport:
    """An upstream that streams 60 chunks, 50 ms apart, then ``[DONE]``."""

    async def handle(request: httpx.Request) -> httpx.Response:
        async def gen():
            for i in range(60):
                yield f'data: {{"id": "c", "model": "m", "choices": [{{"index": 0, "delta": {{"content": "tok{i} "}}, "finish_reason": null}}]}}\n\n'.encode()
                await asyncio.sleep(0.05)
            yield b"data: [DONE]\n\n"

        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=gen())

    return httpx.MockTransport(handle)


async def _serve(app) -> tuple[str, asyncio.Task, uvicorn.Server]:
    from crucible.world.packaging.sidecar import free_port

    port = free_port()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning", lifespan="on"))
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.02)
    return f"http://127.0.0.1:{port}", task, server


async def _rows(path: Path, timeout_s: float = 5.0) -> list[dict]:
    for _ in range(int(timeout_s / 0.05)):
        if path.exists() and path.read_text().strip():
            break
        await asyncio.sleep(0.05)
    return [json.loads(x) for x in path.read_text().splitlines()]


async def test_an_abandoned_stream_is_client_disconnected(tmp_path):
    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up.invalid/v1", "k"), transport=_slow_stream())
    key = proxy.register("s")
    base, task, server = await _serve(proxy.app())
    try:
        async with httpx.AsyncClient(base_url=base) as c:
            body = {"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}
            async with c.stream("POST", "/v1/chat/completions", json=body, headers={"authorization": f"Bearer {key}"}) as r:
                n = 0
                async for line in r.aiter_lines():
                    n += line.startswith("data:")
                    if n >= 5:
                        break  # the agent is killed mid-stream: leaving the block closes the connection
        rows = await _rows(tmp_path / "cap" / "s.jsonl")
    finally:
        server.should_exit = True
        await task
    assert len(rows) == 1
    row = rows[0]
    assert row["status"] == 200 and row["complete"] is False and row["error"] == "client disconnected"  # was "CancelledError: ..."
    assert row["response"]["choices"][0]["message"]["content"].startswith("tok0 ")  # the partial reply is kept
    assert verdict_of(row)[0] == "client_left"  # was "infra": the upstream was blamed
    assert proxy.inflight.get("s", 0) == 0
