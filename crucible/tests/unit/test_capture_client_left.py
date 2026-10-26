"""M3 audit (2026-10-20) R-1 and R-2: an exchange whose client left (``docs/research/improvements/2026-10-20/M03-connectors.md``).

R-1: a client that abandons a stream is recorded as "client disconnected" when the capture proxy is served by uvicorn, as
Crucible serves it (``coding/runtime/r3.py``). Starlette cancels the stream task on a disconnect, so the row read
``CancelledError: ...`` and M10's ``verdict_of`` blamed the upstream (``infra``), masking an evidenced deadline stop as
unknown blame.

R-2: a non-streaming answer whose client already left (a deadline kill, a client timeout) was recorded as an ordinary 200,
so its tool calls became the trace's actions (a write that never happened), a trainable call, and, after a retry, a second
copy of the turn. Such an exchange stays as evidence and is never an action, a trained turn or a retry copy."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from crucible.connectors.base import SourceRef
from crucible.connectors.capture_proxy import CaptureLogConnector, CaptureProxy, Upstream
from crucible.interop.agentenv.harness import calls_from_capture
from crucible.interop.harbor.reconcile import reconcile
from crucible.runtime.agent_evidence import capture_spend, verdict_of

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


# ---------------------------------------------------------------- R-2: non-streaming exchanges whose client left
MSGS = [{"role": "system", "content": "You are a coding agent."}, {"role": "user", "content": "Fix the bug in a.py"}]


def _reply(cid: str, cmd: str) -> dict:
    return {
        "id": cid,
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {"role": "assistant", "content": None, "tool_calls": [{"id": f"{cid}-call", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": cmd})}}]},
            }
        ],
    }


def _actions(path: Path) -> tuple[int, list[tuple[str, list | None]]]:
    con = CaptureLogConnector()
    raw = con.fetch(SourceRef(connector="capture_proxy", key=path.stem, meta={"path": str(path)}))
    ev = con.normalize(raw).events
    return len(raw.payload["exchanges"]), [(e.data["call_id"], e.data.get("writes")) for e in ev if e.type == "tool_call"]


async def test_a_deadline_kill_mid_call_is_evidence_not_an_action(tmp_path):
    async def upstream(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(1.0)
        return httpx.Response(200, json=_reply("K", "echo hi > out.txt"))

    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up.invalid/v1", "k"), transport=httpx.MockTransport(upstream))
    killed, waiting = proxy.register("killed"), proxy.register("waiting")
    base, task, server = await _serve(proxy.app())
    try:

        async def ask(key: str, timeout: float) -> None:
            async with httpx.AsyncClient(base_url=base, timeout=timeout) as c:
                await c.post("/v1/chat/completions", json={"model": "m", "messages": MSGS}, headers={"authorization": f"Bearer {key}"})

        with pytest.raises(httpx.ReadTimeout):  # the agent is killed while its call is in flight
            await ask(killed, 0.3)
        proxy.unregister(killed)  # the harness revokes the key at the deadline, then drains
        await ask(waiting, 5.0)  # a client that waited for its answer
        assert await proxy.drain("killed", 5) == 0
    finally:
        server.should_exit = True
        await task
    row = json.loads((tmp_path / "cap" / "killed.jsonl").read_text())
    assert row["status"] == 200 and row["error"] == "client disconnected" and "complete" not in row  # was an ordinary 200
    assert row["response"]["usage"]["completion_tokens"] == 4 and verdict_of(row)[0] == "client_left"
    assert _actions(tmp_path / "cap" / "killed.jsonl") == (1, [])  # was [("K-call", ["file:out.txt"])]: a write that never happened
    (call,) = calls_from_capture(tmp_path / "cap" / "killed.jsonl", "agent")
    assert call["trainable"] is False and call["error"] == "client disconnected"
    assert capture_spend(tmp_path / "cap" / "killed.jsonl", {"m": [1_000_000.0, 1_000_000.0]}, self_hosted=False) == (13.0, 0.0)  # billed, certain
    ok = json.loads((tmp_path / "cap" / "waiting.jsonl").read_text())
    assert "error" not in ok and verdict_of(ok)[0] == "ok"
    assert _actions(tmp_path / "cap" / "waiting.jsonl") == (1, [("K-call", ["file:out.txt"])])
    assert "trainable" not in calls_from_capture(tmp_path / "cap" / "waiting.jsonl", "agent")[0]


async def test_a_timed_out_original_answered_after_its_retry_is_not_a_turn(tmp_path):
    n = {"first": 0}

    async def upstream(request: httpx.Request) -> httpx.Response:
        if len(json.loads(request.content)["messages"]) > 2:  # the agent's next turn
            return httpx.Response(200, json=_reply("D", "pytest"))
        n["first"] += 1
        if n["first"] == 1:  # the original: slower than the client's timeout; the upstream still answers it
            await asyncio.sleep(1.0)
            return httpx.Response(200, json=_reply("C", "rm -rf a.py"))
        return httpx.Response(200, json=_reply("B", "sed -n 1,40p a.py"))

    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up.invalid/v1", "k"), transport=httpx.MockTransport(upstream))
    key = proxy.register("s")
    base, task, server = await _serve(proxy.app())
    try:

        async def ask(messages: list, retry: str, timeout: float) -> dict:
            async with httpx.AsyncClient(base_url=base, timeout=timeout) as c:
                r = await c.post("/v1/chat/completions", json={"model": "m", "messages": messages}, headers={"authorization": f"Bearer {key}", "x-stainless-retry-count": retry})
                return r.json()

        with pytest.raises(httpx.ReadTimeout):
            await ask(MSGS, "0", 0.3)
        seen = (await ask(MSGS, "1", 5.0))["choices"][0]["message"]  # the SDK's retry: the reply the agent acts on
        await ask([*MSGS, seen, {"role": "tool", "tool_call_id": "B-call", "content": "print(0)"}], "0", 5.0)
        assert await proxy.drain("s", 5) == 0  # the abandoned original is answered and written last
    finally:
        server.should_exit = True
        await task
    rows = [json.loads(x) for x in (tmp_path / "cap" / "s.jsonl").read_text().splitlines()]
    assert [(r["response"]["id"], r.get("error")) for r in rows] == [("B", None), ("D", None), ("C", "client disconnected")]
    assert _actions(tmp_path / "cap" / "s.jsonl") == (2, [("B-call", None), ("D-call", None)])  # was 3 exchanges and C-call's write
    assert len(calls_from_capture(tmp_path / "cap" / "s.jsonl", "agent")) == 2


def _row(digest: str, cid: str, retry: str, **kw) -> dict:
    return {
        "api": "openai",
        "digest": digest,
        "request": {"model": "m", "messages": MSGS if digest == "d" else [*MSGS, {"role": "user", "content": "go on"}]},
        "headers": {"x-stainless-retry-count": retry},
        "status": 200,
        "response": _reply(cid, f"echo {cid} > {cid}.txt"),
        **kw,
    }


@pytest.mark.parametrize(
    "rows, kept",
    [
        # the agent re-sent the same body (its own retry loop: a fresh SDK request, retry count 0), turns later answered late
        ([_row("d", "B", "0"), _row("e", "D", "0"), _row("d", "C", "0", error="client disconnected")], ["B-call", "D-call"]),
        # the abandoned original answered first, then the re-sent body: the agent saw the second
        ([_row("d", "C", "0", error="client disconnected"), _row("d", "B", "0")], ["B-call"]),
        # a client-left row with no other attempt stays (the deadline kill) but is no action
        ([_row("d", "B", "0"), _row("e", "C", "0", error="client disconnected")], ["B-call"]),
    ],
)
def test_fetch_folds_a_client_left_attempt_in_either_order(tmp_path, rows, kept):
    p = tmp_path / "s.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    n, acts = _actions(p)
    assert [c for c, _ in acts] == kept and n == len({r["digest"] for r in rows})  # one exchange per body


def test_reconcile_leaves_out_a_call_the_agent_never_received():
    atif = {"schema_version": "ATIF-v1.7", "steps": [{"source": "user", "message": "do it"}, {"source": "agent", "message": "x", "metrics": {"completion_tokens": 4}}]}
    calls = [{"call_id": "agent:cap1", "usage": {"completion_tokens": 4}}, {"call_id": "agent:cap2", "usage": {"completion_tokens": 7}, "error": "client disconnected", "trainable": False}]
    rep = reconcile(atif, calls)
    assert rep.to_dict()["code"] == "turns_match" and not rep.aux_call_ids  # was atif_aux_calls: the killed call read as auxiliary
