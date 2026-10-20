"""M10 lab audit R-1 (2026-10-15), slice 1: the evidence external-agent harnesses decide blame from. The capture proxy
tells a dropped stream from a finished one and a capped session from a live one; captured calls keep status, error and
completeness; ``runtime.agent_evidence`` is the Crucible-owned classification table and settles spend from the capture.
No Harbor, no agentenv, no network: fake upstreams behind ``httpx.MockTransport``."""

from __future__ import annotations

import asyncio
import dataclasses
import json
from pathlib import Path

import httpx
import pytest

from crucible.actors.llm import UNPRICED_USD_PER_M
from crucible.connectors.base import SourceRef
from crucible.connectors.capture_proxy import (
    LIMIT_STOP_KIND,
    LIMIT_STOP_ROWS,
    LIMIT_STOP_TEXT,
    CaptureLogConnector,
    CaptureProxy,
    Upstream,
    assemble_anthropic,
    assemble_openai,
)
from crucible.runtime.agent_evidence import (
    AgentEnd,
    UpstreamEvidence,
    capture_cost,
    capture_spend,
    classify_agent_end,
    confirm_upstream,
    host_secret_names,
    is_local_base,
    leaked_host_secrets,
    read_upstream_evidence,
    self_hosted_apis,
    verdict_of,
)
from crucible.runtime.infra import LimitPolicy, failure

KEY = "sk-ant-api03-" + "Q" * 40  # credential-shaped: must never survive into a stored reason or call


# Harbor's classes, by name only (agents/installed/base.py:34-151): the table matches names, never imports Harbor
class NonZeroAgentExitCodeError(RuntimeError):
    pass


class ApiRateLimitError(NonZeroAgentExitCodeError):
    pass


class ContextWindowExceededError(NonZeroAgentExitCodeError):
    pass


# ------------------------------------------------------------------ helpers


def _proxy(tmp_path: Path, handler) -> CaptureProxy:
    # both upstreams configured: forward() answers 404 before any cap check for a dialect with no upstream
    return CaptureProxy(tmp_path / "cap", openai=Upstream("http://up/v1", "k"), anthropic=Upstream("http://up/v1", "k"),
                        transport=httpx.MockTransport(handler))


def _rows(tmp_path: Path, session: str = "s") -> list[dict]:
    return [json.loads(x) for x in (tmp_path / "cap" / f"{session}.jsonl").read_text().splitlines() if x.strip()]


def _sse(events: list, *, named: bool = False) -> bytes:
    out = []
    for e in events:
        if isinstance(e, str):
            out.append(f"data: {e}\n\n")
        else:
            out.append((f"event: {e['type']}\n" if named and "type" in e else "") + f"data: {json.dumps(e)}\n\n")
    return "".join(out).encode()


def _streaming(body: bytes):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body)
    return handler


async def _post(proxy: CaptureProxy, key: str, path: str, body: dict) -> httpx.Response:
    h = {"x-api-key": key, "anthropic-version": "2023-06-01"} if path == "/v1/messages" else {"authorization": f"Bearer {key}"}
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as c:
        return await c.post(path, json=body, headers=h)


def _events(text: str) -> list[dict]:
    return [json.loads(x[5:].strip()) for x in text.splitlines() if x.startswith("data:") and x[5:].strip() != "[DONE]"]


def _chunk(content: str | None = None, finish: str | None = None) -> dict:
    return {"id": "r1", "model": "m", "choices": [{"index": 0, "delta": {"content": content} if content else {}, "finish_reason": finish}]}


ANTHROPIC_OK = [{"type": "message_start", "message": {"id": "m1", "model": "c", "usage": {"input_tokens": 3}}},
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hi"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 1}},
                {"type": "message_stop"}]


def _body(path: str, n: int = 0, stream: bool = False) -> dict:
    if path == "/v1/responses":
        return {"model": "m", "input": f"turn {n}", "stream": stream}
    return {"model": "m", "max_tokens": 10, "stream": stream, "messages": [{"role": "user", "content": f"turn {n}"}]}


# ------------------------------------------------------------------ stream terminal marker


async def test_openai_stream_without_terminal_marker_is_stream_drop(tmp_path):
    proxy = _proxy(tmp_path, _streaming(_sse([_chunk("he"), _chunk("llo")])))  # no finish_reason, no [DONE]
    key = proxy.register("s")
    r = await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", stream=True))
    assert r.status_code == 200
    [row] = _rows(tmp_path)
    assert row["complete"] is False and row["error"].startswith("stream_drop")  # was complete: a cut turn read as finished
    assert row["response"]["choices"][0]["message"]["content"] == "hello"  # what the agent did see is still recorded


@pytest.mark.parametrize(("path", "events"), [
    ("/v1/chat/completions", [_chunk("he"), _chunk("llo"), "[DONE]"]),
    ("/v1/chat/completions", [_chunk("he"), _chunk(None, "stop")]),  # a finish_reason with no [DONE] is a terminal marker
    ("/v1/messages", ANTHROPIC_OK),
], ids=["openai-done", "openai-finish-reason", "anthropic-message-stop"])
async def test_streams_with_a_terminal_marker_are_complete(tmp_path, path, events):
    proxy = _proxy(tmp_path, _streaming(_sse(events, named=path == "/v1/messages")))
    key = proxy.register("s")
    await _post(proxy, key, path, _body(path, stream=True))
    [row] = _rows(tmp_path)
    assert row["complete"] is True and "error" not in row


@pytest.mark.parametrize(("path", "events"), [
    ("/v1/messages", [ANTHROPIC_OK[0], {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]),
    ("/v1/chat/completions", [_chunk("he"), {"error": {"message": "overloaded"}}, "[DONE]"]),  # an error chunk, then [DONE]
], ids=["anthropic-error-event", "openai-error-chunk"])
async def test_stream_error_events_are_incomplete(tmp_path, path, events):
    proxy = _proxy(tmp_path, _streaming(_sse(events, named=path == "/v1/messages")))
    key = proxy.register("s")
    await _post(proxy, key, path, _body(path, stream=True))
    [row] = _rows(tmp_path)
    assert row["complete"] is False and row["error"].startswith("stream_error") and "overloaded" in row["error"].lower()


# ------------------------------------------------------------------ per-session call cap


def _json_upstream(hits: list[str]):
    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(request.url.path)
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json={"id": "m1", "type": "message", "role": "assistant", "model": "c", "content": [{"type": "text", "text": "ok"}],
                                             "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2}})
        if request.url.path.endswith("/responses"):
            return httpx.Response(200, json={"object": "response", "status": "completed", "model": "m",
                                             "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]})
        return httpx.Response(200, json={"id": "r", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                                         "usage": {"prompt_tokens": 3, "completion_tokens": 2}})
    return handler


async def test_call_cap_answers_a_stop_without_an_upstream_call(tmp_path):
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    key = proxy.register("s", max_calls=2)
    replies = [await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", n)) for n in range(3)]  # distinct bodies: no retry collapse
    assert len(hits) == 2 and proxy.forwarded["s"] == 2  # the third call never left the proxy
    third = replies[2]
    assert third.status_code == 200
    choice = third.json()["choices"][0]
    assert choice["message"]["content"] == LIMIT_STOP_TEXT and choice["finish_reason"] == "stop" and not choice["message"].get("tool_calls")
    rows = _rows(tmp_path)
    assert [r.get("kind") for r in rows] == [None, None, LIMIT_STOP_KIND] and sum(1 for r in rows if "request" in r) == 2
    assert rows[2]["calls"] == 2 and rows[2]["max_calls"] == 2 and LIMIT_STOP_TEXT not in json.dumps(rows[2])  # the stop is never captured as a turn
    c = CaptureLogConnector()
    raw = c.fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(tmp_path / "cap" / "s.jsonl")}))
    assert len(raw.payload["exchanges"]) == 2 and raw.payload["limit_stops"] == 1
    trace = c.normalize(raw)  # a limit_stop row has no request: it used to KeyError here
    assert LIMIT_STOP_TEXT not in json.dumps([e.text for e in trace.events])


@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages", "/v1/responses"])
async def test_capped_stream_gets_a_stop_in_its_dialect(tmp_path, path):
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    key = proxy.register("s", max_calls=1)  # 0 means unlimited
    assert (await _post(proxy, key, path, _body(path, 0))).status_code == 200
    r = await _post(proxy, key, path, _body(path, 1, stream=True))
    assert len(hits) == 1 and r.status_code == 200 and r.headers["content-type"].startswith("text/event-stream")
    assert LIMIT_STOP_TEXT in r.text
    evs = _events(r.text)
    if path == "/v1/chat/completions":
        assert "data: [DONE]" in r.text
        msg = assemble_openai(evs)["choices"][0]
        assert msg["message"]["content"] == LIMIT_STOP_TEXT and msg["finish_reason"] == "stop"
    elif path == "/v1/messages":
        assert "event: message_stop" in r.text
        msg = assemble_anthropic(evs)
        assert msg["content"][0]["text"] == LIMIT_STOP_TEXT and msg["stop_reason"] == "end_turn"
    else:
        assert "event: response.completed" in r.text
        assert evs[-1]["response"]["output"][0]["content"][0]["text"] == LIMIT_STOP_TEXT
    nonstream = await _post(proxy, key, path, _body(path, 2))  # the non-streamed stop, in the same dialect
    assert LIMIT_STOP_TEXT in nonstream.text and len(hits) == 1
    assert sum(1 for x in _rows(tmp_path) if x.get("kind") == LIMIT_STOP_KIND) == 2


async def test_unregister_drops_the_cap(tmp_path):
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    key = proxy.register("s", max_calls=1)
    await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", 0))
    assert proxy.forwarded["s"] == 1
    proxy.unregister(key)
    proxy.unregister(key)  # idempotent
    assert "s" not in proxy.max_calls and "s" not in proxy.forwarded and not proxy.sessions and "s" not in proxy.limit_stops
    assert (await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", 1))).status_code == 401  # revoked
    key2 = proxy.register("s", max_calls=3)
    assert proxy.forwarded["s"] == 0 and proxy.max_calls["s"] == 3
    assert (await _post(proxy, key2, "/v1/chat/completions", _body("/v1/chat/completions", 2))).json()["choices"][0]["message"]["content"] == "ok"
    with pytest.raises(ValueError):
        proxy.register("t", max_calls=-1)


# ------------------------------------------------------------------ captured calls keep status, error, completeness


def test_calls_from_capture_keeps_status_error_complete(tmp_path):
    from crucible.interop.agentenv.harness import calls_from_capture  # imports only httpx at top level: no agent_env needed

    req = lambda n: {"model": "m", "messages": [{"role": "user", "content": f"turn {n}"}]}  # noqa: E731
    rows = [{"api": "openai", "digest": "d1", "request": req(1), "status": 200, "latency_s": 0.5,
             "response": {"choices": [{"message": {"content": "a"}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}},
            {"api": "openai", "digest": "d2", "request": req(2), "status": 503, "response": {"error": {"message": f"overloaded, key {KEY}"}}},
            {"api": "openai", "digest": "d3", "request": req(3), "status": 200, "stream": True, "complete": False,
             "error": "stream_drop: the upstream body ended without a terminal marker",
             "response": {"choices": [{"index": 0, "message": {"content": "par"}, "finish_reason": None}], "usage": None}},
            {"kind": LIMIT_STOP_KIND, "api": "openai", "calls": 3, "max_calls": 3}]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    calls = calls_from_capture(p, "agent")
    assert [c["status"] for c in calls] == [200, 503, 200]  # the limit_stop row is not a call
    assert "error" not in calls[0] and "complete" not in calls[0] and calls[0]["latency_s"] == 0.5
    assert "overloaded" in calls[1]["error"] and KEY not in calls[1]["error"]  # scrubbed: a provider can echo a key
    assert calls[2]["error"].startswith("stream_drop") and calls[2]["complete"] is False
    assert calls_from_capture(None, "agent") == [] and calls_from_capture(tmp_path / "missing.jsonl", "agent") == []


# ------------------------------------------------------------------ upstream verdicts


@pytest.mark.parametrize(("row", "verdict", "detail"), [
    ({"status": 503, "response": {"error": "down"}}, "infra", "HTTP 503"),
    ({"status": 429, "response": {"error": {"message": "slow down"}}}, "infra", "HTTP 429"),
    ({"status": 408, "response": {}}, "infra", "HTTP 408"),
    ({"status": 502, "complete": False, "error": "ConnectError: connection refused"}, "infra", "connect"),  # the proxy's own record
    ({"status": 200, "stream": True, "complete": False, "error": "stream_drop: cut", "response": {"choices": []}}, "infra", "stream_drop"),
    ({"status": 200, "stream": True, "complete": False, "error": "stream_error: {'type': 'overloaded_error'}", "response": {}}, "infra", "stream_error"),
    ({"status": 200, "stream": True, "complete": False, "error": "client disconnected", "response": {}}, "client_left", None),
    ({"status": 402, "response": {"error": "pay"}}, "credits", None),
    ({"status": 429, "response": {"error": {"code": "insufficient_quota"}}}, "credits", None),  # OpenAI's out-of-credit 429
    ({"status": 401, "response": {"error": "bad key"}}, "auth", None),
    ({"status": 403, "response": {"error": "forbidden"}}, "auth", None),
    ({"status": 404, "response": {"error": "no such model"}}, "not_found", None),
    ({"status": 400, "response": {"error": {"message": "This model's maximum context length is 8192 tokens"}}}, "context_overflow", None),
    ({"status": 400, "response": {"error": {"message": "invalid tool schema"}}}, "client_error", None),
    ({"status": 200, "response": {"error": "x"}}, "infra", "error_body"),
    ({"status": 200, "response": {"choices": [{"message": {"content": "ok"}}]}}, "ok", None),
])
def test_upstream_verdicts(row, verdict, detail):
    v, d = verdict_of(row)
    assert v == verdict
    if detail is not None:
        assert d.startswith(detail)


def test_read_upstream_evidence_uses_the_last_real_row(tmp_path):
    p = tmp_path / "s.jsonl"
    assert read_upstream_evidence(p) == UpstreamEvidence() and read_upstream_evidence(None).verdict == "none"
    ok = {"digest": "a", "request": {}, "status": 200, "response": {"choices": []}}
    bad = {"digest": "b", "request": {}, "status": 503, "response": {"error": f"overloaded {KEY}"}}
    stop = {"kind": LIMIT_STOP_KIND, "calls": 2, "max_calls": 2}
    p.write_text("\n".join(json.dumps(r) for r in (ok, bad, stop)) + "\n" + '{"digest": "c", "requ')  # a torn last line
    up = read_upstream_evidence(p)
    assert (up.verdict, up.detail, up.calls, up.limit_stops, up.last_status) == ("infra", "HTTP 503", 2, 1, 503)
    assert up.to_dict()["verdict"] == "infra" and up.failed == 1 and up.probe is None
    p.write_text("\n".join(json.dumps(r) for r in (bad, ok)) + "\n")  # an agent that retried through a transient 5xx
    assert read_upstream_evidence(p).verdict == "ok"  # the last call decides (AReaL's any-call rule, relaxed on purpose)
    p.write_text(json.dumps({"digest": "d", "request": {}, "status": 502, "complete": False, "error": f"ConnectError: {KEY}"}) + "\n")
    up = read_upstream_evidence(p)
    assert up.verdict == "infra" and KEY not in json.dumps(up.to_dict())


# ------------------------------------------------------------------ the classification table


SAME = "same"


def _up(verdict: str = "ok", detail: str = "HTTP 200", limit_stops: int = 0, probe: str | None = SAME) -> UpstreamEvidence:
    """Evidence of one recorded call; by default Crucible's probe agreed with it (confirmed evidence)."""
    return UpstreamEvidence(calls=1, limit_stops=limit_stops, verdict=verdict, detail=detail, model="m", probe=verdict if probe == SAME else probe)


def test_classification_table():
    # acceptance: an upstream 503 followed by a non-zero agent exit is masked provider_unavailable
    end = classify_agent_end(upstream=_up("infra", "HTTP 503"), error=NonZeroAgentExitCodeError("exit 1"))
    assert end.failure.kind == "provider_unavailable" and end.failure.masks and end.failure.who == "infra"
    # acceptance: an agent that prints "rate limit" with a healthy proxy is scored (its own outcome)
    end = classify_agent_end(upstream=_up(), error=ApiRateLimitError("rate limit"))
    assert end.failure.kind == "agent_run_error" and end.failure.who == "policy" and not end.failure.masks
    assert "ApiRateLimitError" in end.failure.reason
    # host evidence
    end = classify_agent_end(upstream=_up(), error=NonZeroAgentExitCodeError("x"), container_lost="removed")
    assert end.failure.kind == "crucible:env_container_lost" and end.failure.masks
    end = classify_agent_end(upstream=_up("infra", "HTTP 503"), unreachable="3 polls", container_lost="removed")  # unreachable first
    assert end.failure.kind == "transport_unreachable" and end.failure.masks and end.failure.detail == "agent_unreachable"
    # Crucible's own limits are evidenced stops
    end = classify_agent_end(upstream=UpstreamEvidence(), deadline=True)
    assert end.stop == "timeout" and end.failure is None and end.evidenced
    end = classify_agent_end(upstream=_up(limit_stops=1), error=RuntimeError("format errors"))
    assert end.stop == "max_steps" and end.failure is None and end.evidenced
    # mini-swe-agent's typed exits: scored stops, agent-reported
    for status, stop in (("LimitsExceeded", "max_steps"), ("TimeExceeded", "timeout"), ("RepeatedFormatError", "too_many_errors"), ("Submitted", "agent_done")):
        end = classify_agent_end(upstream=_up(), exit_status=status)
        assert end.stop == stop and end.failure is None and not end.evidenced
    assert classify_agent_end(upstream=_up(), exit_status="TimeExceeded").why == "agent-reported"
    # overflow: the agent's class name (scored, agent-reported) or the upstream's own answer (recorded, not evidenced: M10 lab
    # audit R-1, the policy can provoke one with an oversized request of its own)
    end = classify_agent_end(upstream=_up(), error=ContextWindowExceededError("too long"))
    assert end.stop == "context_overflow" and end.failure is None and not end.evidenced
    end = classify_agent_end(upstream=_up("context_overflow", "HTTP 400"), error=NonZeroAgentExitCodeError("x"))
    assert end.stop == "context_overflow" and not end.evidenced and "upstream" in end.why
    # the upstream's own refusals
    end = classify_agent_end(upstream=_up("credits", "HTTP 402"), error=NonZeroAgentExitCodeError("x"))
    assert end.failure.kind == "crucible:credits_exhausted" and end.failure.masks
    end = classify_agent_end(upstream=_up("auth", "HTTP 401"), error=NonZeroAgentExitCodeError("x"))
    assert end.failure.kind == "crucible:harness_unsupported" and end.failure.retryable is False and end.failure.masks
    # A2A self-reports
    end = classify_agent_end(upstream=_up("infra", "HTTP 503"), self_report="failed:agent_error")
    assert end.failure.kind == "provider_unavailable" and end.failure.masks
    end = classify_agent_end(upstream=_up("client_error", "HTTP 400"), self_report="failed:agent_error")
    assert end.failure.who == "policy" and not end.failure.masks
    end = classify_agent_end(upstream=_up(), self_report="failed:infra_error")  # an unconfirmed infra claim: unknown, masked
    assert end.failure.who == "unknown" and end.failure.masks
    assert classify_agent_end(upstream=_up(), self_report="canceled").failure.who == "unknown"
    assert classify_agent_end(upstream=_up(), self_report="completed").stop == "agent_done"
    # anything else: unknown
    end = classify_agent_end(upstream=_up(), error=RuntimeError("boom"))
    assert end.failure.who == "unknown" and end.failure.detail == "unmapped:RuntimeError"
    end = classify_agent_end(upstream=_up(), exit_status="APIConnectionError")
    assert end.failure.who == "unknown" and end.failure.detail == "unmapped:APIConnectionError"
    # a TimeoutError raised inside the agent is not Crucible's deadline
    end = classify_agent_end(upstream=_up(), error=TimeoutError("inner"))
    assert end.stop != "timeout" and end.failure.who == "unknown"


@pytest.mark.parametrize(("verdict", "kind"), [("infra", "provider_unavailable"), ("credits", "crucible:credits_exhausted"),
                                               ("auth", "crucible:harness_unsupported"), ("not_found", "crucible:harness_unsupported"),
                                               ("client_error", None), ("ok", None), ("context_overflow", None)])
def test_deadline_checks_the_upstream_first(verdict, kind):
    """A provider outage that kept the agent retrying until Crucible's deadline is the provider's, not a scored timeout."""
    end = classify_agent_end(upstream=_up(verdict, "HTTP 503" if verdict == "infra" else "x"), deadline=True)
    if kind is None:
        assert end.stop == "timeout" and end.failure is None and end.evidenced
    else:
        assert end.failure.kind == kind and end.failure.masks


def test_only_evidenced_stops_follow_a_mask_policy():
    """P23: a stop read from an agent-writable log or parsed from the agent's output is always scored."""
    masked = LimitPolicy(on_timeout="mask", on_context_window="mask", on_steps="mask")
    for end in (classify_agent_end(upstream=_up(), exit_status="TimeExceeded"), classify_agent_end(upstream=_up(), exit_status="LimitsExceeded"),
                classify_agent_end(upstream=_up(), error=ContextWindowExceededError("x")),
                classify_agent_end(upstream=_up("context_overflow", "HTTP 400"), error=NonZeroAgentExitCodeError("x"))):
        assert end.rule(masked) == "score" and end.penalty(LimitPolicy(on_timeout="penalty:-1", on_steps="penalty:-1", on_context_window="penalty:-1")) is None
    for end in (classify_agent_end(upstream=_up(), deadline=True), classify_agent_end(upstream=_up(limit_stops=2))):
        assert end.rule(masked) == "mask"
    assert AgentEnd(stop="timeout", evidenced=True).penalty(LimitPolicy(on_timeout="penalty:-0.5")) == -0.5
    assert AgentEnd(stop="timeout").rule(LimitPolicy()) == "score"


def test_context_prefix_is_scrubbed_and_capped():
    end = classify_agent_end(upstream=_up("infra", "HTTP 503"), self_report="failed:agent_error", context=f"A2A task failed: echoed {KEY} " + "x" * 900)
    assert end.failure.reason.startswith("A2A task failed") and KEY not in end.failure.reason and len(end.failure.reason) <= 500
    end = classify_agent_end(upstream=_up(), error=RuntimeError(f"boom {KEY}"))
    assert KEY not in end.failure.reason and KEY not in end.failure.model_dump_json()


# ------------------------------------------------------------------ spend from the capture


def test_capture_cost():
    c = {"model": "m", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}}
    assert capture_cost([c], {"m": [1.0, 2.0]}, self_hosted=False) == (3.0, 0.0) and c["cost_usd"] == 3.0
    u = {"model": "hosted-x", "usage": {"prompt_tokens": 1_000_000, "completion_tokens": 0}}
    assert capture_cost([u], {}, self_hosted=False) == (0.0, UNPRICED_USD_PER_M[0]) and u["cost_uncertain_usd"] == UNPRICED_USD_PER_M[0]
    assert capture_cost([dict(u)], {"*": [7.0, 7.0]}, self_hosted=False) == (0.0, 7.0)  # prices["*"] is the fallback
    assert capture_cost([dict(u)], {}, self_hosted=True) == (0.0, 0.0)  # a self-hosted engine costs nothing
    assert capture_cost([dict(c)], {"m": [1.0, 2.0]}, self_hosted=True) == (3.0, 0.0)  # a configured price still applies
    a = {"model": "claude", "usage": {"input_tokens": 100_000, "cache_read_input_tokens": 900_000, "cache_creation_input_tokens": None, "output_tokens": 500_000}}
    assert capture_cost([a], {"claude": [3.0, 15.0]}, self_hosted=False) == (10.5, 0.0)  # 1M in at 3 + 0.5M out at 15
    failed = {"model": "m", "usage": {}, "status": 503}
    assert capture_cost([failed], {"m": [1.0, 2.0]}, self_hosted=False) == (0.0, 0.0) and "cost_usd" not in failed
    assert capture_cost([{"model": None, "usage": None}], {}, self_hosted=False) == (0.0, 0.0)
    both = capture_cost([dict(c), dict(u)], {"m": [1.0, 2.0]}, self_hosted=False)
    assert both == (3.0, UNPRICED_USD_PER_M[0])  # disjoint: the ledger settles their sum
    assert is_local_base("http://127.0.0.1:8000/v1") and is_local_base("http://gpu-box.local:8000/v1")
    assert not is_local_base("https://api.openai.com/v1") and not is_local_base(None)


def test_leaked_host_secrets(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-host-123456")
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    assert leaked_host_secrets({"OPENAI_API_KEY": "cap-x", "X": "sk-real-host-123456"}) == ["X"]
    assert leaked_host_secrets({"P": "/usr/bin:/bin"}) == []  # PATH is not secret-shaped
    monkeypatch.setenv("SOME_TOKEN", "cap-abcdefgh12345")
    assert leaked_host_secrets({"OPENAI_API_KEY": "cap-abcdefgh12345"}, allow={"cap-abcdefgh12345"}) == []  # the capture key
    host = {"ANTHROPIC_API_KEY": "sk-ant-host-999999", "SHORT_KEY": "abc"}
    assert leaked_host_secrets({"A": "sk-ant-host-999999", "B": "abc"}, environ=host) == ["A"]  # under 8 characters: not a credential


# ------------------------------------------------------------------ NORMALIZE (boundary fallback)


def test_normalize_harbor_setup_and_overflow_names():
    f = failure("AgentSetupTimeoutError", "seed")
    assert f.kind == "crucible:backend_error" and f.masks
    f = failure("ContextWindowExceededError", "agent")
    assert f.kind == "agent_timeout" and f.detail == "context_overflow" and not f.masks
    assert failure("OutputTokenExceededError", "agent").detail == "context_overflow"
    f = failure("ApiRateLimitError", "agent")
    assert f.who == "unknown" and f.detail == "unmapped:ApiRateLimitError"  # an evidence-free name never masks as infra


# ------------------------------------------------------------------ M10 lab audit R-1 review round 1


def test_a_verdict_crucibles_probe_does_not_confirm_is_unknown():
    """The agent's commands hold the capture key and pick what the upstream answers (a bogus model: 404; an unfundable
    max_tokens: 402; a request that outlives the proxy's timeout: a 502 row), so a non-policy verdict needs Crucible's own
    probe to agree; otherwise it is unknown blame (masked, counted by the 5% stop), never credits_exhausted."""
    for verdict in ("not_found", "credits", "infra", "auth"):
        for probe in ("ok", None):
            for kw in ({"error": NonZeroAgentExitCodeError("x")}, {"deadline": True}, {"self_report": "failed:agent_error"}):
                end = classify_agent_end(upstream=_up(verdict, "HTTP 4xx", probe=probe), **kw)
                assert end.failure.who == "unknown" and end.failure.detail == f"unconfirmed:{verdict}" and end.failure.masks, (verdict, probe, kw)
                assert end.failure.kind != "crucible:credits_exhausted" and "probe" in end.failure.reason
    # confirmed: the existing mapping
    assert classify_agent_end(upstream=_up("credits", "HTTP 402"), deadline=True).failure.kind == "crucible:credits_exhausted"


async def test_confirm_upstream_probes_the_first_model_only_when_it_would_blame_the_upstream(tmp_path):
    p = tmp_path / "s.jsonl"
    first = {"api": "openai", "digest": "a", "request": {"model": "m", "messages": []}, "status": 200, "response": {"choices": []}}
    forged = {"api": "openai", "digest": "b", "request": {"model": "bogus", "messages": []}, "status": 404, "response": {"error": "no such model"}}
    p.write_text("\n".join(json.dumps(r) for r in (first, forged)) + "\n")
    up = read_upstream_evidence(p)
    assert (up.api, up.model, up.verdict) == ("openai", "m", "not_found")  # the first request: made before the policy wrote anything
    asked: list[tuple[str, str]] = []

    async def probe(api, model):
        asked.append((api, model))
        return {"status": 200, "response": {"choices": [{"message": {"content": "OK"}}]}}

    got = await confirm_upstream(up, probe, up.model)
    assert asked == [("openai", "m")] and got.probe == "ok" and got.verdict == "not_found"
    assert (await confirm_upstream(up, None, "m")).probe is None and (await confirm_upstream(up, probe, None)).probe is None

    async def broken(api, model):
        raise RuntimeError("probe failed")

    assert (await confirm_upstream(up, broken, "m")).probe is None  # no answer is no confirmation
    healthy = UpstreamEvidence(calls=1, verdict="ok", detail="HTTP 200", model="m")
    assert await confirm_upstream(healthy, probe, "m") is healthy and len(asked) == 1  # nothing to confirm: never probed


async def test_the_proxy_probe_is_unlogged_and_row_shaped(tmp_path):
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    row = await proxy.probe("openai", "m")
    assert hits == ["/v1/chat/completions"] and verdict_of(row)[0] == "ok"
    assert verdict_of(await proxy.probe("anthropic", "c"))[0] == "ok" and hits[-1] == "/v1/messages"
    assert not list((tmp_path / "cap").glob("*.jsonl"))  # outside every session: nothing logged
    down = _proxy(tmp_path / "down", lambda r: httpx.Response(503, json={"error": "overloaded"}))
    assert verdict_of(await down.probe("openai", "m"))[0] == "infra"
    assert await CaptureProxy(tmp_path / "none").probe("openai", "m") == {}


def test_an_installed_agent_that_never_reached_its_model_is_not_the_policys():
    """An installed agent (agent_is_harness) that ends abnormally or at the deadline before the proxy recorded any
    exchange never let the policy answer: unknown (P16). An A2A agent is the policy: unchanged."""
    none = UpstreamEvidence()
    for kw in ({"error": NonZeroAgentExitCodeError("Command failed (exit 127): ...")}, {"error": ApiRateLimitError("rate limit")},
               {"error": ContextWindowExceededError("x")}, {"error": RuntimeError("APIConnectionError")}, {"deadline": True}):
        end = classify_agent_end(upstream=none, agent_is_harness=True, **kw)
        assert end.failure.who == "unknown" and end.failure.masks and end.failure.detail == "no_upstream_call", kw
        assert "never answered" in end.failure.reason
    end = classify_agent_end(upstream=_up(), error=ApiRateLimitError("rate limit"), agent_is_harness=True)
    assert end.failure.who == "policy" and not end.failure.masks
    assert classify_agent_end(upstream=_up(), deadline=True, agent_is_harness=True).stop == "timeout"
    assert classify_agent_end(upstream=none, container_lost="removed", agent_is_harness=True).failure.kind == "crucible:env_container_lost"
    assert classify_agent_end(upstream=none, exit_status="Submitted", agent_is_harness=True).stop == "agent_done"
    assert classify_agent_end(upstream=none, self_report="failed:agent_error").failure.who == "policy"  # A2A
    assert classify_agent_end(upstream=none, deadline=True).stop == "timeout"


def test_call_cap_yields_to_upstream_failure_evidence():
    """The proxy's cap counts failed calls, so a cap an outage burned is the provider's, not max_steps (confirmed by
    Crucible's probe, else unknown)."""
    burst = UpstreamEvidence(calls=4, failed=1, limit_stops=1, verdict="infra", detail="HTTP 503", model="m", probe="infra")
    end = classify_agent_end(upstream=burst, exit_status="RepeatedFormatError")
    assert end.failure.kind == "provider_unavailable" and end.failure.masks and end.failure.who == "infra"
    spent = UpstreamEvidence(calls=4, failed=2, limit_stops=1, verdict="ok", detail="HTTP 200", model="m", probe="infra")
    assert classify_agent_end(upstream=spent).failure.detail == "cap_spent_on_upstream_failures"
    unconfirmed = UpstreamEvidence(calls=4, failed=2, limit_stops=1, verdict="ok", detail="HTTP 200", model="m")
    end = classify_agent_end(upstream=unconfirmed)
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:cap_spent_on_upstream_failures"
    healthy = UpstreamEvidence(calls=4, failed=1, limit_stops=1, verdict="ok", detail="HTTP 200")
    end = classify_agent_end(upstream=healthy)
    assert end.stop == "max_steps" and end.failure is None and end.evidenced
    credits = UpstreamEvidence(calls=2, limit_stops=1, verdict="credits", detail="HTTP 402", probe="credits")
    assert classify_agent_end(upstream=credits).failure.kind == "crucible:credits_exhausted"


async def test_confirm_upstream_probes_a_cap_spent_on_failures():
    asked: list[str] = []

    async def probe(api, model):
        asked.append(model)
        return {"status": 503, "response": {"error": "overloaded"}}

    spent = UpstreamEvidence(calls=4, failed=2, limit_stops=1, verdict="ok", detail="HTTP 200", model="m")
    assert (await confirm_upstream(spent, probe, "m")).probe == "infra" and asked == ["m"]


async def test_a_flood_past_the_cap_writes_bounded_rows_and_keeps_the_count(tmp_path):
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    key = proxy.register("s", max_calls=1)
    n = LIMIT_STOP_ROWS + 25
    replies = [await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", i)) for i in range(n + 1)]
    assert len(hits) == 1 and all(LIMIT_STOP_TEXT in r.text for r in replies[1:])  # every over-cap call still gets its stop
    assert sum(1 for r in _rows(tmp_path) if r.get("kind") == LIMIT_STOP_KIND) == LIMIT_STOP_ROWS  # the flood writes no more rows
    proxy.unregister(key)
    proxy.unregister(key)  # idempotent: the coalesced row is written once
    stops = [r for r in _rows(tmp_path) if r.get("kind") == LIMIT_STOP_KIND]
    assert len(stops) == LIMIT_STOP_ROWS + 1 and stops[-1]["count"] == n - LIMIT_STOP_ROWS and "s" not in proxy.limit_stops
    p = tmp_path / "cap" / "s.jsonl"
    assert read_upstream_evidence(p).limit_stops == n and read_upstream_evidence(p).calls == 1
    raw = CaptureLogConnector().fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(p)}))
    assert raw.payload["limit_stops"] == n and len(raw.payload["exchanges"]) == 1


PRICED = {"m": [1_000_000.0, 1_000_000.0]}  # 1 USD per token in and out


def test_capture_spend_prices_every_forwarded_row(tmp_path):
    """Spend is what the upstream billed: every forwarded row, though fetch collapses retries into one exchange; a
    usage-less or cut call is booked as an uncertain bound; a refused call costs nothing."""
    req = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}
    ok = {"api": "openai", "digest": "d", "request": req, "status": 200, "headers": {},
          "response": {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}}}
    stream_req = {"model": "m", "max_tokens": 10, "stream": True, "messages": [{"role": "user", "content": "go"}]}
    no_usage = {"api": "openai", "digest": "e", "request": stream_req, "status": 200, "stream": True, "complete": True,
                "response": {"choices": [{"message": {"content": "x"}}], "usage": None}}
    ant_req = {"model": "m", "max_tokens": 20, "messages": [{"role": "user", "content": "go"}]}
    cut = {"api": "anthropic", "digest": "f", "request": ant_req, "status": 200, "stream": True, "complete": False, "error": "stream_drop: cut",
           "response": {"content": [], "usage": {"input_tokens": 3, "output_tokens": 1}}}
    rows = [ok, dict(ok), {**ok, "headers": {"x-stainless-retry-count": "1"}},
            {"api": "openai", "digest": "g", "request": req, "status": 503, "response": {"error": "down"}},
            {"api": "openai", "digest": "h", "request": req, "status": 502, "complete": False, "error": "ConnectError: refused"},
            {"kind": LIMIT_STOP_KIND, "calls": 3, "max_calls": 3}, no_usage, cut]
    p = tmp_path / "s.jsonl"
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n" + '{"digest": "c", "requ')  # a torn last line
    exchanges = CaptureLogConnector().fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(p)})).payload["exchanges"]
    assert sum(1 for x in exchanges if x.get("digest") == "d") == 1  # the connector collapses the three; the bill does not
    bound = lambda r: (len(json.dumps(r)) // 4) * 1.0 + r["max_tokens"] * 1.0  # noqa: E731 -- PRICED: 1 USD per token
    certain, uncertain = capture_spend(p, PRICED, self_hosted=False)
    assert certain == pytest.approx(15.0) and uncertain == pytest.approx(bound(stream_req) + bound(ant_req))
    assert capture_spend(p, PRICED, self_hosted=True) == (pytest.approx(15.0), pytest.approx(bound(stream_req) + bound(ant_req)))  # priced
    assert capture_spend(tmp_path / "missing.jsonl", PRICED, self_hosted=False) == (0.0, 0.0)
    # self-hosted per dialect: unpriced openai rows are free, unpriced anthropic rows are booked at the fallback
    a = {"api": "anthropic", "digest": "i", "request": {"model": "claude"}, "status": 200,
         "response": {"content": [], "usage": {"input_tokens": 1_000_000, "output_tokens": 0}}}
    o = {"api": "openai", "digest": "j", "request": {"model": "local"}, "status": 200, "response": {"choices": [], "usage": {"prompt_tokens": 1_000_000}}}
    q = tmp_path / "t.jsonl"
    q.write_text(json.dumps(a) + "\n" + json.dumps(o) + "\n")
    assert capture_spend(q, {}, self_hosted={"openai", "responses"}) == (0.0, pytest.approx(UNPRICED_USD_PER_M[0]))


def test_self_hosted_apis():
    local, remote = Upstream("http://127.0.0.1:8000/v1"), Upstream("https://api.anthropic.com/v1", "k")
    assert self_hosted_apis({"openai": remote}, self_hosted=True) == frozenset({"openai", "anthropic", "responses"})
    assert self_hosted_apis({"openai": local}, self_hosted=False) == frozenset()
    assert self_hosted_apis({"openai": local, "anthropic": remote}) == frozenset({"openai", "responses"})
    assert self_hosted_apis({"openai": Upstream("https://openrouter.ai/api/v1"), "anthropic": remote}, policy_version=3) == frozenset({"openai", "responses"})
    assert self_hosted_apis({"anthropic": Upstream("http://gpu.local/v1")}) == frozenset({"anthropic"})


def test_host_secret_names():
    env = {"CURSOR_API_KEY": "crsr-123456789", "CLAUDE_CODE_OAUTH_TOKEN": "sk-ant-oat01-xyz", "SHORT_KEY": "abc", "PATH": "/usr/bin:/bin", "HOME": "/root"}
    assert host_secret_names(env) == ["CLAUDE_CODE_OAUTH_TOKEN", "CURSOR_API_KEY"]
    assert leaked_host_secrets({"X": "crsr-123456789"}, environ=env) == ["X"]  # one rule for both


# ------------------------------------------------------------------ M10 lab audit R-1 review round 2


async def test_confirm_upstream_probes_the_configured_dialect():
    """An A2A agent is the policy and writes the session's first request too, so its route picked the upstream Crucible
    probed (a policy-made Anthropic 404 was confirmed by a probe of Anthropic). ``api`` is the configured dialect; without
    it the first request's dialect is probed (Harbor: made before the policy wrote anything)."""
    asked: list[tuple[str, str]] = []

    async def probe(api, model):
        asked.append((api, model))
        return {"status": 200, "response": {"choices": [{"message": {"content": "OK"}}]}}

    up = UpstreamEvidence(calls=2, last_status=404, verdict="not_found", detail="HTTP 404", api="anthropic", model="whatever")
    got = await confirm_upstream(up, probe, "gpt-x", api="openai")
    assert asked == [("openai", "gpt-x")] and got.probe == "ok" and got.probed == "openai:gpt-x"
    end = classify_agent_end(upstream=got, self_report="failed:agent_error")
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:not_found" and "openai:gpt-x" in end.failure.reason
    assert (await confirm_upstream(up, probe, "gpt-x")).probed == "anthropic:gpt-x" and asked[-1] == ("anthropic", "gpt-x")


def _iso(s: float) -> str:
    from datetime import UTC, datetime, timedelta

    return (datetime(2026, 10, 15, 12, 0, 0, tzinfo=UTC) + timedelta(seconds=s)).isoformat().replace("+00:00", "Z")


def _timed_log(p: Path, n: int, spacing_s: float, last_status: int = 429, retry_after: str | None = None) -> Path:
    """``n`` forwarded rows ``spacing_s`` apart (the proxy's own request_time and time), the last answered ``last_status``."""
    rows = []
    for i in range(n):
        last = i == n - 1
        row = {"api": "openai", "digest": f"d{i}", "request": {"model": "m", "messages": [{"role": "user", "content": f"turn {i}"}]},
               "request_time": _iso(i * spacing_s), "time": _iso(i * spacing_s + 0.5), "status": last_status if last else 200,
               "response": {"error": {"message": "rate limited"}} if last else {"choices": [{"message": {"content": "ok"}}]}}
        if last and retry_after is not None:
            row["retry_after"] = retry_after
        rows.append(row)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


async def test_a_session_under_its_own_load_never_confirms_an_upstream_failure(tmp_path):
    """The probe shares the session's upstream key, rate-limit bucket and queue: a policy that floods through the capture
    key and then fails had its 429 confirmed (``provider_unavailable``: masked and re-sampled). Past ``SELF_LOAD_CALLS``
    calls in the window the infra verdict is not probed (unknown, counted by the 5% stop); a sequential session is."""
    from crucible.runtime import agent_evidence

    asked: list[str] = []

    async def probe(api, model):
        asked.append(model)
        return {"status": 429, "response": {"error": {"message": "rate limited"}}}

    flood = read_upstream_evidence(_timed_log(tmp_path / "flood.jsonl", agent_evidence.SELF_LOAD_CALLS + 10, 0.5))
    assert flood.verdict == "infra" and flood.recent > agent_evidence.SELF_LOAD_CALLS
    got = await confirm_upstream(flood, probe, flood.model)
    assert asked == [] and got.probe is None
    end = classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True)
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:infra" and end.failure.masks and "may be its own" in end.failure.reason

    calm = read_upstream_evidence(_timed_log(tmp_path / "calm.jsonl", 2, 20.0))
    assert calm.recent == 2
    got = await confirm_upstream(calm, probe, calm.model)
    assert asked == ["m"] and got.probe == "infra"
    end = classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True)
    assert end.failure.kind == "provider_unavailable"

    spread = read_upstream_evidence(_timed_log(tmp_path / "spread.jsonl", agent_evidence.SELF_LOAD_CALLS + 10, 10.0))
    assert spread.recent <= agent_evidence.SELF_LOAD_CALLS  # many calls, but not in the last minute: probed


async def test_a_429_is_probed_after_its_retry_after(tmp_path, monkeypatch):
    from crucible.runtime import agent_evidence

    up = read_upstream_evidence(_timed_log(tmp_path / "s.jsonl", 1, 1.0, retry_after="120"))
    assert up.retry_after == 120.0 and up.last_status == 429
    assert read_upstream_evidence(_timed_log(tmp_path / "t.jsonl", 1, 1.0, retry_after="Wed, 21 Oct 2026 07:28:00 GMT")).retry_after is None
    monkeypatch.setattr(agent_evidence, "RETRY_AFTER_CAP_S", 0.05)  # waited, capped
    order: list[str] = []

    async def probe(api, model):
        order.append("probe")
        return {"status": 200, "response": {"choices": [{"message": {"content": "OK"}}]}}

    t0 = asyncio.get_running_loop().time()
    got = await confirm_upstream(up, probe, "m")
    assert got.probe == "ok" and order == ["probe"] and asyncio.get_running_loop().time() - t0 >= 0.04


async def test_the_proxy_records_a_refusals_retry_after(tmp_path):
    proxy = _proxy(tmp_path, lambda r: httpx.Response(429, headers={"retry-after": "7"}, json={"error": {"message": "slow down"}}))
    key = proxy.register("s")
    assert (await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions"))).status_code == 429
    [row] = _rows(tmp_path)
    assert row["retry_after"] == "7" and read_upstream_evidence(tmp_path / "cap" / "s.jsonl").retry_after == 7.0


@pytest.mark.parametrize("cap", [1, 0])  # 0: no cap, as the A2A harness registers
async def test_a_key_revoked_while_the_body_is_read_is_refused(tmp_path, cap):
    """The proxy resolved the session from the key, then awaited the body: a request whose headers arrived before the
    key was revoked (the agent phase ended) and whose body came after was forwarded uncapped (``unregister`` drops the
    cap) and recorded, maybe after the evidence read. A session registered again does not take it either."""
    hits: list[str] = []
    proxy = _proxy(tmp_path, _json_upstream(hits))
    key = proxy.register("s", max_calls=cap)
    assert (await _post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions", 0))).status_code == 200
    body = json.dumps(_body("/v1/chat/completions", 1)).encode()

    async def held():  # runs inside the app's request.body(), after the key was looked up
        proxy.unregister(key)
        proxy.register("s", max_calls=cap)
        yield body

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as c:
        r = await c.post("/v1/chat/completions", content=held(), headers={"authorization": f"Bearer {key}", "content-type": "application/json"})
    assert r.status_code == 401 and hits == ["/v1/chat/completions"]
    assert [x.get("digest") is not None for x in _rows(tmp_path)] == [True]  # one exchange: the held request was never recorded
    assert proxy.forwarded["s"] == 0  # nor counted against the new key


# ------------------------------------------------------------------ M10 lab audit R-1 review round 3


def _failing_row(i: int, start: float, end: float, *, last: bool, status: int = 429, error: str | None = None) -> dict:
    row = {"api": "openai", "digest": f"d{i}", "request": {"model": "m", "messages": [{"role": "user", "content": f"turn {i}"}]},
           "request_time": _iso(start), "time": _iso(end), "status": status if last else 200}
    if last and error is not None:  # the proxy's own record of an httpx error: no response
        row.update(complete=False, error=error)
    else:
        row["response"] = {"error": {"message": "rate limited"}} if last else {"choices": [{"message": {"content": "ok"}}]}
    return row


def _write(p: Path, rows: list[dict]) -> Path:
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return p


async def test_calls_that_ran_into_the_window_count_as_the_sessions_load(tmp_path):
    """The guard counted calls that started in the last minute only: a flood of long generations started earlier, each
    recorded when it ends, still loaded the upstream the probe shares, and was invisible (the probe then queued behind it
    and its timeout "confirmed" the infra verdict). A call counts when it ran into the window, wherever it started."""
    from crucible.runtime import agent_evidence

    n = agent_evidence.SELF_LOAD_CALLS + 5
    # every call started within 4 s of the episode's start and ended 300 s later or more: none started in the last minute
    rows = [_failing_row(i, i * 0.1, 300 + i, last=i == n - 1) for i in range(n)]
    up = read_upstream_evidence(_write(tmp_path / "long.jsonl", rows))
    assert up.verdict == "infra" and up.recent == n
    asked: list[str] = []

    async def probe(api, model):
        asked.append(model)
        return {"status": 429, "response": {"error": {"message": "rate limited"}}}

    got = await confirm_upstream(up, probe, up.model)
    assert asked == [] and got.probe is None
    end = classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True)
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:infra" and "may be its own" in end.failure.reason


async def test_a_session_with_calls_still_in_flight_is_never_probed():
    """A forwarded call whose client the stop killed still runs upstream (the proxy's handler is not cancelled) and has no
    row yet: the harness's ``drain`` waits for it, and what is still in flight keeps Crucible's probe off (it would queue
    behind the session's own calls), a refused connection included."""
    up = UpstreamEvidence(calls=3, last_status=503, verdict="infra", detail="HTTP 503", api="openai", model="m", recent=1, inflight=2)
    asked: list[str] = []

    async def probe(api, model):
        asked.append(model)
        return {"status": 503, "response": {"error": {"message": "overloaded"}}}

    got = await confirm_upstream(up, probe, "m")
    assert asked == [] and got.probe is None
    end = classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True)
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:infra" and "still in flight" in end.failure.reason
    refused = dataclasses.replace(up, last_status=502, last_error="ConnectError: [Errno 111] Connection refused",
                                  detail="connect: ConnectError: [Errno 111] Connection refused")
    assert (await confirm_upstream(refused, probe, "m")).probe is None and asked == []
    assert (await confirm_upstream(dataclasses.replace(up, inflight=0), probe, "m")).probe == "infra" and asked == ["m"]


async def test_a_refused_connection_is_probed_whatever_the_sessions_rate(tmp_path):
    """An honest agent against a fast self-hosted engine makes more than ``SELF_LOAD_CALLS`` calls a minute; when the
    engine restarts, the proxy's own record of the refused connection (httpx ConnectError) was never probed and the outage
    stayed unknown (the 5% stop halted the run). No bucket or queue the session fills refuses a connection, so it is
    probed; a timeout (which a flood can cause) is not."""
    from crucible.runtime import agent_evidence

    n = agent_evidence.SELF_LOAD_CALLS + 10
    refused = [_failing_row(i, i * 1.5, i * 1.5 + 1.0, last=i == n - 1, status=502, error="ConnectError: [Errno 111] Connection refused")
               for i in range(n)]
    up = read_upstream_evidence(_write(tmp_path / "fast.jsonl", refused))
    assert up.recent > agent_evidence.SELF_LOAD_CALLS and up.detail.startswith("connect: ConnectError")
    asked: list[str] = []

    async def probe(api, model):
        asked.append(model)
        return {"status": 502, "complete": False, "error": "ConnectError: refused"}

    got = await confirm_upstream(up, probe, up.model)
    assert asked == ["m"] and got.probe == "infra"
    assert classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True).failure.kind == "provider_unavailable"

    timed_out = [_failing_row(i, i * 1.5, i * 1.5 + 1.0, last=i == n - 1, status=502, error="ReadTimeout: timed out") for i in range(n)]
    up = read_upstream_evidence(_write(tmp_path / "slow.jsonl", timed_out))
    got = await confirm_upstream(up, probe, up.model)
    assert asked == ["m"] and got.probe is None
    end = classify_agent_end(upstream=got, error=NonZeroAgentExitCodeError("exit 1"), agent_is_harness=True)
    assert end.failure.who == "unknown" and end.failure.detail == "unconfirmed:infra"


async def test_calls_still_in_flight_after_the_key_is_revoked_are_counted_and_drained(tmp_path):
    """``unregister`` revokes the key, but a forwarded call already sent keeps running upstream and leaves no row until
    it ends: the proxy counts it in flight past the revocation, ``drain`` waits for it (returning what is left), and its
    row is written and its count cleared when it ends. A stream is in flight until its relay records it."""
    gate = asyncio.Event()

    async def held(request: httpx.Request) -> httpx.Response:
        await gate.wait()
        return httpx.Response(200, json={"id": "r", "model": "m", "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"},
                                                                                "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 2}})

    proxy = _proxy(tmp_path, held)
    key = proxy.register("s")
    call = asyncio.create_task(_post(proxy, key, "/v1/chat/completions", _body("/v1/chat/completions")))
    for _ in range(500):
        if proxy.inflight.get("s"):
            break
        await asyncio.sleep(0.01)
    proxy.unregister(key)
    assert proxy.inflight["s"] == 1 and await proxy.drain("s", 0.05) == 1
    assert not (tmp_path / "cap" / "s.jsonl").exists() or _rows(tmp_path) == []  # no row while it runs
    gate.set()
    assert await proxy.drain("s", 5.0) == 0 and "s" not in proxy.inflight
    assert (await call).status_code == 200 and len(_rows(tmp_path)) == 1
    assert await proxy.drain("never-registered", 0.05) == 0

    streamed = _proxy(tmp_path / "stream", _streaming(_sse([_chunk("hi", "stop"), "[DONE]"])))
    key = streamed.register("t")
    r = await _post(streamed, key, "/v1/chat/completions", _body("/v1/chat/completions", stream=True))
    assert r.status_code == 200 and "t" not in streamed.inflight and _rows(tmp_path / "stream", "t")[0]["complete"] is True
