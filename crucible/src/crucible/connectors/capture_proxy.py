"""``capture_proxy``: a recording proxy in front of live agents (OpenAI chat completions and Anthropic messages),
and the connector that turns its capture logs into traces (DESIGN §5.1; PATTERN: verifiers
``interception/server.py``, NeMo-Gym ``token_id_capture`` and ``anthropic_converter``, OpenEnv RFC 006).

Point an agent's SDK at the proxy (``OPENAI_BASE_URL=http://host:port/v1`` or ``ANTHROPIC_BASE_URL=http://host:port``)
with a per-session key from ``register()`` as its API key: the key identifies the session (verifiers' capability
idea), and the real upstream key stays on the proxy, never in the agent's environment. Each
``POST /v1/chat/completions`` or ``POST /v1/messages`` is forwarded upstream unchanged and appended to
``<capture_dir>/<session>.jsonl``. The record keeps the request (messages, tools, parameters), the response
(assembled from the SSE events for streaming calls), usage, logprobs when the upstream returns them, and latency.
Every record carries a digest of its request body. A client retry (the same body again, e.g. after a timeout) is
recorded too; the connector keeps only the last attempt, which is the response the agent actually saw. Nothing
about the agent changes.

``CaptureLogConnector`` turns a session log into a Trace. Each request contributes the messages that are new since
the previous request (user turns, tool results); each response contributes the agent's message and tool calls. A
request that does not extend the previous one (the agent reset or forked its context) adds a ``context_reset``
event.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import secrets
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.connectors.agent_events import REJECTED, annotate_probes, user_type, writes_of
from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # becomes a file name: no separators, no leading dot
MAX_TEXT = 4000


@dataclass
class Upstream:
    base: str  # OpenAI: ".../v1"; Anthropic: "https://api.anthropic.com/v1"
    key: str | None = None


class CaptureProxy:
    """``transport`` (an httpx transport) is for tests: the proxy can be exercised against a fake upstream."""

    def __init__(self, capture_dir: Path, *, openai: Upstream | None = None, anthropic: Upstream | None = None,
                 timeout_s: float = 600, transport: Any = None):
        self.upstreams = {k: v for k, v in (("openai", openai), ("anthropic", anthropic)) if v is not None}
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_s = timeout_s
        self.transport = transport
        self.sessions: dict[str, str] = {}  # key -> session id

    def register(self, session: str) -> str:
        if not SESSION.fullmatch(session):
            raise ValueError(f"session id must match {SESSION.pattern}: {session!r}")
        key = "cap-" + secrets.token_urlsafe(18)
        self.sessions[key] = session
        return key

    def _record(self, session: str, entry: dict[str, Any]) -> None:
        with (self.capture_dir / f"{session}.jsonl").open("a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def app(self):
        import httpx
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        from starlette.routing import Route

        client = httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport)
        proxy = self

        def error(api: str, status: int, message: str):
            if api == "anthropic":
                return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": message}}, status_code=status)
            return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)

        async def forward(request: Request, api: str, path: str):
            up = proxy.upstreams.get(api)
            if up is None:
                return error(api, 404, f"no {api} upstream configured")
            presented = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            session = proxy.sessions.get(presented)
            if not session:  # only registered keys: the key is the capability
                return error(api, 401, "unknown session key")
            raw = await request.body()
            try:
                body = json.loads(raw)
            except ValueError:
                return error(api, 400, "request body is not JSON")
            headers = {"content-type": "application/json"}
            if api == "anthropic":
                headers["anthropic-version"] = request.headers.get("anthropic-version", "2023-06-01")
                if request.headers.get("anthropic-beta"):
                    headers["anthropic-beta"] = request.headers["anthropic-beta"]
                if up.key:
                    headers["x-api-key"] = up.key
            elif up.key:
                headers["authorization"] = f"Bearer {up.key}"
            entry = {"api": api, "digest": hashlib.blake2b(raw, digest_size=16).hexdigest(), "request": body}
            t0 = time.monotonic()
            url = f"{up.base.rstrip('/')}/{path}"
            if body.get("stream"):
                async def relay() -> AsyncIterator[bytes]:
                    events: list[dict[str, Any]] = []
                    status = None
                    async with client.stream("POST", url, content=raw, headers=headers) as r:
                        status = r.status_code
                        async for line in r.aiter_lines():
                            if line.startswith("data:") and line.strip() != "data: [DONE]":
                                with contextlib.suppress(ValueError):
                                    events.append(json.loads(line[5:].strip()))
                            yield (line + "\n").encode()
                    response = assemble_anthropic(events) if api == "anthropic" else assemble_openai(events)
                    proxy._record(session, {**entry, "time": _now(), "latency_s": round(time.monotonic() - t0, 3), "status": status,
                                            "stream": True, "response": response})
                return StreamingResponse(relay(), media_type="text/event-stream")
            r = await client.post(url, content=raw, headers=headers)
            out = r.json() if r.headers.get("content-type", "").startswith("application/json") else {"error": r.text[:2000]}
            proxy._record(session, {**entry, "time": _now(), "latency_s": round(time.monotonic() - t0, 3), "status": r.status_code, "response": out})
            return JSONResponse(out, status_code=r.status_code)

        async def chat(request: Request):
            return await forward(request, "openai", "chat/completions")

        async def messages(request: Request):
            return await forward(request, "anthropic", "messages")

        async def health(_request: Request):
            return JSONResponse({"ok": True, "upstreams": sorted(proxy.upstreams), "sessions": len(set(proxy.sessions.values()))})

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            yield
            await client.aclose()

        return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/messages", messages, methods=["POST"]),
                                 Route("/health", health)], lifespan=lifespan)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def assemble_openai(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """OpenAI SSE chunks -> one chat.completion (content, tool calls, finish reason, usage)."""
    content: list[str] = []
    calls: dict[int, dict[str, Any]] = {}
    finish, usage, model, rid = None, None, None, None
    for ch in chunks:
        model, rid = ch.get("model", model), ch.get("id", rid)
        usage = ch.get("usage") or usage
        for c in ch.get("choices", []):
            d = c.get("delta") or {}
            if d.get("content"):
                content.append(d["content"])
            for tc in d.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
            finish = c.get("finish_reason") or finish
    msg: dict[str, Any] = {"role": "assistant", "content": "".join(content) or None}
    if calls:
        msg["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"id": rid, "model": model, "object": "chat.completion", "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage}


def assemble_anthropic(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Anthropic message SSE events (message_start, content_block_start/delta/stop, message_delta) -> one message."""
    msg: dict[str, Any] = {"type": "message", "role": "assistant", "content": [], "usage": {}}
    blocks: dict[int, dict[str, Any]] = {}
    partial: dict[int, str] = {}
    for ev in events:
        t = ev.get("type")
        if t == "message_start":
            msg.update({k: v for k, v in (ev.get("message") or {}).items() if k != "content"})
        elif t == "content_block_start":
            blocks[ev["index"]] = dict(ev.get("content_block") or {})
        elif t == "content_block_delta":
            b, d = blocks.setdefault(ev["index"], {"type": "text", "text": ""}), ev.get("delta") or {}
            if d.get("type") == "text_delta":
                b["text"] = b.get("text", "") + d.get("text", "")
            elif d.get("type") == "thinking_delta":
                b["thinking"] = b.get("thinking", "") + d.get("thinking", "")
            elif d.get("type") == "input_json_delta":
                partial[ev["index"]] = partial.get(ev["index"], "") + d.get("partial_json", "")
        elif t == "message_delta":
            msg.update({k: v for k, v in (ev.get("delta") or {}).items() if v is not None})
            msg["usage"] = {**(msg.get("usage") or {}), **(ev.get("usage") or {})}
        elif t == "error":
            msg["error"] = ev.get("error")
    for i, s in partial.items():
        try:
            blocks[i]["input"] = json.loads(s) if s else {}
        except ValueError:
            blocks[i]["input"] = {"_unparsed": s}
    msg["content"] = [blocks[i] for i in sorted(blocks)]
    return msg


def _text(c: Any) -> str:
    if isinstance(c, list):
        return "\n".join(p.get("text", "") for p in c if isinstance(p, dict) and p.get("type", "text") == "text")
    return c or ""


def _args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


class CaptureLogConnector:
    """scope: {"dir": capture_dir}: one trace per session log."""

    name = "capture_proxy"

    def __init__(self, family: str | None = None, privacy: str = "internal"):
        self.family, self.privacy = family, privacy

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        for p in sorted(Path(scope["dir"]).glob("*.jsonl")):
            yield SourceRef(connector=self.name, key=p.stem, meta={"path": str(p)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        exchanges: list[dict[str, Any]] = []
        for x in Path(ref.meta["path"]).read_text().splitlines():
            if not x.strip():
                continue
            ex = json.loads(x)
            if exchanges and ex.get("digest") and exchanges[-1].get("digest") == ex["digest"]:
                exchanges[-1] = ex  # a client retry: the agent saw this attempt's response, not the earlier one
            else:
                exchanges.append(ex)
        return RawRecord(ref=ref, payload={"exchanges": exchanges}, privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        events: list[TraceEvent] = []
        call_seq: dict[str, int] = {}
        models: set[str] = set()
        prev: list[dict[str, Any]] = []
        system: str | None = None

        def add(time_: str | None, **kw: Any) -> int:
            events.append(TraceEvent(seq=len(events), time=time_, **kw))
            return len(events) - 1

        def user(t: str | None, text: str) -> None:
            if text.strip():
                add(t, kind="message", actor="user", type=user_type(text), text=text[:MAX_TEXT])

        def result(t: str | None, call_id: str | None, text: str, error: bool = False) -> None:
            data = {"call_id": call_id, "chars": len(text), **({"error": True} if error else {}), **({"rejected": True} if REJECTED.search(text) else {})}
            add(t, kind="observation", actor="system", type="tool_result", text=text[:MAX_TEXT], reply_to=call_seq.get(call_id or ""), data=data)

        def call(t: str | None, call_id: str | None, name: str, args: Any) -> None:
            arg_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
            w = writes_of(name, args)
            call_seq[call_id or ""] = add(t, kind="action", actor="agent", type="tool_call", text=f"{name}({arg_text[:200]})",
                                          data={"tool": name, "call_id": call_id, "arguments": arg_text[:MAX_TEXT], **({"writes": w} if w else {})})

        for ex in raw.payload["exchanges"]:
            t, api, req = ex.get("time"), ex.get("api", "openai"), ex["request"]
            msgs = req.get("messages", [])
            extends = bool(prev) and msgs[: len(prev)] == prev
            if prev and not extends:
                add(t, kind="observation", actor="system", type="context_reset", text=f"request does not extend the previous one ({len(msgs)} messages)")
            if api == "anthropic" and req.get("system") and _text(req["system"]) != system:
                system = _text(req["system"])
                add(t, kind="message", actor="system", type="system_message", text=system[:MAX_TEXT])
            for m in msgs[len(prev):] if extends else msgs:
                role, content = m.get("role"), m.get("content")
                if role == "assistant":
                    continue  # recorded from the response that produced it
                if api == "anthropic":
                    if isinstance(content, str):
                        user(t, content)
                        continue
                    texts = []
                    for b in content or []:
                        if b.get("type") == "tool_result":
                            result(t, b.get("tool_use_id"), _text(b.get("content")), bool(b.get("is_error")))
                        elif b.get("type") == "text":
                            texts.append(b.get("text", ""))
                    user(t, "\n".join(texts))
                elif role == "tool":
                    result(t, m.get("tool_call_id"), _text(content))
                elif role in ("system", "developer"):
                    add(t, kind="message", actor="system", type="system_message", text=_text(content)[:MAX_TEXT])
                else:
                    user(t, _text(content))
            prev = msgs
            resp = ex.get("response") or {}
            model = resp.get("model") or req.get("model")
            if model:
                models.add(model)
            data: dict[str, Any] = {"model": model, "usage": resp.get("usage"), "latency_s": ex.get("latency_s"), "status": ex.get("status")}
            if resp.get("error") or (ex.get("status") or 200) >= 400:
                data["error"] = resp.get("error") or f"HTTP {ex.get('status')}"
            if api == "anthropic":
                blocks = resp.get("content") or []
                thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
                data.update({"finish": resp.get("stop_reason"), **({"reasoning": thinking[:MAX_TEXT]} if thinking else {})})
                add(t, kind="message", actor="agent", type="llm_call", text=_text(blocks)[:MAX_TEXT], data=data)
                for b in blocks:
                    if b.get("type") == "tool_use":
                        call(t, b.get("id"), b.get("name", ""), b.get("input") or {})
            else:
                choice = (resp.get("choices") or [{}])[0]
                m = choice.get("message") or {}
                data.update({"finish": choice.get("finish_reason"), **({"logprobs": True} if choice.get("logprobs") else {}),
                             **({"reasoning": str(m["reasoning_content"])[:MAX_TEXT]} if m.get("reasoning_content") else {})})
                add(t, kind="message", actor="agent", type="llm_call", text=_text(m.get("content"))[:MAX_TEXT], data=data)
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    call(t, tc.get("id"), fn.get("name") or "", _args(fn.get("arguments") or ""))
        probes = annotate_probes(events)
        actors = [TraceActor(id="user", role="user", kind="human"),
                  TraceActor(id="agent", role="agent", kind="agent", handle=",".join(sorted(models)) or None),
                  TraceActor(id="system", role="system", kind="system")]
        return Trace(trace_id=f"capture:{raw.ref.key}", family=self.family,
                     source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=actors, events=events, outcome=TraceOutcome(details={"probes": probes} if probes["attempted"] else {}))
