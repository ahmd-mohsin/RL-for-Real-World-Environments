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

``CaptureLogConnector`` turns a session log into a Trace. Each request contributes the messages past the longest
prefix it shares with an earlier request (user turns, tool results); each response contributes the agent's message and
tool calls. A request that shares no prefix is a side call (a title request, a subagent: ``data.thread`` > 0) when
the main conversation continues after it, else a ``context_reset`` (compaction, a fresh start).

M3 audit (2026-10-12):
- Proxy: an upstream error keeps its status and ``retry-after`` (a streamed 429 reached the agent as HTTP 200, so its
  SDK never retried). Every exchange is recorded in a ``finally``: a stream that fails or that the user interrupts is
  kept with ``complete: false``. Logs are ASCII-safe (a lone surrogate lost the record after the upstream billed it)
  and owner-only. The query string is forwarded. ``count_tokens``, ``/v1/models`` and ``/v1/responses`` pass through,
  and Responses calls are recorded. One live key per session. Bodies are capped. Multi-line SSE data is joined.
- Assembly: choices are kept apart (n > 1 concatenated them); streamed reasoning is kept; tool names are set, not
  appended ("bashbash"); calls without an index get their own slot; a mid-stream error chunk is kept.
- Connector: cache markers and re-rendered system prompts are ignored when matching prefixes (each caused a false
  ``context_reset`` and re-emitted the whole history); harness text (``<system-reminder>``, command output) is a
  ``harness_message``, not the human; images leave a marker; server tools (web search) are tool calls; retries
  collapse only when marked (SDK ``x-stainless-retry-count``, or a failed earlier attempt); a torn last line is
  skipped; user and tool messages carry the request time.

M10 lab audit R-1 (2026-10-15): the proxy log is the evidence an external agent's blame is decided from
(``runtime.agent_evidence``), so it must tell a dropped stream from a finished one and a capped session from a live one.
- A streamed exchange is ``complete`` only when it ended with its dialect's terminal marker (OpenAI ``[DONE]`` or a
  ``finish_reason``; Anthropic ``message_stop``; Responses ``response.completed`` / ``response.incomplete``) and carried
  no error event. A body that just ends is ``complete: false`` with ``error: "stream_drop: ..."``; an error event gives
  ``"stream_error: ..."`` (PATTERN: prime-agent ``crates/pa-ai/src/providers/openai_completions/stream.rs:704-715``).
  Every stream used to be recorded complete when the upstream closed it, cut or not.
- ``register(session, max_calls=n)`` caps the model calls the proxy forwards for a session (no agent obeys its own
  step setting). Past the cap the proxy answers, in the caller's dialect, a non-empty assistant stop message
  (``LIMIT_STOP_TEXT``; an empty one reads as a failed generation and is retried) that is never forwarded and never
  recorded as an exchange: the log gets one ``{"kind": "limit_stop"}`` row per stop up to ``LIMIT_STOP_ROWS``, and the
  rest as one row with ``count`` when the key is revoked (a flood past the cap cannot grow the log without bound);
  ``CaptureLogConnector`` and ``read_upstream_evidence`` add them up and skip them, so no turn the model did not generate
  can reach training data (PATTERN: OpenEnv ``src/openenv/core/harness/capture/sessions.py:158-178``,
  ``server.py:975-999``). A forwarded call that failed counts.
- ``probe(api, model)``: one minimal generation Crucible itself sends straight to the upstream, outside every session and
  never logged. The session log is writable by the agent's commands (they hold the capture key), so a non-policy upstream
  verdict read from it counts only when this probe of the same upstream answers the same (``agent_evidence``).
- The key is checked again after a request's body is read: a key revoked meanwhile gets 401 (it was forwarded uncapped,
  since ``unregister`` drops the cap, and recorded after the evidence read). A refused exchange keeps the upstream's
  ``retry_after``, so the probe of a 429 waits it out.
- Review round 3: ``inflight[session]`` counts the session's forwarded calls not yet answered and recorded, past
  ``unregister`` (a killed client does not cancel the proxy's forward, so a revoked session's calls still load the
  upstream and leave no row until they end); ``drain(session, timeout_s)`` waits for them. The self-load guard of
  ``agent_evidence`` counts what is left, so Crucible's probe never queues behind the session's own calls.

M3 audit R-1 (2026-10-20): a stream ended from the downstream side (``GeneratorExit``, or the ``CancelledError`` Starlette
raises on a disconnect under uvicorn) is ``error: "client disconnected"``. Under uvicorn it read ``CancelledError: ...``,
which M10 took for an upstream failure, so a deadline kill of a streaming agent became unknown blame. A mid-stream
shutdown of the proxy itself is labelled the same.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import re
import secrets
import time
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.agent_events import REJECTED, annotate_probes, user_type, writes_of
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

SESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # becomes a file name: no separators, no leading dot
MAX_TEXT = 4000
MAX_BODY = 64 * 1024 * 1024
PASS_HEADERS = ("retry-after", "x-request-id", "request-id", "anthropic-ratelimit-requests-remaining", "x-ratelimit-remaining-requests")
RETRY_HEADERS = ("x-stainless-retry-count", "idempotency-key", "x-stainless-timeout")
# M10 lab audit R-1: the per-session call cap's row kind and its stop message (non-empty: OpenEnv found an empty
# assistant message is read as a failed generation and retried, server.py:975-982)
LIMIT_STOP_KIND = "limit_stop"
LIMIT_STOP_TEXT = "[crucible] The model-call limit for this attempt is reached. Stop now."
STREAM_DROP = "stream_drop: the upstream body ended without a terminal marker"
# M10 lab audit R-1: limit-stop rows per session. Past the cap the proxy's answer is local, instant and free, so one row
# per answer let a policy that floods the proxy (its agent phase held open) grow the log on the Crucible host without
# bound. Stops past this many are counted in memory and written as one coalesced row when the key is revoked (OpenEnv
# keeps only an in-memory count, sessions.py:173 ``budget_stop_count``)
LIMIT_STOP_ROWS = 8
PROBE_TEXT = "Reply OK."


def limit_stop_count(row: dict[str, Any]) -> int:
    """How many stop answers a ``limit_stop`` row stands for: 1, or a coalesced row's ``count`` (M10 lab audit R-1)."""
    n = row.get("count", 1)
    return n if isinstance(n, int) and not isinstance(n, bool) and n > 0 else 1


def _upstream_headers(api: str, up: Upstream, version: str = "2023-06-01", beta: str | None = None) -> dict[str, str]:
    """The headers of a request to the upstream: its own key, never the agent's (Anthropic: version and beta)."""
    headers = {"content-type": "application/json"}
    if api == "anthropic":
        headers["anthropic-version"] = version
        if beta:
            headers["anthropic-beta"] = beta
        if up.key:
            headers["x-api-key"] = up.key
    elif up.key:
        headers["authorization"] = f"Bearer {up.key}"
    return headers


@dataclass
class Upstream:
    base: str  # OpenAI: ".../v1"; Anthropic: "https://api.anthropic.com/v1"
    key: str | None = None


class CaptureProxy:
    """``transport`` (an httpx transport) is for tests: the proxy can be exercised against a fake upstream."""

    def __init__(self, capture_dir: Path, *, openai: Upstream | None = None, anthropic: Upstream | None = None,
                 timeout_s: float = 600, transport: Any = None, max_body: int = MAX_BODY):
        self.upstreams = {k: v for k, v in (("openai", openai), ("anthropic", anthropic)) if v is not None}
        self.capture_dir = Path(capture_dir)
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(self.capture_dir, 0o700)  # raw prompts and outputs: the owner only (redaction runs at ingest)
        self.timeout_s, self.max_body = timeout_s, max_body
        self.transport = transport
        self.sessions: dict[str, str] = {}  # key -> session id
        # M10 lab audit R-1: per session, the call cap (0 = unlimited) and the calls forwarded so far (counted on forward,
        # a failed call included: OpenEnv sessions.py:160-163, a call the upstream rejected still cost an attempt)
        self.max_calls: dict[str, int] = {}
        self.forwarded: dict[str, int] = {}
        self.limit_stops: dict[str, int] = {}  # M10 lab audit R-1: per session, the stops answered past its cap
        # M10 lab audit R-1 review round 3: per session, forwarded calls not yet answered and recorded (kept past unregister)
        self.inflight: dict[str, int] = {}

    def register(self, session: str, *, max_calls: int = 0) -> str:
        """A capability key for one session. A session has one live key (two would interleave two runs in one log).
        ``max_calls`` > 0 caps the calls forwarded for it; past the cap the proxy answers a stop itself (module doc)."""
        if not SESSION.fullmatch(session):
            raise ValueError(f"session id must match {SESSION.pattern}: {session!r}")
        if session in self.sessions.values():
            raise ValueError(f"session {session!r} already has a live key (unregister it first)")
        if max_calls < 0:
            raise ValueError(f"max_calls must be >= 0 (0 = unlimited), got {max_calls}")
        key = "cap-" + secrets.token_urlsafe(18)
        self.sessions[key] = session
        self.max_calls[session], self.forwarded[session], self.limit_stops[session] = int(max_calls), 0, 0
        return key

    def unregister(self, key: str) -> None:
        """Idempotent: an unknown (or already revoked) key does nothing."""
        session = self.sessions.pop(key, None)
        if session is not None:
            self.max_calls.pop(session, None)
            self.forwarded.pop(session, None)
            over = self.limit_stops.pop(session, 0) - LIMIT_STOP_ROWS
            if over > 0:  # M10 lab audit R-1: the stops past the row bound, as one row; harnesses revoke the key before reading the log
                with contextlib.suppress(OSError):  # unregister runs in finally blocks: a full disk must not replace the result
                    self._record(session, {"kind": LIMIT_STOP_KIND, "time": _now(), "count": over, "coalesced": True})

    def _done(self, session: str) -> None:
        """M10 lab audit R-1 review round 3: one forwarded call of ``session`` answered and recorded (or abandoned)."""
        n = self.inflight.get(session, 0) - 1
        if n > 0:
            self.inflight[session] = n
        else:
            self.inflight.pop(session, None)

    async def drain(self, session: str, timeout_s: float) -> int:
        """M10 lab audit R-1 review round 3: wait up to ``timeout_s`` for the session's forwarded calls still in flight
        upstream (``inflight``; a revoked session's too); returns how many are left. A harness calls it after the agent
        phase and before it reads the session's log, so those calls are read and billed, and what is left is counted as
        the session's own load (``agent_evidence._self_loaded``)."""
        end = time.monotonic() + max(0.0, timeout_s)
        while self.inflight.get(session, 0) > 0 and time.monotonic() < end:
            await asyncio.sleep(min(0.2, max(0.0, end - time.monotonic())))
        return self.inflight.get(session, 0)

    async def probe(self, api: str, model: str) -> dict[str, Any]:
        """M10 lab audit R-1: one minimal generation of ``model`` sent by Crucible itself straight to the upstream of ``api``
        with the upstream key, outside every session and never written to any log, as a row-shaped dict (``status``,
        ``response``; the proxy's own 502 shape on a transport error; ``{}`` with no upstream for ``api``). The session log
        is writable by the agent's commands (they hold the capture key and pick the model, ``max_tokens`` and when to stop),
        so ``agent_evidence.confirm_upstream`` accepts a non-policy upstream verdict only when this answers the same. No
        token-limit key on the OpenAI body: gpt-5-family models reject ``max_tokens``."""
        import httpx

        up = self.upstreams.get("openai" if api == "responses" else api)
        if up is None:
            return {}
        if api == "anthropic":
            path, body = "messages", {"model": model, "max_tokens": 16, "messages": [{"role": "user", "content": PROBE_TEXT}]}
        elif api == "responses":
            path, body = "responses", {"model": model, "input": PROBE_TEXT, "max_output_tokens": 16}
        else:
            path, body = "chat/completions", {"model": model, "messages": [{"role": "user", "content": PROBE_TEXT}]}
        try:
            async with httpx.AsyncClient(timeout=min(self.timeout_s, 60), transport=self.transport) as c:
                r = await c.post(f"{up.base.rstrip('/')}/{path}", json=body, headers=_upstream_headers(api, up))
        except httpx.HTTPError as e:
            return {"status": 502, "complete": False, "error": f"{type(e).__name__}: {e}"}
        return {"status": r.status_code, "response": _json_or_text(r.content, r.headers.get("content-type", ""))}

    def _record(self, session: str, entry: dict[str, Any]) -> None:
        # ASCII-safe: a lone surrogate in an agent's request made the write fail after the upstream had billed the call
        line = json.dumps(entry, ensure_ascii=True, default=str) + "\n"
        fd = os.open(self.capture_dir / f"{session}.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as f:
            f.write(line)

    def app(self):
        import httpx
        from starlette.applications import Starlette
        from starlette.requests import Request
        from starlette.responses import JSONResponse, Response, StreamingResponse
        from starlette.routing import Route

        client = httpx.AsyncClient(timeout=self.timeout_s, transport=self.transport)
        proxy = self

        def error(api: str, status: int, message: str):
            if api == "anthropic":
                return JSONResponse({"type": "error", "error": {"type": "invalid_request_error", "message": message}}, status_code=status)
            return JSONResponse({"error": {"message": message, "type": "invalid_request_error"}}, status_code=status)

        def session_of(request: Request) -> str | None:
            presented = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
            for key, sess in proxy.sessions.items():  # constant-time compare: the key is the capability
                if secrets.compare_digest(key.encode(), presented.encode()):
                    return sess
            return None

        def upstream_headers(request: Request, api: str, up: Upstream) -> dict[str, str]:
            return _upstream_headers(api, up, request.headers.get("anthropic-version", "2023-06-01"), request.headers.get("anthropic-beta"))

        def url_of(request: Request, up: Upstream, path: str) -> str:
            q = request.url.query
            return f"{up.base.rstrip('/')}/{path}" + (f"?{q}" if q else "")  # Claude Code sends ?beta=true

        def passed(r: Any) -> dict[str, str]:
            return {h: r.headers[h] for h in PASS_HEADERS if h in r.headers}

        def retry_after(r: Any) -> dict[str, str]:
            # M10 lab audit R-1: a refusal's Retry-After is recorded, so Crucible's probe of a 429 waits it out
            return {"retry_after": r.headers["retry-after"]} if r.status_code >= 400 and "retry-after" in r.headers else {}

        async def passthrough(request: Request, api: str, path: str):
            """Unrecorded calls the SDKs make besides generation (token counting, model listing)."""
            up = proxy.upstreams.get(api)
            if up is None:
                return error(api, 404, f"no {api} upstream configured")
            session = session_of(request)
            if not session:
                return error(api, 401, "unknown session key")
            raw = await request.body()
            if session_of(request) != session:  # M10 lab audit R-1: the key was revoked while the body was read
                return error(api, 401, "unknown session key")
            r = await client.request(request.method, url_of(request, up, path), content=raw or None, headers=upstream_headers(request, api, up))
            return Response(r.content, status_code=r.status_code, media_type=r.headers.get("content-type"), headers=passed(r))

        async def forward(request: Request, api: str, path: str):
            up = proxy.upstreams.get("openai" if api == "responses" else api)
            if up is None:
                return error(api, 404, f"no {api} upstream configured")
            session = session_of(request)
            if not session:  # only registered keys: the key is the capability
                return error(api, 401, "unknown session key")
            raw = await request.body()
            # M10 lab audit R-1: the body read awaits; a key revoked meanwhile (the agent phase ended) is refused, never
            # forwarded uncapped (unregister drops the cap) or recorded after the evidence read. The same session, not just
            # a live one: a session id registered again must not take a request held under its old key
            if session_of(request) != session:
                return error(api, 401, "unknown session key")
            if len(raw) > proxy.max_body:
                return error(api, 413, f"request body over {proxy.max_body} bytes")
            try:
                body = json.loads(raw)
            except ValueError:
                return error(api, 400, "request body is not JSON")
            if not isinstance(body, dict):
                return error(api, 400, "request body is not a JSON object")
            # M10 lab audit R-1: the session's call cap. Past it the proxy answers a stop itself: nothing is forwarded and
            # no exchange is recorded (only a limit_stop row), so the stop can never become a trained turn
            cap, n = proxy.max_calls.get(session, 0), proxy.forwarded.get(session, 0)
            if cap and n >= cap:
                # one row per stop up to LIMIT_STOP_ROWS, then counted in memory (unregister writes the rest as one row); a
                # nonzero cap means the key is still live
                k = proxy.limit_stops[session] = proxy.limit_stops.get(session, 0) + 1
                if k <= LIMIT_STOP_ROWS:
                    proxy._record(session, {"kind": LIMIT_STOP_KIND, "api": api, "time": _now(), "calls": n, "max_calls": cap,
                                            "stream": bool(body.get("stream"))})
                return _limit_stop_response(api, body)
            # no await between the check and this count, so concurrent calls cannot overshoot the cap; the key was re-checked
            # after the body read, with no await since
            proxy.forwarded[session] = n + 1
            # M10 lab audit R-1 review round 3: in flight until its row is written (relay's finally for a stream, else this
            # handler's); no await since the count above
            proxy.inflight[session] = proxy.inflight.get(session, 0) + 1
            handed: list[bool] = []  # set when a stream's relay() takes the call over (its finally marks it done)
            try:
                return await forward_call(request, api, path, up, session, raw, body, handed)
            finally:
                if not handed:
                    proxy._done(session)

        async def forward_call(request: Request, api: str, path: str, up: Upstream, session: str, raw: bytes, body: dict[str, Any],
                               handed: list[bool]):
            """One forwarded call of ``forward`` (counted in flight there): to the upstream, recorded, relayed."""
            headers = upstream_headers(request, api, up)
            entry = {"api": api, "digest": hashlib.blake2b(raw, digest_size=16).hexdigest(), "request": body, "request_time": _now(),
                     "headers": {h: request.headers[h] for h in RETRY_HEADERS if h in request.headers}}
            t0 = time.monotonic()
            url = url_of(request, up, path)
            assemble = {"anthropic": assemble_anthropic, "responses": assemble_responses}.get(api, assemble_openai)
            if body.get("stream"):
                try:
                    r = await client.send(client.build_request("POST", url, content=raw, headers=headers), stream=True)
                except httpx.HTTPError as e:
                    proxy._record(session, {**entry, "time": _now(), "status": 502, "stream": True, "complete": False, "error": f"{type(e).__name__}: {e}"})
                    return error(api, 502, f"upstream unreachable: {type(e).__name__}")
                if r.status_code >= 400:  # an error is not a stream: the agent's SDK needs the status (and retry-after) to retry
                    content = await r.aread()
                    await r.aclose()
                    out = _json_or_text(content, r.headers.get("content-type", ""))
                    proxy._record(session, {**entry, "time": _now(), "latency_s": round(time.monotonic() - t0, 3), "status": r.status_code,
                                            "stream": True, "response": out, **retry_after(r)})
                    return Response(content, status_code=r.status_code, media_type=r.headers.get("content-type"), headers=passed(r))

                async def relay() -> AsyncIterator[bytes]:
                    events: list[dict[str, Any]] = []
                    data: list[str] = []
                    complete, err = False, None
                    terminal, stream_err = False, None  # M10 lab audit R-1: did the stream end with its marker, or an error event

                    def take(payload: str) -> None:
                        nonlocal terminal, stream_err
                        if payload.strip() == "[DONE]":
                            terminal = True
                            return
                        try:
                            ev = json.loads(payload)
                        except ValueError:
                            return
                        events.append(ev)
                        if isinstance(ev, dict):
                            done, bad = _stream_mark(api, ev)
                            terminal = terminal or done
                            stream_err = stream_err or bad

                    try:
                        async for line in r.aiter_lines():
                            if line.startswith("data:"):
                                data.append(line[5:].lstrip())
                            elif not line.strip() and data:  # an SSE event ends at a blank line; its data lines join
                                payload = "\n".join(data)
                                data = []
                                take(payload)
                            yield (line + "\n").encode()
                        if data:
                            take("\n".join(data))
                        # M10 lab audit R-1: a body that just ends was cut mid-flight (prime-agent stream.rs:704-715); it
                        # used to be recorded complete, so a dropped turn read as the agent's own answer
                        complete = terminal and stream_err is None
                        err = stream_err or (None if terminal else STREAM_DROP)
                    except BaseException as e:  # upstream failure or the client left (an interrupted turn is the strongest signal)
                        # M3 audit R-1 (2026-10-20): under uvicorn (ASGI 2.3) Starlette cancels the stream task on a disconnect,
                        # so a client that left raises CancelledError, not GeneratorExit (PATTERN: NeMo-Gym
                        # base_responses_api_model.py:1166-1169)
                        left = isinstance(e, (GeneratorExit, asyncio.CancelledError))
                        err = "client disconnected" if left else f"{type(e).__name__}: {e}"
                        raise
                    finally:
                        try:
                            await r.aclose()
                            proxy._record(session, {**entry, "time": _now(), "latency_s": round(time.monotonic() - t0, 3), "status": r.status_code,
                                                    "stream": True, "complete": complete, **({"error": err} if err else {}), "response": assemble(events)})
                        finally:
                            proxy._done(session)  # M10 lab audit R-1 review round 3: the call is no longer in flight
                # M10 lab audit R-1 review round 3: relay() marks the call done; a relay that never starts (the client left
                # first) leaves it counted, which only keeps the probe off (conservative)
                handed.append(True)
                return StreamingResponse(relay(), media_type="text/event-stream", headers=passed(r))
            try:
                r = await client.post(url, content=raw, headers=headers)
            except httpx.HTTPError as e:
                proxy._record(session, {**entry, "time": _now(), "status": 502, "complete": False, "error": f"{type(e).__name__}: {e}"})
                return error(api, 502, f"upstream unreachable: {type(e).__name__}")
            out = _json_or_text(r.content, r.headers.get("content-type", ""))
            proxy._record(session, {**entry, "time": _now(), "latency_s": round(time.monotonic() - t0, 3), "status": r.status_code, "response": out,
                                    **retry_after(r)})
            return JSONResponse(out, status_code=r.status_code, headers=passed(r))

        async def chat(request: Request):
            return await forward(request, "openai", "chat/completions")

        async def messages(request: Request):
            return await forward(request, "anthropic", "messages")

        async def responses(request: Request):
            return await forward(request, "responses", "responses")

        async def count_tokens(request: Request):
            return await passthrough(request, "anthropic", "messages/count_tokens")

        async def models(request: Request):
            api = "anthropic" if request.headers.get("x-api-key") and "anthropic" in proxy.upstreams else "openai"
            return await passthrough(request, api, "models")

        async def health(_request: Request):
            return JSONResponse({"ok": True, "upstreams": sorted(proxy.upstreams), "sessions": len(set(proxy.sessions.values()))})

        @contextlib.asynccontextmanager
        async def lifespan(_app):
            yield
            await client.aclose()

        return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/messages", messages, methods=["POST"]),
                                 Route("/v1/messages/count_tokens", count_tokens, methods=["POST"]), Route("/v1/responses", responses, methods=["POST"]),
                                 Route("/v1/models", models, methods=["GET"]), Route("/health", health)], lifespan=lifespan)


def _now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _json_or_text(content: bytes, ctype: str) -> Any:
    if "json" in ctype:
        with contextlib.suppress(ValueError):
            return json.loads(content)
    return {"error": content.decode("utf-8", "replace")[:2000]}


def _stream_error(x: Any) -> str:
    return "stream_error: " + str(x)[:300]


def _stream_mark(api: str, ev: dict[str, Any]) -> tuple[bool, str | None]:
    """(is this event its dialect's terminal marker, the error it carries). M10 lab audit R-1: a stream is complete only
    with a terminal marker and no error event (``[DONE]`` is handled by the caller; it is not JSON)."""
    t = ev.get("type")
    if api == "anthropic":
        if t == "message_stop":
            return True, None
        return False, (_stream_error(ev.get("error") or ev) if t == "error" else None)
    if api == "responses":
        if t in ("response.completed", "response.incomplete"):
            return True, None
        if t == "response.failed":
            return False, _stream_error((ev.get("response") or {}).get("error") or t)
        if t == "error":
            return False, _stream_error(ev.get("error") or ev.get("message") or ev)
        return False, None
    if ev.get("error"):  # an OpenAI-compatible mid-stream error chunk (assemble_openai keeps it in the response)
        return False, _stream_error(ev["error"])
    choices = ev.get("choices")
    done = isinstance(choices, list) and any(isinstance(c, dict) and c.get("finish_reason") for c in choices)
    return done, None


def _limit_stop_response(api: str, body: dict[str, Any]) -> Any:
    """The proxy's own answer past a session's call cap (M10 lab audit R-1), in the dialect and mode the caller asked for:
    a non-empty assistant message that ends the turn (OpenAI ``finish_reason: stop``, Anthropic ``end_turn``, Responses
    ``completed``) and carries no tool call, so the agent's loop ends cleanly (OpenEnv server.py:975-999)."""
    from starlette.responses import JSONResponse, StreamingResponse

    model, created = body.get("model"), int(time.time())
    stream = bool(body.get("stream"))

    def sse(events: list[tuple[str | None, Any]]) -> Any:
        lines = [(f"event: {name}\n" if name else "") + f"data: {data if isinstance(data, str) else json.dumps(data)}\n\n" for name, data in events]
        return StreamingResponse(iter(lines), media_type="text/event-stream")

    if api == "anthropic":
        msg = {"id": "msg_crucible_limit_stop", "type": "message", "role": "assistant", "model": model, "content": [{"type": "text", "text": LIMIT_STOP_TEXT}],
               "stop_reason": "end_turn", "stop_sequence": None, "usage": {"input_tokens": 0, "output_tokens": 0}}
        if not stream:
            return JSONResponse(msg)
        evs: list[dict[str, Any]] = [
            {"type": "message_start", "message": {**msg, "content": [], "stop_reason": None}},
            {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": LIMIT_STOP_TEXT}},
            {"type": "content_block_stop", "index": 0},
            {"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": None}, "usage": {"output_tokens": 0}},
            {"type": "message_stop"}]
        return sse([(e["type"], e) for e in evs])
    if api == "responses":
        resp = {"id": "resp_crucible_limit_stop", "object": "response", "created_at": created, "status": "completed", "model": model,
                "output": [{"type": "message", "id": "msg_crucible_limit_stop", "status": "completed", "role": "assistant",
                            "content": [{"type": "output_text", "text": LIMIT_STOP_TEXT, "annotations": []}]}],
                "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}}
        if not stream:
            return JSONResponse(resp)
        return sse([("response.created", {"type": "response.created", "response": {**resp, "status": "in_progress", "output": []}}),
                    ("response.completed", {"type": "response.completed", "response": resp})])
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if not stream:
        return JSONResponse({"id": "chatcmpl-crucible-limit-stop", "object": "chat.completion", "created": created, "model": model,
                             "choices": [{"index": 0, "message": {"role": "assistant", "content": LIMIT_STOP_TEXT}, "finish_reason": "stop"}], "usage": usage})
    chunk = {"id": "chatcmpl-crucible-limit-stop", "object": "chat.completion.chunk", "created": created, "model": model,
             "choices": [{"index": 0, "delta": {"role": "assistant", "content": LIMIT_STOP_TEXT}, "finish_reason": "stop"}], "usage": usage}
    return sse([(None, chunk), (None, "[DONE]")])


def assemble_openai(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    """OpenAI SSE chunks -> one chat.completion: one message per choice index (content, reasoning, tool calls, finish
    reason), usage, and a mid-stream ``error`` chunk if one came."""
    per: dict[int, dict[str, Any]] = {}
    usage, model, rid, err = None, None, None, None
    for ch in chunks:
        model, rid = ch.get("model", model), ch.get("id", rid)
        usage = ch.get("usage") or usage
        err = ch.get("error") or err
        for c in ch.get("choices", []):
            st = per.setdefault(int(c.get("index") or 0), {"content": [], "reasoning": [], "calls": {}, "by_id": {}, "last": None, "finish": None})
            d = c.get("delta") or {}
            if d.get("content"):
                st["content"].append(d["content"])
            for k in ("reasoning_content", "reasoning"):
                if isinstance(d.get(k), str) and d[k]:
                    st["reasoning"].append(d[k])
            for tc in d.get("tool_calls") or []:
                idx = tc.get("index")
                if idx is None:  # no index: the call's id names its slot, else it continues the last call
                    idx = st["by_id"].get(tc.get("id")) if tc.get("id") else st["last"]
                    if idx is None:
                        idx = max(st["calls"], default=-1) + 1
                slot = st["calls"].setdefault(idx, {"id": None, "type": "function", "function": {"name": "", "arguments": ""}})
                if tc.get("id"):
                    slot["id"] = tc["id"]
                    st["by_id"][tc["id"]] = idx
                fn = tc.get("function") or {}
                if fn.get("name") and not slot["function"]["name"]:  # set once: some providers repeat the name in every chunk
                    slot["function"]["name"] = fn["name"]
                slot["function"]["arguments"] += fn.get("arguments") or ""
                st["last"] = idx
            st["finish"] = c.get("finish_reason") or st["finish"]
    choices = []
    for i in sorted(per) or [0]:
        st = per.get(i) or {"content": [], "reasoning": [], "calls": {}, "finish": None}
        msg: dict[str, Any] = {"role": "assistant", "content": "".join(st["content"]) or None}
        if st["reasoning"]:
            msg["reasoning_content"] = "".join(st["reasoning"])
        if st["calls"]:
            msg["tool_calls"] = [st["calls"][k] for k in sorted(st["calls"])]
        choices.append({"index": i, "message": msg, "finish_reason": st["finish"]})
    return {"id": rid, "model": model, "object": "chat.completion", "choices": choices, "usage": usage, **({"error": err} if err else {})}


def assemble_responses(events: list[dict[str, Any]]) -> dict[str, Any]:
    """OpenAI Responses SSE events -> the final response object (``response.completed`` carries it whole)."""
    final = next((e.get("response") for e in reversed(events) if e.get("type") in ("response.completed", "response.incomplete", "response.failed")), None)
    if final:
        return final
    text = "".join(e.get("delta", "") for e in events if e.get("type") == "response.output_text.delta")
    return {"object": "response", "output": [{"type": "message", "content": [{"type": "output_text", "text": text}]}] if text else []}


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
        """Retries collapse onto their earlier attempt only when marked: an SDK retry count > 0, or an earlier attempt
        that failed or did not complete. A repeated body is otherwise a real second call (best-of-n). Logs from clients
        that send no retry information keep the old rule (an adjacent identical body is a retry)."""
        exchanges: list[dict[str, Any]] = []
        limit_stops = 0
        lines = Path(ref.meta["path"]).read_text(encoding="utf-8").splitlines()
        for n, x in enumerate(lines):
            if not x.strip():
                continue
            try:
                ex = json.loads(x)
            except ValueError:
                if n == len(lines) - 1:  # the proxy is still writing this line
                    continue
                raise
            if ex.get("kind") == LIMIT_STOP_KIND:  # M10 lab audit R-1: the proxy's own stop, never an exchange (no request)
                limit_stops += limit_stop_count(ex)
                continue
            hdr = {k.lower(): v for k, v in (ex.get("headers") or {}).items()}
            retry = hdr.get("x-stainless-retry-count")
            same = [i for i in range(len(exchanges) - 1, max(-1, len(exchanges) - 6), -1) if ex.get("digest") and exchanges[i].get("digest") == ex["digest"]]
            target = None
            if same:
                i = same[0]
                earlier = exchanges[i]
                failed = earlier.get("complete") is False or (earlier.get("status") or 200) >= 400
                if retry is not None:
                    target = i if (str(retry) not in ("", "0") or failed) else None
                elif failed or i == len(exchanges) - 1:
                    target = i
            if target is not None:
                exchanges[target] = ex  # the agent saw this attempt's response, not the earlier one
            else:
                exchanges.append(ex)
        return RawRecord(ref=ref, payload={"exchanges": exchanges, "limit_stops": limit_stops}, privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        events: list[TraceEvent] = []
        call_seq: dict[str, int] = {}
        models: set[str] = set()
        exchanges = raw.payload["exchanges"]
        reqs = [_conversation(ex) for ex in exchanges]
        threads = _threads(reqs, [_system_of(ex) for ex in exchanges])
        systems: dict[int, str] = {}

        def add(time_: str | None, **kw: Any) -> int:
            events.append(TraceEvent(seq=len(events), time=time_, **kw))
            return len(events) - 1

        def user(t: str | None, text: str, thread: int) -> None:
            human, harness = split_harness(text)
            for h in harness:
                add(t, kind="message", actor="system", type="harness_message", text=h[:MAX_TEXT], data={"thread": thread})
            if not human.strip():
                return
            if thread:  # a side call's "user" text is written by the agent or the harness (a title prompt, a subagent task)
                add(t, kind="message", actor="system", type="side_call_input", text=human[:MAX_TEXT], data={"thread": thread})
            else:
                add(t, kind="message", actor="user", type=user_type(human), text=human[:MAX_TEXT])

        def result(t: str | None, call_id: str | None, text: str, error: bool, thread: int) -> None:
            data = {"call_id": call_id, "chars": len(text), **({"error": True} if error else {}), **({"rejected": True} if REJECTED.search(text) else {}),
                    **({"thread": thread} if thread else {})}
            add(t, kind="observation", actor="system", type="tool_result", text=text[:MAX_TEXT], reply_to=call_seq.get(call_id or ""), data=data)

        def call(t: str | None, call_id: str | None, name: str, args: Any, thread: int, server: bool = False) -> None:
            arg_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False, default=str)
            w = writes_of(name, args)
            call_seq[call_id or ""] = add(t, kind="action", actor="agent", type="tool_call", text=f"{name}({arg_text[:200]})",
                                          data={"tool": name, "call_id": call_id, "arguments": arg_text[:MAX_TEXT], **({"writes": w} if w else {}),
                                                **({"thread": thread} if thread else {}), **({"server_tool": True} if server else {})})

        for i, ex in enumerate(exchanges):
            t_req, t = ex.get("request_time") or ex.get("time"), ex.get("time")
            api, req = ex.get("api", "openai"), ex["request"]
            thread, start, reset = threads[i]
            if reset:
                add(t_req, kind="observation", actor="system", type="context_reset", text=f"request does not extend the conversation ({len(reqs[i])} messages)")
            system = _text(req.get("system")) if api == "anthropic" else (req.get("instructions") if api == "responses" else None)
            for m in _messages(api, req):
                if m.get("role") in ("system", "developer"):
                    system = _text(m.get("content"))
            if system and systems.get(thread) != system:  # a re-rendered system prompt is not a new conversation
                systems[thread] = system
                add(t_req, kind="message", actor="system", type="system_message", text=system[:MAX_TEXT], data={"thread": thread} if thread else {})
            for m in [m for m in _messages(api, req) if m.get("role") not in ("system", "developer")][start:]:  # start counts these
                role, content = m.get("role"), m.get("content")
                if role in ("assistant", "system", "developer") or m.get("type") in ("function_call", "reasoning"):
                    continue  # the agent's turns are recorded from the responses that produced them
                if m.get("type") == "function_call_output":
                    result(t_req, m.get("call_id"), _text(m.get("output")), False, thread)
                elif api == "anthropic" and isinstance(content, list):
                    texts = []
                    for b in content:
                        if b.get("type") == "tool_result":
                            result(t_req, b.get("tool_use_id"), _text(b.get("content")), bool(b.get("is_error")), thread)
                        elif b.get("type") in ("text", "image", "document"):
                            texts.append(b.get("text", "") if b.get("type") == "text" else f"[{b['type']}]")
                    user(t_req, "\n".join(x for x in texts if x), thread)
                elif role == "tool":
                    result(t_req, m.get("tool_call_id"), _text(content), False, thread)
                else:
                    user(t_req, _text(content), thread)
            resp = ex.get("response") or {}
            model = resp.get("model") or req.get("model")
            if model:
                models.add(model)
            data: dict[str, Any] = {"model": model, "usage": resp.get("usage"), "latency_s": ex.get("latency_s"), "status": ex.get("status"),
                                    **({"thread": thread} if thread else {}), **({"complete": False} if ex.get("complete") is False else {})}
            if resp.get("error") or ex.get("error") or (ex.get("status") or 200) >= 400:
                data["error"] = resp.get("error") or ex.get("error") or f"HTTP {ex.get('status')}"
            if api == "anthropic":
                blocks = resp.get("content") or []
                thinking = "\n".join(b.get("thinking", "") for b in blocks if b.get("type") == "thinking")
                data.update({"finish": resp.get("stop_reason"), **({"reasoning": thinking[:MAX_TEXT]} if thinking else {})})
                add(t, kind="message", actor="agent", type="llm_call", text=_text(blocks)[:MAX_TEXT], data=data)
                for b in blocks:
                    if b.get("type") in ("tool_use", "server_tool_use"):  # server tools (web search, code execution) are calls too
                        call(t, b.get("id"), b.get("name", ""), b.get("input") or {}, thread, server=b["type"] == "server_tool_use")
                    elif b.get("type", "").endswith("_tool_result"):
                        result(t, b.get("tool_use_id"), _text(b.get("content")) if not isinstance(b.get("content"), list) else json.dumps(b["content"])[:MAX_TEXT],
                               False, thread)
            elif api == "responses":
                out = resp.get("output") or []
                text = "\n".join(_text(o.get("content")) for o in out if o.get("type") == "message")
                data.update({"finish": resp.get("status")})
                add(t, kind="message", actor="agent", type="llm_call", text=text[:MAX_TEXT], data=data)
                for o in out:
                    if o.get("type") in ("function_call", "custom_tool_call", "local_shell_call"):
                        call(t, o.get("call_id") or o.get("id"), o.get("name") or o["type"], _args(o.get("arguments") or o.get("input") or o.get("action") or ""), thread)
            else:
                choice = (resp.get("choices") or [{}])[0]
                m = choice.get("message") or {}
                data.update({"finish": choice.get("finish_reason"), **({"logprobs": True} if choice.get("logprobs") else {}),
                             **({"reasoning": str(m.get("reasoning_content") or m.get("reasoning"))[:MAX_TEXT]} if m.get("reasoning_content") or m.get("reasoning") else {}),
                             **({"choices": len(resp["choices"])} if len(resp.get("choices") or []) > 1 else {})})
                add(t, kind="message", actor="agent", type="llm_call", text=_text(m.get("content"))[:MAX_TEXT], data=data)
                for tc in m.get("tool_calls") or []:
                    fn = tc.get("function") or {}
                    call(t, tc.get("id"), fn.get("name") or "", _args(fn.get("arguments") or ""), thread)
        probes = annotate_probes(events)
        actors = [TraceActor(id="user", role="user", kind="human"),
                  TraceActor(id="agent", role="agent", kind="agent", handle=",".join(sorted(models)) or None),
                  TraceActor(id="system", role="system", kind="system")]
        return Trace(trace_id=f"capture:{raw.ref.key}", family=self.family,
                     source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=actors, events=events, outcome=TraceOutcome(details={"probes": probes} if probes["attempted"] else {}))


# ---------------------------------------------------------------- conversation matching
_HARNESS = re.compile(r"<(system-reminder|command-name|command-message|command-args|local-command-stdout|local-command-stderr|user-prompt-submit-hook)>"
                      r"[\s\S]*?</\1>|^Caveat: The messages below were generated by the user while running local commands[^\n]*$", re.MULTILINE)


def split_harness(text: str) -> tuple[str, list[str]]:
    """(what the person typed, harness-injected blocks): reminders, slash-command echoes and hook output arrive in the
    user role but were never typed by the person (M4 read them as user feedback)."""
    harness = [m.group(0) for m in _HARNESS.finditer(text)]
    return _HARNESS.sub("", text).strip(), harness


def _messages(api: str, req: dict[str, Any]) -> list[dict[str, Any]]:
    if api == "responses":
        inp = req.get("input")
        return [{"role": "user", "content": inp}] if isinstance(inp, str) else list(inp or [])
    return list(req.get("messages") or [])


def _strip_markers(x: Any) -> Any:
    if isinstance(x, dict):
        return {k: _strip_markers(v) for k, v in x.items() if k not in ("cache_control", "citations")}
    if isinstance(x, list):
        return [_strip_markers(v) for v in x]
    return x


def _conversation(ex: dict[str, Any]) -> list[str]:
    """A request's messages for prefix matching: system turns dropped (re-rendered every turn by some harnesses), cache
    markers dropped (a rolling ``cache_control`` changed the previous user turn), one canonical string each."""
    out = []
    for m in _messages(ex.get("api", "openai"), ex.get("request") or {}):
        if m.get("role") in ("system", "developer"):
            continue
        c = _strip_markers(m.get("content"))
        if isinstance(c, list) and len(c) == 1 and isinstance(c[0], dict) and c[0].get("type") in ("text", "input_text") and set(c[0]) <= {"type", "text"}:
            c = c[0].get("text")  # [{"type": "text", "text": x}] and x are the same turn
        out.append(json.dumps({**_strip_markers({k: v for k, v in m.items() if k != "content"}), "content": c}, sort_keys=True, ensure_ascii=False))
    return out


def _system_of(ex: dict[str, Any]) -> str:
    req = ex.get("request") or {}
    return _text(req.get("system")) if ex.get("api") == "anthropic" else str(req.get("instructions") or "")


def _common(a: list[str], b: list[str]) -> int:
    n = 0
    while n < len(a) and n < len(b) and a[n] == b[n]:
        n += 1
    return n


def _threads(reqs: list[list[str]], systems: list[str] | None = None) -> list[tuple[int, int, bool]]:
    """Per request: (thread, first new message, context_reset). A request continues the earlier request it shares the
    longest prefix with, unless it adds nothing to it under another system prompt (a title call repeating the user's
    first message). One that continues none is a side call (its own thread) when the main conversation continues
    after it, else the main conversation was reset (compaction, a fresh start) and continues from it."""
    systems = systems or [""] * len(reqs)
    out: list[tuple[int, int, bool]] = []
    thread_of: list[int] = []
    main_since = 0  # index of the request that (re)started the main thread
    nxt = 1
    for i, r in enumerate(reqs):
        best, parent = 0, None
        for j in range(i):
            c = _common(reqs[j], r)
            if c and c >= len(r) and systems[j] != systems[i]:
                continue  # nothing new, another system prompt: a side call reusing the conversation, not a turn of it
            if c > best or (c == best and c and parent is not None and thread_of[j] == 0):
                best, parent = c, j
        if parent is not None:
            # new messages: those past what the parent request itself had (its response is recorded already)
            thread_of.append(thread_of[parent])
            out.append((thread_of[parent], min(best, len(reqs[parent])), False))
            continue
        if i == 0:
            thread_of.append(0)
            out.append((0, 0, False))
            continue
        main_reqs = [j for j in range(main_since, i) if thread_of[j] == 0]
        resumed = any(_common(reqs[k], reqs[j]) > 0 for k in range(i + 1, len(reqs)) for j in main_reqs)
        if resumed:
            thread_of.append(nxt)
            out.append((nxt, 0, False))
            nxt += 1
        else:
            thread_of.append(0)
            out.append((0, 0, True))
            main_since = i
    return out
