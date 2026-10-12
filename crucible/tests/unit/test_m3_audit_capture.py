"""M3 audit (2026-10-12), batch C: agent-session capture and import (capture proxy, shared probe vocabulary, ATIF,
chat transcripts, Claude Code sessions, OTel). Each test reproduced a defect before its fix
(``docs/audit/2026-10-12/M03-connectors.md`` §2.3)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from pathlib import Path

import httpx
import pytest

from crucible.connectors.base import SourceRef
from crucible.connectors.capture_proxy import CaptureLogConnector, CaptureProxy, Upstream, assemble_openai
from crucible.core.agent_events import INTERRUPT, REJECTED, annotate_probes, probes_of, writes_of
from crucible.core.schema.trace import TraceEvent


def _log(tmp_path: Path, rows: list[dict]) -> list:
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    c = CaptureLogConnector()
    return c.normalize(c.fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(tmp_path / "s.jsonl")}))).events


def _ex(messages, reply="ok", system=None, digest=None, **kw):
    req = {"messages": messages, **({"system": system} if system else {})}
    import hashlib

    return {"api": "anthropic", "digest": digest or hashlib.sha256(json.dumps(req).encode()).hexdigest()[:32], "request": req, "response": {"content": [{"type": "text", "text": reply}]}, **kw}


def test_C1_cache_markers_and_side_calls_are_not_context_resets(tmp_path):
    u1 = {"role": "user", "content": [{"type": "text", "text": "fix the bug", "cache_control": {"type": "ephemeral"}}]}
    u1b = {"role": "user", "content": [{"type": "text", "text": "fix the bug"}]}
    a1 = {"role": "assistant", "content": [{"type": "tool_use", "id": "t1", "name": "Bash", "input": {"command": "ls"}}]}
    r1 = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py"}]}
    side = _ex([{"role": "user", "content": "fix the bug"}], reply="Bug fix", system="Generate a title")
    ev = _log(tmp_path, [_ex([u1]), side, _ex([u1b, a1, r1], system=None)])
    types = [e.type for e in ev]
    assert "context_reset" not in types and types.count("user_message") == 1  # was: 2 resets, 3 user_messages for one human turn
    assert any(e.data.get("thread", 0) != 0 for e in ev)  # the title call is its own thread
    ev = _log(tmp_path, [_ex([u1b]), _ex([{"role": "user", "content": "summary of earlier work"}])])  # history replaced, never resumed
    assert "context_reset" in [e.type for e in ev]


def test_C14_harness_text_is_not_the_human(tmp_path):
    u = {"role": "user", "content": [{"type": "text", "text": "<system-reminder>todo list is empty</system-reminder>"}, {"type": "text", "text": "hi"},
                                     {"type": "image", "source": {"type": "base64", "data": "AAAA"}}]}
    ev = _log(tmp_path, [_ex([u])])
    human = [e for e in ev if e.actor == "user"]
    assert [e.text for e in human] == ["hi\n[image]"] and any(e.type == "harness_message" and "todo list" in e.text for e in ev)


def test_C10_retries_collapse_only_when_marked(tmp_path):
    same = {"role": "user", "content": "roll a die"}
    ev = _log(tmp_path, [_ex([same], reply="4", digest="d", headers={"x-stainless-retry-count": "0"}),
                         _ex([same], reply="2", digest="d", headers={"x-stainless-retry-count": "0"})])
    assert [e.text for e in ev if e.type == "llm_call"] == ["4", "2"]  # best-of-n kept 1 of 2
    ev = _log(tmp_path, [_ex([same], reply="lost", digest="d", status=200, headers={"x-stainless-retry-count": "0"}),
                         _ex([{"role": "user", "content": "other"}], reply="x", digest="e"),
                         _ex([same], reply="seen", digest="d", headers={"x-stainless-retry-count": "1"})])
    assert [e.text for e in ev if e.type == "llm_call"].count("lost") == 0  # a marked retry replaces its earlier attempt


def test_C12_a_torn_last_line_is_skipped(tmp_path):
    (tmp_path / "s.jsonl").write_text(json.dumps(_ex([{"role": "user", "content": "hi"}])) + "\n" + '{"api": "anthropic", "dige')
    c = CaptureLogConnector()
    assert len(c.fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(tmp_path / "s.jsonl")})).payload["exchanges"]) == 1


def test_C5_C7_openai_stream_assembly():
    chunks = [{"choices": [{"index": 0, "delta": {"content": "Hello"}}]}, {"choices": [{"index": 1, "delta": {"content": "Bye"}}]},
              {"choices": [{"index": 0, "delta": {"reasoning_content": "think"}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "a", "function": {"name": "ls", "arguments": "{}"}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"id": "b", "function": {"name": "cat", "arguments": '{"p":1}'}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 2, "id": "c", "function": {"name": "bash"}}]}}]},
              {"choices": [{"index": 0, "delta": {"tool_calls": [{"index": 2, "function": {"name": "bash", "arguments": "{}"}}]}}]},
              {"error": {"message": "overloaded"}}]
    r = assemble_openai(chunks)
    m = r["choices"][0]["message"]
    assert m["content"] == "Hello" and m["reasoning_content"] == "think" and len(r["choices"]) == 2  # was "HelloBye"
    assert [c["function"]["name"] for c in m["tool_calls"]] == ["ls", "cat", "bash"]  # was "lscat" and "bashbash"
    assert r["error"] == {"message": "overloaded"}


def _upstream(status: int = 429, stream_fail: bool = False):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def chat(request):
        await request.json()
        if stream_fail:
            async def gen():
                yield 'data: {"choices": [{"index": 0, "delta": {"content": "par"}}]}\n\n'
                raise httpx.ReadError("upstream died")
            return StreamingResponse(gen(), media_type="text/event-stream")
        return JSONResponse({"error": {"message": "slow down"}}, status_code=status, headers={"retry-after": "7"})

    async def models(request):
        return JSONResponse({"data": [{"id": "m"}]})

    async def count(request):
        return JSONResponse({"input_tokens": 3, "query": request.url.query})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/models", models),
                             Route("/v1/messages/count_tokens", count, methods=["POST"])])


def test_C2_C9_streamed_errors_keep_status_and_routes_pass_through(tmp_path):
    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up/v1", "k"), anthropic=Upstream("http://up/v1", "k"),
                         transport=httpx.ASGITransport(app=_upstream()))
    key = proxy.register("s")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as c:
            h = {"authorization": f"Bearer {key}"}
            r = await c.post("/v1/chat/completions", json={"model": "m", "stream": True, "messages": [{"role": "user", "content": "hi"}]}, headers=h)
            assert r.status_code == 429 and r.headers.get("retry-after") == "7"  # was 200 + text/event-stream: the SDK never retried
            assert (await c.get("/v1/models", headers=h)).json()["data"][0]["id"] == "m"  # was 404
            cnt = await c.post("/v1/messages/count_tokens?beta=true", json={"model": "m", "messages": []}, headers={"x-api-key": key})
            assert cnt.json() == {"input_tokens": 3, "query": "beta=true"}  # the query string was dropped

    asyncio.run(run())
    rec = [json.loads(x) for x in (tmp_path / "cap" / "s.jsonl").read_text().splitlines()]
    assert rec[0]["status"] == 429 and "request_time" in rec[0]
    assert (tmp_path / "cap").stat().st_mode & 0o077 == 0 and (tmp_path / "cap" / "s.jsonl").stat().st_mode & 0o077 == 0


def test_C3_C4_a_failed_stream_and_odd_bytes_are_still_recorded(tmp_path):
    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up/v1", "k"), transport=httpx.ASGITransport(app=_upstream(stream_fail=True)))
    key = proxy.register("s")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as c:
            body = '{"model": "m", "stream": true, "messages": [{"role": "user", "content": "bin \\udcff"}]}'
            with contextlib.suppress(Exception):  # the client may see the broken stream
                await c.post("/v1/chat/completions", content=body, headers={"authorization": f"Bearer {key}", "content-type": "application/json"})

    asyncio.run(run())
    rec = [json.loads(x) for x in (tmp_path / "cap" / "s.jsonl").read_text().splitlines()]
    assert len(rec) == 1 and rec[0]["complete"] is False and "ReadError" in rec[0]["error"]  # 0 records: the turn was lost
    with pytest.raises(ValueError):
        proxy.register("s")  # a second live key would interleave two runs in one log


@pytest.mark.parametrize("cmd", ["git --no-pager log --all --oneline", "git -c core.pager=cat log --all"])
def test_C15_git_history_probes_with_global_options(cmd):
    assert any(p["kind"] == "git_history" for p in probes_of("Bash", json.dumps({"command": cmd})))


def test_C15_argv_commands_and_escaped_newlines():
    assert any(p["kind"] == "git_history" for p in probes_of("shell", json.dumps({"command": ["git", "log", "--all"]})))
    assert any(p["kind"] == "package_registry" for p in probes_of("shell", json.dumps({"command": ["pip", "download", "django==5.0.1"]})))
    assert not any(p["kind"] == "git_history" for p in probes_of("Bash", json.dumps({"command": "git log --oneline -3\ncat /tmp/upstream/notes.txt"})))


def _probe(tool: str, args: dict, out: str, **rdata) -> dict:
    ev = [TraceEvent(seq=0, kind="action", type="tool_call", data={"tool": tool, "arguments": json.dumps(args), "call_id": "c"}),
          TraceEvent(seq=1, kind="observation", type="tool_result", text=out, reply_to=0, data={"call_id": "c", **rdata})]
    annotate_probes(ev)
    return ev[0].data


def test_C16_a_probe_that_worked_is_not_scored_as_failed():
    d = _probe("Bash", {"command": "git log --all --oneline"}, "9f1e2d3 Fix file not found error in loader (#4521)\n1a2b3c4 base")
    assert d["probe_ran"] and d["probe_exposed"] == ["git_history"]  # "not found" in a commit subject made it "failed"


def test_C17_ordinary_paths_and_searches_are_not_exposures():
    assert not _probe("Bash", {"command": "cd /root/project && python -m pytest"}, "3 passed").get("probes")
    assert _probe("WebSearch", {"query": "python asyncio gather docs"}, "asyncio.gather runs awaitables concurrently")["probe_exposed"] == []
    assert _probe("WebFetch", {"url": "https://example.com"}, "see https://github.com/acme/lib/pull/4521 diff --git a/x b/x")["probe_exposed"] == ["web"]


def test_C18_writes_of():
    assert writes_of("str_replace_based_edit_tool", {"command": "view", "path": "tests/test_a.py"}) == []  # a read counted as a (test) edit
    assert writes_of("mcp__filesystem__write_file", {"path": "a.py"}) == ["file:a.py"]
    assert writes_of("write_file", {"file_path": "b.py"}) == ["file:b.py"]  # Gemini CLI
    assert writes_of("apply_patch", "*** Begin Patch\n*** Update File: c.py\n*** Move to: d.py\n*** End Patch") == ["file:c.py", "file:d.py"]
    bash = writes_of("bash", {"command": "sed -i 's/a/b/' e.py && echo x > f.txt && cat <<EOF >> g.py\nx\nEOF\nls 2>/dev/null | tee h.log"})
    assert bash == ["file:e.py", "file:f.txt", "file:g.py", "file:h.log"]  # mini-swe-agent edits only through bash


def test_C19_rejections_and_interrupts_are_the_harness_sentences():
    assert not REJECTED.search("abc123 Show toast when user rejected the invite")
    assert REJECTED.search("The user doesn't want to proceed with this tool use. The tool use was rejected")
    assert not INTERRUPT.search("grep -n 'Interrupted by user' agent_events.py") and INTERRUPT.search("[Request interrupted by user]")


def _atif(steps):
    from harbor.models.trajectories import Trajectory

    return Trajectory.model_validate({"schema_version": "ATIF-v1.7", "session_id": "s1", "agent": {"name": "claude-code", "version": "1", "model_name": "m"},
                                      "steps": steps})


def test_C20_C23_atif_sidechains_copied_context_and_errors():
    pytest.importorskip("harbor")
    from crucible.core.schema.trace import TraceSource
    from crucible.interop.harbor.atif_import import trajectory_to_trace

    steps = [{"step_id": 1, "source": "user", "message": "port the kernel"},
             {"step_id": 2, "source": "user", "message": "Explore the repo and report", "extra": {"is_sidechain": True, "agent_id": "sub1"}},
             {"step_id": 3, "source": "agent", "message": "", "extra": {"is_sidechain": True, "agent_id": "sub1"},
              "tool_calls": [{"tool_call_id": "w", "function_name": "Write", "arguments": {"file_path": "x.cu", "content": ""}}],
              "observation": {"results": [{"source_call_id": "w", "content": "denied", "extra": {"tool_result_is_error": True}}]}},
             {"step_id": 4, "source": "agent", "message": "", "is_copied_context": True,
              "tool_calls": [{"tool_call_id": "g", "function_name": "Bash", "arguments": {"command": "git log --all"}}]}]
    t = trajectory_to_trace(_atif(steps), trace_id="t", source=TraceSource(connector="atif", ref="r"))
    human = [e for e in t.events if e.actor == "user"]
    assert [e.text for e in human] == ["port the kernel"]  # the subagent's prompt was a human user_message
    write = next(e for e in t.events if e.type == "tool_call")
    assert write.data["subagent"] == "sub1"
    res = next(e for e in t.events if e.type == "tool_result")
    assert res.data.get("error") is True
    assert not any(e.data.get("tool") == "Bash" for e in t.events)  # a copied-context step was re-emitted as a new probe


def test_C26_unlinked_results_answer_the_oldest_open_call():
    from crucible.connectors.chat_messages import messages_to_trace
    from crucible.core.schema.trace import TraceSource

    msgs = [{"role": "user", "content": "read both"},
            {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "read", "arguments": '{"p": "A"}'}}, {"function": {"name": "read", "arguments": '{"p": "B"}'}}]},
            {"role": "tool", "content": "contents of A"}, {"role": "tool", "content": "contents of B"}]
    t = messages_to_trace(msgs, trace_id="t", source=TraceSource(connector="chat", ref="r"))
    calls = {json.loads(e.data["arguments"])["p"] if isinstance(e.data["arguments"], str) else e.data["arguments"]["p"]: e.seq for e in t.events if e.type == "tool_call"}
    res = {e.text: e.reply_to for e in t.events if e.type == "tool_result"}
    assert res["contents of A"] == calls["A"] and res["contents of B"] == calls["B"]  # swapped


def test_C27_C29_otel_import(tmp_path):
    from crucible.connectors.otel_import import OtelConnector

    def span(sid, name, start, end, attrs, parent=None, events=()):
        return {"traceId": "t1", "spanId": sid, "parentSpanId": parent or "", "name": name, "startTimeUnixNano": str(start), "endTimeUnixNano": str(end),
                "attributes": [{"key": k, "value": {"stringValue": v}} for k, v in attrs.items()], "events": list(events)}

    ev = lambda n, t, v: {"name": n, "timeUnixNano": str(t), "attributes": [{"key": "content", "value": {"stringValue": v}}]}
    spans = [span("a", "invoke_agent", 1000, 99000, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "bot"}),
             span("b", "execute_tool write_file", 1000, 9000000, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "write_file",
                                                                   "gen_ai.tool.call.arguments": '{"path": "a.py"}'}, parent="a"),
             span("c", "chat", 2000, 3000, {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m"}, parent="b",
                  events=[ev("gen_ai.user.message", 2001, "first"), ev("gen_ai.user.message", 2002, "second")]),
             span("d", "tool.call", 5000, 6000, {"openinference.span.kind": "TOOL", "tool.name": "bash", "input.value": '{"command": "git log --all"}'}, parent="a")]
    spans.append(dict(spans[1]))  # a duplicate export of one span
    doc = {"resourceSpans": [{"resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "svc"}}]}, "scopeSpans": [{"spans": spans}]}]}
    (tmp_path / "t.json").write_text(json.dumps(doc))
    c = OtelConnector()
    [ref] = list(c.discover({"glob": str(tmp_path / "t.json")}))
    t = c.normalize(c.fetch(ref))
    times = [e.time for e in t.events]
    assert times == sorted(times)  # a tool result at 9 ms came before an llm call at 2 us
    assert sum(1 for e in t.events if e.type == "tool_call" and e.data.get("tool") == "write_file") == 1  # duplicate export
    call = next(e for e in t.events if e.type == "tool_call" and e.data.get("tool") == "write_file")
    assert call.data.get("writes") == ["file:a.py"]
    llm = next(e for e in t.events if e.type == "llm_call")
    assert "first" in json.dumps(llm.data) and "second" in json.dumps(llm.data)  # the second event overwrote the first
    assert any(e.type == "tool_call" and e.data.get("tool") == "bash" and e.data.get("probes") for e in t.events)  # OpenInference span
    assert t.events[0].time.endswith("Z") and ".000001" in t.events[0].time  # ns times went through a float


def test_C24_C25_claude_code_sessions_convert_one_session_and_refresh_grown_copies(tmp_path, monkeypatch):
    import crucible_pack_coding.connectors.claude_code_sessions as ccs

    project = tmp_path / "proj"
    project.mkdir()
    for sid in ("aaa", "bbb"):
        (project / f"{sid}.jsonl").write_text(f'{{"task": "{sid}"}}\n')
    seen: list[list[str]] = []

    def fake_convert(d):
        seen.append(sorted(p.name for p in Path(d).glob("*.jsonl")))

    monkeypatch.setattr(ccs, "convert_claude_code_sessions", fake_convert)
    c = ccs.ClaudeCodeSessionConnector(tmp_path / "frozen", tasks=set())
    for ref in c.discover({"project_dir": project, "freeze": False}):
        with pytest.raises(ValueError):
            c.normalize(c.fetch(ref))
    assert seen == [["aaa.jsonl"], ["bbb.jsonl"]]  # each trace held every session of the project
    list(c.discover({"project_dir": project}))
    assert (tmp_path / "frozen").stat().st_mode & 0o077 == 0  # raw transcripts: the owner only
    (project / "aaa.jsonl").write_text('{"task": "aaa"}\n{"more": 1}\n')  # the live session grew
    list(c.discover({"project_dir": project}))
    assert "more" in (tmp_path / "frozen" / "aaa" / "aaa.jsonl").read_text()  # frozen once, stale forever
