"""M3 core connectors: redaction canaries, file_drop, sql_audit_log, otel_import, capture_proxy, idempotent ingest."""

from __future__ import annotations

import asyncio
import json
import sqlite3

import httpx
import pytest

from crucible.connectors.base import RawRecord, SourceRef, ingest
from crucible.connectors.capture_proxy import CaptureLogConnector, CaptureProxy, Upstream
from crucible.connectors.file_drop import FieldMap, FileDropConnector
from crucible.connectors.otel_import import OtelConnector
from crucible.connectors.redact import RedactionPolicy, redact
from crucible.connectors.sql_audit_log import SqlAuditLogConnector
from crucible.connectors.trace_store import TraceStore
from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource
from crucible.core.store import Store

# Every format the redactor claims to remove, built at runtime so this file holds no literal credential.
CANARIES = {
    "openai": "sk-proj-" + "Q" * 40,
    "openrouter": "sk-or-v1-" + "a1b2" * 12,
    "anthropic": "sk-ant-api03-" + "Z9" * 20,
    "github_classic": "ghp_" + "A" * 36,
    "github_fine": "github_pat_" + "B" * 50,
    "huggingface": "hf_" + "C" * 34,
    "aws": "AKIA" + "ABCDEFGHIJKLMNOP",
    "slack": "xoxb-" + "1234567890-abcdefghij",
    "gitlab": "glpat-" + "D" * 20,
    "google": "AIza" + "E" * 35,
    "jwt": "eyJ" + "h" * 20 + ".eyJ" + "p" * 20 + "." + "s" * 20,
    "private_key": "-----BEGIN RSA PRIVATE KEY-----\nMIIEow" + "F" * 40 + "\n-----END RSA PRIVATE KEY-----",
    "password": "password=" + "hunter2hunter2",
    "email": "jane.doe@acme-industries.com",
    "host_corp": "build01.acme.corp",
    "host_ts": "lab-gpu.tail1234.ts.net",
    "ip_10": "10.20.30.40",
    "ip_192": "192.168.1.77",
    "ip_tailscale": "100.100.101.36",
}
SECRET_VALUES = {k: v.split("=", 1)[1] if k == "password" else v for k, v in CANARIES.items()}


def _trace() -> Trace:
    blob = " | ".join(f"{k}: {v}" for k, v in CANARIES.items())
    events = [
        TraceEvent(seq=0, kind="message", actor="jane.doe", type="comment", text=f"see {blob}"),
        TraceEvent(seq=1, kind="action", actor="bot", type="tool_call", text="run", data={"env": {"KEY": CANARIES["openai"]}, "hosts": [CANARIES["host_corp"], CANARIES["ip_10"]]}),
        TraceEvent(seq=2, kind="observation", actor="jane.doe", type="reply", text="ok", reply_to=0),
    ]
    sig = Signal(id="s1", type="correction", polarity=-1, grades=[1], source_event=0, actor="jane.doe", text=f"key {CANARIES['github_classic']}", meta={"mail": CANARIES["email"]})
    return Trace(trace_id="t1", source=TraceSource(connector="test", ref=f"ticket by {CANARIES['email']}"),
                 actors=[TraceActor(id="jane.doe", role="reviewer", kind="human", handle="jdoe"), TraceActor(id="bot", role="agent", kind="agent")],
                 events=events, signals=[sig], outcome=TraceOutcome(details={"by": CANARIES["email"], "token": CANARIES["huggingface"]}))


def test_redaction_removes_every_planted_canary():
    out = redact(_trace(), RedactionPolicy())
    dumped = out.model_dump_json()
    leaked = [k for k, v in SECRET_VALUES.items() if v in dumped]
    assert leaked == []
    assert "jane.doe" not in dumped and "jdoe" not in dumped  # human actor id and handle are pseudonymized everywhere
    human = next(a for a in out.actors if a.role == "reviewer")
    assert out.events[0].actor == human.id == out.events[2].actor == out.signals[0].actor
    assert next(a for a in out.actors if a.role == "agent").id == "bot"  # agents and systems keep their ids


def test_redaction_is_stable_and_respects_allowlists():
    a, b = redact(_trace()), redact(_trace())
    assert a.model_dump_json() == b.model_dump_json()  # same input, same pseudonyms: threads stay linkable
    keep = redact(_trace(), RedactionPolicy(keep_emails=[CANARIES["email"]]))  # e.g. a world's synthetic addresses
    assert CANARIES["email"] in keep.events[0].text
    t = _trace()
    t.events[2].text = "ship it before Project Falcon launches"
    assert redact(t, RedactionPolicy(terms=["project falcon"])).events[2].text == "ship it before <redacted> launches"
    other = redact(_trace(), RedactionPolicy(salt="tenant-2"))
    assert other.actors[0].id != a.actors[0].id  # per-tenant salt


def test_file_drop_csv_and_jsonl(tmp_path):
    (tmp_path / "a.csv").write_text("case,time,actor,kind,type,text,object,writes,id,reply_to\n"
                                    "T1,2026-01-01T10:00:00Z,alice,action,status_change,opened,ticket:T1,ticket:T1.status,1,\n"
                                    "T1,2026-01-01T11:00:00Z,bob,message,comment,please fix,ticket:T1,,2,1\n"
                                    "T2,2026-01-02T09:00:00Z,alice,action,status_change,opened,ticket:T2,ticket:T2.status;ticket:T2.owner,3,\n")
    (tmp_path / "b.jsonl").write_text(json.dumps({"case": "T1", "time": "2026-01-01T12:00:00Z", "actor": "agent-7", "actor_kind": "agent",
                                                  "kind": "action", "type": "resolve", "writes": ["ticket:T1.status"], "id": "4", "reply_to": "2"}) + "\n")
    c = FileDropConnector(FieldMap(family="toy/ticket"))
    refs = list(c.discover({"glob": str(tmp_path / "*")}))
    assert [r.key for r in refs] == ["T1", "T2"]
    t1 = c.normalize(c.fetch(refs[0]))
    assert [e.type for e in t1.events] == ["status_change", "comment", "resolve"]
    assert t1.events[1].reply_to == 0 and t1.events[2].reply_to == 1
    assert t1.events[0].data["writes"] == ["ticket:T1.status"] and t1.events[2].data["writes"] == ["ticket:T1.status"]
    assert {a.id: a.kind for a in t1.actors} == {"alice": "human", "bob": "human", "agent-7": "agent"}
    t2 = c.normalize(c.fetch(refs[1]))
    assert t2.events[0].data["writes"] == ["ticket:T2.status", "ticket:T2.owner"]


def test_ingest_is_idempotent(tmp_path):
    (tmp_path / "log.jsonl").write_text("\n".join(json.dumps({"case": "C1", "time": f"2026-01-01T0{i}:00:00Z", "actor": "u", "type": "x", "text": f"step {i}"}) for i in range(3)))
    c = FileDropConnector()
    ts = TraceStore(Store(tmp_path / "store"))
    refs = list(c.discover({"glob": str(tmp_path / "*.jsonl")}))
    k1 = ingest(c, refs, ts)
    k2 = ingest(FileDropConnector(), list(FileDropConnector().discover({"glob": str(tmp_path / "*.jsonl")})), ts)
    assert k1 == k2 and len(ts.store.list_docs("trace")) == 1


def test_sql_audit_log(tmp_path):
    db = tmp_path / "erp.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE audit(id INTEGER, invoice TEXT, at TEXT, who TEXT, op TEXT, table_name TEXT, row_id TEXT, detail TEXT)")
    con.executemany("INSERT INTO audit VALUES (?,?,?,?,?,?,?,?)", [
        (1, "INV-1", "2026-02-01T09:00:00Z", "clerk", "insert", "invoices", "INV-1", "created"),
        (2, "INV-1", "2026-02-01T10:00:00Z", "approver", "update", "invoices", "INV-1", "approved"),
        (3, "INV-2", "2026-02-02T09:00:00Z", "clerk", "insert", "invoices", "INV-2", None),
    ])
    con.commit()
    con.close()
    fm = FieldMap(case="invoice", time="at", actor="who", type="op", text="detail", kind="kind", id="id", family="ap/invoice")
    c = SqlAuditLogConnector(fm)
    scope = {"dsn": str(db), "query": "SELECT * FROM audit"}
    refs = list(c.discover(scope))
    assert [r.key for r in refs] == ["INV-1", "INV-2"]
    t = SqlAuditLogConnector(fm).normalize(SqlAuditLogConnector(fm).fetch(refs[0]))  # a fresh connector re-queries
    assert [(e.kind, e.type, e.data["writes"]) for e in t.events] == [("state", "insert", ["invoices:INV-1"]), ("state", "update", ["invoices:INV-1"])]
    with pytest.raises((ValueError, sqlite3.OperationalError)):  # read-only: a connector never writes the system of record (M3 audit: refused before it runs)
        from crucible.connectors.sql_audit_log import _query
        _query(str(db), "DELETE FROM audit")


def _attr(k, v):
    key = {str: "stringValue", int: "intValue", bool: "boolValue", float: "doubleValue"}[type(v)]
    return {"key": k, "value": {key: str(v) if key == "intValue" else v}}


def test_otel_import(tmp_path, monkeypatch):
    t0 = 1_760_000_000_000_000_000
    span = lambda sid, parent, start, attrs, name, status=None: {
        "traceId": "abc", "spanId": sid, "parentSpanId": parent, "name": name, "startTimeUnixNano": str(start), "endTimeUnixNano": str(start + 5_000_000),
        "attributes": [_attr(k, v) for k, v in attrs.items()], **({"status": status} if status else {})}
    doc = {"resourceSpans": [{"resource": {"attributes": [_attr("service.name", "invoice-agent")]}, "scopeSpans": [{"spans": [
        span("s1", "", t0, {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "ap-agent"}, "invoke_agent ap-agent"),
        span("s2", "s1", t0 + 1, {"gen_ai.operation.name": "chat", "gen_ai.request.model": "m-1", "gen_ai.usage.input_tokens": 120, "gen_ai.agent.name": "ap-agent"}, "chat m-1"),
        span("s3", "s1", t0 + 2, {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": "post_invoice", "gen_ai.tool.call.id": "c1",
                                  "gen_ai.tool.call.arguments": '{"id": "INV-1"}', "gen_ai.agent.name": "ap-agent"}, "execute_tool post_invoice",
             {"code": 2, "message": "duplicate invoice"}),
        span("s4", "s1", t0 + 3, {"db.system": "postgresql"}, "SELECT invoices"),
    ]}]}]}
    (tmp_path / "export.json").write_text(json.dumps(doc))
    monkeypatch.chdir(tmp_path)
    c = OtelConnector(family="ap/invoice")
    refs = list(c.discover({"glob": "*.json"}))
    t = OtelConnector().normalize(OtelConnector().fetch(refs[0]))
    # time order: the tool's result lands at its end time, after the SELECT span that started meanwhile (M3 audit)
    assert [e.type for e in t.events] == ["agent_invocation", "llm_call", "tool_call", "span", "tool_result"]
    assert t.events[1].data["model"] == "m-1" and t.events[1].data["usage"] == {"input_tokens": 120}
    assert t.events[4].reply_to == 2 and t.events[4].data["error"] == "duplicate invoice"
    assert t.events[2].data["parent"] == "s1" and t.family == "ap/invoice"


# ---------------- capture proxy (fake upstreams: no network, no API) ----------------

def _fake_upstream(seen: list[dict]):
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    async def chat(request):
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        n = len(body["messages"])
        if body.get("stream"):
            chunks = [{"id": "r1", "model": "fake-1", "choices": [{"index": 0, "delta": {"content": "Let me "}}]},
                      {"id": "r1", "model": "fake-1", "choices": [{"index": 0, "delta": {"content": "edit."}}]},
                      {"id": "r1", "model": "fake-1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "c2", "function": {"name": "write_file", "arguments": '{"path": "a.'}}]}}]},
                      {"id": "r1", "model": "fake-1", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": 'py"}'}}]}, "finish_reason": "tool_calls"}],
                      "usage": {"prompt_tokens": 10, "completion_tokens": 5}}]
            return StreamingResponse(iter([f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]), media_type="text/event-stream")
        msg = {"role": "assistant", "content": f"turn {n}", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"cmd": "ls"}'}}]}
        return JSONResponse({"id": f"r{n}", "model": "fake-1", "choices": [{"index": 0, "message": msg, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": n}})

    async def messages(request):
        body = await request.json()
        seen.append({"headers": dict(request.headers), "body": body})
        if body.get("stream"):
            evs = [{"type": "message_start", "message": {"id": "m1", "model": "fake-claude", "usage": {"input_tokens": 7}}},
                   {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                   {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "plan"}},
                   {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
                   {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Editing now"}},
                   {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "tu1", "name": "Edit", "input": {}}},
                   {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"file_path": "src/k'}},
                   {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '.cu"}'}},
                   {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 9}},
                   {"type": "message_stop"}]
            return StreamingResponse(iter([f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in evs]), media_type="text/event-stream")
        return JSONResponse({"id": "m2", "type": "message", "role": "assistant", "model": "fake-claude", "content": [{"type": "text", "text": "Done."}],
                             "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 1}})

    return Starlette(routes=[Route("/v1/chat/completions", chat, methods=["POST"]), Route("/v1/messages", messages, methods=["POST"])])


def test_capture_proxy_openai_and_anthropic(tmp_path):
    seen: list[dict] = []
    upstream_key = "upstream-" + "k" * 20
    proxy = CaptureProxy(tmp_path / "cap", openai=Upstream("http://up/v1", upstream_key), anthropic=Upstream("http://up/v1", upstream_key),
                         transport=httpx.ASGITransport(app=_fake_upstream(seen)))
    with pytest.raises(ValueError):
        proxy.register("../escape")
    k_oa, k_an = proxy.register("oa-session"), proxy.register("an-session")

    async def run():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=proxy.app()), base_url="http://proxy") as c:
            assert (await c.post("/v1/chat/completions", json={"model": "x", "messages": []}, headers={"authorization": "Bearer nope"})).status_code == 401
            h = {"authorization": f"Bearer {k_oa}"}
            m1 = [{"role": "system", "content": "be careful"}, {"role": "user", "content": "fix the kernel"}]
            r1 = (await c.post("/v1/chat/completions", json={"model": "x", "messages": m1}, headers=h)).json()
            await c.post("/v1/chat/completions", json={"model": "x", "messages": m1}, headers=h)  # client retry: same body
            m2 = [*m1, r1["choices"][0]["message"], {"role": "tool", "tool_call_id": "c1", "content": "a.py b.py"},
                  {"role": "user", "content": "[Request interrupted by user] use b.py"}]
            r2 = await c.post("/v1/chat/completions", json={"model": "x", "messages": m2, "stream": True}, headers=h)
            assert "data: [DONE]" in r2.text
            ha = {"x-api-key": k_an, "anthropic-version": "2023-06-01"}
            a1 = {"model": "c", "system": "sys prompt", "max_tokens": 50, "stream": True, "messages": [{"role": "user", "content": "port it"}]}
            await c.post("/v1/messages", json=a1, headers=ha)
            a2 = {**a1, "stream": False, "messages": [*a1["messages"], {"role": "assistant", "content": [{"type": "tool_use", "id": "tu1", "name": "Edit", "input": {}}]},
                                                      {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "tu1", "content": "The user doesn't want to proceed with this tool use.", "is_error": True},
                                                                                   {"type": "text", "text": "no, keep the old API"}]}]}
            assert (await c.post("/v1/messages", json=a2, headers=ha)).json()["content"][0]["text"] == "Done."

    asyncio.run(run())
    # the agent's session key never reaches the upstream; the proxy's key does
    assert all(upstream_key in (s["headers"].get("authorization", "") + s["headers"].get("x-api-key", "")) for s in seen)
    assert not any(k_oa in json.dumps(s["headers"]) or k_an in json.dumps(s["headers"]) for s in seen)
    assert len((tmp_path / "cap" / "oa-session.jsonl").read_text().splitlines()) == 3  # the retry is recorded ...

    c = CaptureLogConnector(family="coding/live_agent")
    refs = {r.key: r for r in c.discover({"dir": tmp_path / "cap"})}
    oa = c.normalize(c.fetch(refs["oa-session"]))  # ... and collapsed to the attempt the agent saw
    assert [e.type for e in oa.events] == ["system_message", "user_message", "llm_call", "tool_call", "tool_result", "interrupt", "llm_call", "tool_call"]
    assert oa.events[4].reply_to == 3 and oa.events[7].data["writes"] == ["file:a.py"] and oa.events[6].text == "Let me edit."
    assert oa.events[6].data["usage"] == {"prompt_tokens": 10, "completion_tokens": 5}
    an = c.normalize(c.fetch(refs["an-session"]))
    assert [e.type for e in an.events] == ["system_message", "user_message", "llm_call", "tool_call", "tool_result", "user_message", "llm_call"]
    assert an.events[2].data["reasoning"] == "plan" and an.events[3].data["writes"] == ["file:src/k.cu"]
    assert an.events[4].reply_to == 3 and an.events[4].data.get("rejected") and an.events[4].data.get("error")


def test_capture_connector_marks_context_reset(tmp_path):
    rows = [{"api": "openai", "digest": "d1", "request": {"messages": [{"role": "user", "content": "a"}]}, "response": {"choices": [{"message": {"content": "x"}}]}},
            {"api": "openai", "digest": "d2", "request": {"messages": [{"role": "user", "content": "fresh start"}]}, "response": {"choices": [{"message": {"content": "y"}}]}}]
    (tmp_path / "s.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    c = CaptureLogConnector()
    t = c.normalize(c.fetch(SourceRef(connector="capture_proxy", key="s", meta={"path": str(tmp_path / "s.jsonl")})))
    assert [e.type for e in t.events] == ["user_message", "llm_call", "context_reset", "user_message", "llm_call"]


def test_connectors_satisfy_protocol():
    from crucible.connectors.base import Connector

    for c in (FileDropConnector(), SqlAuditLogConnector(), OtelConnector(), CaptureLogConnector()):
        assert isinstance(c, Connector)
    assert RawRecord(ref=SourceRef(connector="x", key="k"), payload={}).privacy == "internal"
