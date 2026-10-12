"""``otel_import``: OpenTelemetry traces of enterprise agents (OTLP/JSON export) -> one Trace per OTel trace id.

Agents in production are increasingly instrumented with the OpenTelemetry GenAI semantic conventions
(``gen_ai.operation.name`` = chat / text_completion / execute_tool / invoke_agent, ``gen_ai.request.model``,
``gen_ai.usage.*``, ``gen_ai.tool.name`` / ``gen_ai.tool.call.id`` / arguments / result, message content in
``gen_ai.input.messages`` / ``gen_ai.output.messages`` or span events). Spans become events in start-time order:

  execute_tool -> action ``tool_call`` (+ observation ``tool_result`` when the result was recorded)
  chat / text_completion / generate_content -> message ``llm_call`` (model, token usage, output text if captured)
  invoke_agent / create_agent -> action ``agent_invocation``
  anything else -> observation ``span``

Span status ERROR is kept (``data.error``): failed tool calls are evidence M4 can use. Parent links are kept in
``data.parent`` (span ids), so the call tree survives.

OpenInference spans (``openinference.span.kind`` TOOL / LLM / AGENT: Phoenix, LangChain, Harbor's atif2otel) map the
same way. M3 audit (2026-10-12): events are ordered by time with results at their end time and parents before
children (a tool result at 9 ms came before an llm call at 2 us); repeated span events are all kept (the second
``gen_ai.user.message`` overwrote the first); a span exported twice counts once; tool calls carry ``writes``; times
keep their nanoseconds (they went through a float); absolute export paths work.
"""

from __future__ import annotations

import glob as _glob
import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.agent_events import annotate_probes, writes_of
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

LLM_OPS = {"chat", "text_completion", "generate_content"}
AGENT_OPS = {"invoke_agent", "create_agent"}
OI_KIND = {"TOOL": "execute_tool", "LLM": "chat", "AGENT": "invoke_agent"}  # OpenInference span kinds


def _value(v: dict[str, Any]) -> Any:
    for k in ("stringValue", "boolValue", "doubleValue"):
        if k in v:
            return v[k]
    if "intValue" in v:
        return int(v["intValue"])
    if "arrayValue" in v:
        return [_value(x) for x in v["arrayValue"].get("values", [])]
    if "kvlistValue" in v:
        return {x["key"]: _value(x["value"]) for x in v["kvlistValue"].get("values", [])}
    return None


def attrs(items: list[dict[str, Any]] | None) -> dict[str, Any]:
    return {a["key"]: _value(a.get("value") or {}) for a in items or []}


def _iso(nanos: str | int | None) -> str | None:
    if nanos in (None, "", "0", 0):
        return None
    n = int(nanos)  # integer arithmetic: a float lost everything below the microsecond
    return f"{datetime.fromtimestamp(n // 1_000_000_000, tz=UTC):%Y-%m-%dT%H:%M:%S}.{n % 1_000_000_000:09d}Z"


def spans_of(doc: dict[str, Any]) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    for rs in doc.get("resourceSpans", []):
        res = attrs((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans", []) + rs.get("instrumentationLibrarySpans", []):
            for sp in ss.get("spans", []):
                yield res, sp


def _text(x: Any, limit: int = 4000) -> str | None:
    if x is None:
        return None
    s = x if isinstance(x, str) else json.dumps(x, ensure_ascii=False, default=str)
    return s[:limit]


class OtelConnector:
    """scope: {"glob": "exports/*.json", "family": "..."} (OTLP/JSON, one document per file or JSONL of documents)."""

    name = "otel_import"

    def __init__(self, family: str | None = None, privacy: str = "internal"):
        self.family, self.privacy = family, privacy
        self._spans: dict[str, list[tuple[dict, dict]]] = {}

    @staticmethod
    def _load(pattern: str) -> dict[str, list[tuple[dict, dict]]]:
        out: dict[str, list[tuple[dict, dict]]] = {}
        for p in map(Path, sorted(_glob.glob(pattern, recursive=True))):  # Path().glob refused absolute patterns
            text = p.read_text()
            docs = [json.loads(text)] if text.lstrip().startswith("{") and "\n{" not in text.strip() else [json.loads(x) for x in text.splitlines() if x.strip()]
            for doc in docs:
                for res, sp in spans_of(doc):
                    spans = out.setdefault(sp["traceId"], [])
                    if not any(s.get("spanId") == sp.get("spanId") for _, s in spans):  # an exporter retry sends a span twice
                        spans.append((res, sp))
        return out

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        self._spans = self._load(scope["glob"])
        for tid in sorted(self._spans):
            yield SourceRef(connector=self.name, key=tid, meta={"family": scope.get("family", self.family), "glob": scope["glob"]})

    def fetch(self, ref: SourceRef) -> RawRecord:
        if ref.key not in self._spans:  # a ref from an earlier run: read the export again
            self._spans = self._load(ref.meta["glob"])
        return RawRecord(ref=ref, payload={"spans": [{"resource": r, "span": s} for r, s in self._spans[ref.key]]}, privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        items = raw.payload["spans"]
        by_id = {it["span"].get("spanId"): it for it in items}
        attrs_of = {sid: attrs(it["span"].get("attributes")) for sid, it in by_id.items()}

        def depth(sid: str | None, seen: int = 0) -> int:
            parent = (by_id.get(sid) or {}).get("span", {}).get("parentSpanId") if sid else None
            return 0 if not parent or parent not in by_id or seen > 64 else 1 + depth(parent, seen + 1)

        def agent_of(sp: dict[str, Any], res: dict[str, Any]) -> str:
            sid, hops = sp.get("spanId"), 0
            while sid in by_id and hops < 64:  # the nearest enclosing agent, else the service
                a = attrs_of[sid]
                name = a.get("gen_ai.agent.name") or (a.get("agent.name") if a.get("openinference.span.kind") == "AGENT" else None)
                if name:
                    return str(name)
                sid, hops = by_id[sid]["span"].get("parentSpanId"), hops + 1
            return str(res.get("service.name") or "agent")

        actors: dict[str, TraceActor] = {"system": TraceActor(id="system", role="system", kind="system")}
        staged: list[tuple[tuple, dict[str, Any], int | None]] = []  # (sort key, event fields, index of the call it answers)
        for k, item in enumerate(items):
            res, sp = item["resource"], item["span"]
            a = attrs_of.get(sp.get("spanId")) or attrs(sp.get("attributes"))
            op = a.get("gen_ai.operation.name") or OI_KIND.get(str(a.get("openinference.span.kind") or "").upper())
            who = agent_of(sp, res)
            actors.setdefault(who, TraceActor(id=who, role="agent", kind="agent", handle=who))
            status = sp.get("status") or {}
            t0, t1 = int(sp.get("startTimeUnixNano") or 0), int(sp.get("endTimeUnixNano") or 0)
            base = {"span_id": sp.get("spanId"), "parent": sp.get("parentSpanId") or None, "name": sp.get("name"), "start_ns": t0,
                    "duration_ms": round((t1 - t0) / 1e6, 3),
                    **({"error": status.get("message") or "error"} if status.get("code") in (2, "STATUS_CODE_ERROR") else {})}
            key = (t0, depth(sp.get("spanId")), k)
            ev_attrs: dict[str, list[dict[str, Any]]] = {}
            for e in sp.get("events", []):
                ev_attrs.setdefault(e.get("name"), []).append(attrs(e.get("attributes")))
            if op == "execute_tool":
                tool = a.get("gen_ai.tool.name") or a.get("tool.name") or sp.get("name")
                args = a.get("gen_ai.tool.call.arguments") or a.get("input.value") or a.get("tool.parameters")
                parsed = json.loads(args) if isinstance(args, str) and args.strip().startswith("{") else args
                w = writes_of(str(tool), parsed if isinstance(parsed, dict) else (args or {}))
                staged.append((key, {"time": _iso(t0), "kind": "action", "actor": who, "type": "tool_call", "text": f"{tool}({_text(args, 200) or ''})",
                                         "data": {**base, "tool": tool, "call_id": a.get("gen_ai.tool.call.id"), "arguments": _text(args), **({"writes": w} if w else {})}}, None))
                call_at = len(staged) - 1
                result = a.get("gen_ai.tool.call.result") if a.get("gen_ai.tool.call.result") is not None else a.get("output.value")
                if result is not None or base.get("error"):
                    staged.append(((t1 or t0, depth(sp.get("spanId")), k), {"time": _iso(t1) or _iso(t0), "kind": "observation", "actor": "system", "type": "tool_result",
                                   "text": _text(result) or base.get("error"), "data": {"call_id": a.get("gen_ai.tool.call.id"), **({"error": base["error"]} if base.get("error") else {})}},
                                   call_at))
            elif op in LLM_OPS:
                details = [x for x in ev_attrs.get("gen_ai.client.inference.operation.details", [])]
                out = (a.get("gen_ai.output.messages") or next((d.get("gen_ai.output.messages") for d in details if d.get("gen_ai.output.messages")), None)
                       or [x for x in ev_attrs.get("gen_ai.choice", [])] or a.get("gen_ai.completion") or a.get("output.value"))
                inputs = (a.get("gen_ai.input.messages") or next((d.get("gen_ai.input.messages") for d in details if d.get("gen_ai.input.messages")), None)
                          or [x for n in ("gen_ai.system.message", "gen_ai.user.message", "gen_ai.assistant.message", "gen_ai.tool.message") for x in ev_attrs.get(n, [])]
                          or a.get("input.value"))
                usage = {k2.removeprefix("gen_ai.usage."): v for k2, v in a.items() if k2.startswith("gen_ai.usage.")}
                staged.append((key, {"time": _iso(t0), "kind": "message", "actor": who, "type": "llm_call", "text": _text(out) or sp.get("name"),
                                         "data": {**base, "model": a.get("gen_ai.request.model") or a.get("gen_ai.response.model") or a.get("llm.model_name"),
                                                  "usage": usage, "input": _text(inputs)}}, None))
            elif op in AGENT_OPS:
                staged.append((key, {"time": _iso(t0), "kind": "action", "actor": who, "type": "agent_invocation", "text": sp.get("name"), "data": {**base, "op": op}}, None))
            else:
                staged.append((key, {"time": _iso(t0), "kind": "observation", "actor": who, "type": "span", "text": sp.get("name"),
                                         "data": {**base, "attributes": {k2: v for k2, v in a.items() if not k2.startswith("gen_ai.")}}}, None))
        order = sorted(range(len(staged)), key=lambda i: staged[i][0])
        seq_of = {i: n for n, i in enumerate(order)}
        events = [TraceEvent(seq=seq_of[i], **staged[i][1], reply_to=seq_of[staged[i][2]] if staged[i][2] is not None else None) for i in order]
        probes = annotate_probes(events)
        return Trace(trace_id=f"otel:{raw.ref.key}", family=raw.ref.meta.get("family"), source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=list(actors.values()), events=events, outcome=TraceOutcome(details={"probes": probes} if probes["attempted"] else {}))
