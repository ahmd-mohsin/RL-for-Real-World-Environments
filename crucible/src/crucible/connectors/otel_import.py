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
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from crucible.connectors.agent_events import annotate_probes
from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

LLM_OPS = {"chat", "text_completion", "generate_content"}
AGENT_OPS = {"invoke_agent", "create_agent"}


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
    return datetime.fromtimestamp(int(nanos) / 1e9, tz=UTC).isoformat().replace("+00:00", "Z")


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
        for p in sorted(Path().glob(pattern)):
            text = p.read_text()
            docs = [json.loads(text)] if text.lstrip().startswith("{") and "\n{" not in text.strip() else [json.loads(x) for x in text.splitlines() if x.strip()]
            for doc in docs:
                for res, sp in spans_of(doc):
                    out.setdefault(sp["traceId"], []).append((res, sp))
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
        spans = sorted(raw.payload["spans"], key=lambda x: int(x["span"].get("startTimeUnixNano") or 0))
        actors: dict[str, TraceActor] = {"system": TraceActor(id="system", role="system", kind="system")}
        events: list[TraceEvent] = []
        for item in spans:
            res, sp = item["resource"], item["span"]
            a = attrs(sp.get("attributes"))
            op = a.get("gen_ai.operation.name")
            who = str(a.get("gen_ai.agent.name") or res.get("service.name") or "agent")
            actors.setdefault(who, TraceActor(id=who, role="agent", kind="agent", handle=who))
            status = sp.get("status") or {}
            base = {"span_id": sp.get("spanId"), "parent": sp.get("parentSpanId") or None, "name": sp.get("name"),
                    "duration_ms": round((int(sp.get("endTimeUnixNano") or 0) - int(sp.get("startTimeUnixNano") or 0)) / 1e6, 3),
                    **({"error": status.get("message") or "error"} if status.get("code") in (2, "STATUS_CODE_ERROR") else {})}
            t = _iso(sp.get("startTimeUnixNano"))
            ev_attrs = {e.get("name"): attrs(e.get("attributes")) for e in sp.get("events", [])}
            if op == "execute_tool":
                tool = a.get("gen_ai.tool.name") or sp.get("name")
                args = a.get("gen_ai.tool.call.arguments")
                seq = len(events)
                events.append(TraceEvent(seq=seq, time=t, kind="action", actor=who, type="tool_call", text=f"{tool}({_text(args, 200) or ''})",
                                         data={**base, "tool": tool, "call_id": a.get("gen_ai.tool.call.id"), "arguments": _text(args)}))
                result = a.get("gen_ai.tool.call.result")
                if result is not None or base.get("error"):
                    events.append(TraceEvent(seq=len(events), time=_iso(sp.get("endTimeUnixNano")) or t, kind="observation", actor="system", type="tool_result",
                                             text=_text(result) or base.get("error"), reply_to=seq, data={"call_id": a.get("gen_ai.tool.call.id"), **({"error": base["error"]} if base.get("error") else {})}))
            elif op in LLM_OPS:
                out = a.get("gen_ai.output.messages") or ev_attrs.get("gen_ai.choice") or a.get("gen_ai.completion")
                usage = {k.removeprefix("gen_ai.usage."): v for k, v in a.items() if k.startswith("gen_ai.usage.")}
                events.append(TraceEvent(seq=len(events), time=t, kind="message", actor=who, type="llm_call", text=_text(out) or sp.get("name"),
                                         data={**base, "model": a.get("gen_ai.request.model") or a.get("gen_ai.response.model"), "usage": usage,
                                               "input": _text(a.get("gen_ai.input.messages") or ev_attrs.get("gen_ai.user.message"))}))
            elif op in AGENT_OPS:
                events.append(TraceEvent(seq=len(events), time=t, kind="action", actor=who, type="agent_invocation", text=sp.get("name"), data={**base, "op": op}))
            else:
                events.append(TraceEvent(seq=len(events), time=t, kind="observation", actor=who, type="span", text=sp.get("name"),
                                         data={**base, "attributes": {k: v for k, v in a.items() if not k.startswith("gen_ai.")}}))
        probes = annotate_probes(events)
        return Trace(trace_id=f"otel:{raw.ref.key}", family=raw.ref.meta.get("family"), source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                     actors=list(actors.values()), events=events, outcome=TraceOutcome(details={"probes": probes} if probes["attempted"] else {}))
