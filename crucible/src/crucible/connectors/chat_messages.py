"""``chat_messages``: conversations in the OpenAI chat format -> traces (M3 generic connector).

Most public agent-trajectory corpora store a run as one chat transcript: ``system`` / ``user`` / ``assistant`` (with
``tool_calls``) / ``tool`` messages (OpenHands, SWE-agent, SWE-Gym and nebius trajectory sets; any agent logged
through an OpenAI-compatible client). One transcript becomes one trace:

  system -> ``system_message``; user -> ``user_message`` / ``interrupt``; assistant text -> ``agent_message``;
  each tool call -> ``tool_call`` (``data.writes`` for file editors; answer-key probes tagged, P27);
  tool -> ``tool_result`` replying to its call (by ``tool_call_id``, else the latest unanswered call).

``messages_to_trace`` is the function the dataset adapters call; ``ChatMessagesConnector`` reads JSONL files of
``{"id": ..., "messages": [...], "outcome": ...}`` records.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.agent_events import REJECTED, annotate_probes, user_type, writes_of
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource

MAX_TEXT = 4000


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
    return content or ""


def _args(raw: Any) -> Any:
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def messages_to_trace(messages: list[dict[str, Any]], *, trace_id: str, source: TraceSource, family: str | None = None,
                      outcome: TraceOutcome | None = None, max_text: int = MAX_TEXT, keep_arguments: bool = True) -> Trace:
    """``keep_arguments``: store each call's full arguments as JSON (needed to replay edits); otherwise bounded."""
    events: list[TraceEvent] = []
    open_calls: dict[str, int] = {}
    order: list[str] = []

    def add(**kw: Any) -> int:
        events.append(TraceEvent(seq=len(events), **kw))
        return len(events) - 1

    for i, m in enumerate(messages):
        role, text = m.get("role"), _text(m.get("content"))
        base = {"turn": i}
        if role == "system":
            add(kind="message", actor="system", type="system_message", text=text[:max_text], data=base)
        elif role == "user":
            add(kind="message", actor="user", type=user_type(text), text=text[:max_text], data=base)
        elif role == "assistant":
            if text.strip():
                add(kind="message", actor="agent", type="agent_message", text=text[:max_text], data=base)
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                name, args = fn.get("name") or "", _args(fn.get("arguments"))
                arg_text = args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)
                w = writes_of(name, args)
                cid = tc.get("id") or f"call_{len(events)}"
                seq = add(kind="action", actor="agent", type="tool_call", text=f"{name}({arg_text[:200]})",
                          target=SignalTarget(app="workspace", object=w[0]) if w else None,
                          data={**base, "tool": name, "call_id": cid, "arguments": arg_text if keep_arguments else arg_text[:max_text],
                                **({"writes": w} if w else {})})
                open_calls[cid] = seq
                order.append(cid)
        elif role == "tool":
            cid = m.get("tool_call_id") or ""
            reply = open_calls.pop(cid, None)
            if reply is None and order:  # unlinked result: answers the latest unanswered call
                for c in reversed(order):
                    if c in open_calls:
                        reply = open_calls.pop(c)
                        break
            add(kind="observation", actor="system", type="tool_result", text=text[:max_text], reply_to=reply,
                data={**base, "call_id": cid, "tool": m.get("name"), "chars": len(text), **({"rejected": True} if REJECTED.search(text) else {})})
    probes = annotate_probes(events)
    out = outcome or TraceOutcome()
    if probes["attempted"]:
        out = out.model_copy(update={"details": {**out.details, "probes": probes}})
    actors = [TraceActor(id="user", role="user", kind="human"), TraceActor(id="agent", role="agent", kind="agent"),
              TraceActor(id="system", role="system", kind="system")]
    return Trace(trace_id=trace_id, family=family, source=source, actors=actors, events=events, outcome=out)


class ChatMessagesConnector:
    """scope: {"glob": "runs/*.jsonl", "family": ...}: each JSONL line {"id", "messages", "outcome"?: good|bad|unknown}."""

    name = "chat_messages"

    def __init__(self, privacy: str = "internal"):
        self.privacy = privacy

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        for p in sorted(Path().glob(scope["glob"])):
            for n, line in enumerate(p.read_text().splitlines()):
                if line.strip():
                    rid = json.loads(line).get("id") or f"{p.stem}:{n}"
                    yield SourceRef(connector=self.name, key=str(rid), meta={"path": str(p), "line": n, "family": scope.get("family")})

    def fetch(self, ref: SourceRef) -> RawRecord:
        line = Path(ref.meta["path"]).read_text().splitlines()[ref.meta["line"]]
        return RawRecord(ref=ref, payload=json.loads(line), privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        p = raw.payload
        label = p.get("outcome") if p.get("outcome") in ("good", "bad", "unknown") else "unknown"
        return messages_to_trace(p["messages"], trace_id=f"chat:{raw.ref.key}", family=raw.ref.meta.get("family"),
                                 source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy),  # type: ignore[arg-type]
                                 outcome=TraceOutcome(label=label, label_source="recorded" if label != "unknown" else None))
