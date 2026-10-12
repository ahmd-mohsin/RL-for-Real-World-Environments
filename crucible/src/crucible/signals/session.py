"""Signals every agent session carries, whatever the pack (DESIGN §5.2, "User interrupt or rejected edit -> negative
on that tool call"). ``extract`` runs these rules beside the pack's mapper, so a Claude Code, mini-swe-agent, OTel,
capture-proxy or World trace yields its human interrupts and rejections without a mapper of its own (M4 audit: no
family outside the packs had one, and the importers' ``interrupt`` / ``rejected`` marks reached nothing).

  ``interrupt`` (a person stopped the agent)            -> ``interrupt`` -1 on the agent's last action before it
  a tool result marked ``rejected`` (permission denied)  -> ``rejection`` -1 on the call it answers
  a user turn of type ``rejection`` (mini-swe-agent)     -> ``rejection`` -1 on the agent's last action before it

A rejected command never ran, so it wrote nothing: the proposing call is graded directly, not the last writer of
something (the last-writer rule would grade an earlier, unrelated edit).
"""

from __future__ import annotations

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceEvent

BY = "mapper:session"


class SessionSignalMapper:
    def map(self, trace: Trace) -> list[Signal]:
        from crucible.signals.extract import signal_id

        kinds = {a.id: a.kind for a in trace.actors}
        by_seq = {e.seq: e for e in trace.events}

        def is_agent_action(e: TraceEvent) -> bool:
            if e.kind != "action":
                return False
            k = kinds.get(e.actor or "")
            return k == "agent" if k is not None else e.actor not in (None, "user", "system")

        def last_action(before: int) -> TraceEvent | None:
            return next((e for e in reversed(trace.events) if e.seq < before and is_agent_action(e)), None)

        out: list[Signal] = []

        def emit(e: TraceEvent, type_: str, graded: TraceEvent | None, rule: str) -> None:
            meta = {"by": BY, "rule": rule, "automated": False}
            if graded is None:
                meta |= {"attribution": "unattributed", "reason": "no_agent_action_before"}
            out.append(Signal(id=signal_id(trace.trace_id, type_, e), type=type_, polarity=-1, grades=[graded.seq] if graded else [],  # type: ignore[arg-type]
                              source_event=e.seq, actor=e.actor, target=graded.target if graded else None, text=(e.text or "")[:500] or None, meta=meta))

        for e in trace.events:
            if e.type == "interrupt":
                emit(e, "interrupt", last_action(e.seq), "interrupt")
            elif e.type == "rejection" and e.kind == "message":
                emit(e, "rejection", last_action(e.seq), "rejected_command")
            elif e.type == "tool_result" and e.data.get("rejected"):
                call = by_seq.get(e.reply_to) if e.reply_to is not None else None
                emit(e, "rejection", call if call is not None and call.kind == "action" else last_action(e.seq), "rejected_tool_use")
        return out
