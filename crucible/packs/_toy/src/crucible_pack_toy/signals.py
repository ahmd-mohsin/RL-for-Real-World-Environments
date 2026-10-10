"""Support-ticket events -> generic expert signals (M4 mapper for the ``toy/support_ticket`` family).

| ticket event | signal | grades |
|---|---|---|
| ``policy_flag`` (a supervisor flags a broken rule) | ``violation`` -1, ``rule`` in meta | the last write to the flagged object |
| ``refund_reversed`` | ``revert`` -1 | the refund it reverses |
| ``supervisor_edit`` (corrected resolution text) | ``correction`` -1 (the reply it replaces is the wrong half) | the agent reply it corrects |
| ``reopened`` by a customer reply | ``rework`` -1 | the last status write |
| ``reassigned`` | ``escalation`` -1 | the last assignee write |
| ``sla_breach`` | ``sla_breach`` -1, ``sla`` in meta | the trace's last write to the object |
| ``supervisor_approval`` | ``approval`` +1 | the approved object |
| ``csat`` | ``outcome``-like ``check_result`` (score >= 4: +1, <= 2: -1) | the whole trace |

Rule names come from the event when the connector kept them, else from the text (the stored traces keep text and
writes only).
"""

from __future__ import annotations

import re

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace
from crucible.signals.extract import signal_id

RULES = [("refund_over_50_needs_approval", r"refund above 50|without supervisor approval"), ("identity_check_before_deletion", r"identity check"),
         ("retention_offer_before_closure", r"retention offer"), ("pro_rata_annual", r"pro rata")]
MAP = {"policy_flag": ("violation", -1), "refund_reversed": ("revert", -1), "supervisor_edit": ("correction", -1), "reopened": ("rework", -1),
       "reassigned": ("escalation", -1), "sla_breach": ("sla_breach", -1), "supervisor_approval": ("approval", 1)}


def rule_of(text: str, data: dict) -> str | None:
    if data.get("rule"):
        return str(data["rule"])
    return next((r for r, pat in RULES if re.search(pat, text or "", re.IGNORECASE)), None)


class ToyTicketSignalMapper:
    def map(self, trace: Trace) -> list[Signal]:
        out: list[Signal] = []
        for e in trace.events:
            if e.type in MAP:
                t, pol = MAP[e.type]
                meta = {"event": e.type}
                if e.type == "policy_flag":
                    meta["rule"] = rule_of(e.text or "", e.data)
                if e.type == "sla_breach":
                    meta["sla"] = e.data.get("sla") or ("reopen_response_4h" if "reopen" in (e.text or "").lower() else "first_response_4h")
                if e.type == "supervisor_edit":
                    meta["exclude_self"] = True
                out.append(Signal(id=signal_id(trace.trace_id, t, e.seq), type=t, polarity=pol, source_event=e.seq, actor=e.actor, target=e.target,  # type: ignore[arg-type]
                                  text=(e.text or "")[:500] or None, meta=meta))
            elif e.type == "csat":
                score = e.data.get("score")
                if score is None:
                    m = re.search(r"\b([1-5])\b", e.text or "")
                    score = int(m.group(1)) if m else None
                if score is not None and score != 3:
                    out.append(Signal(id=signal_id(trace.trace_id, "check_result", e.seq, "csat"), type="check_result", polarity=1 if int(score) >= 4 else -1,
                                      source_event=e.seq, actor=e.actor, text=(e.text or "")[:500] or None, meta={"event": "csat", "score": score}))
        return out
