"""Support-ticket events -> generic expert signals (M4 mapper for the ``toy/support_ticket`` family).

| ticket event | signal | grades |
|---|---|---|
| ``policy_flag`` (a supervisor flags a broken rule) | ``violation`` -1, ``rule`` in meta | the last write to the flagged object |
| ``refund_reversed`` | ``revert`` -1 | the refund it reverses |
| ``supervisor_edit`` (corrected resolution text) | ``correction`` -1 (the reply it replaces is the wrong half) | the agent reply it corrects |
| ``reopened`` by a customer reply | ``rework`` -1 | the last status write |
| ``reassigned`` | ``escalation`` -1 | the last answer given before it (none if there was none) |
| ``sla_breach`` | ``sla_breach`` -1, ``sla`` in meta | the trace's last write to the object |
| ``supervisor_approval`` | ``approval`` +1 | the approved object |
| ``customer_reply`` | negative about the answer: ``rejection`` -1; positive: ``approval`` +1; a question: none | the agent reply it answers |
| ``csat`` | ``outcome`` (trace level; score >= 4: +1, <= 2: -1, 3: 0) | the whole trace |

Rule names come from the event when the connector kept them, else from the text (the stored traces keep text and
writes only).

M4 audit (2026-10-12): the hand labels (written before this mapper) were never scored; scored, the mapper missed every
customer reply, typed CSAT as a ``check_result`` (the taxonomy's ``outcome`` is the customer's verdict on the whole
ticket), graded escalations on the queue's routing rather than the answer that failed, read CSAT "3" (a string) as
negative and crashed on "4.5", took the first digit of "Took 2 days, but rated 5/5", and its rule and SLA fallbacks
mislabelled "Refund of 75.00 exceeds the 50 limit" and "Resolution SLA (24h) breached".
"""

from __future__ import annotations

import re

from crucible.core.schema.signal import Signal, SignalTarget
from crucible.core.schema.trace import Trace, TraceEvent
from crucible.signals.extract import signal_id

RULES = [("refund_over_50_needs_approval", r"refund (?:above|over) \$?50\b|exceeds? the \$?50\b|refund\b.{0,40}\b(?:without|lacks?) (?:supervisor )?approval|without supervisor approval"),
         ("identity_check_before_deletion", r"(?:without|skipped|no|missing)\b.{0,30}\b(?:identity|id) (?:check|verification)|(?:identity|id) (?:check|verification) (?:was )?(?:skipped|missing|not done)"),
         ("retention_offer_before_closure", r"retention offer"), ("pro_rata_annual", r"pro[- ]rata")]
SLAS = [("reopen_response_4h", r"\breopen\w*\b.*\b(?:4 ?h|4 hours|waited)"), ("first_response_4h", r"\bfirst response\b"),
        ("resolution_24h", r"\bresolution\b.*\b24 ?h")]
MAP = {"policy_flag": ("violation", -1), "refund_reversed": ("revert", -1), "supervisor_edit": ("correction", -1), "reopened": ("rework", -1),
       "reassigned": ("escalation", -1), "sla_breach": ("sla_breach", -1), "supervisor_approval": ("approval", 1)}
NEGATIVE = re.compile(r"\b(?:still|not|no|nope|didn'?t|doesn'?t|don'?t|never|wrong|again|isn'?t|can'?t|cannot|won'?t|same problem|broken|useless)\b", re.IGNORECASE)
POSITIVE = re.compile(r"\b(?:thanks?|thank you|works?|worked|perfect|great|got it|received|cheers|solved|fixed|did the trick|back in|all good)\b", re.IGNORECASE)
CSAT_SCALE = (1, 5)


def rule_of(text: str, data: dict) -> str | None:
    if data.get("rule"):
        return str(data["rule"])
    return next((r for r, pat in RULES if re.search(pat, text or "", re.IGNORECASE)), None)


def sla_of(text: str, data: dict) -> str | None:
    if data.get("sla"):
        return str(data["sla"])
    return next((n for n, pat in SLAS if re.search(pat, text or "", re.IGNORECASE)), None)


def csat_score(text: str, data: dict) -> float | None:
    """The score on the declared 1-5 scale: the row's value (numeric text accepted), else an explicit "N/5" or
    "rated N" in the text; anything else, or off the scale, is no score."""
    raw = data.get("score")
    if raw is None:
        m = re.search(r"\b([1-5](?:\.\d)?)\s*/\s*5\b|\brated\s+([1-5](?:\.\d)?)\b", text or "", re.IGNORECASE)
        raw = (m.group(1) or m.group(2)) if m else None
    try:
        score = float(raw) if raw is not None else None
    except (TypeError, ValueError):
        return None
    return score if score is not None and CSAT_SCALE[0] <= score <= CSAT_SCALE[1] else None


def reply_sentiment(text: str) -> str | None:
    t = (text or "").strip()
    if not t or t.endswith("?"):
        return None  # a question is no verdict on the answer
    neg, pos = bool(NEGATIVE.search(t)), bool(POSITIVE.search(t))
    return "negative" if neg and not pos else "positive" if pos and not neg else None


class ToyTicketSignalMapper:
    def map(self, trace: Trace) -> list[Signal]:
        out: list[Signal] = []
        replies = [e for e in trace.events if e.type in ("agent_reply", "supervisor_edit")]

        def last_answer(before: int) -> TraceEvent | None:
            return next((e for e in reversed(replies) if e.seq < before), None)

        def emit(e: TraceEvent, t: str, pol: int, meta: dict, graded: TraceEvent | None = None, *, explicit: bool = False, extra: str = "") -> None:
            if explicit:  # this rule names the step itself; the generic last-writer rule is not asked
                meta = {**meta, **({} if graded else {"attribution": "unattributed", "reason": "no answer before it"})}
            out.append(Signal(id=signal_id(trace.trace_id, t, e, extra), type=t, polarity=pol, source_event=e.seq, actor=e.actor,  # type: ignore[arg-type]
                              grades=[graded.seq] if graded else [], target=(graded.target if graded else None) if explicit else e.target,
                              text=(e.text or "")[:500] or None, meta=meta))

        for e in trace.events:
            if e.type == "reassigned":
                emit(e, "escalation", -1, {"event": e.type}, last_answer(e.seq), explicit=True)
            elif e.type in MAP:
                t, pol = MAP[e.type]
                meta: dict = {"event": e.type}
                if e.type == "policy_flag":
                    meta["rule"] = rule_of(e.text or "", e.data)
                if e.type == "sla_breach":
                    meta["sla"] = sla_of(e.text or "", e.data)
                if e.type == "supervisor_edit":
                    meta["exclude_self"] = True
                emit(e, t, pol, meta)
            elif e.type == "customer_reply":
                mood = reply_sentiment(e.text or "")
                if mood:
                    answered = next((x for x in replies if x.seq == e.reply_to), None) or last_answer(e.seq)
                    emit(e, "rejection" if mood == "negative" else "approval", -1 if mood == "negative" else 1, {"event": e.type, "sentiment": mood}, answered, explicit=True)
            elif e.type == "csat":
                score = csat_score(e.text or "", e.data)
                if score is not None:  # 3 is a verdict too: a neutral one
                    out.append(Signal(id=signal_id(trace.trace_id, "outcome", e, "csat"), type="outcome", polarity=1 if score >= 4 else -1 if score <= 2 else 0,
                                      source_event=e.seq, actor=e.actor, target=SignalTarget(object=e.target.object if e.target else None),
                                      text=(e.text or "")[:500] or None, meta={"event": "csat", "score": score, "scale": list(CSAT_SCALE)}))
        return out
