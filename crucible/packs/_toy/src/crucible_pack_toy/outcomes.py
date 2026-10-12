"""Outcome labels for support tickets (the toy M4 guideline): CSAT <= 2, or reopened without a later resolution -> bad;
resolved with CSAT >= 4 or no CSAT -> good; CSAT 3 -> unknown. M4 audit: the rule lived only in the M5/M6 pipeline,
and no ``outcome_labeler`` was registered, so every toy trace stayed ``unknown`` (outcome agreement 0.08) and toy
reliability never saw an outcome."""

from __future__ import annotations

from crucible.core.schema.trace import Trace, TraceOutcome


def ticket_outcome(t: Trace) -> str:
    from crucible_pack_toy.signals import csat_score

    scores = [csat_score(e.text or "", e.data) for e in t.events if e.type == "csat"]
    score = scores[-1] if scores else None
    statuses = [e.data.get("status") for e in t.events if e.data.get("status")]
    last_status = statuses[-1] if statuses else None
    reopened = any(e.type == "reopened" for e in t.events)
    if (score is not None and score <= 2) or (reopened and last_status != "resolved"):
        return "bad"
    if score is not None and 2 < score < 4:
        return "unknown"
    return "good" if last_status == "resolved" else "unknown"


class TicketOutcome:
    def label(self, trace: Trace) -> TraceOutcome:
        lab = ticket_outcome(trace)
        return TraceOutcome(label=lab, label_source=f"ticket guideline: {lab}", details=dict(trace.outcome.details))  # type: ignore[arg-type]
