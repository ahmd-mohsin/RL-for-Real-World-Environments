"""Outcome labels for pull requests (DESIGN §5.2): merged and kept for 30 days without a revert or hotfix -> good;
reverted / hotfixed within 30 days, or closed without merging -> bad; open, or merged less than 30 days before
the trace was fetched -> unknown (the window is still open; the label would be a guess)."""

from __future__ import annotations

from datetime import datetime, timedelta

from crucible.core.schema.trace import Trace, TraceOutcome

WINDOW = timedelta(days=30)


class PullRequestOutcome:
    def label(self, trace: Trace) -> TraceOutcome:
        details = dict(trace.outcome.details)
        merged = next((e for e in trace.events if e.type == "merged"), None)
        reverted = [e.seq for e in trace.events if e.type == "post_merge_commit" and e.data.get("is_revert")]
        hotfixed = [e.seq for e in trace.events if e.type == "post_merge_commit" and e.data.get("fixlike") and e.data.get("touches_seq")]
        if reverted:
            return TraceOutcome(label="bad", label_source="reverted within 30 days of the merge", details={**details, "evidence": reverted})
        if hotfixed:
            return TraceOutcome(label="bad", label_source="hotfixed within 30 days of the merge", details={**details, "evidence": hotfixed})
        if any(e.type == "closed_unmerged" for e in trace.events):
            return TraceOutcome(label="bad", label_source="closed without merging", details=details)
        if merged is None:
            return TraceOutcome(label="unknown", label_source="still open", details=details)
        now = datetime.fromisoformat(trace.source.fetched_at) if trace.source.fetched_at else None
        if now is None or now - datetime.fromisoformat(merged.time) < WINDOW:
            return TraceOutcome(label="unknown", label_source="merged less than 30 days ago", details=details)
        return TraceOutcome(label="good", label_source="merged + 30 days without revert or hotfix", details=details)
