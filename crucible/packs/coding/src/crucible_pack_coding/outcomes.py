"""Outcome labels for pull requests (DESIGN §5.2): merged and kept for 30 days without a revert or hotfix -> good;
reverted / hotfixed within 30 days, or closed without merging -> bad; open, or merged less than 30 days before
the trace was fetched -> unknown (the window is still open; the label would be a guess).

M4 audit (2026-10-12): the M3 audit took the fetch time out of PR traces (it changed the hash on every read), and this
labeler read it, so every merged PR became "unknown" ("merged less than 30 days ago" for one merged 280 days earlier)
and the M4 gate's outcome agreement fell to 0.21. The connector now records what it observed of the window
(``outcome.details.window``: ``checked``, ``until``, ``closed``), a fact that changes once, when the window closes.
Times are parsed as instants (a naive fetch time raised against the API's Z times, a missing merge time raised), a
revert or hotfix counts only inside the window, and a merge whose later history was never read is unknown, not good.
"""

from __future__ import annotations

from datetime import timedelta

from crucible.connectors.base import parse_time
from crucible.core.schema.trace import Trace, TraceOutcome

WINDOW = timedelta(days=30)


class PullRequestOutcome:
    def label(self, trace: Trace) -> TraceOutcome:
        details = dict(trace.outcome.details)

        def unknown(why: str) -> TraceOutcome:
            return TraceOutcome(label="unknown", label_source=why, details=details)

        def inside(e) -> bool:
            d = e.data.get("days_after_merge")
            return d is None or d <= WINDOW.days

        merged = next((e for e in trace.events if e.type == "merged"), None)
        reverted = [e.seq for e in trace.events if e.type == "post_merge_commit" and e.data.get("is_revert") and inside(e)]
        hotfixed = [e.seq for e in trace.events if e.type == "post_merge_commit" and e.data.get("fixlike") and e.data.get("touches_seq") and inside(e)]
        if reverted:
            return TraceOutcome(label="bad", label_source="reverted within 30 days of the merge", details={**details, "evidence": reverted})
        if hotfixed:
            return TraceOutcome(label="bad", label_source="hotfixed within 30 days of the merge", details={**details, "evidence": hotfixed})
        if any(e.type == "closed_unmerged" for e in trace.events):
            return TraceOutcome(label="bad", label_source="closed without merging", details=details)
        if merged is None:
            return unknown("still open")
        merged_at = parse_time(merged.time)
        if merged_at is None:
            return unknown("merge time unknown")
        window = details.get("window")
        if isinstance(window, dict):
            if not window.get("checked"):
                return unknown("history after the merge not available")
            if not window.get("closed"):
                return unknown("merged less than 30 days before the history was read")
            return TraceOutcome(label="good", label_source="merged + 30 days without revert or hotfix", details=details)
        now = parse_time(trace.source.fetched_at)  # traces stored before the window fact
        if now is None:
            return unknown("observation time unknown")
        if now - merged_at < WINDOW:
            return unknown("merged less than 30 days ago")
        return TraceOutcome(label="good", label_source="merged + 30 days without revert or hotfix", details=details)
