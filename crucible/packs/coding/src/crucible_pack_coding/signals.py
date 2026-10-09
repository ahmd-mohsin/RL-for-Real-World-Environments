"""Coding events -> generic expert signals (M4 mapper for the ``coding/pull_request`` family).

The decision for a root review comment combines what the reviewer *said* with what happened *afterwards*, which
the GitHub connector computes from git (blame within the PR, line tracking, final code):

    its concrete fix (``suggestion`` block or a "suggested fix" code block) is in the final code -> suggestion_accepted
    it asks for a change and the commented code (or the line next to it) changed afterwards     -> correction
    it asks for a change and nothing there changed                                              -> rejection
    a clarifying question after which the code changed                                          -> correction
    praise                                                                                       -> approval
    anything else (questions answered without change, discussion)                               -> no signal

Every step signal grades the PR commit that last wrote the commented line (blame at the comment's commit), or
no step when that line predates the PR. Replies in a thread and the author's own comments carry no signal: the
thread's root does. Reviews: APPROVED -> approval, CHANGES_REQUESTED -> rejection on the reviewed commit. CI runs
on a commit -> check_result (+1 pass, -1 fail, 0 flaky -- both outcomes for one check on one commit, the case
SWE-smith's ``harness/repair.py`` removes). After the merge (``vcs_history``): a revert -> revert, a fix-like
commit that changes lines the PR wrote -> correction (hotfix), both grading the PR commits that wrote them.
"""

from __future__ import annotations

import re

from crucible.core.schema.signal import Signal
from crucible.core.schema.trace import Trace, TraceEvent
from crucible.signals.extract import signal_id

REQUEST_CUES = (
    r"\bplease\b", r"\bshould(?:n't| not)?\b", r"\bmust\b", r"\bneeds? (?:to|a|an|the)\b", r"\bconsider\w*\b", r"\b(?:can|could|would) you\b",
    r"\bI would (?:move|use|prefer|suggest|rather|keep|remove|add|rename|change|do|go)\b", r"\bI'd (?:move|use|prefer|suggest|rather)\b",
    r"\b(?:can|could|should) (?:also )?be (?:removed|dropped|deleted|simplified|replaced|moved|made|marked|merged|combined)\b",
    r"\bnit(?:pick)?\b", r"\btypo\b", r"\bmissing\b", r"\bwrong\b", r"\bincorrect\b", r"\bbug\b", r"\bwill (?:cause|fail|break|crash|leak)\b",
    r"\bleads? to\b", r"\bundefined behaviou?r\b", r"\binconsistent\b", r"\bnot (?:be )?(?:needed|necessary|required)\b", r"\bunnecessary\b",
    r"\binstead of\b", r"\bprefer\b", r"\bwhy not\b", r"\bdon'?t\b", r"\bdo not\b", r"\bavoid\b", r"\bremove\b", r"\brename\b", r"\breplace\b",
    r"\bmove (?:this|these|it)\b", r"\badd (?:a|an|some|the)\b", r"\bswitch to\b", r"\bchange (?:this|it|these|to)\b", r"\bmisleading\b",
    r"\bwe need\b", r"\bis (?:not|n't) (?:needed|used|thread[- ]safe|correct)\b", r"\bdoes(?:n't| not) match\b", r"\bnowhere\b",
    r"\bcould it make sense\b", r"\bwouldn't it be (?:better|cleaner|simpler)\b", r"\bthe (?:same|similar) issue\b", r"\bsame (?:issue|problem|here)\b",
    r"\buse `", r"\bupdate (?:the|this)\b", r"\bguard\b",
)
REQUEST = re.compile("|".join(REQUEST_CUES), re.IGNORECASE)
CLARIFY = re.compile(r"\b(?:intentional|on purpose|why (?:is|are|do|does|did)|what (?:does|is)|is this|do we|does this|are we)\b.*\?|\?\s*$", re.IGNORECASE | re.DOTALL)
PRAISE = re.compile(r"^\s*(?:lgtm|looks good|nice|great|good catch|thanks?|thank you|perfect|👍|:\+1:)[\s!.👍]*$", re.IGNORECASE)
FAILED = {"failure", "timed_out"}
IGNORED_CI = {"neutral", "skipped", "cancelled", "action_required", "stale", None}
OVERVIEW = re.compile(r"pull request overview|wasn't able to review|reviewed \d+ out of \d+ changed files", re.IGNORECASE)


def intent(text: str) -> str:
    t = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)  # code blocks say nothing about intent
    t = re.sub(r"^>.*$", " ", t, flags=re.MULTILINE)  # quoted text is someone else's
    if PRAISE.match(t):
        return "praise"
    if REQUEST.search(t):
        return "request"
    if CLARIFY.search(t):
        return "clarify"
    return "discussion"


class CodingSignalMapper:
    """Mapper for traces from ``crucible_pack_coding.connectors.github_pr``."""

    def map(self, trace: Trace) -> list[Signal]:
        roles = {a.id: a.role for a in trace.actors}
        kinds = {a.id: a.kind for a in trace.actors}
        commit_seq = {e.data["sha"]: e.seq for e in trace.events if e.type == "commit"}
        final_commits = [e.seq for e in trace.events if e.type == "commit" and not e.data.get("superseded")]
        out: list[Signal] = []

        def sig(e: TraceEvent, type_: str, polarity: int, grades: list[int], **meta) -> None:
            out.append(Signal(id=signal_id(trace.trace_id, type_, e.seq), type=type_, polarity=polarity, grades=grades,  # type: ignore[arg-type]
                              source_event=e.seq, actor=e.actor, target=e.target, text=(e.text or "")[:500] or None,
                              meta={"automated": kinds.get(e.actor or "") != "human", **meta}))

        ci_by_name: dict[tuple[str, str], set[str]] = {}
        for e in trace.events:
            if e.type == "ci_run":
                ci_by_name.setdefault((e.data["name"], e.data["sha"]), set()).add(e.data["conclusion"])

        for e in trace.events:
            if e.type == "review_comment":
                if e.reply_to is not None or roles.get(e.actor or "") != "reviewer":
                    continue
                d = e.data
                grades = [d["introduced_by_seq"]] if d.get("introduced_by_seq") is not None else []
                acted = bool(d.get("changed_by"))
                fixed = {"acted_on": acted, "fixed_by_seq": d.get("changed_by_seq")}
                what = intent(e.text or "")
                if d.get("suggestion") is not None and d.get("suggestion_applied"):
                    sig(e, "suggestion_accepted", -1, grades, fix=d["suggestion"][:2000], intent=what, **fixed)
                elif what == "request" or (d.get("suggestion") is not None and what != "praise"):
                    sig(e, "correction" if acted else "rejection", -1, grades, intent=what, **fixed)
                elif what == "clarify" and acted:
                    sig(e, "correction", -1, grades, intent=what, **fixed)
                elif what == "praise":
                    sig(e, "approval", 1, grades, intent=what)
            elif e.type == "review":
                if roles.get(e.actor or "") != "reviewer":
                    continue
                reviewed = commit_seq.get(e.data.get("commit") or "")
                grades = [reviewed] if reviewed is not None else []
                state = e.data.get("state")
                if state == "APPROVED":
                    sig(e, "approval", 1, grades, state=state)
                elif state == "CHANGES_REQUESTED" or state == "COMMENTED" and (e.text or "").strip() and not OVERVIEW.search(e.text or "") and intent(e.text or "") == "request":
                    sig(e, "rejection", -1, grades, state=state, acted_on=None)
            elif e.type == "ci_run":
                c = e.data.get("conclusion")
                if c in IGNORED_CI:
                    continue
                seen = ci_by_name.get((e.data["name"], e.data["sha"]), set())
                flaky = "success" in seen and bool(seen & FAILED)
                pol = 0 if flaky else (1 if c == "success" else -1 if c in FAILED else 0)
                g = commit_seq.get(e.data["sha"])
                sig(e, "check_result", pol, [g] if g is not None else [], check=e.data["name"], conclusion=c, flaky=flaky)
            elif e.type == "post_merge_commit":
                touched = e.data.get("touches_seq") or []
                if e.data.get("is_revert"):
                    sig(e, "revert", -1, touched or final_commits, days_after_merge=e.data.get("days_after_merge"), acted_on=True)
                elif e.data.get("fixlike") and touched:
                    sig(e, "correction", -1, touched, hotfix=True, days_after_merge=e.data.get("days_after_merge"), acted_on=True)
        return out



