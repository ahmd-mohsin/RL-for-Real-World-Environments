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

from crucible.core.schema.signal import Signal, SignalTarget
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
# M4 audit: praise had to be the whole comment in a fixed spelling ("LGTM, thanks!", "**LGTM**", "+1" were missed); a
# negated cue was a request ("No need to change this", "This is not a bug"); any cue beat a question mark ("Should
# this also handle the empty case?" became a rejection); plain imperatives and defect statements were missed.
PRAISE_HEAD = re.compile(r"^\W*(?:lgtm|looks (?:good|great|fine)(?: to me)?|nice(?: one| catch| work| change)?|great(?: work| catch)?|good catch|"
                         r"thanks?(?: (?:a lot|for (?:fixing|the|this|doing)\b[^.!?\n]*))?|thank you(?: for\b[^.!?\n]*)?|perfect|\+1|👍\S*|:\+1:|ship it|awesome)",
                         re.IGNORECASE)
NEGATED = re.compile(r"\bno need (?:to|for)\b|\bdon'?t worry\b|\b(?:is|it'?s|that'?s|this is) (?:fine|ok|okay|correct|intended)(?: as[- ]is)?\b|"
                     r"\bnot an? (?:bug|problem|issue)\b|\bI don'?t think (?:we|you) (?:need|have) to\b|\bno changes? (?:needed|required)\b|"
                     r"\b(?:handles|handled) (?:it|this) correctly\b|\bnothing to (?:change|do)\b", re.IGNORECASE)
STRONG = re.compile(r"\bplease\b|\bmust\b", re.IGNORECASE)
IMPERATIVE = re.compile(r"^\W*(?:use|rename|remove|add|move|replace|drop|avoid|check|fix|make|keep|split|extract|inline|mark|wrap|return|initiali[sz]e|"
                        r"prefer|delete|revert|change|update|simplify|guard|handle|pass|declare|document|reorder|merge)\b(?!\s*\?)", re.IGNORECASE)
DEFECT = re.compile(r"\b(?:leaks?|memory leak|overflows?|can overflow|off[- ]by[- ]one|race(?: condition)?|deadlocks?|dangling|out of bounds|"
                    r"null (?:pointer|dereference)|use[- ]after[- ]free|crash(?:es)?|segfaults?|double free)\b", re.IGNORECASE)
POLITE = re.compile(r"\b(?:can|could|would) you (?!(?:explain|clarify|elaborate|say|tell|confirm why|share))\w+|\bwhy not\b|\bplease\b|\bwhat about (?:using|moving|adding)\b|"
                    r"\b(?:should|shall|can|could) we (?:also )?(?:add|remove|use|rename|move|drop|keep|make|change|replace|mark|document|deprecate|split|merge|update|guard|check)\b",
                    re.IGNORECASE)  # a suggestion asked as a question ("Should we add a note ...?"), not "Should this handle ...?"
HEDGED = re.compile(r"^\W*(?:I think|I believe|it seems|seems like|looks like|I suspect)\b", re.IGNORECASE)  # "I think X is nowhere replaced?" asserts a defect
PRAISE = PRAISE_HEAD  # kept for importers
FAILED = {"failure", "timed_out", "startup_failure", "error"}
PASSED = {"success"}
IGNORED_CI = {"neutral", "skipped", "cancelled", "action_required", "stale", None}
OVERVIEW = re.compile(r"pull request overview|wasn't able to review|reviewed \d+ out of \d+ changed files", re.IGNORECASE)


def _sentences(t: str) -> list[str]:
    return [x.strip() for x in re.split(r"(?<=[.!?])\s+|\n+", t) if x.strip()]


def intent(text: str) -> str:
    """request / clarify / praise / discussion. Abstains (discussion) on what it cannot read, e.g. non-English text."""
    t = re.sub(r"```.*?```", " ", text or "", flags=re.DOTALL)  # code blocks say nothing about intent
    t = re.sub(r"^>.*$", " ", t, flags=re.MULTILINE)  # quoted text is someone else's
    t = re.sub(r"`[^`\n]+`", "`code`", t).strip()  # inline code is not prose ("calls `os.remove`" is not "remove")
    head = PRAISE_HEAD.match(t)
    if head:
        rest = t[head.end():]
        if not (REQUEST.search(rest) or IMPERATIVE.search(rest.lstrip(" ,.!-—:;")) or DEFECT.search(rest) or "?" in rest) or NEGATED.search(rest):
            return "praise"
    asks, wonders = False, False
    for sent in _sentences(t):
        if NEGATED.search(sent) and not STRONG.search(sent):
            continue  # "no need to change this", "this is not a bug"
        if sent.endswith("?"):
            if POLITE.search(sent) or (HEDGED.search(sent) and (REQUEST.search(sent) or DEFECT.search(sent))):
                asks = True
            else:
                wonders = True
            continue
        if REQUEST.search(sent) or IMPERATIVE.search(sent) or DEFECT.search(sent):
            asks = True
    if asks:
        return "request"
    if wonders or CLARIFY.search(t):
        return "clarify"
    return "discussion"


class CodingSignalMapper:
    """Mapper for traces from ``crucible_pack_coding.connectors.github_pr``."""

    def map(self, trace: Trace) -> list[Signal]:
        roles = {a.id: a.role for a in trace.actors}
        kinds = {a.id: a.kind for a in trace.actors}
        commit_seq = {e.data["sha"]: e.seq for e in trace.events if e.type == "commit"}
        by_seq = {e.seq: e for e in trace.events}
        out: list[Signal] = []

        def sig(e: TraceEvent, type_: str, polarity: int, grades: list[int], **meta) -> None:
            out.append(Signal(id=signal_id(trace.trace_id, type_, e), type=type_, polarity=polarity, grades=grades,  # type: ignore[arg-type]
                              source_event=e.seq, actor=e.actor, target=e.target, text=(e.text or "")[:500] or None,
                              meta={"automated": kinds.get(e.actor or "") != "human", **meta}))

        def ci_key(e: TraceEvent) -> tuple[str, str, str]:  # two workflows may both have a job called "build"
            return (str(e.data.get("app") or e.actor or ""), str(e.data.get("name")), str(e.data.get("sha")))

        ci_seen: dict[tuple[str, str, str], set[str]] = {}
        for e in trace.events:
            if e.type == "ci_run":
                ci_seen.setdefault(ci_key(e), set()).add(str(e.data.get("conclusion") or "").lower())

        def root_of(e: TraceEvent) -> TraceEvent:
            while e.reply_to is not None and e.reply_to in by_seq:
                e = by_seq[e.reply_to]
            return e

        answered: set[int] = set()  # author-rooted threads a reviewer already spoke in
        for e in trace.events:
            if e.type == "review_comment":
                if roles.get(e.actor or "") != "reviewer":
                    continue
                if e.reply_to is not None:
                    # the thread's root carries the signal; when the author opened it, the first reviewer reply does
                    # (M4 audit: "No, please move it back" under the author's question was dropped)
                    root = root_of(e)
                    if roles.get(root.actor or "") == "reviewer" or root.seq in answered:
                        continue
                    answered.add(root.seq)
                d = e.data
                grades = [d["introduced_by_seq"]] if d.get("introduced_by_seq") is not None else []
                # None: the connector could not tell (a LEFT-side or file-level comment): no rejection is made of it
                acted = bool(d["changed_by"]) if "changed_by" in d else None
                fixed = {"acted_on": acted, "fixed_by_seq": d.get("changed_by_seq")}
                what = intent(e.text or "")
                has_fix = d.get("suggestion") is not None
                if has_fix and d.get("suggestion_applied"):
                    sig(e, "suggestion_accepted", -1, grades, fix=d["suggestion"][:2000], intent=what, **fixed)
                elif what == "request" or has_fix:  # a concrete fix is a request, whatever praise surrounds it
                    if acted is not None:
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
                c = str(e.data.get("conclusion") or "").lower() or None  # the GraphQL API spells them in capitals
                if c in IGNORED_CI:
                    continue
                seen = ci_seen.get(ci_key(e), set())
                flaky = bool(seen & PASSED) and bool(seen & FAILED)
                pol = 0 if flaky else (1 if c in PASSED else -1 if c in FAILED else 0)
                g = commit_seq.get(e.data.get("sha") or "")
                sig(e, "check_result", pol, [g] if g is not None else [], check=e.data.get("name"), conclusion=c, flaky=flaky)
            elif e.type == "post_merge_commit":
                touched = e.data.get("touches_seq") or []
                days = e.data.get("days_after_merge")
                if e.data.get("is_revert"):
                    # a revert that touched no PR line known to us grades no step (it graded every commit: M4 audit)
                    sig(e, "revert", -1, touched, days_after_merge=days, acted_on=True,
                        **({} if touched else {"attribution": "unattributed", "reason": "the revert touched no line a PR commit wrote"}))
                elif e.data.get("fixlike") and touched:
                    sig(e, "correction", -1, touched, hotfix=True, days_after_merge=days, acted_on=True)
        return out


def _reward(x) -> float | None:
    """A finite number, or None (Harbor refuses non-numeric and non-finite rewards; -1 is a harness's 'not graded')."""
    import math

    try:
        r = float(x)
    except (TypeError, ValueError):
        return None
    return r if math.isfinite(r) and r >= 0 else None


def _passed(x) -> bool:
    return x is True or (isinstance(x, (int, float)) and x >= 1) or (isinstance(x, str) and x.strip().lower() in ("true", "pass", "passed", "1", "yes"))


class AgentRunSignalMapper:
    """Frontier-run traces (``coding/agent_run``) -> gate-level outcome signals (ARCHITECTURE §6, M4 row: "frontier
    failures -> gate-level outcomes"). Each failing hidden-verifier gate becomes a ``check_result`` -1 signal on the
    run's final patch, targeting ``gate:<name>``; a passing run yields one ``check_result`` +1. These are the privileged
    failure knowledge the forge turns into trap tasks (DESIGN §7.1): what a frontier model got wrong, verified by the
    grader, not by an opinion.

    M4 audit: a green run reached through the answer key (``contaminated``) gave +1 (P27: never trusted); a graded run
    whose gates all passed gave nothing; a reward of "1.0" raised, NaN and the -1 "not graded" sentinel became -1, 0.5
    counted as 0; ``"pass": "false"`` was a pass; a gate named twice aborted extraction on a duplicate id."""

    def map(self, trace: Trace) -> list[Signal]:
        det = (trace.outcome.details if trace.outcome else None) or {}
        fin = next((e for e in reversed(trace.events) if e.type == "final_patch"), None)
        if fin is None:
            return []
        reward = _reward(det.get("reward"))
        contaminated = bool(det.get("contaminated"))
        base = {"task": det.get("task"), "run": det.get("run"), "model": det.get("model"), "reward": reward, "automated": True}

        def check(src: TraceEvent, gate: str, pol: int, text: str, **meta) -> Signal:
            return Signal(id=signal_id(trace.trace_id, "check_result", src, gate), type="check_result", polarity=pol, source_event=src.seq,  # type: ignore[arg-type]
                          grades=[fin.seq], actor="hidden_verifier", target=SignalTarget(object=f"gate:{gate}"), text=text, meta={**base, "gate": gate, **meta})

        out: list[Signal] = []
        graded = [e for e in trace.events if e.type == "grade_gate"]
        if graded:  # the grade report's own gate events (the run was graded with its report kept)
            seen: set[str] = set()
            for e in graded:
                g = str(e.data.get("gate") or f"#{e.seq}")
                if g in seen:
                    continue
                seen.add(g)
                if not _passed(e.data.get("pass")):
                    out.append(check(e, g, -1, f"{det.get('model')} failed hidden gate {g}"))
            if not out and not contaminated:
                out.append(check(fin, "*", 1, "every hidden gate passed"))
            return out
        gates = list(dict.fromkeys(str(g) for g in det.get("failing_gates") or []))
        if gates:
            return [check(fin, g, -1, f"{det.get('model')} failed hidden gate {g}") for g in gates]
        if reward is None or det.get("exit_status") in ("LimitsExceeded", "TimeExceeded"):
            return []  # not graded, an unreadable reward, or a run stopped by its limits: unscored, not a failure
        if reward >= 1:
            return [] if contaminated else [check(fin, "*", 1, f"hidden verifier reward {reward}")]
        if reward <= 0:
            return [check(fin, "*", -1, f"hidden verifier reward {reward}")]
        return [check(fin, "*", 0, f"hidden verifier reward {reward} (partial)", partial=True)]


class CIRunSignalMapper:
    """CI-log traces (``coding/ci_run``, from ``connectors.ci_logs``) -> one ``check_result`` per test or build target:
    pass +1, fail or error -1, skipped nothing. A run that failed for infrastructure reasons, or was cut off, is not
    scored (P16), and a test the runner itself marked flaky (pytest RERUN) is polarity 0. Tests that both passed and
    failed on one commit across runs (``ci_logs.flaky``) can be passed in. M4 audit: no mapper existed, so per-test
    results and infra masking never reached M4 (``extract`` raised ``LookupError``)."""

    def __init__(self, flaky_tests: dict[str, list[str]] | None = None):
        self.flaky_tests = flaky_tests or {}

    def map(self, trace: Trace) -> list[Signal]:
        run = trace.events[0].data if trace.events and trace.events[0].type == "ci_run" else {}
        if not run or run.get("failure_kind") == "infra" or run.get("cut_off"):
            return []
        known_flaky = set(self.flaky_tests.get(run.get("commit") or "", []))
        out = []
        for e in trace.events[1:]:
            st = e.data.get("status")
            if e.type != "test_case" or st not in ("pass", "fail", "error"):
                continue
            flaky = bool(e.data.get("flaky")) or e.data.get("name") in known_flaky
            pol = 0 if flaky else (1 if st == "pass" else -1)
            out.append(Signal(id=signal_id(trace.trace_id, "check_result", e), type="check_result", polarity=pol, source_event=e.seq,  # type: ignore[arg-type]
                              actor=e.actor, target=SignalTarget(object=f"test:{e.data.get('name')}"), text=(e.text or "")[:500] or None,
                              meta={"automated": True, "check": e.data.get("name"), "status": st, "flaky": flaky, "commit": run.get("commit"),
                                    "attribution": "unattributed", "reason": "a CI log holds no commit events: the run grades the commit it names"}))
        return out
