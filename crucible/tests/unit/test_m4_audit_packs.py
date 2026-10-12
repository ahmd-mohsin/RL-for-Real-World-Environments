"""M4 audit (docs/audit/2026-10-12/M04-signals.md): the packs' side of M4 -- the PR connector's facts, the coding
mappers, PR outcomes and the toy pack. Ids follow the studies (M = mapping, R = reliability, A = attribution, H = harness
sweep)."""

from __future__ import annotations

import json
from pathlib import Path

from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource
from tests.unit.test_m3_audit_logs import _git, _repo


def ev(seq, type_, kind="action", actor=None, text=None, reply_to=None, time=None, target=None, **data):
    from crucible.core.schema.signal import SignalTarget

    return TraceEvent(seq=seq, kind=kind, type=type_, actor=actor, text=text, reply_to=reply_to, time=time, data=data,
                      target=SignalTarget(object=target) if target else None)


def pr_trace(events, actors=(), label="unknown", details=None, family="coding/pull_request", fetched_at=None):
    return Trace(trace_id="github:acme/repo#12", family=family, source=TraceSource(connector="github_pr", ref="r", fetched_at=fetched_at),
                 actors=list(actors), events=events, outcome=TraceOutcome(label=label, details=details or {}))


AUTHOR = TraceActor(id="au", role="author", kind="human")
REV = TraceActor(id="rv", role="reviewer", kind="human")
CI = TraceActor(id="ci", role="ci", kind="system")


def coding(t):
    from crucible_pack_coding.signals import CodingSignalMapper

    return CodingSignalMapper().map(t)


# ------------------------------------------------------------------------------------------------ PR connector + outcome
def _cached_pr(tmp_path: Path, gd: Path, work: Path, *, comments=(), reviews=(), now="2026-03-01T00:00:00+00:00"):
    from crucible_pack_coding.connectors.github_pr import GitHubPRConnector
    from crucible_pack_coding.github_reviews import API

    head = _git(work, "rev-parse", "feat").strip()
    base = _git(work, "rev-parse", "feat~1").strip()
    merge = _git(work, "log", "--format=%H", "--merges", "-1", "--grep", "#12").strip()
    c = GitHubPRConnector(tmp_path / "cache", gd, offline=True, now=now)
    url = f"{API}/repos/acme/repo/pulls/12"
    pr = {"number": 12, "title": "Tweak launch bounds", "body": "", "user": {"login": "alice"}, "state": "closed", "html_url": "u",
          "created_at": "2026-01-02T00:00:00Z", "merged_at": "2026-01-03T00:00:00Z", "closed_at": "2026-01-03T00:00:00Z", "merge_commit_sha": merge,
          "merged_by": {"login": "maint"}, "head": {"sha": head, "ref": "feat"}, "base": {"sha": base, "ref": "master"}}
    for u, body in [(url, pr), (f"{url}/reviews?per_page=100", list(reviews)), (f"{url}/comments?per_page=100", list(comments)),
                    (f"{API}/repos/acme/repo/commits/{head}/check-runs?per_page=100", {"check_runs": []})]:
        c.gh._path(u).parent.mkdir(parents=True, exist_ok=True)
        c.gh._path(u).write_text(json.dumps(body))
    from crucible.connectors.base import SourceRef

    return c, c.normalize(c.fetch(SourceRef(connector="github_pr", key="acme/repo#12", meta={"repo": "acme/repo", "pr": 12}))), head


def test_R1_a_merged_pr_past_its_window_is_good_and_an_open_window_is_unknown(tmp_path):
    from crucible_pack_coding.connectors.github_pr import GitHubPRConnector
    from crucible_pack_coding.outcomes import PullRequestOutcome

    _, gd = _repo(tmp_path, branch="master", after=[("docs", {"README": "x\n"}, "2026-02-20T00:00:00Z")])
    c = GitHubPRConnector(tmp_path / "cache", gd, offline=True, now="2026-03-01T00:00:00+00:00")
    [ref] = list(c.discover_git({"repo": "acme/repo"}))
    t = c.normalize(c.fetch(ref))
    assert PullRequestOutcome().label(t).label == "good"  # "unknown | merged less than 30 days ago" for every merged PR
    c2 = GitHubPRConnector(tmp_path / "cache", gd, offline=True, now="2026-01-10T00:00:00+00:00")
    assert PullRequestOutcome().label(c2.normalize(c2.fetch(ref))).label == "unknown"


def test_R2_R3_R4_outcome_times_and_window():
    from crucible_pack_coding.outcomes import PullRequestOutcome

    merged = ev(1, "merged", kind="state", time="2026-01-02T00:00:00Z")
    assert PullRequestOutcome().label(pr_trace([ev(0, "commit"), merged], fetched_at="2026-10-09T12:00:00")).label == "good"  # TypeError
    assert PullRequestOutcome().label(pr_trace([ev(0, "commit"), ev(1, "merged", kind="state")], fetched_at="2026-10-09T12:00:00Z")).label == "unknown"
    late = ev(2, "post_merge_commit", is_revert=True, days_after_merge=199.0)
    assert PullRequestOutcome().label(pr_trace([ev(0, "commit"), merged, late], fetched_at="2026-10-09T12:00:00Z")).label == "good"


def test_R5_unread_history_is_unknown_not_good():
    from crucible_pack_coding.outcomes import PullRequestOutcome

    t = pr_trace([ev(0, "commit"), ev(1, "merged", kind="state", time="2026-01-02T00:00:00Z")], details={"window": {"checked": False, "closed": True}},
                 fetched_at="2026-10-09T12:00:00Z")
    assert PullRequestOutcome().label(t).label == "unknown"  # "good": no revert seen, because nothing after the merge was read


def test_R15_M2_M3_pr_events_have_row_ids_one_ai_reviewer_and_suggestions_need_a_change(tmp_path):
    work, gd = _repo(tmp_path, branch="master")
    _git(work, "checkout", "-q", "feat")
    head = _git(work, "rev-parse", "HEAD").strip()
    _git(work, "checkout", "-q", "master")
    hunk = "@@ -1,5 +1,5 @@\n a\n b\n+C2\n+D2\n e"
    comments = [{"id": 101, "user": {"login": "Copilot", "type": "Bot"}, "body": "```suggestion\ne\n```", "path": "gpu/k.cu", "original_line": 3,
                 "line": 3, "side": "RIGHT", "original_commit_id": head, "commit_id": head, "diff_hunk": hunk, "created_at": "2026-01-02T12:00:00Z", "position": 3}]
    reviews = [{"id": 7, "user": {"login": "copilot-pull-request-reviewer[bot]", "type": "Bot"}, "state": "COMMENTED", "body": "", "commit_id": head,
                "submitted_at": "2026-01-02T12:01:00Z"}]
    _, t, _ = _cached_pr(tmp_path, gd, work, comments=comments, reviews=reviews)
    keys = [e.data.get("row_id") for e in t.events]
    assert all(keys) and len(set(keys)) == len(keys)
    bots = {e.actor for e in t.events if e.type in ("review", "review_comment")}
    assert len(bots) == 1 and next(a for a in t.actors if a.id in bots).kind == "agent"  # two actors of two kinds
    cm = next(e for e in t.events if e.type == "review_comment")
    assert cm.data["suggestion_applied"] is False  # "e" is elsewhere in the file; the commented line never changed


def test_M8_M9_revert_and_fix_wording():
    from crucible_pack_coding.connectors.github_pr import is_fixlike, revert_of

    assert revert_of('Revert "Tweak launch bounds (#123)"', set()) == (True, False)
    assert revert_of('Revert "Revert "Tweak launch bounds""', set())[0] is False  # a re-land
    assert revert_of("Reapply \"Tweak\" after the revert of #130", set())[0] is False
    assert revert_of("Fix regression from #123; do not revert, adjust bounds instead", set())[0] is False
    assert revert_of("Undo\n\nThis reverts commit abcdef1234.", {"abcdef1234567"}) == (True, True)
    assert not is_fixlike("refactor: rename helpers", "fixes lint warning")
    assert not is_fixlike("docs: explain the bug-report template", "")
    assert is_fixlike("Fix launch bounds on sm_90", "")


# ------------------------------------------------------------------------------------------------ coding mapper
def test_M1_intent():
    from crucible_pack_coding.signals import intent

    cases = {"No need to change this, it's fine as is.": "discussion", "This is not a bug, it is the documented behaviour.": "discussion",
             "This calls `os.remove` safely, nice.": "discussion", "Should this also handle the empty cloud case?": "clarify",
             "Could you explain why this is needed?": "clarify", "LGTM, thanks!": "praise", "**LGTM**": "praise", "+1": "praise",
             "Good catch, I missed that.": "praise", "Use size_t here.": "request", "This leaks memory when n == 0.": "request",
             "Should we add a note when this should be removed?": "request", "I think this value is nowhere replaced?": "request",
             "LGTM, but please rename x": "request", "Bitte diese Zeile entfernen.": "discussion"}
    assert {t: intent(t) for t in cases} == cases


def test_M4_M10_M11_comment_rules():
    rc = {"kind": "message", "actor": "rv"}
    # M4: the connector could not tell whether the line changed (LEFT side): no rejection
    t = pr_trace([ev(0, "commit", sha="a"), ev(1, "review_comment", text="Please keep these comments", side="LEFT", **rc)], [AUTHOR, REV])
    assert coding(t) == []
    # M10: a reviewer's reply under the author's own root speaks for the thread
    t = pr_trace([ev(0, "commit", actor="au", sha="a"), ev(1, "review_comment", kind="message", actor="au", text="I moved the guard here, OK?"),
                  ev(2, "review_comment", reply_to=1, text="No, please move it back.", changed_by="b", introduced_by_seq=0, **rc)], [AUTHOR, REV])
    assert [s.type for s in coding(t)] == ["correction"]
    # M11: a suggestion with praise words is a request, not an approval
    t = pr_trace([ev(0, "commit", sha="a"), ev(1, "review_comment", text="Nice!\n```suggestion\nreturn x\n```", suggestion="return x",
                                                  suggestion_applied=False, changed_by=None, **rc)], [AUTHOR, REV])
    assert [s.type for s in coding(t)] == ["rejection"]


def test_M12_M13_ci_spelling_job_keys_and_untouched_reverts():
    t = pr_trace([ev(0, "commit", sha="a"), ev(1, "ci_run", kind="check", actor="ci", name="build", sha="a", conclusion="STARTUP_FAILURE", app="gha"),
                  ev(2, "ci_run", kind="check", actor="ci", name="build", sha="a", conclusion="success", app="azure")], [AUTHOR, CI])
    pol = [s.polarity for s in coding(t)]
    assert pol == [-1, 1]  # 0/0: polarity 0 for the spelling, then both "flaky" across two workflows
    t = pr_trace([ev(0, "commit", sha="a"), ev(1, "commit", sha="b"), ev(2, "post_merge_commit", is_revert=True, touches_seq=[])], [AUTHOR])
    [s] = coding(t)
    assert s.grades == [] and s.meta["attribution"] == "unattributed"  # graded every commit


def test_M6_ci_log_traces_are_extracted(tmp_path):
    from crucible.signals.extract import extract
    from tests.unit.test_m3_audit_logs import _ci

    t = _ci(tmp_path, '<testsuite><testcase classname="a" name="t1"/><testcase classname="a" name="t2"><failure message="x"/></testcase></testsuite>', "junit.xml")
    out = extract(t)  # LookupError: no signal_mapper registered for family 'coding/ci_run'
    assert sorted(s.polarity for s in out.signals) == [-1, 1]
    infra = t.model_copy(update={"events": [t.events[0].model_copy(update={"data": {**t.events[0].data, "failure_kind": "infra"}}), *t.events[1:]]})
    assert extract(infra).signals == []


def test_M14_M15_M16_frontier_run_rewards():
    from crucible_pack_coding.signals import AgentRunSignalMapper

    from crucible.signals.extract import extract

    def run(details, gates=()):
        events = [ev(0, "final_patch", kind="state")] + [ev(1 + i, "grade_gate", kind="check", gate=g, **{"pass": p}) for i, (g, p) in enumerate(gates)]
        return pr_trace(events, details=details, family="coding/agent_run")

    m = AgentRunSignalMapper()
    assert m.map(run({"reward": 1.0, "contaminated": True})) == []  # +1 through the answer key
    [ok] = m.map(run({}, [("tests", True), ("perf", "true")]))
    assert ok.polarity == 1  # a graded run whose gates all passed gave nothing
    assert [s.meta["gate"] for s in m.map(run({}, [("tests", "false")]))] == ["tests"]  # bool("false") was a pass
    assert m.map(run({"reward": "1.0"}))[0].polarity == 1  # TypeError
    assert m.map(run({"reward": float("nan")})) == []  # -1
    assert m.map(run({"reward": 0.5}))[0].polarity == 0  # same as 0.0
    assert len(extract(run({"failing_gates": ["tests", "tests"]}), mapper=m).signals) == 1  # duplicate id aborted extraction


# ------------------------------------------------------------------------------------------------ toy pack
def test_M17_R26_R27_R28_the_toy_gate_runs_and_passes():
    from crucible_pack_toy.m4.gate import run

    rep = run("test")  # the labels named another family and nothing loaded them; forced, type 0.87 and outcome 0.08
    assert rep["passed"] and rep["n_events"] == 84 and rep["label_errors"] == 0


def test_M17_toy_rules():
    from crucible_pack_toy.signals import ToyTicketSignalMapper

    actors = [TraceActor(id="c", role="customer", kind="human"), TraceActor(id="a", role="agent", kind="human"), TraceActor(id="q", role="system", kind="system")]
    t = Trace(trace_id="toy/support_ticket:T1", family="toy/support_ticket", source=TraceSource(connector="file_drop", ref="T1"), actors=actors,
              events=[ev(0, "assigned", actor="q"), ev(1, "agent_reply", kind="message", actor="a", text="Reinstall the app."),
                      ev(2, "customer_reply", kind="message", actor="c", text="Reinstalled, it still crashes."),
                      ev(3, "reassigned", actor="a", text="escalated to mobile"), ev(4, "csat", kind="message", actor="c", text="ok", score="3")])
    got = {s.type: s for s in ToyTicketSignalMapper().map(t)}
    assert got["rejection"].grades == [1] and got["escalation"].grades == [1]  # escalation graded the queue's routing
    assert got["outcome"].polarity == 0  # "3" (a string) was -1; CSAT was a check_result


def test_M18_M19_csat_rule_and_sla_text():
    from crucible_pack_toy.signals import csat_score, rule_of, sla_of

    assert csat_score("Took 2 days, but rated 5/5", {}) == 5.0
    assert csat_score("", {"score": "4.5"}) == 4.5
    assert csat_score("", {"score": 6}) is None
    assert rule_of("Refund of 75.00 exceeds the 50 limit", {}) == "refund_over_50_needs_approval"
    assert rule_of("Deletion after the identity check done.", {}) is None
    assert sla_of("Resolution SLA (24h) breached.", {}) == "resolution_24h"


def test_H5_M16_reward_and_pass_parsing():
    from crucible_pack_coding.signals import _passed, _reward

    assert _reward(-1) is None and _reward("x") is None and _reward(float("inf")) is None and _reward("0.5") == 0.5
    assert _passed("pass") and not _passed("false") and not _passed(0)
