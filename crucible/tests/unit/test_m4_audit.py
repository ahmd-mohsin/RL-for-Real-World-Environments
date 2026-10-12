"""M4 audit (docs/audit/2026-10-12/M04-signals.md): the core signal layer -- ids, extraction, attribution, session
signals, reliability, decision points and the exit gate. Ids follow the studies: A = attribution study, R = reliability
study, M = mapping study, H = harness sweep. Each test failed on the commit before the audit, except where noted."""

from __future__ import annotations

import pytest

from crucible.core.schema.signal import Signal, SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource


def ev(seq, type_, kind="action", actor=None, writes=None, target=None, text=None, reply_to=None, time=None, **data):
    if writes is not None:
        data["writes"] = writes
    return TraceEvent(seq=seq, kind=kind, type=type_, actor=actor, text=text, reply_to=reply_to, time=time,
                      target=SignalTarget(app=target[0], object=target[1]) if isinstance(target, tuple) else (SignalTarget(object=target) if target else None),
                      data=data)


def tr(events, signals=(), label="unknown", family="test/fam", tid="t", actors=None):
    return Trace(trace_id=tid, family=family, source=TraceSource(connector="x", ref=tid), events=events, signals=list(signals),
                 outcome=TraceOutcome(label=label), actors=actors or [])


def sig(id_, type_, pol, grades, src, target=None, actor=None, **meta):
    return Signal(id=id_, type=type_, polarity=pol, grades=grades, source_event=src, actor=actor,
                  target=SignalTarget(app=target[0], object=target[1]) if isinstance(target, tuple) else (SignalTarget(object=target) if target else None),
                  meta=meta)


class Fixed:
    """A mapper returning given signals."""

    def __init__(self, *signals):
        self.signals = list(signals)

    def map(self, trace):
        return list(self.signals)


# ------------------------------------------------------------------------------------------------ attribution
def _graded(events, s):
    from crucible.signals.attribute import attribute

    return attribute(tr(events), [s])[0]


def test_A1_absolute_relative_and_dotted_paths_match():
    events = [ev(0, "tool_call", writes=["file:/home/u/repo/src/a.py"]), ev(1, "review_comment", kind="message")]
    assert _graded(events, sig("s", "correction", -1, [], 1, "file:src/a.py")).grades == [0]  # was []
    events = [ev(0, "tool_call", writes=["file:./a.py"]), ev(1, "check", kind="check")]
    assert _graded(events, sig("s", "correction", -1, [], 1, "file:a.py")).grades == [0]


def test_A2_a_backup_copy_or_a_prefix_name_does_not_take_the_grade():
    events = [ev(0, "tool_call", writes=["file:a.py"]), ev(1, "tool_call", writes=["file:a.py.bak"]), ev(2, "check", kind="check")]
    assert _graded(events, sig("s", "correction", -1, [], 2, "file:a.py")).grades == [0]  # was [1], the backup
    events = [ev(0, "tool_call", writes=["file:setup.py"]), ev(1, "tool_call", writes=["file:setup"]), ev(2, "check", kind="check")]
    assert _graded(events, sig("s", "correction", -1, [], 2, "file:setup.py")).grades == [0]
    events = [ev(0, "tool_call", writes=["ticket:T1.status"]), ev(1, "x", kind="message")]  # records keep "." fields
    assert _graded(events, sig("s", "rework", -1, [], 1, "ticket:T1")).grades == [0]


def test_A3_a_peer_in_another_workspace_is_not_credited():
    events = [ev(0, "tool_call", actor="a", writes=["file:a.py"], target=("ws_a", "file:a.py")),
              ev(1, "tool_call", actor="b", writes=["file:a.py"], target=("ws_b", "file:a.py")), ev(2, "check", kind="check")]
    assert _graded(events, sig("s", "correction", -1, [], 2, ("ws_a", "file:a.py"))).grades == [0]  # was [1]


def test_A4_deletions_and_moves_are_writes_and_a_directory_covers_its_files():
    from crucible.core.agent_events import writes_of

    assert writes_of("bash", {"command": "rm a.py"}) == ["file:a.py"]
    assert writes_of("bash", {"command": "git mv src lib"}) == ["file:lib", "file:src"]
    events = [ev(0, "tool_call", writes=["file:a/x.py"]), ev(1, "tool_call", writes=["file:lib"]), ev(2, "check", kind="check")]
    assert _graded(events, sig("s", "correction", -1, [], 2, "file:lib/a.py")).grades == [1]


def test_A7_A8_A9_self_exclusion_and_unattributed_reason():
    # A7: an anonymous source with exclude_self does not credit an anonymous rewrite
    events = [ev(0, "tool_call", actor="agent", writes=["ticket:T1.resolution"]), ev(1, "tool_call", writes=["ticket:T1.resolution"]),
              ev(2, "supervisor_edit", kind="message")]
    assert _graded(events, sig("s", "correction", -1, [], 2, "ticket:T1.resolution", exclude_self=True)).grades == [0]
    # A8: a reviewer's own fixup commit is not the step their rejection grades
    events = [ev(0, "commit", actor="author", writes=["file:a.py"]), ev(1, "commit", actor="rev", writes=["file:a.py"]),
              ev(2, "review_comment", kind="message", actor="rev")]
    assert _graded(events, sig("s", "rejection", -1, [], 2, "file:a.py", actor="rev")).grades == [0]
    # A9: "could not attribute" says why
    got = _graded([ev(0, "x"), ev(1, "check", kind="check")], sig("s", "correction", -1, [], 1, "file:nobody.py"))
    assert got.grades == [] and got.meta["attribution"] == "unattributed" and got.meta["reason"] == "no_writer"


def test_A5_atif_subagent_events_follow_the_call_that_delegated_them():
    pytest.importorskip("harbor")
    from harbor.models.trajectories import Trajectory

    from crucible.interop.harbor.atif_import import trajectory_to_trace

    sub = {"schema_version": "ATIF-v1.7", "session_id": "s1", "trajectory_id": "sub-1", "agent": {"name": "claude-code", "version": "1", "model_name": "m"},
           "steps": [{"step_id": 1, "source": "user", "message": "edit a.py"},
                     {"step_id": 2, "source": "agent", "message": "", "timestamp": "2026-01-01T10:02:00Z",
                      "tool_calls": [{"tool_call_id": "e", "function_name": "Edit", "arguments": {"file_path": "a.py", "old_string": "x", "new_string": "y"}}]}]}
    main = Trajectory.model_validate({"schema_version": "ATIF-v1.7", "session_id": "s1", "agent": {"name": "claude-code", "version": "1", "model_name": "m"},
                                      "steps": [{"step_id": 1, "source": "user", "message": "port it", "timestamp": "2026-01-01T10:00:00Z"},
                                                {"step_id": 2, "source": "agent", "message": "", "timestamp": "2026-01-01T10:00:30Z",
                                                 "tool_calls": [{"tool_call_id": "t", "function_name": "Task", "arguments": {"prompt": "edit a.py"}}],
                                                 "observation": {"results": [{"source_call_id": "t", "content": "done", "subagent_trajectory_ref": [{"trajectory_id": "sub-1"}]}]}},
                                                {"step_id": 3, "source": "user", "message": "that change to a.py is wrong", "timestamp": "2026-01-01T10:05:00Z"}],
                                      "subagent_trajectories": [sub]})
    t = trajectory_to_trace(main, trace_id="t", source=TraceSource(connector="atif", ref="r"))
    edit = next(e for e in t.events if e.data.get("tool") == "Edit")
    task = next(e for e in t.events if e.data.get("tool") == "Task")
    user = next(e for e in t.events if e.text == "that change to a.py is wrong")
    assert task.seq < edit.seq < user.seq  # the subagent's edit was appended after the user's later correction
    assert task.data["delegated_to"] == ["sub-1"]


def test_A6_business_app_calls_record_the_records_they_write():
    from crucible.core.agent_events import app_writes

    assert app_writes("tickets.update_ticket", {"ticket_id": "T7", "status": "resolved"}) == ["ticket:T7.status"]
    assert app_writes("erp.approve_invoice", {"invoice_id": "INV-1"}) == ["invoice:INV-1"]
    assert app_writes("api.update_resource", {"collection": "orders", "id": "O1", "patch": {"status": "x"}}) == ["orders:O1.status"]
    assert app_writes("tickets.get_ticket", {"ticket_id": "T7"}) == []  # a read
    assert app_writes("tickets.close_ticket", {"ticket_id": "T7"}, read_only=True) == []
    from crucible.signals.attribute import attribute

    t = tr([ev(0, "tool_call", actor="agent", writes=app_writes("tickets.issue_refund", {"order_id": "O-5", "amount": 80})), ev(1, "policy_flag", kind="message")])
    assert attribute(t, [sig("v", "violation", -1, [], 1, "order:O-5")])[0].grades == [0]


# ------------------------------------------------------------------------------------------------ extraction
def test_A11_M21_signal_ids_survive_an_earlier_row():
    from crucible.core.schema.trace import event_key
    from crucible.signals.extract import signal_id

    a = ev(0, "approval_evt", kind="message", text="ok", row_id="r9")
    b = ev(1, "approval_evt", kind="message", text="ok", row_id="r9")  # same row one position later
    assert signal_id("t", "approval", a) == signal_id("t", "approval", b)
    assert event_key(a) == "row:r9"


def test_A12_extract_keeps_hand_added_signals_and_known_reliability():
    from crucible.signals.extract import extract

    t = tr([ev(0, "tool_call", writes=["file:a.py"]), ev(1, "review", kind="message")],
           signals=[sig("hand", "interrupt", -1, [0], 1, by="hand"), sig("m1", "rejection", -1, [0], 1).model_copy(update={"reliability": 0.83})])
    out = extract(t, mapper=Fixed(sig("m1", "rejection", -1, [0], 1)))
    by = {s.id: s for s in out.signals}
    assert set(by) == {"m1", "hand"} and by["m1"].reliability == 0.83  # both were dropped


def test_A13_A14_bad_mapper_output_fails_at_extraction_and_late_grades_are_flagged():
    from crucible.signals.extract import ExtractError, extract

    t = tr([ev(0, "tool_call"), ev(1, "review", kind="message"), ev(2, "x")])
    with pytest.raises(ExtractError, match="unknown events"):
        extract(t, mapper=Fixed(sig("s", "rejection", -1, [999], 1)))
    with pytest.raises(ExtractError, match="polarity"):
        extract(t, mapper=Fixed(sig("s", "suggestion_accepted", 1, [0], 1)))
    late = extract(t, mapper=Fixed(sig("s", "approval", 1, [0, 2], 1))).signals[0]
    assert late.grades == [0] and late.meta["dropped_grades"] == [2]


def test_A15_duplicate_ids_name_the_trace():
    from crucible.signals.extract import ExtractError, extract

    t = tr([ev(0, "tool_call"), ev(1, "ci", kind="check")], tid="pr#9")
    with pytest.raises(ExtractError, match="pr#9"):
        extract(t, mapper=Fixed(sig("d", "check_result", 1, [0], 1), sig("d", "check_result", -1, [0], 1)))


def test_A16_H1_H2_agent_sessions_yield_interrupts_and_rejections_without_a_pack_mapper():
    from crucible.core.agent_events import user_type
    from crucible.signals.extract import extract

    assert user_type("Commands not executed. The user rejected your commands with the following message: no") == "rejection"  # mini-swe-agent
    actors = [TraceActor(id="agent", role="agent", kind="agent"), TraceActor(id="user", role="user", kind="human"), TraceActor(id="system", role="system", kind="system")]
    t = tr([ev(0, "tool_call", actor="agent", writes=["file:a.py"]), ev(1, "tool_call", actor="agent", text="Bash(rm -rf build)"),
            ev(2, "tool_result", kind="observation", actor="system", reply_to=1, rejected=True),
            ev(3, "tool_call", actor="agent"), ev(4, "interrupt", kind="message", actor="user", text="[Request interrupted by user]")],
           family="coding/construction_session", actors=actors)
    out = extract(t)  # LookupError: no signal_mapper registered
    got = {(s.type, tuple(s.grades)) for s in out.signals}
    assert ("rejection", (1,)) in got  # the proposing call, not the last writer of anything (it never ran)
    assert ("interrupt", (3,)) in got


def test_A19_a_family_without_a_mapper_is_still_labelled():
    from crucible.signals.extract import extract

    class L:
        def label(self, trace):
            return TraceOutcome(label="good")

    assert extract(tr([ev(0, "x")], family="nobody/fam"), labeler=L()).outcome.label == "good"


# ------------------------------------------------------------------------------------------------ reliability
def test_R7_corrections_can_be_contradicted_and_need_evidence():
    from crucible.signals.reliability import verdict

    events = [ev(0, "commit"), ev(1, "review_comment", kind="message"), ev(2, "commit"), ev(3, "post_merge_commit")]
    c = sig("c", "correction", -1, [0], 1, fixed_by_seq=2, acted_on=True)
    assert verdict(tr(events, [c], label="unknown"), c) is None  # was True: confirmed by construction
    rv = sig("r", "revert", -1, [2], 3)
    assert verdict(tr(events, [c, rv], label="bad"), c) is False  # the fix itself was rolled back
    assert verdict(tr(events, [c], label="good"), c) is True
    assert verdict(tr(events, [rv], label="bad"), rv) is None  # the revert defines the outcome: no independent evidence


def test_R9_another_jobs_failure_does_not_contradict_a_passing_check():
    from crucible.signals.reliability import verdict

    events = [ev(0, "commit"), ev(1, "ci_run", kind="check"), ev(2, "ci_run", kind="check")]
    lint = sig("l", "check_result", 1, [0], 1, check="lint")
    build = sig("b", "check_result", -1, [0], 2, check="build")
    assert verdict(tr(events, [lint, build], label="good"), lint) is True  # was False


def test_R10_R11_R14_rows_per_family_and_actor_class_unjudgeable_signals_get_none():
    from crucible.signals.reliability import apply, estimate

    events = [ev(0, "commit"), ev(1, "review", kind="message"), ev(2, "review", kind="message")]
    human = sig("h", "approval", 1, [0], 1)
    bot = sig("b", "approval", 1, [0], 2, automated=True)
    flaky = sig("f", "check_result", 0, [0], 2, check="x")
    t1 = tr(events, [human, bot, flaky], label="good", family="coding/pull_request")
    t2 = tr(events, [sig("h2", "approval", 1, [0], 1)], label="good", family="toy/support_ticket")
    table = estimate([t1, t2])
    assert ("coding/pull_request", "approval", "automated") in table and ("toy/support_ticket", "approval", "human") in table
    out = {s.id: s.reliability for s in apply(t1, table).signals}
    assert out["f"] is None and out["h"] is not None  # flaky got the type mean
    # R14: an approval at seq 0 vs a later rejection: order is known through `is not None`, not truthiness
    from crucible.signals.reliability import verdict

    e2 = [ev(0, "commit"), ev(1, "x"), ev(2, "y")]
    a = sig("a", "approval", 1, [0], 1)
    r = sig("r", "rejection", -1, [0], 2)
    assert verdict(tr(e2, [a, r], label="good"), a) is False


# ------------------------------------------------------------------------------------------------ pivots and pairs
def test_A20_A21_A22_pivot_pairs():
    from crucible.signals.pivots import pivots
    from crucible.studio.datasets import from_trace

    actors = [TraceActor(id="agent", role="agent", kind="agent"), TraceActor(id="sup", role="supervisor", kind="human")]
    events = [ev(0, "refund_issued", actor="agent", text="refund"), ev(1, "status_change", actor="agent", text="resolved"),
              ev(2, "policy_flag", kind="message", actor="sup", text="no approval"), ev(3, "refund_reversed", actor="agent", text="reverse")]
    rv = sig("rv", "revert", -1, [0, 1], 3)
    t = tr(events, [rv], label="bad", actors=actors)
    item = from_trace(t)
    assert item.pivots == [] and item.pivots_unpaired >= 1  # "reverse the refund" was the chosen action at the state before it
    c = sig("c", "correction", -1, [0], 2, fixed_by_seq=1)
    undone = sig("u", "revert", -1, [1], 3)
    assert 0 not in pivots(tr(events, [c, undone], label="bad", actors=actors))  # its fix was itself reverted


# ------------------------------------------------------------------------------------------------ the gate
def _gate_trace(order=(0, 1, 2)):
    rows = [ev(0, "commit", row_id="commit:a", writes=["file:a.py"]), ev(1, "review_comment", kind="message", row_id="comment:1"),
            ev(2, "review_comment", kind="message", row_id="comment:2")]
    events = [rows[i].model_copy(update={"seq": k}) for k, i in enumerate(order)]
    seq = {e.data["row_id"]: e.seq for e in events}
    s = sig("s1", "correction", -1, [seq["commit:a"]], seq["comment:1"])
    return tr(events, [s], label="good", tid="t")


def _labels(n=1, outcomes=1, **kw):
    ev_ = {"trace_id": "t", "event": "row:comment:1", "type": "correction", "grades": ["row:commit:a"], **kw}
    return {"pack": "p", "events": [ev_] * n, "outcomes": [{"trace_id": "t", "label": "good"}] * outcomes}


def test_R15_labels_by_event_key_survive_reordering():
    from crucible.signals.gate import evaluate

    for order in [(0, 1, 2), (1, 0, 2)]:  # the commit moved after the comment (M3's instant sort moved events like this)
        rep = evaluate({"t": _gate_trace(order)}, _labels())
        assert rep.type_accuracy == 1.0


def test_R16_R17_R18_R21_fail_closed_scoring():
    from crucible.signals.gate import evaluate

    rep = evaluate({}, {"events": [{"trace_id": "nope", "event": "row:x", "type": "none"}]})  # KeyError
    assert rep.label_errors == 1 and rep.type_accuracy == 0.0
    t = _gate_trace()
    t2 = t.model_copy(update={"signals": t.signals + [sig("s2", "approval", 1, [0], 1)]})
    assert evaluate({"t": t2}, _labels()).type_accuracy == 0.0  # the spurious approval was invisible
    rep = evaluate({"t": t}, _labels(grades=None))
    assert rep.attribution_precision == 0.0  # a step graded where the label says none was never penalised
    small = evaluate({"t": t}, _labels())
    assert small.type_accuracy == 1.0 and not small.passed  # one label passed the gate
    big = evaluate({"t": t}, _labels(n=50, outcomes=10))
    assert big.passed and big.intervals["type_accuracy"][0] < 1.0


def test_R25_outcome_coverage_is_reported_apart():
    from crucible.signals.gate import evaluate

    t = _gate_trace().model_copy(update={"outcome": TraceOutcome(label="unknown")})
    rep = evaluate({"t": t}, {"events": [], "outcomes": [{"trace_id": "t", "label": "unknown"}]})
    assert rep.outcome_agreement == 1.0 and rep.outcome_coverage == 0.0
