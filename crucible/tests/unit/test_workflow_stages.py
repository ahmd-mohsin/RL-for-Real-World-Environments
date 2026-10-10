"""M5 stage induction, the workflow graph document, the Workflow State Tracker and event-log state."""

from __future__ import annotations

from crucible.connectors.file_drop import FieldMap, rows_to_trace
from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck
from crucible.core.schema.state import Query
from crucible.workflow.graph import WorkflowGraph
from crucible.workflow.logstate import LogStateView, states
from crucible.workflow.stages import induce, name_stages, precedence, stage_of
from crucible.workflow.tracker import WorkflowTracker


def m(i: str, field: str, req=()) -> Milestone:
    return Milestone(id=f"log/{i}", requires=list(req), check=MilestoneCheck(app="t", query=Query(op="exists", args={"object": f"ticket:{{case}}.{field}"}),
                                                                             expect=Expectation(kind="truthy")))


MS = [m("assignee", "assignee"), m("resolution", "resolution"), m("status", "status")]


def test_precedence_and_layers():
    ach = [{"log/assignee": 0.1, "log/resolution": 0.5, "log/status": 0.9}] * 4 + [{"log/assignee": 0.2, "log/resolution": 0.4, "log/status": 0.8}]
    edges = precedence(ach, min_support=0.9, min_traces=3)
    assert {(e.a, e.b) for e in edges} == {("log/assignee", "log/resolution"), ("log/assignee", "log/status"), ("log/resolution", "log/status")}
    model = name_stages(induce(MS, ach))
    assert [s.milestones for s in model.stages] == [["log/assignee"], ["log/resolution"], ["log/status"]]
    assert [s.name for s in model.stages] == ["assignee", "resolution", "status"]
    # unordered milestones share a layer; contradictory data never yields an edge
    mixed = [{"log/assignee": 0.1, "log/resolution": 0.2}, {"log/assignee": 0.3, "log/resolution": 0.1}] * 3
    assert precedence(mixed, min_support=0.9) == []


def test_stage_of_and_graph_round_trip():
    model = induce(MS, [{"log/assignee": 0.1, "log/resolution": 0.5, "log/status": 0.9}] * 3)
    assert stage_of(model, MS, {"log/assignee": 1.0}) == 1 and stage_of(model, MS, {}) == 0
    assert stage_of(model, MS, {k.id: 1.0 for k in MS}) == 3
    g = WorkflowGraph.build("toy/x", MS, name_stages(model))
    back = WorkflowGraph.model_validate_json(g.model_dump_json())
    assert back.stage_model().stage_index() == {"log/assignee": 0, "log/resolution": 1, "log/status": 2}


async def test_log_state_and_tracker():
    rows = [{"case": "T1", "time": "2026-01-01T09:00:00Z", "actor": "a", "type": "ticket_opened", "writes": ["ticket:T1.request"]},
            {"case": "T1", "time": "2026-01-01T09:01:00Z", "actor": "q", "type": "assigned", "writes": ["ticket:T1.assignee"], "team": "tier1"},
            {"case": "T1", "time": "2026-01-01T09:20:00Z", "actor": "b", "type": "agent_reply", "writes": ["ticket:T1.resolution"]},
            {"case": "T1", "time": "2026-01-01T09:21:00Z", "actor": "b", "type": "status_change", "writes": ["ticket:T1.status"], "status": "resolved"},
            {"case": "T1", "time": "2026-01-01T10:00:00Z", "actor": "w", "type": "reopened", "writes": ["ticket:T1.status"], "status": "open"}]
    t = rows_to_trace(rows, FieldMap(extra=["team", "status"]), case="T1", connector="file_drop")
    st = states(t, case="T1")
    assert st[1][1].objects["ticket:T1.assignee"] == "tier1" and st[3][1].objects["ticket:T1.status"] == "resolved" and st[4][1].objects["ticket:T1.status"] == "open"
    view = LogStateView(objects=st[3][1].objects, case="T1")
    r = await view.query("t", Query(op="match", args={"object": "ticket:{case}.status", "pattern": "^resolved$"}))
    assert r.value == {"exists": True, "matches": True}
    resolved = Milestone(id="log/status=resolved", check=MilestoneCheck(app="t", query=Query(op="match", args={"object": "ticket:{case}.status", "pattern": "^resolved$"}),
                                                                         expect=Expectation(kind="equals", value={"exists": True, "matches": True})))
    ms = [MS[0], MS[1], resolved]
    graph = WorkflowGraph.build("toy/x", ms, induce(ms, [{"log/assignee": 0.1, "log/resolution": 0.5, "log/status=resolved": 0.9}] * 3))
    tr = WorkflowTracker(graph)
    hist = [await tr.observe(None, v) for _, v in st]
    assert [h.stage for h in hist] == [0, 1, 2, 3, 2]  # the reopen sends the ticket back to the resolve stage
    assert hist[3].gained == ["log/status=resolved"] and hist[4].lost == ["log/status=resolved"] and "regressed:log/status=resolved" in hist[4].deviations
    assert hist[1].next_expected == ["log/resolution"] and abs(hist[3].phi - 1.0) < 1e-9
