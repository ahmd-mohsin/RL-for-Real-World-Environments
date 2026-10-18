"""P1 typed hand-offs and the family task repo (factory/handoff.py; docs/research/swarm-pipeline/00 §3.1, §3.3, §3.4): a
schema per port plus the full original spec (never a summary alone), stored for attribution (hand-off loss / grader error /
actor error); the task repo with immutable commits, ``informed_by=`` lineage, in-flight announcements checked before a lease,
free abstention and the 40% herding guard, wired into a machine as the S0 lease and S17 accept states."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from crucible.factory.handoff import (
    DuplicateWork,
    HandoffError,
    ImmutableError,
    TaskRepo,
    accept_state,
    attribute,
    build_handoff,
    check_handoff,
    lease_state,
)
from crucible.factory.machine import MachineRun, StateContext, canonicalize

SPEC = {"intent": "dependency-aware refactors", "request": "add mul and pow to calc", "source": "issue #12"}
PLAN = {"milestones": [{"id": "m1", "text": "mul"}, {"id": "m2", "text": "pow", "requires": ["m1"]}]}


def machine() -> dict[str, Any]:
    return canonicalize({"states": [
        {"id": "plan", "entry": True, "fn": "plan", "outputs": [{"name": "plan", "type": "json"}]},
        {"id": "world", "fn": "world", "inputs": [{"name": "plan", "type": "json", "from": "plan.plan"},
                                                  {"name": "note", "type": "text", "from": "params.note", "optional": True}],
         "outputs": [{"name": "world", "type": "json", "schema": {"type": "object", "required": ["test_cmd"]}}]},
        {"id": "done", "terminal": "done"}],
        "transitions": [{"from": "plan", "to": "world"}, {"from": "world", "to": "done"}]})


def ctx(inputs: dict[str, Any], params: dict[str, Any] | None = None) -> StateContext:
    st = next(s for s in machine()["states"] if s["id"] == "world")
    return StateContext("run1", "world", 1, 1, inputs, ["carried finding"], {"spec": SPEC, **(params or {})}, "world_expert", state=st)


def states() -> dict[str, Any]:
    return {s["id"]: s for s in machine()["states"]}


def test_handoff_carries_every_port_with_schema_and_the_full_spec() -> None:
    h = build_handoff(ctx({"plan": PLAN, "note": None}), states())
    assert h.ports["plan"] == PLAN and h.sources["plan"] == "plan.plan" and h.spec == SPEC and h.findings == ["carried finding"]
    assert "milestones" in json.dumps(h.schemas["plan"])  # PORT_SCHEMAS["plan"]
    assert check_handoff(h, original_spec=SPEC) == []
    with pytest.raises(HandoffError, match="milestones"):
        build_handoff(ctx({"plan": {"steps": []}}), states())
    with pytest.raises(HandoffError, match="original spec"):
        build_handoff(StateContext("run1", "world", 1, 1, {"plan": PLAN}, [], {}, state=states()["world"]), states())


def test_check_handoff_refuses_a_summary_spec_and_an_edit() -> None:
    h = json.loads(build_handoff(ctx({"plan": PLAN}), states()).to_json())
    summary = {**h, "spec": "add mul and pow"}
    assert any("summary" in p for p in check_handoff(summary, original_spec=SPEC))
    edited = {**h, "ports": {**h["ports"], "plan": {"milestones": []}}}
    probs = check_handoff(edited, original_spec=SPEC)
    assert any("edited after sealing" in p for p in probs) and any("port 'plan'" in p for p in probs)


def test_attribution_three_ways() -> None:
    h = build_handoff(ctx({"plan": PLAN}), states())
    good = {"plan": {"plan": PLAN}}
    assert attribute(h, producer_outputs=good, original_spec=SPEC).label == "actor_error"
    lossy = json.loads(h.to_json())
    lossy["ports"]["plan"] = {"milestones": [{"id": "m1", "text": "mul"}]}  # a dropped milestone in transit
    a = attribute(lossy, producer_outputs=good, original_spec=SPEC)
    assert a.label == "handoff_loss" and any("differs from what plan.plan settled" in e for e in a.evidence)
    assert attribute(h, producer_outputs=good, original_spec=SPEC, grader_pass=False, regrade_pass=True).label == "grader_error"
    canaried = {"plan": {"plan": {**PLAN, "_canary": "CRUCIBLE-CANARY-x"}}}  # the settle's canary is not a loss
    assert attribute(h, producer_outputs=canaried, original_spec=SPEC).label == "actor_error"


def test_task_repo_commits_are_immutable_and_carry_lineage(tmp_path: Path) -> None:
    repo = TaskRepo(tmp_path / "fam", "calc")
    a = repo.commit({"runs/r1/plan.json": json.dumps(PLAN)}, "plan", trailers={"state": "plan"})
    b = repo.commit({"runs/r1/world.json": "{}"}, "world", informed_by=[a])
    c = repo.commit({"runs/r1/tests/t.py": "assert 1"}, "tests", informed_by=[b])
    assert repo.trailers(c)["informed_by"] == [b] and repo.lineage(c) == [b, a]
    assert repo.commit({"runs/r1/plan.json": json.dumps(PLAN)}, "same content") == c  # a no-op, not a rewrite
    with pytest.raises(ImmutableError):
        repo.commit({"runs/r1/plan.json": "{}"}, "rewrite")
    with pytest.raises(ImmutableError):
        repo.commit({"../escape.txt": "x"}, "escape")
    assert [r["subject"] for r in repo.log("runs/r1/plan.json")] == ["plan"]


def test_announcements_conflicts_expiry_abstain_and_herding(tmp_path: Path) -> None:
    now = {"t": 1000.0}
    repo = TaskRepo(tmp_path / "fam", "calc", clock=lambda: now["t"])
    a1 = repo.announce("w1", {"repo": "calc", "intent": "mul", "files": ["calc.py"]}, ["upstream fixes mul"], ttl_s=100)
    with pytest.raises(DuplicateWork) as e:
        repo.announce("w2", {"repo": "calc", "intent": "mul"}, ["x"])
    assert e.value.conflicts[0]["id"] == a1["id"]
    repo.announce("w2", {"repo": "calc", "intent": "pow"}, ["x"])  # a different intent is not a duplicate
    repo.announce("w3", {"repo": "other", "intent": "mul"}, ["x"])
    with pytest.raises(ValueError, match="abort criteria"):
        repo.announce("w4", {"repo": "z"}, [])
    now["t"] += 101  # a1 expired: its scope reopens
    a4 = repo.announce("w4", {"repo": "calc", "intent": "mul"}, ["x"])
    repo.abstain(a4["id"])
    assert a4["id"] not in [a["id"] for a in repo.inflight()]
    assert json.loads((repo.root / f"retired/{a4['id']}.json").read_text())["outcome"] == "abstained"
    # the herding guard: a parent may hold at most 40% of accepted tasks once there are 5
    for i, parent in enumerate(["adapter-x", "adapter-x", "issue-1", "issue-2"]):
        a = repo.announce(f"h{i}", {"cell": f"c{i}"}, ["x"])
        repo.retire(a["id"], "accepted", task_id=f"t{i}", parent=parent)
    v = repo.herding_check("adapter-x")
    assert not v.ok and v.share == 3 / 5 and "60%" in v.reason
    assert repo.herding_check("issue-3").ok
    with pytest.raises(ValueError, match="parent idea"):
        repo.retire(a1["id"], "accepted", task_id="t9")


async def test_lease_and_accept_states_in_a_machine(tmp_path: Path) -> None:
    repo = TaskRepo(tmp_path / "fam", "calc")

    async def work(ctx: StateContext) -> dict:
        return {"out": "task built"}

    m = {"states": [{"id": "lease", "entry": True, "fn": "lease", "outputs": [{"name": "lease", "type": "json"}]},
                    {"id": "work", "fn": "work", "outputs": [{"name": "out", "type": "text"}]},
                    {"id": "accept", "fn": "accept", "inputs": [{"name": "lease", "type": "json", "from": "lease.lease"},
                                                                 {"name": "parent", "type": "text", "from": "params.parent"}],
                     "outputs": [{"name": "accept", "type": "json"}]},
                    {"id": "accepted", "terminal": "done"}, {"id": "abstained", "terminal": "rejected"}, {"id": "herded", "terminal": "rejected"}],
         "transitions": [{"from": "lease", "to": "work", "when": "lease.ok == true"}, {"from": "lease", "to": "abstained", "when": "lease.ok == false"},
                         {"from": "work", "to": "accept"},
                         {"from": "accept", "to": "accepted", "when": "accept.ok == true"}, {"from": "accept", "to": "herded", "when": "accept.ok == false"}]}
    h = {"lease": lease_state(repo), "work": work, "accept": accept_state(repo, floor=3)}

    def params(i: int, scope: str, parent: str) -> dict:
        return {"scope": {"cell": scope}, "abort_criteria": ["cell filled"], "parent": parent, "task_id": f"task-{i}", "holder": f"root/{i}"}

    res = await MachineRun(m, run_dir=tmp_path / "r1", handlers=h, params=params(1, "a", "p1")).run()
    assert res.status == "done" and res.outputs["accept"]["accept"]["task_id"] == "task-1"
    # an in-flight duplicate is visible before the lease starts: the run abstains
    live = repo.announce("other-root", {"cell": "b"}, ["x"])
    res = await MachineRun(m, run_dir=tmp_path / "r2", handlers=h, params=params(2, "b", "p2")).run()
    assert res.status == "rejected" and res.reason.endswith("abstained") and res.outputs["lease"]["lease"]["conflicts"] == [live["id"]]
    res = await MachineRun(m, run_dir=tmp_path / "r3", handlers=h, params=params(3, "c", "p1")).run()
    assert res.status == "done"
    res = await MachineRun(m, run_dir=tmp_path / "r4", handlers=h, params=params(4, "d", "p1")).run()  # p1 would hold 3/3
    assert res.status == "rejected" and res.reason.endswith("herded") and "would hold 3/3" in res.outputs["accept"]["accept"]["reason"]
    assert sorted(a["task_id"] for a in repo.accepted()) == ["task-1", "task-3"]
