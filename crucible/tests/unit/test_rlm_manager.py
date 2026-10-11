"""m14: the RLM task manager and the continual harness. The measured harness accepts an edit only when the same-seed A/B
improves the primary metric without a guard regression, refuses base-prompt edits and uncompilable skills, and rolls back
to any version. The task FSM (Planner -> Implementer (child RLM) -> Reviewer -> objective M9 Verifier) repairs a forged task
with M9's B.15 hole re-introduced: a first, wrong check is caught by the verifier (A16 still finds the hack) and loops back;
the second adds the in-episode approval check, passes A1 + A16 and the review, and is committed. The manager RLM drives it
through its ``forge`` proxy; a crashed FSM resumes from its journal without re-implementing."""

from __future__ import annotations

import json

import pytest

from crucible.actors.llm import LLMResponse
from crucible.actors.rlm.harness_state import HarnessError, HarnessState
from crucible.runtime.inject import FakeLLM


def code(src: str, i: int = 0) -> LLMResponse:
    return LLMResponse(text="", tool_calls=[{"id": f"py{i}", "name": "python", "arguments": {"code": src}}])


# ---- the measured continual harness -----------------------------------------------------------------------------------
async def test_harness_measured_edits_rollback_and_digest(tmp_path):
    h = HarnessState(tmp_path / "harness.json")
    v0 = h.version
    mem = {"action": "create", "kind": "memory", "id": "approval_events", "title": "approval needs event grounding",
           "content": "an approver name in state is not an approval: grade the in-episode message", "family": "toy/helpdesk_v2"}

    async def helps(state, seed):  # the metric depends on whether the memory is present
        return {"repair_rate": 0.8 if "approval_events" in state else 0.4, "cost": 1.0}

    ev = await h.evaluate([mem], helps, seeds=[0, 1, 2], primary="repair_rate", margin=0.1, guards=["cost"], higher_is_better={"cost": False})
    assert ev["accepted"] and "approval_events" in h.entries and h.version == v0 + 1

    async def hurts(state, seed):
        return {"repair_rate": 0.8, "cost": 3.0 if "noise" in state else 1.0}

    noise = {"action": "create", "kind": "note", "id": "noise", "title": "always add five checks", "content": "more checks are better"}
    ev2 = await h.evaluate([noise], hurts, seeds=[0, 1], primary="repair_rate", margin=-1, guards=["cost"], higher_is_better={"cost": False})
    assert not ev2["accepted"] and "noise" not in h.entries and any("guard cost regressed" in r for r in ev2["reasons"])
    with pytest.raises(HarnessError, match="immutable"):
        h.propose([{"action": "update", "kind": "note", "id": "base_system_prompt", "title": "x", "content": "y"}])
    with pytest.raises(HarnessError, match="does not compile"):
        h.propose([{"action": "create", "kind": "skill", "id": "s", "title": "s", "content": "def f(:\n  pass"}])
    d = h.digest("grade the approval message", family="toy/helpdesk_v2")
    assert "approval_events" in d and h.entries["approval_events"].shown == 1
    h.rollback_to(v0)
    assert "approval_events" not in h.entries
    again = HarnessState(tmp_path / "harness.json")  # persisted: versions, events and the rollback
    assert again.fingerprint == h.fingerprint and len(again.events) == 3


# ---- the FSM on the B.15 hole ------------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def weak_t2():
    import asyncio

    from crucible_pack_toy.forge.world_synth import forge

    col: list = []
    asyncio.run(forge(collect=col))
    t = next(x for x, _ in col if x.task_id.endswith("t2_approval"))
    return t


def _rebuild(task, checks):
    from crucible_pack_toy.forge.task import ToyForgedTask

    return ToyForgedTask(task.spec, task.cs.model_copy(update={"checks": list(checks)}), task.oracle, task.negatives, pack_root=task.pack_root,
                         world_file=task.world_file, scenario=task.sc, canary=task.canary)


def workspace(weak_t2, tmp_path):
    from crucible.actors.rlm.manager import ForgeWorkspace

    weak = _rebuild(weak_t2, [c for c in weak_t2.cs.checks if c.id != "policy/approval_given_in_episode"])
    return ForgeWorkspace({"t2": weak}, rebuild=_rebuild, checks=("A1", "A16"), cache_root=tmp_path / "accept")


WRONG = {"id": "outcome/extra_note", "gate": "outcome", "kind": "predicate", "app": "tickets", "sql": "SELECT COUNT(*) FROM notes",
         "expect": {"kind": "between", "lo": 1}, "states": ["note"], "provenance": "implementer guess"}
RIGHT = {"id": "policy/approval_given_in_episode", "gate": "policy", "gate_kind": "gate", "kind": "event",
         "spec": {"rule": "approval_grounded", "approver": "supervisor"}, "states": ["policy"], "provenance": "M9 A13 forged_approval (B.15)"}


def implementer_llm():
    return FakeLLM([code("t = await forge.show(task_id='t2')\nprint(len(t['checks']))", 1),
                    code(f"v = await forge.add_check(task_id='t2', check={WRONG!r})\nrlm.finish(str(v))", 2),
                    code(f"v = await forge.add_check(task_id='t2', check={RIGHT!r})\nrlm.finish(str(v))", 3)], model="implementer-model")


async def test_fsm_repairs_b15_with_a_loop_back(weak_t2, tmp_path):
    from crucible.actors.rlm.manager import TaskFSM, llm_reviewer, rlm_implementer

    ws = workspace(weak_t2, tmp_path)
    plans = []

    async def planner(task, findings):
        plans.append(findings)
        return "close the A16 hack: the approval must be grounded in an in-episode supervisor message"

    reviewer = llm_reviewer(FakeLLM([json.dumps({"approved": True, "findings": []})] * 3, model="reviewer-model"))
    fsm = TaskFSM(ws, planner=planner, implementer=rlm_implementer(implementer_llm(), ws), reviewer=reviewer, journal=tmp_path / "fsm.jsonl")
    st = await fsm.run("t2")
    assert st.phase == "committed" and st.round == 2, st.history
    assert any("A16" in f for f in plans[0])  # the base verification found the hack
    loop = next(h for h in st.history if h["event"] == "loop back")
    assert loop["verified"] is False and any("A16" in f for f in loop["findings"])  # the wrong check did not close it
    committed = ws.task("t2")
    assert "policy/approval_given_in_episode" in {c.id for c in committed.cs.checks} and ws.committed["t2"] == 2
    final = await ws.verify("t2")
    assert final["passed"] and final["checks"]["A16"]["status"] == "pass"


async def test_commit_refuses_unverified_and_fsm_resumes(weak_t2, tmp_path):
    from crucible.actors.rlm.manager import TaskFSM, rlm_implementer

    ws = workspace(weak_t2, tmp_path)
    v = await ws.add_check("t2", RIGHT)
    with pytest.raises(PermissionError, match="no passing verification"):
        await ws.commit("t2", v)
    calls = {"impl": 0, "review": 0}
    impl = rlm_implementer(FakeLLM([code(f"rlm.finish(str(await forge.add_check(task_id='t2', check={RIGHT!r})))", 1)]), ws)

    async def counting_impl(*a):
        calls["impl"] += 1
        return await impl(*a)

    async def planner(task, findings):
        return "add the in-episode approval check"

    async def flaky_review(summary):
        calls["review"] += 1
        if calls["review"] == 1:
            raise RuntimeError("reviewer provider down")
        return {"approved": True, "findings": []}

    fsm = TaskFSM(ws, planner=planner, implementer=counting_impl, reviewer=flaky_review, journal=tmp_path / "fsm.jsonl")
    with pytest.raises(RuntimeError, match="provider down"):
        await fsm.run("t2")
    st = await TaskFSM(ws, planner=planner, implementer=counting_impl, reviewer=flaky_review, journal=tmp_path / "fsm.jsonl").run("t2")
    assert st.phase == "committed" and calls["impl"] == 1  # resumed at review/verify with the same candidate


async def test_manager_rlm_drives_the_repair(weak_t2, tmp_path):
    from crucible.actors.rlm.manager import RLMTaskManager, TaskFSM, rlm_implementer

    ws = workspace(weak_t2, tmp_path)

    async def planner(task, findings):
        return "add the in-episode approval check"

    async def reviewer(summary):
        return {"approved": True, "findings": []}

    impl = rlm_implementer(FakeLLM([code(f"rlm.finish(str(await forge.add_check(task_id='t2', check={RIGHT!r})))", 1)]), ws)
    fsm = TaskFSM(ws, planner=planner, implementer=impl, reviewer=reviewer)
    h = HarnessState()
    h.apply_unmeasured([{"action": "create", "kind": "memory", "id": "m1", "title": "approvals", "content": "approval checks must read events"}],
                       reason="seed lesson")
    mgr = RLMTaskManager(FakeLLM([code("ts = await forge.tasks()\nbad = [t['task_id'] for t in ts if not (await forge.verify(task_id=t['task_id']))['passed']]\nbad", 1),
                                  code("res = [await forge.repair(task_id=t) for t in bad]\nrlm.finish(str([r['phase'] for r in res]))", 2)]),
                         ws, fsm, harness=h)
    run = await mgr.manage("Make every task in the workspace pass its acceptance checks.")
    assert run.finished and run.answer == "['committed']", run.messages[-1]
    assert mgr.fsm_results["t2"].phase == "committed" and "approval checks must read events" in run.messages[1]["content"]  # the digest reached it


async def test_candidates_cannot_weaken_their_audit_and_must_conform(weak_t2, tmp_path):
    """m14 live findings: a candidate that drops a check is still audited against the committed version's checks (so dropping
    the in-episode approval check from a fixed task fails A16 instead of passing it), and add_check rejects fields the pack's
    grammar does not understand."""
    from crucible_pack_toy.forge.vocabulary import VOCABULARY

    from crucible.actors.rlm.manager import ForgeWorkspace

    ws = ForgeWorkspace({"t2": weak_t2}, rebuild=_rebuild, checks=("A1", "A16"), cache_root=tmp_path / "accept", vocabulary=VOCABULARY)
    assert (await ws.verify("t2"))["passed"]  # the fixed task passes
    weaker = await ws.drop_check("t2", "policy/approval_given_in_episode")
    rep = await ws.verify("t2", weaker)
    assert not rep["passed"] and rep["checks"]["A16"]["status"] == "fail" and any("A16 evidence" in f for f in rep["findings"])
    with pytest.raises(ValueError, match="not understood by rule"):
        await ws.add_check("t2", {**RIGHT, "id": "policy/x", "spec": {"rule": "approval_grounded", "approver": "supervisor", "match_amount": True}})
    with pytest.raises(ValueError, match="is not one of"):
        await ws.add_check("t2", {**RIGHT, "id": "policy/y", "spec": {"rule": "made_up_rule"}})
    with pytest.raises(ValueError, match="already exists"):
        await ws.add_check("t2", RIGHT)
