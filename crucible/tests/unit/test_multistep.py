"""M1 item 4: multi-step tasks with per-step gating (``TaskStep``, ``NativeHarness._run_steps``; Harbor multi-step trials).

Three steps in one ticket World: open T1; then (after a setup that files T2) close T2; then delete T1. Each step has its
own grader. Checked: state and the policy's context carry across steps; the trace marks each step; a step below its
``min_reward`` ends the task; the three reward strategies; a fresh context per step; separate verification at every
step, with the policy's World kept open between steps and closed only for the last.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from crucible.actors.policy import EpisodeLimits, ScriptedPolicy
from crucible.core.schema.actions import Done, ToolCall
from crucible.core.schema.world import WorldSpec
from crucible.core.store import Store
from crucible.runtime.harness.native import NativeHarness, PolicyRef, TaskStep
from crucible.runtime.run import Runner, RunSpec
from crucible.verify.runtime.plan import CheckResult, GradeReport
from crucible.world.world import World


def _doc(world) -> dict:
    return json.loads(Path(world.apps["api"].state_params["path"]).read_text())


def _report(task_id: str, ok: bool, outcome: float | None = None) -> GradeReport:
    return GradeReport(task_id, "full", [CheckResult("c", "g", "pass" if ok else "fail")], {"g": "pass" if ok else "fail"}, 1.0 if ok else 0.0,
                       float(ok) if outcome is None else outcome, {"status": "ok"}, True, 0.0, meta={"kinds": {"g": "outcome"}})


class TicketTask:
    world_id = "t/tickets"

    def __init__(self, *, gates=(None, None, None), strategy: str = "mean", fresh: bool = False, separate: bool = False):
        self.id = f"t/tickets/{strategy}@1"
        self.step_reward = strategy
        self.separate = separate
        self.graded_in: list[World] = []
        self.opened: list[World] = []

        async def g1(world, d):
            self.graded_in.append(world)
            return _report(self.id, "T1" in _doc(world)["tickets"])

        async def g2(world, d):
            self.graded_in.append(world)
            return _report(self.id, _doc(world)["tickets"].get("T2", {}).get("status") == "closed")

        async def g3(world, d):
            self.graded_in.append(world)
            return _report(self.id, "T1" not in _doc(world)["tickets"])

        async def file_t2(world):
            r = await world._call_tool("helpdesk", ToolCall(tool_call_id="s1", function_name="api.create_resource",
                                                           arguments={"collection": "tickets", "id": "T2", "data": {"status": "open"}}), system=True)
            assert r.ok, r.content

        self.steps = [TaskStep("open", "Open ticket T1.", g1, min_reward=gates[0]),
                      TaskStep("close", "A customer filed T2. Close it.", g2, min_reward=gates[1], setup=file_t2),
                      TaskStep("cleanup", "Delete T1.", g3, min_reward=gates[2], fresh_context=fresh)]

    def identity_config(self) -> dict:
        """M10 lab audit R-4: the steps as data (their graders are closures: named by the step, not described) and the strategy; never
        ``graded_in`` / ``opened``. ``step_grades_once`` is set by tests that need the harness's once-only rule."""
        return {"step_reward": self.step_reward, "separate": self.separate, "step_grades_once": getattr(self, "step_grades_once", False),
                "steps": [{"name": s.name, "instruction": s.instruction, "min_reward": s.min_reward, "max_steps": s.max_steps, "fresh_context": s.fresh_context}
                          for s in self.steps]}

    def spec(self):
        d = {"world": "t/tickets",
             "apps": [{"id": "api", "template": "common/json_api", "state": {"adapter": "json_doc", "params": {"path": "api/doc.json"}},
                       "params": {"collections": ["tickets"]}}],
             "actors": [{"id": "agent", "kind": "policy", "tools": ["api.*"]}, {"id": "helpdesk", "kind": "system", "tools": ["api.*"]}]}
        if self.separate:
            d["verifier"] = {"mode": "separate"}
        return WorldSpec.model_validate(d)

    async def open(self, workdir: Path) -> World:
        w = World(self.spec(), instructions={"agent": "(set per step)"}, store=Store(Path(workdir) / "s"), workdir=Path(workdir) / "w")
        await w.start()
        await w.reset()
        self.opened.append(w)
        return w


def _t(i, name, **args):
    return ToolCall(tool_call_id=f"a{i}", function_name=f"api.{name}", arguments=args)


GOOD = [_t(1, "create_resource", collection="tickets", id="T1", data={"status": "open"}), Done(reason="opened"),
        _t(2, "update_resource", collection="tickets", id="T2", patch={"status": "closed"}), Done(reason="closed"),
        _t(3, "delete_resource", collection="tickets", id="T1"), Done(reason="deleted")]
SKIPS_STEP_2 = [GOOD[0], GOOD[1], Done(reason="nothing to do"), GOOD[4], GOOD[5]]


class Counting:
    def __init__(self, actions):
        self.actions, self.made = actions, 0

    def ref(self) -> PolicyRef:
        def make(attempt):
            self.made += 1
            return {"agent": ScriptedPolicy("agent", list(self.actions))}
        return PolicyRef(id="scripted:steps", make=make, config={})  # M10 lab audit R-4: declared configuration


async def _run(tmp_path, task, actions, rid):
    pol = Counting(actions)
    await Runner(tmp_path / "runs").run(RunSpec(run_id=rid, tasks=[task], policies=[pol.ref()],
                                                harnesses=[NativeHarness("native:scripted", limits=EpisodeLimits(max_steps=10))], k=1))
    adir = next((tmp_path / "runs" / rid / "samples").glob("*/a1"))
    return json.loads((adir / "record.json").read_text()), adir, pol


async def test_three_steps_carry_state_and_context(tmp_path):
    task = TicketTask()
    rec, adir, pol = await _run(tmp_path, task, GOOD, "good")
    assert rec["reward"] == 1.0 and rec["failure"] is None, rec
    assert pol.made == 1  # one policy object for all three steps: its context carries over
    steps = rec["meta"]["task_steps"]
    assert [(x["name"], x["reward"], x["passed_gate"]) for x in steps] == [("open", 1.0, True), ("close", 1.0, True), ("cleanup", 1.0, True)]
    assert rec["meta"]["step_reward"] == "mean" and rec["meta"]["gates"] == {"step/open": 1, "step/close": 1, "step/cleanup": 1}
    trace = json.loads((adir / "trace.json").read_text())
    marks = [e["data"]["name"] for e in trace["events"] if e.get("type") == "task_step"]
    assert marks == ["open", "close", "cleanup"]
    for i, name in enumerate(["open", "close", "cleanup"]):
        assert json.loads((adir / "steps" / f"{i:02d}-{name}" / "verifier" / "report.json").read_text())["reward"] == 1.0
    assert all(w is task.opened[0] for w in task.graded_in)  # in-place grading: the one World


@pytest.mark.parametrize(("strategy", "expected"), [("mean", round(1 / 3, 6)), ("mean_executed", 0.5), ("final", 0.0)])
async def test_gate_stops_remaining_steps(tmp_path, strategy, expected):
    task = TicketTask(gates=(None, 1.0, None), strategy=strategy)
    rec, adir, _ = await _run(tmp_path, task, SKIPS_STEP_2, strategy)
    assert rec["reward"] == expected
    assert not (adir / "steps" / "02-cleanup").exists()  # the step after the failed gate never ran
    assert rec["meta"]["task_steps"][-1]["abort"] == "reward=0.0 below min_reward 1.0"
    assert len(task.graded_in) == 2


async def test_without_gate_every_step_runs(tmp_path):
    task = TicketTask()
    rec, adir, _ = await _run(tmp_path, task, SKIPS_STEP_2, "nogate")
    assert rec["reward"] == round(2 / 3, 6) and (adir / "steps" / "02-cleanup").exists()


async def test_dict_gate_on_flat_reward_keys(tmp_path):
    task = TicketTask(gates=({"outcome": 1.0, "reward": 0.5}, None, None))
    rec, _, _ = await _run(tmp_path, task, GOOD, "dict")
    assert rec["reward"] == 1.0
    task = TicketTask(gates=({"check/missing": 0.0}, None, None))  # a gated key the report lacks counts as below
    rec, _, _ = await _run(tmp_path, task, GOOD, "missing")
    assert rec["reward"] == round(1 / 3, 6)


async def test_fresh_context_makes_new_policies(tmp_path):
    task = TicketTask(fresh=True)
    actions_after_fresh = GOOD  # the new policy replays its script from the start: create T1 again, then stop
    rec, _, pol = await _run(tmp_path, task, actions_after_fresh, "fresh")
    assert pol.made == 2 and rec["reward"] == round(2 / 3, 6)  # step 3 got a new policy, which did not delete T1


async def test_separate_verification_per_step(tmp_path):
    task = TicketTask(separate=True)
    rec, adir, _ = await _run(tmp_path, task, GOOD, "sep")
    assert rec["reward"] == 1.0, rec
    agent = task.opened[0]
    assert len(task.graded_in) == 3 and all(w is not agent for w in task.graded_in)  # every step graded in a copy
    assert len({id(w) for w in task.graded_in}) == 3 and agent.closed
    for i, name in enumerate(["open", "close", "cleanup"]):
        assert (adir / "steps" / f"{i:02d}-{name}" / "verifier" / "handoff.json").exists()
