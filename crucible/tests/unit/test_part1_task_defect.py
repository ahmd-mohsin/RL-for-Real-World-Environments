"""Part 1 V9 (2026-10-30, docs/research/env-synthesis/part1-framework-improvements.md §2(c)): a ``report_task_defect``
exit. The agent had ``GIVE_UP:`` (graded as it stands) but no way to say the task itself is broken, the exit that cut
cheating in ImpossibleBench (2510.20270). A ``TASK_DEFECT: <what is wrong>`` line in the final message now ends the
episode with reward 0 (never negative) and the report on the rollout; the trainer logs the share per step and
framework_health clusters the reports per task. The lane's epilogue announces it to newly authored tasks only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from crucible_pack_toy.runtime import FibTask

from crucible.actors.policy import LLMToolPolicy
from crucible.core.schema.rollout import RolloutRecord
from crucible.factory import framework_health as fh
from crucible.factory.lanes.windows import EPILOGUE
from crucible.runtime.harness.native import PolicyRef
from crucible.runtime.harness.workspace import WorkspaceHarness, task_defect_of
from crucible.runtime.inject import FakeLLM, tool_reply
from crucible.runtime.run import Runner, RunSpec
from crucible.train.bridges.tinker.trainer import Group, behaviour_metrics
from tests.unit.test_framework_health import view

FIB = "a, b = 0, 1\nfor _ in range(10):\n    print(a)\n    a, b = b, a + b\n"


async def _episode(tmp_path, final: str):
    acts = [tool_reply("ws__write_file", {"path": "fib.py", "content": FIB}, 1), final]
    pol = PolicyRef(id="fake:fib", make=lambda a: {"agent": LLMToolPolicy("agent", FakeLLM(list(acts)))}, config={"final": final[:20]})
    await Runner(tmp_path).run(RunSpec(run_id="v9", tasks=[FibTask()], policies=[pol], harnesses=[WorkspaceHarness()], k=1))
    adir = min((tmp_path / "v9" / "samples").glob("*/a*"))
    return json.loads((adir / "record.json").read_text()), RolloutRecord.model_validate_json((adir / "rollout.json").read_text())


async def test_a_reported_defect_scores_0_and_travels_with_the_rollout(tmp_path) -> None:
    rec, ro = await _episode(tmp_path, "I wrote fib.py.\nTASK_DEFECT: the request asks for 11 numbers but the checks pin 10")
    assert rec["reward"] == 0.0 and not rec["mask_sample"]  # before: 1.0, the correct fix's grade
    assert ro.progress["task_defect"] == {"report": "the request asks for 11 numbers but the checks pin 10", "graded_reward": 1.0}


async def test_without_the_line_nothing_changes(tmp_path) -> None:
    rec, ro = await _episode(tmp_path, "done; the word TASK_DEFECT: mid-line is no report")
    assert rec["reward"] == 1.0 and "task_defect" not in (ro.progress or {})


def test_only_the_final_messages_line_counts() -> None:
    def ev(kind, typ, text=None, data=None):
        return SimpleNamespace(kind=kind, type=typ, text=text, data=data or {})

    tr = SimpleNamespace(events=[ev("action", "done", "TASK_DEFECT: an earlier actor's"), ev("action", "tool_call"), ev("action", "done", "fine")])
    assert task_defect_of(tr) is None  # the last done decides
    assert task_defect_of(SimpleNamespace(events=[ev("action", "done", "x", {"answer": "  TASK_DEFECT:"})])) == ""  # empty is still a report
    assert "TASK_DEFECT: <what is wrong" in EPILOGUE and EPILOGUE.startswith("How this works")  # the 40-char marker other code finds


@pytest.mark.parametrize(("graded", "severity", "cls"), [(0.0, "HIGH", "suspect"), (1.0, "LOW", "agent")])
def test_framework_health_clusters_the_reports_per_task_and_never_pauses(graded, severity, cls) -> None:
    td = {"report": "the checks pin 10 numbers", "graded_reward": graded}
    vs = [view(k, rec={"meta": {"progress": {"task_defect": td}}}) for k in range(3)]
    h = fh.health_report(vs)
    [d] = [d for d in h["defects"] if d["kind"] == "task_defect_report"]
    assert d["severity"] == severity and d["cls"] == cls and len(d["episodes"]) == 3 and not h["pause"]


def test_the_trainer_logs_the_share_per_step() -> None:
    def rec(rid, td=None):
        return RolloutRecord.model_validate({"rollout_id": rid, "sample": rid, "task": "t", "harness": "h", "policy": "p", "reward": 0.0,
                                             "progress": {"task_defect": td} if td else {}})

    batch = [Group(0, None, "t", "v1", "g", [rec("a", {"report": "x", "graded_reward": 0.0}), rec("b"), rec("c"), rec("d")], 0, 0.0)]
    assert behaviour_metrics(batch, [])["task_defect/share"] == pytest.approx(0.25)
