"""P0 ``LongTask`` contract (swarm-pipeline/00 §1, ``factory.longtask``): ``WindowTask`` conforms; the conformance checker
finds what a broken task lacks; the executed ground truth, the null agent (S8) and clean-World solvability (S5) run on any
LongTask and catch a grader that cannot fail and a task nobody can solve; ``simplify('prerequisites')``; the card and
the serialisable ref."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_m17_w_task import REQS, make_task

from crucible.factory.cards import TaskCard
from crucible.factory.lanes.windows import WindowTask, window_checkset
from crucible.factory.longtask import (
    GroundCase,
    LongTask,
    LongTaskRef,
    check_longtask,
    clean_world_solvable,
    ground_truth,
    new_file_patch,
    null_agent_score,
)
from crucible.factory.pathcheck import PathRules


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return make_task(tmp_path_factory.mktemp("p0lt"))


def test_window_task_conforms(built) -> None:
    task, rec, _ = built
    assert isinstance(task, LongTask)
    assert check_longtask(task) == []
    w = task.world
    assert w["network"] == "none" and w["history"] == "cut" and w["base_dir"] == str(task.base_dir) and w["repos"] == ["toyrepo"]
    assert task.prompt == task.instruction and isinstance(task.path_rules, PathRules)
    assert set(task.path_rules.grader_paths) == set(rec["tests_b"])
    assert {m.milestone for m in task.mutants()} == {"m01", "m02", "m03"}
    assert [c.id for c in task.floors()] == ["no_op", "claims_only", "test_edit", "pass_stub"] and all(c.kind == "floor" for c in task.floors())
    assert task.oracle_patch().startswith("diff --git") and "pkg/gamma.py" in task.oracle_patch()


def test_check_longtask_names_each_defect(built) -> None:
    task, rec, _ = built
    assert check_longtask(SimpleNamespace(task_id="x")) [:2] == ["member: missing world", "member: missing prompt"]
    # a requires edge the prober labelled patch-conflict (a textual dependency) is not an executed edge
    bad = WindowTask(task.lane, task.mined, {**rec, "milestones": [{**m, "edges": {u: "patch-conflict" for u in m.get("edges") or {}}} for m in rec["milestones"]]},
                     task_id="bad", prompt=task.instruction, requirements=REQS, spans=task.spans, checkset=task.checkset)
    found = check_longtask(bad)
    assert "milestone m03: requires m02 without an executed edge" in found and "short: no executed requires edge (long, not long-horizon)" in found
    cyc = [{**m, "requires": ["m03"] if m["id"] == "m02" else m.get("requires") or []} for m in rec["milestones"]]
    bad2 = WindowTask(task.lane, task.mined, {**rec, "milestones": cyc}, task_id="cyc", prompt=task.instruction, requirements=REQS[:2], spans=task.spans,
                      checkset=task.checkset)
    found2 = check_longtask(bad2)
    assert any(f.startswith("milestones: requires cycle") for f in found2) and "ledger: milestone m03 has no requirement" in found2
    unauthored = WindowTask(task.lane, task.mined, rec, task_id="u", prompt="", requirements=[], spans={}, checkset=window_checkset(rec, []))
    assert "prompt: empty" in check_longtask(unauthored) and check_longtask(unauthored, need_prompt=False) == []


def test_ground_truth_null_agent_and_clean_world(built) -> None:
    task, _, _ = built
    gt = asyncio.run(ground_truth(task))
    assert gt["ok"], gt["failed"]
    assert gt["rows"]["test_edit"]["path_violations"] == ["path:protected"] and gt["rows"]["test_edit"]["reward"] == 0.0
    assert gt["rows"]["revert_m02"]["milestones"]["m02"] is False and gt["rows"]["oracle_steps"]["reward"] == 1.0
    null = asyncio.run(null_agent_score(task))
    assert null["ok"] and set(null["rows"]) == {"empty", "claims_only"}
    cw = asyncio.run(clean_world_solvable(task))
    assert cw["ok"] and cw["reward"] == 1.0 and cw["phi"] == 1.0
    with pytest.raises(TypeError):
        asyncio.run(clean_world_solvable(task, in_world=True))  # needs a workdir (and a Docker World)


def test_null_agent_and_solvability_catch_broken_tasks(built) -> None:
    task, rec, _ = built
    # a grader that cannot fail: milestone m01 "checks" a test that already passes at the start state
    keep = "tests/test_keep.py::test_keep"
    ms = [{**m, "tests": [keep]} if m["id"] == "m01" else m for m in rec["milestones"]]
    leaky = WindowTask(task.lane, task.mined, {**rec, "milestones": ms}, task_id="leaky", prompt=task.instruction, requirements=REQS, spans=task.spans,
                       checkset=window_checkset({**rec, "milestones": ms}, REQS))
    null = asyncio.run(null_agent_score(leaky))
    assert not null["ok"] and null["rows"]["empty"]["phi"] > 0
    # an unsolvable task: a milestone whose check no end state can pass
    ms2 = [{**m, "tests": ["tests/test_core.py::test_nobody_wrote"]} if m["id"] == "m02" else m for m in rec["milestones"]]
    dead = WindowTask(task.lane, task.mined, {**rec, "milestones": ms2}, task_id="dead", prompt=task.instruction, requirements=REQS, spans=task.spans,
                      checkset=window_checkset({**rec, "milestones": ms2}, REQS))
    cw = asyncio.run(clean_world_solvable(dead))
    assert not cw["ok"] and cw["reward"] < 1.0


def test_simplify_pre_applies_the_prerequisites(built, tmp_path: Path) -> None:
    task, _, _ = built
    simple = task.simplify("prerequisites", root=tmp_path)
    sf = simple.window["simplified_from"]
    assert sf["pre_applied"] == ["m02"] and sf["targets"] == ["m03"] and sf["mode"] == "exact"
    assert [m["id"] for m in simple.milestones] == ["m01", "m03"] and simple.milestones[1]["requires"] == []
    assert (simple.base_dir / "pkg" / "beta.py").exists() and not (simple.base_dir / "pkg" / "gamma.py").exists()
    assert "[R2]" not in simple.prompt and "[R3]" in simple.prompt and [r["id"] for r in simple.requirements] == ["R1", "R3"]
    assert check_longtask(simple, long_horizon=False) == []
    assert asyncio.run(simple.milestone_status("")) == {"m01": False, "m03": False}
    gamma = new_file_patch("pkg/gamma.py", "from pkg.beta import beta\n\n\ndef gamma():\n    return beta() + 1\n")
    assert asyncio.run(simple.milestone_status(gamma)) == {"m01": False, "m03": True}  # beta is already there
    assert asyncio.run(task.milestone_status(gamma))["m03"] is False  # on the original start it is not
    assert asyncio.run(ground_truth(simple))["ok"]
    assert {m.milestone for m in simple.mutants()} <= {"m01", "m03"}
    with pytest.raises(ValueError):
        task.simplify("faulty_function")


def test_card_and_ref(built) -> None:
    task, _, _ = built
    card = task.card(intent="I1")
    assert isinstance(card, TaskCard) and card.verifier.verifier_version == task.verifier_version and card.lineage["generator"] == "mined_window"
    assert card.ledger_problems({}) is not None
    ref = task.ref()
    back = LongTaskRef.from_json(json.loads(ref.dumps()))
    assert back == ref and back.source_id == task.mined.id and back.split == "train" and back.verifier_version == task.verifier_version


def test_new_file_patch_applies(built) -> None:
    task, _, _ = built
    p = new_file_patch("NOTES.md", "a\nb")
    assert asyncio.run(task.grade(p))[0].meta["apply_errors"] == []
    assert GroundCase("x", "floor", p).solution == p


def test_simplify_uses_only_executed_edges_and_a_relative_root(built, tmp_path: Path, monkeypatch) -> None:
    task, rec, _ = built
    # a record mined before report 11 F2 lists patch-conflict units in requires_units: they are never pre-applied
    textual = [{**m, "edges": {u: "patch-conflict" for u in m.get("edges") or {}}} for m in rec["milestones"]]
    old = WindowTask(task.lane, task.mined, {**rec, "milestones": textual}, task_id="old", prompt=task.instruction, requirements=REQS, spans=task.spans,
                     checkset=task.checkset)
    with pytest.raises(ValueError, match="executed prerequisites"):
        old.simplify("prerequisites", root=tmp_path)
    monkeypatch.chdir(tmp_path)  # git runs inside the new start: a relative root is resolved first
    simple = task.simplify("prerequisites", root=Path("rel"))
    assert simple.base_dir.is_absolute() and (tmp_path / "rel").is_dir()
