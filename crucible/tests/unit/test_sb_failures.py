"""Failure mining (factory/failures.py; docs/research/swarm-pipeline/08): audited student traces -> findings (milestones
missed and lost, the first wrong step, the cause, a verbatim quote) -> builder-intent requests the swarm planner and the
root curator consume. Bundles are built in memory: no Docker, no model."""

from __future__ import annotations

import pytest

from crucible.factory.audit import AuditLabel
from crucible.factory.failures import MODES, builder_intents, finding, mine, mode_of
from crucible.factory.traceaudit import CheckpointRow, EpisodeBundle, ToolStep, facts, first_wrong_step, render_bundle

PATCH_A = "diff --git a/pkg/core.py b/pkg/core.py\n--- a/pkg/core.py\n+++ b/pkg/core.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"
PATCH_B = PATCH_A + "diff --git a/pkg/util.py b/pkg/util.py\n--- a/pkg/util.py\n+++ b/pkg/util.py\n@@ -1 +1 @@\n-y = 1\n+y = 3\n"


def bundle(i: int, *, kind: str = "lost", policy: str = "solver:student:Qwen/Qwen3.5-4B@mlx", task: str = "w1~abc", masked: bool = False) -> EpisodeBundle:
    steps = [ToolStep(1, "c1", "repo__read_file", {"path": "pkg/core.py"}, "x = 1"),
             ToolStep(2, "c2", "repo__write_file", {"path": "pkg/core.py", "content": "x = 2"}, "ok"),
             ToolStep(3, "c3", "repo__write_file", {"path": "pkg/util.py", "content": "y = 3"}, "ok"),
             ToolStep(4, "c4", "repo__shell", {"command": "python -m pytest -q"}, "1 failed")]
    if kind == "lost":
        cps = [CheckpointRow(0, 2, ["edit"], 0.5, True, {"m1": True, "m2": False}, "a", PATCH_A),
               CheckpointRow(1, 3, ["edit"], 0.0, True, {"m1": False, "m2": False}, "b", PATCH_B)]
        ms = {"m1": False, "m2": False}
    elif kind == "plateau":
        cps = [CheckpointRow(0, 2, ["edit"], 0.5, True, {"m1": True, "m2": False}, "a", PATCH_A)]
        ms = {"m1": True, "m2": False}
    else:  # claimed done, nothing moved
        cps = [CheckpointRow(0, 3, ["edit"], 0.0, True, {"m1": False, "m2": False}, "b", PATCH_B)]
        ms = {"m1": False, "m2": False}
    return EpisodeBundle(id=f"run/s{i}/a0", dir="/nowhere", run="run", task=task, window="w1", policy=policy, model="student:Qwen/Qwen3.5-4B",
                         family="qwen", prompt="Please add the feature.", final_message="All the requirements are implemented." if kind == "claim" else "Stopping.",
                         steps=steps, checkpoints=cps, phi0=0.0, stop="agent_done", masked=masked, reward=0.0, milestones=ms, phi_final=cps[-1].phi,
                         patch=PATCH_B if kind != "plateau" else PATCH_A, path={"checks": []})


def test_a_lost_milestone_is_a_finding_with_its_first_wrong_step_and_a_verbatim_quote() -> None:
    b = bundle(1)
    f = finding(b, numerics="dev")
    assert f is not None and f.mode == "lost_progress" and f.cause is None and f.cause_source == "unlabelled"
    assert f.milestones_missed == ["m1", "m2"] and f.milestones_lost == ["m1"]
    assert f.first_wrong_step["kind"] == "lost" and f.first_wrong_step["step"] == 3 and f.first_wrong_step["files"] == ["pkg/util.py"]
    assert f.quote and f.quote in render_bundle(b) and "pkg/util.py" in f.quote
    assert f.numerics == "dev" and f.routed is None and f.signature.startswith("unlabelled|lost")


def test_modes_from_facts_and_the_first_wrong_step() -> None:
    for kind, mode in (("plateau", "stalled"), ("claim", "claimed_done_without_progress")):
        b = bundle(2, kind=kind)
        assert mode_of(b, facts(b), first_wrong_step(b)) == mode
    m = finding(bundle(3, masked=True))
    assert m is not None and m.mode == "env_flake" and m.cause == "env_flake" and m.routed == "infra"
    assert MODES["harness_misuse"] is None and MODES["lost_progress"]


def test_an_auditor_label_wins_and_its_quote_is_kept_only_if_verbatim() -> None:
    b = bundle(4)
    lab = AuditLabel(b.id, "world_bug", False, evidence="1 failed", status="labelled")
    f = finding(b, label=lab)
    assert f is not None and f.cause == "world_bug" and f.cause_source == "auditor" and f.quote == "1 failed" and f.routed == "world"
    f2 = finding(b, label=AuditLabel(b.id, "skill_gap", False, evidence="words never in the trace", status="labelled"))
    assert f2 is not None and f2.quote != "words never in the trace" and f2.quote in render_bundle(b)


def test_passes_are_not_findings_and_the_policy_stamp_sets_numerics() -> None:
    p = bundle(5)
    p.reward = 1.0
    assert finding(p) is None
    fs = mine([bundle(6), bundle(7, policy="solver:tinker:Qwen/Qwen3.6-35B-A3B@tinker"), bundle(8, policy="solver:tinker:Qwen/Qwen3.5-4B@tinker"), p])
    assert [f.numerics for f in fs] == ["dev", "final", "dev"]  # MLX, the target on Tinker, the dev model on Tinker
    assert [f.student for f in fs] == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B", "Qwen/Qwen3.5-4B"]


def test_intents_are_keyed_by_student_model() -> None:
    tgt = "solver:tinker:Qwen/Qwen3.6-35B-A3B@tinker"
    fs = mine([bundle(20), bundle(21, task="w2~def"), bundle(22, policy=tgt), bundle(23, policy=tgt, task="w2~def")])
    intents, _ = builder_intents(fs, min_traces=2)
    assert sorted(i.student for i in intents) == ["Qwen/Qwen3.5-4B", "Qwen/Qwen3.6-35B-A3B"] and all(i.n == 2 for i in intents)
    by = {i.student: i for i in intents}
    assert by["Qwen/Qwen3.6-35B-A3B"].numerics == "final" and by["Qwen/Qwen3.5-4B"].numerics == "dev"
    assert "qwen3-6-35b-a3b" in by["Qwen/Qwen3.6-35B-A3B"].id and by["Qwen/Qwen3.6-35B-A3B"].planner_intent()["student"] == "Qwen/Qwen3.6-35B-A3B"
    assert by["Qwen/Qwen3.6-35B-A3B"].cell(approved_by="human:muahmed").observations[0]["student"] == "Qwen/Qwen3.6-35B-A3B"


def test_builder_intents_cluster_skill_modes_and_route_the_rest() -> None:
    fs = mine([bundle(10), bundle(11, task="w2~def"), bundle(12, kind="plateau"), bundle(13, masked=True)])
    intents, routed = builder_intents(fs, min_traces=2)
    assert [i.mode for i in intents] == ["lost_progress"] and [r["owner"] for r in routed] == ["infra"]
    bi = intents[0]
    assert bi.n == 2 and bi.tasks == ["w1~abc", "w2~def"] and bi.numerics == "dev" and bi.milestones == ["m1"] and bi.files == ["pkg/util.py"]
    assert bi.behaviour == "Keeps earlier work working while extending it" and len(bi.quotes) == 2 and bi.status == "awaiting_human"
    card = bi.intent_card()
    assert card["status"] == "candidate" and card["evidence"] == bi.evidence and card["coverage"]["behaviour"] == bi.behaviour
    pi = bi.planner_intent()
    assert pi["id"] == bi.id and pi["behaviour"] == bi.behaviour and "never a vaguer request" in pi["ask"] and pi["observed"]
    cell = bi.cell(approved_by="human:muahmed")
    assert cell.intent["status"] == "approved" and cell.intent["approved_by"] == "human:muahmed" and len(cell.observations) == 2
    with pytest.raises(ValueError):
        bi.cell(approved_by="nobody")  # only a human (or a named actor) approves an intent
    assert builder_intents(fs, min_traces=3)[0] == []
