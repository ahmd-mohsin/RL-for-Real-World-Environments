"""M17 P: cards.py validation (ids, Literal types, spans, pairs) and the fixed interface names."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from crucible.factory import cards as C
from tests.unit.m17_p_fixtures import NOW, PROMPT, SHA, card, ledger, obs


def test_interface_names_exist():
    for name in ("ObservationRecord", "IntentCard", "Requirement", "CheckRef", "RequirementLedger", "FailureCause",
                 "VerifierCard", "TaskCard", "Label", "Decision"):
        assert hasattr(C, name)


@pytest.mark.parametrize("t", ["real_pr_tests", "expert_end_state", "human_graded_outcome", "production_trace", "recorded_session",
                               "observed_model_failure", "reference_derived", "simulator_state"])
def test_first_class_property(t):
    assert obs(type=t).first_class == (t not in {"reference_derived", "simulator_state"})


def test_observation_validation():
    with pytest.raises(ValidationError):
        obs(type="vibes")
    with pytest.raises(ValidationError):
        obs(by="alice")
    with pytest.raises(ValidationError):
        obs(id="has space")
    with pytest.raises(ValidationError):
        obs(id="../x")
    with pytest.raises(ValidationError):
        C.ObservationRecord(id="o", type="real_pr_tests", pointer="p", sha256="xyz", created=NOW, by="human:a")
    with pytest.raises(ValidationError):
        C.ObservationRecord(id="o", type="real_pr_tests", pointer="p", sha256=SHA, created="yesterday", by="human:a")
    with pytest.raises(ValidationError):
        C.ObservationRecord(id="o", type="real_pr_tests", pointer="p", sha256=SHA, created=NOW, by="human:a", extra=1)


def test_intent_card_rules():
    base = {"id": "i1", "behaviour": "b", "failure_mode": "f"}
    assert C.IntentCard(**base).status == "candidate"
    with pytest.raises(ValidationError, match="approved_by"):
        C.IntentCard(**base, status="approved", evidence=["o1"], coverage={"behaviour": "x", "surface": "y"})
    with pytest.raises(ValidationError, match="evidence"):
        C.IntentCard(**base, status="approved", approved_by="human:a", coverage={"behaviour": "x", "surface": "y"})
    with pytest.raises(ValidationError, match="coverage"):
        C.IntentCard(**base, status="approved", approved_by="human:a", evidence=["o1"])
    with pytest.raises(ValidationError, match="has an approver"):
        C.IntentCard(**base, status="rejected", approved_by="human:a")
    with pytest.raises(ValidationError):
        C.IntentCard(**base, coverage={"colour": "red"})
    with pytest.raises(ValidationError, match="twice"):
        C.IntentCard(**base, evidence=["o1", "o1"])
    with pytest.raises(ValidationError):
        C.IntentCard(**base, status="done")
    ok = C.IntentCard(**base, status="approved", approved_by="human:a", evidence=["o1"], coverage={"behaviour": "x", "surface": "y"})
    assert ok.approved_by == "human:a"


def test_requirement_span_rules():
    for bad in [(-1, 3), (5, 5), (6, 5)]:
        with pytest.raises(ValidationError):
            C.Requirement(id="r", text="t", prompt_span=bad, kind="k")


def test_prompt_span_must_lie_inside_the_prompt():
    c = card()
    assert c.ledger_problems({"o1": obs("o1"), "o2": obs("o2")}) == []
    bad = C.RequirementLedger(requirements=[C.Requirement(id="r1", text="x", prompt_span=(0, len(PROMPT) + 1), kind="k")])
    with pytest.raises(ValidationError, match="runs past the prompt"):
        card(ledger=bad)
    edge = C.RequirementLedger(requirements=[C.Requirement(id="r1", text="Reply with a one-line summary.", prompt_span=(len(PROMPT) - 30, len(PROMPT)), kind="k")])
    assert card(ledger=edge)  # ending exactly at the prompt's end is inside


def test_duplicate_ids_in_ledger_rejected():
    r = C.Requirement(id="r1", text="t", prompt_span=(0, 3), kind="k")
    with pytest.raises(ValidationError, match="duplicate requirement"):
        C.RequirementLedger(requirements=[r, r])
    c = C.CheckRef(id="c", requirement_ids=["r1"])
    with pytest.raises(ValidationError, match="duplicate check"):
        C.RequirementLedger(requirements=[r], checks=[c, c])


def test_verifier_card_pairs_and_ranges():
    assert C.VerifierCard(verifier_version="v1", fp=(0, 40)).fp == (0, 40)
    for kw in ({"fp": (5, 4)}, {"fn": (-1, 4)}, {"mutant_kill": (1, 0)}, {"flake_upper": 1.5}, {"kappa": 2.0}, {"hacks_confirmed": -1}):
        with pytest.raises(ValidationError):
            C.VerifierCard(verifier_version="v1", **kw)
    with pytest.raises(ValidationError):
        C.VerifierCard(verifier_version="v1", kappa=float("nan"))


def test_task_card_status_literal_and_roundtrip():
    with pytest.raises(ValidationError):
        card(status="published")
    c = card()
    assert C.TaskCard.model_validate_json(c.model_dump_json()) == c
    assert c.content_hash() == card().content_hash()
    assert card(prompt=PROMPT + " ").content_hash() != c.content_hash()


def test_label_and_decision_validation():
    ok = {"id": "l1", "item": "it", "item_sha256": SHA, "labeller": "human:a", "value": "pass", "at": NOW}
    assert C.Label(**ok).supersedes is None
    for patch in ({"labeller": "alice"}, {"item_sha256": "abc"}, {"value": ""}, {"at": "soon"}, {"supersedes": "l1"}, {"item": "a b"}):
        with pytest.raises(ValidationError):
            C.Label(**{**ok, **patch})
    d = {"id": "d1", "subject": "t1", "actor": "human:a", "action": "drop", "reason": "r", "criteria": ["amb"], "recipe_version": "r1", "at": NOW}
    assert C.Decision(**d).criteria == ["amb"]
    for patch in ({"action": "maybe"}, {"actor": "bob"}, {"reason": ""}, {"recipe_version": "r 1"}, {"criteria": ["a b"]}):
        with pytest.raises(ValidationError):
            C.Decision(**{**d, **patch})


def test_failure_cause_literal_is_the_closed_set():
    from typing import get_args

    assert set(get_args(C.FailureCause)) == {"skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake"}
    assert ledger()  # fixture sanity
