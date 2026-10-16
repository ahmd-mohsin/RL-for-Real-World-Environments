"""Shared builders for the M17 owner-P tests."""

from __future__ import annotations

from crucible.factory.cards import CheckRef, ObservationRecord, Requirement, RequirementLedger, TaskCard, VerifierCard

SHA = "a" * 64
NOW = "2026-10-14T12:00:00Z"
PROMPT = ("Process the invoice batch. Refunds over $50 must be approved by a manager. "
          "Never pay a duplicate invoice.\nReply with a one-line summary.")


def obs(id: str = "o1", type: str = "real_pr_tests", by: str = "miner:git") -> ObservationRecord:
    return ObservationRecord(id=id, type=type, pointer=f"store://{id}", sha256=SHA, created=NOW, by=by)


def span(text: str, prompt: str = PROMPT) -> tuple[int, int]:
    i = prompt.index(text)
    return (i, i + len(text))


def ledger() -> RequirementLedger:
    return RequirementLedger(
        requirements=[Requirement(id="r1", text="Refunds over $50 must be approved by a manager.", prompt_span=span("Refunds over $50 must be approved by a manager."), kind="approval"),
                      Requirement(id="r2", text="Never pay a duplicate invoice.", prompt_span=span("Never pay a duplicate invoice."), kind="duplicate")],
        checks=[CheckRef(id="c1", requirement_ids=["r1"], observation_ids=["o1"], gating=True),
                CheckRef(id="c2", requirement_ids=["r2"], observation_ids=["o1", "o2"], gating=True)])


def card(id: str = "t1", status: str = "accepted", family: str = "fam", **kw) -> TaskCard:
    base = {"id": id, "family": family, "intent": "intent1", "surface": {"app": "ap"}, "prompt": PROMPT, "ledger": ledger(),
            "world_ref": "crucible_pack_toy.runtime:HelpdeskTask", "checkset_ref": "checks/t1", "oracle_ref": "oracle/t1",
            "verifier": VerifierCard(verifier_version="v1"), "recipe_version": "r1", "status": status}
    base.update(kw)
    return TaskCard(**base)
