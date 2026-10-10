"""C4, validated rubric items (DESIGN §6.1, §6.5): binary items from written policy or recurring review patterns, that
earn terminal partial credit only after they predict outcomes on later, held-out traces.

Item schema follows MiMo's ``verifier_meta`` (1,427 items over 266 environments, research 04 §1.10): ``tier``
(critical | important | sanity), ``method`` (rule | llm), ``question``, ``pass_anchor`` (the ground truth plus explicit
accept / reject equivalences: the reason binary LLM items reproduce), an optional rule ``check`` (a milestone-style
query), ``weight`` and provenance. Lifecycle: ``candidate`` -> ``validated`` (time-split AUC ≥ 0.6, same direction on
every subset; weight = max(0, 2·(AUC − 0.5))) or ``retired``. Every LLM item ships a known-good and a known-bad
fixture that must score 1 and 0 before it is used (prime-environments' judges were silently blinded by wiring bugs).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.milestone import MilestoneCheck, Provenance
from crucible.credit.validity import validate

ITEM_PROMPT = """You check one requirement of a delivered result. Answer strictly from the evidence shown.

Requirement: {question}
Ground truth and accepted equivalents: {anchor}

Evidence:
{evidence}

Rules: pass only if the evidence shows the requirement is met as delivered; anything missing, truncated or only claimed
in prose fails (say "evidence truncated" if that is the reason). Reply with one line: VERDICT: PASS or VERDICT: FAIL,
then one sentence why."""


class PassAnchor(CrucibleModel):
    truth: str
    accept: list[str] = Field(default_factory=list)
    reject: list[str] = Field(default_factory=list)


class RubricItem(CrucibleModel):
    id: str
    tier: Literal["critical", "important", "sanity"] = "important"
    method: Literal["rule", "llm"] = "rule"
    question: str
    pass_anchor: PassAnchor | None = None
    check: MilestoneCheck | None = None  # method == rule
    weight: float = 0.0
    status: Literal["candidate", "validated", "retired"] = "candidate"
    validity: dict[str, Any] = Field(default_factory=dict)
    fixtures: dict[str, str] = Field(default_factory=dict)  # {"good": evidence that must PASS, "bad": evidence that must FAIL}
    provenance: list[Provenance] = Field(default_factory=list)

    def prompt(self, evidence: str) -> str:
        a = self.pass_anchor
        anchor = "n/a" if a is None else f"{a.truth}; accept: {', '.join(a.accept) or '-'}; reject: {', '.join(a.reject) or '-'}"
        return ITEM_PROMPT.format(question=self.question, anchor=anchor, evidence=evidence)


def parse_verdict(text: str) -> bool | None:
    import re

    m = re.findall(r"VERDICT:\s*(PASS|FAIL)\b", text or "", re.IGNORECASE)
    return None if not m else m[-1].upper() == "PASS"


def lifecycle(item: RubricItem, scores: Sequence[float], labels: Sequence[int], *, subsets: Sequence[Any] | None = None,
              groups: Sequence[Any] | None = None, min_auc: float = 0.6) -> RubricItem:
    """Validate an item on held-out (later) traces: kept items get a weight, the rest are retired."""
    v = validate(item.id, scores, labels, subsets=subsets, groups=groups, min_auc=min_auc)
    return item.model_copy(update={"status": "validated" if v.kept else "retired", "weight": v.weight, "validity": v.to_dict()})


def share_met(items: Sequence[RubricItem], verdicts: dict[str, bool | None]) -> float:
    """p in the reward formula: weighted share of validated items met (unmeasured items count as unmet)."""
    live = [i for i in items if i.status == "validated" and i.weight > 0]
    tot = sum(i.weight for i in live)
    return sum(i.weight for i in live if verdicts.get(i.id)) / tot if tot else 0.0
