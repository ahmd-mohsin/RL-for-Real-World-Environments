"""LLM-judged rubric items in a grade (DESIGN §6.5, engineering rule 3): **validated items only, never a gate**.

A rubric item (``credit/operators/rubric_item.py``: tier, question, ``pass_anchor``, fixtures) enters a grade only after
its lifecycle validated it on held-out outcomes (time-split AUC >= 0.6, consistent direction). Judged items feed the
``rubric`` channel of ``reward.json`` (the small ``alpha * rubric`` bonus on successes); they can never decide a gate,
because no gate is decided by an LLM. An unparseable verdict is unmeasured (``None``), never a fail.
"""

from __future__ import annotations

from collections.abc import Sequence

from crucible.credit.operators.rubric_item import RubricItem, parse_verdict, share_met


class UnvalidatedItem(ValueError):
    pass


def admit(items: Sequence[RubricItem]) -> list[RubricItem]:
    bad = [i.id for i in items if i.status != "validated"]
    if bad:
        raise UnvalidatedItem(f"rubric items not validated (candidate or retired items never grade): {bad}")
    return list(items)


async def judge_items(llm, items: Sequence[RubricItem], evidence: str, *, max_tokens: int = 2500) -> dict:
    """One strict binary judgment per validated item; returns ``{"items": {id: bool|None}, "met": share}``."""
    verdicts: dict[str, bool | None] = {}
    for it in admit(items):
        resp = await llm.complete([{"role": "user", "content": it.prompt(evidence)}], max_tokens=max_tokens, temperature=0)
        verdicts[it.id] = parse_verdict(getattr(resp, "text", "") or "")
    return {"items": verdicts, "met": share_met(items, verdicts), "unmeasured": [k for k, v in verdicts.items() if v is None]}
