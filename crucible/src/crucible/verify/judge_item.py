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


async def judge_items(llm, items: Sequence[RubricItem], evidence: str, *, max_tokens: int = 2500, votes: int = 1, retries: int = 1) -> dict:
    """Strict binary judgments per validated item; returns ``{"items": {id: bool|None}, "met": share, "unmeasured": [...],
    "mask": bool}``. MiMo's handling (``verify.py:311-382``, M5-M7 sweep): ``votes`` samples per item, an unparseable reply
    is asked again up to ``retries`` times, the verdict is the majority of usable votes (ties: no verdict), and no usable
    vote is unmeasured. An unmeasured **critical** item masks the rubric channel (``mask``); a lesser one only leaves
    the share's denominator. M6 audit: unmeasured items counted as unmet, and nothing called this or tested it."""
    verdicts: dict[str, bool | None] = {}
    for it in admit(items):
        got: list[bool] = []
        for _ in range(max(1, votes)):
            v = None
            for _try in range(1 + max(0, retries)):
                resp = await llm.complete([{"role": "user", "content": it.prompt(evidence)}], max_tokens=max_tokens, temperature=0 if votes == 1 else 0.7)
                v = parse_verdict(getattr(resp, "text", "") or "")
                if v is not None:
                    break
            if v is not None:
                got.append(v)
        yes, no = sum(got), len(got) - sum(got)
        verdicts[it.id] = None if not got or yes == no else yes > no
    unmeasured = [k for k, v in verdicts.items() if v is None]
    critical = {i.id for i in items if i.tier == "critical"}
    return {"items": verdicts, "met": share_met(items, verdicts), "unmeasured": unmeasured, "mask": bool(set(unmeasured) & critical)}
