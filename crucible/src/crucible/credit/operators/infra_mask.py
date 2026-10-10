"""C6, the infrastructure mask (DESIGN P16; research 04 §1.12): "a failed measurement is not a zero".

An episode is masked when anything other than the policy decided its outcome: a failure kind that masks
(``core/schema/run.py``), an answer source the environment leaked (``crucible:answer_exposed``, P27), a verifier that
could not measure (a milestone or gate returned None at the terminal state), or a judge that stayed unparseable after
its retries. A masked episode carries reward ``None``; trainers exclude it from group statistics (and impute the
group mean for its rows), never score it 0 — one −999 left in a GRPO group collapsed the pass/fail advantage gap from
2.15 to 0.004 (verl fork test, research 04).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from crucible.core.schema.reward import InfraStatus
from crucible.core.schema.run import POLICY_OUTCOME_KINDS, is_failure_kind


def mask_for(*, failures: Iterable[Mapping[str, Any]] = (), terminal_phis: Mapping[str, float | None] | None = None,
             gates: Mapping[str, float | None] | None = None, judge_unparseable: bool = False) -> InfraStatus:
    for f in failures:
        kind = str(f.get("kind", ""))
        masks = f.get("mask")
        if masks is None:
            masks = kind not in POLICY_OUTCOME_KINDS
        if masks:
            return InfraStatus(status="masked", failure_kind=kind if is_failure_kind(kind) else "verifier_error", reason=str(f.get("reason", ""))[:300])
    unmeasured = [k for k, v in (gates or {}).items() if v is None]
    if unmeasured:
        return InfraStatus(status="masked", failure_kind="verifier_error", reason=f"gates not measured: {unmeasured[:5]}")
    if terminal_phis is not None and terminal_phis and all(v is None for v in terminal_phis.values()):
        return InfraStatus(status="masked", failure_kind="verifier_error", reason="no milestone could be measured at the terminal state")
    if judge_unparseable:
        return InfraStatus(status="masked", failure_kind="judge_unparseable", reason="judge output unparseable after retries")
    return InfraStatus()
