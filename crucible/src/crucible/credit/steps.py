"""The dense channel ``steps.jsonl`` (DESIGN §4.3.2): one row per scored step boundary, aligned with the trace's steps
(M5 segmentation) and keyed by the model call whose sampled tokens the credit attaches to (``call_id``), so trainer
adapters can place per-turn advantages on the right branch even when a harness compacts or spawns subagents
(verifiers v1 message graph; OpenEnv capture graph).

``rows_for`` builds the rows from a Φ trajectory (C1) and the step list; ``write`` / ``read`` serialise them;
``to_turn_shaping`` gives the per-turn ΔΦ list a trainer's ``llm_turn_spans`` operator consumes.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from crucible.core.schema.reward import StepRow
from crucible.credit.operators.potential import PotentialStep
from crucible.workflow.steps import Step


def rows_for(steps: Sequence[Step], potential: Sequence[PotentialStep], *, call_ids: Sequence[str | None] | None = None,
             harness: str | None = None, violations: Sequence[Sequence[str]] | None = None) -> list[StepRow]:
    if len(steps) != len(potential):
        raise ValueError(f"{len(steps)} steps but {len(potential)} potential values")
    out = []
    for i, (s, p) in enumerate(zip(steps, potential, strict=True)):
        out.append(StepRow(step=s.index, call_id=call_ids[i] if call_ids else None, harness=harness, actor=s.actor, phi=round(p.phi, 6),
                           delta_phi=round(p.delta, 6), milestones_new=p.new, milestones_lost=p.lost,
                           violations=list(violations[i]) if violations else [], extra={"kind": s.kind, "partial": p.partial} if p.partial else {"kind": s.kind}))
    return out


def write(rows: Sequence[StepRow], path: Path) -> None:
    path.write_text("".join(r.model_dump_json(exclude_none=True) + "\n" for r in rows))


def read(path: Path) -> list[StepRow]:
    return [StepRow.model_validate(json.loads(x)) for x in path.read_text().splitlines() if x.strip()]


def to_turn_shaping(rows: Sequence[StepRow]) -> list[float]:
    """Per-turn ΔΦ in step order (the ``turn_shaping`` list placed next to ``llm_turn_spans``)."""
    return [r.delta_phi for r in sorted(rows, key=lambda r: r.step)]
