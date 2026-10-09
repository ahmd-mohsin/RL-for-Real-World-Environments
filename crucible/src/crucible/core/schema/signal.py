"""Expert signals: the generic taxonomy (DESIGN §5.2, ARCHITECTURE M4).

Packs map their raw events (a review comment, an ERP override, a ticket reopen) onto these types.
A signal *grades* one or more trace events; polarity says whether it is evidence for (+1) or
against (-1) the graded step.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

SignalType = Literal[
    "outcome",  # terminal outcome label of the whole trace
    "correction",  # someone fixed what the actor did (paired wrong/right versions)
    "rejection",  # reviewer / approver / customer rejected a step or its output
    "approval",  # explicit approval of a step or output
    "rework",  # the actor had to redo the step (amend, force-push, resubmission)
    "revert",  # the output was rolled back later (revert, recall, refund)
    "escalation",  # handed to someone more senior / another team
    "violation",  # written policy or constraint broken
    "sla_breach",  # took longer than an agreed bound
    "interrupt",  # a human stopped an agent mid-action
    "suggestion_accepted",  # a reviewer's concrete fix was applied verbatim
    "check_result",  # automated check (CI, validation rule) result on a state
    "target",  # a quantitative target set by an expert (perf bar, budget)
]


class SignalTarget(CrucibleModel):
    app: str | None = None
    object: str | None = None  # file path, table:row id, ticket id ...
    span: list[int] | None = None  # e.g. [first_line, last_line]


class Signal(CrucibleModel):
    id: str
    type: SignalType
    polarity: Literal[-1, 0, 1]
    grades: list[int] = Field(default_factory=list)  # event seqs this signal is evidence about
    source_event: int | None = None  # the event seq that carries the signal
    actor: str | None = None  # who produced the signal (reviewer id, ci, customer)
    target: SignalTarget | None = None
    text: str | None = None
    reliability: float | None = None  # estimated by M4 from agreement with later outcomes
    meta: dict[str, Any] = Field(default_factory=dict)
