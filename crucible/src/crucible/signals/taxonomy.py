"""The expert-signal taxonomy (DESIGN §5.2, ARCHITECTURE M4): what each generic type means, which way it points,
what it grades, and the prior on how often it is right before any data says otherwise.

Packs never add types; they map their raw events (a review comment, a ticket reopen, an ERP override) onto these.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, get_args

from crucible.core.schema.signal import SignalType


@dataclass(frozen=True)
class SignalKind:
    type: SignalType
    polarity: Literal[-1, 0, 1]  # default direction for the graded step (+1 evidence for, -1 against)
    grades: Literal["step", "trace"]  # a step (an event that changed something) or the whole trace
    description: str
    prior: tuple[float, float] = (1.0, 1.0)  # Beta prior on reliability (agreement with later evidence)


TAXONOMY: dict[str, SignalKind] = {k.type: k for k in [
    SignalKind("outcome", 0, "trace", "terminal outcome of the whole trace (merged and kept, ticket solved, invoice paid)"),
    SignalKind("correction", -1, "step", "someone fixed what the actor did: a (wrong, corrected) pair exists", (3.0, 1.0)),
    SignalKind("rejection", -1, "step", "a reviewer / approver / customer rejected a step or its output"),
    SignalKind("approval", 1, "step", "explicit approval of a step or its output"),
    SignalKind("rework", -1, "step", "the actor had to redo the step (reopened ticket, resubmission, amend)"),
    SignalKind("revert", -1, "step", "the output was rolled back later (revert, recall, refund reversal)", (3.0, 1.0)),
    SignalKind("escalation", -1, "step", "handed to someone more senior or another team"),
    SignalKind("violation", -1, "step", "a written policy or constraint was broken"),
    SignalKind("sla_breach", -1, "step", "took longer than an agreed bound"),
    SignalKind("interrupt", -1, "step", "a human stopped an agent mid-action"),
    SignalKind("suggestion_accepted", -1, "step", "a reviewer's concrete fix was applied verbatim (the graded step was wrong, the fix is known)", (4.0, 1.0)),
    SignalKind("check_result", 0, "step", "an automated check (CI, validation rule) on the state after a step; polarity = pass/fail"),
    SignalKind("target", 0, "trace", "a quantitative target an expert set (perf bar, budget)"),
]}

assert set(TAXONOMY) == set(get_args(SignalType)), "taxonomy and SignalType out of sync"

NEGATIVE = frozenset(t for t, k in TAXONOMY.items() if k.polarity < 0)
