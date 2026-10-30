"""The lemma bank of the synthesizer loop (part2-paper-core §9.5) and its acceptance test.

A **lemma** is a design rule abstracted from a (counterexample, repair) pair, written for the roles that build tasks:
the proposer, the contract writer and the repairer (``roles``). It cites the counterexamples it came from, so a lemma
that only restates one task (a task id, an entity, a value) is visible to the leakage critic (``update.leakage_critic``
reads ``text`` as the diff). The bank is versioned data: ``add`` and ``retire`` return a new bank with ``version + 1``,
and it serialises to plain JSON, so ``update.ThetaStore`` versions it as part of the S component.

**Acceptance** (``lemma_acceptance``): a lemma batch is accepted when the **first-round confirmed-counterexample rate**
falls against an **equal-budget control** (the same cells generated without the batch, at the same spend) and J rises
(the caller passes the update rule's verdict on J, R1–R9). The rate test is one of:
- **paired** (the same cells built twice, with and without the batch): an exact one-sided McNemar test on the discordant
  pairs (b: the control had a counterexample and the treated build did not; c: the reverse); significant when
  P(X ≥ b | b + c, ½) < α;
- **unpaired**: the Newcombe hybrid-score interval for the difference of two proportions (from the two Wilson
  intervals); significant when its upper end is below 0.
Unequal spend (beyond ``budget_tol``) refuses the batch outright: a lower rate bought with more budget is not a lemma's
effect (R9).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.synth.canon import content_hash
from crucible.synth.meta.recall import Z95, wilson

Role = Literal["proposer", "writer", "repairer"]


class Lemma(CrucibleModel):
    id: str
    text: str
    roles: list[Role] = Field(default_factory=lambda: ["proposer", "writer", "repairer"])
    sources: list[str] = Field(default_factory=list)  # counterexample ids
    defects: list[str] = Field(default_factory=list)
    round: int = 0


class LemmaBank(CrucibleModel):
    version: int = 1
    lemmas: list[Lemma] = Field(default_factory=list)
    retired: list[str] = Field(default_factory=list)

    def add(self, *lemmas: Lemma) -> LemmaBank:
        have = {x.id for x in self.lemmas}
        dup = [x.id for x in lemmas if x.id in have]
        if dup:
            raise ValueError(f"lemmas already in the bank: {dup}")
        return LemmaBank(version=self.version + 1, lemmas=[*self.lemmas, *lemmas], retired=list(self.retired))

    def retire(self, *ids: str) -> LemmaBank:
        gone = set(ids)
        return LemmaBank(version=self.version + 1, lemmas=[x for x in self.lemmas if x.id not in gone], retired=[*self.retired, *sorted(gone)])

    def for_role(self, role: Role) -> list[Lemma]:
        return [x for x in self.lemmas if role in x.roles]

    def prompt_block(self, role: Role) -> str:
        """The lemmas a role's prompt carries (M4's role prompts include this block)."""
        return "\n".join(f"- {x.text}" for x in self.for_role(role))

    def digest(self) -> str:
        return content_hash(self.model_dump(mode="json"), 24)


def lemma_id(text: str) -> str:
    return "lem-" + content_hash(text.strip(), 12)


# ---------------------------------------------------------------------------------------------------------- acceptance

class Rate(CrucibleModel):
    """First-round confirmed counterexamples: ``k`` of ``n`` tasks had one, at ``spend`` dollars."""

    k: int
    n: int
    spend: float

    @property
    def p(self) -> float:
        return self.k / self.n if self.n else 0.0


class LemmaDecision(CrucibleModel):
    accepted: bool
    reasons: list[str] = Field(default_factory=list)
    treated: tuple[float, float, float] | None = None  # rate, Wilson lo, hi
    control: tuple[float, float, float] | None = None
    diff_interval: tuple[float, float] | None = None
    mcnemar_p: float | None = None


def newcombe(t: Rate, c: Rate, z: float = Z95) -> tuple[float, float]:
    """Newcombe's hybrid-score interval for p_t − p_c (method 10), from the two Wilson intervals."""
    l1, u1 = wilson(t.k, t.n, z)
    l2, u2 = wilson(c.k, c.n, z)
    d = t.p - c.p
    lo = d - math.sqrt((t.p - l1) ** 2 + (u2 - c.p) ** 2)
    hi = d + math.sqrt((u1 - t.p) ** 2 + (c.p - l2) ** 2)
    return lo, hi


def mcnemar_one_sided(pairs: Sequence[tuple[bool, bool]]) -> tuple[int, int, float]:
    """Exact one-sided McNemar on (control_had_cx, treated_had_cx) pairs: (b, c, P(X >= b | b + c, 1/2))."""
    b = sum(1 for ctl, tr in pairs if ctl and not tr)
    c = sum(1 for ctl, tr in pairs if tr and not ctl)
    n = b + c
    if n == 0:
        return b, c, 1.0
    p = sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n
    return b, c, p


def lemma_acceptance(treated: Rate, control: Rate, *, j_rises: bool, pairs: Sequence[tuple[bool, bool]] | None = None, alpha: float = 0.05,
                     budget_tol: float = 0.05) -> LemmaDecision:
    """See the module docstring."""
    why: list[str] = []
    if control.spend <= 0 or abs(treated.spend - control.spend) > budget_tol * control.spend:
        why.append(f"unequal budget: treated {treated.spend:.2f} vs control {control.spend:.2f}")
    lt, ut = wilson(treated.k, treated.n)
    lc, uc = wilson(control.k, control.n)
    dec = LemmaDecision(accepted=False, treated=(treated.p, lt, ut), control=(control.p, lc, uc))
    if pairs is not None:
        _, _, p = mcnemar_one_sided(pairs)
        dec.mcnemar_p = p
        if not p < alpha:
            why.append(f"paired: the rate did not fall (McNemar p = {p:.4f})")
    else:
        z = Z95 if abs(alpha - 0.05) < 1e-12 else math.sqrt(2) * _erfinv(1 - alpha)
        lo, hi = newcombe(treated, control, z)
        dec.diff_interval = (lo, hi)
        if not hi < 0:
            why.append(f"unpaired: the rate did not fall (difference interval {lo:.3f}..{hi:.3f})")
    if not j_rises:
        why.append("J did not rise under the update rule")
    dec.accepted = not why
    dec.reasons = why
    return dec


def _erfinv(y: float) -> float:
    """Inverse error function by bisection (only for a non-default alpha)."""
    lo, hi = -6.0, 6.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if math.erf(mid) < y:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
