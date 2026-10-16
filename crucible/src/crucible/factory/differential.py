"""Second verifier and differential (05 §6.4; the Rust-rewrite parity pattern): two CheckSets, written independently
(the second by another model family from the same prompt and spec), judge every solver end state. Every disagreement
opens a defect whose owner (prompt, spec, the first verifier or the second) the trace auditor decides; this stage only
lists them, with the requirements on which the two disagree.

Rules:
- the two verdicts must be on the **same end state** (equal fingerprints when both report one): a different end state
  is environment nondeterminism, reported apart (``state_mismatch``), never a disagreement;
- a masked verdict on either side is unmeasured, never agreement;
- per-requirement disagreement uses each CheckSet's own ledger (``check id -> requirement ids``): a requirement holds when
  every check of it passes.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from crucible.factory.floors import Graded, call, graded_from, requirement_map

OWNERS = ("prompt", "spec", "verifier", "second_verifier")
Owner = Literal["prompt", "spec", "verifier", "second_verifier"]


@dataclass
class EndState:
    """One solver end state: its id (trace or episode) and the submission that reproduces it."""

    id: str
    solution: Any
    meta: dict = field(default_factory=dict)


@dataclass
class Disagreement:
    item: str
    primary: bool
    secondary: bool
    requirements: list[str]  # requirements on which the two verdicts differ (when both are ledger-bound)
    primary_failing: list[str]
    secondary_failing: list[str]
    owner: Owner | None = None  # decided by the auditor (``assign``)
    reason: str = ""


@dataclass
class DifferentialReport:
    disagreements: list[Disagreement]
    compared: int
    unmeasured: list[str] = field(default_factory=list)
    state_mismatch: list[str] = field(default_factory=list)
    agree_pass: int = 0
    agree_fail: int = 0

    @property
    def card(self) -> tuple[int, int]:
        """``VerifierCard.differential_disagreements``."""
        return (len(self.disagreements), self.compared)

    def open(self) -> list[Disagreement]:
        return [d for d in self.disagreements if d.owner is None]

    def assign(self, item: str, owner: Owner, reason: str) -> None:
        if owner not in OWNERS:
            raise ValueError(f"owner must be one of {OWNERS}, got {owner!r}")
        for d in self.disagreements:
            if d.item == item:
                d.owner, d.reason = owner, reason
                return
        raise KeyError(item)

    @property
    def ok(self) -> bool:
        return not self.disagreements and not self.state_mismatch and self.compared > 0

    def to_dict(self) -> dict:
        return {"disagreements": [d.__dict__ for d in self.disagreements], "compared": self.compared, "unmeasured": self.unmeasured,
                "state_mismatch": self.state_mismatch, "agree_pass": self.agree_pass, "agree_fail": self.agree_fail, "ok": self.ok}


def _req_status(g: Graded, ledger: Any) -> dict[str, bool]:
    m = requirement_map(ledger)
    out: dict[str, bool] = {}
    for c, ok in g.checks.items():
        for r in m.get(c, ()):
            out[r] = out.get(r, True) and ok
    return out


def compare(item: str, a: Graded, b: Graded, *, ledger_a: Any = None, ledger_b: Any = None, pass_at: float = 1.0) -> Disagreement | None | str:
    """One item: a ``Disagreement``, ``None`` (agreement), ``"unmeasured"`` or ``"state_mismatch"``."""
    if a.masked or b.masked or not a.finite or not b.finite:
        return "unmeasured"
    if a.fingerprint is not None and b.fingerprint is not None and a.fingerprint != b.fingerprint:
        return "state_mismatch"
    pa, pb = bool(a.passed(pass_at)), bool(b.passed(pass_at))
    reqs: list[str] = []
    if ledger_a is not None and ledger_b is not None:
        ra, rb = _req_status(a, ledger_a), _req_status(b, ledger_b)
        reqs = sorted(r for r in set(ra) & set(rb) if ra[r] != rb[r])
    if pa == pb and not reqs:
        return None
    return Disagreement(item, pa, pb, reqs, a.failing(), b.failing())


async def differential(states: Sequence[EndState], primary: Callable[[Any], Awaitable[Any]], secondary: Callable[[Any], Awaitable[Any]], *,
                       ledger_a: Any = None, ledger_b: Any = None, pass_at: float = 1.0) -> DifferentialReport:
    """Judge every end state with both verifiers (``primary(solution)``, ``secondary(solution)``)."""
    out = DifferentialReport([], 0)
    for s in states:
        a, b = graded_from(await call(primary, s.solution)), graded_from(await call(secondary, s.solution))
        r = compare(s.id, a, b, ledger_a=ledger_a, ledger_b=ledger_b, pass_at=pass_at)
        if r == "unmeasured":
            out.unmeasured.append(s.id)
            continue
        if r == "state_mismatch":
            out.state_mismatch.append(s.id)
            continue
        out.compared += 1
        if isinstance(r, Disagreement):
            out.disagreements.append(r)
        elif a.passed(pass_at):
            out.agree_pass += 1
        else:
            out.agree_fail += 1
    return out
