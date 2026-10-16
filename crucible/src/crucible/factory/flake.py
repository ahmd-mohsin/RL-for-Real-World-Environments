"""Flake (05 §6.6, R12): oracle reruns at several concurrency levels, with the one-sided Clopper–Pearson upper bound on
the false-negative rate on the card (A4's estimator, ``verify.calibrate.clopper_pearson_upper``), and the two sources of
variation kept apart:

- **verifier nondeterminism**: the *same* end state (equal fingerprints) got different verdicts. Always a defect.
- **state variation**: different end states. For oracle replays it is environment nondeterminism; for policy rollouts it
  is agent stochasticity, which is not a flake at all (abc-bench: 113 of 224 tasks stochastic, never separated).

*v1* gate: ≥ 10 reruns at each of ≥ 2 concurrency levels and an upper bound ≤ 5%. Zero failures in n reruns bound the rate
at ``1 - 0.05**(1/n)``: 13.9% at n = 20, 4.9% at n = 60. So the default is 30 reruns per level (60 in all); 10 per level
can never pass the 5% gate.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.factory.floors import Graded, call, graded_from


@dataclass
class FlakeConfig:
    reruns: int = 30  # per concurrency level (see the module doc for why not 10)
    concurrency: tuple[int, ...] = (1, 4)
    min_reruns: int = 10  # v1, per level
    min_levels: int = 2  # v1
    upper_max: float = 0.05  # v1
    alpha: float = 0.05  # one-sided
    pass_at: float = 1.0


def separate(grades: Sequence[Graded], *, pass_at: float = 1.0) -> dict:
    """Split verdict variation into verifier nondeterminism (same fingerprint, different verdicts) and state variation
    (different fingerprints). Grades without a fingerprint cannot be attributed and are counted apart."""
    by_fp: dict[str, set] = {}
    outcome: dict[str, set] = {}
    unattributed = 0
    for g in grades:
        if g.masked:
            continue
        if g.fingerprint is None:
            unattributed += 1
            continue
        by_fp.setdefault(g.fingerprint, set()).add(g.verdict_key())
        outcome.setdefault(g.fingerprint, set()).add(bool(g.passed(pass_at)))
    inconsistent = sorted(fp for fp, v in by_fp.items() if len(v) > 1)
    states_pass = sum(1 for v in outcome.values() if v == {True})
    states_fail = sum(1 for v in outcome.values() if v == {False})
    return {"states": len(by_fp), "verifier_inconsistent_states": inconsistent, "states_all_pass": states_pass, "states_all_fail": states_fail,
            "unattributed": unattributed}


@dataclass
class FlakeReport:
    levels: dict[int, dict]  # concurrency -> {"n", "fails", "masked"}
    n: int
    false_negatives: int
    masked: int
    upper: float
    attribution: dict
    config: FlakeConfig = field(default_factory=FlakeConfig)

    @property
    def verifier_nondeterministic(self) -> bool:
        return bool(self.attribution["verifier_inconsistent_states"])

    @property
    def environment_nondeterministic(self) -> bool:
        """Oracle replays ending in more than one end state."""
        return self.attribution["states"] > 1

    def reasons(self) -> list[str]:
        c, out = self.config, []
        measured = [lv for lv, r in self.levels.items() if r["n"] >= c.min_reruns]
        if len(measured) < c.min_levels:
            out.append(f"{len(measured)} concurrency levels with >= {c.min_reruns} measured reruns, need {c.min_levels}")
        if self.upper > c.upper_max:
            out.append(f"flake upper bound {self.upper:.3f} > {c.upper_max} ({self.false_negatives} oracle false negatives in {self.n})")
        if self.verifier_nondeterministic:
            out.append(f"verifier nondeterminism: same end state, different verdicts ({len(self.attribution['verifier_inconsistent_states'])} states)")
        return out

    @property
    def ok(self) -> bool:
        return not self.reasons()

    def to_dict(self) -> dict:
        return {"levels": self.levels, "n": self.n, "false_negatives": self.false_negatives, "masked": self.masked, "upper": round(self.upper, 5),
                "attribution": self.attribution, "verifier_nondeterministic": self.verifier_nondeterministic,
                "environment_nondeterministic": self.environment_nondeterministic, "ok": self.ok, "reasons": self.reasons()}


def summarize(by_level: dict[int, list[Graded]], *, config: FlakeConfig | None = None) -> FlakeReport:
    from crucible.verify.calibrate import clopper_pearson_upper

    cfg = config or FlakeConfig()
    levels, all_g = {}, []
    for lv, gs in by_level.items():
        meas = [g for g in gs if not g.masked and g.finite]
        levels[lv] = {"n": len(meas), "fails": sum(1 for g in meas if not g.passed(cfg.pass_at)), "masked": len(gs) - len(meas)}
        all_g += gs
    n = sum(r["n"] for r in levels.values())
    fails = sum(r["fails"] for r in levels.values())
    return FlakeReport(levels, n, fails, sum(r["masked"] for r in levels.values()), clopper_pearson_upper(fails, n, cfg.alpha),
                       separate(all_g, pass_at=cfg.pass_at), cfg)


async def measure_flake(run: Callable[[int], Awaitable[Any]], *, config: FlakeConfig | None = None) -> FlakeReport:
    """``run(i)``: one oracle replay graded from a fresh start (``Graded`` or anything ``graded_from`` reads). At each
    concurrency level ``c``, ``reruns`` runs go with at most ``c`` in flight."""
    cfg = config or FlakeConfig()
    by_level: dict[int, list[Graded]] = {}
    base = 0
    for c in cfg.concurrency:
        sem = asyncio.Semaphore(max(1, c))

        async def one(i: int, sem: asyncio.Semaphore = sem) -> Graded:
            async with sem:
                try:
                    return graded_from(await call(run, i))
                except Exception as e:  # noqa: BLE001 - a crashed run is a masked draw, counted apart
                    return Graded(None, {}, None, {"error": str(e)[:200]})

        by_level[c] = list(await asyncio.gather(*(one(base + i) for i in range(cfg.reruns))))
        base += cfg.reruns
    return summarize(by_level, config=cfg)
