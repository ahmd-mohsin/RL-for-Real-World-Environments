"""Reward composition (DESIGN §6.4) and per-turn advantages.

**Terminal scalar** (what goes into ``reward.json["reward"]`` and the trainer's ``rm_scores``):

    R = G(s_T) · [ O(s_T) · (1 + α·p + β·e)  +  κ · (1 − O(s_T)) · Φ(s_T) ],   κ ≤ 0.3, α ≤ 0.1, β ≤ 0.05

- a failed gate gives 0, so no gate is traded for partial credit;
- a full success is worth at least 1, a failure at most κ < 1;
- partial credit κ(1−O)Φ lives *in the scalar*, so all-fail groups keep variance and survive dynamic-sampling
  filters that evict zero-std groups (verl ``filter_groups``; research 04 §5);
- a masked episode (C6: infrastructure failure, unmeasurable verifier) has reward ``None``: never 0.

**Per-turn advantages.** Two estimators, because they do different things:

- ``mode="baseline"`` — DESIGN's original formula, η·(shaping return-to-go). With Φ(terminal) := 0 and γ = 1 the
  shaping return-to-go from turn t telescopes to −Φ(s_{t−1}): a *state baseline*. It is unbiased and reduces
  variance, but it carries no information about the action at t (it even lowers the advantage of turns taken from
  good states). Use it only as variance reduction.
- ``mode="delta"`` — local credit: η·ΔΦ_t, the potential gained by the turn's own action. This is a process-reward
  term: biased (it changes the objective, P13 type 2), so it is kept small (η), centred to zero net mass within the
  group (a whole-turn negative shift flattens the policy and lengthens outputs: research 04 §1.4), scaled by the group's
  outcome spread (the MiMo recipes use Dr.GRPO, no std normalisation), and it never exceeds the outcome advantage
  of a success over a failure.

Episodes with ``reward is None`` (masked) are excluded from every group statistic and get zero advantage.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from crucible.core.schema.reward import InfraStatus, RewardJson

MAX_KAPPA, MAX_ALPHA, MAX_BETA = 0.3, 0.1, 0.05


@dataclass(frozen=True)
class Coefficients:
    kappa: float = 0.2
    alpha: float = 0.05
    beta: float = 0.02

    def __post_init__(self) -> None:
        if not (0 <= self.kappa <= MAX_KAPPA and 0 <= self.alpha <= MAX_ALPHA and 0 <= self.beta <= MAX_BETA):
            raise ValueError(f"coefficients out of bounds (κ ≤ {MAX_KAPPA}, α ≤ {MAX_ALPHA}, β ≤ {MAX_BETA}): {self}")


DEFAULT = Coefficients()


def compose(*, gates: Mapping[str, bool], outcome: bool, phi: float, rubric: float = 0.0, efficiency: float = 0.0,
            coef: Coefficients | None = None, masked: InfraStatus | None = None,
            rubric_items: Mapping[str, float] | None = None, components: Mapping[str, float] | None = None) -> RewardJson:
    """The gated scalar plus every reported channel. ``phi``, ``rubric``, ``efficiency`` in [0, 1]; ``efficiency``
    may exceed 1 only for record-relative objectives (capped at 2)."""
    coef = coef or DEFAULT
    for name, v in (("phi", phi), ("rubric", rubric)):
        if not 0 <= v <= 1:
            raise ValueError(f"{name} must be in [0, 1], got {v}")
    if not 0 <= efficiency <= 2:
        raise ValueError(f"efficiency must be in [0, 2], got {efficiency}")
    g = {k: int(bool(v)) for k, v in gates.items()}
    if masked is not None and masked.status == "masked":
        return RewardJson(reward=0.0, gates=g, outcome=float(outcome), progress=phi, infra=masked, components=dict(components or {}),
                          rubric=dict(rubric_items or {}))
    o = 1.0 if outcome else 0.0
    gate = 1.0 if all(g.values()) else 0.0
    r = gate * (o * (1 + coef.alpha * rubric + coef.beta * efficiency) + coef.kappa * (1 - o) * phi)
    return RewardJson(reward=r, gates=g, outcome=o, progress=phi, rubric={"validated_items_met": rubric, **(rubric_items or {})},
                      efficiency=efficiency, components={"kappa": coef.kappa, "alpha": coef.alpha, "beta": coef.beta, **(components or {})})


def scalar(rj: RewardJson) -> float | None:
    """The training scalar, ``None`` for a masked episode (never 0)."""
    return None if rj.infra.status == "masked" else rj.reward


@dataclass
class Episode:
    reward: float | None  # composed R; None = masked
    phis: Sequence[float]  # Φ after each turn t = 0..T-1 (Φ before the first turn is phi0)
    phi0: float = 0.0
    turn_weights: Sequence[float] | None = None  # e.g. completion tokens per turn (zero-mass centring); default 1
    advantages: list[float] = field(default_factory=list)


def turn_advantages(group: Sequence[Episode], *, eta: float = 0.1, mode: str = "delta", normalize_outcome: bool = False) -> list[list[float]]:
    """Per-turn advantages for one GRPO group (all episodes share a prompt/task and a harness)."""
    live = [e for e in group if e.reward is not None]
    out: list[list[float]] = []
    if not live:
        return [[0.0] * len(e.phis) for e in group]
    rs = [e.reward for e in live]
    mu = statistics.fmean(rs)  # type: ignore[arg-type]
    sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0  # type: ignore[arg-type]
    # local terms per live episode
    terms: dict[int, list[float]] = {}
    for k, e in enumerate(group):
        if e.reward is None:
            continue
        prev = [e.phi0, *e.phis[:-1]]
        if mode == "delta":
            terms[k] = [p - q for p, q in zip(e.phis, prev, strict=True)]
        elif mode == "baseline":
            terms[k] = [-q for q in prev]  # shaping return-to-go with Φ(terminal) = 0
        else:
            raise ValueError("mode is 'delta' or 'baseline'")
    # zero net mass across the group (weighted by turn tokens when given)
    num = den = 0.0
    for k, ts in terms.items():
        ws = group[k].turn_weights or [1.0] * len(ts)
        num += sum(w * t for w, t in zip(ws, ts, strict=True))
        den += sum(ws)
    centre = num / den if den else 0.0
    spread = statistics.pstdev([t for ts in terms.values() for t in ts]) if sum(len(t) for t in terms.values()) > 1 else 0.0
    scale = (sd if sd > 0 else 1.0) / spread if spread > 0 else 0.0
    for k, e in enumerate(group):
        if e.reward is None:
            out.append([0.0] * len(e.phis))
            continue
        a_out = (e.reward - mu) / (sd if normalize_outcome and sd > 0 else 1.0)
        local = [eta * scale * (t - centre) for t in terms[k]]
        out.append([a_out + x for x in local])
    return out
