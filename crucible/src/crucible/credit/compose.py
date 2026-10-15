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
  term: biased (it changes the objective, P13 type 2), so it is kept small (η), centred to zero net mass within each
  episode (a whole-turn negative shift flattens the policy and lengthens outputs: research 04 §1.4), scaled by the
  group's outcome spread (the MiMo recipes use Dr.GRPO, no std normalisation), bounded by the turn's outcome
  advantage (its sign never flips), never positive on a gate-failed or hacked episode, and absent from a group with
  no outcome spread.

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
        return RewardJson(reward=None, gates=g, outcome=float(outcome), progress=phi, infra=masked, components=dict(components or {}),
                          rubric=dict(rubric_items or {}))
    o = 1.0 if outcome else 0.0
    gate = 1.0 if all(g.values()) else 0.0
    r = gate * (o * (1 + coef.alpha * rubric + coef.beta * efficiency) + coef.kappa * (1 - o) * phi)
    return RewardJson(reward=r, gates=g, outcome=o, progress=phi, rubric={"validated_items_met": rubric, **(rubric_items or {})},
                      efficiency=efficiency, components={"kappa": coef.kappa, "alpha": coef.alpha, "beta": coef.beta, **(components or {})})


def scalar(rj: RewardJson) -> float | None:
    """The training scalar, ``None`` for a masked episode (never 0)."""
    return None if rj.infra.status == "masked" else rj.reward


def without_progress(rj: RewardJson | Mapping[str, object]) -> float | None:
    """The same composition with Φ = 0: the outcome-only reward of a composed one (its gates, outcome, rubric share,
    efficiency and coefficients kept). What ``crucible label`` pays when the partial-credit channel is not validated
    (``credit.channels``); ``None`` for a masked episode."""
    rj = rj if isinstance(rj, RewardJson) else RewardJson.model_validate(rj)
    if rj.infra.status == "masked":
        return None
    c = rj.components
    coef = Coefficients(kappa=c.get("kappa", DEFAULT.kappa), alpha=c.get("alpha", DEFAULT.alpha), beta=c.get("beta", DEFAULT.beta))
    return compose(gates={k: bool(v) for k, v in rj.gates.items()}, outcome=bool(rj.outcome), phi=0.0, rubric=rj.rubric.get("validated_items_met", 0.0),
                   efficiency=rj.efficiency or 0.0, coef=coef).reward


@dataclass
class Episode:
    reward: float | None  # composed R; None = masked
    phis: Sequence[float]  # Φ after each turn t = 0..T-1 (Φ before the first turn is phi0)
    phi0: float = 0.0
    turn_weights: Sequence[float] | None = None  # e.g. completion tokens per turn (zero-mass centring); default 1
    advantages: list[float] = field(default_factory=list)
    gated: bool = False  # a hard gate failed: no positive local credit (M6 audit V3: gate-failed turns earned +0.143)
    hack: bool = False  # a confirmed hack (level-2 judge): same
    vetoed: Sequence[bool] | None = None  # per turn: evidence of wrongdoing at that turn (violations, probes, regressions)


def turn_advantages(group: Sequence[Episode], *, eta: float = 0.1, mode: str = "delta", normalize_outcome: bool = False,
                    zeta: float | None = None) -> list[list[float]]:
    """Per-turn advantages for one GRPO group (all episodes share a prompt/task and a harness).

    ``delta`` mode, after the M6 audit:
    - the local term is centred **within each episode** (it only redistributes the episode's outcome advantage across
      its turns): group-wide centring paid every extra turn when the group's mean ΔΦ was negative, so an oscillating
      episode out-earned a monotone one (0.495 vs 0.254);
    - it is bounded by the turn's outcome advantage and never flips its sign (the docstring promised this; at 800
      turns one turn of a failure out-ranked every turn of the success);
    - a gate-failed or hacked episode gets no positive local term, and a group whose live episodes all have one
      reward gets none at all (it learned from pure process reward);
    - a vetoed turn (AReaL GVPO's veto, fed by Crucible's own step evidence) cannot be positive, and in a flat group
      it gets ``-zeta``."""
    zeta = eta if zeta is None else zeta
    live = [e for e in group if e.reward is not None]
    out: list[list[float]] = []
    if not live:
        return [[0.0] * len(e.phis) for e in group]
    rs = [e.reward for e in live]
    mu = statistics.fmean(rs)  # type: ignore[arg-type]
    sd = statistics.pstdev(rs) if len(rs) > 1 else 0.0  # type: ignore[arg-type]
    terms: dict[int, list[float]] = {}
    for k, e in enumerate(group):
        if e.reward is None:
            continue
        prev = [e.phi0, *e.phis[:-1]]
        if mode == "delta":
            raw = [p - q for p, q in zip(e.phis, prev, strict=True)]
        elif mode == "baseline":
            raw = [-q for q in prev]  # shaping return-to-go with Φ(terminal) = 0
        else:
            raise ValueError("mode is 'delta' or 'baseline'")
        ws = e.turn_weights or [1.0] * len(raw)
        c = sum(w * t for w, t in zip(ws, raw, strict=True)) / sum(ws) if sum(ws) else 0.0
        terms[k] = [t - c for t in raw]
    spread = statistics.pstdev([t for ts in terms.values() for t in ts]) if sum(len(t) for t in terms.values()) > 1 else 0.0
    scale = (sd / spread) if (spread > 0 and sd > 0) else 0.0  # no outcome spread: no process credit
    for k, e in enumerate(group):
        if e.reward is None:
            out.append([0.0] * len(e.phis))
            continue
        a_out = (e.reward - mu) / (sd if normalize_outcome and sd > 0 else 1.0)
        local = [eta * scale * t for t in terms[k]]
        peak = max((abs(x) for x in local), default=0.0)
        if peak > abs(a_out):  # bounded by the outcome advantage (sign never flips), scaled as a whole: zero mass kept
            local = [x * abs(a_out) / peak for x in local] if peak else local
        if e.gated or e.hack:
            local = [min(0.0, x) for x in local]
        row = []
        for i, x in enumerate(local):
            total = a_out + x
            if e.vetoed and i < len(e.vetoed) and e.vetoed[i]:
                total = min(total, 0.0) - (zeta if sd == 0 else 0.0)
            row.append(total)
        out.append(row)
    return out


def turn_credit(group: Sequence[Episode], **kw) -> list[list[float]]:
    """The process part only (per-turn advantage minus the episode's outcome advantage): what a trainer adds on each
    call's tokens next to the sequence advantage it computes itself."""
    adv = turn_advantages(group, **kw)
    live = [e.reward for e in group if e.reward is not None]
    mu = statistics.fmean(live) if live else 0.0
    sd = statistics.pstdev(live) if len(live) > 1 else 0.0
    norm = kw.get("normalize_outcome", False)
    return [[0.0] * len(row) if e.reward is None else [x - (e.reward - mu) / (sd if norm and sd > 0 else 1.0) for x in row]
            for e, row in zip(group, adv, strict=True)]
