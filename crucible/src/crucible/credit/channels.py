"""The validity gate on the label path (DESIGN §5.3 filter (d), §6.5): a pack's Φ reaches a dataset only through a
held-out verdict, one per use of it.

A pack's ``credit`` hook gives ``crucible label``, per trace, Φ after each event (``phi_by_seq``) and a reward that may
pay partial credit κ·(1−O)·Φ. Those are two **channels**, judged separately because they test different things:

- **step** (per-turn weights, the ΔΦ ``--mask-negative`` and DPO turn evidence read): Φ part-way through a trace must
  predict the outcome. The statistic is M6 gate S2's, Φ at the midpoint event (``step_statistic``: the one place to
  change when S2 is revised);
- **terminal** (the partial-credit term of the reward): Φ at the end state must predict the outcome.

**Held-out traces** are the ones the hook marks with a split other than ``train``: the hook knows which traces its
milestones were mined from, so only it can name traces they were not mined from. A family without them has no
evidence and nothing is released.

**Decision.** ``validity.validate`` on the held-out traces with a known outcome (bootstrap grouped by the hook's
``group``, default the trace), released iff AUC ≥ 0.6 **and** the 95 % lower bound > 0.5: the rule the rubric-item gate
applies (``operators/rubric_item.py``). The point rule alone passes the toy's end-state Φ at 0.667 on three failures,
with an interval of [0.50, 1.00].

**Fallback** (``apply``): an unreleased step channel trains no Φ (rows carry weight 0.0 with ``credit: false``, the
path of a pack with no dense credit); an unreleased terminal channel recomposes the reward with Φ = 0
(``compose.without_progress``). Per-milestone AUCs are reported beside the verdict, never used as weights: at tens of
traces they are noise (M5 study 2026-10-14, R1).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.credit.compose import without_progress
from crucible.credit.validity import Validity, validate
from crucible.workflow.potential import fraction_index

MIN_AUC = 0.6  # DESIGN §6.5
MIN_LOWER_BOUND = 0.5  # the 95 % lower bound must exclude chance
NO_HOLDOUT = "no held-out traces: every trace is train"
NO_PHI = "the credit hook gives no Φ"


def _phi(phi_by_seq: Mapping[Any, float] | None) -> list[tuple[int, float]]:
    return sorted((int(k), float(v)) for k, v in (phi_by_seq or {}).items())


def step_statistic(seqs: Sequence[int], phi_by_seq: Mapping[Any, float] | None) -> float:
    """Φ at the midpoint event of the trace (``fraction_index(n, 0.5)`` over its events in seq order): the last Φ
    recorded at or before that event, 0.0 if none (nothing measured yet is no progress)."""
    if not seqs:
        return 0.0
    mid = sorted(seqs)[fraction_index(len(seqs), 0.5)]
    return next((v for s, v in reversed(_phi(phi_by_seq)) if s <= mid), 0.0)


def terminal_statistic(row: Mapping[str, Any]) -> float:
    """Φ at the end state: the ``progress`` the reward was composed with when the hook gives ``reward_json``, else the
    last Φ recorded, else 0.0."""
    rj = row.get("reward_json") or {}
    if rj.get("progress") is not None:
        return float(rj["progress"])
    phis = _phi(row.get("phi_by_seq"))
    return phis[-1][1] if phis else 0.0


@dataclass
class ChannelVerdict:
    channel: str  # step | terminal
    statistic: str
    released: bool = False
    reason: str = ""
    validity: Validity | None = None

    def to_dict(self) -> dict[str, Any]:
        v = self.validity.to_dict() if self.validity else {}
        return {"statistic": self.statistic, "released": self.released, "reason": self.reason,
                **{k: v[k] for k in ("auc", "ci", "n", "positives") if k in v}}

    def label(self) -> str:
        return "released" if self.released else f"unreleased: {self.reason}"


@dataclass
class FamilyVerdict:
    family: str
    step: ChannelVerdict
    terminal: ChannelVerdict
    holdout: dict[str, Any] = field(default_factory=dict)
    milestones: dict[str, dict[str, Any]] = field(default_factory=dict)  # reported only

    def to_dict(self) -> dict[str, Any]:
        return {"holdout": self.holdout, "step": self.step.to_dict(), "terminal": self.terminal.to_dict(), "milestones": self.milestones}


def _judge(channel: str, statistic: str, scores: list[float], labels: list[int], groups: list[str]) -> ChannelVerdict:
    v = validate(channel, scores, labels, groups=groups, min_auc=MIN_AUC)
    out = ChannelVerdict(channel, statistic, validity=v)
    lo = v.ci[0]
    if not v.kept:
        out.reason = v.reason
    elif lo is None or lo <= MIN_LOWER_BOUND:
        out.reason = f"95% lower bound {'none' if lo is None else f'{lo:.2f}'} is not above {MIN_LOWER_BOUND}"
    else:
        out.released, out.reason = True, f"held-out AUC {v.auc:.3f}, 95% lower bound {lo:.2f}"
    return out


def decide(family: str, traces: Sequence[Any], credit: Mapping[str, Mapping[str, Any]]) -> FamilyVerdict:
    """The two verdicts for one family. ``traces``: the family's traces with their final outcome labels; ``credit``: the
    hook's rows by trace id. Pure and deterministic (``validate``'s bootstrap is seeded)."""
    rows = [(t, credit[t.trace_id]) for t in traces if t.trace_id in credit and "error" not in credit[t.trace_id]]
    held = [(t, r) for t, r in rows if str(r.get("split") or "train") != "train"]
    known = [(t, r) for t, r in held if t.outcome.label in ("good", "bad")]
    holdout = {"rule": "credit hook split != train", "traces": len(held), "good": sum(t.outcome.label == "good" for t, _ in known),
               "bad": sum(t.outcome.label == "bad" for t, _ in known), "unknown": len(held) - len(known)}
    step, terminal = ChannelVerdict("step", "phi_at_midpoint_event"), ChannelVerdict("terminal", "phi_at_end_state")
    if not any(r.get("phi_by_seq") or (r.get("reward_json") or {}).get("progress") for _, r in rows):
        step.reason = terminal.reason = NO_PHI
        return FamilyVerdict(family, step, terminal, holdout)
    if not held:
        step.reason = terminal.reason = NO_HOLDOUT
        return FamilyVerdict(family, step, terminal, holdout)
    labels = [int(t.outcome.label == "good") for t, _ in known]
    groups = [str(r.get("group") or t.trace_id) for t, r in known]
    step = _judge("step", step.statistic, [step_statistic([e.seq for e in t.events], r.get("phi_by_seq")) for t, r in known], labels, groups)
    terminal = _judge("terminal", terminal.statistic, [terminal_statistic(r) for _, r in known], labels, groups)
    milestones: dict[str, dict[str, Any]] = {}
    for mid in sorted({m for _, r in known for m in (r.get("phi_end_by_milestone") or {})}):
        scores = [float((r.get("phi_end_by_milestone") or {}).get(mid) or 0.0) for _, r in known]
        v = validate(mid, scores, labels, groups=groups, min_auc=MIN_AUC).to_dict()
        milestones[mid] = {k: v[k] for k in ("auc", "ci", "n", "positives")}
    return FamilyVerdict(family, step, terminal, holdout, milestones)


def apply(row: Mapping[str, Any], verdict: FamilyVerdict, *, hook: str) -> tuple[dict[str, Any], dict[Any, float] | None]:
    """(the trace's ``rewards.jsonl`` row, the Φ per event to train on). The row keeps the hook's ``phi_by_seq`` (a
    measurement) and gains ``channels``; an unreleased terminal channel replaces ``reward`` with the Φ = 0 composition
    and keeps the hook's value as ``reward_with_progress``. A hook that pays dense credit without ``reward_json``
    cannot be recomposed: refused, never guessed."""
    out = dict(row)
    if "error" in row:
        return out, None
    out["channels"] = {"step": verdict.step.label(), "terminal": verdict.terminal.label()}
    phi = dict(row["phi_by_seq"]) if verdict.step.released and row.get("phi_by_seq") else None
    if not verdict.terminal.released and row.get("reward") is not None:
        rj = row.get("reward_json")
        if rj is None:
            if row.get("phi_by_seq"):
                raise ValueError(f"credit hook {hook} returned dense credit without reward_json: its partial credit cannot be withheld")
        else:
            reward = without_progress(rj)
            if reward != row["reward"]:
                out["reward"], out["reward_with_progress"] = reward, row["reward"]
    return out, phi
