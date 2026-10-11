"""``reward.json`` (terminal, multi-key) and ``steps.jsonl`` (dense channel), DESIGN §4.3.1–4.3.2.

On disk the full document is ``crucible_reward.json``. ``reward.json`` is its flat numeric projection
(``RewardJson.flat()``: ``reward``, ``outcome``, ``gate/<g>``, ``infra/masked``, ...), because Harbor rejects any
non-numeric value in ``reward.json`` (``harbor/verifier/verifier.py:104-121``; corrected 2026-10-09, M7). NeMo-Gym
consumers read ``reward_components``. ``steps.jsonl`` rows align with ATIF step ids.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument


class InfraStatus(CrucibleModel):
    status: Literal["ok", "masked"] = "ok"
    failure_kind: str | None = None
    reason: str | None = None


class RewardJson(VersionedDocument):
    SCHEMA = "crucible.reward/v1"
    reward: float
    gates: dict[str, int] = Field(default_factory=dict)  # gate -> 0/1
    outcome: float = 0.0
    progress: float | None = None  # Φ(s_T) in [0, 1]
    rubric: dict[str, float] = Field(default_factory=dict)
    efficiency: float | None = None
    infra: InfraStatus = Field(default_factory=InfraStatus)
    components: dict[str, float] = Field(default_factory=dict)  # extra named channels

    @model_validator(mode="after")
    def _consistent(self) -> RewardJson:
        if any(v not in (0, 1) for v in self.gates.values()):
            raise ValueError("gates are 0/1")
        if self.gates and min(self.gates.values()) == 0 and self.reward != 0 and self.infra.status == "ok":
            raise ValueError("a failed gate forces reward 0 (DESIGN §6.4)")
        if self.progress is not None and not 0 <= self.progress <= 1:
            raise ValueError("progress is Φ in [0, 1]")
        return self

    def flat(self) -> dict[str, float]:
        """The numeric ``reward.json`` Harbor accepts."""
        out = {"reward": self.reward, **self.reward_components(), "infra/masked": float(self.infra.status == "masked")}
        return {k: float(v) for k, v in out.items()}

    def reward_components(self) -> dict[str, float]:
        """Flat NeMo-Gym-style components."""
        out: dict[str, float] = {"outcome": self.outcome}
        out.update({f"gate/{k}": float(v) for k, v in self.gates.items()})
        if self.progress is not None:
            out["progress"] = self.progress
        out.update({f"rubric/{k}": v for k, v in self.rubric.items()})
        if self.efficiency is not None:
            out["efficiency"] = self.efficiency
        out.update(self.components)
        return out


class StepRow(CrucibleModel):
    step: int
    atif_step_id: str | None = None
    call_id: str | None = None  # the model call (capture-proxy node / ATIF step) whose sampled tokens this step's credit attaches to
    harness: str | None = None  # harness@version: credit is conditioned on it (one harness per GRPO group)
    actor: str | None = None
    phi: float
    delta_phi: float
    milestones_new: list[str] = Field(default_factory=list)
    milestones_lost: list[str] = Field(default_factory=list)
    violations: list[str] = Field(default_factory=list)
    cost_s: float | None = None
    extra: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _potential(self) -> StepRow:
        if not 0 <= self.phi <= 1:
            raise ValueError("phi is a potential in [0, 1]")
        return self
