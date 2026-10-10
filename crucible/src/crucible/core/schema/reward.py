"""``reward.json`` (terminal, multi-key) and ``steps.jsonl`` (dense channel), DESIGN §4.3.1–4.3.2.

``reward.json`` stays readable by Harbor (it reads the multi-key dict and uses ``reward``) and by
NeMo-Gym consumers (``reward_components``). ``steps.jsonl`` rows align with ATIF step ids.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel


class InfraStatus(CrucibleModel):
    status: Literal["ok", "masked"] = "ok"
    failure_kind: str | None = None
    reason: str | None = None


class RewardJson(CrucibleModel):
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
