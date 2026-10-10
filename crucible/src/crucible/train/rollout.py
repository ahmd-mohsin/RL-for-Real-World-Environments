"""The rollout record contract ``crucible.rollout/v1`` (DESIGN §10.3 point 1; ARCHITECTURE M12 ``rollout.py``).

The trainer's only input from an episode. Per model call (``Turn``): the prompt and completion token ids and the
behaviour logprobs exactly as the inference engine sampled them (token-in-token-out; never detokenized and
re-tokenized), the policy version that generated the span, and the sampling parameters. Per rollout: harness@version,
truncation, compaction segments, timings, infra status, reward channels and per-turn credit, and the World trace id.

``valid_for_training`` is the gate every trainer bridge applies: a rollout missing any token record is masked, never
guessed (the corrections in ``mismatch.py`` and the staleness cap need both the logprobs and the version).
Shapes: verl ``TokenTrace`` / uni_agent ``Trajectory`` with a per-span version (research m9-m10/02 §2, verl row).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

SCHEMA = "crucible.rollout/v1"


class Turn(CrucibleModel):
    call_id: str
    actor: str
    kind: str = "turn"  # turn | compaction (the policy's own summary: trained with the episode's advantage)
    segment: int = 0
    world_step: int | None = None
    # Token storage is linear in the trajectory, not quadratic (research long-horizon/01: verifiers MessageNode): within a
    # segment a call stores only the prompt tokens it adds after the previous call's prompt + completion; the first call of
    # a segment stores its whole (compacted) prompt. ``prefix_break``: the engine's prompt did not extend the previous one.
    prompt_is_delta: bool = False
    prefix_break: bool = False
    prompt_token_ids: list[int] | None = None
    completion_token_ids: list[int] | None = None
    logprobs: list[float] | None = None  # behaviour logprobs, one per completion token
    policy_version: int | str | None = None
    sampling: dict[str, Any] = Field(default_factory=dict)  # temperature, top_p, max_tokens, seed
    finish: str | None = None
    routing: list[list[int]] | None = None  # MoE routing ids per token, when the engine exposes them
    advantage: float | None = None  # per-turn credit from steps.jsonl (M6), attached to this call's tokens
    atif_step_ids: list[int] = Field(default_factory=list)

    def token_complete(self) -> bool:
        n = len(self.completion_token_ids or [])
        return self.prompt_token_ids is not None and n > 0 and self.logprobs is not None and len(self.logprobs) == n and self.policy_version is not None


class Segment(CrucibleModel):
    """A context-compaction segment: the turns generated under one context (``compaction.py``)."""

    index: int
    start_call: str | None = None
    end_call: str | None = None
    summary_call: str | None = None  # the compaction call that opened it
    context_before: int | None = None
    context_after: int | None = None


class RolloutRecord(CrucibleModel):
    schema_: Literal["crucible.rollout/v1"] = Field(SCHEMA, alias="schema")
    rollout_id: str  # the attempt id
    sample: str
    task: str
    harness: str  # harness@version: one harness per GRPO group
    policy: str
    group: str | None = None  # the GRPO group key (task + seed family)
    turns: list[Turn] = Field(default_factory=list)
    segments: list[Segment] = Field(default_factory=list)
    reward: float | None = None  # None = masked: never trained on as 0
    reward_components: dict[str, float] = Field(default_factory=dict)
    gates: dict[str, bool] = Field(default_factory=dict)
    truncated: bool = False
    stop: str | None = None
    failure_kind: str | None = None
    blame: str | None = None
    mask_sample: bool = False
    timings: dict[str, float] = Field(default_factory=dict)
    trace_id: str | None = None
    tier: str = "fast"
    trajectory_tokens: int | None = None  # every token that entered the policy's context over the episode, before compaction
    progress: dict[str, Any] = Field(default_factory=dict)  # dense progress summary (phi0, phi_final, sum_delta, drift)

    def oldest_version(self) -> int | str | None:
        vs = [t.policy_version for t in self.turns if t.policy_version is not None]
        return min(vs, key=lambda v: (isinstance(v, str), v)) if vs else None

    def valid_for_training(self) -> tuple[bool, str]:
        if self.mask_sample or self.reward is None:
            return False, f"masked ({self.failure_kind or 'no reward'})"
        if not self.turns:
            return False, "no model calls recorded"
        bad = [t.call_id for t in self.turns if not t.token_complete() or t.prefix_break]
        if bad:
            return False, f"token record incomplete for {len(bad)}/{len(self.turns)} calls (first: {bad[0]})"
        return True, "ok"
