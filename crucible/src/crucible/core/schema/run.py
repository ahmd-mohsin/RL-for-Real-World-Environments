"""Episode / run records and the failure vocabulary.

``FailureKind`` reuses NeMo-Gym's shared vocabulary (``nemo_gym/failure_kinds.py``, Apache-2.0)
verbatim so masks and metrics line up with NeMo-RL / verl consumers, and adds Crucible's own
incident classes (DESIGN Appendix B.5) as namespaced ``crucible:<kind>`` names, the extension form
NeMo-Gym allows. The rule (DESIGN P16): an infrastructure failure is *masked*, never scored 0.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

# --- NeMo-Gym shared names (verbatim) ---
NEMO_GYM_FAILURE_KINDS = frozenset(
    {
        "transport_unreachable", "transport_peer_drop", "transport_timeout", "transport_local_resource",
        "agent_run_error", "agent_request_failed", "agent_timeout", "agent_command_not_allowed",
        "judge_failed", "judge_unparseable", "verifier_error",
        "provider_unavailable", "provider_quota_exhausted", "provider_oom_killed",
        "session_lost", "session_expired", "session_release_failed",
        "cancelled", "kill_shaped", "shutdown", "persistence_failed", "cohort_incomplete",
    }
)

# --- Crucible incident classes (DESIGN Appendix B.5); all are infrastructure, never policy ---
CRUCIBLE_FAILURE_KINDS = frozenset(
    {
        "crucible:empty_collection",  # artifact/patch collection produced nothing because the hook failed
        "crucible:usage_blank",  # model API returned a response without usage; harness aborted
        "crucible:credits_exhausted",  # model API out of credits / budget stop
        "crucible:zombie_wait",  # PID-1 / reaping problem stalled the episode
        "crucible:stale_image",  # verifier image fingerprint did not match the task's
        "crucible:node_contention",  # perf grading contaminated by foreign load
        "crucible:snapshot_mismatch",  # restore did not reproduce the snapshot fingerprint
        "crucible:backend_error",  # world backend failed to provision / exec
        "crucible:sim_failure",  # a simulated actor broke character / stayed inconsistent (NeMo-Gym masks user_failure)
        "crucible:tool_crashed",  # an App raised outside its contract (a bug in the environment, not the policy's doing)
        "crucible:policy_misbehavior",  # the policy defeated a verification boundary (unkillable cohort, unsafe paths): scored 0
    }
)

FAILURE_KINDS = NEMO_GYM_FAILURE_KINDS | CRUCIBLE_FAILURE_KINDS
# Kinds that are the policy's own outcome: scored (normally 0), never masked (Karotte: misbehavior is not an infra error).
POLICY_OUTCOME_KINDS = frozenset({"agent_timeout", "agent_command_not_allowed", "crucible:policy_misbehavior"})
FailureStage = Literal["admission", "seed", "agent", "verification", "cleanup"]


def is_failure_kind(name: str) -> bool:
    return name in FAILURE_KINDS


class Failure(CrucibleModel):
    kind: str
    stage: FailureStage
    reason: str = ""  # occurrence detail; never used as a metric label
    retryable: bool = True
    mask: bool | None = None  # None: masked unless the kind is a policy outcome (POLICY_OUTCOME_KINDS)

    @property
    def masks(self) -> bool:
        return self.mask if self.mask is not None else self.kind not in POLICY_OUTCOME_KINDS


class RunRecord(CrucibleModel):
    """One episode of one policy on one task/world."""

    run_id: str
    task: str | None = None  # task content hash
    world: str  # world spec content hash
    policy: str  # model / harness id
    seed: int = 0
    started_at: str
    ended_at: str | None = None
    steps: int = 0
    outcome: Literal["completed", "failed", "masked", "running"] = "running"
    failure: Failure | None = None
    mask_sample: bool = False  # True for infrastructure failures (P16)
    reward: float | None = None
    trajectory: str | None = None  # store key of the ATIF trajectory
    final_state: str | None = None  # store key of the final WorldRef
    cost_usd: float = 0.0
    tokens: dict[str, int] = Field(default_factory=dict)
    meta: dict[str, Any] = Field(default_factory=dict)

    def mark_failure(self, failure: Failure) -> None:
        if not is_failure_kind(failure.kind):
            raise ValueError(f"unknown failure kind {failure.kind!r}")
        self.failure = failure
        self.outcome = "masked" if failure.masks else "failed"
        self.mask_sample = failure.masks
