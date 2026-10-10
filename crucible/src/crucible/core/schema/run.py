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
        "crucible:answer_exposed",  # the policy reached the published fix / hidden tests / future history: the environment leaked (P21)
        "crucible:tool_crashed",  # an App raised outside its contract (a bug in the environment, not the policy's doing)
        "crucible:policy_misbehavior",  # the policy defeated a verification boundary (unkillable cohort, unsafe paths): scored 0
        # added with M10 (2026-10-09; research m9-m10/02 §3.3)
        "crucible:stale_build",  # consumers built against stale headers/config (ABI mismatch, Appendix B.5): the task's fault
        "crucible:harness_self_kill",  # the harness's own control channel killed itself (self-matching pkill -f, B.5)
        "crucible:capture_incomplete",  # a training rollout lacks token ids / behaviour logprobs (crucible.rollout/v1)
        "crucible:agent_broke_env",  # the policy broke its environment (deleted the repo, killed services): scored 0, never masked
    }
)

FAILURE_KINDS = NEMO_GYM_FAILURE_KINDS | CRUCIBLE_FAILURE_KINDS
# Kinds that are the policy's own outcome: scored (normally 0), never masked (Karotte: misbehavior is not an infra error).
POLICY_OUTCOME_KINDS = frozenset({"agent_timeout", "agent_command_not_allowed", "crucible:policy_misbehavior", "crucible:agent_broke_env"})
FailureStage = Literal["admission", "seed", "agent", "verification", "cleanup"]
Blame = Literal["infra", "harness", "task", "policy", "unknown"]
# The blame a kind carries by default (M10 taxonomy; research m9-m10/02 §3.3). policy => scored; everything else => masked.
DEFAULT_BLAME: dict[str, str] = {
    **{k: "infra" for k in NEMO_GYM_FAILURE_KINDS},
    "agent_timeout": "policy", "agent_command_not_allowed": "policy", "agent_request_failed": "harness", "agent_run_error": "harness",
    "verifier_error": "task", "judge_failed": "task", "judge_unparseable": "task",
    "crucible:empty_collection": "harness", "crucible:usage_blank": "infra", "crucible:credits_exhausted": "infra", "crucible:zombie_wait": "infra",
    "crucible:stale_image": "infra", "crucible:node_contention": "infra", "crucible:snapshot_mismatch": "infra", "crucible:backend_error": "infra",
    "crucible:sim_failure": "task", "crucible:answer_exposed": "task", "crucible:tool_crashed": "task", "crucible:policy_misbehavior": "policy",
    "crucible:stale_build": "task", "crucible:harness_self_kill": "harness", "crucible:capture_incomplete": "infra", "crucible:agent_broke_env": "policy",
}


def is_failure_kind(name: str) -> bool:
    return name in FAILURE_KINDS


class Failure(CrucibleModel):
    kind: str
    stage: FailureStage
    reason: str = ""  # occurrence detail; never used as a metric label
    retryable: bool = True
    mask: bool | None = None  # None: derived from blame (policy => scored, else masked)
    blame: Blame | None = None  # None: the kind's default (DEFAULT_BLAME)
    delivery: Literal["not_sent", "possibly_delivered", "delivered"] = "delivered"  # did the model call reach the provider (budget settle)
    detail: str | None = None  # the original sub-kind (e.g. no_reward_json, RewardFileNotFoundError)

    @property
    def who(self) -> str:
        return self.blame or DEFAULT_BLAME.get(self.kind, "unknown")

    @property
    def masks(self) -> bool:
        if self.mask is not None:
            return self.mask
        return self.who != "policy"


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
    cost_uncertain_usd: float = 0.0  # spend that could not be accounted exactly (blank usage), held uncertain in the ledger
    tokens: dict[str, int] = Field(default_factory=dict)
    sample: dict[str, Any] | None = None  # runtime.keys.SampleKey (task, world, policy, harness, sample_idx, seed, tier)
    attempt: str | None = None  # "<sample_id>-a<N>"
    meta: dict[str, Any] = Field(default_factory=dict)

    def mark_failure(self, failure: Failure) -> None:
        if not is_failure_kind(failure.kind):
            raise ValueError(f"unknown failure kind {failure.kind!r}")
        self.failure = failure
        self.outcome = "masked" if failure.masks else "failed"
        self.mask_sample = failure.masks
        if failure.masks:
            self.reward = None  # a masked sample has no reward: no aggregate may read it as 0
        elif self.reward is None:
            self.reward = 0.0  # a policy-caused failure is scored

    def scored_reward(self) -> float | None:
        """The only accessor aggregates use: None for masked samples."""
        return None if self.mask_sample or self.outcome in ("masked", "running") else self.reward
