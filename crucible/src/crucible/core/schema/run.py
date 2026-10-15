"""Episode / run records and the failure vocabulary.

``FailureKind`` reuses NeMo-Gym's shared vocabulary (``nemo_gym/failure_kinds.py``, Apache-2.0)
verbatim so masks and metrics line up with NeMo-RL / verl consumers, and adds Crucible's own
incident classes (DESIGN Appendix B.5) as namespaced ``crucible:<kind>`` names, the extension form
NeMo-Gym allows. The rule (DESIGN P16): an infrastructure failure is *masked*, never scored 0.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.text import SafeText, safe_text

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
        # added with the M10 audit (2026-10-12)
        "crucible:policy_error",  # the policy's own code or output raised inside the episode (bad tool arguments): scored 0
        "crucible:harness_unsupported",  # the harness cannot run this task (configuration): masked, never retried
        "crucible:replay_incomplete",  # a foreign transcript could not be replayed whole (cut, unmapped tool names): masked
        "crucible:replay_divergence",  # the replayed episode grades differently from the live one: masked
        "crucible:reissue_limit",  # an engine call kept aborting with no sample barrier (M12 audit H3 V9): masked
        # added with the M13 audit (2026-10-13; bench campaign B1, B2, B12)
        "crucible:grader_tampered",  # the policy changed the grader's own toolchain (a shadowed or replaced tool, a startup hook): scored 0
        "crucible:env_container_lost",  # the task container was removed or stopped from outside, or its daemon failed: masked
        "crucible:grader_network_error",  # the grader's own grade-time setup lost the network (container detached, DNS down): masked
        "crucible:env_image_unavailable",  # the task image could not be pulled (missing, access denied, registry down): masked
        "crucible:env_build_failed",  # the task's Dockerfile did not build: masked
    }
)
# Kinds a retry cannot fix (sweep G2 D3: ``retryable`` was never False, so a stale image was retried three times)
NON_RETRYABLE_KINDS = frozenset({"crucible:stale_image", "crucible:stale_build", "crucible:harness_unsupported", "crucible:credits_exhausted", "agent_request_failed",
                                 "crucible:answer_exposed"})

FAILURE_KINDS = NEMO_GYM_FAILURE_KINDS | CRUCIBLE_FAILURE_KINDS
# Kinds that are the policy's own outcome: scored (normally 0), never masked (Karotte: misbehavior is not an infra error).
POLICY_OUTCOME_KINDS = frozenset({"agent_timeout", "agent_command_not_allowed", "crucible:policy_misbehavior", "crucible:agent_broke_env",
                                  "crucible:policy_error", "crucible:grader_tampered"})
FailureStage = Literal["admission", "seed", "agent", "collect", "verification", "cleanup"]
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
    "crucible:policy_error": "policy", "crucible:harness_unsupported": "harness", "crucible:replay_incomplete": "harness",
    "crucible:replay_divergence": "harness", "crucible:reissue_limit": "infra",
    "crucible:grader_tampered": "policy", "crucible:env_container_lost": "infra", "crucible:grader_network_error": "infra",
    "crucible:env_image_unavailable": "infra", "crucible:env_build_failed": "infra",
}


# Pack-registered kinds (``<pack>:<kind>``, NeMo-Gym's namespaced extension form, ``nemo_gym/failure_kinds.py``):
# a pack names its own environment failures without editing core, each with a default blame.
_NAMESPACED = re.compile(r"^[a-z0-9_]+:[a-z0-9_]+$")
_REGISTERED: dict[str, str] = {}
MAX_REASON_CHARS = 2000  # NeMo-Gym caps failure_reason at 2000 characters


def register_failure_kind(name: str, *, blame: str) -> None:
    """Register a pack's failure kind ``<pack>:<kind>`` with its default blame (policy => scored, else masked)."""
    if not _NAMESPACED.match(name) or name.startswith("crucible:"):
        raise ValueError(f"a pack failure kind is '<pack>:<kind>' outside the crucible namespace, got {name!r}")
    if blame not in ("infra", "harness", "task", "policy", "unknown"):
        raise ValueError(f"unknown blame {blame!r}")
    if name in _REGISTERED and _REGISTERED[name] != blame:
        raise ValueError(f"failure kind {name!r} is already registered with blame {_REGISTERED[name]!r}")
    _REGISTERED[name] = blame


def is_failure_kind(name: str) -> bool:
    return name in FAILURE_KINDS or name in _REGISTERED


def default_blame(name: str) -> str:
    return DEFAULT_BLAME.get(name) or _REGISTERED.get(name, "unknown")


def _iso(v: str | None) -> str | None:
    if v is not None:
        datetime.fromisoformat(v)  # raises ValueError on a malformed timestamp
    return v


class Failure(CrucibleModel):
    kind: str
    stage: FailureStage
    reason: str = ""  # occurrence detail; never used as a metric label
    retryable: bool = True
    mask: bool | None = None  # None: derived from blame (policy => scored, else masked)
    blame: Blame | None = None  # None: the kind's default (DEFAULT_BLAME)
    delivery: Literal["not_sent", "possibly_delivered", "delivered"] = "delivered"  # did the model call reach the provider (budget settle)
    detail: SafeText | None = None  # the original sub-kind (e.g. no_reward_json, RewardFileNotFoundError)

    @field_validator("reason")
    @classmethod
    def _bounded(cls, v: str) -> str:
        from crucible.core.secrets import scrub_secrets

        v = safe_text(v)  # an error naming a non-UTF-8 file must not make the record unwritable (R-10)
        v = scrub_secrets(v)  # a reason is stored in records and journals: never a credential (M10 audit V12)
        return v if len(v) <= MAX_REASON_CHARS else v[: MAX_REASON_CHARS - 14] + " …[truncated]"

    @property
    def who(self) -> str:
        return self.blame or default_blame(self.kind)

    @property
    def masks(self) -> bool:
        if self.mask is not None:
            return self.mask
        return self.who != "policy"


class RunRecord(VersionedDocument):
    """One episode of one policy on one task/world."""

    SCHEMA = "crucible.run/v1"

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

    @field_validator("started_at", "ended_at")
    @classmethod
    def _timestamps(cls, v: str | None) -> str | None:
        return _iso(v)

    @model_validator(mode="after")
    def _consistent(self) -> RunRecord:
        if (self.outcome == "masked") != self.mask_sample:
            raise ValueError(f"outcome={self.outcome!r} contradicts mask_sample={self.mask_sample}")
        if self.mask_sample and self.reward is not None:
            raise ValueError("a masked sample has no reward")
        return self

    def mark_failure(self, failure: Failure) -> None:
        if not is_failure_kind(failure.kind):
            raise ValueError(f"unknown failure kind {failure.kind!r}")
        # one consistent update: per-field assignment would pass through states the validator rightly rejects
        reward = None if failure.masks else (0.0 if self.reward is None else self.reward)  # masked: no reward; policy: scored
        for k, v in (("failure", failure), ("outcome", "masked" if failure.masks else "failed"),
                     ("mask_sample", failure.masks), ("reward", reward)):
            object.__setattr__(self, k, v)

    def scored_reward(self) -> float | None:
        """The only accessor aggregates use: None for masked samples."""
        return None if self.mask_sample or self.outcome in ("masked", "running") else self.reward


class LockChange(CrucibleModel):
    """A change of settings a resume was allowed to make (``RunSpec.allow_change``), kept in the lock's history."""

    at: str
    changed: list[str]  # the dotpaths that changed
    before: str  # fingerprint
    after: str


class RunLock(VersionedDocument):
    """What a run is (R-1): the projected settings of its whole ``RunSpec`` (``core.fingerprint``) and their fingerprint.
    Written to ``<run>/lock.json`` before the first dispatch; a resume whose settings differ is refused unless every
    changed setting is allowed by name, and then the change is recorded in ``changes``."""

    SCHEMA: ClassVar[str] = "crucible.run_lock/v1"
    run_id: str
    scheme: str
    fingerprint: str
    excluded: list[str] = Field(default_factory=list)  # operational globs left out of the identity
    settings: dict[str, Any] = Field(default_factory=dict)  # dotpath -> value
    opaque: list[str] = Field(default_factory=list)  # components that declared no settings: only their id counts
    created_at: str
    changes: list[LockChange] = Field(default_factory=list)
