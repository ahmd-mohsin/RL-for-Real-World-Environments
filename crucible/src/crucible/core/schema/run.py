"""Episode / run records and the failure vocabulary.

``FailureKind`` reuses NeMo-Gym's shared vocabulary (``nemo_gym/failure_kinds.py``, Apache-2.0)
verbatim so masks and metrics line up with NeMo-RL / verl consumers, and adds Crucible's own
incident classes (DESIGN Appendix B.5) as namespaced ``crucible:<kind>`` names, the extension form
NeMo-Gym allows. The rule (DESIGN P16): an infrastructure failure is *masked*, never scored 0.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any, ClassVar, Literal

from pydantic import Field, field_validator, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument, register_upcaster
from crucible.core.schema.build import BuildInfo
from crucible.core.schema.usage import Timing, UsageTotals
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
        # added with the M0 audit (2026-10-14, R-2)
        "crucible:task_changed",  # the task's content digest changed between plan, dispatch and the end of the attempt: never graded, masked
    }
)
# Kinds a retry cannot fix (sweep G2 D3: ``retryable`` was never False, so a stale image was retried three times)
NON_RETRYABLE_KINDS = frozenset({"crucible:stale_image", "crucible:stale_build", "crucible:harness_unsupported", "crucible:credits_exhausted", "agent_request_failed",
                                 "crucible:answer_exposed", "crucible:task_changed"})

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
    "crucible:env_image_unavailable": "infra", "crucible:env_build_failed": "infra", "crucible:task_changed": "task",
}


# Pack-registered kinds (``<pack>:<kind>``, NeMo-Gym's namespaced extension form, ``nemo_gym/failure_kinds.py``):
# a pack names its own environment failures without editing core, each with a default blame. Each part starts with a
# letter and the whole name must match (R-27; PORT of NeMo-Gym's ``_NAMESPACED_PATTERN`` with ``fullmatch``, Apache-2.0):
# ``^...$`` let "pack:kind\n" (``$`` matches before a final newline) and "9probe:_x" register.
_NAMESPACED = re.compile(r"[a-z][a-z0-9_]*:[a-z][a-z0-9_]*")
_REGISTERED: dict[str, str] = {}
_WARNED: set[str] = set()
_log = logging.getLogger(__name__)
MAX_REASON_CHARS = 2000  # NeMo-Gym caps failure_reason at 2000 characters


def register_failure_kind(name: str, *, blame: str) -> None:
    """Register a pack's failure kind ``<pack>:<kind>`` with its default blame (policy => scored, else masked)."""
    if not isinstance(name, str) or not _NAMESPACED.fullmatch(name) or name.startswith("crucible:"):
        raise ValueError(f"a pack failure kind is '<pack>:<kind>' outside the crucible namespace, got {name!r}")
    if blame not in ("infra", "harness", "task", "policy", "unknown"):
        raise ValueError(f"unknown blame {blame!r}")
    if name in _REGISTERED and _REGISTERED[name] != blame:
        raise ValueError(f"failure kind {name!r} is already registered with blame {_REGISTERED[name]!r}")
    _REGISTERED[name] = blame


def is_failure_kind(name: str) -> bool:
    return name in FAILURE_KINDS or name in _REGISTERED


def default_blame(name: str) -> str:
    """The kind's default blame; an unknown kind (a misspelling, a pack not installed here) is ``unknown``, and masked.
    Refusing it would strand the record, so it is read, with one warning per kind and process (R-27)."""
    blame = DEFAULT_BLAME.get(name) or _REGISTERED.get(name)
    if blame is None:
        if name not in _WARNED:
            _WARNED.add(name)
            _log.warning("failure kind %r is not registered here: its blame is unknown (masked); a misspelling, or its pack is not installed", name)
        return "unknown"
    return blame


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
    # M10 lab audit R-3: {probe: alive|dead|unprobed, control: passed|failed|none, status} on every masked failure (a
    # non-policy blame carries its evidence); M10 lab audit R-5: the (task, harness) breaker counts a task-blamed grade failure and its
    # ``breaker_open`` / ``quarantine`` events report ``control``. None until the harness or the Runner
    # fills it; dropped from canonical JSON while None, so no content hash moves
    evidence: dict[str, Any] | None = Field(default=None, exclude_if=lambda v: v is None)  # crucible.run/v2.1
    # M10 lab audit R-5 item 3: the seconds the provider asked to wait (HTTP Retry-After on a 429 or 503), read by
    # ``RetryRule.backoff`` (a floor under the jittered wait) and by the provider breaker's cool-down. None when unknown;
    # dropped from canonical JSON while None (the ``evidence`` pattern), so no content hash moves
    retry_after: float | None = Field(default=None, ge=0, exclude_if=lambda v: v is None)  # crucible.run/v2.1

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
    """One episode of one policy on one task/world.

    ``crucible.run/v2`` (R-1 step 3): the record names the lock and launch it ran under (``lock.json`` fingerprint,
    ``launches/<n>.json``), and carries the task digest (R-2), token usage and stage timing (R-19) as typed fields; v1's
    never-filled ``tokens`` is gone. ``None`` is unknown: a v1 record, or a producer outside the runner.

    ``crucible.run/v2.1`` (M10 lab audit R-3, R-5): ``Failure.evidence`` and ``Failure.retry_after``, left out while unset."""

    SCHEMA = "crucible.run/v2.1"

    run_id: str
    task: str | None = None  # the task id; its content is task_digest
    task_digest: str | None = None  # the task's content digest at dispatch (R-2); None: the task declares none
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
    usage: UsageTotals | None = None  # the attempt's own model calls summed, with coverage (R-19)
    timing: Timing | None = None  # wall time per stage, with the model's share of the agent stage (R-19)
    sample: dict[str, Any] | None = None  # runtime.keys.SampleKey (task, world, policy, harness, sample_idx, seed, tier)
    attempt: str | None = None  # "<sample_id>-a<N>"
    lock: str | None = None  # the run lock's fingerprint for the launch that wrote this record (lock.json, R-1)
    launch: int | None = Field(None, ge=1)  # that launch: launches/<n>.json (R-1)
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


def _run_v1_to_v2(d: dict[str, Any]) -> dict[str, Any]:
    """``crucible.run/v1`` -> v2: the task digest, usage and timing leave ``meta`` for their fields (a record from before
    R-19 has ``meta.timings``, its stages); ``tokens`` is dropped (no producer ever filled it). A v1 record cannot know
    its lock or launch: they stay None."""
    d, meta = dict(d), dict(d.get("meta") or {})
    d.pop("tokens", None)
    for key in ("task_digest", "usage", "timing"):
        if key in meta:
            d[key] = meta.pop(key)
    stages = meta.pop("timings", None)
    if isinstance(stages, dict) and "timing" not in d:
        d["timing"] = Timing.from_stages(stages).model_dump()
    d["meta"] = meta
    return d


register_upcaster("crucible.run", 1, _run_v1_to_v2)


class LockChange(CrucibleModel):
    """A change of settings a resume made (the run identity accepted it, ``RunSpec.accept_drift``), kept in the lock's history."""

    at: str
    changed: list[str]  # the dotpaths that changed
    before: str  # fingerprint
    after: str


class RunLock(VersionedDocument):
    """What a run is (R-1): the projected settings of its whole ``RunSpec`` (``core.fingerprint``) and their fingerprint.
    Written to ``<run>/lock.json`` before the first dispatch; a resume whose settings differ records the change in
    ``changes`` (refusing one is the run identity's, ``runtime.identity``)."""

    SCHEMA: ClassVar[str] = "crucible.run_lock/v1"
    run_id: str
    scheme: str
    fingerprint: str
    excluded: list[str] = Field(default_factory=list)  # operational globs left out of the identity
    settings: dict[str, Any] = Field(default_factory=dict)  # dotpath -> value
    opaque: list[str] = Field(default_factory=list)  # components that declared no settings: only their id counts
    created_at: str
    changes: list[LockChange] = Field(default_factory=list)


class RunLaunch(VersionedDocument):
    """One invocation of a run (R-1 step 2), in ``<run>/launches/<n>.json``: what the lock leaves out because it may
    change between resumes. The command line (scrubbed), the operational settings used (concurrency, budget, ...),
    the redacted configuration and the code that ran (``core.build``). The lock says what a run is; launches say how
    each resume ran it."""

    SCHEMA: ClassVar[str] = "crucible.run_launch/v1"
    run_id: str
    n: int  # 1 for the first invocation
    started_at: str
    lock: str  # the lock fingerprint this launch ran under
    lock_status: Literal["created", "adopted", "same", "changed"]
    argv: list[str] = Field(default_factory=list)
    operational: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)  # core.config.Config.redacted()
    build: BuildInfo
    python: str
