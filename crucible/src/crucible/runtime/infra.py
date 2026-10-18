"""Failure classification, retry rules, limit policy, and aggregates that cannot read a masked sample as 0
(research m9-m10/02 §3.3; DESIGN P16, Appendix B.5).

Every surveyed runner scores some infrastructure failures as 0 (Harbor ``metrics/base.py``, verifiers ``trace.py``,
agentenv's missing reward file, mini-swe-agent's batch resume...). Here:

- a failure has a ``blame`` (infra, harness, task, policy, unknown). Only ``policy`` is scored; everything else is
  masked with ``reward = None``;
- ``normalize`` maps foreign and legacy names (``no_reward_json``, ``check_infra``, ``grader_exception``, Harbor exception
  names, MiMo categories) onto the shared vocabulary, so ``RunRecord.mark_failure`` never raises on a known source;
- ``boundary(stage)`` wraps a run phase and turns exceptions into typed failures (pattern: verifiers ``boundary()``),
  walking ``__cause__`` up to 8 links for a classifiable cause (M10 lab audit R-3; pattern: NeMo-RL ``failures.py``);
  ``classify_exception`` is that classification without the raise;
- ``episode_failure`` rebuilds an episode's failure dict as a ``Failure`` with its blame, mask and retryable kept;
- ``RetryRule`` retries only infra/harness/task blame (never policy: "retry until ok" is survivorship bias), and a
  verification-stage failure only as a re-grade of persisted work (M10 lab audit R-3: never a fresh agent run);
- ``LimitPolicy`` decides whether hitting a wall-clock/step/context limit is scored (default) or masked (Karotte);
- the aggregates (``pass_at_k``, ``mean_reward``, ``coverage``) drop masked samples from numerators **and**
  denominators and report coverage = scored / planned.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field

from crucible.core.schema.run import DEFAULT_BLAME, NON_RETRYABLE_KINDS, Failure, RunRecord, is_failure_kind

NORMALIZE = {
    # Crucible verifier kinds (M7)
    "no_reward_json": ("verifier_error", "no_reward_json"), "check_infra": ("verifier_error", "check_infra"),
    "grader_exception": ("verifier_error", "grader_exception"), "verifier_timeout": ("verifier_error", "verifier_timeout"),
    # M10 lab audit R-3 item 2: a reward the grader could not turn into a number (a NaN or infinite reward file or kit mask,
    # named so by ``HarborDirTask.grade_patch``): answer-sensitive, decided by the task's control
    "non_finite_reward": ("verifier_error", "non_finite_reward"),
    # Harbor exception names
    "RewardFileNotFoundError": ("verifier_error", "RewardFileNotFoundError"), "RewardFileEmptyError": ("verifier_error", "RewardFileEmptyError"),
    "VerifierOutputParseError": ("verifier_error", "VerifierOutputParseError"), "VerifierTimeoutError": ("verifier_error", "VerifierTimeoutError"),
    "AgentTimeoutError": ("agent_timeout", "AgentTimeoutError"), "EnvironmentStartTimeoutError": ("crucible:backend_error", "EnvironmentStartTimeoutError"),
    # Crucible World readiness (M1 item 6)
    "HealthcheckError": ("crucible:backend_error", "HealthcheckError"), "ResetHookError": ("crucible:backend_error", "ResetHookError"),
    # an App's seed copy failed (M14 campaign B3: a shared seed tarball was masked with unknown blame)
    "WorldSeedError": ("crucible:backend_error", "WorldSeedError"),
    "ApiInternalServerError": ("provider_unavailable", "ApiInternalServerError"), "ApiOverloadedError": ("provider_unavailable", "ApiOverloadedError"),
    # M10 lab audit R-1: Harbor's agent-setup deadline (trial/errors.py:4) is the harness's install, never the policy's; a
    # context or output overflow is a scored limit stop, as Crucible's own ContextOverflow is. Harbor's other regex classes
    # (ApiRateLimitError, ApiUsageLimitError, ApiConnectionClosedError, AgentAuthenticationError, ModelNotFoundError,
    # NetworkConnectionError, NonZeroAgentExitCodeError; agents/installed/base.py:34-170) are deliberately NOT mapped: they
    # read the agent's own output, and a non-policy blame needs proxy or host evidence (``runtime.agent_evidence``).
    # Unmapped, an evidence-free name stays unknown blame here
    "AgentSetupTimeoutError": ("crucible:backend_error", "AgentSetupTimeoutError"),
    "ContextWindowExceededError": ("agent_timeout", "context_overflow"), "OutputTokenExceededError": ("agent_timeout", "context_overflow"),
    # MiMo verl categories
    "setup/failed": ("crucible:backend_error", "mimo:setup/failed"), "rollout/pod_conn_timeout": ("transport_peer_drop", "mimo:pod_conn_timeout"),
    # M11 importers and clients (M11 audit G5 D10: these normalized to unmapped and counted toward the unknown-blame stop)
    "ambiguous_reward": ("verifier_error", "ambiguous_reward"), "no_reward": ("verifier_error", "no_reward"), "mask_sample": ("verifier_error", "mask_sample"),
    "AddTestsDirError": ("crucible:backend_error", "AddTestsDirError"), "DownloadVerifierDirError": ("crucible:backend_error", "DownloadVerifierDirError"),
    "env_unreachable": ("transport_unreachable", "env_unreachable"), "env_session_lost": ("session_lost", "env_session_lost"),
    "env_admission_timeout": ("transport_timeout", "env_admission_timeout"), "env_seed_failed": ("crucible:backend_error", "env_seed_failed"),
    "env_server_error": ("crucible:backend_error", "env_server_error"),
    # LLM client
    "UsageBlank": ("crucible:usage_blank", "UsageBlank"), "BudgetExceeded": ("crucible:credits_exhausted", "BudgetExceeded"),
    "ContextOverflow": ("agent_timeout", "context_overflow"),
}


def normalize(kind: str) -> tuple[str, str | None]:
    """Foreign or legacy name -> (shared kind, detail). A kind a pack registered keeps its name and blame (sweep G3 D5:
    it was mapped to ``agent_run_error``, so a policy-blamed pack kind was masked and retried)."""
    if is_failure_kind(kind):
        return kind, None
    if kind in NORMALIZE:
        return NORMALIZE[kind]
    return "agent_run_error", f"unmapped:{kind}"


def failure(kind: str, stage: str, reason: str = "", **kw) -> Failure:
    k, detail = normalize(kind)
    detail = kw.pop("detail", None) or detail
    if detail and detail.startswith("unmapped:") and "blame" not in kw:
        kw["blame"] = "unknown"  # an unmapped foreign name is unclassified: counted by the unknown-blame stop rule (sweep G5)
    return Failure(kind=k, stage=stage, reason=reason[:500], detail=detail, **kw)  # type: ignore[arg-type]


class StageFailure(Exception):
    """Raised by run phases with an already-classified failure."""

    def __init__(self, f: Failure):
        super().__init__(f"{f.kind}@{f.stage}: {f.reason}")
        self.failure = f


def _unencodable_text(ex: BaseException) -> bool:
    """A serialization error caused by a string with a lone surrogate (``UnicodeEncodeError``, or pydantic's
    ``PydanticSerializationError`` wrapping one)."""
    if isinstance(ex, UnicodeEncodeError):
        return True
    return type(ex).__name__ == "PydanticSerializationError" and "surrogates not allowed" in str(ex)


_MAX_CAUSE_DEPTH = 8  # NeMo-RL failures.py:76-78: bounded, so a self-referential or pathologically deep chain cannot spin


def _causes(ex: BaseException) -> list[BaseException]:
    """``ex``, ``ex.__cause__``, ... : at most ``_MAX_CAUSE_DEPTH`` links, stopping at an exception already seen (a cycle).
    Only the explicit ``raise ... from`` link: an implicit ``__context__`` is whatever happened to be handled nearby."""
    out: list[BaseException] = []
    cur: BaseException | None = ex
    while cur is not None and len(out) < _MAX_CAUSE_DEPTH and not any(cur is seen for seen in out):
        out.append(cur)
        cur = cur.__cause__
    return out


def classify_exception(ex: BaseException, stage: str, *, unknown_kind: str = "agent_run_error") -> Failure:
    """The typed failure ``boundary(stage)`` raises for ``ex`` (its rules, below). M10 lab audit R-3: a function, so a
    caller that must decide before re-raising (``grading.grade_in_place``: is another grade of this work worth it?) reads
    the same classification the boundary would give."""
    name = type(ex).__name__
    if stage == "agent" and _unencodable_text(ex):
        # M14 campaign B1: a string the policy produced that UTF-8 cannot encode (a lone surrogate) reached a writer the
        # World's own escaping did not cover. The policy's output, not the environment: scored 0, never an unknown mask
        return failure("crucible:policy_misbehavior", stage, f"unencodable text in the policy's output: {name}: {ex}"[:500], mask=False)
    for i, link in enumerate(_causes(ex)):
        lname = type(link).__name__
        via = "" if i == 0 else f" (cause of {name}: {ex})"
        if isinstance(link, StageFailure):  # a typed failure wrapped by a caller that did not know it: its own classification
            return link.failure
        if lname in NORMALIZE:
            return failure(lname, stage, f"{link}{via}"[:500])
        if isinstance(link, TimeoutError):
            return failure("transport_timeout", stage, f"{link}{via}"[:500])
        if isinstance(link, (ConnectionError, OSError)):
            return failure("transport_unreachable", stage, f"{lname}: {link}{via}"[:500])
    return Failure(kind=unknown_kind, stage=stage, reason=f"{name}: {ex}"[:500], blame="unknown")  # type: ignore[arg-type]


@contextlib.contextmanager
def boundary(stage: str, *, unknown_kind: str = "agent_run_error") -> Iterator[None]:
    """Typed failures pass through; known exception classes are mapped; anything else is ``unknown`` blame (masked and
    counted against the unknown-fraction stop rule). The classification is ``classify_exception``'s.

    M10 lab audit R-3 (pattern: NeMo-RL ``experience/failures.py:76-81,141-264``, classified on the side that raised): the
    non-policy rules (a wrapped ``StageFailure``, a NORMALIZE name, ``TimeoutError``, ``ConnectionError``/``OSError``) are
    tried on ``ex`` and then on each ``__cause__`` link, at most 8 links, the first match winning: a ``RuntimeError`` raised
    ``from`` a ``VerifierTimeoutError`` or a socket error was an unknown mask. The policy-blaming unencodable-text rule
    applies to ``ex`` itself only: an infrastructure exception wrapping a ``UnicodeEncodeError`` of a host path or a backend
    message is not the policy's output, so it stays unknown rather than becoming the policy's 0."""
    try:
        yield
    except StageFailure:
        raise
    except Exception as ex:
        raise StageFailure(classify_exception(ex, stage, unknown_kind=unknown_kind)) from ex


def episode_failure(f: dict, stage: str = "agent", prefix: str = "") -> Failure:
    """An episode's failure dict (``EpisodeResult.failures``: a responder's, a policy's, the World's dumped ``Failure``) as
    a ``Failure``, keeping what the episode already classified.

    M10 lab audit R-3 (b): ``_masked_episode`` and the workspace harness rebuilt ``failure(kind)`` without the blame, so an
    unclassified policy exception (``actors/policy.py:586-587``, blame unknown) fell back to ``agent_run_error``'s harness
    default, was retried, and ``Health.unknown`` never counted it. Only the keys that are present (not None) are passed, so
    ``failure()``'s unmapped -> unknown rule and the kind's default blame still apply when the dict names no blame."""
    keep = {k: f[k] for k in ("blame", "mask", "retryable", "detail", "delivery") if f.get(k) is not None}
    return failure(str(f["kind"]), stage, prefix + str(f.get("reason", "")), **keep)


@dataclass
class RetryRule:
    kinds: tuple[str, ...] = ()  # empty = any kind of the allowed blames
    blames: tuple[str, ...] = ("infra", "harness", "task")
    max_attempts: int = 3

    def allows(self, f: Failure, attempt: int, *, regrade: bool = False) -> bool:
        """``regrade``: the attempt persisted its work for a grade-only retry (``HarnessResult.regradable``)."""
        if f.who == "policy" or not f.retryable or attempt >= self.max_attempts or f.kind in NON_RETRYABLE_KINDS:
            return False
        if f.kind == "agent_timeout":  # a limit stop masked by LimitPolicy: re-running until it fits is survivorship (G3 D6)
            return False
        if f.stage == "verification" and not regrade:
            # M10 lab audit R-3: a failure after the agent phase is retried only by grading the same work again. NativeHarness
            # had no re-grade, so a masked grade re-ran the agent: a policy whose answer hung the tests escaped its 0 and
            # pass@k became conditional on "did not hang". Without persisted work the attempt ends terminal-masked
            # (fails closed for every caller that does not say otherwise)
            return False
        return f.who in self.blames and (not self.kinds or f.kind in self.kinds)


@dataclass
class LimitPolicy:
    on_timeout: str = "score"  # "score" (policy's outcome) | "mask" | "penalty:<c>" (m13 C-8, A2-24: the reward becomes c)
    on_context_window: str = "score"
    on_steps: str = "score"

    def for_stop(self, stop: str) -> str:
        return {"timeout": self.on_timeout, "max_steps": self.on_steps, "context_overflow": self.on_context_window}.get(stop, "score")

    def penalty(self, stop: str) -> float | None:
        """The reward a limit stop gets under ``"penalty:<c>"``; None when the policy scores or masks it."""
        rule = self.for_stop(stop)
        if not rule.startswith("penalty:"):
            return None
        return float(rule.split(":", 1)[1])


# --------------------------------------------------------------------------------------------- aggregates


def scored(records: Iterable[RunRecord]) -> list[RunRecord]:
    return [r for r in records if r.scored_reward() is not None]


def mean_reward(records: Iterable[RunRecord]) -> float | None:
    s = [r.scored_reward() for r in scored(records)]
    return sum(s) / len(s) if s else None


def coverage(records: Iterable[RunRecord], planned: int) -> float:
    return len(scored(records)) / planned if planned else 0.0


def pass_at_k(records: Iterable[RunRecord], k: int, *, threshold: float = 1.0) -> float | None:
    """Unbiased pass@k per sample group (task, policy, harness), averaged; masked samples are not samples."""
    groups: dict[tuple, list[float]] = {}
    for r in scored(records):
        sk = r.sample or {}
        groups.setdefault((sk.get("task", r.task), sk.get("policy", r.policy), sk.get("harness")), []).append(r.scored_reward())
    vals = []
    for rewards in groups.values():
        n, c = len(rewards), sum(1 for x in rewards if x >= threshold)
        if n < k:
            continue
        vals.append(1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k))
    return sum(vals) / len(vals) if vals else None


@dataclass
class Health:
    planned: int
    terminal: int = 0
    masked: int = 0
    unknown: int = 0
    by_kind: dict[str, int] = field(default_factory=dict)

    def add(self, r: RunRecord) -> None:
        self.terminal += 1
        if r.mask_sample:
            self.masked += 1
            k = r.failure.kind if r.failure else "?"
            self.by_kind[k] = self.by_kind.get(k, 0) + 1
            if r.failure and r.failure.who == "unknown":
                self.unknown += 1

    def stop_reason(self, *, max_mask_fraction: float = 0.2, max_unknown_fraction: float = 0.05, min_terminal: int = 20) -> str | None:
        if self.terminal < min_terminal:
            return None
        if self.masked / self.terminal > max_mask_fraction:
            return f"mask fraction {self.masked}/{self.terminal} > {max_mask_fraction}: the environment is broken, not noisy ({self.by_kind})"
        if self.unknown / self.terminal > max_unknown_fraction:
            return f"unknown-blame fraction {self.unknown}/{self.terminal} > {max_unknown_fraction}: the classifier needs a rule"
        return None


def classify_verifier(infra: dict) -> Failure:
    """A masked grade -> a typed failure: contamination is the node's, a stale image or build the task's, else
    ``verifier_error`` with the original kind kept as detail."""
    reason = (infra.get("reason") or "").lower()
    kind = infra.get("failure_kind") or "grader_exception"
    if "contaminated" in reason or "never quiet" in reason or "node_contention" in reason:
        return failure("crucible:node_contention", "verification", infra.get("reason") or "", detail=kind)
    if "fingerprint" in reason or "stale image" in reason or "stale_image" in reason:
        return failure("crucible:stale_image", "verification", infra.get("reason") or "", detail=kind)
    if "stale build" in reason or "stale_build" in reason or "abi" in reason.split():
        return failure("crucible:stale_build", "verification", infra.get("reason") or "", detail=kind)
    return failure(kind, "verification", infra.get("reason") or "")


def default_blame(kind: str) -> str:
    return DEFAULT_BLAME.get(kind, "unknown")
