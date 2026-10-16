"""M17 task-factory records (``docs/research/task-factory/06-build.md``, fixed interfaces; method in 05 §3, §6, §8).

Pure data with validation, no I/O and no framework imports:

- ``ObservationRecord`` (R3): where a ground truth came from. ``first_class`` is false for ``reference_derived`` and
  ``simulator_state``: those may gate only together with executed mutants (05 §6.2).
- ``IntentCard``: a behaviour a policy fails at, with the observed failures that prove it.
- ``Requirement`` / ``CheckRef`` / ``RequirementLedger`` (R1): every requirement in the prompt has a check and every check
  traces to a requirement; ``ledger.py`` holds the two-way check, ``RequirementLedger.problems`` delegates to it.
- ``VerifierCard``: every number with its denominator.
- ``TaskCard``: one task with its ledger, verifier card, stats and lineage.
- ``Label`` / ``Decision``: the append-only human-judgement records stored by ``labels.py``.

Validation here is about shape: ids are ``crucible.core.ids.validate_id`` ids, enums are ``Literal`` types, counts are
``(numerator, denominator)`` pairs with ``0 <= n <= d``, a requirement's ``prompt_span`` lies inside its prompt. Whether
a ledger is *complete* is a ``problems()`` question, not a validation error: a half-built card must be storable.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from crucible.core.ids import validate_id
from crucible.core.schema.base import CrucibleModel

ObservationType = Literal["real_pr_tests", "expert_end_state", "human_graded_outcome", "production_trace",
                          "recorded_session", "observed_model_failure", "reference_derived", "simulator_state"]
SECOND_CLASS: frozenset[str] = frozenset({"reference_derived", "simulator_state"})
FailureCause = Literal["skill_gap", "spec_defect", "unstated_format", "world_bug", "verifier_defect", "budget", "env_flake"]
DecisionAction = Literal["keep", "drop", "revise", "approve", "reject"]

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_ACTOR = re.compile(r"^(human|model|miner):\S+$")
RECIPE_VERSION = re.compile(r"^\S+$")


def check_id(value: str) -> str:
    """A factory id: the shared ``validate_id`` rules, plus no whitespace (ids appear in sentences and JSON keys)."""
    value = validate_id(value)
    if re.search(r"\s", value):
        raise ValueError(f"id {value!r} contains whitespace")
    return value


def check_sha256(value: str) -> str:
    if not _SHA256.match(value):
        raise ValueError(f"expected 64 lower-case hex characters, got {value!r:.80}")
    return value


def check_timestamp(value: str) -> str:
    """An ISO-8601 timestamp (``Z`` accepted), kept as written."""
    try:
        datetime.fromisoformat(value)
    except (ValueError, AttributeError):
        raise ValueError(f"expected an ISO-8601 timestamp, got {value!r:.80}") from None
    return value


def check_actor(value: str) -> str:
    """``human:<id>`` | ``model:<id>`` | ``miner:<id>``."""
    if not _ACTOR.match(value):
        raise ValueError(f"expected 'human:<id>', 'model:<id>' or 'miner:<id>', got {value!r:.80}")
    return value


def _fraction(v: tuple[int, int], what: str) -> tuple[int, int]:
    n, d = v
    if not (0 <= n <= d):
        raise ValueError(f"{what} must be (numerator, denominator) with 0 <= numerator <= denominator, got {v}")
    return v


class ObservationRecord(CrucibleModel):
    """A pointer to a ground-truth source (R3): what, where, a content hash, when, by whom."""

    id: str
    type: ObservationType
    pointer: str = Field(min_length=1)
    sha256: str
    created: str
    by: str  # "human:<id>" | "model:<id>" | "miner:<id>"

    _id = field_validator("id")(check_id)
    _sha = field_validator("sha256")(check_sha256)
    _created = field_validator("created")(check_timestamp)
    _by = field_validator("by")(check_actor)

    @property
    def first_class(self) -> bool:
        return self.type not in SECOND_CLASS


class IntentCard(CrucibleModel):
    """A behaviour worth training: what policies get wrong, with evidence (ObservationRecord ids)."""

    id: str
    behaviour: str = Field(min_length=1)
    failure_mode: str = Field(min_length=1)
    evidence: list[str] = Field(default_factory=list)
    difficulty_axes: list[str] = Field(default_factory=list)
    coverage: dict[str, str] = Field(default_factory=dict)  # {"behaviour": ..., "surface": ...}
    probe_set: str | None = None
    status: Literal["candidate", "approved", "rejected", "retired"] = "candidate"
    approved_by: str | None = None

    _id = field_validator("id")(check_id)

    @field_validator("evidence")
    @classmethod
    def _evidence(cls, v: list[str]) -> list[str]:
        for e in v:
            check_id(e)
        if len(set(v)) != len(v):
            raise ValueError("evidence lists the same observation twice")
        return v

    @field_validator("coverage")
    @classmethod
    def _coverage(cls, v: dict[str, str]) -> dict[str, str]:
        extra = set(v) - {"behaviour", "surface"}
        if extra:
            raise ValueError(f"coverage keys are 'behaviour' and 'surface'; got {sorted(extra)}")
        return v

    @model_validator(mode="after")
    def _approval(self) -> IntentCard:
        if self.status == "approved":
            if not self.approved_by:
                raise ValueError("an approved intent card names who approved it (approved_by)")
            check_actor(self.approved_by)
            if not self.evidence:
                raise ValueError("an approved intent card needs evidence (observed model failures)")
            if set(self.coverage) != {"behaviour", "surface"}:
                raise ValueError("an approved intent card declares coverage: {'behaviour': ..., 'surface': ...}")
        elif self.approved_by is not None:
            raise ValueError(f"approved_by is set on a {self.status!r} card; only an approved card has an approver")
        return self


class Requirement(CrucibleModel):
    """A thing the prompt asks for. ``prompt_span`` is a half-open character range into the task prompt."""

    id: str
    text: str = Field(min_length=1)
    prompt_span: tuple[int, int]
    kind: str = Field(min_length=1)  # keys the probe library

    _id = field_validator("id")(check_id)

    @field_validator("prompt_span")
    @classmethod
    def _span(cls, v: tuple[int, int]) -> tuple[int, int]:
        if v[0] < 0 or v[1] <= v[0]:
            raise ValueError(f"prompt_span must be (start, end) with 0 <= start < end, got {v}")
        return v


class CheckRef(CrucibleModel):
    """A check in the check set: the requirements it enforces and the observations that ground it.
    ``gating`` checks contribute to the reward; others are diagnostic."""

    id: str
    requirement_ids: list[str] = Field(default_factory=list)
    observation_ids: list[str] = Field(default_factory=list)
    gating: bool = False

    _id = field_validator("id")(check_id)

    @field_validator("requirement_ids", "observation_ids")
    @classmethod
    def _ids(cls, v: list[str]) -> list[str]:
        for e in v:
            check_id(e)
        return v


class RequirementLedger(CrucibleModel):
    """R1, enforced both ways. ``problems`` returns one named message per defect; empty means the ledger is sound."""

    requirements: list[Requirement] = Field(default_factory=list)
    checks: list[CheckRef] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique(self) -> RequirementLedger:
        for what, items in (("requirement", self.requirements), ("check", self.checks)):
            ids = [x.id for x in items]
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                raise ValueError(f"duplicate {what} ids: {dup}")
        return self

    def problems(self, observations: dict[str, ObservationRecord], *, prompt: str | None = None) -> list[str]:
        """Every ledger defect, each as ``"<code>: <detail>"`` (codes in ``ledger.CODES``). ``prompt`` enables the span
        checks (span inside the prompt, span text overlaps the requirement); ``TaskCard.ledger_problems`` passes it."""
        from crucible.factory.ledger import problems as _problems

        return _problems(self, observations, prompt=prompt)


class VerifierCard(CrucibleModel):
    """05 §6: every number with its denominator. Pairs are ``(numerator, denominator)``."""

    verifier_version: str = Field(min_length=1)
    floors: dict[str, float] = Field(default_factory=dict)
    mutant_kill: tuple[int, int] = (0, 0)
    alternatives_pass: tuple[int, int] = (0, 0)
    hacks_confirmed: int = Field(default=0, ge=0)
    differential_disagreements: tuple[int, int] = (0, 0)
    fp: tuple[int, int] = (0, 0)
    fn: tuple[int, int] = (0, 0)
    flake_upper: float | None = Field(default=None, ge=0.0, le=1.0)
    kappa: float | None = Field(default=None, ge=-1.0, le=1.0)
    horizon: dict[str, Any] | None = None

    @field_validator("mutant_kill", "alternatives_pass", "differential_disagreements", "fp", "fn")
    @classmethod
    def _pairs(cls, v: tuple[int, int], info: Any) -> tuple[int, int]:
        return _fraction(v, info.field_name)


class TaskCard(CrucibleModel):
    id: str
    family: str
    intent: str  # IntentCard id
    surface: dict[str, str] = Field(default_factory=dict)
    prompt: str = Field(min_length=1)
    ledger: RequirementLedger = Field(default_factory=RequirementLedger)
    world_ref: str
    checkset_ref: str
    oracle_ref: str
    verifier: VerifierCard
    stats: dict[str, Any] = Field(default_factory=dict)  # pass rates with Wilson intervals per (policy, harness, k); cause mix
    lineage: dict[str, Any] = Field(default_factory=dict)
    recipe_version: str = Field(min_length=1)
    status: Literal["draft", "accepted", "rejected", "quarantined", "retired"] = "draft"

    _id = field_validator("id", "family", "intent")(check_id)

    @model_validator(mode="after")
    def _spans_inside_prompt(self) -> TaskCard:
        n = len(self.prompt)
        for r in self.ledger.requirements:
            if r.prompt_span[1] > n:
                raise ValueError(f"requirement {r.id!r}: prompt_span {r.prompt_span} runs past the prompt ({n} characters)")
        return self

    def ledger_problems(self, observations: dict[str, ObservationRecord]) -> list[str]:
        return self.ledger.problems(observations, prompt=self.prompt)


class Label(CrucibleModel):
    """One labeller's verdict on one item at one content hash. Never edited; a changed mind is a new label that
    ``supersedes`` the old one (``labels.LabelStore`` enforces who may supersede what)."""

    id: str
    item: str
    item_sha256: str
    labeller: str
    value: str = Field(min_length=1)
    reason: str = ""
    at: str
    supersedes: str | None = None

    _id = field_validator("id", "item")(check_id)
    _labeller = field_validator("labeller")(check_actor)  # model-vs-human agreement needs the kind of labeller
    _sha = field_validator("item_sha256")(check_sha256)
    _at = field_validator("at")(check_timestamp)

    @field_validator("supersedes")
    @classmethod
    def _supersedes(cls, v: str | None) -> str | None:
        return None if v is None else check_id(v)

    @model_validator(mode="after")
    def _not_self(self) -> Label:
        if self.supersedes == self.id:
            raise ValueError("a label cannot supersede itself")
        return self


class Decision(CrucibleModel):
    """The decision store (outer loop): a keep, drop, revise, approve or reject, with the reason and the named criteria
    that reason falls under, at the recipe version that made it."""

    id: str
    subject: str
    actor: str
    action: DecisionAction
    reason: str = Field(min_length=1)
    criteria: list[str] = Field(default_factory=list)
    recipe_version: str
    at: str

    _id = field_validator("id", "subject")(check_id)
    _actor = field_validator("actor")(check_actor)
    _at = field_validator("at")(check_timestamp)

    @field_validator("recipe_version")
    @classmethod
    def _rv(cls, v: str) -> str:
        if not RECIPE_VERSION.match(v):
            raise ValueError(f"recipe_version must be one token without whitespace, got {v!r:.80}")
        return v

    @field_validator("criteria")
    @classmethod
    def _criteria(cls, v: list[str]) -> list[str]:
        for c in v:
            check_id(c)
        return v
