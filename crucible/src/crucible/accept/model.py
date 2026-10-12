"""M9 data model (research m9-m10/01 §15.2): statuses, needs, costs, evidence, check results, the acceptance key and the
acceptance report.

- ``pass`` / ``fail`` need evidence (a pass without evidence is invalid: Repo2RLEnv ``report.py``).
- ``not_applicable`` only with a reason from the closed ``NA_REASONS`` vocabulary, derived from task facts, never asserted
  by an author (agentenv's skip policy).
- ``pending``: a declared need is unavailable (GPU busy, no paid budget, no human sign-off yet). ``error``: the check
  itself crashed or its infrastructure failed. Neither counts as a pass.
- ``AcceptanceReport.accepted`` is *derived* from results each time it is read, never stored as a boolean.
"""

from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel

CONTRACT_VERSION = "accept-1"


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NA = "not_applicable"
    PENDING = "pending"
    ERROR = "error"


NA_REASONS = frozenset({
    "no_perf_bars",  # A1 headroom: the task has no performance bars
    "no_fresh_gates",  # A1 fresh no-crash, A4 draws: no gate draws a fresh grade-time seed
    "no_tolerance_checks",  # A4 induced draws
    "no_reference_build",  # A4 self-consistency, A9 fingerprints
    "no_images",  # A5 image audit, A9 digests: the task runs in a World, not a container image
    "horizon<H4",  # S1 human sign-off
    "single_tier",  # A12 tier agreement: the plan has no fast tier distinct from full
    "no_agent_surface",  # A13: nothing the policy can submit reaches the grader except World tool calls (covered by A6)
    "not_composite",  # A14 / A15 / A17: the task is not a composition (m13 G-3)
    "dependent_composite",
    "no_property_audit",
    "not_perturbed",  # A18: the task is not a perturbation of another  # A16: no independent audit of the stated properties, so a hack cannot be told from a success  # A17: later components read state earlier ones produce, so parts are not independent
})


class Need(StrEnum):
    DOCKER = "docker"
    GPU = "gpu"
    LAB_GPU = "lab_gpu"
    PAID = "paid"
    HUMAN = "human"
    HARBOR = "harbor"


COST_CLASSES = ("static", "cpu", "docker", "gpu", "paid", "human")


class Cost(CrucibleModel):
    grades: int = 0
    gate_subset: list[str] = Field(default_factory=list)
    gpu_min_per_grade: float = 0.0
    usd: float = 0.0


class Evidence(CrucibleModel):
    path: str  # relative to the acceptance cache dir of the key
    sha256: str
    kind: str = "measured"


class CheckResult(CrucibleModel):
    id: str
    version: int
    status: Status
    measured: dict[str, Any] = Field(default_factory=dict)
    reasons: list[str] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    seconds: float = 0.0
    detectors: list[str] = Field(default_factory=list)  # registry entries this result covers
    key_digest: str | None = None
    na_reason: str | None = None

    @model_validator(mode="after")
    def _rules(self) -> CheckResult:
        if self.status == Status.NA and self.na_reason not in NA_REASONS:
            raise ValueError(f"{self.id}: not_applicable needs a reason from NA_REASONS, got {self.na_reason!r}")
        return self


class AcceptanceKey(CrucibleModel):
    task_id: str
    harbor_hash: str = ""
    overlay_hash: str
    images: dict[str, str] = Field(default_factory=dict)  # role -> image digest
    kit_hash: str
    contract_version: str = CONTRACT_VERSION
    policy_version: str
    registry_version: str

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()).hexdigest()[:24]


class Override(CrucibleModel):
    check: str
    by: str
    reason: str
    date: str
    key_digest: str | None = None  # the acceptance key it was granted for: an override of another key does not apply (V16)


# a human sign-off and the composite checks cannot be overridden either (M9 audit V16)
NO_OVERRIDE = frozenset({"A1", "A2", "A3", "A4", "A5", "A6", "A13", "A14", "A15", "A16", "A17", "A18", "S1"})


class AcceptanceReport(CrucibleModel):
    schema_: Literal["crucible.acceptance/v1"] = Field("crucible.acceptance/v1", alias="schema")
    task_id: str
    key: AcceptanceKey
    key_digest: str
    results: dict[str, CheckResult] = Field(default_factory=dict)
    required: list[str] = Field(default_factory=list)
    overrides: list[Override] = Field(default_factory=list)
    stats: dict[str, Any] = Field(default_factory=dict)

    def blocking(self) -> dict[str, str]:
        """Required checks that keep the task out of the pool, with their status."""
        over = {o.check for o in self.overrides if o.check not in NO_OVERRIDE and o.by.strip() and o.date.strip()
                and o.key_digest in (None, self.key_digest)}
        out = {}
        for c in self.required:
            r = self.results.get(c)
            st = r.status if r else "missing"
            if st not in (Status.PASS, Status.NA) and c not in over:
                out[c] = str(st)
            elif r is not None and r.status in (Status.PASS, Status.FAIL) and not r.evidence:
                out[c] = "no_evidence"
            elif r is not None and r.key_digest not in (None, self.key_digest):
                out[c] = "stale_result"
        return out

    @property
    def accepted(self) -> bool:
        return not self.blocking()

    def current_for(self, key: AcceptanceKey) -> bool:
        return key.digest() == self.key_digest

    def summary(self) -> dict[str, Any]:
        return {"task_id": self.task_id, "key": self.key_digest, "accepted": self.accepted, "blocking": self.blocking(),
                "statuses": {k: str(v.status) for k, v in sorted(self.results.items())}}
