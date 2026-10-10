"""Versioned acceptance thresholds (research m9-m10/01 §15.1 ``policy.py``). The policy version is a hash of the
values, so changing any threshold makes every acceptance non-current (it is part of the acceptance key)."""

from __future__ import annotations

import hashlib
import json

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

ALL_CHECKS = ("A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10", "A11", "A12", "A13", "S1")


class Policy(CrucibleModel):
    name: str = "accept-v1"
    a1_grades: int = 3
    a1_fresh_grades: int = 10
    a1_headroom: float = 0.15
    a2_exact_zero: bool = True
    a3_mutation_min: float = 0.9
    a3_min_negatives_h3: int = 4
    a4_draws: int = 40
    a4_stable_vectors: int = 8
    a4_reference_runs: int = 3
    a8_min_share: float = 0.8
    a10_min_runs: int = 2
    a10_max_runs: int = 8  # sequential top-up cap: a point estimate outside the band whose interval still overlaps it samples more first
    a12_spearman: float = 0.8
    a12_timeout_margin: float = 0.8  # the slowest full grade must fit within this share of the verifier timeout
    infra_retries: int = 2
    required: list[str] = Field(default_factory=lambda: list(ALL_CHECKS))

    def version(self) -> str:
        return f"{self.name}:{hashlib.sha256(json.dumps(self.model_dump(mode='json'), sort_keys=True).encode()).hexdigest()[:12]}"


DEFAULT = Policy()
