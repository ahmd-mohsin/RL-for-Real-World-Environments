"""Sample identity (research m9-m10/02 §3.2): what a run's resume keys on, which no surveyed framework has.

A ``SampleKey`` names one planned sample: (task, world, policy, harness@version#config digest, sample_idx, seed, tier; M10 lab
audit R-4: the digest makes a harness with other limits another harness, so another sample). Its id is a
hash of the canonical JSON, so the same plan always yields the same ids and a re-run resumes instead of re-sampling.
Seeds are derived, not counted: ``derive_seed(run_seed, task, policy, harness, idx)`` gives every sample its own sampling
seed (sent to the endpoint and used as the LLM cache key), so k samples never collapse onto one replayed completion.
An attempt of a sample is ``<sample_id>-a<N>`` (NeMo-Gym ``rollout_id-a<N>``); sandboxes, containers, capture sessions
and logs are named after it.
"""

from __future__ import annotations

import hashlib
import json
from typing import Literal

from crucible.core.schema.base import CrucibleModel


def _h(obj, n: int = 16) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:n]


def derive_seed(run_seed: int, task: str, policy: str, harness: str, idx: int) -> int:
    return int(_h([run_seed, task, policy, harness, idx], 15), 16) % (2**31 - 1)


class SampleKey(CrucibleModel):
    task: str  # task id or content hash, pinned version
    world: str = ""  # world spec hash
    policy: str  # model id or checkpoint@policy_version
    harness: str  # "native:llm-tool@1#<digest>" | "native:bash-loop@1#<digest>" | "harbor:mini-swe-agent@2.4.6#<digest>" | ... (M10 lab audit R-4)
    sample_idx: int
    seed: int
    tier: Literal["fast", "full"] = "full"

    def id(self) -> str:
        return _h(self.model_dump(mode="json"))


def attempt_id(sample: SampleKey | str, n: int) -> str:
    sid = sample.id() if isinstance(sample, SampleKey) else sample
    return f"{sid}-a{n}"


def identity(obj) -> str:
    return _h(obj, 24)
