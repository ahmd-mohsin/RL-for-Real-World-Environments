"""``crucible_ref``: a Crucible store object (WorldRef, StateRef, grade, trace, rollout) as an agentenv artifact
(research m11-m12/05 §4.4). Versioned and immutable like every agentenv artifact; ``derived_from`` records provenance."""

from __future__ import annotations

from typing import Literal

from agent_env.artifact.artifact import Artifact
from pydantic import Field


class CrucibleRefArtifact(Artifact):
    type: Literal["crucible_ref"] = "crucible_ref"
    store_key: str = ""
    kind: Literal["world_ref", "state_ref", "grade", "trace", "rollout"] = "grade"
    derived_from: list[str] = Field(default_factory=list)
