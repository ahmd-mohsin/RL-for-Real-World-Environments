"""``pack.yaml``: a domain pack's manifest (ARCHITECTURE §4 M14; M15 ``crucible pack new / check``).

The shape follows NeMo-Gym's ``EnvironmentManifest`` (a pydantic model with a cross-field contract and an exported JSON
Schema) and OpenEnv's ``openenv.yaml`` (reward range, determinism, capabilities). Hooks are ``module:attr`` strings
resolved against the pack's ``src/``. Every studio command reads them, so the core imports no pack.

| hook | signature | used by |
|---|---|---|
| ``credit`` | ``async (traces) -> {trace_id: {"reward": float, "phi_by_seq": {seq: Φ}}}`` | ``label`` (dense credit) |
| ``tasks`` | ``() -> [(RunTask, gold actions)]`` | ``pack check`` (gold passes, no-op fails), ``bench`` anchors |
| ``generator`` | ``(seed: int) -> RunTask`` | ``bench`` fresh items |
| ``stages`` | ``(task_id) -> [(stage, [milestone ids])]`` | ``bench`` stage reach |
| ``forge`` | ``async (out_dir, store_dir) -> [(task, resolve)]`` | ``build`` |
| ``policies`` | ``() -> [PolicyRef]`` | ``bench`` / ``build`` without a model |
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel


class IngestSpec(CrucibleModel):
    connector: Literal["file_drop"] = "file_drop"
    family: str
    glob: str  # relative to the pack root
    extra: list[str] = Field(default_factory=list)


class RewardSpec(CrucibleModel):
    range: tuple[float, float] = (0.0, 1.0)
    higher_is_better: bool = True


class LabelSpec(CrucibleModel):
    policy_roles: list[str] = Field(default_factory=lambda: ["agent"])  # whose actions are the policy's (assistant turns)
    threshold: float = 1.0  # reward at or above which an episode counts as solved


class PackHooks(CrucibleModel):
    credit: str | None = None
    tasks: str | None = None
    generator: str | None = None
    stages: str | None = None
    forge: str | None = None
    policies: str | None = None


class PackManifest(CrucibleModel):
    schema_version: Literal[1] = 1
    name: str
    version: str
    description: str
    license: str
    authors: list[str] = Field(default_factory=list)
    kind: Literal["domain", "coding", "toy"] = "domain"
    package: str  # the import name under src/
    entry_point: str  # module:register (the crucible.plugins entry point)
    template_version: int | None = None  # the _template version the pack was scaffolded from (a later 3-way upgrade)
    reward: RewardSpec = Field(default_factory=RewardSpec)
    determinism: Literal["seeded", "nondeterministic"] = "seeded"
    ingest: list[IngestSpec] = Field(default_factory=list)
    label: LabelSpec = Field(default_factory=LabelSpec)
    hooks: PackHooks = Field(default_factory=PackHooks)

    @model_validator(mode="after")
    def _contract(self) -> PackManifest:
        lo, hi = self.reward.range
        if not lo < hi:
            raise ValueError(f"reward.range must be increasing, got {self.reward.range}")
        if self.hooks.credit and not self.ingest:
            raise ValueError("hooks.credit labels ingested traces: declare at least one ingest source")
        if ":" not in self.entry_point:
            raise ValueError("entry_point must be module:attr")
        for k, v in self.hooks.model_dump().items():
            if v is not None and ":" not in v:
                raise ValueError(f"hooks.{k} must be module:attr, got {v!r}")
        return self
