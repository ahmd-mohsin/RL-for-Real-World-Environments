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
| ``fixtures`` | ``() -> [{"name", "task", "actions", "reward", "fails_gate"?}]`` | ``pack check --level fixture`` (exact expected grades) |
| ``model`` | ``async (traces) -> {"milestones": [...], "stages": [...], ...}`` | ``crucible model`` (the M5 workflow model) |
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel


class IngestSpec(CrucibleModel):
    """One M3 source. ``connector`` is any registered connector (``crucible registry``). ``file_drop`` takes ``glob`` (+
    ``extra`` fields); every other connector is constructed with ``init`` and discovered with ``scope`` through its
    ``discover`` method (e.g. ``discover_git``). String values in ``init`` / ``scope`` / ``root`` that start with ``./`` or
    ``../`` are resolved against the pack root, so a pack can point at data outside itself without absolute paths.
    ``redaction_env``: a dotenv file whose credential values are redacted as well (``RedactionPolicy.with_env_file``)."""

    connector: str = "file_drop"
    family: str
    glob: str | None = None  # file_drop: relative to ``root`` (default: the pack root)
    root: str | None = None
    extra: list[str] = Field(default_factory=list)
    init: dict[str, Any] = Field(default_factory=dict)
    scope: dict[str, Any] = Field(default_factory=dict)
    discover: str = "discover"
    redaction_env: str | None = None
    label: bool = True  # labelled by ``crucible label`` (sources like policy documents are context, not episodes)

    @model_validator(mode="after")
    def _file_drop_needs_glob(self) -> IngestSpec:
        if self.connector == "file_drop" and not self.glob:
            raise ValueError("a file_drop ingest source needs a glob")
        return self


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
    fixtures: str | None = None
    model: str | None = None


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
    requires: list[Literal["docker", "gpu", "network", "paid"]] = Field(default_factory=list)  # what running its tasks needs (semantic checks skip without it)
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
