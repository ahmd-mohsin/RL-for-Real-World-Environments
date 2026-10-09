"""The Crucible task extension and task card (DESIGN §4.3, §4.3.4).

A Crucible task is a valid Harbor task directory (validated by ``crucible.interop.harbor``) plus a
``crucible/`` folder:

    crucible/task.yaml        CrucibleTask   - family, horizon, world, milestones, negatives
    crucible/world.yaml       WorldSpec      - the world the task runs in
    crucible/card.yaml        TaskCard       - provenance and value proof
    crucible/milestones.yaml  MilestoneSet
    crucible/negatives/*.patch

The core never imports Harbor: Harbor's ``TaskConfig`` is checked at the interop boundary, and the
core keeps only what it needs from it (``HarborTaskSummary``).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel, VersionedDocument

Horizon = Literal["H0", "H1", "H2", "H3", "H4", "H5"]


class Negative(CrucibleModel):
    id: str
    path: str  # relative to the task dir, e.g. crucible/negatives/no_guard_band.patch
    expected_gate: str  # the gate this negative must fail
    description: str = ""
    provenance: str | None = None  # signal or incident it encodes


class CrucibleTask(VersionedDocument):
    SCHEMA = "crucible.task/v1"
    id: str  # "<pack>/<family>/<name>@<version>"
    family: str
    horizon: Horizon
    world: str = "crucible/world.yaml"
    milestones: str | None = "crucible/milestones.yaml"
    card: str = "crucible/card.yaml"
    negatives: list[Negative] = Field(default_factory=list)
    oracle: str = "solution/solution.patch"
    # When the crucible/ folder is an overlay kept outside the Harbor dir (e.g. seed tasks in a pack),
    # this points at the Harbor task dir, relative to the overlay dir.
    harbor_dir: str | None = None
    seed_of: list[str] = Field(default_factory=list)  # task ids derived from this one (span tasks)
    derived_from: str | None = None  # parent task id if this is a derived (span/trap) task


class FrontierResult(CrucibleModel):
    model: str
    harness: str | None = None
    task_version: str | None = None
    score: float
    passed: bool
    failing_gates: list[str] = Field(default_factory=list)
    steps: int | None = None
    cost_usd: float | None = None
    run_ref: str | None = None  # path or store key
    note: str | None = None


class OracleProof(CrucibleModel):
    grades: int  # number of full grades run
    passes: int
    min_headroom: float | None = None  # smallest perf headroom vs its bar


class FlakeProof(CrucibleModel):
    draws: int
    false_negatives: int
    induced: int = 0


class ValueProof(CrucibleModel):
    frontier: list[FrontierResult] = Field(default_factory=list)
    oracle: OracleProof | None = None
    mutation_score: float | None = None
    negatives_killed: str | None = None  # "6/6"
    flake: FlakeProof | None = None
    leak_audit: Literal["pass", "fail", "not_run"] = "not_run"


class Grounding(CrucibleModel):
    traces: list[str] = Field(default_factory=list)
    oracle_source: Literal["expert_end_state", "builder_oracle", "both"] = "builder_oracle"
    checks_with_provenance: str | None = None  # "41/44"


class Splits(CrucibleModel):
    train: bool = False
    bench: bool = False
    ood_axes: list[str] = Field(default_factory=list)
    showcase: bool = False  # MVP showcase only (tasks 01-04): never a published OOD split


class TaskCard(VersionedDocument):
    SCHEMA = "crucible.card/v1"
    id: str
    family: str
    horizon: Horizon
    title: str | None = None
    grounding: Grounding = Field(default_factory=Grounding)
    value_proof: ValueProof = Field(default_factory=ValueProof)
    splits: Splits = Field(default_factory=Splits)
    history: list[dict[str, Any]] = Field(default_factory=list)  # version notes


class HarborTaskSummary(CrucibleModel):
    """What the core keeps from a Harbor ``TaskConfig`` (filled by interop.harbor)."""

    name: str
    schema_version: str
    task_hash: str  # Harbor packager content hash of the task dir
    gpus: int = 0
    cpus: float | None = None
    memory_mb: int | None = None
    docker_image: str | None = None
    dockerfile: str | None = None
    verifier_mode: Literal["shared", "separate"] = "shared"
    network_mode: str = "public"  # effective for the agent phase
    baseline_network_mode: str = "public"  # [environment] baseline (setup / build)
    agent_timeout_sec: float | None = None
    verifier_timeout_sec: float | None = None
    collect_commands: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    ignored_keys: list[str] = Field(default_factory=list)  # task.toml keys Harbor drops silently
    metadata: dict[str, Any] = Field(default_factory=dict)
