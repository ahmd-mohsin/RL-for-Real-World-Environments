"""Role I/O documents of the forge (DESIGN §7.2). Every role's output is one of these, stored content-addressed.

What each role may see is fixed by the orchestrator's staging (``Node.visible``), not by these models; the models make
the hand-offs explicit and checkable (the Test writer's ``CheckSet`` has no field for oracle material, and the
Auditor's findings cite files or signal ids).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel


class SignalRef(CrucibleModel):
    id: str  # resolvable in the M4 signal store (A8)
    type: str
    trace_id: str | None = None
    text: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


class StatedProperty(CrucibleModel):
    id: str
    text: str  # the sentence of the instruction that states it (A7: every graded property is stated)
    kind: str = "outcome"  # a structured hint for the Test writer: refund, note, status, no_refund, approval, collateral, ...
    args: dict[str, Any] = Field(default_factory=dict)


class TaskSpec(CrucibleModel):
    task_id: str
    title: str
    instruction: str
    properties: list[StatedProperty]
    literals: list[str] = Field(default_factory=list)  # every identifier/amount the instruction names (literal grounding)
    trap: str | None = None  # property-level description of the trap (never its location)
    provenance: list[str] = Field(default_factory=list)  # signal ids / policy sections the task is grounded on
    infeasible: bool = False
    horizon: str = "H1"
    difficulty: dict[str, Any] = Field(default_factory=dict)


class Architecture(CrucibleModel):
    """The Architect's output: the scenario (entities drawn from the seed) and the task specs, tasks before data (AWM)."""

    world_id: str
    scenario: dict[str, Any]
    tasks: list[TaskSpec]


class EnvBuild(CrucibleModel):
    world_id: str
    world_yaml: str
    files: dict[str, str]  # relpath (under the forged pack root) -> content
    frozen_now: str
    rows_served: dict[str, list[str]] = Field(default_factory=dict)  # literal -> the seed rows that make it exist


class OracleBuild(CrucibleModel):
    task_id: str
    kind: Literal["actions", "patch", "state"] = "actions"
    actions: list[dict[str, Any]] = Field(default_factory=list)
    patch: str | None = None
    source: Literal["expert_end_state", "builder_oracle", "both"] = "builder_oracle"
    note: str = ""


class OracleSet(CrucibleModel):
    oracles: list[OracleBuild]


class CheckSpec(CrucibleModel):
    id: str
    gate: str
    gate_kind: Literal["gate", "outcome", "score"] = "outcome"
    kind: Literal["predicate", "invariant", "static", "milestone", "suite"] = "predicate"  # suite: a behavioural gate the kit grader runs (coding)
    app: str | None = None
    sql: str | None = None  # predicate: one scalar; invariant: violating rows
    expect: dict[str, Any] = Field(default_factory=dict)  # Expectation fields for predicates
    states: list[str] = Field(default_factory=list)  # property ids it grades (A7)
    provenance: str | None = None  # signal id / policy section / reference_differential (A8)
    spec: dict[str, Any] = Field(default_factory=dict)  # static / milestone payloads (coding)


class CheckSet(CrucibleModel):
    task_id: str
    checks: list[CheckSpec]


class CheckSets(CrucibleModel):
    sets: list[CheckSet]


class NegativeSpec(CrucibleModel):
    id: str
    task_id: str
    kind: Literal["floor", "trap_reversion", "defect_mutant", "cheat", "valid_alternative"]
    actions: list[dict[str, Any]] = Field(default_factory=list)
    patch: str | None = None
    expected_gate: str | None = None
    expected: Literal["fail", "pass"] = "fail"
    provenance: str | None = None
    description: str = ""


class NegativeSet(CrucibleModel):
    negatives: list[NegativeSpec]


class Finding(CrucibleModel):
    category: str
    blocking: bool = True
    task_id: str | None = None
    detail: str = ""
    citation: str | None = None  # file / signal id / check id


class AuditReport(CrucibleModel):
    findings: list[Finding] = Field(default_factory=list)

    @property
    def sound(self) -> bool:
        return not any(f.blocking for f in self.findings)
