"""Milestones (DESIGN §4.3.3, §6.2): state predicates with prerequisites, weight, cost and provenance.

A milestone's check is a ``Query`` against one App's State Adapter plus an expectation, so the
same structure covers "module builds" (process adapter) and "invoice matched" (sql adapter).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.state import Query


class Expectation(CrucibleModel):
    """How a QueryResult maps to φ ∈ [0, 1]."""

    kind: Literal["ok", "equals", "truthy", "regex", "count_eq", "count_ge", "between", "score"] = "ok"
    value: Any = None
    lo: float | None = None
    hi: float | None = None


class MilestoneCheck(CrucibleModel):
    app: str
    query: Query
    expect: Expectation = Field(default_factory=Expectation)


class Provenance(CrucibleModel):
    signal: str | None = None  # signal id or "<trace>#<seq>"
    source: str | None = None  # "expert_end_state", "reference_differential", "policy:<doc>"
    note: str | None = None


class Milestone(CrucibleModel):
    id: str
    check: MilestoneCheck
    requires: list[str] = Field(default_factory=list)
    weight: float = 1.0
    cost: Literal["cheap", "medium", "expensive"] = "cheap"
    gate: bool = False  # a gate milestone zeroes reward when unmet at the terminal state
    provenance: list[Provenance] = Field(default_factory=list)


class MilestoneSet(VersionedDocument):
    SCHEMA = "crucible.milestones/v1"
    family: str
    milestones: list[Milestone]

    @model_validator(mode="after")
    def _dag(self) -> MilestoneSet:
        ids = [m.id for m in self.milestones]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate milestone ids")
        graph = {m.id: set(m.requires) for m in self.milestones}
        for m in self.milestones:
            if m.weight <= 0:
                raise ValueError(f"milestone {m.id} weight must be > 0")
            unknown = set(m.requires) - set(ids)
            if unknown:
                raise ValueError(f"milestone {m.id} requires unknown {sorted(unknown)}")
        # cycle check (Kahn)
        indeg = {k: len(v) for k, v in graph.items()}
        ready = [k for k, d in indeg.items() if d == 0]
        seen = 0
        while ready:
            k = ready.pop()
            seen += 1
            for other, reqs in graph.items():
                if k in reqs:
                    indeg[other] -= 1
                    if indeg[other] == 0:
                        ready.append(other)
        if seen != len(ids):
            raise ValueError("milestone prerequisites contain a cycle")
        return self
