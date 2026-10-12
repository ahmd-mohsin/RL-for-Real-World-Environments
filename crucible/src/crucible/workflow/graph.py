"""The family Workflow Graph (``crucible.workflow/v1``): the mined, filtered milestones, their stages and precedence
edges, the filter verdicts, and where each came from. It is what the tracker loads and what a task card cites."""

from __future__ import annotations

from typing import Any

from pydantic import Field, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.milestone import Milestone
from crucible.workflow.stages import StageModel


class StageSpec(CrucibleModel):
    index: int
    name: str
    milestones: list[str]
    median_time: float = 0.0


class EdgeSpec(CrucibleModel):
    a: str
    b: str
    support: float
    n: int


class WorkflowGraph(VersionedDocument):
    SCHEMA = "crucible.workflow/v1"
    family: str
    milestones: list[Milestone]
    stages: list[StageSpec] = Field(default_factory=list)
    edges: list[EdgeSpec] = Field(default_factory=list)
    verdicts: dict[str, dict[str, Any]] = Field(default_factory=dict)  # milestone -> filter checks + kept/reason
    sources: list[str] = Field(default_factory=list)  # trace ids mined from
    stats: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _valid(self) -> WorkflowGraph:
        """The ``MilestoneSet`` rules (M5 audit: a weight of -1 gave Φ = -1, an unknown prerequisite zeroed a milestone
        forever, and a stage naming an unknown milestone crashed the tracker)."""
        ids = {m.id for m in self.milestones}
        if len(ids) != len(self.milestones):
            raise ValueError("duplicate milestone ids")
        for m in self.milestones:
            if not m.weight > 0:
                raise ValueError(f"milestone {m.id} has weight {m.weight}; weights must be > 0")
            if bad := [r for r in m.requires if r not in ids]:
                raise ValueError(f"milestone {m.id} requires unknown milestones {bad}")
        for st in self.stages:
            if bad := [x for x in st.milestones if x not in ids]:
                raise ValueError(f"stage {st.index} names unknown milestones {bad}")
        return self

    @classmethod
    def build(cls, family: str, milestones: list[Milestone], model: StageModel, **kw: Any) -> WorkflowGraph:
        return cls(family=family, milestones=milestones,
                   stages=[StageSpec(index=s.index, name=s.name, milestones=s.milestones, median_time=s.median_time) for s in model.stages],
                   edges=[EdgeSpec(a=e.a, b=e.b, support=round(e.support, 4), n=e.n) for e in model.edges], **kw)

    def stage_model(self) -> StageModel:
        from crucible.workflow.stages import Precedence, Stage

        return StageModel(stages=[Stage(index=s.index, milestones=s.milestones, name=s.name, median_time=s.median_time) for s in self.stages],
                          edges=[Precedence(e.a, e.b, e.support, e.n) for e in self.edges])
