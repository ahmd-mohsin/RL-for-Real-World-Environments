"""Crucible schemas (all JSON-serializable, versioned where top-level)."""

from crucible.core.schema.actions import (
    Action, Done, Inbound, Message, Observation, StepResult, ToolCall, ToolResult, ToolSpec,
)
from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.milestone import Expectation, Milestone, MilestoneCheck, MilestoneSet
from crucible.core.schema.reward import InfraStatus, RewardJson, StepRow
from crucible.core.schema.run import FAILURE_KINDS, Failure, RunRecord
from crucible.core.schema.signal import Signal, SignalTarget, SignalType
from crucible.core.schema.state import (
    ClockState, FileChange, PendingEvent, Query, QueryResult, RowChange, StateDiff, StateRef, WorldRef,
)
from crucible.core.schema.task import CrucibleTask, HarborTaskSummary, Negative, TaskCard
from crucible.core.schema.trace import Record, Trace, TraceActor, TraceEvent, TraceOutcome, TraceSource, records_from_trace
from crucible.core.schema.world import (
    ActorSpec, AppSpec, ClockSpec, EventAction, EventSpec, EventWhen, PolicyDoc, Resources, StateSpec, WorldSpec,
)

ALL_DOCUMENTS = {
    "world": WorldSpec,
    "trace": Trace,
    "task": CrucibleTask,
    "card": TaskCard,
    "milestones": MilestoneSet,
    "reward": RewardJson,
    "step_row": StepRow,
    "run": RunRecord,
    "world_ref": WorldRef,
    "record": Record,
    "harbor_task_summary": HarborTaskSummary,
}

__all__ = [n for n in dir() if not n.startswith("_")]
