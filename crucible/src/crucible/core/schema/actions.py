"""Actions, observations and tool specs exchanged between Actors and a World.

Field names follow ATIF (Harbor ``models/trajectories``: ``tool_call_id``, ``function_name``,
``arguments``) and MCP (``name``, ``description``, ``inputSchema``), so episodes convert to ATIF
and tool surfaces convert to MCP without renaming.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel


class ToolSpec(CrucibleModel):
    app: str
    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object", "properties": {}})
    read_only: bool = False  # hint for state tracking: read-only calls cannot change state

    @property
    def qualified(self) -> str:
        return f"{self.app}.{self.name}"

    def to_openai(self) -> dict[str, Any]:
        """OpenAI function-calling schema; names use '__' because '.' is not allowed there."""
        return {
            "type": "function",
            "function": {
                "name": f"{self.app}__{self.name}",
                "description": self.description[:1024],
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }


class ToolCall(CrucibleModel):
    kind: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    function_name: str  # qualified "<app>.<tool>"
    arguments: dict[str, Any] = Field(default_factory=dict)


class Message(CrucibleModel):
    kind: Literal["message"] = "message"
    to: str  # actor id
    text: str
    citations: list[str] = Field(default_factory=list)  # record span ids (grounded actors)


class Done(CrucibleModel):
    kind: Literal["done"] = "done"
    reason: str = ""


Action = Annotated[ToolCall | Message | Done, Field(discriminator="kind")]


class ToolResult(CrucibleModel):
    tool_call_id: str
    function_name: str
    ok: bool
    content: str  # what the actor sees (text; structured results are JSON-encoded)
    data: Any = None  # structured payload, verifier/log side
    error: str | None = None
    duration_s: float | None = None
    truncated: bool = False
    # who is at fault when ok=False: the policy (bad call, denied), the tool (a domain error it reports), or the
    # infrastructure (transport down, App crashed, timeout of a hung service) -- infra faults mask the episode.
    fault: Literal["policy", "tool", "infra"] | None = None
    content_ref: str | None = None  # store key of the full output when ``content`` was bounded


class Inbound(CrucibleModel):
    """A message delivered to an actor: from another actor or from a fired event."""

    frm: str = Field(alias="from")
    text: str
    at: str | None = None  # virtual time
    citations: list[str] = Field(default_factory=list)


class Observation(CrucibleModel):
    actor: str
    step: int
    time: str | None = None  # virtual time
    tool_results: list[ToolResult] = Field(default_factory=list)
    inbox: list[Inbound] = Field(default_factory=list)
    instruction: str | None = None  # only on reset
    tools: list[ToolSpec] = Field(default_factory=list)  # only on reset or when permissions change


class StepResult(CrucibleModel):
    observation: Observation
    done: bool = False
    info: dict[str, Any] = Field(default_factory=dict)  # events fired, permission changes, timings
