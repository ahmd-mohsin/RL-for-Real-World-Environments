"""Actions, observations and tool specs exchanged between Actors and a World.

Field names follow ATIF (Harbor ``models/trajectories``: ``tool_call_id``, ``function_name``,
``arguments``) and MCP (``name``, ``description``, ``inputSchema``), so episodes convert to ATIF
and tool surfaces convert to MCP without renaming.
"""

from __future__ import annotations

import hashlib
import re
from typing import Annotated, Any, Literal

from pydantic import Field

from crucible.core.schema.base import CrucibleModel

_FN_OK = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


def openai_name(qualified: str) -> str:
    """A function name OpenAI-style APIs accept (^[a-zA-Z0-9_-]{1,64}$) for an ``app.tool`` name."""
    name = qualified.replace(".", "__", 1)
    if _FN_OK.match(name):
        return name
    clean = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
    return clean if len(clean) <= 64 else clean[:55] + "_" + hashlib.sha256(qualified.encode()).hexdigest()[:8]


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


class ImageRef(CrucibleModel):
    """An image a tool returned (a screenshot, a rendered frame, a chart). The bytes live in the World's store under
    ``ref`` (content-addressed); ``data_b64`` rides only on the live observation handed to the policy, never in traces."""

    mime: str = "image/png"
    ref: str
    data_b64: str | None = Field(default=None, exclude=True)  # never serialized (records, checkpoints, traces)


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
    # the environment ended the episode with this call (an OpenEnv / Gym ``done``: a final answer, a step budget); the
    # World marks the calling actor done, exactly as if it had sent ``Done``
    ends_episode: bool = False
    ends_for: list[str] | None = None  # actors the termination applies to (default: the caller); a finished game ends every seat
    images: list[ImageRef] = Field(default_factory=list)  # image outputs (vision environments); text stays in ``content``


class Inbound(CrucibleModel):
    """A message delivered to an actor: from another actor or from a fired event."""

    frm: str = Field(alias="from")
    prefill: bool = False  # part of a recorded conversation the episode continues from (NeMo-Gym prefill), not live
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
