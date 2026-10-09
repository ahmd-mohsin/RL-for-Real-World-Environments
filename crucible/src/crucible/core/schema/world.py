"""World spec ``crucible.world/v1`` (DESIGN §4.4): Apps, Actors, Clock, Events, policies.

The same schema describes a single repo with one actor (task 04) and a multi-department
workflow with simulated people and a virtual clock. Templates (``coding/git_repo``,
``common/mailbox`` ...) are resolved by the registry; packs contribute templates, the core never
names a domain.
"""

from __future__ import annotations

import fnmatch
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from crucible.core.schema.base import CrucibleModel, VersionedDocument
from crucible.core.schema.state import AdapterKind

ActorKind = Literal["policy", "simulated", "peer", "human", "system"]


class Resources(CrucibleModel):
    cpus: float | None = None
    memory_mb: int | None = None
    storage_mb: int | None = None
    gpus: int = 0
    gpu_arch: str | None = None  # e.g. "sm_120"


class StateSpec(CrucibleModel):
    adapter: AdapterKind
    # Adapter parameters, e.g. {"root": "/workspace/pcl"} for fs_git, {"path": "state.db"} for sql.
    params: dict[str, Any] = Field(default_factory=dict)


class AppSpec(CrucibleModel):
    id: str
    template: str  # registry key, e.g. "common/sqlite_app", "coding/git_repo", "mimo/sidecar"
    state: StateSpec
    params: dict[str, Any] = Field(default_factory=dict)  # template parameters
    resources: Resources = Field(default_factory=Resources)
    container: str | None = None  # which backend container hosts it (None = backend default)
    description: str | None = None

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError(f"app id must be [A-Za-z0-9_-]+, got {v!r}")
        return v


class ActorSpec(CrucibleModel):
    id: str
    kind: ActorKind
    # Tool access: patterns over "<app>.<tool>" (fnmatch). The policy actor sees ONLY these tools.
    tools: list[str] = Field(default_factory=list)
    model: str | None = None  # LLM id for simulated/policy actors (resolved by config)
    persona: str | None = None  # simulated actors: who they are and what they want
    grounded_on: list[str] = Field(default_factory=list)  # record sources (store keys / paths)
    policy_doc: str | None = None  # path within an App (e.g. "policy:approval_matrix.md")
    transport: str | None = None  # human/peer: "a2a", "file_queue", "console"
    params: dict[str, Any] = Field(default_factory=dict)

    def allows(self, qualified_tool: str) -> bool:
        return any(fnmatch.fnmatchcase(qualified_tool, pat) for pat in self.tools)


class ClockSpec(CrucibleModel):
    """Virtual time.

    ``step`` mode: time moves only by ``step_dt_s`` per world step and by explicit advances, so
    episodes are reproducible (RL default). ``wall`` mode: agentenv gateway semantics,
    now = t0 + real_elapsed * rate (urn:agentenv:clock/v1).
    """

    mode: Literal["step", "wall"] = "step"
    t0: str = "2026-01-05T09:00:00Z"
    rate: float = 1.0  # wall mode: virtual seconds per real second (max 86400)
    step_dt_s: float = 60.0  # step mode: virtual seconds per world step

    @field_validator("rate")
    @classmethod
    def _rate(cls, v: float) -> float:
        if not (0 <= v <= 86400):
            raise ValueError("rate must be within [0, 86400] (agentenv clock bound: 1 real s = 1 virtual day)")
        return v


class EventWhen(CrucibleModel):
    """When an event fires (subset of urn:agentenv:triggers/v1 schedules)."""

    at: str | None = None  # absolute RFC 3339 virtual time
    after: str | None = None  # duration from t0, e.g. "2d", "90m", "30s"
    every: str | None = None  # repeat interval
    count: int | None = None  # max firings for `every`
    on_tool: str | None = None  # fire after a call to "<app>.<tool>" (by any actor, unless on_actor)
    on_actor: str | None = None
    repeat: bool = False  # on_tool: fire on every matching call (agentenv action-trigger "repeat"); default once

    @model_validator(mode="after")
    def _one_anchor(self) -> EventWhen:
        anchors = [x for x in (self.at, self.after, self.on_tool) if x]
        if len(anchors) != 1:
            raise ValueError("event `when` needs exactly one of at / after / on_tool")
        if self.count is not None and not self.every:
            raise ValueError("`count` requires `every`")
        return self


class EventAction(CrucibleModel):
    """What happens (agentenv trigger action types: nl, tool, permission)."""

    type: Literal["nl", "tool", "permission"]
    text: str | None = None  # nl: message delivered to `to`
    to: str | None = None  # nl: recipient actor (default: every policy actor)
    tool: str | None = None  # tool: "<app>.<tool>" executed by the system
    args: dict[str, Any] = Field(default_factory=dict)
    grant: list[str] = Field(default_factory=list)  # permission: tool patterns granted to `actor`
    revoke: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _fields(self) -> EventAction:
        if self.type == "nl" and not self.text:
            raise ValueError("nl action needs text")
        if self.type == "tool" and not self.tool:
            raise ValueError("tool action needs tool")
        if self.type == "permission" and not (self.grant or self.revoke):
            raise ValueError("permission action needs grant or revoke")
        return self


class EventSpec(CrucibleModel):
    id: str
    when: EventWhen
    actor: str  # the actor in whose name the event acts (system actors included)
    action: EventAction


class PolicyDoc(CrucibleModel):
    id: str
    app: str
    path: str
    applies_to: list[str] = Field(default_factory=list)  # actor ids


class IsolationSpec(CrucibleModel):
    """How the World protects itself from the policy (DESIGN P17; Karotte / Repo2RLEnv lessons).

    - ``scrub_messages``: simulated actors' replies to policy actors are scrubbed of fix pointers
      (PR / issue / commit / CVE ids, code-host URLs, "fixed in vX") before delivery;
    - ``hide_citations``: record ids cited by grounded actors (they name repos and PRs) stay in the trace,
      never in the policy's observation;
    - ``policy_user``: container backends run every policy tool as this unprivileged user (created at
      provision); root-only paths stay unreadable, and verifier operations stay root-side;
    - ``quiesce``: before any snapshot or verification, kill every process the policy user left behind;
    - ``egress_probe``: for ``no-network`` / ``allowlist`` worlds, refuse to start if the policy user can
      reach the internet (Karotte's self-test);
    - ``pids_limit`` / ``memory``: resource caps for the policy's sandbox (fork bombs, memory floods).
    """

    scrub_messages: bool = True
    hide_citations: bool = True
    policy_user: str | None = None
    policy_uid: int | None = None  # None: the guard allocates a free uid (image accounts are never reused)
    quiesce: bool = True
    egress_probe: bool = True
    pids_limit: int | None = 4096
    memory: str | None = None


class WorldSpec(VersionedDocument):
    SCHEMA = "crucible.world/v1"
    world: str  # "<pack>/<name>"
    description: str | None = None
    backend: str = "local"  # registry key: local | harbor | agentenv | ...
    backend_params: dict[str, Any] = Field(default_factory=dict)
    apps: list[AppSpec]
    actors: list[ActorSpec]
    clock: ClockSpec | None = None
    events: list[EventSpec] = Field(default_factory=list)
    policies: list[PolicyDoc] = Field(default_factory=list)
    network: Literal["no-network", "public", "allowlist"] = "no-network"
    allowed_hosts: list[str] = Field(default_factory=list)  # network == "allowlist" (e.g. the model API)
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    seed: int = 0

    @model_validator(mode="after")
    def _references(self) -> WorldSpec:
        app_ids = [a.id for a in self.apps]
        actor_ids = [a.id for a in self.actors]
        for ids, what in ((app_ids, "app"), (actor_ids, "actor"), ([e.id for e in self.events], "event")):
            dup = {x for x in ids if ids.count(x) > 1}
            if dup:
                raise ValueError(f"duplicate {what} ids: {sorted(dup)}")
        if not any(a.kind == "policy" for a in self.actors):
            raise ValueError("a world needs at least one policy actor")
        for actor in self.actors:
            for pat in actor.tools:
                app = pat.split(".", 1)[0]
                if app not in app_ids and not any(fnmatch.fnmatchcase(a, app) for a in app_ids):
                    raise ValueError(f"actor {actor.id!r} tool pattern {pat!r} names unknown app {app!r}")
        for ev in self.events:
            if ev.actor not in actor_ids:
                raise ValueError(f"event {ev.id!r} acts as unknown actor {ev.actor!r}")
            if ev.action.to and ev.action.to not in actor_ids:
                raise ValueError(f"event {ev.id!r} messages unknown actor {ev.action.to!r}")
            if (ev.when.at or ev.when.after) and self.clock is None:
                raise ValueError(f"event {ev.id!r} is time-scheduled but the world has no clock")
        for pol in self.policies:
            if pol.app not in app_ids:
                raise ValueError(f"policy {pol.id!r} lives in unknown app {pol.app!r}")
        return self

    def app(self, app_id: str) -> AppSpec:
        for a in self.apps:
            if a.id == app_id:
                return a
        raise KeyError(app_id)

    def actor(self, actor_id: str) -> ActorSpec:
        for a in self.actors:
            if a.id == actor_id:
                return a
        raise KeyError(actor_id)

    @property
    def policy_actors(self) -> list[ActorSpec]:
        return [a for a in self.actors if a.kind == "policy"]
