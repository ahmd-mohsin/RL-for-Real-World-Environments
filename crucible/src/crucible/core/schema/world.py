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
    gpu_type: str | None = None  # cloud GPU class, e.g. "H100", "A100", "L4" (Modal/Harbor providers); None = any


class SandboxSpec(CrucibleModel):
    """Where a World's state and commands live when the backend puts them in a container.

    ``provider``: ``docker`` (this machine's daemon, or a remote one via ``gpu_host``/``docker_host``, e.g. the lab
    RTX 5090), or any agentenv sandbox provider (``modal``, ``e2b``, ...). GPUs come from the Apps' ``resources``
    (count, ``gpu_type``); ``gpu`` overrides with a provider spec such as ``"H100:2"``.
    """

    provider: str = "docker"  # or agentenv providers, a comma-separated fallback chain ("modal,e2b")
    image: str
    gpu_host: str | None = None  # crucible.toml [gpu_hosts.<name>]
    docker_host: str | None = None
    gpu: str | None = None
    gpu_mode: Literal["runtime", "devices"] = "runtime"  # docker: nvidia runtime (lab 5090) or --gpus
    cpus: float | None = None
    memory_mb: int | None = None
    timeout_s: int = Field(7200, ge=60, le=86400)  # sandbox lifetime; long-horizon worlds go above 2 h (up to 24 h)
    attribution: dict[str, str] = Field(default_factory=dict)  # provider tags (run_id, sample_key, harness); "world" is always set
    auto_sleep_idle_s: int | None = None  # Sail: sleep the VM after this idle time (human waits), resume on the next exec
    workdir: str = "/crucible"  # where relative App paths live inside the sandbox
    # docker: runc (default), runsc (a daemon with gVisor registered), or runsc-in-container (gVisor inside a privileged
    # container: no daemon change, m13 S5)
    runtime: Literal["runc", "runsc", "runsc-in-container"] = "runc"
    pool: int = 0  # docker: keep this many pre-started containers per launch signature (world/pool.py, m13 S10)
    pool_reuse: bool = False  # reuse released containers after verified hygiene (default: destroy, start fresh)


class StateSpec(CrucibleModel):
    adapter: AdapterKind
    # Adapter parameters, e.g. {"root": "/workspace/pcl"} for fs_git, {"path": "state.db"} for sql.
    params: dict[str, Any] = Field(default_factory=dict)


class ProbeSpec(CrucibleModel):
    """A shell ``command`` run in the App's exec (exit 0 = success), or an HTTP request to ``http`` (a status in
    ``expect_status``, default 2xx/3xx, = success). Used for health checks and reset hooks (M1 item 6)."""

    command: str | None = None
    http: str | None = None
    method: Literal["GET", "POST", "PUT", "DELETE"] = "GET"
    body: dict[str, Any] | None = None
    expect_status: list[int] | None = None
    timeout_s: float = Field(default=30.0, gt=0)

    @model_validator(mode="after")
    def _one(self) -> ProbeSpec:
        if bool(self.command) == bool(self.http):
            raise ValueError("give exactly one of command or http")
        return self


class HealthcheckSpec(ProbeSpec):
    """Docker HEALTHCHECK semantics (Harbor ``HealthcheckConfig``): during ``start_period_s`` failures do not count and
    checks repeat every ``start_interval_s``; after it, ``retries`` consecutive failures fail the check."""

    interval_s: float = Field(default=5.0, ge=0)
    start_period_s: float = Field(default=0.0, ge=0)
    start_interval_s: float = Field(default=5.0, ge=0)
    retries: int = Field(default=3, ge=1)


class AppSpec(CrucibleModel):
    id: str
    template: str  # registry key, e.g. "common/sqlite_app", "coding/git_repo", "mimo/sidecar"
    state: StateSpec
    params: dict[str, Any] = Field(default_factory=dict)  # template parameters
    resources: Resources = Field(default_factory=Resources)
    container: str | None = None  # which backend container hosts it (None = backend default)
    description: str | None = None
    healthcheck: HealthcheckSpec | None = None  # run after the App starts and after every fresh reset (M1 item 6)
    reset_hook: ProbeSpec | None = None  # run on every fresh reset: a live service whose state no adapter captures (M1 item 6)

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        if not v or not all(c.isalnum() or c in "_-" for c in v):
            raise ValueError(f"app id must be [A-Za-z0-9_-]+, got {v!r}")
        return v


class IdentitySpec(CrucibleModel):
    """Who an actor is inside the World's systems (M1 item 3; EnterpriseOps-Gym ``context`` / ``user_info``).

    ``roles`` name ``WorldSpec.rbac`` entries; the World refuses a tool call no role allows (``permission_denied``, the
    policy's error, visible like a 403). Apps receive the identity on every call (``App.identity(actor)``); servers
    reached over HTTP get it as headers: ``x-user-id``, ``x-user-email``, ``x-user-name``, ``x-user-roles`` plus
    ``headers`` (EnterpriseOps converts every context key ``k`` to ``x-k``)."""

    user_id: str | int | None = None
    name: str | None = None
    email: str | None = None
    roles: list[str] = Field(default_factory=list)
    attrs: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)

    def http_headers(self) -> dict[str, str]:
        out = {k: str(v) for k, v in (("x-user-id", self.user_id), ("x-user-email", self.email), ("x-user-name", self.name)) if v is not None}
        if self.roles:
            out["x-user-roles"] = ",".join(self.roles)
        for k, v in self.headers.items():
            out[k if k.lower().startswith("x-") else "x-" + k.lower().replace("_", "-")] = str(v)
        return out


class RoleSpec(CrucibleModel):
    """A role's tool rights: ``allow`` and ``deny`` patterns over ``<app>.<tool>``; deny wins."""

    allow: list[str] = Field(default_factory=list)
    deny: list[str] = Field(default_factory=list)


class CatalogSpec(CrucibleModel):
    """Which of its permitted tools an actor is shown (M1 item 3; EnterpriseOps-Gym modes ``oracle``, ``plus_N_tools``).

    ``oracle`` patterns are always shown; ``distractors`` more are drawn, reproducibly from the World seed, from the
    other tools (``-1``: all of them); ``hide`` patterns are never shown (``restricted_tools``). A tool outside the
    catalog does not exist for the actor: calling it is ``unknown_tool``. Simulated actors and events are unaffected."""

    oracle: list[str] = Field(default_factory=list)
    distractors: int = Field(default=0, ge=-1)
    hide: list[str] = Field(default_factory=list)


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
    identity: IdentitySpec | None = None
    catalog: CatalogSpec | None = None

    def allows(self, qualified_tool: str) -> bool:
        return any(fnmatch.fnmatchcase(qualified_tool, pat) for pat in self.tools)


class ClockSpec(CrucibleModel):
    """Virtual time.

    ``step`` mode: time moves only by ``step_dt_s`` per world step and by explicit advances, so
    episodes are reproducible (RL default). ``wall`` mode: agentenv gateway semantics,
    now = t0 + real_elapsed * rate (urn:agentenv:clock/v1). ``sim`` mode: a simulation owns time; it moves only
    when an App advances it (``AppContext.clock.advance_to``: a discrete-event simulator, an imported environment
    reporting its sim time), never per step; time-triggered events fire when the simulation passes their due time.
    """

    mode: Literal["step", "wall", "sim"] = "step"
    t0: str = "2026-01-05T09:00:00Z"
    rate: float = 1.0  # wall mode: virtual seconds per real second (max 86400)
    step_dt_s: float = 60.0  # step mode: virtual seconds per world step

    @field_validator("t0")
    @classmethod
    def _t0(cls, v: str) -> str:
        from datetime import datetime

        datetime.fromisoformat(v)  # raises ValueError on a malformed timestamp
        return v

    @field_validator("step_dt_s")
    @classmethod
    def _step_dt(cls, v: float) -> float:
        if v < 0:
            raise ValueError("step_dt_s must be >= 0 (time never runs backwards)")
        return v

    @field_validator("rate")
    @classmethod
    def _rate(cls, v: float) -> float:
        if not (0 <= v <= 86400):
            raise ValueError("rate must be within [0, 86400] (agentenv clock bound: 1 real s = 1 virtual day)")
        return v


class EveryDist(CrucibleModel):
    """A stochastic recurrence: exponential inter-arrivals with ``mean`` (agentenv ``every: {dist: exp, mean}``)."""

    dist: Literal["exp"] = "exp"
    mean: str  # duration


class EventWhen(CrucibleModel):
    """When an event fires: the urn:agentenv:triggers/v1 schedules (m13 C8: the whole language).

    Exactly one anchor: ``at``, ``after`` (a duration from t0), ``on_tool`` (an action trigger, with ``where``
    predicates over ``args.*`` / ``result.*``), ``state`` (a check pipeline evaluated after each tool call), or
    ``after_event`` + ``offset`` (anchored to another event's firing). ``every`` alone recurs from t0. ``every`` is a
    duration or ``EveryDist`` (seeded, so arrivals reproduce after a restore); ``count`` and ``until`` bound it.
    """

    at: str | None = None  # absolute RFC 3339 virtual time (or a P-duration from t0)
    after: str | None = None  # duration from t0, e.g. "2d", "90m", "30s"
    every: str | EveryDist | None = None  # repeat interval
    count: int | None = None  # max firings for `every`
    until: str | None = None  # no recurrence after this instant (RFC 3339 or a P-duration from t0)
    seed: int | None = None  # stochastic `every`
    on_tool: str | None = None  # fire after a successful call to "<app>.<tool>" (by any actor, unless on_actor)
    on_actor: str | None = None
    repeat: bool = False  # on_tool: fire on every matching call (agentenv action-trigger "repeat"); default once
    # on_tool: {"args.x" | "result.y" | "x": {regex|equals|exists}}. New fields default to None: canonical JSON drops None,
    # so specs that do not use them keep their content hash (acceptance keys, WorldRefs)
    where: dict[str, dict[str, Any]] | None = None
    state: dict[str, Any] | None = None  # a check pipeline {steps: [{tool, args, extract, bind, predicate}]}
    after_event: str | None = None  # anchor: another event's id
    offset: str | None = None  # after_event: fire this long after the anchor fired

    @model_validator(mode="after")
    def _one_anchor(self) -> EventWhen:
        from crucible.core.triggers import validate_check, validate_predicate

        anchors = [x for x in (self.at, self.after, self.on_tool, self.state, self.after_event) if x]
        if len(anchors) > 1 or (not anchors and not self.every):
            raise ValueError("event `when` needs exactly one of at / after / on_tool / state / after_event (or a recurring `every` alone)")
        if (self.count is not None or self.until) and not self.every:
            raise ValueError("`count` and `until` require `every`")
        if self.seed is not None and not isinstance(self.every, EveryDist):
            raise ValueError("`seed` is only valid with a stochastic `every`")
        if isinstance(self.every, EveryDist) and self.seed is None:
            raise ValueError("a stochastic `every` requires a `seed` (reproducible arrivals)")
        if bool(self.after_event) != bool(self.offset):
            raise ValueError("`after_event` and `offset` go together")
        if self.where and not self.on_tool:
            raise ValueError("`where` is only valid on on_tool events")
        for f, pred in (self.where or {}).items():
            validate_predicate(pred, f"where.{f}")
        if self.state is not None:
            object.__setattr__(self, "state", validate_check(self.state, "when.state"))
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
    verify: dict[str, Any] | None = None  # a check pipeline the action's effect must pass (logged verify_ok / verify_failed)

    @model_validator(mode="after")
    def _fields(self) -> EventAction:
        if self.type == "nl" and not self.text:
            raise ValueError("nl action needs text")
        if self.type == "tool" and not self.tool:
            raise ValueError("tool action needs tool")
        if self.type == "permission" and not (self.grant or self.revoke):
            raise ValueError("permission action needs grant or revoke")
        if self.verify is not None:
            from crucible.core.triggers import validate_check

            object.__setattr__(self, "verify", validate_check(self.verify, "action.verify"))
        return self


class EventSpec(CrucibleModel):
    id: str
    when: EventWhen
    actor: str  # the actor in whose name the event acts (system actors included)
    action: EventAction
    # on_tool events: hold the provoking call's response until the action ran, so the caller observes its effect. In
    # step mode Crucible always runs on_tool events before the step's observation; the field is delegated in wall mode.
    barrier: Literal["provoking_call"] | None = None

    @model_validator(mode="after")
    def _barrier(self) -> EventSpec:
        if self.barrier and not self.when.on_tool:
            raise ValueError("a provoking_call barrier needs an on_tool event")
        return self


class PolicyDoc(CrucibleModel):
    id: str
    app: str
    path: str
    applies_to: list[str] = Field(default_factory=list)  # actor ids


class LimitsSpec(CrucibleModel):
    """Bounds the World enforces at its single tool-call choke point (``World._call_tool``)."""

    tool_timeout_s: float = 3600  # a hung tool becomes an infra-fault result, never a hung episode
    max_tool_output_chars: int = 30000  # one head+tail cut with a banner; the full text goes to the store
    # per-episode call budgets: fnmatch pattern over qualified tool names -> max calls (all policy actors together),
    # e.g. {"env.check_plan": 10}; an exhausted budget is a policy-fault result ("quota_exhausted"), never a crash.
    # None (not {}) by default: canonical JSON drops None, so specs without quotas keep their content hash
    tool_quota: dict[str, int] | None = None

    @model_validator(mode="after")
    def _bounds(self) -> LimitsSpec:
        if self.tool_timeout_s <= 0:
            raise ValueError("tool_timeout_s must be > 0")
        if self.max_tool_output_chars <= 0:
            raise ValueError("max_tool_output_chars must be > 0")
        if self.tool_quota and any(n < 0 for n in self.tool_quota.values()):
            raise ValueError("tool_quota counts must be >= 0")
        return self


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
    - ``pids_limit`` / ``memory``: resource caps for the policy's sandbox (fork bombs, memory floods);
    - ``firewall`` (None = ``auto``, ``off``, ``required``): per-uid iptables rules for the policy (Karotte): in
      ``allowlist`` worlds it may reach only localhost, the sandbox's own addresses and the allowlisted hosts' IPs;
      in every world the localhost ports of App servers and ``blocked_ports`` are closed to it, so it cannot reach an
      App except through the World's tool gate. ``auto`` applies the rules where the sandbox allows (CAP_NET_ADMIN)
      and leans on the provider and the egress probe otherwise; ``required`` refuses to start without them.
    """

    scrub_messages: bool = True
    hide_citations: bool = True
    policy_user: str | None = None
    policy_uid: int | None = None  # None: the guard allocates a free uid (image accounts are never reused)
    quiesce: bool = True
    egress_probe: bool = True
    pids_limit: int = Field(default=4096, ge=0)  # 0 = no limit (None was the pre-audit spelling and did not round-trip)
    memory: str | None = None
    firewall: Literal["auto", "off", "required"] | None = None  # None: auto (keeps the hashes of specs written before M1 item 5)
    blocked_ports: list[int] | None = None

    @field_validator("pids_limit", mode="before")
    @classmethod
    def _no_limit(cls, v: Any) -> Any:
        return 0 if v is None else v


class VerifierSpec(CrucibleModel):
    """Where and how the World is graded (M1; Harbor's separate verifier environment, Karotte's submission custody).

    - ``mode``: ``in_world`` grades the live World the policy acted in (the pre-M1 behaviour). ``separate`` quiesces
      the policy's World, snapshots it, closes it, and restores the snapshot into a fresh World built from the same
      spec, where the grader runs: nothing the policy left running, and no file it can still change, reaches the grader;
    - ``backend`` / ``backend_params``: the verifier World's backend (default: the World's own), e.g. a verifier image
      that carries the hidden tests the policy's image never had;
    - ``artifacts``: ``"<app>:<glob>"`` patterns naming what the policy hands in. Custody rules apply to them
      (``world/custody.py``): no symlinks (unless ``allow_symlinks``), no FIFOs, sockets or devices, no oversize file,
      at most ``max_entries`` entries and ``max_depth`` levels, at most ``max_total_bytes`` in all;
    - ``protected``: ``"<app>:<glob>"`` patterns the policy must not change (hidden tests, graders, fixtures): the
      snapshot's entries must equal the episode start's, and no new entry may appear under them.

    A custody violation is the policy's outcome (``crucible:policy_misbehavior``): scored 0, never masked.
    """

    mode: Literal["in_world", "separate"] = "in_world"
    backend: str | None = None
    backend_params: dict[str, Any] = Field(default_factory=dict)
    artifacts: list[str] = Field(default_factory=list)
    protected: list[str] = Field(default_factory=list)
    allow_symlinks: bool = False
    max_file_bytes: int = Field(default=256 * 1024**2, gt=0)
    max_total_bytes: int = Field(default=1024**3, gt=0)
    max_entries: int = Field(default=10_000, gt=0)
    max_depth: int = Field(default=32, gt=0)

    @field_validator("artifacts", "protected")
    @classmethod
    def _app_glob(cls, v: list[str]) -> list[str]:
        for pat in v:
            app, sep, glob = pat.partition(":")
            if not sep or not app or not glob:
                raise ValueError(f"custody pattern {pat!r} must be '<app>:<glob>'")
            if glob.startswith("/") or ".." in glob.split("/"):
                raise ValueError(f"custody pattern {pat!r}: the glob is relative to the app's state root and stays inside it")
        return v


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
    # online interception rules active at World._call_tool (m13 C-4, ``world/intercept.py``): None = the defaults
    # (remote_code_fetch, grader_paths); [] = off; a task whose job is to download (internet is the task) keeps only
    # ["grader_paths"]
    intercept: list[str] | None = None
    isolation: IsolationSpec = Field(default_factory=IsolationSpec)
    limits: LimitsSpec = Field(default_factory=LimitsSpec)
    verifier: VerifierSpec | None = None  # None: graded in the live World (``in_world``); hashes of older specs unchanged
    rbac: dict[str, RoleSpec] | None = None  # role -> tool rights, enforced for actors with an identity (M1 item 3)
    serial: bool | None = None  # True: one episode of this world at a time on this host (shared live services; M1 item 6)
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
        event_ids = [e.id for e in self.events]
        for ev in self.events:
            if ev.when.after_event and ev.when.after_event not in event_ids:
                raise ValueError(f"event {ev.id!r} is anchored on unknown event {ev.when.after_event!r}")
            if ev.actor not in actor_ids:
                raise ValueError(f"event {ev.id!r} acts as unknown actor {ev.actor!r}")
            if ev.action.to and ev.action.to not in actor_ids:
                raise ValueError(f"event {ev.id!r} messages unknown actor {ev.action.to!r}")
            timed = ev.when.at or ev.when.after or ev.when.after_event or (ev.when.every and not ev.when.on_tool and not ev.when.state)
            if timed and self.clock is None:
                raise ValueError(f"event {ev.id!r} is time-scheduled but the world has no clock")
        def known_app(pattern: str) -> bool:
            app = pattern.split(".", 1)[0]
            return app in app_ids or any(fnmatch.fnmatchcase(a, app) for a in app_ids)

        for ev in self.events:
            if ev.action.tool and not known_app(ev.action.tool):
                raise ValueError(f"event {ev.id!r} runs a tool of unknown app: {ev.action.tool!r}")
            for pat in [*ev.action.grant, *ev.action.revoke]:
                if not known_app(pat):
                    raise ValueError(f"event {ev.id!r} grants or revokes a pattern of unknown app: {pat!r}")
        if self.network == "allowlist" and not self.allowed_hosts:
            raise ValueError("network 'allowlist' needs allowed_hosts")
        if self.network != "allowlist" and self.allowed_hosts:
            raise ValueError(f"allowed_hosts is only meaningful with network 'allowlist', not {self.network!r}")
        for pol in self.policies:
            if pol.app not in app_ids:
                raise ValueError(f"policy {pol.id!r} lives in unknown app {pol.app!r}")
        for actor in self.actors:
            roles = actor.identity.roles if actor.identity else []
            if roles and self.rbac is None:
                raise ValueError(f"actor {actor.id!r} has roles {roles} but the world defines no rbac")
            unknown = sorted(set(roles) - set(self.rbac or {}))
            if unknown:
                raise ValueError(f"actor {actor.id!r} has unknown roles {unknown} (rbac defines {sorted(self.rbac or {})})")
        for role, rs in (self.rbac or {}).items():
            for pat in [*rs.allow, *rs.deny]:
                if not known_app(pat):
                    raise ValueError(f"rbac role {role!r} names a pattern of unknown app: {pat!r}")
        if self.verifier is not None:
            for pat in [*self.verifier.artifacts, *self.verifier.protected]:
                app = pat.partition(":")[0]
                if app not in app_ids:
                    raise ValueError(f"verifier custody pattern {pat!r} names unknown app {app!r}")
                if self.app(app).state.adapter not in ("fs", "fs_git"):
                    raise ValueError(f"verifier custody pattern {pat!r}: app {app!r} holds {self.app(app).state.adapter!r} state; custody covers file trees (fs, fs_git)")
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
