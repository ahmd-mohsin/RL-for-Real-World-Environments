"""World episodes -> ATIF trajectories (DEPEND: ``harbor.models.trajectories``, ATIF v1.8).

One trajectory per policy actor, validated by Harbor's own pydantic models: the instruction and every
message / event delivered to the actor are ``user`` steps; each action is an ``agent`` step (tool calls
carry their observation results, matched by ``tool_call_id``; messages to other actors carry
``extra.to``). Virtual timestamps are kept. The result plugs into Harbor's viewer, trace exporters
(ShareGPT / HF) and anything else that reads ATIF.
"""

from __future__ import annotations

from typing import Any

from harbor.models.trajectories import Agent, Observation, ObservationResult, Step, ToolCall, Trajectory

from crucible.core.schema.trace import Trace


def _ts(t: str | None) -> str | None:
    return t.replace("Z", "+00:00") if t else None


def episode_to_atif(trace: Trace, actor: str, *, instruction: str | None = None, agent_name: str = "crucible-policy",
                    agent_version: str = "0.1.0", model_name: str | None = None, session_id: str | None = None) -> Trajectory:
    steps: list[Step] = []

    def add(**kw: Any) -> Step:
        s = Step(step_id=len(steps) + 1, **kw)
        steps.append(s)
        return s

    if instruction:
        add(source="user", message=instruction)
    pending: dict[str, Step] = {}
    for e in trace.events:
        to = e.data.get("to")
        if e.actor == actor and e.type == "tool_call":
            cid = str(e.data.get("id"))
            s = add(source="agent", timestamp=_ts(e.time), message="", model_name=model_name,
                    tool_calls=[ToolCall(tool_call_id=cid, function_name=e.data["function"], arguments=e.data.get("arguments") or {})])
            pending[cid] = s
        elif e.actor == actor and e.type == "tool_result":
            cid = str(e.data.get("id"))
            s = pending.pop(cid, None)
            if s is None:
                continue
            s.observation = Observation(results=[ObservationResult(source_call_id=cid, content=e.text or "",
                                                                   extra={"ok": e.data.get("ok"), "error": e.data.get("error")})])
        elif e.actor == actor and e.type == "message":
            add(source="agent", timestamp=_ts(e.time), message=e.text or "", model_name=model_name, extra={"to": to})
        elif e.actor == actor and e.type == "done":
            add(source="agent", timestamp=_ts(e.time), message=e.text or "", model_name=model_name, extra={"done": True})
        elif e.type == "message" and to == actor:
            add(source="user", timestamp=_ts(e.time), message=f"[from {e.actor}] {e.text or ''}", extra={"from": e.actor, "citations": e.data.get("citations") or []})
        elif e.type == "event:nl" and (to in (None, actor)):
            add(source="user", timestamp=_ts(e.time), message=f"[from {e.actor}] {e.text or ''}", extra={"event": e.data.get("event")})
    if not steps:
        add(source="system", message="(empty episode)")
    return Trajectory(session_id=session_id or trace.trace_id, agent=Agent(name=agent_name, version=agent_version, model_name=model_name),
                      steps=steps, extra={"world": trace.family, "crucible_trace": trace.trace_id, "actor": actor})
