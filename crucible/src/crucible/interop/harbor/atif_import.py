"""ATIF trajectories -> ``crucible.trace/v1`` (M3 ``atif_import``; DESIGN §5.1 ``agent_sessions``).

Harbor converts 40+ agent harnesses' native logs to ATIF (``src/harbor/agents/installed/``: Claude Code session
JSONL, mini-swe-agent, Codex, OpenHands, ...). Crucible depends on those converters instead of writing its own
and maps ATIF once, here, onto trace events:

  user step       -> message ``user_message`` (``interrupt`` when the user stopped the agent)
  agent step      -> message ``agent_message`` (text), then one action ``tool_call`` per tool call
  observation     -> observation ``tool_result``, ``reply_to`` the call it answers (``data.rejected`` when the user
                     refused the call)
  subagent steps  -> the same events, ``data.subagent`` naming the subagent trajectory

File-editing tool calls record what they wrote (``data.writes = ["file:<path>"]``), which M4 uses to attribute
later corrections to the step that made the edit. The full ATIF document can be kept in the store; events then
carry ``atif_ref = "<key>#<step_id>"`` and only a bounded text.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from harbor.models.trajectories import Trajectory

from crucible.connectors.base import RawRecord, SourceRef
from crucible.core.agent_events import INTERRUPT, REJECTED, annotate_probes, writes_of
from crucible.core.schema.signal import SignalTarget
from crucible.core.schema.trace import Trace, TraceActor, TraceEvent, TraceSource

MAX_TEXT = 4000


def text_of(message: Any) -> str:
    if message is None:
        return ""
    if isinstance(message, str):
        return message
    return "\n".join(p.text for p in message if getattr(p, "text", None))


def _short(v: Any, n: int = 300) -> str:
    s = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, default=str)
    return s if len(s) <= n else s[:n] + "…"


def trajectory_to_trace(traj: Trajectory, *, trace_id: str, source: TraceSource, family: str | None = None,
                        atif_key: str | None = None, max_text: int = MAX_TEXT) -> Trace:
    model = traj.agent.model_name if traj.agent else None
    actors = [TraceActor(id="user", role="user", kind="human"), TraceActor(id="agent", role="agent", kind="agent", handle=model or (traj.agent.name if traj.agent else None)),
              TraceActor(id="system", role="system", kind="system")]
    events: list[TraceEvent] = []
    call_seq: dict[str, int] = {}

    def add(**kw: Any) -> int:
        seq = len(events)
        events.append(TraceEvent(seq=seq, **kw))
        return seq

    def walk(t: Trajectory, sub: str | None) -> None:
        for step in t.steps:
            ref = f"{atif_key}#{step.step_id}" if atif_key else None
            extra = step.extra or {}
            # Claude Code subagents are merged into the main step list and marked only in ``extra`` (Harbor): their
            # prompts were human user_messages and their writes the main agent's (M3 audit)
            here = sub or (str(extra.get("agent_id") or "sidechain") if extra.get("is_sidechain") else None)
            base = {"step_id": step.step_id, **({"subagent": here} if here else {})}
            msg = text_of(step.message)
            if step.source == "user":
                if here:  # the task the main agent gave its subagent: written by the agent, not a person
                    add(time=step.timestamp, kind="message", actor="agent", type="subagent_prompt", text=msg[:max_text], atif_ref=ref, data=base)
                    continue
                typ = "interrupt" if INTERRUPT.search(msg) else ("copied_context" if step.is_copied_context else "user_message")
                add(time=step.timestamp, kind="message", actor="user", type=typ, text=msg[:max_text], atif_ref=ref, data=base)
                continue
            if step.is_copied_context:  # replayed from an earlier trajectory: not new work, not training data (Harbor step.py)
                continue
            if step.source == "system":
                add(time=step.timestamp, kind="message", actor="system", type="system_message", text=msg[:max_text], atif_ref=ref, data=base)
                continue
            usage = step.metrics.model_dump(exclude_none=True, exclude={"prompt_token_ids", "completion_token_ids", "logprobs"}) if step.metrics else {}
            if msg.strip() or step.reasoning_content:
                add(time=step.timestamp, kind="message", actor="agent", type="agent_message", text=msg[:max_text], atif_ref=ref,
                    data={**base, "model": step.model_name, "usage": usage, **({"reasoning": step.reasoning_content[:max_text]} if step.reasoning_content else {})})
            step_calls: list[int] = []
            for tc in step.tool_calls or []:
                writes = writes_of(tc.function_name, tc.arguments or {})
                seq = add(time=step.timestamp, kind="action", actor="agent", type="tool_call", text=f"{tc.function_name}({_short(tc.arguments, 200)})", atif_ref=ref,
                          target=SignalTarget(app="workspace", object=writes[0]) if writes else None,
                          data={**base, "tool": tc.function_name, "call_id": tc.tool_call_id, "arguments": _short(tc.arguments, max_text),
                                **({"writes": writes} if writes else {})})
                call_seq[tc.tool_call_id] = seq
                step_calls.append(seq)
            for i, r in enumerate(step.observation.results if step.observation else []):
                out = text_of(r.content) if not isinstance(r.content, str) else r.content
                is_error = (getattr(r, "extra", None) or {}).get("tool_result_is_error") or extra.get("tool_result_is_error")
                add(time=step.timestamp, kind="observation", actor="system", type="tool_result", text=(out or "")[:max_text], atif_ref=ref,
                    # some converters (mini-swe-agent) leave source_call_id empty: results then answer the step's calls in order
                    reply_to=call_seq.get(r.source_call_id or "", step_calls[i] if i < len(step_calls) else None),
                    data={**base, "call_id": r.source_call_id, "chars": len(out or ""), **({"rejected": True} if REJECTED.search(out or "") else {}),
                          **({"error": True} if is_error else {})})
        for s in t.subagent_trajectories or []:
            walk(s, s.trajectory_id or s.session_id or "subagent")  # siblings inherit the parent's session id

    walk(traj, None)
    probes = annotate_probes(events)
    fm = traj.final_metrics.model_dump(exclude_none=True) if traj.final_metrics else {}
    return Trace(trace_id=trace_id, family=family, source=source, actors=actors, events=events,
                 outcome={"label": "unknown", "details": {"agent": traj.agent.name if traj.agent else None, "model": model, "atif": atif_key,  # type: ignore[arg-type]
                                                          "schema": traj.schema_version, "final_metrics": fm,
                                                          **({"probes": probes} if probes["attempted"] else {})}})


def convert_mini_swe_agent(data: dict[str, Any], session_id: str) -> Trajectory:
    from harbor.agents.installed.mini_swe_agent import convert_mini_swe_agent_to_atif

    return convert_mini_swe_agent_to_atif(data, session_id=session_id)


def convert_claude_code_sessions(session_dir: Path) -> Trajectory | None:
    """A Claude Code project/session directory (``<id>.jsonl`` + ``subagents/*.jsonl``) -> one ATIF trajectory."""
    import tempfile

    from harbor.agents.installed.claude_code import ClaudeCode

    with tempfile.TemporaryDirectory(prefix="crucible-cc-") as d:  # was a mkdtemp per conversion, never removed
        return ClaudeCode(logs_dir=Path(d))._convert_events_to_trajectory(Path(session_dir))


class AtifConnector:
    """Any ATIF trajectory files: scope {"glob": "runs/**/trajectory.atif.json", "family": "..."}."""

    name = "atif"

    def __init__(self, family: str | None = None, privacy: str = "internal"):
        self.family, self.privacy = family, privacy

    def discover(self, scope: dict[str, Any]) -> Iterator[SourceRef]:
        for p in sorted(Path().glob(scope["glob"]) if not Path(scope["glob"]).is_absolute() else Path("/").glob(scope["glob"].lstrip("/"))):
            yield SourceRef(connector=self.name, key=str(p), meta={"path": str(p), "family": scope.get("family", self.family)})

    def fetch(self, ref: SourceRef) -> RawRecord:
        return RawRecord(ref=ref, payload=json.loads(Path(ref.meta["path"]).read_text()), privacy=self.privacy)  # type: ignore[arg-type]

    def normalize(self, raw: RawRecord) -> Trace:
        traj = Trajectory.model_validate(raw.payload)
        # a session's continuation segments share its session id: the trajectory id tells them apart
        sid, trid = traj.session_id, getattr(traj, "trajectory_id", None)
        tid = f"{sid}:{trid}" if sid and trid and trid != sid else (sid or trid or Path(raw.ref.key).stem)
        return trajectory_to_trace(traj, trace_id=f"atif:{tid}", family=raw.ref.meta.get("family"),
                                   source=TraceSource(connector=self.name, ref=raw.ref.key, privacy=raw.privacy))
