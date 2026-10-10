"""Step segmentation (DESIGN §5.3.1): a trace's events -> the steps credit is assigned to.

A **step** is the unit a reward channel scores and an advantage is attached to: one thing an actor did that may have
moved the state, plus what it observed in return.

- **Agent sessions** (events ``tool_call`` / ``tool_result``, from ATIF, the capture proxy, OTel or a World): one step
  per tool call that can change state. Consecutive read-only calls by the same actor merge into a single
  ``investigate`` step, so "read five files" is one step, not five. Agent messages and reasoning attach to the
  step they precede.
- **Pull requests** (``commit``, ``review_comment``, ``ci_run``, ...): one step per commit (the state changes), one per
  review round (consecutive comments by one reviewer), one per CI run.
- **Event logs** (tickets, ERP audit rows): one step per event that writes an object; non-writing events by the
  same actor fold into the next writing one.

Packs may register a ``segmenter`` for a family (registry kind ``segmenter``; the longest family prefix wins, as for
signal mappers); the generic segmenter handles all three shapes and is what the tracker uses on World episodes.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import Field

from crucible.core.schema.base import CrucibleModel
from crucible.core.schema.trace import Trace, TraceEvent
from crucible.workflow.actions import ActionKind, classify

StepKind = ActionKind | Literal["commit", "review", "ci", "message", "state"]
STATE_CHANGING: frozenset[str] = frozenset({"edit", "build", "test", "run", "act", "submit", "commit", "state"})


class Step(CrucibleModel):
    index: int
    seqs: list[int]  # trace events that make up the step (call, result, preceding message)
    actor: str | None = None
    kind: str  # StepKind
    tools: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    state_ref: str | None = None  # the state after the step, when the trace records it (git:<sha>, world:<ref>)
    t0: str | None = None
    t1: str | None = None
    calls: int = 0
    label: str | None = None  # short human-readable summary
    data: dict[str, Any] = Field(default_factory=dict)

    @property
    def last_seq(self) -> int:
        return max(self.seqs)


@runtime_checkable
class Segmenter(Protocol):
    def segment(self, trace: Trace) -> list[Step]: ...


def _tool_and_args(e: TraceEvent) -> tuple[str, Any]:
    return str(e.data.get("tool") or e.data.get("function") or ""), e.data.get("arguments")


class GenericSegmenter:
    """Agent sessions, PR traces and event logs (see the module docstring)."""

    def segment(self, trace: Trace) -> list[Step]:
        types = {e.type for e in trace.events}
        if "tool_call" in types:
            return self._agent(trace)
        if "commit" in types:
            return self._pr(trace)
        return self._log(trace)

    # ------------------------------------------------------------------ agent sessions
    def _agent(self, trace: Trace) -> list[Step]:
        steps: list[Step] = []
        pending: list[int] = []  # messages / reasoning waiting for the next call
        results = {e.reply_to: e for e in trace.events if e.type == "tool_result" and e.reply_to is not None}
        consumed: set[int] = set()

        def new(kind: str, e: TraceEvent, seqs: list[int], tool: str, label: str) -> None:
            steps.append(Step(index=len(steps), seqs=seqs, actor=e.actor, kind=kind, tools=[tool] if tool else [], writes=list(e.data.get("writes") or []),
                              t0=e.time, t1=e.time, calls=1 if tool else 0, label=label[:160], data={"probes": e.data["probes"]} if e.data.get("probes") else {}))

        for e in trace.events:
            if e.type == "tool_call":
                tool, args = _tool_and_args(e)
                kind = classify(tool, args)
                r = results.get(e.seq)
                seqs = [*pending, e.seq, *([r.seq] if r is not None else [])]
                if r is not None:
                    consumed.add(r.seq)
                pending = []
                prev = steps[-1] if steps else None
                if kind == "investigate" and prev is not None and prev.kind == "investigate" and prev.actor == e.actor:
                    prev.seqs = sorted({*prev.seqs, *seqs})  # read-only calls in a row are one investigation
                    prev.calls += 1
                    if tool and tool not in prev.tools:
                        prev.tools = [*prev.tools, tool]
                    prev.t1 = e.time or prev.t1
                    continue
                new(kind, e, seqs, tool, e.text or tool)
            elif e.type == "tool_result":
                if e.seq not in consumed and steps:  # an unlinked result belongs to the latest step
                    steps[-1].seqs = sorted({*steps[-1].seqs, e.seq})
            elif e.type in ("user_message", "interrupt", "system_message", "copied_context", "context_reset"):
                if pending:
                    steps.append(Step(index=len(steps), seqs=pending, actor="agent", kind="message", t0=None))
                    pending = []
                steps.append(Step(index=len(steps), seqs=[e.seq], actor=e.actor, kind="message", t0=e.time, t1=e.time, label=(e.text or e.type)[:160],
                                  data={"type": e.type}))
            else:  # agent_message, llm_call, done, reset, ...
                pending.append(e.seq)
        if pending:
            steps.append(Step(index=len(steps), seqs=pending, actor="agent", kind="message"))
        return steps

    # ------------------------------------------------------------------ pull requests
    def _pr(self, trace: Trace) -> list[Step]:
        steps: list[Step] = []
        for e in sorted(trace.events, key=lambda x: (x.time or "", x.seq)):
            if e.type == "commit":
                steps.append(Step(index=len(steps), seqs=[e.seq], actor=e.actor, kind="commit", writes=list(e.data.get("writes") or []),
                                  state_ref=e.state_ref, t0=e.time, t1=e.time, label=(e.text or "").splitlines()[0][:160] if e.text else None,
                                  data={k: e.data[k] for k in ("sha", "superseded") if k in e.data}))
            elif e.type in ("review_comment", "review"):
                prev = steps[-1] if steps else None
                if prev is not None and prev.kind == "review" and prev.actor == e.actor:
                    prev.seqs = [*prev.seqs, e.seq]
                    prev.t1 = e.time or prev.t1
                    continue
                steps.append(Step(index=len(steps), seqs=[e.seq], actor=e.actor, kind="review", t0=e.time, t1=e.time, label=(e.text or "")[:160]))
            elif e.type in ("ci_run", "check_run"):
                steps.append(Step(index=len(steps), seqs=[e.seq], actor=e.actor, kind="ci", state_ref=e.state_ref, t0=e.time, t1=e.time,
                                  label=f"{e.data.get('name')}: {e.data.get('conclusion')}"))
            else:
                steps.append(Step(index=len(steps), seqs=[e.seq], actor=e.actor, kind="message" if e.kind == "message" else "state",
                                  writes=list(e.data.get("writes") or []), state_ref=e.state_ref, t0=e.time, t1=e.time, label=e.type))
        return steps

    # ------------------------------------------------------------------ event logs
    def _log(self, trace: Trace) -> list[Step]:
        steps: list[Step] = []
        pending: list[TraceEvent] = []
        for e in trace.events:
            writes = list(e.data.get("writes") or [])
            if not writes:
                pending.append(e)
                continue
            seqs = [p.seq for p in pending if p.actor == e.actor] + [e.seq]
            for p in [p for p in pending if p.actor != e.actor]:  # others' events stay their own steps
                steps.append(Step(index=len(steps), seqs=[p.seq], actor=p.actor, kind="message" if p.kind == "message" else "state", t0=p.time, t1=p.time, label=p.type))
            pending = []
            steps.append(Step(index=len(steps), seqs=sorted(seqs), actor=e.actor, kind="message" if e.kind == "message" else ("act" if e.kind == "action" else "state"),
                              writes=writes, state_ref=e.state_ref, t0=e.time, t1=e.time, label=e.type))
        for p in pending:
            steps.append(Step(index=len(steps), seqs=[p.seq], actor=p.actor, kind="message" if p.kind == "message" else "state", t0=p.time, t1=p.time, label=p.type))
        return steps


def segmenter_for(family: str | None) -> Segmenter:
    from crucible.core.registry import get_registry

    reg = get_registry()
    names = sorted((n for n in reg.names("segmenter") if family and (family == n or family.startswith(n + "/"))), key=len, reverse=True)
    return reg.get("segmenter", names[0])() if names else GenericSegmenter()


def segment(trace: Trace) -> list[Step]:
    return segmenter_for(trace.family).segment(trace)
