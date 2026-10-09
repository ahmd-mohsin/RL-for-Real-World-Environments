"""Deterministic system actors (no LLM): rule-based responders and check runners.

- ``RelayActor``: ordered rules ``(pattern, reply | tool call)``; first match wins. Models bots and
  procedural colleagues ("send me the PO number" -> look it up and answer) exactly and for free.
- ``CheckRunnerActor``: a CI bot. When messaged, it runs its configured checks through a *verifier-side*
  State Adapter query (so check scripts can live outside the policy's reach) and answers with
  pass/fail per check name only — never the check's code or expected values.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.base import ActorBase
from crucible.core.schema.actions import Action, Message, ToolCall
from crucible.world.world import ActorTurn, VerifierView


@dataclass
class Rule:
    pattern: str
    reply: str | None = None  # may use {0}, {1}... regex groups and {tool_result}
    tool: str | None = None  # "<app>.<tool>", called before replying
    args: dict[str, Any] = field(default_factory=dict)  # values may use regex groups "{1}"

    def __post_init__(self) -> None:
        self._rx = re.compile(self.pattern, re.IGNORECASE | re.DOTALL)


class RelayActor(ActorBase):
    def __init__(self, actor_id: str, rules: list[Rule], *, default: str | None = None, reply_latency: str | float | None = None):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.rules, self.default = rules, default
        self._pending: dict[str, tuple[Rule, re.Match, str]] = {}

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for tr in turn.tool_results:
            pending = self._pending.pop(tr.tool_call_id, None)
            if pending:
                rule, m, to = pending
                out.append(Message(to=to, text=(rule.reply or "{tool_result}").format(*m.groups(), tool_result=tr.content)))
        for msg in turn.inbox:
            for rule in self.rules:
                m = rule._rx.search(msg.text)
                if not m:
                    continue
                if rule.tool:
                    cid = f"relay-{uuid.uuid4().hex[:8]}"
                    args = {k: (v.format(*m.groups()) if isinstance(v, str) else v) for k, v in rule.args.items()}
                    self._pending[cid] = (rule, m, msg.frm)
                    out.append(ToolCall(tool_call_id=cid, function_name=rule.tool, arguments=args))
                else:
                    out.append(Message(to=msg.frm, text=(rule.reply or "").format(*m.groups())))
                break
            else:
                if self.default:
                    out.append(Message(to=msg.frm, text=self.default))
            self.history.append({"from": msg.frm, "text": msg.text, "at": turn.time})
        return out


@dataclass
class Check:
    name: str
    app: str
    cmd: str
    timeout_s: float = 1800


class CheckRunnerActor(ActorBase):
    """CI bot. Needs a VerifierView (given by the task/episode setup, never by the policy)."""

    def __init__(self, actor_id: str, view: VerifierView, checks: list[Check], *, trigger: str = r"\b(run|ci|check|test)", reply_latency: str | float | None = "10m"):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.view, self.checks, self._rx = view, checks, re.compile(trigger, re.IGNORECASE)
        self.runs = 0

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "runs": self.runs}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.runs = int(state.get("runs", 0))

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for msg in turn.inbox:
            if not self._rx.search(msg.text):
                out.append(Message(to=msg.frm, text="CI: say 'run' to start the checks."))
                continue
            self.runs += 1
            lines = []
            for c in self.checks:
                r = await self.view.query(c.app, {"op": "run", "args": {"cmd": c.cmd, "timeout_s": c.timeout_s}})
                status = "TIMEOUT" if (r.value or {}).get("timed_out") else ("PASS" if r.ok else "FAIL")
                lines.append(f"{c.name}: {status}")
            out.append(Message(to=msg.frm, text=f"CI run #{self.runs}\n" + "\n".join(lines)))
        return out
