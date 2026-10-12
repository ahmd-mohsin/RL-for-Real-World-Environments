"""Deterministic system actors (no LLM): rule-based responders and check runners.

- ``RelayActor``: ordered rules ``(pattern, reply | tool call)``; first match wins. Models bots and
  procedural colleagues ("send me the PO number" -> look it up and answer) exactly and for free.
- ``CheckRunnerActor``: a CI bot that answers with pass/fail per check name only, never the check's code or
  expected values. With ``ci_app`` it calls the World's ``ci`` App (``world/apps/ci.py``): a separate runner holds
  the hidden checks and runs them unprivileged, budgeted and traced. Without it (``view`` + ``checks``), each
  command runs in the policy's own tree, so candidate code executes during the check; that path is for checks the
  policy cannot subvert or for trusted local runs (M2 audit: it ran as the verifier, root, and called an infra
  fault "FAIL").
"""

from __future__ import annotations

import re
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


_SLOT = re.compile(r"\{(\d+|tool_result)\}")


def fill(template: str, groups: tuple, tool_result: str | None = None) -> str:
    """Substitute only ``{0}``, ``{1}``, ... and ``{tool_result}``; every other brace is literal (a JSON reply template).
    (M2 audit: ``str.format`` raised ``KeyError`` on any literal brace and the bot went silent.)"""

    def sub(m: re.Match) -> str:
        k = m.group(1)
        if k == "tool_result":
            return tool_result if tool_result is not None else m.group(0)
        i = int(k)
        return (groups[i] or "") if i < len(groups) else m.group(0)

    return _SLOT.sub(sub, template)


class RelayActor(ActorBase):
    def __init__(self, actor_id: str, rules: list[Rule], *, default: str | None = None, reply_latency: str | float | None = None):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.rules, self.default = rules, default
        self._pending: dict[str, tuple[int, tuple, str]] = {}  # call id -> (rule index, regex groups, reply-to): serializable

    def state_dict(self) -> dict[str, Any]:  # a lookup in flight survives snapshot / restore (it used to be lost)
        return {**super().state_dict(), "pending": {k: [i, list(g), to] for k, (i, g, to) in self._pending.items()}}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self._pending = {k: (int(i), tuple(g), str(to)) for k, (i, g, to) in (state.get("pending") or {}).items()}

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for tr in turn.tool_results:
            pending = self._pending.pop(tr.tool_call_id, None)
            if pending:
                i, groups, to = pending
                out.append(Message(to=to, text=fill(self.rules[i].reply or "{tool_result}", groups, tr.content)))
        for msg in turn.inbox:
            for i, rule in enumerate(self.rules):
                m = rule._rx.search(msg.text)
                if not m:
                    continue
                if rule.tool:
                    cid = f"relay-{len(self.history)}-{i}"  # deterministic: identical replays give identical traces
                    args = {k: (fill(v, m.groups()) if isinstance(v, str) else v) for k, v in rule.args.items()}
                    self._pending[cid] = (i, m.groups(), msg.frm)
                    out.append(ToolCall(tool_call_id=cid, function_name=rule.tool, arguments=args))
                else:
                    out.append(Message(to=msg.frm, text=fill(rule.reply or "", m.groups())))
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
    expect: str | None = None  # a regex the output must match for PASS (exit 0 alone can be faked by candidate code)


class CheckRunnerActor(ActorBase):
    """CI bot. ``ci_app``: the World's ``ci`` App, called as this actor (its ACL must grant ``<ci_app>.run_ci``).
    Otherwise ``view`` (a VerifierView from the task setup, never from the policy) and ``checks``: each runs as
    ``check_user`` where the exec can switch users, PASS honours ``expect``, an adapter or exec fault records a
    masking failure instead of a "FAIL", and ``max_runs`` bounds the runs."""

    def __init__(self, actor_id: str, view: VerifierView | None = None, checks: list[Check] | None = None, *, ci_app: str | None = None,
                 trigger: str = r"\b(run|ci|check|test)", reply_latency: str | float | None = "10m", check_user: str | None = "65534",
                 max_runs: int | None = None):
        super().__init__(actor_id, reply_latency=reply_latency)
        if ci_app is None and (view is None or not checks):
            raise ValueError("CheckRunnerActor needs ci_app, or a VerifierView and checks")
        self.view, self.checks, self.ci_app, self._rx = view, checks or [], ci_app, re.compile(trigger, re.IGNORECASE)
        self.check_user, self.max_runs = check_user, max_runs
        self.runs = 0
        self._pending: dict[str, str] = {}  # run_ci call id -> who asked

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "runs": self.runs, "pending": dict(self._pending)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.runs = int(state.get("runs", 0))
        self._pending = dict(state.get("pending") or {})

    async def respond(self, turn: ActorTurn) -> list[Action]:
        out: list[Action] = []
        for tr in turn.tool_results:
            to = self._pending.pop(tr.tool_call_id, None)
            if to is None:
                continue
            if tr.ok:
                out.append(Message(to=to, text=f"CI run #{self.runs}\n{tr.content}"))
            elif tr.fault in ("policy", "tool"):  # e.g. the CI budget is spent: the policy's to know
                out.append(Message(to=to, text=f"CI: {tr.error or tr.content}"))
            else:
                self.failures.append({"kind": "crucible:ci_infra", "reason": f"run_ci failed: {tr.error or tr.content}"[:300], "at": turn.time})
        for msg in turn.inbox:
            if not self._rx.search(msg.text):
                out.append(Message(to=msg.frm, text="CI: say 'run' to start the checks."))
                continue
            if self.max_runs is not None and self.runs >= self.max_runs:
                out.append(Message(to=msg.frm, text=f"CI: the run budget ({self.max_runs}) is spent."))
                continue
            self.runs += 1
            if self.ci_app is not None:
                cid = f"ci-{self.runs}"
                self._pending[cid] = msg.frm
                out.append(ToolCall(tool_call_id=cid, function_name=f"{self.ci_app}.run_ci", arguments={}))
                continue
            lines = await self._run_local(turn)
            if lines is not None:
                out.append(Message(to=msg.frm, text=f"CI run #{self.runs}\n" + "\n".join(lines)))
        return out

    async def _run_local(self, turn: ActorTurn) -> list[str] | None:
        assert self.view is not None
        lines = []
        for c in self.checks:
            exec_ = getattr(self.view.adapter(c.app), "exec", None)
            user = None if exec_ is None or getattr(exec_, "is_local", True) else self.check_user
            r = await self.view.query(c.app, {"op": "run", "args": {"cmd": c.cmd, "timeout_s": c.timeout_s, **({"user": user} if user else {})}})
            if r.exit_code is None and not (r.value or {}).get("timed_out"):  # the adapter or exec failed: no verdict on the code
                self.failures.append({"kind": "crucible:ci_infra", "reason": f"check {c.name!r}: {r.error}"[:300], "at": turn.time})
                return None
            out = (r.stdout or "") + (r.stderr or "")
            passed = r.ok and (c.expect is None or re.search(c.expect, out) is not None)
            lines.append(f"{c.name}: {'TIMEOUT' if (r.value or {}).get('timed_out') else ('PASS' if passed else 'FAIL')}")
        return lines
