"""Scenario-driven simulated person (user, customer, approver, another department).

The persona prompt combines three public designs:
- verifiers v1 ``UserSimEnv`` (role-play the USER; never do the assistant's work; reveal details only
  when asked; reply with a done marker when the goal is met);
- NeMo-Gym ``conversational_tool_use_simulation`` scenario fields (persona, reason for contact, known
  info, unknown info, task instructions, outside-policy scope) and its rule that a simulator failure
  masks the sample;
- tau2 simulation guidelines (never hallucinate — information not in the scenario is unknown;
  disclose progressively; an out-of-scope token when the scenario cannot answer).

Crucible adds: an optional per-reply **consistency judge** from a different model family (NeMo-Gym
judges each user message), bounded regeneration, and tool use under the actor's own permissions
(an approver can actually update the ERP).
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from pydantic import Field

from crucible.actors.base import ActorBase
from crucible.actors.llm import BudgetExceeded, LLMClient, LLMError
from crucible.core.schema.actions import Action, Done, Message, ToolCall
from crucible.core.schema.base import CrucibleModel
from crucible.world.world import ActorTurn

DONE = "###DONE###"
OUT_OF_SCOPE = "###OUT-OF-SCOPE###"


class Scenario(CrucibleModel):
    persona: str
    reason_for_contact: str = ""
    known_info: str = ""
    unknown_info: str = ""
    task_instructions: str = ""
    outside_policy_scope: bool = False
    opening: str | None = None  # first message to send if the actor opens the conversation
    extra: dict[str, Any] = Field(default_factory=dict)

    def render(self) -> str:
        parts = [f"Persona: {self.persona}"]
        if self.reason_for_contact:
            parts.append(f"Reason for contact: {self.reason_for_contact}")
        if self.known_info:
            parts.append(f"What you know (you may share it when asked): {self.known_info}")
        if self.unknown_info:
            parts.append(f"What you do NOT know (say you don't know if asked): {self.unknown_info}")
        if self.task_instructions:
            parts.append(f"Instructions: {self.task_instructions}")
        return "\n".join(parts)


PERSONA = """You are role-playing a person in a workplace simulation. This is your situation:

<scenario>
{scenario}
</scenario>

Rules:
- Stay in character: short, natural messages; you are NOT an AI assistant and you never do the other party's work for them.
- Never make up information that is not in the scenario. Anything not stated there is unknown to you.
- Reveal details progressively, only when asked, as a real person would; do not recite the scenario.
- If you have tools, use them only as this person would in their job.
- When your goal is fully met, or you are convinced it cannot be met, reply with exactly {done} and nothing else.
- If you are asked something the scenario gives you no basis to answer, reply with exactly {oos}."""

JUDGE = """You audit a simulated person's reply for consistency.
Scenario:
<scenario>
{scenario}
</scenario>
Conversation so far (the simulated person is "SIM"):
{history}
Candidate reply from SIM:
<reply>{reply}</reply>
Return only JSON {{"consistent": true|false, "reason": "<one short sentence>"}}. A reply is inconsistent if it states a fact not supported by the
scenario or the conversation, contradicts either, breaks character (acts as an assistant / does the other party's task), or
reveals information the scenario says SIM does not know."""


class SimulatedActor(ActorBase):
    def __init__(self, actor_id: str, scenario: Scenario, llm: LLMClient, *, judge: LLMClient | None = None, max_regenerations: int = 2,
                 use_tools: bool = False, max_tokens: int = 300, reply_latency: str | float | None = None):
        super().__init__(actor_id, reply_latency=reply_latency)
        self.scenario, self.llm, self.judge = scenario, llm, judge
        self.max_regenerations, self.use_tools, self.max_tokens = max_regenerations, use_tools, max_tokens
        self.audit: list[dict[str, Any]] = []

    def state_dict(self) -> dict[str, Any]:
        return {**super().state_dict(), "audit": list(self.audit)}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        super().load_state_dict(state)
        self.audit = list(state.get("audit", []))

    def opening_message(self, to: str) -> Message | None:
        if not self.scenario.opening:
            return None
        self.history.append({"role": "assistant", "content": self.scenario.opening})
        return Message(to=to, text=self.scenario.opening)

    def _messages(self) -> list[dict[str, Any]]:
        sys = PERSONA.format(scenario=self.scenario.render(), done=DONE, oos=OUT_OF_SCOPE)
        return [{"role": "system", "content": sys}, *self.history]

    async def respond(self, turn: ActorTurn) -> list[Action]:
        if self.finished:
            return []
        for m in turn.inbox:
            self.history.append({"role": "user", "content": f"[{m.frm}] {m.text}"})
        for tr in turn.tool_results:
            self.history.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content[:6000]})
        if not turn.inbox and not turn.tool_results:
            return []
        reply_to = turn.inbox[-1].frm if turn.inbox else (self.history_last_sender() or "")
        tools = [t.to_openai() for t in turn.tools] if self.use_tools and turn.tools else None
        try:
            for attempt in range(self.max_regenerations + 1):
                r = await self.llm.complete(self._messages(), tools=tools, max_tokens=self.max_tokens, temperature=0.0 if attempt == 0 else 0.7, seed=attempt)
                if r.tool_calls:
                    calls = []
                    msg_calls = []
                    for tc in r.tool_calls:
                        fname = tc["name"].replace("__", ".", 1)
                        cid = tc["id"] or f"sim-{uuid.uuid4().hex[:8]}"
                        calls.append(ToolCall(tool_call_id=cid, function_name=fname, arguments=tc["arguments"]))
                        msg_calls.append({"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}})
                    self.history.append({"role": "assistant", "content": r.text or "", "tool_calls": msg_calls})
                    return list(calls)
                text = (r.text or "").strip()
                marker = DONE if DONE in text else (OUT_OF_SCOPE if OUT_OF_SCOPE in text else None)
                if marker == DONE and not any(h["role"] == "assistant" for h in self.history):
                    verdict = {"consistent": False, "reason": "ended the conversation before saying anything"}  # NeMo-Gym: no STOP first
                elif marker:
                    verdict = {"consistent": True}
                else:
                    verdict = await self._judge(text) if self.judge else {"consistent": True}
                self.audit.append({"reply": text[:500], **verdict, "attempt": attempt})
                if verdict.get("consistent", True):
                    self.history.append({"role": "assistant", "content": text})
                    if marker:  # the conversation ends; the World records why (the policy never sees the marker)
                        self.finished = True
                        before = text.split(marker, 1)[0].strip()
                        out: list[Action] = [Message(to=reply_to, text=before)] if before else []
                        return [*out, Done(reason="user_stop" if marker == DONE else "out_of_scope")]
                    return [Message(to=reply_to, text=text)]
            self.failures.append({"kind": "crucible:sim_failure", "reason": "inconsistent after regenerations", "at": turn.time})
            return []
        except (LLMError, BudgetExceeded) as e:
            self.failures.append({"kind": getattr(e, "failure_kind", "provider_unavailable"), "reason": str(e)[:300], "at": turn.time})
            return []

    def history_last_sender(self) -> str | None:
        for h in reversed(self.history):
            if h.get("role") == "user" and h["content"].startswith("["):
                return h["content"][1:].split("]", 1)[0]
        return None

    async def _judge(self, reply: str) -> dict[str, Any]:
        assert self.judge is not None
        hist = "\n".join(f"{'SIM' if h['role'] == 'assistant' else 'OTHER'}: {h.get('content', '')}" for h in self.history if h["role"] in ("user", "assistant"))
        try:
            v = await self.judge.complete_json([{"role": "user", "content": JUDGE.format(scenario=self.scenario.render(), history=hist or "(none)", reply=reply)}], max_tokens=400)
            return {"consistent": bool(v.get("consistent", False)), "reason": str(v.get("reason", ""))[:300]}
        except LLMError as e:
            import re

            m = re.search(r'"consistent"\s*:\s*(true|false)', str(e))  # verdict survives a truncated reason
            if m:
                return {"consistent": m.group(1) == "true", "reason": "(truncated)"}
            self.failures.append({"kind": "judge_failed", "reason": str(e)[:300]})
            return {"consistent": True, "reason": "judge unavailable"}
        except BudgetExceeded as e:
            self.failures.append({"kind": "judge_failed", "reason": str(e)[:300]})
            return {"consistent": True, "reason": "judge unavailable"}
