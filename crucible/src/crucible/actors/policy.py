"""Policy actors and the episode loop.

A *policy* is the actor under training or evaluation: it alone calls ``World.step``. Here:
- ``ScriptedPolicy``: a fixed list of actions or a function of the observation (oracles, replays, tests);
- ``LLMToolPolicy``: any OpenAI-compatible chat model with function calling over the World's tools
  (a minimal harness for smoke tests and cheap baselines; real harnesses — Claude Code, Codex,
  mini-swe-agent, OpenHands — run through Harbor agents / MCP in M10);
- ``run_episode``: drives one or more policies round-robin until all are done or a step cap is hit,
  and reports simulator failures so the sample can be masked.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.llm import BudgetExceeded, LLMClient, LLMError
from crucible.core.schema.actions import Action, Done, Message, Observation, ToolCall
from crucible.core.schema.state import WorldRef
from crucible.core.schema.trace import Trace
from crucible.world.world import World


class Policy:
    actor_id: str

    async def act(self, obs: Observation) -> Action:  # pragma: no cover - interface
        raise NotImplementedError


class ScriptedPolicy(Policy):
    def __init__(self, actor_id: str, script: list[Action] | Callable[[Observation], Action | None]):
        self.actor_id = actor_id
        self.script = script
        self._i = 0

    async def act(self, obs: Observation) -> Action:
        if callable(self.script):
            a = self.script(obs)
            return a if a is not None else Done(reason="script finished")
        if self._i >= len(self.script):
            return Done(reason="script finished")
        a = self.script[self._i]
        self._i += 1
        return a


SYSTEM = """You are an agent working inside a simulated workplace. Use the tools to complete your instruction.
Messages from other people arrive as user messages starting with [from <name>]. To message someone, call the tool
`send_message` with fields `to` and `text`. When the work is complete, reply with a final summary and no tool call."""

SEND_MESSAGE = {"type": "function", "function": {"name": "send_message", "description": "Send a message to another person (actor) in the world.",
                "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}}}


NUDGE = "Your last reply was empty. Continue the task: call a tool, or reply with a final summary if the work is complete."


class LLMToolPolicy(Policy):
    def __init__(self, actor_id: str, llm: LLMClient, *, system: str = SYSTEM, max_tokens: int = 1200, max_tool_output: int = 6000,
                 max_nudges: int = 2):
        self.actor_id, self.llm, self.system = actor_id, llm, system
        self.max_tokens, self.max_tool_output = max_tokens, max_tool_output
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.tools: list[dict[str, Any]] = []
        self._queue: list[Action] = []
        self.error: str | None = None
        self._nudges, self.max_nudges = 0, max_nudges

    def _ingest(self, obs: Observation) -> None:
        if obs.tools:
            self.tools = [t.to_openai() for t in obs.tools] + [SEND_MESSAGE]
        if obs.instruction:
            self.messages.append({"role": "user", "content": obs.instruction})
        for tr in obs.tool_results:
            self.messages.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": tr.content[: self.max_tool_output]})
        for m in obs.inbox:
            self.messages.append({"role": "user", "content": f"[from {m.frm}{' at ' + m.at if m.at else ''}] {m.text}"})

    async def act(self, obs: Observation) -> Action:
        self._ingest(obs)
        if self._queue:
            return self._queue.pop(0)
        try:
            r = await self.llm.complete(self.messages, tools=self.tools or None, max_tokens=self.max_tokens)
        except (LLMError, BudgetExceeded) as e:
            self.error = f"{getattr(e, 'failure_kind', 'provider_unavailable')}: {e}"
            return Done(reason=f"policy error: {e}")
        if not r.tool_calls and not (r.text or "").strip() and self._nudges < self.max_nudges:
            # A blank turn (e.g. a reasoning model that spent its budget thinking) is not a decision to stop.
            self._nudges += 1
            self.messages.append({"role": "user", "content": NUDGE})  # no empty assistant turn: some providers reject it
            return await self.act(Observation(actor=obs.actor, step=obs.step))
        if not r.tool_calls:
            self.messages.append({"role": "assistant", "content": r.text})
            return Done(reason=r.text[:2000])
        calls = []
        actions: list[Action] = []
        for tc in r.tool_calls:
            cid = tc["id"] or f"call-{uuid.uuid4().hex[:8]}"
            calls.append({"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}})
            if tc["name"] == "send_message":
                actions.append(Message(to=str(tc["arguments"].get("to", "")), text=str(tc["arguments"].get("text", ""))))
            else:
                actions.append(ToolCall(tool_call_id=cid, function_name=tc["name"].replace("__", ".", 1), arguments=tc["arguments"]))
        self.messages.append({"role": "assistant", "content": r.text or "", "tool_calls": calls})
        for a, c in zip(actions, calls, strict=True):
            if isinstance(a, Message):  # every tool call id needs a tool reply; send_message is acknowledged here
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"message sent to {a.to}"})
        self._queue = actions[1:]
        return actions[0]


@dataclass
class EpisodeResult:
    steps: int
    done: bool
    trace: Trace
    final: WorldRef
    masked: bool
    failures: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, list[Observation]] = field(default_factory=dict)


async def run_episode(world: World, policies: dict[str, Policy], *, max_steps: int = 50, reset: bool = True) -> EpisodeResult:
    obs = await world.reset() if reset else {a: Observation(actor=a, step=world.step_index) for a in policies}
    log: dict[str, list[Observation]] = {a: [o] for a, o in obs.items()}
    order = list(policies)
    current = {a: obs.get(a) or Observation(actor=a, step=0) for a in order}
    steps, done = 0, False
    while steps < max_steps and not done:
        progressed = False
        for a in order:
            if a in world.done:
                continue
            action = await policies[a].act(current[a])
            res = await world.step(a, action)
            current[a] = res.observation
            log.setdefault(a, []).append(res.observation)
            steps += 1
            progressed = True
            done = res.done
            if done or steps >= max_steps:
                break
        if not progressed:
            break
    failures = [f for r in world.responders.values() for f in getattr(r, "failures", [])]
    failures += [{"kind": "policy_error", "reason": p.error} for p in policies.values() if getattr(p, "error", None)]
    final = await world.snapshot(quiesce=True)  # the episode is over: nothing the policy started may keep changing state
    return EpisodeResult(steps=steps, done=done, trace=world.to_trace(), final=final, masked=bool(failures), failures=failures, observations=log)
