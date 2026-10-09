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

import asyncio
import json
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMClient, LLMError
from crucible.core.schema.actions import Action, Done, Message, Observation, ToolCall, openai_name
from crucible.core.schema.state import WorldRef
from crucible.core.schema.trace import Trace
from crucible.world.world import World, WorldError


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
    """A minimal function-calling harness. Parallel tool calls in one turn are executed in order and their results
    are appended together before any message that arrived meanwhile (a valid history for OpenAI-style APIs);
    provider reasoning fields are replayed verbatim; tool output is bounded once, by the World."""

    def __init__(self, actor_id: str, llm: LLMClient, *, system: str = SYSTEM, max_tokens: int = 1200, max_tool_output: int | None = None,
                 max_nudges: int = 2, temperature: float = 0.0):
        self.actor_id, self.llm, self.system = actor_id, llm, system
        self.max_tokens, self.max_tool_output, self.temperature = max_tokens, max_tool_output, temperature
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.tools: list[dict[str, Any]] = []
        self._names: dict[str, str] = {}  # openai function name -> qualified tool name
        self._queue: list[Action] = []
        self._pending: set[str] = set()  # tool call ids of the current batch still waiting for results
        self._held: list[dict[str, Any]] = []  # messages that arrived while a batch was in flight
        self.error: str | None = None
        self.failure_kind: str | None = None
        self.stop: str | None = None
        self.sample_key: str | None = None
        self.model_calls: list[dict[str, Any]] = []
        self._nudges, self.max_nudges = 0, max_nudges

    def begin_episode(self, episode_id: str) -> None:
        self.sample_key = episode_id

    def _ingest(self, obs: Observation) -> None:
        if obs.tools:
            self.tools, self._names = [], {}
            for t in obs.tools:
                fn = t.to_openai()
                name = openai_name(t.qualified)
                fn["function"]["name"] = name
                self._names[name] = t.qualified
                self.tools.append(fn)
            self.tools.append(SEND_MESSAGE)
        if obs.instruction:
            self.messages.append({"role": "user", "content": obs.instruction})
        for tr in obs.tool_results:
            content = tr.content
            if self.max_tool_output and len(content) > self.max_tool_output:
                h = self.max_tool_output * 2 // 3
                content = content[:h] + f"\n[... {len(content) - self.max_tool_output} characters truncated ...]\n" + content[-(self.max_tool_output - h):]
            self.messages.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": content})
            self._pending.discard(tr.tool_call_id)
        for m in obs.inbox:
            self._held.append({"role": "user", "content": f"[from {m.frm}{' at ' + m.at if m.at else ''}] {m.text}"})
        if not self._pending:  # the batch is complete: now the messages that arrived during it
            self.messages += self._held
            self._held = []

    async def act(self, obs: Observation) -> Action:
        self._ingest(obs)
        if self._queue:
            return self._queue.pop(0)
        try:
            r = await self.llm.complete(self.messages, tools=self.tools or None, max_tokens=self.max_tokens, temperature=self.temperature,
                                        sample_key=self.sample_key)
        except ContextOverflow as e:
            self.stop = "context_overflow"  # truncated, scored: a long-horizon failure is data, not an outage
            return Done(reason=f"context overflow: {e}")
        except (LLMError, BudgetExceeded) as e:
            self.failure_kind = getattr(e, "failure_kind", None) or "provider_unavailable"
            self.error = f"{self.failure_kind}: {e}"
            return Done(reason=f"policy error: {e}")
        self.model_calls.append({"model": r.model, "finish": r.raw_finish, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                                 "cached": r.cached, "response_id": r.response_id, "tool_calls": len(r.tool_calls)})
        if not r.tool_calls and not (r.text or "").strip() and self._nudges < self.max_nudges:
            # A blank turn (e.g. a reasoning model that spent its budget thinking) is not a decision to stop.
            self._nudges += 1
            self.messages.append({"role": "user", "content": NUDGE})  # no empty assistant turn: some providers reject it
            return await self.act(Observation(actor=obs.actor, step=obs.step))
        if not r.tool_calls:
            self.messages.append({**r.raw_message, "role": "assistant", "content": r.text})
            self.stop = "agent_done"
            return Done(reason=r.text[:2000])
        calls = []
        actions: list[Action] = []
        for tc in r.tool_calls:
            cid = tc["id"] or f"call-{uuid.uuid4().hex[:8]}"
            raw_args = tc["arguments"].get("__unparseable__") if "__unparseable__" in tc["arguments"] else None
            calls.append({"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": raw_args if raw_args is not None else json.dumps(tc["arguments"])}})
            if tc["name"] == "send_message":
                actions.append(Message(to=str(tc["arguments"].get("to", "")), text=str(tc["arguments"].get("text", ""))))
            else:
                qualified = self._names.get(tc["name"]) or tc["name"].replace("__", ".", 1)
                actions.append(ToolCall(tool_call_id=cid, function_name=qualified, arguments=tc["arguments"]))
        # provider fields (reasoning, reasoning_details, thinking signatures) replay verbatim: reasoning models degrade without them
        self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or "", "tool_calls": calls})
        for a, c in zip(actions, calls, strict=True):
            if isinstance(a, Message):  # every tool call id needs a tool reply; send_message is acknowledged here
                self.messages.append({"role": "tool", "tool_call_id": c["id"], "content": f"message sent to {a.to}"})
            else:
                self._pending.add(c["id"])
        self._queue = actions[1:]
        return actions[0]


@dataclass
class EpisodeLimits:
    """Stop conditions of one episode (verifiers' RolloutLimits / tau2's max_errors)."""

    max_steps: int = 50
    max_tool_errors: int | None = 25  # consecutive failed policy calls: a policy stuck in an error loop ends "too_many_errors"
    timeout_s: float | None = None  # wall clock for the whole episode; the final snapshot is still taken
    end_on: tuple[str, ...] = ()  # simulated actors whose end of conversation ends the episode (stop "user_stop" / "out_of_scope")


@dataclass
class EpisodeResult:
    steps: int
    done: bool
    trace: Trace
    final: WorldRef
    masked: bool
    failures: list[dict[str, Any]] = field(default_factory=list)
    observations: dict[str, list[Observation]] = field(default_factory=dict)
    stop_condition: str = "running"  # agent_done | all_done | max_steps | too_many_errors | timeout | context_overflow | <failure kind>
    truncated: bool = False
    model_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


async def run_episode(world: World, policies: dict[str, Policy], *, max_steps: int = 50, reset: bool = True,
                      limits: EpisodeLimits | None = None) -> EpisodeResult:
    limits = limits or EpisodeLimits(max_steps=max_steps)
    obs = await world.reset() if reset else {a: Observation(actor=a, step=world.step_index) for a in policies}
    for p in policies.values():
        if hasattr(p, "begin_episode"):
            p.begin_episode(world.episode_id)
    log: dict[str, list[Observation]] = {a: [o] for a, o in obs.items()}
    order = list(policies)
    current = {a: obs.get(a) or Observation(actor=a, step=0) for a in order}
    state = {"steps": 0, "done": False, "stop": "running", "errors": 0}

    async def loop() -> None:
        while state["steps"] < limits.max_steps and not state["done"]:
            progressed = False
            for a in order:
                if a in world.done:
                    continue
                action = await policies[a].act(current[a])
                res = await world.step(a, action)
                current[a] = res.observation
                log.setdefault(a, []).append(res.observation)
                state["steps"] += 1
                progressed = True
                state["done"] = res.done
                bad = [r for r in res.observation.tool_results if not r.ok and r.fault in ("policy", "tool")]
                state["errors"] = state["errors"] + 1 if bad else (0 if res.observation.tool_results else state["errors"])
                if limits.max_tool_errors and state["errors"] >= limits.max_tool_errors:
                    state["stop"] = "too_many_errors"
                    return
                ended = [world.finished[x] for x in limits.end_on if x in world.finished]
                if ended:
                    state["stop"] = ended[0]
                    return
                if state["done"] or state["steps"] >= limits.max_steps:
                    break
            if not progressed:
                break

    try:
        await asyncio.wait_for(loop(), timeout=limits.timeout_s)
    except TimeoutError:
        state["stop"] = "timeout"
    failures = [f for r in world.responders.values() for f in getattr(r, "failures", [])]
    failures += [{"kind": p.failure_kind, "reason": p.error} for p in policies.values() if getattr(p, "error", None)]
    failures += [f.model_dump() for f in world.failures]  # infra faults at the World's choke point (tool crashed, transport)
    try:
        final = await world.snapshot(quiesce=True)  # the episode is over: nothing the policy started may keep changing state
    except WorldError as e:  # the policy defeated the verification boundary: its outcome (score 0), never an infra mask
        failures.append({"kind": "crucible:policy_misbehavior", "reason": str(e)[:500], "mask": False})
        final = await world.snapshot()
    stops = {getattr(p, "stop", None) for p in policies.values()}
    if state["stop"] == "running":
        if "context_overflow" in stops:
            state["stop"] = "context_overflow"
        elif failures and any(_masks(f) for f in failures):
            state["stop"] = next(f["kind"] for f in failures if _masks(f))
        elif state["done"]:
            state["stop"] = "all_done" if len(policies) > 1 else "agent_done"
        elif state["steps"] >= limits.max_steps:
            state["stop"] = "max_steps"
        else:
            state["stop"] = "agent_done"
    calls = {a: list(getattr(p, "model_calls", [])) for a, p in policies.items()}
    return EpisodeResult(steps=state["steps"], done=state["done"], trace=world.to_trace(), final=final, masked=any(_masks(f) for f in failures),
                         failures=failures, observations=log, stop_condition=state["stop"],
                         truncated=state["stop"] in ("max_steps", "timeout", "context_overflow", "too_many_errors"), model_calls=calls)


def _masks(f: dict[str, Any]) -> bool:
    from crucible.core.schema.run import POLICY_OUTCOME_KINDS

    if f.get("mask") is not None:
        return bool(f["mask"])
    return f.get("kind") not in POLICY_OUTCOME_KINDS
