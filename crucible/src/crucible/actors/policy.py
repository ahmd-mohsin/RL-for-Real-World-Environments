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
from collections.abc import Callable, Sequence
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

    def state_dict(self) -> dict:
        return {"i": self._i}

    def load_state_dict(self, d: dict) -> None:
        self._i = d["i"]


SYSTEM = """You are an agent working inside a simulated workplace. Use the tools to complete your instruction.
Messages from other people arrive as user messages starting with [from <name>]. To message someone, call the tool
`send_message` with fields `to` and `text`. When the work is complete, reply with a final summary and no tool call."""

SEND_MESSAGE = {"type": "function", "function": {"name": "send_message", "description": "Send a message to another person (actor) in the world.",
                "parameters": {"type": "object", "properties": {"to": {"type": "string"}, "text": {"type": "string"}}, "required": ["to", "text"]}}}


CONTEXT_COMPACT = {"type": "function", "function": {
    "name": "context_compact",
    "description": "Replace your conversation history with a summary you write, when the context is getting long. Include what is done, "
                   "what is pending, every rule that changes how you act, and the exact ids and values you still need. Other tool calls in "
                   "the same reply are not executed.",
    "parameters": {"type": "object", "properties": {"summary": {"type": "string"}, "keep_recent_turns": {"type": "integer", "minimum": 0}},
                   "required": ["summary"]}}}
SUMMARY_HEAD = "[Summary of your earlier work in this episode; the full history was compacted]\n"

COMPACT = ("Your context is nearly full. Write a summary of your work so far that lets you continue without the earlier history: "
           "what is done, what is pending and why (including anything you are waiting for), every rule or announcement that changes how "
           "you must act, and the exact ids and amounts you still need. Reply with the summary only, no tool call.")


@dataclass
class ContextConfig:
    """Context management for long horizons (DESIGN §10.4). ``budget_tokens``: the context the policy may use;
    at ``trigger`` × budget the policy compacts: it writes the summary itself (a recorded model call, trained with the
    episode's advantage), and the new context is system + the task verbatim + the summary + the last
    ``keep_recent_turns`` complete turns. Each compaction starts a new segment."""

    budget_tokens: int = 128_000
    trigger: float = 0.8
    keep_recent_turns: int = 3
    summary_max_tokens: int = 2_000
    # "compact": the harness triggers ``strategy`` at trigger × budget; "action": the policy has a ``context_compact``
    # tool and compacts only when it decides to (MiniMax; no automatic trigger); "none": overflow ends the episode as
    # context_overflow (scored)
    mode: str = "compact"
    # m13 C-7 (A2-46) strategies for mode="compact": "summary" (the policy writes a summary: a recorded, trained call),
    # "keep_recent" (GLM: drop all but the last ``keep_recent_turns`` turns; GLM uses k = 5), "discard_all" (keep only
    # system + task), "tiered" (HCM: first shrink older tool outputs to ``tier_tool_chars``; summarize only if still
    # over), "hybrid" (LongCat: ``hybrid_summaries`` summaries, then discard-all keeping the latest summary)
    strategy: str = "summary"
    tier_tool_chars: int = 200
    hybrid_summaries: int = 1


IMAGE_TOKENS = 768  # estimate per image part (a 512-px tile class image on current vision APIs)


def approx_tokens(messages: list[dict[str, Any]]) -> int:
    """A provider-independent estimate (4 characters per token) used for the compaction trigger and the accounting;
    an image part counts ``IMAGE_TOKENS``, not its base64 length."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                n += IMAGE_TOKENS * 4 if isinstance(part, dict) and part.get("type") == "image_url" else len(json.dumps(part, default=str))
        else:
            n += len(c) if isinstance(c, str) else len(json.dumps(c, default=str)) if c else 0
        if m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"], default=str))
    return n // 4 + 4 * len(messages)


CUT_OFF = "Your last reply was cut off at the output limit before you called a tool. Continue: make one tool call (keep any reasoning short)."
NUDGE = "Your last reply was empty. Continue the task: call a tool, or reply with a final summary if the work is complete."
class LLMToolPolicy(Policy):
    """A minimal function-calling harness. Parallel tool calls in one turn are executed in order and their results
    are appended together before any message that arrived meanwhile (a valid history for OpenAI-style APIs);
    provider reasoning fields are replayed verbatim; tool output is bounded once, by the World."""

    def __init__(self, actor_id: str, llm: LLMClient, *, system: str = SYSTEM, max_tokens: int = 1200, max_tool_output: int | None = None,
                 max_nudges: int = 2, temperature: float = 0.0, context: ContextConfig | None = None, vision: bool = False):
        self.actor_id, self.llm, self.system = actor_id, llm, system
        self.vision = vision  # send tool images to the model (vision models); otherwise a placeholder line per image
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
        self._last_full: list[int] | None = None  # prompt + completion of the last captured call in this segment (delta capture)
        self._last_seg: int | None = None
        self._nudges, self.max_nudges = 0, max_nudges
        self.context = context
        self.segment = 0
        self.segments: list[dict[str, Any]] = []  # one per compaction: the boundary and what was kept
        self.trajectory_tokens = 0  # every token that entered the context, before any compaction (instruction, observations, replies)
        self._seen = 0  # messages already counted

    def begin_episode(self, episode_id: str) -> None:
        self.sample_key = episode_id

    # ---- long horizons: accounting, compaction, checkpoint state ------------------------------------------------------
    def _count_new(self) -> None:
        """Inputs that entered the context (instruction, observations, messages) by estimate; the policy's own replies are
        counted from the provider's completion tokens (they include reasoning that never appears in the content)."""
        if self._seen < len(self.messages):
            self.trajectory_tokens += approx_tokens([m for m in self.messages[self._seen:] if m.get("role") != "assistant"])
            self._seen = len(self.messages)

    def _compress(self, tok: dict[str, Any] | None) -> dict[str, Any] | None:
        """Linear token capture (m13 B-3): a call whose prompt extends the previous call's prompt + completion (same segment,
        exact TITO) stores only the new tail (``prompt_delta``) and the prefix length; the full prompt is rebuilt in
        ``runtime.record.build``. Capture memory and checkpoints are O(T) instead of O(turns²)."""
        if not tok or tok.get("prompt_token_ids") is None:
            return tok
        full = list(tok["prompt_token_ids"])
        out = dict(tok)
        if self._last_full is not None and self._last_seg == self.segment and full[: len(self._last_full)] == self._last_full:
            out["prompt_token_ids"] = None
            out["prompt_delta"] = full[len(self._last_full):]
            out["prompt_prefix_len"] = len(self._last_full)
        self._last_full = full + list(tok.get("completion_token_ids") or [])
        self._last_seg = self.segment
        return out

    def _count_reply(self, usage: dict[str, Any], fallback: list[dict[str, Any]]) -> None:
        self.trajectory_tokens += int(usage.get("completion_tokens") or approx_tokens(fallback))

    def _rebuild(self, keep_turns: int, summary: str | None, *, keep_summary: bool = False) -> int:
        """New context: system + task (verbatim) + [summary] + the last ``keep_turns`` complete turns. Returns turns kept."""
        task = next((m for m in self.messages if m["role"] == "user"), None)
        prev_summary = next((m for m in reversed(self.messages) if m["role"] == "user" and str(m.get("content", "")).startswith(SUMMARY_HEAD)), None)
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        cut = starts[-keep_turns] if keep_turns and len(starts) >= keep_turns else (starts[0] if (starts and keep_turns) else len(self.messages))
        recent = self.messages[cut:] if keep_turns else []
        head = [self.messages[0], *([task] if task is not None and task is not self.messages[0] else [])]
        if summary is not None:
            head.append({"role": "user", "content": SUMMARY_HEAD + summary})
        elif keep_summary and prev_summary is not None:
            head.append(prev_summary)
        else:
            head.append({"role": "user", "content": "[Earlier history of this episode was discarded to fit the context; continue from the current state]"})
        self.messages = [*head, *recent]
        self._seen = len(self.messages)
        return min(keep_turns, len(starts))

    def _new_segment(self, strategy: str, before: int, kept: int, summary_call: str | None = None) -> None:
        self.segments.append({"index": self.segment + 1, "strategy": strategy, "summary_call": summary_call, "context_before": before,
                              "context_after": approx_tokens(self.messages), "kept_turns": kept})
        self.segment += 1

    def _shrink_tool_outputs(self, keep_turns: int, chars: int) -> bool:
        starts = [i for i, m in enumerate(self.messages) if m["role"] == "assistant"]
        cut = starts[-keep_turns] if keep_turns and len(starts) >= keep_turns else len(self.messages)
        changed = False
        for i in range(cut):
            m = self.messages[i]
            c = m.get("content")
            if m["role"] == "tool" and isinstance(c, str) and len(c) > chars:
                self.messages[i] = {**m, "content": c[:chars] + f"\n[... older tool output elided ({len(c) - chars} characters) ...]"}
                changed = True
        return changed

    async def _maybe_compact(self) -> None:
        cfg = self.context
        if cfg is None or cfg.mode != "compact" or self._queue or self._pending:
            return
        if approx_tokens(self.messages) < cfg.trigger * cfg.budget_tokens:
            return
        before = approx_tokens(self.messages)
        strategy = cfg.strategy
        if strategy not in ("summary", "keep_recent", "discard_all", "tiered", "hybrid"):
            raise ValueError(f"unknown context strategy {strategy!r}")
        if strategy == "hybrid":
            done = sum(1 for sg in self.segments if sg.get("strategy") in ("summary", "hybrid:summary"))
            strategy = "hybrid:summary" if done < cfg.hybrid_summaries else "hybrid:discard_all"
        if strategy == "tiered":
            if self._shrink_tool_outputs(cfg.keep_recent_turns, cfg.tier_tool_chars) and approx_tokens(self.messages) < cfg.trigger * cfg.budget_tokens:
                self._seen = len(self.messages)
                self._new_segment("tiered:shrink", before, len(self.messages))
                return
            strategy = "tiered:summary"
        if strategy == "keep_recent":
            self._new_segment("keep_recent", before, self._rebuild(cfg.keep_recent_turns, None))
            return
        if strategy in ("discard_all", "hybrid:discard_all"):
            self._new_segment(strategy, before, self._rebuild(0, None, keep_summary=strategy == "hybrid:discard_all"))
            return
        r = await self.llm.complete([*self.messages, {"role": "user", "content": COMPACT}], tools=None, max_tokens=cfg.summary_max_tokens,
                                    temperature=self.temperature, sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
        self._count_reply(r.usage or {}, [{"role": "assistant", "content": r.text or ""}])
        self.model_calls.append({"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "compaction", "segment": self.segment,
                                 "model": r.model, "finish": r.raw_finish, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                                 "cached": r.cached, "actions": [], "tokens": self._compress(r.tokens), "context_tokens": before})
        kept = self._rebuild(cfg.keep_recent_turns, r.text or "")
        self._new_segment(strategy, before, kept, summary_call=self.model_calls[-1]["call_id"])

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue this policy mid-episode (``runtime.checkpoint``): its context, tools, queued actions,
        bookkeeping and segment structure. Resuming re-prefills one segment, never the whole trajectory."""
        return {"messages": self.messages, "tools": self.tools, "names": self._names, "queue": [a.model_dump(mode="json") for a in self._queue],
                "pending": sorted(self._pending), "held": self._held, "stop": self.stop, "nudges": self._nudges, "model_calls": self.model_calls,
                "segment": self.segment, "segments": self.segments, "trajectory_tokens": self.trajectory_tokens, "seen": self._seen,
                "last_full": self._last_full, "last_seg": self._last_seg,
                "sample_key": self.sample_key, "error": self.error, "failure_kind": self.failure_kind}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        from pydantic import TypeAdapter

        self.messages, self.tools, self._names = d["messages"], d["tools"], d["names"]
        self._queue = [TypeAdapter(Action).validate_python(a) for a in d["queue"]]
        self._pending, self._held = set(d["pending"]), d["held"]
        self.stop, self._nudges, self.model_calls = d["stop"], d["nudges"], d["model_calls"]
        self.segment, self.segments, self.trajectory_tokens, self._seen = d["segment"], d["segments"], d["trajectory_tokens"], d["seen"]
        self._last_full, self._last_seg = d.get("last_full"), d.get("last_seg")
        self.sample_key, self.error, self.failure_kind = d.get("sample_key"), d.get("error"), d.get("failure_kind")

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
            if self.context is not None and self.context.mode == "action":
                self.tools.append(CONTEXT_COMPACT)
        if obs.instruction:
            self.messages.append({"role": "user", "content": obs.instruction})
        for tr in obs.tool_results:
            content = tr.content
            if self.max_tool_output and len(content) > self.max_tool_output:
                h = self.max_tool_output * 2 // 3
                content = content[:h] + f"\n[... {len(content) - self.max_tool_output} characters truncated ...]\n" + content[-(self.max_tool_output - h):]
            if tr.images and not self.vision:
                content += "".join(f"\n[image {i.mime} omitted: this model receives text only]" for i in tr.images)
            self.messages.append({"role": "tool", "tool_call_id": tr.tool_call_id, "content": content})
            self._pending.discard(tr.tool_call_id)
            if tr.images and self.vision:  # OpenAI-style tool messages carry text only: images follow the batch as a user turn
                self._held.append({"role": "user", "content": [{"type": "text", "text": f"[images returned by tool call {tr.tool_call_id}]"},
                                                               *({"type": "image_url", "image_url": {"url": f"data:{i.mime};base64,{i.data_b64}"}}
                                                                 for i in tr.images if i.data_b64)]})
        for m in obs.inbox:
            if m.frm == self.actor_id:  # its own line from a prefilled conversation
                self._held.append({"role": "assistant", "content": m.text})
            else:
                self._held.append({"role": "user", "content": f"[from {m.frm}{' at ' + m.at if m.at else ''}] {m.text}"})
        if not self._pending:  # the batch is complete: now the messages that arrived during it
            self.messages += self._held
            self._held = []

    async def act(self, obs: Observation) -> Action:
        self._ingest(obs)
        self._count_new()
        if self._queue:
            return self._queue.pop(0)
        await self._maybe_compact()
        try:
            r = await self.llm.complete(self.messages, tools=self.tools or None, max_tokens=self.max_tokens, temperature=self.temperature,
                                        sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)  # sampled branches must differ
        except ContextOverflow as e:
            self.stop = "context_overflow"  # truncated, scored: a long-horizon failure is data, not an outage
            return Done(reason=f"context overflow: {e}")
        except (LLMError, BudgetExceeded) as e:
            self.failure_kind = getattr(e, "failure_kind", None) or "provider_unavailable"
            self.error = f"{self.failure_kind}: {e}"
            return Done(reason=f"policy error: {e}")
        if (r.tokens or {}).get("new_segment") and self.model_calls:  # the engine re-rendered the history (thinking mode): a new segment
            self._new_segment("rerender", approx_tokens(self.messages), len(self.messages))
        call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "turn", "segment": self.segment, "context_tokens": approx_tokens(self.messages),
                "step": obs.step, "model": r.model, "finish": r.raw_finish, "usage": r.usage,
                "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "cached": r.cached, "response_id": r.response_id,
                "tool_calls": len(r.tool_calls), "actions": [], "tokens": self._compress(r.tokens)}
        self.model_calls.append(call)  # ``actions`` (filled below, in execution order) joins this call to its World actions and ATIF steps
        self._count_reply(r.usage or {}, [{"role": "assistant", "content": r.text or "", "tool_calls": r.tool_calls or None}])
        if not r.tool_calls and not (r.text or "").strip() and self._nudges < self.max_nudges:
            # A blank turn (e.g. a reasoning model that spent its budget thinking) is not a decision to stop.
            self._nudges += 1
            self.messages.append({"role": "user", "content": NUDGE})  # no empty assistant turn: some providers reject it
            return await self.act(Observation(actor=obs.actor, step=obs.step))
        if not r.tool_calls and r.raw_finish == "length" and self._nudges < self.max_nudges:
            # cut off mid-reply: not a decision to stop (mini-swe-agent's "you were cut off" format error)
            self._nudges += 1
            self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or "(cut off)"})
            self.messages.append({"role": "user", "content": CUT_OFF})
            return await self.act(Observation(actor=obs.actor, step=obs.step))
        if not r.tool_calls:
            self.messages.append({**r.raw_message, "role": "assistant", "content": r.text})
            self.stop = "agent_done"
            call["actions"] = ["done"]
            return Done(reason=r.text[:2000], answer=r.text if len(r.text) > 2000 else None)
        compact = next((tc for tc in r.tool_calls if tc["name"] == "context_compact"), None)
        if compact is not None and self.context is not None and self.context.mode == "action":
            # m13 C-7 (A2-47): the policy compacts by its own decision; the summary is this (recorded, trained) call's output
            before = approx_tokens(self.messages)
            call["kind"], call["actions"] = "compaction", ["context_compact"]
            keep = int(compact["arguments"].get("keep_recent_turns", 0) or 0) if isinstance(compact["arguments"], dict) else 0
            summary = str(compact["arguments"].get("summary", "")) if isinstance(compact["arguments"], dict) else ""
            self._new_segment("action", before, self._rebuild(max(0, keep), summary), summary_call=call["call_id"])
            self._compactions_in_row = getattr(self, "_compactions_in_row", 0) + 1
            if self._compactions_in_row > 2:  # compacting again and again without acting is not progress
                self.stop = "too_many_compactions"
                return Done(reason="the policy kept compacting without acting")
            return await self.act(Observation(actor=obs.actor, step=obs.step))
        self._compactions_in_row = 0
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
        call["actions"] = ["message" if isinstance(a, Message) else f"tool:{c['id']}" for a, c in zip(actions, calls, strict=True)]
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
    snapshot_every: int = 0  # >0: a WorldRef every N steps (deduplicated by fingerprint) for state-based credit and branching


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
    snapshots: list[tuple[int, WorldRef]] = field(default_factory=list)  # (step, ref) when limits.snapshot_every > 0
    truncated: bool = False
    model_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    observed: list[dict[str, Any]] = field(default_factory=list)  # each observer's on_end summary (e.g. dense progress)


async def run_episode(world: World, policies: dict[str, Policy], *, max_steps: int = 50, reset: bool = True,
                      limits: EpisodeLimits | None = None, observers: Sequence[Any] = (), start_steps: int = 0,
                      after_step: Any = None) -> EpisodeResult:
    """``observers``: objects with ``on_start(world)``, ``on_step(world, step, actor, action, result)`` and ``on_end(world)``
    (``verify.progress.ProgressTracker``). ``start_steps``/``reset=False`` continue an episode restored from a checkpoint;
    ``after_step(step)`` is awaited after every step (checkpointing, ``runtime.checkpoint``)."""
    limits = limits or EpisodeLimits(max_steps=max_steps)
    obs = await world.reset() if reset else {a: Observation(actor=a, step=world.step_index) for a in policies}
    for o in observers:
        await o.on_start(world)
    for p in policies.values():
        if hasattr(p, "begin_episode"):
            p.begin_episode(world.episode_id)
    log: dict[str, list[Observation]] = {a: [o] for a, o in obs.items()}
    order = list(policies)
    if limits.snapshot_every:
        series_start = await world.snapshot()
    current = {a: obs.get(a) or Observation(actor=a, step=0) for a in order}
    state = {"steps": start_steps, "done": False, "stop": "running", "errors": 0}
    series: list[tuple[int, WorldRef]] = [(0, series_start)] if limits.snapshot_every else []

    async def capture() -> None:
        ref = await world.snapshot()  # no quiesce mid-episode: the policy's servers may legitimately run
        if not series or series[-1][1].fingerprint != ref.fingerprint:
            series.append((state["steps"], ref))

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
                for o in observers:
                    await o.on_step(world, state["steps"], a, action, res)
                if after_step is not None:
                    await after_step(state["steps"])
                if limits.snapshot_every and state["steps"] % limits.snapshot_every == 0:
                    await capture()
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
    observed = [await o.on_end(world) for o in observers]
    failures = [f for r in world.responders.values() for f in getattr(r, "failures", [])]
    failures += [{"kind": p.failure_kind, "reason": p.error} for p in policies.values() if getattr(p, "error", None)]
    failures += [f.model_dump() for f in world.failures]  # infra faults at the World's choke point (tool crashed, transport)
    try:
        final = await world.snapshot(quiesce=True)  # the episode is over: nothing the policy started may keep changing state
    except WorldError as e:  # the policy defeated the verification boundary: its outcome (score 0), never an infra mask
        failures.append({"kind": "crucible:policy_misbehavior", "reason": str(e)[:500], "mask": False})
        final = await world.snapshot()
    trace = world.to_trace()
    hit = (trace.outcome.details.get("probes") or {}).get("first_strong_exposure")
    if hit:  # the World let the policy reach the answer (P21 says it must not): whatever it scores is not evidence
        failures.append({"kind": "crucible:answer_exposed", "reason": f"{hit['kind']} at event {hit['seq']}: {hit['evidence'][:200]}", "mask": True})
    stops = {getattr(p, "stop", None) for p in policies.values()}
    if state["stop"] == "running":
        if "context_overflow" in stops:
            state["stop"] = "context_overflow"
        elif "too_many_errors" in stops:  # a policy that detected its own stall (RLMPolicy): truncated, not done
            state["stop"] = "too_many_errors"
        elif failures and any(_masks(f) for f in failures):
            state["stop"] = next(f["kind"] for f in failures if _masks(f))
        elif state["done"]:
            state["stop"] = "all_done" if len(policies) > 1 else "agent_done"
        elif state["steps"] >= limits.max_steps:
            state["stop"] = "max_steps"
        else:
            state["stop"] = "agent_done"
    calls = {a: list(getattr(p, "model_calls", [])) for a, p in policies.items()}
    return EpisodeResult(steps=state["steps"], done=state["done"], trace=trace, final=final, masked=any(_masks(f) for f in failures),
                         failures=failures, observations=log, stop_condition=state["stop"],
                         truncated=state["stop"] in ("max_steps", "timeout", "context_overflow", "too_many_errors"), model_calls=calls,
                         snapshots=series, observed=observed)


def _masks(f: dict[str, Any]) -> bool:
    from crucible.core.schema.run import POLICY_OUTCOME_KINDS

    if f.get("mask") is not None:
        return bool(f["mask"])
    return f.get("kind") not in POLICY_OUTCOME_KINDS
