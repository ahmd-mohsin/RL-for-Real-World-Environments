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
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from crucible.actors.llm import BudgetExceeded, ContextOverflow, LLMClient, LLMError
from crucible.core.ids import sha256_text
from crucible.core.schema.actions import Action, Done, Message, Observation, ToolCall, openai_name
from crucible.core.schema.state import WorldRef
from crucible.core.schema.trace import Trace
from crucible.world import hostreach
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
    """Context management for long horizons (DESIGN §10.4). ``budget_tokens``: the context the policy may use (None: the
    model's window when its client knows it, ``LLMClient.context_window``, else ``DEFAULT_BUDGET_TOKENS``; M2 R-14);
    at ``trigger`` × budget the policy compacts: it writes the summary itself (a recorded model call, trained with the
    episode's advantage), and the new context is system + the task verbatim + the summary + the last
    ``keep_recent_turns`` complete turns. Each compaction starts a new segment."""

    budget_tokens: int | None = None
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


DEFAULT_BUDGET_TOKENS = 128_000  # the compaction budget when neither ContextConfig nor the model's client gives one
# of the window, left free when a reply is fitted to it (M2 R-1): the context past the last provider-counted prompt is an estimate
WINDOW_MARGIN = 0.02


IMAGE_TOKENS = 768  # estimate per image part (a 512-px tile class image on current vision APIs)
AUDIO_TOKENS_PER_S = 25  # estimate for audio input parts; the clip length is read as 16 kHz 16-bit mono (32 kB/s)


def media_parts(text: str, media: list[Any] | None, *, vision: bool, audio: bool) -> str | list[dict[str, Any]]:
    """A user turn's content: plain text, or OpenAI chat content parts when media comes with it (M1 item 8). Images
    become ``image_url`` parts (a data URL, or the remote URL) for vision models; audio becomes ``input_audio`` parts for
    audio models; media a model cannot take is named in a placeholder line, so the text still says it was there."""
    if not media:
        return text
    parts: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    for m in media:
        if m.kind == "image" and vision and (m.data_b64 or m.url):
            parts.append({"type": "image_url", "image_url": {"url": m.url or f"data:{m.mime};base64,{m.data_b64}"}})
        elif m.kind == "audio" and audio and m.data_b64:
            fmt = {"audio/mpeg": "mp3", "audio/wav": "wav"}.get(m.mime, m.mime.split("/")[-1])  # canonical types (R-26)
            parts.append({"type": "input_audio", "input_audio": {"data": m.data_b64, "format": fmt}})
        else:
            why = "this model receives text only" if not (vision if m.kind == "image" else audio) else "no data available"
            parts.append({"type": "text", "text": f"[{m.kind} {m.mime} omitted: {why}]"})
    return parts


def approx_tokens(messages: list[dict[str, Any]]) -> int:
    """A provider-independent estimate (4 characters per token) used for the compaction trigger and the accounting;
    an image part counts ``IMAGE_TOKENS``, not its base64 length."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                kind = part.get("type") if isinstance(part, dict) else None
                if kind == "image_url":
                    n += IMAGE_TOKENS * 4
                elif kind == "input_audio":
                    n += int(len(part["input_audio"].get("data", "")) * 3 / 4 / 32000 * AUDIO_TOKENS_PER_S) * 4
                else:
                    n += len(json.dumps(part, default=str))
        else:
            n += len(c) if isinstance(c, str) else len(json.dumps(c, default=str)) if c else 0
        if m.get("tool_calls"):
            n += len(json.dumps(m["tool_calls"], default=str))
    return n // 4 + 4 * len(messages)


# a tool-call id every provider accepts back in the history (Bedrock and Anthropic: [a-zA-Z0-9_-]; the derived ones are also
# Mistral's 9 alphanumerics); others are replaced (M2 R-6)
CALL_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


CUT_OFF = "Your last reply was cut off at the output limit, so nothing in it was run. Continue: make one tool call (keep any reasoning short)."
NUDGE = "Your last reply was empty. Continue the task: call a tool, or reply with a final summary if the work is complete."
TURN_NUDGE = "It is still your turn in the game, which is not over. Choose one of your legal actions with a tool call."
class LLMToolPolicy(Policy):
    """A minimal function-calling harness. Parallel tool calls in one turn are executed in order and their results
    are appended together before any message that arrived meanwhile (a valid history for OpenAI-style APIs);
    provider reasoning fields are replayed verbatim; tool output is bounded once, by the World."""

    def __init__(self, actor_id: str, llm: LLMClient, *, system: str = SYSTEM, max_tokens: int = 1200, max_tool_output: int | None = None,
                 max_nudges: int = 2, temperature: float = 0.0, context: ContextConfig | None = None, vision: bool = False, audio: bool = False,
                 tool_limit: int | None = None, limit_warning: int = 5, status_file: str = "STATUS.json"):
        self.actor_id, self.llm, self.system = actor_id, llm, system
        self.vision = vision  # send tool images to the model (vision models); otherwise a placeholder line per image
        self.audio = audio  # send audio parts (audio-input models); otherwise a placeholder line per clip
        self.max_tokens, self.max_tool_output, self.temperature = max_tokens, max_tool_output, temperature
        self.messages: list[dict[str, Any]] = [{"role": "system", "content": system}]
        self.tools: list[dict[str, Any]] = []
        self._names: dict[str, str] = {}  # openai function name -> qualified tool name
        self._queue: list[Action] = []
        self._pending: set[str] = set()  # tool call ids of the current batch still waiting for results
        self._held: list[dict[str, Any]] = []  # messages that arrived while a batch was in flight
        self.error: str | None = None
        self.failure_kind: str | None = None
        self.retry_after: float | None = None  # M10 lab audit R-5 item 3: the provider's Retry-After of the LLMError that ended the episode (not in state_dict)
        self.stop: str | None = None
        self.sample_key: str | None = None
        self.model_calls: list[dict[str, Any]] = []
        self._last_full: list[int] | None = None  # prompt + completion of the last captured call in this segment (delta capture)
        self._last_seg: int | None = None
        self._nudges, self.max_nudges = 0, max_nudges
        self._turn_nudges = 0  # per turn, reset by any tool call: a game seat replying in text on its own turn has not resigned
        # the episode's tool-call limit, told truthfully (M17 report 11 F5): at ``limit_warning`` calls left one note says so;
        # at the limit, one final turn offers only the write-file tool, for ``status_file``; then the episode ends (max_steps)
        self.tool_limit, self.limit_warning, self.status_file = tool_limit, limit_warning, status_file
        self.tool_calls_issued = 0
        self.limit_events: list[dict[str, Any]] = []
        self._limit_warned = False
        self._final_turn_used = False
        self._act_depth = 0
        self.context = context
        self.segment = 0
        self.segments: list[dict[str, Any]] = []  # one per compaction: the boundary and what was kept
        self.trajectory_tokens = 0  # every token that entered the context, before any compaction (instruction, observations, replies)
        self._seen = 0  # messages already counted
        self._measured: tuple[int, int] | None = None  # (messages in the last prompt, its prompt + completion tokens per the provider)
        self._ids_seen: set[str] = set()  # tool-call ids used in this episode (``_call_ids``; rebuilt from the history on load)

    def begin_episode(self, episode_id: str) -> None:
        self.sample_key = episode_id

    def context_tokens(self) -> int:
        """The context's size: the provider's count of the last prompt and reply plus an estimate of what came after, else
        the chars/4 estimate. chars/4 alone undercounted code and CJK text by 2-3x, so compaction fired after the
        provider had already refused the prompt (M2 audit)."""
        if self._measured is not None and self._measured[0] <= len(self.messages):
            n, tokens = self._measured
            return tokens + approx_tokens(self.messages[n:])
        return approx_tokens(self.messages)

    def _measure(self, usage: dict[str, Any] | None) -> None:
        u = usage or {}
        if u.get("prompt_tokens"):
            self._measured = (len(self.messages), int(u["prompt_tokens"]) + int(u.get("completion_tokens") or 0))

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
        self._measured = None
        return min(keep_turns, len(starts))

    async def _window(self) -> int | None:
        """The model's context window when the client knows it (``LLMClient.context_window``: config, else a self-hosted
        engine's /models card), else None."""
        get = getattr(self.llm, "context_window", None)
        return await get() if get is not None else None

    async def _reply_tokens(self) -> int:
        """``max_tokens`` for the next turn (M2 R-1; PATTERN: AReaL, slime, rLLM and ROLL send min(requested, window - prompt),
        Apache-2.0): ``max_tokens`` cut to the room the model's window leaves after this context and ``WINDOW_MARGIN``.
        Engines refuse a prompt plus ``max_tokens`` over the window, which ended the episode with room left (a 26,172-token
        prompt in a 32,768 window asking for 8,000). No room at all is a ``ContextOverflow`` before the call is paid for;
        an unknown window sends ``max_tokens`` as before."""
        window = await self._window()
        if not window:
            return self.max_tokens
        used = self.context_tokens()
        room = window - used - int(window * WINDOW_MARGIN)
        if room <= 0:
            raise ContextOverflow(f"the context ({used} tokens) leaves no room for a reply in the {window}-token window")
        return min(self.max_tokens, room)

    def _call_ids(self, calls: list[dict[str, Any]], call: dict[str, Any]) -> list[str]:
        """The id each tool call of this reply goes by (M2 R-6; PATTERN: NeMo-Gym's nemo_user_sim checks every call has a
        non-empty id unique in the response, Apache-2.0, repaired here rather than charged to the policy, since a serving
        stack's parser makes these). A provider id is kept when it matches ``CALL_ID`` and is new in this episode; else
        ``c`` + 8 hex of sha256(actor:call:position), the same on every replay. Ids key the World's results, the batch in
        flight and the format penalties: a provider numbering ids per reply charged turn 2's malformed call to turn 1 too,
        two calls sharing an id let a message in between their results, and an id-less reply got a random id, so replays
        missed the cache. Replaced ids are kept on the call record (``provider_ids``)."""
        out: list[str] = []
        for i, tc in enumerate(calls):
            cid, k = str(tc.get("id") or ""), 0
            while not CALL_ID.fullmatch(cid) or cid in self._ids_seen or cid in out:
                cid = "c" + sha256_text(f"{self.actor_id}:{len(self.model_calls)}:{i}" + (f":{k}" if k else ""))[:8]
                k += 1
            out.append(cid)
        self._ids_seen.update(out)
        if any(cid != tc.get("id") for cid, tc in zip(out, calls, strict=True)):
            call["provider_ids"] = [tc.get("id") for tc in calls]
        return out

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
        budget = cfg.budget_tokens or await self._window() or DEFAULT_BUDGET_TOKENS
        if self.context_tokens() < cfg.trigger * budget:
            return
        before = self.context_tokens()
        strategy = cfg.strategy
        if strategy not in ("summary", "keep_recent", "discard_all", "tiered", "hybrid"):
            raise ValueError(f"unknown context strategy {strategy!r}")
        if strategy == "hybrid":
            done = sum(1 for sg in self.segments if sg.get("strategy") in ("summary", "hybrid:summary"))
            strategy = "hybrid:summary" if done < cfg.hybrid_summaries else "hybrid:discard_all"
        if strategy == "tiered":
            if self._shrink_tool_outputs(cfg.keep_recent_turns, cfg.tier_tool_chars) and approx_tokens(self.messages) < cfg.trigger * budget:
                self._seen = len(self.messages)
                self._measured = None
                self._new_segment("tiered:shrink", before, len(self.messages))
                return
            strategy = "tiered:summary"
        if strategy == "keep_recent":
            self._new_segment("keep_recent", before, self._rebuild(cfg.keep_recent_turns, None))
            return
        if strategy in ("discard_all", "hybrid:discard_all"):
            self._new_segment(strategy, before, self._rebuild(0, None, keep_summary=strategy == "hybrid:discard_all"))
            return
        attempts: list[dict[str, Any]] = []
        for attempt in range(2):  # an empty summary (a reasoning model that spent its budget thinking) is resampled once
            r = await self.llm.complete([*self.messages, {"role": "user", "content": COMPACT}], tools=None, max_tokens=cfg.summary_max_tokens,
                                        temperature=self.temperature, sample_key=f"{self.sample_key}:summary{attempt}" if attempt and self.sample_key else self.sample_key,
                                        seed=0 if self.temperature == 0 else None)
            self._count_reply(r.usage or {}, [{"role": "assistant", "content": r.text or ""}])
            if (r.tokens or {}).get("new_segment") and self.model_calls:  # a thinking-template re-render: a segment, not a prefix break (M14 H3 V9)
                self._new_segment("rerender", before, len(self.messages))
            # harness-reward audit 2026-10-28: a resample has the first attempt's prompt, which never extends that attempt's prompt +
            # completion, so as a link of the segment's chain it was a prefix break and train mode masked the whole rollout
            # (capture_incomplete). It is a tree node (edge ``compaction_attempt``, ``Turn.node``): its own sequence, whole prompt
            call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "compaction", "segment": self.segment,
                    "model": r.model, **r.finish_fields(), "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                    "cached": r.cached, "actions": [], "tokens": self._compress(r.tokens), "context_tokens": before,
                    "sampling": getattr(r, "sampling", None) or {}, **({"edge": "compaction_attempt"} if attempt else {})}
            self.model_calls.append(call)
            attempts.append(call)
            if (r.text or "").strip():
                # only the summary the episode continued from is trained (M16 §3.6, verifiers trace.py:515-553): a blank attempt
                # that was discarded is recorded with ``accepted`` False (context, loss mask 0), so an episode's advantage never
                # reinforces an empty summary that was not part of its trajectory
                for a in attempts:
                    a["accepted"] = a is call
                note = "\n[this summary was cut off at its length limit]" if r.finish == "length" else ""
                self._new_segment(strategy, before, self._rebuild(cfg.keep_recent_turns, r.text + note), summary_call=call["call_id"])
                return
        for a in attempts:
            a["accepted"] = False
        # no usable summary: an empty one used to replace the whole history (M2 audit); keep the recent turns instead
        self._new_segment(f"{strategy}:summary_failed", before, self._rebuild(cfg.keep_recent_turns, None, keep_summary=True))

    def state_dict(self) -> dict[str, Any]:
        """Everything needed to continue this policy mid-episode (``runtime.checkpoint``): its context, tools, queued actions,
        bookkeeping and segment structure. Resuming re-prefills one segment, never the whole trajectory."""
        return {"messages": self.messages, "tools": self.tools, "names": self._names, "queue": [a.model_dump(mode="json") for a in self._queue],
                "pending": sorted(self._pending), "held": self._held, "stop": self.stop, "nudges": self._nudges, "model_calls": self.model_calls,
                "segment": self.segment, "segments": self.segments, "trajectory_tokens": self.trajectory_tokens, "seen": self._seen,
                "last_full": self._last_full, "last_seg": self._last_seg,
                "sample_key": self.sample_key, "error": self.error, "failure_kind": self.failure_kind, "measured": self._measured,
                "turn_nudges": self._turn_nudges, "tool_calls_issued": self.tool_calls_issued, "limit_warned": self._limit_warned,
                "final_turn_used": self._final_turn_used, "limit_events": self.limit_events}

    def load_state_dict(self, d: dict[str, Any]) -> None:
        from pydantic import TypeAdapter

        self.messages, self.tools, self._names = d["messages"], d["tools"], d["names"]
        self._queue = [TypeAdapter(Action).validate_python(a) for a in d["queue"]]
        self._pending, self._held = set(d["pending"]), d["held"]
        self.stop, self._nudges, self.model_calls = d["stop"], d["nudges"], d["model_calls"]
        self.segment, self.segments, self.trajectory_tokens, self._seen = d["segment"], d["segments"], d["trajectory_tokens"], d["seen"]
        self._last_full, self._last_seg = d.get("last_full"), d.get("last_seg")
        self.sample_key, self.error, self.failure_kind = d.get("sample_key"), d.get("error"), d.get("failure_kind")
        self._measured = tuple(d["measured"]) if d.get("measured") else None
        self._turn_nudges = d.get("turn_nudges", 0)
        self.tool_calls_issued, self._limit_warned = d.get("tool_calls_issued", 0), d.get("limit_warned", False)
        self._final_turn_used, self.limit_events = d.get("final_turn_used", False), list(d.get("limit_events") or [])
        self._ids_seen = {str(c.get("id")) for m in self.messages for c in m.get("tool_calls") or []}
        self._ids_seen |= {a[5:] for c in self.model_calls for a in c.get("actions") or [] if isinstance(a, str) and a.startswith("tool:")}

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
            self.messages.append({"role": "user", "content": media_parts(obs.instruction, obs.instruction_media, vision=self.vision, audio=self.audio)})
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
                self._held.append({"role": "user", "content": media_parts(f"[from {m.frm}{' at ' + m.at if m.at else ''}] {m.text}", m.attachments,
                                                                          vision=self.vision, audio=self.audio)})
        if not self._pending:  # the batch is complete: now the messages that arrived during it
            self.messages += self._held
            self._held = []

    async def act(self, obs: Observation) -> Action:
        if self.tool_limit is None or self._act_depth:  # off (the default), or a nudge's re-entry: unchanged behaviour
            return await self._act(obs)
        self._ingest(obs)
        self._count_new()
        obs = Observation(actor=obs.actor, step=obs.step, legal=obs.legal, to_move=obs.to_move)  # ingested once
        if self._final_turn_used:
            self.stop = "max_steps"
            return Done(reason=f"the tool-call limit ({self.tool_limit}) was reached")
        if self.tool_calls_issued >= self.tool_limit:
            if self._queue:  # calls of the last reply past the limit are never run; each still gets its tool reply
                for a in self._queue:
                    if isinstance(a, ToolCall):
                        self.messages.append({"role": "tool", "tool_call_id": a.tool_call_id, "content": "not run: the tool-call limit was reached"})
                        self._pending.discard(a.tool_call_id)
                self._queue = []
            return await self._final_status_turn(obs)
        left = self.tool_limit - self.tool_calls_issued
        if not self._limit_warned and left <= self.limit_warning and not self._queue and not self._pending:
            self._limit_warned = True
            note = f"[harness] {left} tool calls remain in this episode. Write {self.status_file} and finish."
            self.messages.append({"role": "user", "content": note})
            self.limit_events.append({"kind": "warning", "at": self.tool_calls_issued, "text": note})
        self._act_depth += 1
        try:
            a = await self._act(obs)
        finally:
            self._act_depth -= 1
        if isinstance(a, ToolCall):
            self.tool_calls_issued += 1
        return a

    async def _final_status_turn(self, obs: Observation) -> Action:
        """One last model turn whose only tool writes files, for the status file; anything else ends the episode."""
        self._final_turn_used = True
        self.stop = "max_steps"
        tools = [t for t in self.tools if t["function"]["name"].split("__")[-1] == "write_file"]
        if not tools or self._pending:
            return Done(reason=f"the tool-call limit ({self.tool_limit}) was reached")
        note = (f"[harness] The tool-call limit ({self.tool_limit}) is reached. One final turn: the only tool is write_file, to write "
                f"{self.status_file}. Reply without a tool call to end without it.")
        self.messages.append({"role": "user", "content": note})
        self.limit_events.append({"kind": "final_turn", "at": self.tool_calls_issued, "text": note})
        try:
            r = await self.llm.complete(self.messages, tools=tools, max_tokens=await self._reply_tokens(), temperature=self.temperature,
                                        sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)
        except (LLMError, BudgetExceeded, ContextOverflow) as e:
            return Done(reason=f"the tool-call limit was reached; the final turn failed: {e}")
        if (r.tokens or {}).get("new_segment") and self.model_calls:  # as in ``_act``: the harness note is a user turn, so a template
            # that drops earlier reasoning there (Qwen3.5, "interleaved") re-rendered the history: a new segment, not a prefix break
            # (Tinker smoke 2026-10-20: the final STATUS turn of every limit-stopped episode was masked capture_incomplete)
            self._new_segment("rerender", approx_tokens(self.messages), len(self.messages))
        call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "turn", "segment": self.segment, "context_tokens": approx_tokens(self.messages),
                "step": obs.step, "model": r.model, "finish": r.raw_finish, "usage": r.usage, "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3),
                "cached": r.cached, "response_id": r.response_id, "tool_calls": len(r.tool_calls), "actions": [], "tokens": self._compress(r.tokens),
                "final_status_turn": True}
        self.model_calls.append(call)
        self._count_reply(r.usage or {}, [{"role": "assistant", "content": r.text or "", "tool_calls": r.tool_calls or None}])
        tc = next((t for t in r.tool_calls if t["name"].split("__")[-1] == "write_file"
                   and str((t.get("arguments") or {}).get("path", "")).lstrip("./") == self.status_file), None)
        if tc is None:
            self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or ""})
            call["actions"] = ["done"]
            self.limit_events.append({"kind": "final_turn_unused"})
            return Done(reason=(r.text or "")[:2000] or f"the tool-call limit ({self.tool_limit}) was reached")
        [cid] = self._call_ids([tc], call)
        self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or "",
                              "tool_calls": [{"id": cid, "type": "function", "function": {"name": tc["name"], "arguments": json.dumps(tc["arguments"])}}]})
        self._pending.add(cid)
        call["actions"] = [f"tool:{cid}"]
        self.limit_events.append({"kind": "final_turn_status_written"})
        return ToolCall(tool_call_id=cid, function_name=self._names.get(tc["name"]) or tc["name"].replace("__", ".", 1), arguments=tc["arguments"])

    async def _act(self, obs: Observation) -> Action:
        self._ingest(obs)
        self._count_new()
        if self._queue:
            return self._queue.pop(0)
        try:
            await self._maybe_compact()  # inside: a failed summary call ends the turn like a failed turn (it crashed run_episode)
            max_tokens = await self._reply_tokens()
            r = await self.llm.complete(self.messages, tools=self.tools or None, max_tokens=max_tokens, temperature=self.temperature,
                                        sample_key=self.sample_key, seed=0 if self.temperature == 0 else None)  # sampled branches must differ
        except ContextOverflow as e:
            self.stop = "context_overflow"  # truncated, scored: a long-horizon failure is data, not an outage
            return Done(reason=f"context overflow: {e}")
        except (LLMError, BudgetExceeded) as e:
            self.failure_kind = getattr(e, "failure_kind", None) or "provider_unavailable"
            self.error = f"{self.failure_kind}: {e}"
            self.retry_after = getattr(e, "retry_after", None)
            return Done(reason=f"policy error: {e}")
        if (r.tokens or {}).get("new_segment") and self.model_calls:  # the engine re-rendered the history (thinking mode): a new segment
            self._new_segment("rerender", approx_tokens(self.messages), len(self.messages))
        call = {"call_id": f"{self.actor_id}:c{len(self.model_calls) + 1}", "kind": "turn", "segment": self.segment, "context_tokens": approx_tokens(self.messages),
                "step": obs.step, "model": r.model, **r.finish_fields(), "usage": r.usage,
                "cost_usd": r.cost_usd, "latency_s": round(r.latency_s, 3), "cached": r.cached, "response_id": r.response_id,
                "tool_calls": len(r.tool_calls), "actions": [], "tokens": self._compress(r.tokens),
                "sampling": getattr(r, "sampling", None) or {},  # what the call sent (R-3): the rollout's Turn.sampling
                **({"max_tokens_requested": self.max_tokens} if max_tokens < self.max_tokens else {})}  # fitted to the window (M2 R-1)
        self.model_calls.append(call)  # ``actions`` (filled below, in execution order) joins this call to its World actions and ATIF steps
        self._count_reply(r.usage or {}, [{"role": "assistant", "content": r.text or "", "tool_calls": r.tool_calls or None}])
        self._measure(r.usage)
        # retries keep the game context: a nudge that dropped ``legal`` let the next text reply resign the seat (M2 audit)
        again = Observation(actor=obs.actor, step=obs.step, legal=obs.legal, to_move=obs.to_move)
        if r.raw_finish in ("content_filter", "refusal") and not r.tool_calls:
            # a provider filter is not the policy choosing to stop: a scored, labelled end (it was read as an empty reply)
            self.stop = "content_filter"
            call["actions"] = ["done"]
            return Done(reason=f"the provider filtered the reply (finish={r.raw_finish})")
        blank = not r.tool_calls and not (r.text or "").strip()
        dropped = f" ({len(r.tool_calls)} unfinished tool call(s) were not run)" if r.tool_calls and r.finish == "length" else ""
        if blank and self._nudges < self.max_nudges:
            # A blank turn (e.g. a reasoning model that spent its budget thinking) is not a decision to stop.
            self._nudges += 1
            call["nudged"] = "blank"  # its output never enters the context; the nudge does
            self.messages.append({"role": "user", "content": NUDGE})  # no empty assistant turn: some providers reject it
            return await self.act(again)
        if r.finish == "length" and self._nudges < self.max_nudges:
            # cut off mid-reply: not a decision to stop (mini-swe-agent's "you were cut off" format error). Tool calls in a
            # cut-off reply may be truncated mid-argument or missing their siblings: they are dropped, never executed.
            self._nudges += 1
            call["nudged"] = "cut_off"
            self.messages.append({**r.raw_message, "role": "assistant", "content": (r.text or "(cut off)") + dropped})
            self.messages.append({"role": "user", "content": CUT_OFF})
            return await self.act(again)
        if blank or r.finish == "length":
            # M2 R-5 (PATTERN: verifiers truncates on a last length finish, mini-swe-agent ends on repeated format errors; MIT):
            # the allowance is spent (or there was none). A cut-off reply's tool calls are still never run (the third one's
            # ran, and reset the allowance), and a stall is not a decision to stop (it ended agent_done, untruncated).
            # Single-turn answers (no allowance) are graded as before, under this label.
            self.stop = "max_output_tokens" if r.finish == "length" else "format_error"
            self.messages.append({**r.raw_message, "role": "assistant", "content": (r.text or "") + dropped})
            call["actions"] = ["done"]
            return Done(reason=(r.text or "")[:2000], answer=r.text if len(r.text or "") > 2000 else None)
        if not r.tool_calls and obs.legal and self._turn_nudges < self.max_nudges:
            # a turn-based game: legal actions are offered only to the seat to move; a text reply is not a resignation
            self._turn_nudges += 1
            call["nudged"] = "turn"
            self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or "(no action)"})
            self.messages.append({"role": "user", "content": TURN_NUDGE})
            return await self.act(again)
        if r.tool_calls:  # a clean step: the nudge allowance is per stall, not per episode (it never reset: M2 audit)
            self._turn_nudges = self._nudges = 0
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
            args = compact["arguments"] if isinstance(compact["arguments"], dict) else {}
            try:
                keep = int(args.get("keep_recent_turns", 0) or 0)
            except (TypeError, ValueError):
                # the model's malformed argument is a format error it is told about (M10 audit V6: int("all") raised out
                # of the episode and the attempt was masked with unknown blame, which can halt a run)
                if self._nudges >= self.max_nudges:
                    self.stop = "format_error"
                    call["actions"] = ["done"]
                    return Done(reason="context_compact: keep_recent_turns must be an integer")
                self._nudges += 1
                self.messages.append({**r.raw_message, "role": "assistant", "content": r.text or "(context_compact)"})
                self.messages.append({"role": "user", "content": "context_compact failed: keep_recent_turns must be an integer number of turns."})
                return await self.act(again)
            summary = str(args.get("summary", ""))
            self._new_segment("action", before, self._rebuild(max(0, keep), summary), summary_call=call["call_id"])
            self._compactions_in_row = getattr(self, "_compactions_in_row", 0) + 1
            if self._compactions_in_row > 2:  # compacting again and again without acting is not progress
                self.stop = "too_many_compactions"
                return Done(reason="the policy kept compacting without acting")
            return await self.act(again)
        self._compactions_in_row = 0
        calls = []
        actions: list[Action] = []
        for tc, cid in zip(r.tool_calls, self._call_ids(r.tool_calls, call), strict=True):
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
    # actors whose end of conversation ends the episode (stop "user_stop" / "out_of_scope"). None: every simulated person
    # (a responder with a ``scenario``) once all of them have ended, as in every reference loop; () never ends early. The
    # default was () and nothing set it, so after "###DONE###" the policy talked to no one until max_steps (M2 audit).
    end_on: tuple[str, ...] | None = None
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
    stop_condition: str = "running"  # agent_done | all_done | max_steps | timeout | <POLICY_STOP_ORDER label> | <failure kind>
    snapshots: list[tuple[int, WorldRef]] = field(default_factory=list)  # (step, ref) when limits.snapshot_every > 0
    truncated: bool = False
    model_calls: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    observed: list[dict[str, Any]] = field(default_factory=list)  # each observer's on_end summary (e.g. dense progress)
    # M7 R7: observers that raised ([{"observer", "phase", "error"}]); each is off for the rest of the episode and its
    # ``observed`` entry is ``{"failed": <error>}``. An observer measures the episode; its fault never ends or masks it
    observer_failures: list[dict[str, Any]] = field(default_factory=list)


async def run_episode(world: World, policies: dict[str, Policy], *, max_steps: int = 50, reset: bool = True,
                      limits: EpisodeLimits | None = None, observers: Sequence[Any] = (), start_steps: int = 0,
                      after_step: Any = None, start_obs: dict[str, Observation] | None = None, sample_key: str | None = None) -> EpisodeResult:
    """``observers``: objects with ``on_start(world)``, ``on_step(world, step, actor, action, result)`` and ``on_end(world)``
    (``verify.progress.ProgressTracker``). ``start_steps``/``reset=False`` continue an episode restored from a checkpoint;
    ``start_obs`` (with ``reset=False``) are the first observations, e.g. a multi-step task's next instruction
    (``World.begin_task_step``); ``after_step(step)`` is awaited after every step (checkpointing, ``runtime.checkpoint``).
    ``sample_key`` (the harness's attempt id) keys each policy's record/replay cache; it defaults to the World's episode
    id, which is random per run, so only a stable key lets a replay hit (it was always overwritten: M2 audit)."""
    limits = limits or EpisodeLimits(max_steps=max_steps)
    turns = getattr(world, "turns", None)
    if turns is not None and len(policies) > 1:  # a multi-seat game: each seat sees the others' moves and its legal actions
        from crucible.actors.seat_view import MoveLog, SeatView

        if not any(isinstance(p, SeatView) for p in policies.values()) and not any(isinstance(o, MoveLog) for o in observers):
            mlog = MoveLog([s for s in policies if s in getattr(turns, "seats", policies)])
            observers = [*observers, mlog]
            policies = {a: SeatView(p, a, mlog, show_legal=True) if a in mlog.seats else p for a, p in policies.items()}
    if reset:
        obs = await world.reset()
    else:
        obs = dict(start_obs) if start_obs else {a: Observation(actor=a, step=world.step_index) for a in policies}
    observer_failures: list[dict[str, Any]] = []
    broken: dict[int, str] = {}

    async def observe(i: int, o: Any, phase: str, call: Any) -> Any:
        """M7 R7: an observer (a progress tracker's subgoal query) that raises is switched off and recorded; it used to
        escape ``run_episode`` and mask the episode with unknown blame, a mask the policy could choose by its actions."""
        if i in broken:
            return None
        try:
            return await call()
        except asyncio.CancelledError:
            raise
        except Exception as ex:  # noqa: BLE001 -- recorded in observer_failures; the episode goes on
            broken[i] = f"{type(ex).__name__}: {ex}"[:500]
            observer_failures.append({"observer": type(o).__name__, "phase": phase, "error": broken[i]})
            return None

    for i, o in enumerate(observers):
        await observe(i, o, "start", lambda o=o: o.on_start(world))
    for p in [*policies.values(), *world.responders.values()]:  # simulated people key per-episode sampling the same way
        if hasattr(p, "begin_episode"):
            p.begin_episode(sample_key or world.episode_id)
    end_on = limits.end_on
    if end_on is None:
        end_on = tuple(a for a, r in world.responders.items() if getattr(r, "scenario", None) is not None)
        end_all = True
    else:
        end_all = False
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
                turns = getattr(world, "turns", None)
                if turns is not None:
                    if not turns.may_act(a):
                        continue
                    late = turns.take_deferred(a)  # a simultaneous round's results this seat has not seen
                    if late:
                        current[a] = current[a].model_copy(update={"tool_results": [*current[a].tool_results, *late]})
                        log.setdefault(a, []).append(current[a])
                    # the seat's last observation came from its own previous move, when another seat was to move: its turn
                    # and legal actions are the turn manager's now (found live: a stale legal=None made an LLM seat resign)
                    current[a] = current[a].model_copy(update={"to_move": list(turns.to_move), "legal": (turns.legal or {}).get(a)})
                try:
                    action = await policies[a].act(current[a])
                except asyncio.CancelledError:
                    raise
                except Exception as ex:  # noqa: BLE001 -- classified (M10 audit V6, V7)
                    why = f"{a}: {type(ex).__name__}: {ex}"[:500]
                    kind = getattr(ex, "failure_kind", None)
                    if kind is None and isinstance(ex, TimeoutError):
                        kind = "transport_timeout"  # an inference socket, not the episode clock: infra, masked
                    elif kind is None and isinstance(ex, (ConnectionError, OSError)):
                        kind = "transport_unreachable"  # a lost worker or endpoint: infra, masked
                    if kind is not None:
                        state["policy_failure"], state["stop"] = {"kind": kind, "reason": why}, "error"
                    elif isinstance(ex, (ValueError, IndexError, KeyError, TypeError)):
                        # the policy's own logic choked on its output (a malformed argument, an out-of-range action):
                        # its outcome, scored 0 (it was masked with unknown blame, enough of which halts a run)
                        state["policy_failure"], state["stop"] = {"kind": "crucible:policy_error", "reason": why, "mask": False}, "policy_error"
                    else:  # anything else stays unclassified: masked and counted by the unknown-blame stop rule
                        state["policy_failure"], state["stop"] = {"kind": "agent_run_error", "reason": why, "blame": "unknown"}, "error"
                    return
                res = await world.step(a, action)
                current[a] = res.observation
                log.setdefault(a, []).append(res.observation)
                state["steps"] += 1
                progressed = True
                for i, o in enumerate(observers):
                    await observe(i, o, "step", lambda o=o, a=a, action=action, res=res: o.on_step(world, state["steps"], a, action, res))
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
                ended = [world.finished[x] for x in end_on if x in world.finished]
                if ended and (not end_all or len(ended) == len(end_on)):
                    state["stop"] = ended[-1] if end_all else ended[0]
                    return
                broke = _masking_failure(world)
                if broke is not None:  # the episode is no longer evidence: stop now, not after paying for max_steps (M2 audit)
                    state["stop"] = broke
                    return
                if state["done"] or state["steps"] >= limits.max_steps:
                    break
            if not progressed:
                break

    try:
        # the agent clock stands still while a remote host's outage pauses the episode (``world.hostreach``, 2026-10-26)
        async with asyncio.timeout(limits.timeout_s) as agent_clock:
            with hostreach.clock(agent_clock):
                await loop()
    except TimeoutError:
        state["stop"] = "timeout"
    observed = []
    for i, o in enumerate(observers):
        got = await observe(i, o, "end", lambda o=o: o.on_end(world))
        observed.append({"failed": broken[i]} if i in broken else got)
    failures = [f for r in world.responders.values() for f in getattr(r, "failures", [])]
    # M10 lab audit R-5 item 3: a provider's Retry-After travels with the policy's failure (``episode_failure`` keeps it)
    failures += [{"kind": p.failure_kind, "reason": p.error, **({"retry_after": ra} if (ra := getattr(p, "retry_after", None)) is not None else {})}
                 for p in policies.values() if getattr(p, "error", None)]
    if state.get("policy_failure"):
        failures.append(state["policy_failure"])
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
    stops = [s for p in policies.values() if (s := getattr(p, "stop", None)) not in (None, "agent_done")]
    if state["stop"] == "running":
        # a policy's own stop label (context_overflow, max_output_tokens, too_many_errors from a self-detected stall,
        # too_many_compactions, format_error, content_filter) is the episode's: these were reported as a voluntary agent_done (M2 audit)
        if stops:
            state["stop"] = next((s for s in POLICY_STOP_ORDER if s in stops), stops[0])
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
                         truncated=state["stop"] in TRUNCATING_STOPS, model_calls=calls,
                         snapshots=series, observed=observed, observer_failures=observer_failures)


# max_output_tokens (M2 R-5, verifiers' name): every reply up to the nudge allowance was cut off at max_tokens
POLICY_STOP_ORDER = ("context_overflow", "max_output_tokens", "too_many_errors", "too_many_compactions", "format_error", "content_filter")
TRUNCATING_STOPS = ("max_steps", "timeout", "context_overflow", "max_output_tokens", "too_many_errors", "too_many_compactions")  # format_error and content_filter are terminal


def _masking_failure(world: World) -> str | None:
    """The kind of the first failure that masks the episode (a simulator broke, an infra fault), else ``None``."""
    for r in world.responders.values():
        for f in getattr(r, "failures", None) or ():
            if _masks(f):
                return str(f.get("kind"))
    for f in world.failures:
        d = f.model_dump()
        if _masks(d):
            return str(d.get("kind"))
    return None


def _masks(f: dict[str, Any]) -> bool:
    from crucible.core.schema.run import POLICY_OUTCOME_KINDS

    if f.get("mask") is not None:
        return bool(f["mask"])
    return f.get("kind") not in POLICY_OUTCOME_KINDS
